"""排样业务编排：参数模型 -> 核心算法 -> 响应装配 -> 落库。

切割执行会话：从已保存方案启动会话，按 (panel_no, step) 顺序逐刀推进；
“完成当前刀”事件、离线回传批次与换刀确认在单个 SQLite 写事务中完成幂等
判定、版本与对账校验、状态推进和事件/批次/换刀记录，并发/重试语义见
submit_cut_event、submit_cut_batch 与 change_blade。

刀具寿命检查点（可选）：启动会话时设置 blade_life（累计切长上限 mm）与
initial_used 后，每次推进前按方案快照中的切割长度预判——下一刀会超过寿命
则拒绝推进（next_action=blade_change），换刀确认把已用切长归零并递增版本。
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from . import database as db
from .gcode import generate_gcode, validate_plan_cuts
from .models import (
    BatchCutIn,
    BladeChangeRequest,
    CutBatchRequest,
    CutEventRequest,
    GcodeRequest,
    PlanRequest,
)
from .nesting import build_payload, solve


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_plan(req: PlanRequest) -> dict:
    """纯计算（不碰数据库），便于单元测试。"""
    parts = [
        {
            "id": p.id,
            "width": p.width,
            "height": p.height,
            "quantity": p.quantity,
            "rotatable": p.rotatable,
        }
        for p in req.parts
    ]
    internal = solve(
        board_w=req.board_width,
        board_h=req.board_height,
        kerf=req.kerf_width,
        max_boards=req.max_boards,
        parts=parts,
    )
    return build_payload(internal)


def create_plan(req_dict: dict) -> dict:
    """幂等创建：相同请求返回同一方案。"""
    request_json = db.canonical_request(req_dict)
    plan_id = db.plan_id_from_request(req_dict)

    with db.get_conn() as conn:
        existing = db.get_plan(plan_id, conn)
        if existing is not None:
            return existing

        payload = compute_plan(PlanRequest.model_validate(req_dict))
        payload["plan_id"] = plan_id
        payload["created_at"] = _now_iso()
        db.save_plan(plan_id, request_json, json.dumps(payload, ensure_ascii=False),
                     payload["created_at"], conn)
        return payload


class PlanGeometryError(Exception):
    """已存方案未通过导出前几何校验；violations 为结构化错误列表。"""

    def __init__(self, violations: list[dict]):
        super().__init__("方案切割线未通过导出前几何校验")
        self.violations = violations


def export_gcode(plan_id: str, req: GcodeRequest) -> Optional[dict]:
    """导出已存方案的 G-code。

    方案不存在返回 None；几何校验失败抛 PlanGeometryError（不产出任何
    部分 G-code）。
    """
    with db.get_conn() as conn:
        plan = db.get_plan(plan_id, conn)
    if plan is None:
        return None
    violations = validate_plan_cuts(plan)
    if violations:
        raise PlanGeometryError(violations)
    return generate_gcode(
        plan,
        plan_id=plan_id,
        origin=req.origin,
        unit=req.unit,
        safe_z=req.safe_z,
    )


# ---------------------------------------------------------------------------
# 切割执行会话
# ---------------------------------------------------------------------------

def flatten_cuts(plan: dict) -> list[dict]:
    """把方案的切割线展平为全局执行序列（按 panel_no、step 升序）。

    返回项在方案 CutOut 之上附带 panel_no；顺序与 G-code 导出一致：
    先执行的刀把后续刀所在区域切出，母板按编号依次上机。
    """
    seq: list[dict] = []
    for panel in plan.get("panels", []):
        for cut in panel.get("cuts", []):
            seq.append({
                "panel_no": panel["panel_no"],
                "step": cut["step"],
                "axis": cut["axis"],
                "x": cut["x"],
                "y": cut["y"],
                "length": cut["length"],
                "region": cut["region"],
            })
    seq.sort(key=lambda c: (c["panel_no"], c["step"]))
    return seq


def _add_len(used: float, length: float) -> float:
    """累计切长：统一保留 6 位小数（与排样输出精度一致，抑制浮点噪声）。"""
    return round(used + length, 6)


def _would_exceed(blade_life: Optional[float], used: float,
                  length: float) -> bool:
    """预判：在已用 used 的基础上再切 length 是否会超过刀具寿命。"""
    return blade_life is not None and _add_len(used, length) > blade_life


def _session_state(row: dict, plan: Optional[dict] = None) -> dict:
    """由会话行（+ 方案）装配对外会话状态。"""
    if plan is None:
        plan = json.loads(row["plan_json"])
    cuts = flatten_cuts(plan)
    done = row["completed_cuts"]
    nxt = cuts[done] if done < len(cuts) else None
    blade_life = row["blade_life"]
    if blade_life is None:
        blade_used = blade_remaining = None
    else:
        blade_used = row["blade_used"]
        blade_remaining = round(blade_life - blade_used, 6)
    if nxt is None:
        next_action = "none"
    elif _would_exceed(blade_life, row["blade_used"], nxt["length"]):
        next_action = "blade_change"
    else:
        next_action = "cut"
    return {
        "session_id": row["session_id"],
        "plan_id": row["plan_id"],
        "status": row["status"],
        "completed": row["status"] == "completed",
        "version": row["version"],
        "completed_cuts": done,
        "total_cuts": row["total_cuts"],
        "current_panel": nxt["panel_no"] if nxt else None,
        "next_cut": nxt,
        "blade_life": blade_life,
        "blade_used": blade_used,
        "blade_remaining": blade_remaining,
        "next_action": next_action,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "completed_at": row["completed_at"],
    }


def start_session(plan_id: str, blade_life: Optional[float] = None,
                  initial_used: float = 0.0) -> Optional[dict]:
    """从已保存方案启动切割会话；方案不存在返回 None。

    blade_life 为刀具寿命（累计切长上限 mm），initial_used 为启动时刀具
    已用切长；不传 blade_life 则不跟踪（响应中刀具字段为 null）。
    无切割线的方案（零件恰好铺满无余料切刀，或全部未排）启动即完成。
    """
    with db.get_conn() as conn:
        plan = db.get_plan(plan_id, conn)
        if plan is None:
            return None
        total = len(flatten_cuts(plan))
        now = _now_iso()
        session_id = str(uuid.uuid4())
        done = total == 0
        db.insert_session(
            session_id=session_id,
            plan_id=plan_id,
            status="completed" if done else "active",
            version=0,
            completed_cuts=0,
            total_cuts=total,
            blade_life=blade_life,
            blade_used=round(initial_used, 6) if blade_life is not None else 0.0,
            created_at=now,
            updated_at=now,
            completed_at=now if done else None,
            conn=conn,
        )
        conn.commit()
        row = db.get_session(session_id, conn)
    return _session_state(row, plan)


def get_session_state(session_id: str) -> Optional[dict]:
    """查询会话状态；不存在返回 None。"""
    with db.get_conn() as conn:
        row = db.get_session(session_id, conn)
    if row is None:
        return None
    return _session_state(row)


class SessionConflictError(Exception):
    """409 冲突：detail 为结构化错误体（error/message + 上下文）。"""

    def __init__(self, detail: dict):
        super().__init__(detail.get("message", "会话状态冲突"))
        self.detail = detail


def _blade_block_detail(blade_life: float, blade_used: float,
                        cut_length: float, index: Optional[int] = None,
                        batch: bool = False) -> dict:
    """组装“下一刀会超过刀具寿命”的结构化错误体（状态不推进）。"""
    remaining = round(blade_life - blade_used, 6)
    if batch:
        message = (f"批次第 {index} 项（切长 {cut_length}mm）会使累计切长超过"
                   f"刀具寿命（已用 {blade_used}mm / 寿命 {blade_life}mm，"
                   f"剩余 {remaining}mm），请先换刀，整批未推进")
    else:
        message = (f"下一刀切长 {cut_length}mm 会使累计切长超过刀具寿命"
                   f"（已用 {blade_used}mm / 寿命 {blade_life}mm，"
                   f"剩余 {remaining}mm），请先换刀，状态未推进")
    detail = {
        "error": "blade_change_required",
        "message": message,
        "blade_life": blade_life,
        "blade_used": blade_used,
        "blade_remaining": remaining,
        "cut_length": cut_length,
    }
    if index is not None:
        detail["index"] = index
    return detail


def submit_cut_event(session_id: str, req: CutEventRequest) -> Optional[dict]:
    """提交“完成当前刀”事件；会话不存在返回 None。

    在单个 SQLite 写事务（BEGIN IMMEDIATE）中依次：
      1. 幂等判定：同 (session_id, idempotency_key) 已应用且请求体一致 ->
         原样返回首次响应（状态不重复推进）；同键不同请求 -> 409；
      2. 校验会话未完成、expected_version 等于当前版本、所报
         (panel_no, step) 正是当前刀、下一刀不越过刀具寿命检查点 ——
         任一不满足抛 SessionConflictError，事务回滚，状态不推进；
      3. 推进 version/completed_cuts（全部完成时置 completed）并累计
         blade_used，记录事件，一次性提交。

    BEGIN IMMEDIATE 使并发写者在库级串行：并发同版本请求只有一个能通过
    校验，其余读到推进后的版本并以 version_conflict 拒绝。
    """
    request_json = db.canonical_request(req.model_dump())
    with db.transaction() as conn:
        row = db.get_session(session_id, conn)
        if row is None:
            return None

        # 1) 幂等：同键重试返回首次结果（此时版本已推进，必须先于版本校验）
        evt = db.get_session_event(session_id, req.idempotency_key, conn)
        if evt is not None:
            if evt["request_json"] != request_json:
                raise SessionConflictError({
                    "error": "idempotency_key_reuse",
                    "message": "幂等键已被本会话中的不同请求使用",
                })
            return json.loads(evt["response_json"])

        # 2) 校验：会话未完成 -> 版本一致 -> 所报刀正是当前刀
        if row["status"] == "completed":
            raise SessionConflictError({
                "error": "session_completed",
                "message": f"会话已完成（共 {row['total_cuts']} 刀），"
                           "不能再提交切割事件",
                "total_cuts": row["total_cuts"],
            })
        if req.expected_version != row["version"]:
            raise SessionConflictError({
                "error": "version_conflict",
                "message": f"expected_version={req.expected_version} 与当前版本 "
                           f"{row['version']} 不一致，状态未推进",
                "expected_version": req.expected_version,
                "current_version": row["version"],
            })
        plan = json.loads(row["plan_json"])
        cuts = flatten_cuts(plan)
        nxt = cuts[row["completed_cuts"]]
        if req.panel_no != nxt["panel_no"] or req.step != nxt["step"]:
            raise SessionConflictError({
                "error": "cut_mismatch",
                "message": f"请求完成第 {req.panel_no} 板第 {req.step} 刀，"
                           f"但当前刀是第 {nxt['panel_no']} 板第 "
                           f"{nxt['step']} 刀，状态未推进",
                "expected": {"panel_no": nxt["panel_no"], "step": nxt["step"]},
                "received": {"panel_no": req.panel_no, "step": req.step},
            })
        # 刀具寿命检查点：单刀事件不得越过，须先换刀
        blade_life = row["blade_life"]
        blade_used = row["blade_used"]
        if _would_exceed(blade_life, blade_used, nxt["length"]):
            raise SessionConflictError(_blade_block_detail(
                blade_life, blade_used, nxt["length"]))

        # 3) 推进 + 累计已用切长 + 记录事件，同事务提交
        now = _now_iso()
        new_done = row["completed_cuts"] + 1
        finished = new_done >= row["total_cuts"]
        db.update_session_progress(
            session_id=session_id,
            version=row["version"] + 1,
            completed_cuts=new_done,
            status="completed" if finished else "active",
            blade_used=(_add_len(blade_used, nxt["length"])
                        if blade_life is not None else blade_used),
            updated_at=now,
            completed_at=now if finished else None,
            conn=conn,
        )
        state = _session_state(db.get_session(session_id, conn), plan)
        db.insert_session_event(
            session_id=session_id,
            idempotency_key=req.idempotency_key,
            panel_no=req.panel_no,
            step=req.step,
            request_json=request_json,
            response_json=json.dumps(state, ensure_ascii=False),
            created_at=now,
            conn=conn,
        )
        return state


def _batch_mismatch_detail(phase: str, index: int,
                           expected: Optional[dict],
                           received: Optional[BatchCutIn],
                           skip: int, n: int) -> dict:
    """组装批次校验失败的结构化错误体（整批不推进）。

    phase="reconcile"：批次前缀与 base_version 之后服务端已完成区间对账失败；
    phase="apply"：其余后缀与当前方案的下一刀序列不匹配。
    index 为批次 cuts 数组下标（从 0 开始）；expected/received 为 None 分别
    表示方案已无剩余切刀 / 批次未覆盖到该位置。
    """
    exp = ({"panel_no": expected["panel_no"], "step": expected["step"]}
           if expected is not None else None)
    got = ({"panel_no": received.panel_no, "step": received.step}
           if received is not None else None)
    if phase == "reconcile" and received is None:
        message = (f"批次仅 {n} 刀，不足以覆盖 base_version 之后服务端已完成的 "
                   f"{skip} 刀，无法逐项对账，整批未推进")
    elif phase == "reconcile":
        message = (f"cuts[{index}] 为第 {got['panel_no']} 板第 {got['step']} 刀，"
                   f"但 base_version 之后服务端已完成的该刀是第 "
                   f"{exp['panel_no']} 板第 {exp['step']} 刀，整批未推进")
    elif expected is None:
        message = (f"cuts[{index}] 超出方案剩余切刀：会话已无可推进的刀，"
                   "整批未推进")
    else:
        message = (f"cuts[{index}] 为第 {got['panel_no']} 板第 {got['step']} 刀，"
                   f"但当前方案的下一刀是第 {exp['panel_no']} 板第 "
                   f"{exp['step']} 刀，整批未推进")
    return {
        "error": "batch_mismatch",
        "message": message,
        "phase": phase,
        "index": index,
        "expected": exp,
        "received": got,
    }


def submit_cut_batch(session_id: str, req: CutBatchRequest) -> Optional[dict]:
    """提交设备离线期间完成的切刀批次；会话不存在返回 None。

    在单个 SQLite 写事务（BEGIN IMMEDIATE）中依次：
      1. 幂等判定：同 (session_id, batch_id) 已应用且请求体一致 ->
         原样返回首次响应（状态不重复推进）；同键不同请求 -> 409；
      2. 基线校验：base_version 超过当前版本（客户端基线比服务端还新，
         无法对账）-> 409；
      3. 对账：base_version 之后服务端已完成的区间与批次前缀逐项比对，
         一致前缀仅跳过；首个不一致 -> 409，整批不推进；
      4. 其余后缀必须连续匹配当前方案的下一刀序列，任一不符 -> 409；
      5. 刀具寿命检查点：待应用后缀中任一刀会使累计切长越过寿命 ->
         409，整批不推进；
      6. 验证通过后一次性推进 version/completed_cuts 并累计 blade_used
         （全部完成时置 completed），记录批次，一次性提交；
         applied=0 时不改动会话行。

    与单刀事件、换刀确认共用同一写事务机制：批次与单刀、批次与批次、
    批次与换刀并发时在库级串行，后到者基于最新状态重新对账——已被覆盖
    的批次自然转为“全部跳过”或 409，不会重复推进。

    版本换算：version 随切刀与换刀共同递增（version = completed_cuts +
    换刀次数），故 base_version 对应的已完成刀数 = base_version - 截至
    该版本的换刀次数；对账区间刀数 = done - base_done。
    """
    request_json = db.canonical_request(req.model_dump())
    with db.transaction() as conn:
        row = db.get_session(session_id, conn)
        if row is None:
            return None

        # 1) 幂等：同 batch_id 重放首次响应（必须先于一切校验）
        batch = db.get_session_batch(session_id, req.batch_id, conn)
        if batch is not None:
            if batch["request_json"] != request_json:
                raise SessionConflictError({
                    "error": "batch_id_reuse",
                    "message": "batch_id 已被本会话中的不同批次请求使用",
                })
            return json.loads(batch["response_json"])

        version = row["version"]
        done = row["completed_cuts"]
        total = row["total_cuts"]

        # 2) 基线校验：base_version 比服务端还新，对账区间无意义
        if req.base_version > version:
            raise SessionConflictError({
                "error": "version_conflict",
                "message": f"base_version={req.base_version} 超过当前版本 "
                           f"{version}，无法对账，整批未推进",
                "base_version": req.base_version,
                "current_version": version,
            })

        plan = json.loads(row["plan_json"])
        cuts = flatten_cuts(plan)
        n = len(req.cuts)
        # version 随切刀与换刀共同递增：base_version 对应的已完成刀数
        # base_done = base_version - 截至该版本的换刀次数
        base_done = req.base_version - db.count_blade_changes_up_to(
            session_id, req.base_version, conn)
        # 对账区间刀数 = 当前已完成刀数 - 基线已完成刀数
        skip = done - base_done

        # 3) 对账：base_version 之后已完成的区间必须与批次前缀逐项一致
        for i in range(skip):
            expected = cuts[base_done + i]
            received = req.cuts[i] if i < n else None
            if (received is None
                    or received.panel_no != expected["panel_no"]
                    or received.step != expected["step"]):
                raise SessionConflictError(_batch_mismatch_detail(
                    "reconcile", i, expected, received, skip, n))

        # 4) 其余后缀必须连续匹配当前方案的下一刀序列
        for k in range(skip, n):
            pos = done + (k - skip)
            expected = cuts[pos] if pos < total else None
            received = req.cuts[k]
            if (expected is None
                    or received.panel_no != expected["panel_no"]
                    or received.step != expected["step"]):
                raise SessionConflictError(_batch_mismatch_detail(
                    "apply", k, expected, received, skip, n))

        # 5) 刀具寿命检查点：任一待应用刀越过寿命 -> 整批 409 不推进
        blade_life = row["blade_life"]
        blade_used = row["blade_used"]
        if blade_life is not None:
            used = blade_used
            for k in range(skip, n):
                pos = done + (k - skip)
                used = _add_len(used, cuts[pos]["length"])
                if used > blade_life:
                    raise SessionConflictError(_blade_block_detail(
                        blade_life, blade_used, cuts[pos]["length"],
                        index=k, batch=True))
            blade_used = used

        # 6) 验证通过：一次性推进版本与完成数，记录批次，同事务提交
        applied = n - skip
        now = _now_iso()
        if applied > 0:
            new_done = done + applied
            finished = new_done >= total
            db.update_session_progress(
                session_id=session_id,
                version=version + applied,
                completed_cuts=new_done,
                status="completed" if finished else "active",
                blade_used=blade_used,
                updated_at=now,
                completed_at=now if finished else None,
                conn=conn,
            )
            row = db.get_session(session_id, conn)
        result = {
            "skipped": skip,
            "applied": applied,
            "session": _session_state(row, plan),
        }
        db.insert_session_batch(
            session_id=session_id,
            batch_id=req.batch_id,
            base_version=req.base_version,
            cut_count=n,
            skipped=skip,
            applied=applied,
            request_json=request_json,
            response_json=json.dumps(result, ensure_ascii=False),
            created_at=now,
            conn=conn,
        )
        return result


def change_blade(session_id: str, req: BladeChangeRequest) -> Optional[dict]:
    """确认换刀：已用切长归零并递增版本；会话不存在返回 None。

    在单个 SQLite 写事务（BEGIN IMMEDIATE）中依次：
      1. 幂等判定：同 (session_id, idempotency_key) 已应用且请求体一致 ->
         原样返回首次响应；同键不同请求 -> 409；
      2. 校验会话未完成、已开启刀具寿命跟踪、expected_version 等于当前
         版本 —— 任一不满足抛 SessionConflictError，事务回滚；
      3. blade_used 归零、version +1，记录换刀，一次性提交。

    换刀与切刀事件/批次共用同一版本序列与写事务：并发换刀与切刀只有一个
    能通过版本校验，另一方读到推进后的版本并以 version_conflict 拒绝。
    """
    request_json = db.canonical_request(req.model_dump())
    with db.transaction() as conn:
        row = db.get_session(session_id, conn)
        if row is None:
            return None

        # 1) 幂等：同键重试返回首次结果（必须先于版本校验）
        bc = db.get_blade_change(session_id, req.idempotency_key, conn)
        if bc is not None:
            if bc["request_json"] != request_json:
                raise SessionConflictError({
                    "error": "idempotency_key_reuse",
                    "message": "幂等键已被本会话中的不同请求使用",
                })
            return json.loads(bc["response_json"])

        # 2) 校验：会话未完成 -> 已开启寿命跟踪 -> 版本一致
        if row["status"] == "completed":
            raise SessionConflictError({
                "error": "session_completed",
                "message": f"会话已完成（共 {row['total_cuts']} 刀），"
                           "不能再确认换刀",
                "total_cuts": row["total_cuts"],
            })
        if row["blade_life"] is None:
            raise SessionConflictError({
                "error": "blade_not_tracked",
                "message": "会话启动时未设置 blade_life，不跟踪刀具寿命，"
                           "无需也不能换刀",
            })
        if req.expected_version != row["version"]:
            raise SessionConflictError({
                "error": "version_conflict",
                "message": f"expected_version={req.expected_version} 与当前版本 "
                           f"{row['version']} 不一致，未确认换刀",
                "expected_version": req.expected_version,
                "current_version": row["version"],
            })

        # 3) 已用切长归零 + 版本 +1 + 记录换刀，同事务提交
        now = _now_iso()
        db.reset_session_blade(
            session_id=session_id,
            version=row["version"] + 1,
            updated_at=now,
            conn=conn,
        )
        state = _session_state(db.get_session(session_id, conn))
        db.insert_blade_change(
            session_id=session_id,
            idempotency_key=req.idempotency_key,
            version=row["version"] + 1,
            request_json=request_json,
            response_json=json.dumps(state, ensure_ascii=False),
            created_at=now,
            conn=conn,
        )
        return state
