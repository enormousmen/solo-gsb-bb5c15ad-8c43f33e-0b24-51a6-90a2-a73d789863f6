"""刀具寿命检查点：按累计切长预判、换刀确认、批次整批拦截与并发测试。

语义：启动会话时可设置 blade_life（累计切长上限 mm）与 initial_used；
每次推进前按方案快照中的切割长度预判，下一刀会超过寿命则 409
blade_change_required（单刀不得越过；离线批次任一待应用刀越过则整批
409 不推进）；换刀确认带 expected_version 与幂等键，在 SQLite 写事务中
把 blade_used 归零并递增版本，并发换刀与切刀只有一方成功。响应持续给出
blade_used / blade_remaining / next_action（单位 mm，与排样一致）。

服务层并发测试直接用多线程调用 services（每个事务各自开 SQLite 连接，
真实并发）；API 层用 TestClient 做端到端冒烟。需要 fastapi/pydantic 依赖。
"""
import json
import os
import tempfile
import threading
import unittest

from app import database as db
from app.models import (
    BatchCutIn,
    BladeChangeRequest,
    CutBatchRequest,
    CutEventRequest,
    StartSessionRequest,
)
from app.services import (
    SessionConflictError,
    change_blade,
    create_plan,
    flatten_cuts,
    get_session_state,
    start_session,
    submit_cut_batch,
    submit_cut_event,
)

try:
    from app.main import app
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except Exception:
    HAS_FASTAPI = False


# 4 块母板、20 刀的多板方案（与 test_sessions 同款）
REQ = {
    "board_width": 244.0,
    "board_height": 122.0,
    "kerf_width": 3.0,
    "max_boards": 4,
    "parts": [
        {"id": "door", "width": 120, "height": 60, "quantity": 4,
         "rotatable": True},
        {"id": "shelf", "width": 60, "height": 40, "quantity": 6,
         "rotatable": True},
    ],
}

# 零件恰好铺满母板：无任何切割线
ZERO_CUT_REQ = {
    "board_width": 100.0,
    "board_height": 100.0,
    "kerf_width": 3.0,
    "max_boards": 1,
    "parts": [
        {"id": "slab", "width": 100, "height": 100, "quantity": 1,
         "rotatable": False},
    ],
}


def _req(payload):
    """模拟 JSON 往返后的请求体（与线上序列化路径一致）。"""
    return json.loads(json.dumps(payload))


def _lengths(cuts):
    return [c["length"] for c in cuts]


class _DbTestCase(unittest.TestCase):
    """每个用例一个独立临时库，并把模块级 DB_PATH 指过去。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "blade.db")
        db.init_db(self.path)
        self._orig_path = db.DB_PATH
        db.DB_PATH = self.path

    def tearDown(self):
        db.DB_PATH = self._orig_path
        self.tmp.cleanup()

    def _new_session(self, req=REQ, blade_life=None, initial_used=0.0):
        plan = create_plan(_req(req))
        session = start_session(plan["plan_id"], blade_life=blade_life,
                                initial_used=initial_used)
        return plan, session

    def _event(self, session, key, version, cut):
        return submit_cut_event(session["session_id"], CutEventRequest(
            expected_version=version,
            idempotency_key=key,
            panel_no=cut["panel_no"],
            step=cut["step"],
        ))

    @staticmethod
    def _batch(batch_id, base_version, cuts):
        return CutBatchRequest(
            batch_id=batch_id,
            base_version=base_version,
            cuts=[BatchCutIn(panel_no=c["panel_no"], step=c["step"])
                  for c in cuts],
        )

    @staticmethod
    def _blade_change(session, key, version):
        return change_blade(session["session_id"], BladeChangeRequest(
            expected_version=version, idempotency_key=key))


class TestBladeLifecycle(_DbTestCase):
    def test_start_with_blade_life(self):
        plan, sess = self._new_session(blade_life=1000.0, initial_used=100.0)
        cuts = flatten_cuts(plan)
        self.assertEqual(sess["blade_life"], 1000.0)
        self.assertEqual(sess["blade_used"], 100.0)
        self.assertEqual(sess["blade_remaining"], 900.0)
        self.assertEqual(sess["next_action"], "cut")
        # 查询接口返回同一份状态
        self.assertEqual(get_session_state(sess["session_id"]), sess)
        # 无寿命跟踪的会话：刀具字段为 null，next_action 正常
        _, plain = self._new_session()
        self.assertIsNone(plain["blade_life"])
        self.assertIsNone(plain["blade_used"])
        self.assertIsNone(plain["blade_remaining"])
        self.assertEqual(plain["next_action"], "cut")
        self.assertGreater(len(cuts), 3)

    def test_initial_used_counts_toward_checkpoint(self):
        plan, _ = self._new_session()
        lens = _lengths(flatten_cuts(plan))
        # 寿命 = 前两刀长度；initial_used 先占掉第二刀的量
        _, sess = self._new_session(blade_life=lens[0] + lens[1],
                                    initial_used=lens[1])
        self.assertEqual(sess["blade_used"], lens[1])
        self.assertEqual(sess["blade_remaining"], lens[0])
        # 第一刀恰好用尽剩余寿命
        state = self._event(sess, "e0", 0, flatten_cuts(plan)[0])
        self.assertEqual(state["blade_used"], lens[0] + lens[1])
        self.assertEqual(state["blade_remaining"], 0.0)
        self.assertEqual(state["next_action"], "blade_change")

    def test_checkpoint_blocks_and_blade_change_resets(self):
        plan, _ = self._new_session()
        cuts = flatten_cuts(plan)
        lens = _lengths(cuts)
        life = lens[0] + lens[1]  # 恰好容纳前两刀
        _, sess = self._new_session(blade_life=life)
        sid = sess["session_id"]

        self._event(sess, "e0", 0, cuts[0])
        state = self._event(sess, "e1", 1, cuts[1])
        self.assertEqual(state["blade_used"], life)
        self.assertEqual(state["blade_remaining"], 0.0)
        self.assertEqual(state["next_action"], "blade_change")
        # 查询接口同样给出 blade_change
        self.assertEqual(get_session_state(sid)["next_action"], "blade_change")

        # 单刀事件不得越过检查点：409，状态不推进
        with self.assertRaises(SessionConflictError) as ctx:
            self._event(sess, "e2", 2, cuts[2])
        detail = ctx.exception.detail
        self.assertEqual(detail["error"], "blade_change_required")
        self.assertEqual(detail["blade_life"], life)
        self.assertEqual(detail["blade_used"], life)
        self.assertEqual(detail["blade_remaining"], 0.0)
        self.assertEqual(detail["cut_length"], lens[2])
        state = get_session_state(sid)
        self.assertEqual(state["version"], 2)
        self.assertEqual(state["completed_cuts"], 2)
        with db.get_conn() as conn:
            self.assertEqual(db.count_session_events(sid, conn), 2)

        # 换刀：已用切长归零、版本 +1，可以继续切
        state = self._blade_change(sess, "bc-1", 2)
        self.assertEqual(state["version"], 3)
        self.assertEqual(state["completed_cuts"], 2)  # 进度不变
        self.assertEqual(state["blade_used"], 0.0)
        self.assertEqual(state["blade_remaining"], life)
        self.assertEqual(state["next_action"], "cut")
        self.assertEqual(state["next_cut"], cuts[2])

        state = self._event(sess, "e2", 3, cuts[2])
        self.assertEqual(state["version"], 4)
        self.assertEqual(state["blade_used"], lens[2])

    def test_blade_change_version_and_key_conflicts(self):
        plan, _ = self._new_session()
        life = sum(_lengths(flatten_cuts(plan))[:2])
        _, sess = self._new_session(blade_life=life)
        sid = sess["session_id"]

        # 版本不符 -> 409 version_conflict，不换刀
        with self.assertRaises(SessionConflictError) as ctx:
            self._blade_change(sess, "bc-stale", 5)
        detail = ctx.exception.detail
        self.assertEqual(detail["error"], "version_conflict")
        self.assertEqual(detail["expected_version"], 5)
        self.assertEqual(detail["current_version"], 0)
        self.assertEqual(get_session_state(sid)["version"], 0)

        # 正常换刀（刀具未用完也允许：预防性换刀）
        first = self._blade_change(sess, "bc-1", 0)
        self.assertEqual(first["version"], 1)
        self.assertEqual(first["blade_used"], 0.0)

        # 同键同请求重放：逐字段一致，版本不重复推进
        for _ in range(3):
            self.assertEqual(self._blade_change(sess, "bc-1", 0), first)
        self.assertEqual(get_session_state(sid)["version"], 1)

        # 同键不同请求 -> 409 idempotency_key_reuse
        with self.assertRaises(SessionConflictError) as ctx:
            self._blade_change(sess, "bc-1", 1)
        self.assertEqual(ctx.exception.detail["error"],
                         "idempotency_key_reuse")
        self.assertEqual(get_session_state(sid)["version"], 1)

        # 旧版本换刀 -> 409 version_conflict
        with self.assertRaises(SessionConflictError) as ctx:
            self._blade_change(sess, "bc-2", 0)
        self.assertEqual(ctx.exception.detail["error"], "version_conflict")

    def test_blade_change_rejected_when_untracked_or_completed(self):
        # 未跟踪寿命的会话不能换刀
        plan, sess = self._new_session()
        with self.assertRaises(SessionConflictError) as ctx:
            self._blade_change(sess, "bc", 0)
        self.assertEqual(ctx.exception.detail["error"], "blade_not_tracked")

        # 零切割会话启动即完成：next_action=none，换刀 -> session_completed
        plan0, sess0 = self._new_session(ZERO_CUT_REQ, blade_life=100.0,
                                         initial_used=10.0)
        self.assertTrue(sess0["completed"])
        self.assertEqual(sess0["next_action"], "none")
        self.assertEqual(sess0["blade_used"], 10.0)
        self.assertEqual(sess0["blade_remaining"], 90.0)
        with self.assertRaises(SessionConflictError) as ctx:
            self._blade_change(sess0, "bc", 0)
        self.assertEqual(ctx.exception.detail["error"], "session_completed")

        # 跑完整个会话后：next_action=none，换刀 -> session_completed
        plan, sess = self._new_session(blade_life=1e9)
        cuts = flatten_cuts(plan)
        for i, cut in enumerate(cuts):
            state = self._event(sess, f"e{i}", i, cut)
        self.assertTrue(state["completed"])
        self.assertEqual(state["next_action"], "none")
        self.assertEqual(state["blade_used"], sum(_lengths(cuts)))
        with self.assertRaises(SessionConflictError) as ctx:
            self._blade_change(sess, "bc", state["version"])
        self.assertEqual(ctx.exception.detail["error"], "session_completed")

    def test_full_lifecycle_with_blade_changes(self):
        """交替切刀/换刀跑完整个会话：version 与 completed_cuts 守恒。"""
        plan, _ = self._new_session()
        cuts = flatten_cuts(plan)
        lens = _lengths(cuts)
        life = lens[0] + lens[1]  # 每把刀恰好切两刀
        _, sess = self._new_session(blade_life=life)
        sid = sess["session_id"]

        version = 0
        changes = 0
        for i, cut in enumerate(cuts):
            state = get_session_state(sid)
            if state["next_action"] == "blade_change":
                state = self._blade_change(sess, f"bc-{changes}", version)
                version += 1
                changes += 1
                self.assertEqual(state["blade_used"], 0.0)
            state = self._event(sess, f"e{i}", version, cut)
            version += 1
        self.assertTrue(state["completed"])
        self.assertGreater(changes, 0)
        # version = 刀数 + 换刀次数
        self.assertEqual(state["version"], len(cuts) + changes)
        self.assertEqual(state["completed_cuts"], len(cuts))
        with db.get_conn() as conn:
            self.assertEqual(db.count_session_events(sid, conn), len(cuts))

    def test_cut_longer_than_life_stalls_until_larger_life(self):
        """单刀长度超过寿命：永远 blade_change_required（配置问题，如实反馈）。"""
        plan, _ = self._new_session()
        cuts = flatten_cuts(plan)
        life = cuts[0]["length"] / 2
        _, sess = self._new_session(blade_life=life)
        self.assertEqual(sess["next_action"], "blade_change")
        with self.assertRaises(SessionConflictError) as ctx:
            self._event(sess, "e0", 0, cuts[0])
        self.assertEqual(ctx.exception.detail["error"], "blade_change_required")
        # 换刀后依然不够：仍是 blade_change
        state = self._blade_change(sess, "bc", 0)
        self.assertEqual(state["next_action"], "blade_change")
        self.assertEqual(state["blade_used"], 0.0)

    def test_start_session_unknown_plan_still_none(self):
        self.assertIsNone(start_session(
            "00000000-0000-0000-0000-000000000000", blade_life=100.0))


class TestBladeBatch(_DbTestCase):
    def test_batch_crossing_checkpoint_rejected_atomically(self):
        plan, _ = self._new_session()
        cuts = flatten_cuts(plan)
        lens = _lengths(cuts)
        life = lens[0] + lens[1] + lens[2]
        _, sess = self._new_session(blade_life=life)
        sid = sess["session_id"]

        # 前三刀一批：恰好用尽
        result = submit_cut_batch(sid, self._batch("b1", 0, cuts[:3]))
        self.assertEqual((result["skipped"], result["applied"]), (0, 3))
        self.assertEqual(result["session"]["blade_used"], life)
        self.assertEqual(result["session"]["next_action"], "blade_change")

        # 后两刀一批：第一刀就越过检查点 -> 整批 409，不推进
        with self.assertRaises(SessionConflictError) as ctx:
            submit_cut_batch(sid, self._batch("b2", 3, cuts[3:5]))
        detail = ctx.exception.detail
        self.assertEqual(detail["error"], "blade_change_required")
        self.assertEqual(detail["index"], 0)
        self.assertEqual(detail["cut_length"], lens[3])
        state = get_session_state(sid)
        self.assertEqual(state["version"], 3)
        self.assertEqual(state["completed_cuts"], 3)
        with db.get_conn() as conn:
            self.assertEqual(db.count_session_batches(sid, conn), 1)

        # 批次第二刀才越过：同样整批 409（index 指向越界项）
        with self.assertRaises(SessionConflictError) as ctx:
            submit_cut_batch(sid, self._batch("b3", 3, cuts[3:5]))
        self.assertEqual(ctx.exception.detail["index"], 0)

        # 换刀后同批（新 batch_id、base_version 对齐新状态）顺利应用
        self._blade_change(sess, "bc-1", 3)
        result = submit_cut_batch(sid, self._batch("b4", 4, cuts[3:5]))
        self.assertEqual((result["skipped"], result["applied"]), (0, 2))
        self.assertEqual(result["session"]["version"], 6)
        self.assertEqual(result["session"]["blade_used"],
                         round(lens[3] + lens[4], 6))

    def test_batch_checkpoint_mid_batch_index(self):
        """越过点出现在批次中间：index 指向第一个越界项，整批不推进。"""
        plan, _ = self._new_session()
        cuts = flatten_cuts(plan)
        lens = _lengths(cuts)
        life = lens[0] + lens[1]  # 只够前两刀
        _, sess = self._new_session(blade_life=life)
        sid = sess["session_id"]

        with self.assertRaises(SessionConflictError) as ctx:
            submit_cut_batch(sid, self._batch("b1", 0, cuts[:4]))
        detail = ctx.exception.detail
        self.assertEqual(detail["error"], "blade_change_required")
        self.assertEqual(detail["index"], 2)  # cuts[2] 是第一个越界项
        self.assertEqual(detail["cut_length"], lens[2])
        self.assertEqual(get_session_state(sid)["version"], 0)
        with db.get_conn() as conn:
            self.assertEqual(db.count_session_batches(sid, conn), 0)

    def test_batch_base_version_mapping_across_blade_change(self):
        """换刀也递增版本：批次对账须把 base_version 正确换算成已完成刀数。"""
        plan, _ = self._new_session()
        cuts = flatten_cuts(plan)
        _, sess = self._new_session(blade_life=1e9)
        sid = sess["session_id"]

        self._event(sess, "e0", 0, cuts[0])
        self._event(sess, "e1", 1, cuts[1])      # version=2, done=2
        self._blade_change(sess, "bc-1", 2)      # version=3, done=2

        # 设备离线前看到的是 version=3：批次第一刀对应 cuts[2]
        result = submit_cut_batch(sid, self._batch("b1", 3, cuts[2:6]))
        self.assertEqual((result["skipped"], result["applied"]), (0, 4))
        self.assertEqual(result["session"]["version"], 7)
        self.assertEqual(result["session"]["completed_cuts"], 6)

        # 另一台设备离线前看到的是 version=6（= done 5 + 1 次换刀）：
        # 批次第一刀对应 cuts[5]，对账跳过 1 刀后推进 cuts[6:8]
        result = submit_cut_batch(sid, self._batch("b2", 6, cuts[5:8]))
        self.assertEqual((result["skipped"], result["applied"]), (1, 2))
        self.assertEqual(result["session"]["version"], 9)
        self.assertEqual(result["session"]["completed_cuts"], 8)

        # base_version=2（换刀前）：已完成 8 刀，2 刀批次不足以对账 -> 409
        with self.assertRaises(SessionConflictError) as ctx:
            submit_cut_batch(sid, self._batch("b3", 2, cuts[2:4]))
        detail = ctx.exception.detail
        self.assertEqual(detail["error"], "batch_mismatch")
        self.assertEqual(detail["phase"], "reconcile")
        self.assertEqual(get_session_state(sid)["version"], 9)

        # 幂等重放 b1：仍返回首次响应（version 停留在 7）
        replay = submit_cut_batch(sid, self._batch("b1", 3, cuts[2:6]))
        self.assertEqual(replay["session"]["version"], 7)
        self.assertEqual(get_session_state(sid)["version"], 9)

    def test_batch_skipped_prefix_not_counted_twice(self):
        """对账跳过的刀不重复计切长：只累计实际应用的后缀。"""
        plan, _ = self._new_session()
        cuts = flatten_cuts(plan)
        lens = _lengths(cuts)
        life = sum(lens[:4])
        _, sess = self._new_session(blade_life=life)
        sid = sess["session_id"]

        self._event(sess, "e0", 0, cuts[0])
        self._event(sess, "e1", 1, cuts[1])  # used = l0 + l1
        # 批次覆盖前 4 刀：前 2 刀跳过，只累计 cuts[2]、cuts[3]
        result = submit_cut_batch(sid, self._batch("b1", 0, cuts[:4]))
        self.assertEqual((result["skipped"], result["applied"]), (2, 2))
        self.assertEqual(result["session"]["blade_used"], life)
        self.assertEqual(result["session"]["next_action"], "blade_change")


class TestBladeConcurrency(_DbTestCase):
    def test_concurrent_blade_change_and_cut_single_winner(self):
        """并发换刀与切刀（同版本）：恰一个成功，另一个 version_conflict。"""
        plan, _ = self._new_session()
        cuts = flatten_cuts(plan)
        _, sess = self._new_session(blade_life=1e9)
        sid = sess["session_id"]
        self._event(sess, "e0", 0, cuts[0])  # version=1

        n_threads = 2
        barrier = threading.Barrier(n_threads)
        results, lock = [], threading.Lock()

        def cut_worker():
            req = CutEventRequest(expected_version=1,
                                  idempotency_key="race-cut",
                                  panel_no=cuts[1]["panel_no"],
                                  step=cuts[1]["step"])
            barrier.wait()
            try:
                submit_cut_event(sid, req)
                outcome = ("cut", "applied")
            except SessionConflictError as exc:
                outcome = ("cut", exc.detail["error"])
            with lock:
                results.append(outcome)

        def blade_worker():
            req = BladeChangeRequest(expected_version=1,
                                     idempotency_key="race-blade")
            barrier.wait()
            try:
                change_blade(sid, req)
                outcome = ("blade", "applied")
            except SessionConflictError as exc:
                outcome = ("blade", exc.detail["error"])
            with lock:
                results.append(outcome)

        threads = [threading.Thread(target=cut_worker),
                   threading.Thread(target=blade_worker)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 恰一方成功，另一方版本冲突；版本只推进一次
        applied = [r for r in results if r[1] == "applied"]
        conflicts = [r for r in results if r[1] == "version_conflict"]
        self.assertEqual(len(applied), 1)
        self.assertEqual(len(conflicts), 1)
        state = get_session_state(sid)
        self.assertEqual(state["version"], 2)
        if applied[0][0] == "cut":
            self.assertEqual(state["completed_cuts"], 2)
            self.assertEqual(state["blade_used"],
                             round(cuts[0]["length"] + cuts[1]["length"], 6))
        else:
            self.assertEqual(state["completed_cuts"], 1)
            self.assertEqual(state["blade_used"], 0.0)

    def test_concurrent_blade_changes_single_winner(self):
        """并发换刀（不同幂等键、同版本）：恰一个成功。"""
        plan, sess = self._new_session(blade_life=1000.0)
        sid = sess["session_id"]
        n_threads = 6
        barrier = threading.Barrier(n_threads)
        results, lock = [], threading.Lock()

        def worker(i):
            req = BladeChangeRequest(expected_version=0,
                                     idempotency_key=f"bc-race-{i}")
            barrier.wait()
            try:
                change_blade(sid, req)
                outcome = "applied"
            except SessionConflictError as exc:
                outcome = exc.detail["error"]
            with lock:
                results.append(outcome)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(results.count("applied"), 1)
        self.assertEqual(results.count("version_conflict"), n_threads - 1)
        self.assertEqual(get_session_state(sid)["version"], 1)

    def test_concurrent_same_blade_key_all_get_original(self):
        """并发换刀（同键同请求）：全部拿到首次结果，只应用一次。"""
        plan, sess = self._new_session(blade_life=1000.0, initial_used=100.0)
        sid = sess["session_id"]
        n_threads = 6
        barrier = threading.Barrier(n_threads)
        responses, lock = [], threading.Lock()

        def worker():
            req = BladeChangeRequest(expected_version=0,
                                     idempotency_key="shared-bc")
            barrier.wait()
            with lock:
                responses.append(change_blade(sid, req))

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(responses), n_threads)
        for resp in responses:
            self.assertEqual(resp, responses[0])
        self.assertEqual(responses[0]["version"], 1)
        self.assertEqual(responses[0]["blade_used"], 0.0)
        state = get_session_state(sid)
        self.assertEqual(state["version"], 1)
        self.assertEqual(state["blade_used"], 0.0)

    def test_concurrent_soak_with_blade_changes(self):
        """切刀 + 换刀混合并发跑完整个会话：终态一致，推进守恒。"""
        plan, _ = self._new_session()
        cuts = flatten_cuts(plan)
        lens = _lengths(cuts)
        total = len(cuts)
        self.assertGreaterEqual(total, 10)
        life = lens[0] + lens[1]  # 每把刀至多两刀，强制频繁换刀
        _, sess = self._new_session(blade_life=life)
        sid = sess["session_id"]
        errors, lock = [], threading.Lock()
        done = threading.Event()

        def cut_worker():
            while not done.is_set():
                state = get_session_state(sid)
                if state["completed"]:
                    return
                if state["next_action"] == "blade_change":
                    continue
                nxt = state["next_cut"]
                req = CutEventRequest(
                    expected_version=state["version"],
                    idempotency_key=f"c-{threading.get_ident()}-{state['version']}",
                    panel_no=nxt["panel_no"],
                    step=nxt["step"],
                )
                try:
                    submit_cut_event(sid, req)
                except SessionConflictError as exc:
                    if exc.detail["error"] not in (
                            "version_conflict", "session_completed",
                            "blade_change_required"):
                        with lock:
                            errors.append(exc.detail)
                except Exception as exc:
                    with lock:
                        errors.append(str(exc))

        def blade_worker():
            while not done.is_set():
                state = get_session_state(sid)
                if state["completed"]:
                    return
                if state["next_action"] != "blade_change":
                    continue
                req = BladeChangeRequest(
                    expected_version=state["version"],
                    idempotency_key=f"b-{threading.get_ident()}-{state['version']}")
                try:
                    change_blade(sid, req)
                except SessionConflictError as exc:
                    if exc.detail["error"] not in (
                            "version_conflict", "session_completed"):
                        with lock:
                            errors.append(exc.detail)
                except Exception as exc:
                    with lock:
                        errors.append(str(exc))

        threads = ([threading.Thread(target=cut_worker) for _ in range(3)]
                   + [threading.Thread(target=blade_worker) for _ in range(3)])
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        done.set()
        for t in threads:
            t.join()

        self.assertFalse(any(t.is_alive() for t in threads), "线程未在超时内结束")
        self.assertEqual(errors, [])
        final = get_session_state(sid)
        self.assertTrue(final["completed"])
        self.assertEqual(final["completed_cuts"], total)
        self.assertEqual(final["next_action"], "none")
        with db.get_conn() as conn:
            self.assertEqual(db.count_session_events(sid, conn), total)
            # version = 刀数 + 换刀次数
            n_changes = conn.execute(
                "SELECT COUNT(*) AS c FROM cut_session_blade_changes "
                "WHERE session_id = ?", (sid,)).fetchone()["c"]
        self.assertEqual(final["version"], total + n_changes)
        self.assertGreater(n_changes, 0)


@unittest.skipUnless(HAS_FASTAPI, "未安装 fastapi，跳过 API 层测试")
class TestBladeApi(_DbTestCase):
    def setUp(self):
        super().setUp()
        self.client = TestClient(app)

    def _create_plan(self, req=REQ):
        r = self.client.post("/api/v1/plans", json=req)
        self.assertEqual(r.status_code, 200)
        return r.json()

    def _start(self, plan_id, body=None):
        r = self.client.post(f"/api/v1/plans/{plan_id}/sessions", json=body)
        self.assertEqual(r.status_code, 201)
        return r.json()

    def test_session_flow_with_blade_life_via_api(self):
        plan = self._create_plan()
        cuts = flatten_cuts(plan)
        lens = _lengths(cuts)
        life = lens[0] + lens[1]

        sess = self._start(plan["plan_id"],
                           {"blade_life": life, "initial_used": 0})
        sid = sess["session_id"]
        self.assertEqual(sess["blade_life"], life)
        self.assertEqual(sess["blade_used"], 0.0)
        self.assertEqual(sess["blade_remaining"], life)
        self.assertEqual(sess["next_action"], "cut")

        # 无请求体启动：不跟踪寿命
        plain = self._start(plan["plan_id"])
        self.assertIsNone(plain["blade_life"])
        self.assertIsNone(plain["blade_used"])
        self.assertIsNone(plain["blade_remaining"])
        self.assertEqual(plain["next_action"], "cut")

        # 切两刀到检查点
        for i in range(2):
            r = self.client.post(
                f"/api/v1/sessions/{sid}/events",
                json={"expected_version": i, "idempotency_key": f"e{i}",
                      "panel_no": cuts[i]["panel_no"], "step": cuts[i]["step"]})
            self.assertEqual(r.status_code, 200)
        state = r.json()
        self.assertEqual(state["blade_used"], life)
        self.assertEqual(state["blade_remaining"], 0.0)
        self.assertEqual(state["next_action"], "blade_change")

        # 下一刀越过检查点 -> 409 blade_change_required
        r = self.client.post(
            f"/api/v1/sessions/{sid}/events",
            json={"expected_version": 2, "idempotency_key": "e2",
                  "panel_no": cuts[2]["panel_no"], "step": cuts[2]["step"]})
        self.assertEqual(r.status_code, 409)
        detail = r.json()["detail"]
        self.assertEqual(detail["error"], "blade_change_required")
        self.assertEqual(detail["blade_remaining"], 0.0)
        self.assertEqual(detail["cut_length"], lens[2])

        # 换刀确认 -> 200，已用归零、版本 +1
        r = self.client.post(
            f"/api/v1/sessions/{sid}/blade-changes",
            json={"expected_version": 2, "idempotency_key": "bc-1"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["version"], 3)
        self.assertEqual(body["blade_used"], 0.0)
        self.assertEqual(body["blade_remaining"], life)
        self.assertEqual(body["next_action"], "cut")
        self.assertEqual(body["completed_cuts"], 2)

        # 同键重放：逐字段一致
        r = self.client.post(
            f"/api/v1/sessions/{sid}/blade-changes",
            json={"expected_version": 2, "idempotency_key": "bc-1"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), body)

        # 继续切刀（注意版本已被换刀推进到 3）
        r = self.client.post(
            f"/api/v1/sessions/{sid}/events",
            json={"expected_version": 3, "idempotency_key": "e2",
                  "panel_no": cuts[2]["panel_no"], "step": cuts[2]["step"]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["blade_used"], lens[2])

    def test_blade_change_conflicts_via_api(self):
        plan = self._create_plan()
        sess = self._start(plan["plan_id"], {"blade_life": 1e6})
        sid = sess["session_id"]

        # 版本不符 -> 409 version_conflict
        r = self.client.post(
            f"/api/v1/sessions/{sid}/blade-changes",
            json={"expected_version": 9, "idempotency_key": "bc-x"})
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.json()["detail"]["error"], "version_conflict")

        # 同键不同请求 -> 409 idempotency_key_reuse
        r = self.client.post(
            f"/api/v1/sessions/{sid}/blade-changes",
            json={"expected_version": 0, "idempotency_key": "bc-1"})
        self.assertEqual(r.status_code, 200)
        r = self.client.post(
            f"/api/v1/sessions/{sid}/blade-changes",
            json={"expected_version": 1, "idempotency_key": "bc-1"})
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.json()["detail"]["error"], "idempotency_key_reuse")

        # 未跟踪寿命的会话 -> 409 blade_not_tracked
        plain = self._start(plan["plan_id"])
        r = self.client.post(
            f"/api/v1/sessions/{plain['session_id']}/blade-changes",
            json={"expected_version": 0, "idempotency_key": "bc-1"})
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.json()["detail"]["error"], "blade_not_tracked")

        # 会话不存在 -> 404
        missing = "00000000-0000-0000-0000-000000000000"
        r = self.client.post(
            f"/api/v1/sessions/{missing}/blade-changes",
            json={"expected_version": 0, "idempotency_key": "bc-1"})
        self.assertEqual(r.status_code, 404)

    def test_session_start_validation_422(self):
        plan = self._create_plan()
        url = f"/api/v1/plans/{plan['plan_id']}/sessions"
        # initial_used 不能脱离 blade_life
        r = self.client.post(url, json={"initial_used": 10})
        self.assertEqual(r.status_code, 422)
        # initial_used 超过 blade_life
        r = self.client.post(url, json={"blade_life": 100, "initial_used": 200})
        self.assertEqual(r.status_code, 422)
        # blade_life 必须为正
        r = self.client.post(url, json={"blade_life": 0})
        self.assertEqual(r.status_code, 422)
        r = self.client.post(url, json={"blade_life": -5})
        self.assertEqual(r.status_code, 422)
        # initial_used 不能为负
        r = self.client.post(url, json={"blade_life": 100, "initial_used": -1})
        self.assertEqual(r.status_code, 422)
        # 多余字段
        r = self.client.post(url, json={"blade_life": 100, "bogus": 1})
        self.assertEqual(r.status_code, 422)
        # 合法：initial_used == blade_life（启动即需换刀）
        r = self.client.post(url, json={"blade_life": 100, "initial_used": 100})
        self.assertEqual(r.status_code, 201)
        self.assertEqual(r.json()["next_action"], "blade_change")
        self.assertEqual(r.json()["blade_remaining"], 0.0)

    def test_blade_change_validation_422(self):
        plan = self._create_plan()
        sess = self._start(plan["plan_id"], {"blade_life": 1e6})
        url = f"/api/v1/sessions/{sess['session_id']}/blade-changes"
        # 缺 idempotency_key
        r = self.client.post(url, json={"expected_version": 0})
        self.assertEqual(r.status_code, 422)
        # 缺 expected_version
        r = self.client.post(url, json={"idempotency_key": "k"})
        self.assertEqual(r.status_code, 422)
        # 非法版本
        r = self.client.post(url, json={"expected_version": -1,
                                        "idempotency_key": "k"})
        self.assertEqual(r.status_code, 422)
        # 多余字段
        r = self.client.post(url, json={"expected_version": 0,
                                        "idempotency_key": "k", "bogus": 1})
        self.assertEqual(r.status_code, 422)

    def test_openapi_lists_blade_endpoint(self):
        spec = self.client.get("/openapi.json").json()
        paths = spec["paths"]
        self.assertIn("/api/v1/sessions/{session_id}/blade-changes", paths)
        schemas = spec["components"]["schemas"]
        self.assertIn("BladeChangeRequest", schemas)
        self.assertIn("StartSessionRequest", schemas)
        # 会话响应模型包含刀具字段
        props = schemas["SessionResponse"]["properties"]
        for field in ("blade_life", "blade_used", "blade_remaining",
                      "next_action"):
            self.assertIn(field, props)


if __name__ == "__main__":
    unittest.main()
