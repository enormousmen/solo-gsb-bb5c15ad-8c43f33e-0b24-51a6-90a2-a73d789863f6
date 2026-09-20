"""FastAPI 入口。

启动即初始化 SQLite（见 lifespan）；接口：
  GET  /health
  POST /api/v1/plans                          创建/获取排样方案（幂等）
  GET  /api/v1/plans/{plan_id}                查询单个方案
  GET  /api/v1/plans                          查询方案列表（分页）
  POST /api/v1/plans/{plan_id}/gcode          导出方案的 G-code
  POST /api/v1/plans/{plan_id}/sessions       从方案启动切割执行会话
                                              （可选 blade_life/initial_used
                                              开启刀具寿命检查点）
  GET  /api/v1/sessions/{session_id}          查询切割会话状态
  POST /api/v1/sessions/{session_id}/events   提交“完成当前刀”事件（幂等）
  POST /api/v1/sessions/{session_id}/event-batches
                                              离线回传切刀批次（对账 + 幂等）
  POST /api/v1/sessions/{session_id}/blade-changes
                                              确认换刀（已用切长归零 + 版本递增）
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from . import database as db
from .models import (
    BladeChangeRequest,
    CutBatchRequest,
    CutBatchResponse,
    CutEventRequest,
    GcodeRequest,
    GcodeResponse,
    HealthOut,
    PlanListResponse,
    PlanRequest,
    PlanResponse,
    PlanSummary,
    SessionResponse,
    StartSessionRequest,
)
from .services import (
    PlanGeometryError,
    SessionConflictError,
    change_blade,
    create_plan,
    export_gcode,
    get_session_state,
    start_session,
    submit_cut_batch,
    submit_cut_event,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化数据库（表结构幂等）
    db.init_db()
    yield


app = FastAPI(
    title="板材直切排样 API",
    description=(
        "面向小型加工坊的矩形零件母板直切（guillotine）排样服务。"
        "仅生成贯穿当前余料区域的直切方案，锯缝占位，"
        "相同输入返回稳定且幂等的结果。"
        "支持从已保存方案启动持久化的切割执行会话：按 panel_no/step 顺序"
        "逐刀汇报进度，乐观锁版本 + 幂等键在 SQLite 事务中校验，"
        "并发与重试不会破坏执行状态。设备离线期间完成的切刀可按批次回传："
        "服务端在单事务中将 base_version 之后已完成区间与批次前缀逐项对账，"
        "一致前缀仅跳过，其余后缀连续匹配当前方案后一次性推进。"
        "可选的刀具寿命检查点按累计切长（mm）预判：下一刀会超过寿命则"
        "拒绝推进并提示换刀；换刀确认在同事务中把已用切长归零并递增版本，"
        "并发换刀与切刀只有一方成功。"
    ),
    version="1.3.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthOut, tags=["meta"])
def health() -> HealthOut:
    return HealthOut(status="ok")


@app.post(
    "/api/v1/plans",
    response_model=PlanResponse,
    tags=["plans"],
    summary="创建排样方案",
)
def post_plan(req: PlanRequest) -> PlanResponse:
    payload = create_plan(req.model_dump())
    return JSONResponse(payload)


@app.get(
    "/api/v1/plans/{plan_id}",
    response_model=PlanResponse,
    tags=["plans"],
    summary="按编号查询方案",
)
def get_plan(plan_id: str) -> PlanResponse:
    with db.get_conn() as conn:
        payload = db.get_plan(plan_id, conn)
    if payload is None:
        raise HTTPException(status_code=404, detail="方案不存在")
    return JSONResponse(payload)


@app.post(
    "/api/v1/plans/{plan_id}/gcode",
    response_model=GcodeResponse,
    tags=["plans"],
    summary="导出方案的 G-code",
)
def post_plan_gcode(plan_id: str, req: GcodeRequest) -> GcodeResponse:
    try:
        payload = export_gcode(plan_id, req)
    except PlanGeometryError as exc:
        # 校验失败：结构化错误，不输出任何部分 G-code
        raise HTTPException(
            status_code=409,
            detail={
                "error": "invalid_plan_geometry",
                "message": "方案切割线未通过导出前几何校验，未生成任何 G-code",
                "violations": exc.violations,
            },
        ) from exc
    if payload is None:
        raise HTTPException(status_code=404, detail="方案不存在")
    return JSONResponse(payload)


@app.get(
    "/api/v1/plans",
    response_model=PlanListResponse,
    tags=["plans"],
    summary="查询方案列表",
)
def list_plans(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> PlanListResponse:
    with db.get_conn() as conn:
        total = db.count_plans(conn)
        rows = db.list_plans(limit, offset, conn)
    return PlanListResponse(
        total=total,
        plans=[PlanSummary(**row) for row in rows],
    )


@app.post(
    "/api/v1/plans/{plan_id}/sessions",
    response_model=SessionResponse,
    status_code=201,
    tags=["sessions"],
    summary="从已保存方案启动切割执行会话（可选刀具寿命检查点）",
)
def post_plan_session(plan_id: str,
                      req: Optional[StartSessionRequest] = None
                      ) -> SessionResponse:
    blade_life = req.blade_life if req is not None else None
    initial_used = req.initial_used if req is not None else 0.0
    payload = start_session(plan_id, blade_life=blade_life,
                            initial_used=initial_used)
    if payload is None:
        raise HTTPException(status_code=404, detail="方案不存在")
    return JSONResponse(payload, status_code=201)


@app.get(
    "/api/v1/sessions/{session_id}",
    response_model=SessionResponse,
    tags=["sessions"],
    summary="查询切割会话状态",
)
def get_session(session_id: str) -> SessionResponse:
    payload = get_session_state(session_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return JSONResponse(payload)


@app.post(
    "/api/v1/sessions/{session_id}/events",
    response_model=SessionResponse,
    tags=["sessions"],
    summary="提交“完成当前刀”事件（版本 + 幂等键，事务校验）",
)
def post_session_event(session_id: str, req: CutEventRequest) -> SessionResponse:
    try:
        payload = submit_cut_event(session_id, req)
    except SessionConflictError as exc:
        # 版本/当前刀/幂等键冲突或会话已完成：状态未推进
        raise HTTPException(status_code=409, detail=exc.detail) from exc
    if payload is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return JSONResponse(payload)


@app.post(
    "/api/v1/sessions/{session_id}/event-batches",
    response_model=CutBatchResponse,
    tags=["sessions"],
    summary="离线回传切刀批次（对账 + 版本/幂等校验，事务推进）",
)
def post_session_event_batch(session_id: str,
                             req: CutBatchRequest) -> CutBatchResponse:
    try:
        payload = submit_cut_batch(session_id, req)
    except SessionConflictError as exc:
        # 幂等键冲突/基线版本越界/对账或后缀不匹配/越过寿命检查点：整批未推进
        raise HTTPException(status_code=409, detail=exc.detail) from exc
    if payload is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return JSONResponse(payload)


@app.post(
    "/api/v1/sessions/{session_id}/blade-changes",
    response_model=SessionResponse,
    tags=["sessions"],
    summary="确认换刀（版本 + 幂等键，事务中将已用切长归零并递增版本）",
)
def post_session_blade_change(session_id: str,
                              req: BladeChangeRequest) -> SessionResponse:
    try:
        payload = change_blade(session_id, req)
    except SessionConflictError as exc:
        # 版本/幂等键冲突、会话已完成或未跟踪刀具寿命：未确认换刀
        raise HTTPException(status_code=409, detail=exc.detail) from exc
    if payload is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return JSONResponse(payload)
