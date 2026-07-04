"""
main.py
FastAPI 入口 — 含：
  - pydantic-settings 設定管理
  - API Key 驗證
  - Rate Limiting (slowapi)
  - Request ID tracing middleware
  - 結構化 JSON logging
  - 背景 Task 模式（POST /query 立即回 job_id，GET /query/{job_id} 輪詢結果）
"""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from circuit_breaker import mcp_circuit_registry
from config import get_settings
from graph import compile_graph, run_hermes_query
from logging_config import RequestTracingMiddleware, set_thread_id, setup_logging

logger = logging.getLogger("hermes.main")

# ── Rate Limiter（依 IP，可改成 API Key） ─────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── 背景 Task 狀態存放（in-memory；多 worker 環境請換成 Redis） ─────────────────
_jobs: dict[str, dict] = {}
_graph_holder: dict = {}


# ── Auth Dependency ───────────────────────────────────────────────────────────
async def require_api_key(x_api_key: str | None = Header(default=None)):
    settings = get_settings()
    if not settings.require_api_key:
        return
    if x_api_key != settings.hermes_api_key:
        raise HTTPException(status_code=401, detail="缺少或錯誤的 X-API-Key")


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    graph = await compile_graph()
    _graph_holder["graph"] = graph
    logger.info("hermes_started", extra={"env": settings.hermes_env})
    yield
    logger.info("hermes_stopped")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Hermes Agent", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(RequestTracingMiddleware)


# ── Schemas ───────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    user_input: str
    thread_id: str | None = None
    max_steps: int | None = None


class JobAccepted(BaseModel):
    job_id: str
    thread_id: str
    status: Literal["queued"] = "queued"


class JobStatus(BaseModel):
    job_id: str
    thread_id: str
    status: Literal["queued", "running", "done", "error"]
    final_answer: str | None = None
    agent_outputs: dict | None = None
    errors: list[str] | None = None
    completed_agents: list[str] | None = None
    error_detail: str | None = None


# ── Background Worker ─────────────────────────────────────────────────────────
async def _run_job(job_id: str, thread_id: str, user_input: str, max_steps: int):
    _jobs[job_id]["status"] = "running"
    set_thread_id(thread_id)
    try:
        graph = _graph_holder["graph"]
        result = await run_hermes_query(graph, user_input, thread_id, max_steps=max_steps)
        _jobs[job_id].update({"status": "done", **result})
    except Exception as exc:
        logger.exception("job_failed", extra={"job_id": job_id})
        _jobs[job_id].update({"status": "error", "error_detail": str(exc)})


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post(
    "/query",
    response_model=JobAccepted,
    dependencies=[Depends(require_api_key)],
    status_code=202,
)
@limiter.limit("30/minute")
async def submit_query(request: Request, req: QueryRequest):
    """
    立即回 202 + job_id，查詢在背景執行。
    用 GET /query/{job_id} 輪詢結果，避免長查詢卡住 API worker。
    """
    graph = _graph_holder.get("graph")
    if graph is None:
        raise HTTPException(status_code=503, detail="Graph 尚未初始化完成")

    settings = get_settings()
    job_id = str(uuid.uuid4())
    thread_id = req.thread_id or str(uuid.uuid4())
    max_steps = req.max_steps or settings.hermes_max_steps

    _jobs[job_id] = {"status": "queued", "thread_id": thread_id}

    import asyncio
    asyncio.create_task(_run_job(job_id, thread_id, req.user_input, max_steps))

    logger.info("job_queued", extra={"job_id": job_id, "thread_id": thread_id})
    return JobAccepted(job_id=job_id, thread_id=thread_id)


@app.get(
    "/query/{job_id}",
    response_model=JobStatus,
    dependencies=[Depends(require_api_key)],
)
async def get_query_result(job_id: str):
    """輪詢查詢結果。status: queued → running → done | error"""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="找不到此 job_id")
    return JobStatus(job_id=job_id, **job)


@app.get("/health/circuits", dependencies=[Depends(require_api_key)])
async def circuit_status():
    return mcp_circuit_registry.status_report()


@app.get("/health")
async def health():
    return {"status": "ok"}
