#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  Hermes Agent — v1 基礎架構                                ║
║  內容: config + logging_config + circuit_breaker + llm_retry║
║         + state + main(FastAPI) + graph                     ║
║  餵入方式: 複製此檔案內容到 AI 對話視窗                      ║
╚══════════════════════════════════════════════════════════════╝

Hermes Agent 是一個 LLM Agent API server，使用 LangGraph StateGraph
管理多個 agent（orchestrator、research、document、finance、synthesizer），
提供非同步 background task 模式（POST /query → 202 + job_id → GET /query/{job_id} 輪詢結果）。

啟動方式：
  pip install -r requirements.txt
  uvicorn main:app --reload

Requirements 需包含：
  fastapi>=0.110  uvicorn[standard]>=0.29  pydantic>=2.6
  langgraph>=0.2.0  langchain-core>=0.3.0  langchain-anthropic>=0.3.0
  duckduckgo-search>=6.0.0  cachetools>=5.3  tenacity>=8.2
  slowapi>=0.1.9  pydantic-settings>=2.2  pytest>=8.0  pytest-asyncio>=0.24

注意：
  - v2-hermes-agents.py 要與此檔案放在同一目錄
  - 在餵入 AI 時，v1 和 v2 必須合在一起餵
  - agents/orchestrator.py、agents/research.py 等放在 v2
  - v2 的位置：與此檔案同層，路徑為 ./agents/*.py
"""


# ═══════════════════════════════════════════════════════════════
# Part 1 / config.py — pydantic-settings 設定管理
# ═══════════════════════════════════════════════════════════════

from pydantic import field_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Any, Dict, List, Literal


class HermesSettings(BaseSettings):
    """
    環境變數管理。
    啟動時即驗證 ANTHROPIC_API_KEY 是否存在，避免等到呼叫 LLM 才爆炸。
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    anthropic_api_key: str = Field(..., validation_alias="ANTHROPIC_API_KEY")
    hermes_postgres_dsn: str = "postgresql://postgres:postgres@localhost:5432/hermes_db"
    hermes_api_key: Optional[str] = None
    hermes_max_steps: int = 8
    hermes_env: str = "development"

    @field_validator("hermes_max_steps")
    @classmethod
    def max_steps_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("HERMES_MAX_STEPS 必須為正整數")
        return v

    @property
    def require_api_key(self) -> bool:
        """development 環境不強制 API key 驗證，其他環境視 hermes_api_key 是否有設定。"""
        if self.hermes_env == "development":
            return False
        return bool(self.hermes_api_key)


_settings: Optional[HermesSettings] = None


def get_settings() -> HermesSettings:
    global _settings
    if _settings is None:
        _settings = HermesSettings()
    return _settings


# ═══════════════════════════════════════════════════════════════
# Part 2 / logging_config.py — JSON 結構化 logging + ContextVar
# ═══════════════════════════════════════════════════════════════

import json
import logging
import uuid
from contextvars import ContextVar

_request_id: ContextVar[str] = ContextVar("request_id", default="")
_thread_id: ContextVar[str] = ContextVar("thread_id", default="")

_JSON_HANDLER: logging.Handler = None  # type: ignore


def set_request_id(rid: str):
    _request_id.set(rid)


def get_request_id() -> str:
    return _request_id.get()


def set_thread_id(tid: str):
    _thread_id.set(tid)


def get_thread_id() -> str:
    return _thread_id.get()


class JsonFormatter(logging.Formatter):
    """輸出 JSON line 的 log formatter，自動附加 request_id / thread_id。"""
    def format(self, record: logging.LogRecord) -> str:
        obj = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        rid = get_request_id()
        if rid:
            obj["request_id"] = rid
        tid = get_thread_id()
        if tid:
            obj["thread_id"] = tid
        for key, value in record.__dict__.items():
            if key not in ("args", "asctime", "created", "exc_info", "exc_text",
                           "filename", "funcName", "levelname", "levelno", "lineno",
                           "module", "msecs", "message", "msg", "name", "pathname",
                           "process", "processName", "relativeCreated", "stack_info",
                           "thread", "threadName"):
                obj[key] = str(value)
        return json.dumps(obj, ensure_ascii=False)


class RequestTracingMiddleware:
    """
    FastAPI ASGI middleware：
    - 自動產生或沿用 caller 提供的 X-Request-ID
    - 注入 ContextVar _request_id
    - 在 response header 回傳 X-Request-ID
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        rid = None
        for k, v in scope.get("headers", []):
            if k.lower() == b"x-request-id":
                rid = v.decode()
                break

        if not rid:
            rid = str(uuid.uuid4())

        set_request_id(rid)

        async def send_with_header(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"X-Request-ID", rid.encode()))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_header)


def setup_logging():
    """初始化 JSON logger（idempotent）。"""
    global _JSON_HANDLER
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    if _JSON_HANDLER:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    _JSON_HANDLER = handler


# ═══════════════════════════════════════════════════════════════
# Part 3 / circuit_breaker.py — 三態斷路器
# ═══════════════════════════════════════════════════════════════

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(Exception):
    """斷路器為 OPEN 狀態時拒絕請求"""
    pass


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    success_threshold: int = 2


class CircuitBreaker:
    """
    三態斷路器（CLOSED / OPEN / HALF_OPEN）。

    流程：
      CLOSED → (連續失敗達 threshold) → OPEN
      OPEN → (recovery_timeout 經過) → HALF_OPEN
      HALF_OPEN → (連續成功達 success_threshold) → CLOSED
      HALF_OPEN → (一次失敗) → OPEN

    HALF_OPEN 時限制最多 half_open_max_calls 個併發請求。
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN and self._recovery_timeout_elapsed():
            logger_cb.info(f"breaker[{self.name}] 從 OPEN 轉為 HALF_OPEN")
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 0
            self._success_count = 0
        return self._state

    def _recovery_timeout_elapsed(self) -> bool:
        return (time.monotonic() - self._last_failure_time) >= self.config.recovery_timeout

    async def call(self, func: Callable[[], Any], *args, **kwargs) -> Any:
        current_state = self.state
        if current_state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(f"breaker[{self.name}] OPEN，拒絕請求")

        if current_state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.config.half_open_max_calls:
                raise CircuitBreakerOpenError(f"breaker[{self.name}] HALF_OPEN（試探中），已達上限")
            self._half_open_calls += 1

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure()
            raise

    def _on_success(self):
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                logger_cb.info(f"breaker[{self.name}] 從 HALF_OPEN 恢復為 CLOSED")
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0
                self._success_count = 0
        else:
            self._failure_count = 0

    def _on_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._state == CircuitState.HALF_OPEN:
            logger_cb.info(f"breaker[{self.name}] HALF_OPEN 中失敗，退回 OPEN")
            self._state = CircuitState.OPEN
            self._half_open_calls = 0
            self._success_count = 0
        elif self._failure_count >= self.config.failure_threshold:
            logger_cb.warning(f"breaker[{self.name}] 連續失敗 {self._failure_count} 次，轉為 OPEN")
            self._state = CircuitState.OPEN
            self._half_open_calls = 0
            self._success_count = 0


class CircuitBreakerRegistry:
    """全域 circuit breaker registry，依名稱管理獨立 breaker。"""

    def __init__(self, default_config: Optional[CircuitBreakerConfig] = None):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._default_config = default_config or CircuitBreakerConfig()

    def get(self, name: str) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, config=self._default_config)
        return self._breakers[name]

    def status_report(self) -> dict[str, str]:
        return {name: cb.state.value for name, cb in self._breakers.items()}


logger_cb = logging.getLogger("hermes.circuit_breaker")
mcp_circuit_registry = CircuitBreakerRegistry()


# ═══════════════════════════════════════════════════════════════
# Part 4 / llm_retry.py — Exponential Backoff Retry
# ═══════════════════════════════════════════════════════════════

import asyncio
import random
from langchain_core.messages import BaseMessage


def _is_retryable(exc: Exception) -> bool:
    """判斷例外是否可重試（429/503/overloaded 等 transient error）。"""
    msg = str(exc).lower()
    return any(keyword in msg for keyword in [
        "429", "503", "overloaded", "rate limit", "service unavailable"
    ])


async def llm_ainvoke(
    llm: Any,
    messages: List[BaseMessage],
    max_attempts: int = 4,
    wait_min: float = 1.0,
    wait_max: float = 30.0,
) -> Any:
    """
    包裝 LLM.ainvoke，遇上 retryable 錯誤時 exponential backoff + jitter。
    non-retryable 錯誤（ValueError 等）直接拋出，不浪費重試次數。
    """
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await llm.ainvoke(messages)
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc):
                raise
            if attempt == max_attempts:
                raise

            delay = min(wait_min * (2 ** (attempt - 1)) + random.uniform(0, 0.5), wait_max)
            logger_llm.warning(
                f"LLM 呼叫失敗（第 {attempt} 次），{delay:.1f}s 後重試: {exc}"
            )
            await asyncio.sleep(delay)

    raise last_exc  # type: ignore


logger_llm = logging.getLogger("hermes.llm_retry")


# ═══════════════════════════════════════════════════════════════
# Part 5 / state.py — LangGraph 狀態定義
# ═══════════════════════════════════════════════════════════════

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class HermesState(TypedDict, total=False):
    """
    LangGraph 的 shared state schema。

    欄位說明：
      user_input       — 使用者原始輸入
      messages         — LangChain message 歷史（預留給多輪對話）
      completed_agents — 已完成 agent 名稱列表（防止重複分配）
      agent_outputs    — 各 agent 的輸出 dict，key = agent 名稱
      errors           — 流程中發生的錯誤列表
      step_count       — 目前執行的步數（orchestrator 每次呼叫 +1）
      max_steps        — 最大步數防護
      final_answer     — Synthesizer 產生的最終回覆
      next_agent       — Orchestrator 決定的下一個 agent 名稱
    """
    user_input: str
    messages: List[Any]
    completed_agents: List[str]
    agent_outputs: Dict[str, Any]
    errors: List[str]
    step_count: int
    max_steps: int
    final_answer: str
    next_agent: str


# ═══════════════════════════════════════════════════════════════
# Part 6 / graph.py — LangGraph StateGraph 組裝
# ═══════════════════════════════════════════════════════════════

from langgraph.graph import StateGraph, END


logger_graph = logging.getLogger("hermes.graph")


def build_graph_builder() -> StateGraph:
    """
    組裝節點與邊（不含 checkpointer）。
    測試可用 MemorySaver 取代 Postgres；正式環境請用 compile_graph() 接 checkpointer。

    節點：
      orchestrator — 條件路由中樞
      research     — 搜尋引擎查詢
      document     — 結構化文件輸出
      finance      — 財務分析計算
      synthesizer  — 最終彙整輸出

    邊：
      orchestrator → (research|document|finance|synthesizer)  條件路由
      research/document/finance → orchestrator                 回圈（直到轉 synthesizer）
      synthesizer → END                                        流程結束
    """
    # lazy import：v2 的 agent 模組在執行時期才載入
    from agents.orchestrator import orchestrator_node, route_after_orchestrator
    from agents.research import research_node
    from agents.document import document_node
    from agents.finance import finance_node
    from agents.synthesizer import synthesizer_node

    builder = StateGraph(HermesState)

    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("research", research_node)
    builder.add_node("document", document_node)
    builder.add_node("finance", finance_node)
    builder.add_node("synthesizer", synthesizer_node)

    builder.set_entry_point("orchestrator")

    builder.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "research": "research",
            "document": "document",
            "finance": "finance",
            "synthesizer": "synthesizer",
        },
    )

    builder.add_edge("research", "orchestrator")
    builder.add_edge("document", "orchestrator")
    builder.add_edge("finance", "orchestrator")
    builder.add_edge("synthesizer", END)

    return builder


async def compile_graph(checkpointer=None):
    """編譯 LangGraph。傳入 checkpointer 則啟用狀態持久化。"""
    builder = build_graph_builder()
    graph = builder.compile(checkpointer=checkpointer)
    return graph


async def build_hermes_graph(postgres_conn_string: str):
    """使用 AsyncPostgresSaver 建立生產級 graph（含 Postgres 持久化）。"""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    builder = build_graph_builder()
    async with AsyncPostgresSaver.from_conn_string(postgres_conn_string) as checkpointer:
        await checkpointer.setup()
        graph = builder.compile(checkpointer=checkpointer)
        return graph, checkpointer


async def run_hermes_query(
    graph,
    user_input: str,
    thread_id: str,
    max_steps: int = 8,
) -> dict:
    """
    執行單次查詢。thread_id 用於 checkpoint 區分不同對話/任務。

    回傳：
      final_answer    — Synthesizer 產生的最終回覆
      agent_outputs   — 各 agent 的輸出 dict
      errors          — 流程中發生的錯誤列表
      completed_agents — 完成的 agent 列表
    """
    config = {"configurable": {"thread_id": thread_id}}
    initial_state: HermesState = {
        "user_input": user_input,
        "messages": [],
        "completed_agents": [],
        "agent_outputs": {},
        "errors": [],
        "step_count": 0,
        "max_steps": max_steps,
    }

    final_state = await graph.ainvoke(initial_state, config=config)
    return {
        "final_answer": final_state.get("final_answer", ""),
        "agent_outputs": final_state.get("agent_outputs", {}),
        "errors": final_state.get("errors", []),
        "completed_agents": final_state.get("completed_agents", []),
    }


# ═══════════════════════════════════════════════════════════════
# Part 7 / main.py — FastAPI 入口
# ═══════════════════════════════════════════════════════════════

from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logger_main = logging.getLogger("hermes.main")

# ── Rate Limiter ───────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── 背景 Task 狀態存放（in-memory；多 worker 環境請換成 Redis） ──
_jobs: dict[str, dict] = {}
_graph_holder: dict = {}


# ── Auth Dependency ────────────────────────────────────────────
async def require_api_key(x_api_key: str | None = Header(default=None)):
    settings = get_settings()
    if not settings.require_api_key:
        return
    if x_api_key != settings.hermes_api_key:
        raise HTTPException(status_code=401, detail="缺少或錯誤的 X-API-Key")


# ── Lifespan ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    graph = await compile_graph()
    _graph_holder["graph"] = graph
    logger_main.info("hermes_started", extra={"env": settings.hermes_env})
    yield
    logger_main.info("hermes_stopped")


# ── App ────────────────────────────────────────────────────────
app = FastAPI(title="Hermes Agent", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(RequestTracingMiddleware)


# ── Schemas ────────────────────────────────────────────────────
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


# ── Background Worker ──────────────────────────────────────────
async def _run_job(job_id: str, thread_id: str, user_input: str, max_steps: int):
    _jobs[job_id]["status"] = "running"
    set_thread_id(thread_id)
    try:
        graph = _graph_holder["graph"]
        result = await run_hermes_query(graph, user_input, thread_id, max_steps=max_steps)
        _jobs[job_id].update({"status": "done", **result})
    except Exception as exc:
        logger_main.exception("job_failed", extra={"job_id": job_id})
        _jobs[job_id].update({"status": "error", "error_detail": str(exc)})


# ── Endpoints ──────────────────────────────────────────────────
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

    logger_main.info("job_queued", extra={"job_id": job_id, "thread_id": thread_id})
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
