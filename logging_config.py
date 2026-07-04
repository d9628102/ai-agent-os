"""JSON 結構化 logging + Request ID / Thread ID ContextVar tracing。"""

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
        # 把 extra 參數也放進去
        for key, value in record.__dict__.items():
            if key not in ("args", "asctime", "created", "exc_info", "exc_text",
                           "filename", "funcName", "levelname", "levelno", "lineno",
                           "module", "msecs", "message", "msg", "name", "pathname",
                           "process", "processName", "relativeCreated", "stack_info",
                           "thread", "threadName"):
                obj[key] = str(value)
        return json.dumps(obj, ensure_ascii=False)


class RequestTracingMiddleware:
    """FastAPI middleware：自動產生 X-Request-ID 並注入 ContextVar。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        # headers 在 ASGI scope 中是 bytes
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
    """初始化 JSON logger。重複呼叫為安全（handler 不重複加）。"""
    global _JSON_HANDLER
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    if _JSON_HANDLER:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    _JSON_HANDLER = handler
