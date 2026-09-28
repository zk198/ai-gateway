from __future__ import annotations

import logging
import time
import uuid
from contextvars import ContextVar

from starlette.requests import Request

_request_id: ContextVar[str | None] = ContextVar("ai_request_id", default=None)


def set_request_id(value: str):
    return _request_id.set(value)


def reset_request_id(token) -> None:
    _request_id.reset(token)


def request_id() -> str | None:
    return _request_id.get()


def now() -> float:
    return time.perf_counter()


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def audit_event(event: str, *, operation: str, outcome: str, **fields: object) -> None:
    safe = {"request_id": request_id(), "event": event, "operation": operation, "outcome": outcome, **fields}
    logger = logging.getLogger("ai_gateway.audit")
    logger.info("audit_event %s", safe)


def incoming_request_id(request: Request) -> str:
    value = request.headers.get("X-Request-ID", "").strip()
    return value[:128] if value else uuid.uuid4().hex
