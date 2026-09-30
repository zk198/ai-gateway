from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
import logging
import os
import time
import uuid

import httpx

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from .agent_client import AgentClient
from .laya_client import LayaClient
from .auth import authenticate, authenticate_diagnostics
from .db import SessionStore
from .models import AnswerRequest, AnswerResponse, ChatCompletionRequest, Citation, ConversationCreateRequest, ConversationResponse, GatewaySettings, SearchRequest
from .service import RAGService
from .trace_store import TraceStore
from .observability import audit_event, elapsed_ms, incoming_request_id, request_id, reset_request_id, set_request_id

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

settings = GatewaySettings(
    retrieval_url=os.getenv("RAG_RETRIEVAL_URL", "http://rag-retrieval:8100"),
    ingestion_url=os.getenv("RAG_INGESTION_URL", "http://pst-agent:8000"),
    agent_url=os.getenv("AI_AGENT_URL", "http://agent-core:8000"),
    postgres_dsn=os.getenv("AI_POSTGRES_DSN", os.getenv("RAG_POSTGRES_DSN", "postgresql://rag:rag@postgres:5432/rag")),
    timeout_seconds=float(os.getenv("RAG_DOWNSTREAM_TIMEOUT_SECONDS", "60")),
    agent_timeout_seconds=float(os.getenv("AI_AGENT_TIMEOUT_SECONDS", "120")),
    laya_enabled=os.getenv("LAYA_ENABLED", "false").lower() in {"1", "true", "yes", "on"},
    laya_url=os.getenv("LAYA_URL", "http://laya:8000"),
    laya_timeout_seconds=float(os.getenv("LAYA_TIMEOUT_SECONDS", "5")),
    laya_api_key=os.getenv("LAYA_API_KEY") or None,
)
service = RAGService(settings)
agent_client = AgentClient(settings.agent_url, settings.agent_timeout_seconds)
laya_client = LayaClient(settings.laya_url, settings.laya_timeout_seconds, settings.laya_api_key)
session_store = SessionStore(settings.postgres_dsn)
trace_store = TraceStore(int(os.getenv("AI_TRACE_STORE_MAX_TRACES", "1000")))
app = FastAPI(title="AI Gateway", version="0.3.0")
MAX_REQUEST_BYTES = int(os.getenv("AI_GATEWAY_MAX_REQUEST_BYTES", str(25 * 1024 * 1024)))

UI_ORIGINS = [origin.strip() for origin in os.getenv("AI_UI_ORIGINS", "http://localhost:3000").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware, allow_origins=UI_ORIGINS, allow_credentials=True, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["Authorization", "Content-Type", "X-Request-ID"]
)

@app.middleware("http")
async def request_context(request: Request, call_next):
    value = incoming_request_id(request)
    token = set_request_id(value)
    started = time.perf_counter()
    try:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BYTES:
            return Response(content='{"detail":"request body too large"}', status_code=413, media_type="application/json", headers={"X-Request-ID": value})
        response = await call_next(request)
        response.headers["X-Request-ID"] = value
        logger.info("gateway_request request_id=%s method=%s path=%s status=%s gateway_ms=%.1f", value, request.method, request.url.path, response.status_code, elapsed_ms(started))
        return response
    except Exception:
        logger.exception("gateway_request_failed request_id=%s method=%s path=%s gateway_ms=%.1f", value, request.method, request.url.path, elapsed_ms(started))
        raise
    finally:
        reset_request_id(token)

@app.on_event("shutdown")
async def shutdown() -> None:
    await session_store.close()

def downstream_error(exc: Exception) -> HTTPException:
    if isinstance(exc, TimeoutError):
        return HTTPException(504, "downstream timeout")
    if isinstance(exc, ConnectionError):
        return HTTPException(502, "downstream unavailable")
    return HTTPException(502, "downstream request failed")

def store_downstream_trace(exc: Exception, tenant_id: str, user_id: str) -> None:
    response = getattr(exc, "response", None)
    if response is None:
        return
    try:
        payload = response.json()
    except (ValueError, json.JSONDecodeError):
        return
    detail = payload.get("detail") if isinstance(payload, dict) else None
    trace = detail.get("trace") if isinstance(detail, dict) else None
    if isinstance(trace, dict) and trace.get("trace_id"):
        trace_store.put(trace, tenant_id=tenant_id, user_id=user_id)

@app.get("/health", tags=["internal"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/systemone", tags=["diagnostics"])
async def systemone(request: Request, body: dict) -> dict:
    """Privileged gateway boundary for direct System-1 decisions.

    Browser code never connects to Laya directly. This endpoint is intentionally
    protected by the existing diagnostics capability and remains disabled unless
    LAYA_ENABLED is true.
    """
    tenant, user = authenticate_diagnostics(request)
    if not settings.laya_enabled:
        raise HTTPException(status_code=503, detail="Laya disabled")
    state = body.get("state")
    questions = body.get("questions")
    if state is None or not isinstance(questions, dict) or not questions:
        raise HTTPException(status_code=422, detail="state and non-empty questions are required")
    started = time.perf_counter()
    try:
        result = await laya_client.decide(state, questions, model=body.get("model"))
    except Exception as exc:
        audit_event("laya_decision", operation="systemone", outcome="failed", error=type(exc).__name__)
        raise downstream_error(exc) from exc
    latency = elapsed_ms(started)
    trace_id = uuid.uuid4().hex
    trace = {
        "schema_version": "1.0",
        "trace_id": trace_id,
        "request_id": request_id(),
        "started_at": datetime.now(UTC).isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "duration_ms": latency,
        "status": "completed",
        "error": None,
        "logs": [],
        "metrics": {"trace_events": 1, "laya_ms": latency},
        "trace": [{
            "event_id": uuid.uuid4().hex,
            "parent_id": None,
            "sequence": 1,
            "kind": "system1",
            "stage": "laya",
            "name": "laya.systemone",
            "status": "completed",
            "started_at": datetime.now(UTC).isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
            "duration_ms": latency,
            "payload": {
                "routing": result.get("routing"),
                "answers": result.get("answers"),
                "request_id": request_id(),
            },
        }],
    }
    trace_store.put(trace, tenant_id=tenant, user_id=user)
    audit_event("laya_decision", operation="systemone", outcome="succeeded", latency_ms=latency)
    return {"answers": result.get("answers", {}), "routing": result.get("routing"), "trace_id": trace_id}


@app.get("/api/v1/ops/status")
async def ops_status(request: Request) -> dict[str, object]:
    authenticate(request)
    timeout = float(os.getenv("AI_GATEWAY_OPS_PROBE_TIMEOUT_SECONDS", "2"))

    async def probe(name: str, url: str) -> tuple[str, dict[str, object]]:
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url)
            return name, {"status": "ok" if response.is_success else "error", "latency_ms": elapsed_ms(started), "http_status": response.status_code}
        except httpx.TimeoutException:
            return name, {"status": "timeout", "latency_ms": elapsed_ms(started)}
        except httpx.RequestError:
            return name, {"status": "unavailable", "latency_ms": elapsed_ms(started)}

    async def postgres_probe() -> tuple[str, dict[str, object]]:
        started = time.perf_counter()
        try:
            await session_store.conversation_exists("00000000-0000-0000-0000-000000000000", "__ops__", "__ops__")
            return "postgres", {"status": "ok", "latency_ms": elapsed_ms(started)}
        except Exception:
            return "postgres", {"status": "unavailable", "latency_ms": elapsed_ms(started)}

    checks = await asyncio.gather(
        postgres_probe(),
        probe("agent", f"{settings.agent_url.rstrip('/')}/api/v1/ready"),
        probe("retrieval", f"{settings.retrieval_url.rstrip('/')}/health"),
        probe("ingestion", f"{settings.ingestion_url.rstrip('/')}/health"),
    )
    dependencies = dict(checks)
    overall = "ok" if all(item["status"] == "ok" for item in dependencies.values()) else "degraded"
    return {"status": overall, "request_id": request_id(), "dependencies": dependencies}

@app.get("/ready", tags=["internal"])
async def ready() -> dict[str, str]:
    started = time.perf_counter()
    try:
        await session_store.conversation_exists("00000000-0000-0000-0000-000000000000", "__readiness__", "__readiness__")
    except Exception as exc:
        logger.warning("gateway_readiness_failed request_id=%s stage=postgres error=%s", request_id(), type(exc).__name__)
        raise HTTPException(503, "postgres unavailable") from exc
    logger.info("gateway_readiness request_id=%s postgres_ms=%.1f", request_id(), elapsed_ms(started))
    return {"status": "ready"}


@app.post("/api/v1/conversations", response_model=ConversationResponse)
async def create_conversation(body: ConversationCreateRequest, request: Request) -> ConversationResponse:
    tenant, user = authenticate(request)
    conversation = await session_store.create_conversation(tenant, user, body.title)
    return ConversationResponse(
        id=conversation.id, title=conversation.title,
        created_at=conversation.created_at.isoformat(), updated_at=conversation.updated_at.isoformat()
    )

@app.get("/api/v1/conversations", response_model=list[ConversationResponse])
async def list_conversations(request: Request, limit: int = 50) -> list[ConversationResponse]:
    tenant, user = authenticate(request)
    if limit < 1 or limit > 100:
        raise HTTPException(400, "limit must be between 1 and 100")
    rows = await session_store.list_conversations(tenant, user, limit)
    return [
        ConversationResponse(
            id=row.id, title=row.title,
            created_at=row.created_at.isoformat(), updated_at=row.updated_at.isoformat()
        ) for row in rows
    ]

@app.post("/v1/chat/completions", tags=["internal"])
async def chat_completions(body: ChatCompletionRequest, request: Request) -> dict:
    tenant, user = authenticate(request)
    conversation_id = body.conversation_id
    if conversation_id is None:
        conversation_id = (await session_store.create_conversation(tenant, user, None)).id
    elif not await session_store.conversation_exists(conversation_id, tenant, user):
        raise HTTPException(404, "conversation not found")

    history = await session_store.get_history(conversation_id, tenant, user)
    incoming = [message.model_dump() for message in body.messages]

    if history and incoming[: len(history)] == history:
        new_messages = incoming[len(history) :]
        messages = incoming
    elif conversation_id and all(message["role"] == "user" for message in incoming):
        new_messages = incoming
        messages = history + incoming
    elif history:
        raise HTTPException(409, "messages must extend the stored conversation history")
    else:
        new_messages = incoming
        messages = incoming

    if not new_messages:
        raise HTTPException(400, "no new messages supplied")

    if body.stream:
        raise HTTPException(400, "streaming is not yet enabled at the gateway boundary")

    try:
        result = await agent_client.chat(messages, body.model)
    except Exception as exc:
        store_downstream_trace(exc, tenant, user)
        raise downstream_error(exc) from exc

    for message in new_messages:
        await session_store.append_message(conversation_id, tenant, user, message["role"], message["content"])
    content = str(result.get("content", ""))
    await session_store.append_message(conversation_id, tenant, user, "assistant", content)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model or "default",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "conversation_id": conversation_id,
        "usage": {"iterations": result.get("iterations", 0), "tool_calls": result.get("tool_calls", 0)},
    }

def _conversation_messages(request: AnswerRequest, history: list[dict]) -> tuple[list[dict], list[dict]]:
    incoming = [item.model_dump() for item in request.messages] if request.messages else []
    if not incoming and request.question:
        incoming = [{"role": "user", "content": request.question}]
    if not incoming:
        raise HTTPException(status_code=422, detail="question or messages is required")
    if history and incoming[: len(history)] == history:
        new_messages = incoming[len(history) :]
        messages = incoming
    elif all(message["role"] == "user" for message in incoming):
        new_messages = incoming
        messages = history + incoming
    elif history:
        raise HTTPException(status_code=409, detail="messages must extend the stored conversation history")
    else:
        new_messages = incoming
        messages = incoming
    if not new_messages:
        raise HTTPException(status_code=400, detail="no new messages supplied")
    return messages, new_messages


@app.post("/api/v1/answer", response_model=AnswerResponse)
async def answer(body: AnswerRequest, request: Request) -> AnswerResponse:
    tenant, user = authenticate(request)
    conversation_id = body.conversation_id
    if conversation_id is None:
        conversation_id = (await session_store.create_conversation(tenant, user, None)).id
    elif not await session_store.conversation_exists(conversation_id, tenant, user):
        raise HTTPException(status_code=404, detail="conversation not found")
    history = await session_store.get_history(conversation_id, tenant, user)
    messages, new_messages = _conversation_messages(body, history)
    agent_started = time.perf_counter()
    try:
        audit_event("agent_invocation", operation="answer", outcome="started", downstream="agent-core")
        result = await agent_client.answer(messages, body.model)
    except Exception as exc:
        audit_event("agent_invocation", operation="answer", outcome="failed", downstream="agent-core", error=type(exc).__name__)
        store_downstream_trace(exc, tenant, user)
        logger.warning("gateway_stage_failed request_id=%s stage=agent error=%s", request_id(), type(exc).__name__)
        raise downstream_error(exc) from exc
    audit_event("agent_invocation", operation="answer", outcome="succeeded", downstream="agent-core")
    logger.info("gateway_agent_stage request_id=%s agent_ms=%.1f", request_id(), elapsed_ms(agent_started))
    try:
        citations = [Citation(**item) for item in result.get("citations", [])]
        answer_text = str(result.get("answer", ""))
        iterations = int(result.get("iterations", 0))
        tool_calls = int(result.get("tool_calls", 0))
    except (TypeError, ValueError, KeyError) as exc:
        audit_event("downstream_response", operation="answer", outcome="malformed", downstream="agent-core", error=type(exc).__name__)
        raise HTTPException(status_code=502, detail="malformed agent response") from exc
    for message in new_messages:
        await session_store.append_message(conversation_id, tenant, user, message["role"], message["content"])
    await session_store.append_message(conversation_id, tenant, user, "assistant", answer_text)
    trace = result.get("trace")
    if isinstance(trace, dict):
        trace_store.put(trace, tenant_id=tenant, user_id=user)
    return AnswerResponse(
        conversation_id=conversation_id,
        answer=str(result.get("answer", "")),
        citations=[Citation(**item) for item in result.get("citations", [])],
        iterations=int(result.get("iterations", 0)),
        tool_calls=int(result.get("tool_calls", 0)),
        trace_id=str(trace.get("trace_id")) if isinstance(trace, dict) and trace.get("trace_id") else None,
    )


@app.post("/api/v1/answer/stream")
async def answer_stream(body: AnswerRequest, request: Request) -> StreamingResponse:
    tenant, user = authenticate(request)
    conversation_id = body.conversation_id
    if conversation_id is None:
        conversation_id = (await session_store.create_conversation(tenant, user, None)).id
    elif not await session_store.conversation_exists(conversation_id, tenant, user):
        raise HTTPException(status_code=404, detail="conversation not found")
    history = await session_store.get_history(conversation_id, tenant, user)
    messages, new_messages = _conversation_messages(body, history)

    async def events():
        answer_parts: list[str] = []
        stream_started = time.perf_counter()
        try:
            async for event, data in agent_client.stream_answer(messages, body.model):
                payload = json.loads(data)
                if event == "delta":
                    content = str(payload.get("content", ""))
                    answer_parts.append(content)
                    yield f"event: delta\ndata: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"
                elif event == "error":
                    trace = payload.get("trace")
                    if isinstance(trace, dict):
                        trace_store.put(trace, tenant_id=tenant, user_id=user)
                        error_payload = {
                            "detail": "grounded answer dependency failed",
                            "trace_id": str(trace.get("trace_id", "")),
                        }
                    else:
                        error_payload = {"detail": "grounded answer dependency failed"}
                    yield f"event: error\ndata: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
                elif event == "done":
                    answer_text = "".join(answer_parts)
                    for message in new_messages:
                        await session_store.append_message(conversation_id, tenant, user, message["role"], message["content"])
                    await session_store.append_message(conversation_id, tenant, user, "assistant", answer_text)
                    payload["conversation_id"] = str(conversation_id)
                    trace = payload.get("trace")
                    if isinstance(trace, dict):
                        trace_store.put(trace, tenant_id=tenant, user_id=user)
                        payload = {key: value for key, value in payload.items() if key != "trace"}
                        payload["trace_id"] = str(trace.get("trace_id", ""))
                    audit_event("agent_invocation", operation="answer_stream", outcome="succeeded", downstream="agent-core")
                    logger.info("gateway_stream_stage request_id=%s stage=agent agent_ms=%.1f", request_id(), elapsed_ms(stream_started))
                    yield f"event: done\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except Exception as exc:
            audit_event("agent_invocation", operation="answer_stream", outcome="failed", downstream="agent-core", error=type(exc).__name__)
            logger.exception("grounded_answer_stream_failed request_id=%s stage=agent error=%s", request_id(), type(exc).__name__)
            yield f"event: error\ndata: {json.dumps({'detail': 'grounded answer dependency failed'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})



@app.get("/api/v1/traces/{trace_id}", tags=["diagnostics"])
async def get_trace(trace_id: str, request: Request) -> dict:
    tenant, user = authenticate_diagnostics(request)
    trace = trace_store.get(trace_id, tenant_id=tenant, user_id=user)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return trace

@app.post("/search", operation_id="search_knowledge", tags=["llm"])
async def search(request: SearchRequest, http_request: Request) -> list[dict]:
    tenant, user = authenticate(http_request)
    started = time.perf_counter()
    try:
        result = await service.search(request.query, request.limit, tenant, user)
        logger.info("gateway_stage request_id=%s stage=retrieval retrieval_ms=%.1f", request_id(), elapsed_ms(started))
        return result
    except Exception as exc:
        raise downstream_error(exc) from exc

@app.get("/sources", operation_id="list_sources", tags=["llm"])
async def sources(request: Request) -> list[dict]:
    tenant, user = authenticate(request)
    try:
        return await service.get_sources(tenant, user)
    except Exception as exc:
        raise downstream_error(exc) from exc

@app.get("/sources/{source_name}", tags=["internal"])
async def source(request: Request, source_name: str) -> dict:
    tenant, user = authenticate(request)
    try:
        return await service.get_source(source_name, tenant, user)
    except Exception as exc:
        raise downstream_error(exc) from exc

@app.get("/stats", tags=["internal"])
async def stats(request: Request) -> dict:
    tenant, user = authenticate(request)
    try:
        return await service.get_stats(tenant, user)
    except Exception as exc:
        raise downstream_error(exc) from exc

@app.get("/messages/{message_id}", operation_id="get_message", tags=["llm"])
async def message(request: Request, message_id: str) -> dict:
    tenant, user = authenticate(request)
    try:
        return await service.get_message(message_id, tenant, user)
    except Exception as exc:
        raise downstream_error(exc) from exc

@app.get("/documents/{document_id}", operation_id="get_document", tags=["llm"])
async def document(request: Request, document_id: str) -> dict:
    tenant, user = authenticate(request)
    try:
        return await service.get_document(document_id, tenant, user)
    except Exception as exc:
        raise downstream_error(exc) from exc

@app.post("/ingest", tags=["internal"])
async def ingest(request: Request) -> Response:
    tenant, user = authenticate(request)
    response = await service.ingest(await request.body(), request.headers.get("content-type", "application/json"), tenant, user)
    return Response(content=response.content, status_code=response.status_code, media_type=response.headers.get("content-type"))

@app.post("/upload", tags=["internal"])
async def upload(request: Request) -> Response:
    tenant, user = authenticate(request)
    response = await service.upload(await request.body(), request.headers.get("content-type", "application/octet-stream"), tenant, user)
    return Response(content=response.content, status_code=response.status_code, media_type=response.headers.get("content-type"))
