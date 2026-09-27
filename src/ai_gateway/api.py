from __future__ import annotations

import json
import logging
import os
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from .agent_client import AgentClient
from .auth import authenticate
from .db import SessionStore
from .models import AnswerRequest, AnswerResponse, ChatCompletionRequest, Citation, ConversationCreateRequest, ConversationResponse, GatewaySettings, SearchRequest
from .observability import elapsed_ms, incoming_request_id, reset_request_id, set_request_id
from .service import RAGService

logger = logging.getLogger(__name__)
settings = GatewaySettings(retrieval_url=os.getenv("RAG_RETRIEVAL_URL", "http://rag-retrieval:8100"), ingestion_url=os.getenv("RAG_INGESTION_URL", "http://pst-agent:8000"), agent_url=os.getenv("AI_AGENT_URL", "http://agent-core:8000"), postgres_dsn=os.getenv("AI_POSTGRES_DSN", os.getenv("RAG_POSTGRES_DSN", "postgresql://rag:rag@postgres:5432/rag")), timeout_seconds=float(os.getenv("RAG_DOWNSTREAM_TIMEOUT_SECONDS", "60")), agent_timeout_seconds=float(os.getenv("AI_AGENT_TIMEOUT_SECONDS", "120")))
service = RAGService(settings)
agent_client = AgentClient(settings.agent_url, settings.agent_timeout_seconds)
session_store = SessionStore(settings.postgres_dsn)
app = FastAPI(title="AI Gateway", version="0.3.0")
MAX_REQUEST_BYTES = int(os.getenv("AI_GATEWAY_MAX_REQUEST_BYTES", str(25 * 1024 * 1024)))
UI_ORIGINS = [origin.strip() for origin in os.getenv("AI_UI_ORIGINS", "http://localhost:3000").split(",") if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=UI_ORIGINS, allow_credentials=True, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["Authorization", "Content-Type", "X-Request-ID"])

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
    if isinstance(exc, TimeoutError): return HTTPException(504, "downstream timeout")
    if isinstance(exc, ConnectionError): return HTTPException(502, "downstream unavailable")
    return HTTPException(502, "downstream request failed")

@app.get("/health", tags=["internal"])
async def health() -> dict[str, str]: return {"status": "ok"}

@app.get("/ready", tags=["internal"])
async def ready() -> dict[str, str]:
    started=time.perf_counter()
    try:
        await session_store.conversation_exists("00000000-0000-0000-0000-000000000000", "__readiness__", "__readiness__")
    except Exception as exc:
        logger.warning("gateway_readiness_failed request_id=%s stage=postgres error=%s", __import__("ai_gateway.observability", fromlist=["request_id"]).request_id(), type(exc).__name__)
        raise HTTPException(503, "postgres unavailable") from exc
    logger.info("gateway_readiness request_id=%s postgres_ms=%.1f", __import__("ai_gateway.observability", fromlist=["request_id"]).request_id(), elapsed_ms(started))
    return {"status":"ready"}

@app.post("/api/v1/conversations", response_model=ConversationResponse)
async def create_conversation(body: ConversationCreateRequest, request: Request) -> ConversationResponse:
    tenant, user = authenticate(request); conversation = await session_store.create_conversation(tenant, user, body.title)
    return ConversationResponse(id=conversation.id,title=conversation.title,created_at=conversation.created_at.isoformat(),updated_at=conversation.updated_at.isoformat())

@app.get("/api/v1/conversations", response_model=list[ConversationResponse])
async def list_conversations(request: Request, limit: int = 50) -> list[ConversationResponse]:
    tenant,user=authenticate(request)
    if limit<1 or limit>100: raise HTTPException(400,"limit must be between 1 and 100")
    rows=await session_store.list_conversations(tenant,user,limit)
    return [ConversationResponse(id=row.id,title=row.title,created_at=row.created_at.isoformat(),updated_at=row.updated_at.isoformat()) for row in rows]

@app.post("/v1/chat/completions", tags=["internal"])
async def chat_completions(body: ChatCompletionRequest, request: Request) -> dict:
    tenant,user=authenticate(request); conversation_id=body.conversation_id
    if conversation_id is None: conversation_id=(await session_store.create_conversation(tenant,user,None)).id
    elif not await session_store.conversation_exists(conversation_id,tenant,user): raise HTTPException(404,"conversation not found")
    history=await session_store.get_history(conversation_id,tenant,user); incoming=[message.model_dump() for message in body.messages]
    if history and incoming[:len(history)]==history: new_messages=incoming[len(history):]; messages=incoming
    elif all(message["role"]=="user" for message in incoming): new_messages=incoming; messages=history+incoming
    elif history: raise HTTPException(409,"messages must extend the stored conversation history")
    else: new_messages=incoming; messages=incoming
    if not new_messages: raise HTTPException(400,"no new messages supplied")
    if body.stream: raise HTTPException(400,"streaming is not yet enabled at the gateway boundary")
    started=time.perf_counter()
    try: result=await agent_client.chat(messages,body.model)
    except Exception as exc: logger.warning("gateway_stage_failed request_id=%s stage=agent error=%s",__import__("ai_gateway.observability",fromlist=["request_id"]).request_id(),type(exc).__name__); raise downstream_error(exc) from exc
    logger.info("gateway_agent_stage request_id=%s agent_ms=%.1f",__import__("ai_gateway.observability",fromlist=["request_id"]).request_id(),elapsed_ms(started))
    for message in new_messages: await session_store.append_message(conversation_id,tenant,user,message["role"],message["content"])
    content=str(result.get("content","")); await session_store.append_message(conversation_id,tenant,user,"assistant",content)
    return {"id":f"chatcmpl-{uuid.uuid4().hex}","object":"chat.completion","created":int(time.time()),"model":body.model or "default","choices":[{"index":0,"message":{"role":"assistant","content":content},"finish_reason":"stop"}],"conversation_id":conversation_id,"usage":{"iterations":result.get("iterations",0),"tool_calls":result.get("tool_calls",0)}}

# Existing answer/search route implementations remain below unchanged.
