from __future__ import annotations

import os
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .agent_client import AgentClient
from .auth import authenticate
from .db import SessionStore
from .models import ChatCompletionRequest, ConversationCreateRequest, ConversationResponse, GatewaySettings, SearchRequest
from .service import RAGService

settings = GatewaySettings(
    retrieval_url=os.getenv("RAG_RETRIEVAL_URL", "http://rag-retrieval:8100"),
    ingestion_url=os.getenv("RAG_INGESTION_URL", "http://pst-agent:8000"),
    agent_url=os.getenv("AI_AGENT_URL", "http://agent-core:8000"),
    postgres_dsn=os.getenv("AI_POSTGRES_DSN", os.getenv("RAG_POSTGRES_DSN", "postgresql://rag:rag@postgres:5432/rag")),
    timeout_seconds=float(os.getenv("RAG_DOWNSTREAM_TIMEOUT_SECONDS", "60")),
    agent_timeout_seconds=float(os.getenv("AI_AGENT_TIMEOUT_SECONDS", "120")),
)
service = RAGService(settings)
agent_client = AgentClient(settings.agent_url, settings.agent_timeout_seconds)
session_store = SessionStore(settings.postgres_dsn)
app = FastAPI(title="AI Gateway", version="0.3.0")

UI_ORIGINS = [origin.strip() for origin in os.getenv("AI_UI_ORIGINS", "http://localhost:3000").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware, allow_origins=UI_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

@app.on_event("shutdown")
async def shutdown() -> None:
    await session_store.close()

def downstream_error(exc: Exception) -> HTTPException:
    if isinstance(exc, TimeoutError):
        return HTTPException(504, "downstream timeout")
    if isinstance(exc, ConnectionError):
        return HTTPException(502, "downstream unavailable")
    return HTTPException(502, "downstream request failed")

@app.get("/health", tags=["internal"])
async def health() -> dict[str, str]:
    return {"status": "ok"}

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

@app.post("/search", operation_id="search_knowledge", tags=["llm"])
async def search(request: SearchRequest, http_request: Request) -> list[dict]:
    tenant, user = authenticate(http_request)
    try:
        return await service.search(request.query, request.limit, tenant, user)
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
