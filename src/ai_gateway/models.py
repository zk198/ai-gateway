from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    limit: int = Field(default=10, ge=1, le=20)

class SearchResult(BaseModel):
    chunk_id: str | None = None
    parent_kind: str | None = None
    score: float | None = None
    text: str | None = None
    source_name: str | None = None
    parent: dict[str, Any] | None = None

class Source(BaseModel):
    name: str
    account_email: str | None = None
    account_type: str | None = None
    display_name: str | None = None
    description: str | None = None

class GatewaySettings(BaseModel):
    retrieval_url: str
    ingestion_url: str
    agent_url: str
    postgres_dsn: str
    timeout_seconds: float = 60.0
    agent_timeout_seconds: float = 120.0

class ConversationCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=500)

class ConversationResponse(BaseModel):
    id: str
    title: str | None
    created_at: str
    updated_at: str

class ChatMessage(BaseModel):
    role: str = Field(pattern="^(system|user|assistant|tool)$")
    content: str = Field(min_length=1, max_length=100_000)

class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)
    conversation_id: str | None = None
    stream: bool = False
