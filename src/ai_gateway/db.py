from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

SCHEMA = """
CREATE TABLE IF NOT EXISTS ai_conversations (
    id UUID PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    title TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ai_conversations_owner_idx
    ON ai_conversations (tenant_id, user_id, updated_at DESC);
CREATE TABLE IF NOT EXISTS ai_messages (
    id UUID PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES ai_conversations(id) ON DELETE CASCADE,
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ai_messages_conversation_idx
    ON ai_messages (conversation_id, created_at, id);
"""

@dataclass(frozen=True)
class Conversation:
    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime

class SessionStore:
    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 5) -> None:
        self.pool = AsyncConnectionPool(dsn, min_size=min_size, max_size=max_size, open=False, kwargs={"row_factory": dict_row})
        self._opened = False

    async def open(self) -> None:
        if self._opened:
            return
        await self.pool.open()
        self._opened = True
        async with self.pool.connection() as connection:
            await connection.execute(SCHEMA)
            await connection.commit()

    async def close(self) -> None:
        if self._opened:
            await self.pool.close()
            self._opened = False

    async def _ensure_open(self) -> None:
        await self.open()

    async def create_conversation(self, tenant_id: str, user_id: str, title: str | None) -> Conversation:
        await self._ensure_open()
        async with self.pool.connection() as connection:
            result = await connection.execute(
                """INSERT INTO ai_conversations (id, tenant_id, user_id, title)
                   VALUES (%s, %s, %s, %s)
                   RETURNING id, title, created_at, updated_at""",
                (uuid.uuid4(), tenant_id, user_id, title),
            )
            row = await result.fetchone()
            await connection.commit()
        assert row is not None
        return Conversation(**row)

    async def list_conversations(self, tenant_id: str, user_id: str, limit: int) -> list[Conversation]:
        await self._ensure_open()
        async with self.pool.connection() as connection:
            result = await connection.execute(
                """SELECT id, title, created_at, updated_at FROM ai_conversations
                   WHERE tenant_id = %s AND user_id = %s
                   ORDER BY updated_at DESC LIMIT %s""",
                (tenant_id, user_id, limit),
            )
            rows = await result.fetchall()
        return [Conversation(**row) for row in rows]

    async def conversation_exists(self, conversation_id: str, tenant_id: str, user_id: str) -> bool:
        await self._ensure_open()
        async with self.pool.connection() as connection:
            result = await connection.execute(
                "SELECT 1 FROM ai_conversations WHERE id = %s AND tenant_id = %s AND user_id = %s",
                (conversation_id, tenant_id, user_id),
            )
            return await result.fetchone() is not None

    async def get_history(self, conversation_id: str, tenant_id: str, user_id: str) -> list[dict[str, Any]]:
        await self._ensure_open()
        async with self.pool.connection() as connection:
            result = await connection.execute(
                """SELECT m.role, m.content FROM ai_messages AS m
                   JOIN ai_conversations AS c ON c.id = m.conversation_id
                   WHERE m.conversation_id = %s AND c.tenant_id = %s AND c.user_id = %s
                   ORDER BY m.created_at, m.id""",
                (conversation_id, tenant_id, user_id),
            )
            return [dict(row) for row in await result.fetchall()]

    async def append_message(
        self, conversation_id: str, tenant_id: str, user_id: str, role: str, content: str
    ) -> None:
        await self._ensure_open()
        async with self.pool.connection() as connection:
            result = await connection.execute(
                """INSERT INTO ai_messages (id, conversation_id, tenant_id, user_id, role, content)
                   SELECT %s, id, tenant_id, user_id, %s, %s FROM ai_conversations
                   WHERE id = %s AND tenant_id = %s AND user_id = %s""",
                (uuid.uuid4(), role, content, conversation_id, tenant_id, user_id),
            )
            if result.rowcount != 1:
                await connection.rollback()
                raise ValueError("conversation not found")
            await connection.execute(
                "UPDATE ai_conversations SET updated_at = now() WHERE id = %s AND tenant_id = %s AND user_id = %s",
                (conversation_id, tenant_id, user_id),
            )
            await connection.commit()
