import json

from fastapi.testclient import TestClient
import httpx

from ai_gateway import api


def test_health():
    client = TestClient(api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_authenticates_and_injects_derived_context(monkeypatch):
    calls = []

    def fake_authenticate(request):
        assert request.headers["x-rag-tenant-id"] == "attacker-tenant"
        return "token-tenant", "token-user"

    async def fake_search(query, limit, tenant, user):
        calls.append((query, limit, tenant, user))
        return [{"text": "ok"}]

    monkeypatch.setattr(api, "authenticate", fake_authenticate)
    monkeypatch.setattr(api.service, "search", fake_search)

    client = TestClient(api.app)
    response = client.post(
        "/search",
        json={"query": "hello"},
        headers={"X-RAG-Tenant-ID": "attacker-tenant"},
    )

    assert response.status_code == 200
    assert response.json() == [{"text": "ok"}]
    assert calls == [("hello", 10, "token-tenant", "token-user")]


def test_upload_uses_ingestion_backend(monkeypatch):
    async def fake_upload(body, content_type, tenant, user):
        return httpx.Response(201, content=b"ok")

    monkeypatch.setattr(api, "authenticate", lambda request: ("t1", "u1"))
    monkeypatch.setattr(api.service, "upload", fake_upload)

    client = TestClient(api.app)
    response = client.post("/upload", content=b"data")
    assert response.status_code == 201
    assert response.text == "ok"


def test_user_read_routes_require_auth(monkeypatch):
    from fastapi import HTTPException

    def reject(request):
        raise HTTPException(status_code=401, detail="missing token")

    monkeypatch.setattr(api, "authenticate", reject)
    client = TestClient(api.app)
    assert client.get("/sources").status_code == 401


def test_mcp_only_exposes_llm_routes():
    from ai_gateway.mcp import mcp

    assert mcp is not None


def test_chat_completions_persists_history_and_calls_agent(monkeypatch):
    import ai_gateway.api as gateway_api

    monkeypatch.setattr(gateway_api, "authenticate", lambda request: ("tenant-1", "user-1"))

    class FakeStore:
        def __init__(self):
            self.messages = []

        async def create_conversation(self, tenant, user, title):
            from types import SimpleNamespace
            from datetime import datetime, timezone
            return SimpleNamespace(
                id="c1",
                title=title,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

        async def conversation_exists(self, conversation_id, tenant, user):
            return conversation_id == "c1"

        async def get_history(self, conversation_id, tenant, user):
            return list(self.messages)

        async def append_message(self, conversation_id, tenant, user, role, content):
            self.messages.append({"role": role, "content": content})

    store = FakeStore()

    async def fake_chat(messages, model):
        assert messages == [{"role": "user", "content": "hello"}]
        return {"content": "world", "iterations": 1, "tool_calls": 0}

    monkeypatch.setattr(gateway_api, "session_store", store)
    monkeypatch.setattr(gateway_api.agent_client, "chat", fake_chat)

    response = TestClient(gateway_api.app).post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "world"
    assert store.messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]

    async def second_chat(messages, model):
        assert messages == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
            {"role": "user", "content": "again"},
        ]
        return {"content": "second", "iterations": 1, "tool_calls": 0}

    monkeypatch.setattr(gateway_api.agent_client, "chat", second_chat)
    response = TestClient(gateway_api.app).post(
        "/v1/chat/completions",
        json={
            "conversation_id": "c1",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
                {"role": "user", "content": "again"},
            ],
        },
    )
    assert response.status_code == 200
    assert store.messages[-2:] == [
        {"role": "user", "content": "again"},
        {"role": "assistant", "content": "second"},
    ]


def test_grounded_answer_persists_citations_and_history(monkeypatch):
    monkeypatch.setattr(api, "authenticate", lambda request: ("tenant-1", "user-1"))

    class FakeStore:
        def __init__(self):
            self.messages = []

        async def create_conversation(self, tenant, user, title):
            from types import SimpleNamespace
            return SimpleNamespace(id="c1")

        async def conversation_exists(self, conversation_id, tenant, user):
            return conversation_id == "c1"

        async def get_history(self, conversation_id, tenant, user):
            return list(self.messages)

        async def append_message(self, conversation_id, tenant, user, role, content):
            self.messages.append({"role": role, "content": content})

    store = FakeStore()

    async def fake_answer(messages, model):
        assert messages == [{"role": "user", "content": "What?"}]
        return {
            "answer": "Supported [S1].",
            "citations": [
                {"id": "S1", "chunk_id": "c1", "source_name": "mailbox", "text": "Evidence"}
            ],
            "iterations": 2,
            "tool_calls": 1,
        }

    monkeypatch.setattr(api, "session_store", store)
    monkeypatch.setattr(api.agent_client, "answer", fake_answer)

    response = TestClient(api.app).post("/api/v1/answer", json={"question": "What?"})
    assert response.status_code == 200
    assert response.json()["conversation_id"] == "c1"
    assert response.json()["answer"] == "Supported [S1]."
    assert response.json()["citations"][0]["source_name"] == "mailbox"
    assert store.messages == [
        {"role": "user", "content": "What?"},
        {"role": "assistant", "content": "Supported [S1]."},
    ]


def test_grounded_answer_continues_stored_conversation(monkeypatch):
    monkeypatch.setattr(api, "authenticate", lambda request: ("tenant-1", "user-1"))

    class FakeStore:
        def __init__(self):
            self.messages = [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "first answer"},
            ]

        async def create_conversation(self, tenant, user, title):
            from types import SimpleNamespace
            return SimpleNamespace(id="c1")

        async def conversation_exists(self, conversation_id, tenant, user):
            return conversation_id == "c1"

        async def get_history(self, conversation_id, tenant, user):
            return list(self.messages)

        async def append_message(self, conversation_id, tenant, user, role, content):
            self.messages.append({"role": role, "content": content})

    store = FakeStore()

    async def fake_answer(messages, model):
        assert messages == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second"},
        ]
        assert model == "local-model"
        return {
            "answer": "second answer",
            "citations": [],
            "iterations": 1,
            "tool_calls": 0,
        }

    monkeypatch.setattr(api, "session_store", store)
    monkeypatch.setattr(api.agent_client, "answer", fake_answer)

    response = TestClient(api.app).post(
        "/api/v1/answer",
        json={
            "conversation_id": "c1",
            "model": "local-model",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "second"},
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["conversation_id"] == "c1"
    assert store.messages[-2:] == [
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "second answer"},
    ]


def test_grounded_answer_surfaces_downstream_failure(monkeypatch):
    monkeypatch.setattr(api, "authenticate", lambda request: ("tenant-1", "user-1"))

    class FakeStore:
        async def create_conversation(self, tenant, user, title):
            from types import SimpleNamespace
            return SimpleNamespace(id="c1")

        async def conversation_exists(self, conversation_id, tenant, user):
            return conversation_id == "c1"

        async def get_history(self, conversation_id, tenant, user):
            return []

    store = FakeStore()

    async def failing_answer(messages, model):
        raise TimeoutError("agent-core timeout")

    monkeypatch.setattr(api, "session_store", store)
    monkeypatch.setattr(api.agent_client, "answer", failing_answer)

    response = TestClient(api.app).post(
        "/api/v1/answer",
        json={"question": "What?"},
    )

    assert response.status_code == 504
    assert response.json() == {"detail": "downstream timeout"}


def test_grounded_answer_maps_agent_mcp_failure_to_502(monkeypatch):
    monkeypatch.setattr(api, "authenticate", lambda request: ("tenant-1", "user-1"))

    class FakeStore:
        async def create_conversation(self, tenant, user, title):
            from types import SimpleNamespace
            return SimpleNamespace(id="c1")

        async def conversation_exists(self, conversation_id, tenant, user):
            return conversation_id == "c1"

        async def get_history(self, conversation_id, tenant, user):
            return []

    request = httpx.Request("POST", "http://agent-core/api/v1/answer")
    response = httpx.Response(502, request=request)

    async def failing_answer(messages, model):
        raise httpx.HTTPStatusError(
            "agent-core returned MCP failure",
            request=request,
            response=response,
        )

    monkeypatch.setattr(api, "session_store", FakeStore())
    monkeypatch.setattr(api.agent_client, "answer", failing_answer)

    result = TestClient(api.app).post(
        "/api/v1/answer",
        json={"question": "What?"},
    )

    assert result.status_code == 502
    assert result.json() == {"detail": "downstream request failed"}


def test_grounded_answer_stream_serializes_uuid_conversation_id(monkeypatch):
    monkeypatch.setattr(api, "authenticate", lambda request: ("tenant-1", "user-1"))

    class FakeStore:
        async def create_conversation(self, tenant, user, title):
            from types import SimpleNamespace
            return SimpleNamespace(id=__import__("uuid").uuid4())

        async def conversation_exists(self, conversation_id, tenant, user):
            return False

        async def get_history(self, conversation_id, tenant, user):
            return []

        async def append_message(self, conversation_id, tenant, user, role, content):
            pass

    async def fake_stream(messages, model):
        yield "delta", '{"content":"Hello."}'
        yield "done", '{"citations":[],"iterations":1,"tool_calls":0}'

    monkeypatch.setattr(api, "session_store", FakeStore())
    monkeypatch.setattr(api.agent_client, "stream_answer", fake_stream)

    response = TestClient(api.app).post("/api/v1/answer/stream", json={"question": "What?"})

    assert response.status_code == 200
    assert '"conversation_id": "' in response.text
    assert "UUID is not JSON serializable" not in response.text


def test_grounded_answer_stream_persists_completed_answer(monkeypatch):
    monkeypatch.setattr(api, "authenticate", lambda request: ("tenant-1", "user-1"))

    class FakeStore:
        def __init__(self):
            self.messages = []

        async def create_conversation(self, tenant, user, title):
            from types import SimpleNamespace
            return SimpleNamespace(id="c1")

        async def conversation_exists(self, conversation_id, tenant, user):
            return conversation_id == "c1"

        async def get_history(self, conversation_id, tenant, user):
            return list(self.messages)

        async def append_message(self, conversation_id, tenant, user, role, content):
            self.messages.append({"role": role, "content": content})

    store = FakeStore()

    async def fake_stream(messages, model):
        yield "delta", '{"content":"Hello "}'
        yield "delta", '{"content":"world."}'
        yield "done", '{"citations":[{"id":"S1","chunk_id":"c1","source_name":"mailbox","text":"Evidence"}],"iterations":2,"tool_calls":1}'

    monkeypatch.setattr(api, "session_store", store)
    monkeypatch.setattr(api.agent_client, "stream_answer", fake_stream)

    response = TestClient(api.app).post("/api/v1/answer/stream", json={"question": "What?"})
    assert response.status_code == 200
    assert "event: delta\ndata:" in response.text
    assert "Hello " in response.text
    assert "event: done\ndata:" in response.text
    assert '"conversation_id": "c1"' in response.text
    assert '"citations": [{"id": "S1", "chunk_id": "c1", "source_name": "mailbox", "text": "Evidence"}]' in response.text
    assert store.messages == [
        {"role": "user", "content": "What?"},
        {"role": "assistant", "content": "Hello world."},
    ]


def test_request_id_is_propagated_and_returned():
    response = TestClient(api.app).get("/health", headers={"X-Request-ID": "phase1c-test-id"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "phase1c-test-id"


def test_oversized_request_is_rejected(monkeypatch):
    monkeypatch.setattr(api, "MAX_REQUEST_BYTES", 10)
    response = TestClient(api.app).post("/search", content=b"12345678901")
    assert response.status_code == 413


def test_trace_endpoint_requires_diagnostics_permission(monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(
        api,
        "authenticate_diagnostics",
        lambda request: (_ for _ in ()).throw(
            HTTPException(403, "diagnostics:read permission required")
        ),
    )
    response = TestClient(api.app).get("/api/v1/traces/trace-1")
    assert response.status_code == 403


def test_trace_endpoint_is_tenant_and_user_scoped(monkeypatch):
    monkeypatch.setattr(api, "authenticate_diagnostics", lambda request: ("tenant-1", "user-1"))
    api.trace_store.put(
        {"trace_id": "trace-1", "status": "completed"},
        tenant_id="tenant-1",
        user_id="user-1",
    )
    client = TestClient(api.app)
    response = client.get("/api/v1/traces/trace-1")
    assert response.status_code == 200
    assert response.json()["trace_id"] == "trace-1"

    monkeypatch.setattr(api, "authenticate_diagnostics", lambda request: ("tenant-2", "user-1"))
    assert client.get("/api/v1/traces/trace-1").status_code == 404


def test_grounded_answer_stream_persists_failed_trace_and_exposes_only_trace_id(monkeypatch):
    monkeypatch.setattr(api, "authenticate", lambda request: ("tenant-1", "user-1"))

    class FakeStore:
        async def create_conversation(self, tenant, user, title):
            from types import SimpleNamespace
            return SimpleNamespace(id="c1")

        async def conversation_exists(self, conversation_id, tenant, user):
            return conversation_id == "c1"

        async def get_history(self, conversation_id, tenant, user):
            return []

    trace = {
        "schema_version": "1.0",
        "trace_id": "failed-trace-1",
        "status": "failed",
        "error": {"type": "MCPToolArgumentError", "message": "invalid arguments"},
        "trace": [
            {"id": "tool-1", "kind": "tool", "stage": "tool", "name": "web.echo", "status": "failed"}
        ],
        "logs": [],
        "metrics": {},
    }

    async def fake_stream(messages, model):
        yield "delta", '{"content":"partial"}'
        yield "error", json.dumps({
            "detail": "grounded answer dependency failed",
            "trace_id": trace["trace_id"],
            "trace": trace,
        })

    monkeypatch.setattr(api, "session_store", FakeStore())
    monkeypatch.setattr(api.agent_client, "stream_answer", fake_stream)
    monkeypatch.setattr(api, "authenticate_diagnostics", lambda request: ("tenant-1", "user-1"))

    response = TestClient(api.app).post("/api/v1/answer/stream", json={"question": "What?"})
    assert response.status_code == 200

    frames = [frame for frame in response.text.split("\n\n") if frame]
    assert len(frames) == 2
    assert frames[0] == 'event: delta\ndata: {"content": "partial"}'
    assert frames[1].startswith("event: error\ndata: ")
    error_payload = json.loads(frames[1].split("data: ", 1)[1])
    assert error_payload == {
        "detail": "grounded answer dependency failed",
        "trace_id": "failed-trace-1",
    }

    diagnostic = TestClient(api.app).get("/api/v1/traces/failed-trace-1")
    assert diagnostic.status_code == 200
    assert diagnostic.json()["status"] == "failed"
    assert diagnostic.json()["error"]["type"] == "MCPToolArgumentError"
