from fastapi import HTTPException
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
import httpx as httpx2
import jwt
import pytest
from starlette.requests import Request

from ai_gateway import api
from ai_gateway.api import app
from ai_gateway.auth import authenticate
from ai_gateway.mcp import mcp, mcp_app


def request_with_token(token: str):
    return Request(
        {
            "type": "http",
            "headers": [(b"authorization", f"Bearer {token}".encode())],
        }
    )


def make_token(secret: str, claims: dict) -> str:
    return jwt.encode(claims, secret, algorithm="HS256")


@pytest.mark.anyio
async def test_mcp_exposes_only_explicitly_tagged_llm_operations():
    async with Client(mcp) as client:
        tools = await client.list_tools()

    exposed = {tool.name for tool in tools}
    expected = {
        route.operation_id
        for route in app.routes
        if getattr(route, "operation_id", None)
        and "llm" in getattr(route, "tags", [])
    }
    assert exposed == expected


@pytest.mark.anyio
async def test_internal_and_state_changing_routes_are_not_mcp_tools():
    async with Client(mcp) as client:
        tools = await client.list_tools()

    exposed = {tool.name for tool in tools}
    internal_operations = {
        route.operation_id
        for route in app.routes
        if getattr(route, "operation_id", None)
        and "internal" in getattr(route, "tags", [])
    }
    assert not exposed.intersection(internal_operations)


def test_llm_routes_are_explicit_and_reviewable():
    llm_operations = {
        route.operation_id
        for route in app.routes
        if getattr(route, "operation_id", None)
        and "llm" in getattr(route, "tags", [])
    }
    assert llm_operations == {
        "search_knowledge",
        "list_sources",
        "get_message",
        "get_document",
    }


@pytest.mark.anyio
async def test_generated_mcp_http_tool_accepts_bearer_auth_and_uses_single_fastapi_auth_path(
    monkeypatch,
):
    secret = "test-secret-key-with-at-least-32-bytes!!"
    token = make_token(
        secret, {"sub": "token-user", "tenant_id": "token-tenant"}
    )
    monkeypatch.setenv("RAG_JWT_SECRET", secret)
    monkeypatch.delenv("RAG_JWKS_URL", raising=False)

    observed = {}

    async def fake_search(query, limit, tenant, user):
        observed["search"] = (query, limit, tenant, user)
        return [{"text": "ok"}]

    monkeypatch.setattr(api.service, "search", fake_search)

    mcp_app = mcp.http_app(transport="streamable-http", stateless_http=True)

    def httpx_client_factory(**kwargs):
        return httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=mcp_app),
            base_url="http://testserver",
            headers={**kwargs.pop("headers", {}), "Authorization": f"Bearer {token}"},
            **kwargs,
        )

    transport = StreamableHttpTransport(
        "http://testserver/mcp",
        httpx_client_factory=httpx_client_factory,
    )

    async with mcp_app.lifespan(mcp_app):
        async with Client(transport) as client:
            await client.call_tool(
                "search_knowledge", {"query": "hello", "limit": 1}
            )

    assert observed == {
        "search": ("hello", 1, "token-tenant", "token-user"),
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    "authorization",
    [None, "", "Basic not-bearer", "Bearer "],
)
async def test_mcp_missing_or_wrong_auth_is_rejected_before_downstream(
    monkeypatch, authorization
):
    secret = "test-secret-key-with-at-least-32-bytes!!"
    monkeypatch.setenv("RAG_JWT_SECRET", secret)
    monkeypatch.delenv("RAG_JWKS_URL", raising=False)

    called = False

    async def fake_search(query, limit, tenant, user):
        nonlocal called
        called = True
        return [{"text": "should-not-run"}]

    monkeypatch.setattr(api.service, "search", fake_search)

    mcp_app = mcp.http_app(transport="streamable-http", stateless_http=True)

    def httpx_client_factory(**kwargs):
        return httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=mcp_app),
            base_url="http://testserver",
            **kwargs,
        )

    transport = StreamableHttpTransport(
        "http://testserver/mcp",
        httpx_client_factory=httpx_client_factory,
    )

    async with mcp_app.lifespan(mcp_app):
        async with Client(transport) as client:
            with pytest.raises(Exception, match="401|Unauthorized|Bearer"):
                await client.call_tool(
                    "search_knowledge", {"query": "hello", "limit": 1}
                )

    assert called is False


@pytest.mark.anyio
@pytest.mark.parametrize(
    "claims, signing_secret",
    [
        ({"sub": "u", "tenant_id": "t"}, "wrong-signing-secret-with-at-least-32!!"),
        ({"tenant_id": "t"}, "test-secret-key-with-at-least-32-bytes!!"),
        ({"sub": "u"}, "test-secret-key-with-at-least-32-bytes!!"),
    ],
)
async def test_mcp_invalid_jwt_or_missing_identity_claims_never_reach_downstream(
    monkeypatch, claims, signing_secret
):
    secret = "test-secret-key-with-at-least-32-bytes!!"
    monkeypatch.setenv("RAG_JWT_SECRET", secret)

    token = make_token(signing_secret, claims)
    mcp_app = mcp.http_app(transport="streamable-http", stateless_http=True)

    async def fake_search(query, limit, tenant, user):
        raise AssertionError("downstream search must not be called")

    monkeypatch.setattr(api.service, "search", fake_search)

    def httpx_client_factory(**kwargs):
        return httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=mcp_app),
            base_url="http://testserver",
            **kwargs,
        )

    transport = StreamableHttpTransport(
        "http://testserver/mcp",
        httpx_client_factory=httpx_client_factory,
    )

    async with mcp_app.lifespan(mcp_app):
        async with Client(transport) as client:
            with pytest.raises(Exception):
                await client.call_tool(
                    "search_knowledge", {"query": "hello", "limit": 1}
                )


def test_single_jwt_verifier_is_shared_by_rest_and_generated_mcp_tools():
    import ai_gateway.api as gateway_api
    import ai_gateway.auth as gateway_auth

    assert gateway_api.authenticate is gateway_auth.authenticate


def test_mcp_entrypoint_is_http_and_generated_tools_use_streamable_http():
    source = __import__("inspect").getsource(
        __import__("ai_gateway.mcp", fromlist=["mcp"])
    )
    assert 'mcp.run(transport="http"' in source
    assert mcp.http_app(transport="streamable-http", stateless_http=True) is not None


def test_mounted_mcp_path_exists():
    routes = {route.path for route in app.routes}
    assert any(path.startswith("/mcp") for path in routes)
    assert app.router.lifespan_context is mcp_app.lifespan


def test_auth_unit_error_contract(monkeypatch):
    monkeypatch.setenv("RAG_JWT_SECRET", "test-secret-key-with-at-least-32-bytes!!")

    with pytest.raises(HTTPException) as missing:
        authenticate(Request({"type": "http", "headers": []}))
    assert missing.value.status_code == 401

    invalid = make_token(
        "wrong-secret-with-at-least-32-bytes!!!!",
        {"sub": "u", "tenant_id": "t"},
    )
    with pytest.raises(HTTPException) as bad:
        authenticate(request_with_token(invalid))
    assert bad.value.status_code == 401
