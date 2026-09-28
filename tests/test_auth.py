import jwt
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from ai_gateway.auth import authenticate


def request_with_token(token):
    return Request(
        {
            "type": "http",
            "headers": [(b"authorization", f"Bearer {token}".encode())],
        }
    )


def test_missing_bearer_is_rejected():
    request = Request({"type": "http", "headers": []})

    with pytest.raises(HTTPException) as exc:
        authenticate(request)

    assert exc.value.status_code == 401


def test_valid_hs256_token_returns_tenant_and_user(monkeypatch):
    monkeypatch.setenv("RAG_JWT_SECRET", "secret")
    monkeypatch.delenv("RAG_JWKS_URL", raising=False)

    token = jwt.encode(
        {"sub": "u1", "tenant_id": "t1"},
        "secret",
        algorithm="HS256",
    )

    assert authenticate(request_with_token(token)) == ("t1", "u1")


def test_invalid_signature_is_rejected(monkeypatch):
    monkeypatch.setenv("RAG_JWT_SECRET", "secret")

    token = jwt.encode(
        {"sub": "u1", "tenant_id": "t1"},
        "wrong",
        algorithm="HS256",
    )

    with pytest.raises(HTTPException) as exc:
        authenticate(request_with_token(token))

    assert exc.value.status_code == 401


def test_issuer_and_audience_are_verified_when_configured(monkeypatch):
    monkeypatch.setenv("RAG_JWT_SECRET", "secret")
    monkeypatch.setenv("RAG_JWT_ISSUER", "issuer")
    monkeypatch.setenv("RAG_JWT_AUDIENCE", "audience")

    token = jwt.encode(
        {
            "sub": "u1",
            "tenant_id": "t1",
            "iss": "wrong",
            "aud": "audience",
        },
        "secret",
        algorithm="HS256",
    )

    with pytest.raises(HTTPException) as exc:
        authenticate(request_with_token(token))

    assert exc.value.status_code == 401


def test_missing_subject_or_tenant_is_forbidden(monkeypatch):
    monkeypatch.setenv("RAG_JWT_SECRET", "secret")

    token = jwt.encode({"sub": "u1"}, "secret", algorithm="HS256")

    with pytest.raises(HTTPException) as exc:
        authenticate(request_with_token(token))

    assert exc.value.status_code == 403


def test_diagnostics_permission_is_required(monkeypatch):
    monkeypatch.setenv("RAG_JWT_SECRET", "secret")
    token = jwt.encode({"sub": "u1", "tenant_id": "t1"}, "secret", algorithm="HS256")
    with pytest.raises(HTTPException) as exc:
        from ai_gateway.auth import authenticate_diagnostics
        authenticate_diagnostics(request_with_token(token))
    assert exc.value.status_code == 403


def test_diagnostics_permission_is_accepted(monkeypatch):
    monkeypatch.setenv("RAG_JWT_SECRET", "secret")
    token = jwt.encode({"sub": "u1", "tenant_id": "t1", "permissions": ["diagnostics:read"]}, "secret", algorithm="HS256")
    from ai_gateway.auth import authenticate_diagnostics
    assert authenticate_diagnostics(request_with_token(token)) == ("t1", "u1")
