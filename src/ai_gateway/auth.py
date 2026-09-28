from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Any

import jwt

from .observability import audit_event
from fastapi import HTTPException, Request
from jwt import PyJWKClient

_mcp_authorization: ContextVar[str | None] = ContextVar("ai_gateway_mcp_authorization", default=None)


def set_mcp_authorization(header: str | None):
    return _mcp_authorization.set(header)


def reset_mcp_authorization(token) -> None:
    _mcp_authorization.reset(token)


def _reject(status: int, detail: str, reason: str) -> None:
    audit_event("auth_failure", operation="authenticate", outcome="denied", reason=reason)
    raise HTTPException(status, detail)


def _claims(request: Request) -> dict[str, Any]:
    header = request.headers.get("Authorization") or _mcp_authorization.get() or ""
    if not header.startswith("Bearer ") or not header[7:].strip():
        _reject(401, "Bearer token required", "missing_bearer")
    token = header[7:].strip()
    secret = os.getenv("RAG_JWT_SECRET")
    jwks = os.getenv("RAG_JWKS_URL")
    algorithms = [value.strip() for value in os.getenv("RAG_JWT_ALGORITHMS", "HS256" if secret else "RS256").split(",") if value.strip()]
    issuer = os.getenv("RAG_JWT_ISSUER")
    audience = os.getenv("RAG_JWT_AUDIENCE")
    if not algorithms:
        raise HTTPException(500, "JWT algorithms are not configured")
    try:
        if secret:
            claims = jwt.decode(token, secret, algorithms=algorithms, issuer=issuer, audience=audience, options={"verify_iss": issuer is not None, "verify_aud": audience is not None})
        elif jwks:
            key = PyJWKClient(jwks).get_signing_key_from_jwt(token).key
            claims = jwt.decode(token, key, algorithms=algorithms, issuer=issuer, audience=audience, options={"verify_iss": issuer is not None, "verify_aud": audience is not None})
        else:
            raise HTTPException(500, "JWT verifier is not configured")
    except jwt.ExpiredSignatureError as exc:
        audit_event("auth_failure", operation="authenticate", outcome="denied", reason="expired")
        raise HTTPException(401, "invalid token") from exc
    except jwt.PyJWTError as exc:
        audit_event("auth_failure", operation="authenticate", outcome="denied", reason="invalid")
        raise HTTPException(401, "invalid token") from exc
    return claims


def authenticate(request: Request) -> tuple[str, str]:
    claims = _claims(request)
    tenant = claims.get("tenant_id") or claims.get("tid")
    user = claims.get("sub")
    if not tenant or not user:
        _reject(403, "token must contain tenant_id and sub", "missing_identity_claims")
    return str(tenant), str(user)


def authenticate_diagnostics(request: Request) -> tuple[str, str]:
    tenant, user = authenticate(request)
    claims = _claims(request)
    permissions = claims.get("permissions", [])
    scope = claims.get("scope", "")
    if isinstance(permissions, str):
        permissions = permissions.split()
    if not isinstance(permissions, list):
        permissions = []
    if "diagnostics:read" not in permissions and "diagnostics:read" not in str(scope).split():
        raise HTTPException(403, "diagnostics:read permission required")
    return tenant, user
