# AI Gateway

The authenticated application/API boundary for the AI Platform. It exposes client-facing RAG, conversation, answer/chat, streaming and operational diagnostic APIs while keeping internal services private.

## Responsibilities
- Authenticate requests using the configured local JWT mechanism.
- Provide retrieval/search and grounded answer/chat APIs.
- Orchestrate downstream calls to `rag-ingestion`, `rag-retrieval` and `agent-core`.
- Persist conversation/session state and privileged trace/diagnostic data in PostgreSQL.
- Propagate request/correlation IDs and emit operational/audit events.
- Optionally call the Laya System-1 decision service when enabled.

The browser/UI should call this gateway only, never PostgreSQL, Qdrant, ingestion, retrieval, agent, MCP tool or LLM services directly.

## API
The FastAPI application provides health, retrieval/search, answer/chat and streaming, conversation/session, and privileged diagnostics operations. Consult `/docs` in a local deployment for the exact current routes and schemas. Search remains retrieval-only; answer generation and agent orchestration are separate operations.

## Configuration
Required deployment topology includes `RAG_RETRIEVAL_URL`, `RAG_INGESTION_URL`, `AI_AGENT_URL`, `AI_POSTGRES_DSN` and `LAYA_URL`. Timeout and optional Laya controls include `RAG_DOWNSTREAM_TIMEOUT_SECONDS`, `AI_AGENT_TIMEOUT_SECONDS`, `LAYA_ENABLED`, `LAYA_TIMEOUT_SECONDS` and `LAYA_API_KEY`. See the settings/auth modules for the complete environment contract. Do not commit secrets.

## Development
Python 3.11+.
```bash
uv sync --extra test
uv run pytest
```
Run via `ai-infra` with all downstream URLs and PostgreSQL configured explicitly.

## Security and operations
Expose only this service to the UI; keep downstream services private. Use strong JWT/database credentials, restrict privileged diagnostics, configure CORS to trusted UI origins, propagate correlation IDs, and redact prompts, documents, tokens and sensitive payloads from logs.
