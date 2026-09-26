from __future__ import annotations

import httpx

from .models import GatewaySettings

class RAGService:
    def __init__(self, settings: GatewaySettings) -> None:
        self.settings = settings

    async def _request(self, method: str, url: str, tenant: str, user: str, **kwargs: object) -> httpx.Response:
        headers = {"X-RAG-Tenant-ID": tenant, "X-RAG-User-ID": user}
        caller_headers = kwargs.pop("headers", None)
        if isinstance(caller_headers, dict):
            headers.update(caller_headers)
        try:
            async with httpx.AsyncClient(timeout=self.settings.timeout_seconds) as client:
                return await client.request(method, url, headers=headers, **kwargs)
        except httpx.TimeoutException as exc:
            raise TimeoutError("downstream timeout") from exc
        except httpx.RequestError as exc:
            raise ConnectionError("downstream unavailable") from exc

    async def search(self, query: str, limit: int, tenant: str, user: str) -> list[dict]:
        response = await self._request("POST", f"{self.settings.retrieval_url}/search", tenant, user, json={"query": query, "limit": limit})
        response.raise_for_status()
        return response.json()

    async def get_sources(self, tenant: str, user: str) -> list[dict]:
        response = await self._request("GET", f"{self.settings.ingestion_url}/sources", tenant, user)
        response.raise_for_status()
        return response.json()

    async def get_source(self, source_name: str, tenant: str, user: str) -> dict:
        response = await self._request("GET", f"{self.settings.ingestion_url}/sources/{source_name}", tenant, user)
        response.raise_for_status()
        return response.json()

    async def get_stats(self, tenant: str, user: str) -> dict:
        response = await self._request("GET", f"{self.settings.ingestion_url}/stats", tenant, user)
        response.raise_for_status()
        return response.json()

    async def get_message(self, message_id: str, tenant: str, user: str) -> dict:
        response = await self._request("GET", f"{self.settings.ingestion_url}/messages/{message_id}", tenant, user)
        response.raise_for_status()
        return response.json()

    async def get_document(self, document_id: str, tenant: str, user: str) -> dict:
        response = await self._request("GET", f"{self.settings.ingestion_url}/documents/{document_id}", tenant, user)
        response.raise_for_status()
        return response.json()

    async def ingest(self, body: bytes, content_type: str, tenant: str, user: str) -> httpx.Response:
        return await self._request("POST", f"{self.settings.ingestion_url}/ingest", tenant, user, content=body, headers={"content-type": content_type})

    async def upload(self, body: bytes, content_type: str, tenant: str, user: str) -> httpx.Response:
        return await self._request("POST", f"{self.settings.ingestion_url}/upload", tenant, user, content=body, headers={"content-type": content_type})
