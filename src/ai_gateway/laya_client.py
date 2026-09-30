from __future__ import annotations

import httpx

from .observability import request_id


class LayaClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 5.0,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key
        self.transport = transport

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        value = request_id()
        if value:
            headers["X-Request-ID"] = value
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def decide(self, state: object, questions: dict, *, model: str | None = None) -> dict:
        body: dict[str, object] = {"state": state, "questions": questions}
        if model:
            body["model"] = model
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds, transport=self.transport) as client:
                response = await client.post(
                    f"{self.base_url}/v1/systemone",
                    json=body,
                    headers=self._headers(),
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or "answers" not in payload:
                    raise ValueError("malformed Laya response")
                return payload
        except httpx.TimeoutException as exc:
            raise TimeoutError("laya timeout") from exc
        except httpx.RequestError as exc:
            raise ConnectionError("laya unavailable") from exc
