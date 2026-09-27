from __future__ import annotations

import httpx

class AgentClient:
    def __init__(self, base_url: str, timeout_seconds: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def chat(self, messages: list[dict], model: str | None = None) -> dict:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/chat",
                    json={"messages": messages, "model": model},
                )
                response.raise_for_status()
                return response.json()
        except httpx.TimeoutException as exc:
            raise TimeoutError("agent-core timeout") from exc
        except httpx.RequestError as exc:
            raise ConnectionError("agent-core unavailable") from exc


    async def answer(self, messages: list[dict], model: str | None = None) -> dict:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/answer",
                    json={"messages": messages, "model": model},
                )
                response.raise_for_status()
                return response.json()
        except httpx.TimeoutException as exc:
            raise TimeoutError("agent-core timeout") from exc
        except httpx.RequestError as exc:
            raise ConnectionError("agent-core unavailable") from exc

    async def stream_answer(self, messages: list[dict], model: str | None = None):
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/v1/answer/stream",
                    json={"messages": messages, "model": model},
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("event: "):
                            event = line[7:]
                        elif line.startswith("data: "):
                            yield event, line[6:]
        except httpx.TimeoutException as exc:
            raise TimeoutError("agent-core timeout") from exc
        except httpx.RequestError as exc:
            raise ConnectionError("agent-core unavailable") from exc
