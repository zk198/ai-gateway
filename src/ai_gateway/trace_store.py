from __future__ import annotations

from collections import OrderedDict
from typing import Any


class TraceStore:
    """Bounded in-memory Level-3 trace store.

    The interface deliberately hides storage from the API so a persistent backend can
    replace it without changing the UI contract.
    """

    def __init__(self, max_traces: int = 1000) -> None:
        self.max_traces = max_traces
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def put(self, trace: dict[str, Any], *, tenant_id: str, user_id: str) -> None:
        trace_id = str(trace.get("trace_id", ""))
        if not trace_id:
            return
        self._items.pop(trace_id, None)
        self._items[trace_id] = {"tenant_id": tenant_id, "user_id": user_id, "trace": trace}
        while len(self._items) > self.max_traces:
            self._items.popitem(last=False)

    def get(self, trace_id: str, *, tenant_id: str, user_id: str) -> dict[str, Any] | None:
        item = self._items.get(trace_id)
        if item is None:
            return None
        if item["tenant_id"] != tenant_id or item["user_id"] != user_id:
            return None
        self._items.move_to_end(trace_id)
        return item["trace"]
