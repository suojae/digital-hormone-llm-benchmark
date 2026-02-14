
from __future__ import annotations

import json
from typing import Tuple

from hormone_harness.adapters.base import LLMRequest
from hormone_harness.controller import Usage


class DummyAdapter:
    """A deterministic dummy adapter for smoke tests.

    It ignores the prompt and alternates between a few tool calls,
    then returns a SUCCESS final response.

    This is NOT for benchmarking; it only verifies the harness wiring.
    """

    def __init__(self, model: str = "dummy") -> None:
        self.model = model
        self._step = 0

    def generate(self, req: LLMRequest) -> Tuple[str, Usage]:
        # Fake token accounting
        usage = Usage(prompt_tokens=50, completion_tokens=30, total_tokens=80)

        # If caller requests the *final* WebArena-Verified schema, return it directly.
        if req.json_schema and isinstance(req.json_schema, dict):
            props = (req.json_schema.get("properties") or {})
            if "action" in props and "status" in props and "results" in props:
                txt = json.dumps({"action": "retrieve", "status": "SUCCESS", "results": ["ok"], "error_details": None})
                return txt, usage

        self._step += 1
        if self._step < 3:
            obj = {"type": "tool", "tool": "browser.read", "args": {}}
        elif self._step < 5:
            obj = {"type": "tool", "tool": "browser.find", "args": {"query": "target"}}
        else:
            obj = {
                "type": "final",
                "final": {"action": "retrieve", "status": "SUCCESS", "results": ["ok"], "error_details": None},
            }
        txt = json.dumps(obj)
        return txt, usage
