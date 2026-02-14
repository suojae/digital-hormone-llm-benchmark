from __future__ import annotations

from typing import Tuple
from hormone_harness.adapters.base import LLMRequest
from hormone_harness.controller import Usage


class AnthropicAdapter:
    def __init__(self, model: str, api_key_env: str = "ANTHROPIC_API_KEY") -> None:
        self.model = model
        self.api_key_env = api_key_env

    def generate(self, req: LLMRequest) -> Tuple[str, Usage]:
        raise NotImplementedError("Implement provider call here.")