
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple, List

from hormone_harness.controller import Usage


@dataclass
class LLMRequest:
    """A provider-agnostic request.

    Keep this minimal for fairness across GPT/Gemini/Claude, but allow optional fields
    for stricter JSON emission (if the provider supports it).
    """
    system: str
    user: str

    temperature: float = 0.5
    max_tokens: int = 800

    # Optional decoding controls (not all providers expose the same knobs)
    top_p: Optional[float] = None
    stop: Optional[List[str]] = None

    # Optional structured output guidance
    response_format: Optional[str] = None  # e.g., "json"
    json_schema: Optional[Dict[str, Any]] = None  # if provider supports schema-native JSON mode


class ModelAdapter(Protocol):
    """Unify GPT/Gemini/Claude behind the same interface."""

    def generate(self, req: LLMRequest) -> Tuple[str, Usage]:
        """Return (text, usage)."""
        ...
