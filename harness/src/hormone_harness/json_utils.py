
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from jsonschema import Draft7Validator


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
_FIRST_OBJ_RE = re.compile(r"(\{.*\})", re.DOTALL)
_FIRST_ARR_RE = re.compile(r"(\[.*\])", re.DOTALL)


@dataclass
class JsonParseError(Exception):
    message: str
    raw_text: str

    def __str__(self) -> str:
        return self.message


def _strip_code_fences(text: str) -> str:
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def extract_json_candidate(text: str) -> str:
    """Extract a plausible JSON substring from model output.

    Strategy:
    1) If output includes ```json fences, use the fenced content.
    2) Otherwise try to find the largest {...} or [...] span.
    3) Fallback: return stripped text (caller may still try json.loads).
    """
    t = _strip_code_fences(text)

    # Try object
    m = _FIRST_OBJ_RE.search(t)
    if m:
        cand = m.group(1).strip()
        return cand

    # Try array
    m = _FIRST_ARR_RE.search(t)
    if m:
        cand = m.group(1).strip()
        return cand

    return t.strip()


def parse_json(text: str) -> Any:
    """Parse model output into JSON. Raises JsonParseError."""
    cand = extract_json_candidate(text)
    try:
        return json.loads(cand)
    except Exception as e:
        raise JsonParseError(message=f"Failed to parse JSON: {e}", raw_text=text)


def validate_json(instance: Any, schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate instance against a JSON schema. Returns (ok, error_message)."""
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda x: x.path)
    if not errors:
        return True, None
    # Return the first error for concise repair prompts
    err = errors[0]
    path = ".".join([str(p) for p in err.path]) if err.path else "<root>"
    return False, f"{path}: {err.message}"


def make_repair_prompt(schema_name: str, schema: Dict[str, Any], error: str, raw: str) -> str:
    """Prompt to repair invalid JSON outputs with minimal extra tokens."""
    # Avoid dumping the full schema; keep it short and refer to constraints.
    # If you want, you can embed a compact schema summary here.
    return (
        f"You returned invalid JSON for schema '{schema_name}'.\n"
        f"Validation error: {error}\n"
        f"Your previous output was:\n{raw}\n\n"
        "Return ONLY a single JSON object that conforms to the required schema. "
        "Do not include any extra keys, commentary, or markdown fences."
    )
