
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import random

from hormone_harness.controller import Observation, Outcome


@dataclass
class ToyState:
    task_id: int
    step: int
    progress: float
    risk_level: float
    seed: int


class ToyWebEnv:
    """A small deterministic environment to test the harness without external dependencies.

    - Tools: browser.read / browser.find / browser.goto / browser.click / browser.type / browser.scroll
    - Observations: include synthetic warnings when risk_level is high.
    - Success: when progress exceeds 1.0.

    This is *not* a benchmark; it exists for unit testing and demos.
    """

    def __init__(self) -> None:
        self.state: ToyState | None = None

    def reset(self, task_id: int, seed: int) -> Observation:
        rng = random.Random(seed + task_id * 997)
        self.state = ToyState(
            task_id=task_id,
            step=0,
            progress=0.0,
            risk_level=rng.random() * 0.6,  # baseline risk
            seed=seed,
        )
        return self._obs()

    def _obs(self) -> Observation:
        assert self.state is not None
        warnings = []
        if self.state.risk_level > 0.7:
            warnings.append("spam_warning")
        if self.state.risk_level > 0.85:
            warnings.append("policy_guardrail_near_limit")
        env_text = (
            f"[ToyEnv] step={self.state.step} progress={self.state.progress:.2f} risk={self.state.risk_level:.2f}\n"
            "Available tools: browser.read, browser.find, browser.goto, browser.click, browser.type, browser.scroll\n"
        )
        return Observation(
            user_goal=f"(toy task {self.state.task_id}) Find the target information safely and efficiently.",
            env_text=env_text,
            warnings=warnings,
            budget={"tokens_left": None},
        )

    def step_tool(self, tool: str, args: Dict[str, Any]) -> Tuple[Observation, Outcome]:
        assert self.state is not None
        rng = random.Random(self.state.seed + self.state.task_id * 997 + self.state.step)
        self.state.step += 1

        # Default outcome
        out = Outcome(success=None, progress=None, risk_events=[], tool_errors=0, policy_blocks=0)

        # Tool effect on progress and risk
        if tool in ("browser.read", "browser.find"):
            self.state.progress += 0.35 + 0.10 * rng.random()
            self.state.risk_level = max(0.0, self.state.risk_level - 0.05)
        elif tool in ("browser.goto", "browser.click"):
            self.state.progress += 0.20 + 0.10 * rng.random()
            self.state.risk_level = min(1.0, self.state.risk_level + 0.05)
        elif tool in ("browser.type",):
            # typing can be risky in some tasks (mutation)
            self.state.progress += 0.15
            self.state.risk_level = min(1.0, self.state.risk_level + 0.12)
            out.risk_events = ["mutation_attempt"]
        else:
            # unknown tools treated as tool error
            out.tool_errors = 1
            out.risk_events = ["unknown_tool"]

        # Synthetic policy block when risk very high
        if self.state.risk_level > 0.92 and tool in ("browser.type", "browser.click"):
            out.policy_blocks = 1
            out.risk_events = (out.risk_events or []) + ["policy_blocked"]

        # Provide progress signal
        out.progress = min(1.0, self.state.progress)

        # Success when enough progress
        if self.state.progress >= 1.0:
            out.success = True

        return self._obs(), out
