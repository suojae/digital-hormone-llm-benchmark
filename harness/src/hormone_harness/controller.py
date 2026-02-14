
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
import json
import os
import math


def clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def exp_saturate(x: float) -> float:
    """Map x>=0 to (0,1) with diminishing returns."""
    if x <= 0:
        return 0.0
    return 1.0 - math.exp(-x)


@dataclass
class DigitalHormones:
    """Tri-axis internal state.

    Design intent:
    - Functional control signals (state compression), not clinical affect.
    - Values in [0, 1], updated per decision tick.
    """
    dopamine: float = 0.5   # reward / progress / positive feedback
    cortisol: float = 0.0   # risk / stress / guardrail pressure
    energy: float = 1.0     # budget / resources remaining

    def to_dict(self) -> Dict[str, float]:
        return {"dopamine": float(self.dopamine), "cortisol": float(self.cortisol), "energy": float(self.energy)}


@dataclass
class HormoneConfig:
    """Hyperparameters for hormone dynamics.

    Notes:
    - Use small deltas by default to avoid instability.
    - Prefer ablations that show robustness to reasonable ranges.
    """
    # Tick-based decay (homeostasis)
    decay_dopamine: float = 0.90
    decay_cortisol: float = 0.98

    # Event gains
    k_reward: float = 0.12
    k_risk: float = 0.18

    # Risk composition weights
    w_warning: float = 0.10
    w_risk_event: float = 0.15
    w_tool_error: float = 0.20
    w_policy_block: float = 0.35  # strong guardrail block / prohibited action

    # Energy budgeting
    token_budget: int = 200_000
    k_token_cost: float = 1.0  # scale on normalized token cost

    # Hysteresis thresholds (avoid mode-chatter)
    defensive_enter: float = 0.80
    defensive_exit: float = 0.60

    exploratory_enter: float = 0.75
    exploratory_exit: float = 0.55

    low_energy_enter: float = 0.20
    low_energy_exit: float = 0.30


@dataclass
class ControlParams:
    """What the controller outputs to modulate an LLM call."""
    system_modifier: str
    temperature: float
    max_tokens: int
    tool_allowlist: Optional[List[str]] = None
    regime: str = "neutral"  # one of neutral, exploratory, defensive, low_resource, defensive_low_resource


@dataclass
class Observation:
    """A minimal, model-agnostic observation wrapper.

    Adapt this to each benchmark/environment.
    """
    user_goal: str
    env_text: str
    warnings: List[str]
    budget: Dict[str, Any]  # e.g., {"tokens_left": int, "calls_left": int}


@dataclass
class Outcome:
    """Outcome of the last step/action (environment + tooling)."""
    success: Optional[bool] = None
    progress: Optional[float] = None  # optional [0,1]
    risk_events: Optional[List[str]] = None
    tool_errors: Optional[int] = None
    policy_blocks: Optional[int] = None  # times blocked by a safety/permission rule


@dataclass
class Usage:
    """Token/cost accounting returned by the model provider if available."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class DigitalHormoneController:
    """State update + policy mapping wrapper around a base LLM.

    This controller is deliberately lightweight and interpretable. The goal is to provide
    a low-dimensional 'state compression' layer that biases behavior under competing
    objectives (utility, safety/risk, cost).
    """

    def __init__(
        self,
        hormones: Optional[DigitalHormones] = None,
        config: Optional[HormoneConfig] = None,
        persist_path: str = "memory/hormones.json",
        persist_cortisol_only: bool = True,
        reset_dopamine_on_load: bool = True,
        reset_energy_on_load: bool = False,
    ) -> None:
        self.h = hormones or DigitalHormones()
        self.cfg = config or HormoneConfig()
        self.persist_path = persist_path
        self.persist_cortisol_only = persist_cortisol_only
        self.reset_dopamine_on_load = reset_dopamine_on_load
        self.reset_energy_on_load = reset_energy_on_load

        # Regime state for hysteresis
        self._regime = "neutral"

    # ---------------- Persistence ----------------
    def save_state(self) -> None:
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        data = self.h.to_dict()
        if self.persist_cortisol_only:
            data = {"cortisol": data["cortisol"]}
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_state(self) -> None:
        if not os.path.exists(self.persist_path):
            return
        with open(self.persist_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "cortisol" in data:
            self.h.cortisol = clip01(float(data["cortisol"]))
        if not self.reset_dopamine_on_load and "dopamine" in data:
            self.h.dopamine = clip01(float(data["dopamine"]))
        else:
            self.h.dopamine = 0.5
        if not self.reset_energy_on_load and "energy" in data:
            self.h.energy = clip01(float(data["energy"]))
        elif self.reset_energy_on_load:
            self.h.energy = 1.0

    # ---------------- Feature extraction ----------------
    def _reward_signal(self, outcome: Outcome) -> float:
        # Primary: success and optional progress
        sig = 0.0
        if outcome.success is True:
            sig += 1.0
        if outcome.progress is not None:
            sig += clip01(float(outcome.progress))
        # saturate to [0,1]
        return clip01(sig / 2.0)

    def _risk_signal(self, obs: Observation, outcome: Outcome) -> float:
        warnings = len(obs.warnings) if obs.warnings else 0
        risk_events = len(outcome.risk_events) if outcome.risk_events else 0
        tool_errors = int(outcome.tool_errors or 0)
        policy_blocks = int(outcome.policy_blocks or 0)

        # Weighted sum then saturate (diminishing returns)
        raw = (
            self.cfg.w_warning * warnings
            + self.cfg.w_risk_event * risk_events
            + self.cfg.w_tool_error * tool_errors
            + self.cfg.w_policy_block * policy_blocks
        )
        return exp_saturate(raw)

    def _token_cost_signal(self, usage: Usage) -> float:
        # Normalize token consumption to [0,1] by budget
        spent = max(0, int(usage.total_tokens or 0))
        if self.cfg.token_budget <= 0:
            return 0.0
        return clip01(self.cfg.k_token_cost * (spent / float(self.cfg.token_budget)))

    # ---------------- State update ----------------
    def decay(self) -> None:
        self.h.dopamine = clip01(self.h.dopamine * self.cfg.decay_dopamine)
        self.h.cortisol = clip01(self.h.cortisol * self.cfg.decay_cortisol)
        # energy decays only through explicit cost updates

    def update(self, obs: Observation, outcome: Outcome, usage: Usage) -> None:
        """One tick update (decay + event deltas)."""
        self.decay()

        r_sig = self._reward_signal(outcome)
        k_sig = self._risk_signal(obs, outcome)
        c_sig = self._token_cost_signal(usage)

        # Saturating increments: (1 - current) * signal
        self.h.dopamine = clip01(self.h.dopamine + self.cfg.k_reward * (1.0 - self.h.dopamine) * r_sig)

        # Risk accumulates more stubbornly; also saturating
        self.h.cortisol = clip01(self.h.cortisol + self.cfg.k_risk * (1.0 - self.h.cortisol) * k_sig)

        # Energy decreases linearly with normalized cost
        self.h.energy = clip01(self.h.energy - c_sig)

        self.save_state()

    # ---------------- Regime selection (hysteresis) ----------------
    def _update_regime(self) -> str:
        d, c, e = self.h.dopamine, self.h.cortisol, self.h.energy
        cfg = self.cfg

        # Determine flags with hysteresis
        defensive = False
        exploratory = False
        low_energy = False

        if self._regime.startswith("defensive"):
            defensive = c >= cfg.defensive_exit
        else:
            defensive = c >= cfg.defensive_enter

        if self._regime == "exploratory":
            exploratory = d >= cfg.exploratory_exit and c < cfg.defensive_enter
        else:
            exploratory = d >= cfg.exploratory_enter and c < cfg.defensive_exit

        if "low_resource" in self._regime:
            low_energy = e <= cfg.low_energy_exit
        else:
            low_energy = e <= cfg.low_energy_enter

        # Arbitration: defensive > low_energy > exploratory > neutral
        if defensive and low_energy:
            self._regime = "defensive_low_resource"
        elif defensive:
            self._regime = "defensive"
        elif low_energy:
            self._regime = "low_resource"
        elif exploratory:
            self._regime = "exploratory"
        else:
            self._regime = "neutral"

        return self._regime

    # ---------------- Control mapping ----------------
    def control_params(self) -> ControlParams:
        regime = self._update_regime()
        d, c, e = self.h.dopamine, self.h.cortisol, self.h.energy

        # Defaults
        temperature = 0.5
        max_tokens = 800
        tool_allowlist: Optional[List[str]] = None
        system_modifier = (
            "[MODE: neutral]\n"
            "- Be helpful, grounded in observations, and follow the task.\n"
            "- If uncertain, ask a brief clarifying question.\n"
        )

        if regime == "exploratory":
            temperature = 0.8
            max_tokens = 900
            system_modifier = (
                "[MODE: exploratory]\n"
                "- Proactively explore options and recover from dead-ends.\n"
                "- Keep actions grounded in the current observation.\n"
                "- When multiple strategies exist, try the lowest-risk one first.\n"
            )

        if regime in ("defensive", "defensive_low_resource"):
            temperature = 0.1
            max_tokens = 400 if regime == "defensive" else 250
            tool_allowlist = ["browser.goto", "browser.read", "browser.find", "browser.scroll"]
            system_modifier = (
                "[MODE: defensive]\n"
                "- Prioritize safety, correctness, and permission boundaries.\n"
                "- Do NOT attempt actions that could violate policies or account security.\n"
                "- When uncertain, ask for confirmation or provide a safe alternative.\n"
                "- Avoid unnecessary tool use.\n"
            )

        if regime in ("low_resource", "defensive_low_resource"):
            max_tokens = min(max_tokens, 250)
            temperature = min(temperature, 0.4)
            system_modifier += (
                "\n[MODE: low_resource]\n"
                "- Minimize tokens. Prefer short, high-signal responses.\n"
                "- Use tools only if essential.\n"
            )

        return ControlParams(
            system_modifier=system_modifier,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            tool_allowlist=tool_allowlist,
            regime=regime,
        )

    def tick(self, obs: Observation, outcome: Outcome, usage: Usage) -> Tuple[DigitalHormones, ControlParams]:
        """Convenience method: update hormones then compute control params for next call."""
        self.update(obs, outcome, usage)
        params = self.control_params()
        return self.h, params
