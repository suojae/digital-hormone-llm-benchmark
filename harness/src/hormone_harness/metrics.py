from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import math
import random


@dataclass
class EpisodeMetrics:
    utility: float
    risk: float
    cost_tokens: int
    steps: int
    success: bool


def bootstrap_ci(deltas: List[float], iters: int = 5000, alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    """Simple bootstrap CI for paired deltas (ON - OFF)."""
    rng = random.Random(seed)
    n = len(deltas)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    means = []
    for _ in range(iters):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((alpha / 2) * iters)
    hi_idx = int((1 - alpha / 2) * iters) - 1
    return {"mean": sum(deltas) / n, "lo": means[lo_idx], "hi": means[hi_idx]}