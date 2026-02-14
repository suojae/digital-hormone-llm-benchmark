#!/usr/bin/env python3
"""
Plot hormone trajectories for a single run (steps.jsonl).

Usage:
  python scripts/plot_hormones.py --steps outputs/<model>/task_<id>/ON_seed0/steps.jsonl --out figures/hormones.png
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=str, required=True)
    ap.add_argument("--out", type=str, default="hormones.png")
    args = ap.parse_args()

    steps_path = Path(args.steps)
    rows = []
    with steps_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    t = [r["t"] for r in rows]
    d = [r.get("hormones", {}).get("dopamine") if r.get("hormones") else None for r in rows]
    c = [r.get("hormones", {}).get("cortisol") if r.get("hormones") else None for r in rows]
    e = [r.get("hormones", {}).get("energy") if r.get("hormones") else None for r in rows]
    regimes = [r.get("controls", {}).get("regime", "unknown") for r in rows]

    plt.figure(figsize=(7, 4))
    if d[0] is not None:
        plt.plot(t, d, label="dopamine")
        plt.plot(t, c, label="cortisol")
        plt.plot(t, e, label="energy")
        plt.ylim(-0.05, 1.05)
        plt.xlabel("step")
        plt.ylabel("value")
        plt.title("Hormone trajectories")
        plt.legend()
    else:
        plt.text(0.1, 0.5, "No hormone data (condition OFF)", fontsize=12)
        plt.axis("off")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    # Also write regimes as a simple text file for debugging
    (out_path.with_suffix(".regimes.txt")).write_text("\n".join(regimes), encoding="utf-8")
    print(f"Saved {out_path.resolve()}")

if __name__ == "__main__":
    main()
