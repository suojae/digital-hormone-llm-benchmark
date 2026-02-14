#!/usr/bin/env python3
"""
Create Pareto plots from aggregated results CSV.

Usage:
  python scripts/plot_pareto.py --csv results.csv --out_dir figures
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_scatter(df: pd.DataFrame, x: str, y: str, title: str, out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    for cond, g in df.groupby("condition"):
        plt.scatter(g[x], g[y], label=cond, alpha=0.7)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="figures")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Overall plots
    plot_scatter(df, x="utility", y="risk_events", title="Utility vs Risk (all models)", out_path=out_dir / "pareto_utility_risk_all.png")
    plot_scatter(df, x="utility", y="cost_tokens", title="Utility vs Cost (all models)", out_path=out_dir / "pareto_utility_cost_all.png")

    # Per-model plots
    for model_id, g in df.groupby("model_id"):
        plot_scatter(g, x="utility", y="risk_events", title=f"Utility vs Risk ({model_id})", out_path=out_dir / f"pareto_utility_risk_{model_id}.png")
        plot_scatter(g, x="utility", y="cost_tokens", title=f"Utility vs Cost ({model_id})", out_path=out_dir / f"pareto_utility_cost_{model_id}.png")

    print(f"Saved figures to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
