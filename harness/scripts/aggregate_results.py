#!/usr/bin/env python3
"""
Aggregate paired experiment logs into a CSV suitable for paper tables/plots.

Directory convention (produced by runner):
  outputs/<model_id>/task_<task_id>/<condition>_seed<seed>/
    - steps.jsonl
    - agent_response.json

Usage:
  python scripts/aggregate_results.py --root outputs/webarena_verified --out results.csv
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
from pathlib import Path
import pandas as pd


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root output directory")
    ap.add_argument("--out", type=str, default="results.csv")
    args = ap.parse_args()

    root = Path(args.root)
    records = []

    for agent_response_path in root.rglob("agent_response.json"):
        run_dir = agent_response_path.parent
        steps_path = run_dir / "steps.jsonl"
        if not steps_path.exists():
            continue

        steps = read_jsonl(steps_path)
        agent_resp = json.loads(agent_response_path.read_text(encoding="utf-8"))

        # Parse identifiers from path
        # .../<model_id>/task_<task_id>/<condition>_seed<seed>/
        parts = run_dir.parts
        try:
            model_id = parts[-3]
            task_part = parts[-2]
            cond_part = parts[-1]
            task_id = int(task_part.replace("task_", ""))
            condition = cond_part.split("_seed")[0]
            seed = int(cond_part.split("_seed")[1])
        except Exception:
            model_id = steps[0].get("model_id", "unknown") if steps else "unknown"
            task_id = int(steps[0].get("task_id", -1)) if steps else -1
            condition = steps[0].get("condition", "unknown") if steps else "unknown"
            seed = int(steps[0].get("seed", -1)) if steps else -1

        total_tokens = sum(int(s.get("usage", {}).get("total_tokens", 0) or 0) for s in steps)
        total_risk = sum(len((s.get("outcome", {}) or {}).get("risk_events", []) or []) for s in steps)
        steps_n = len(steps)

        success = (agent_resp.get("status") == "SUCCESS")
        utility = 1.0 if success else 0.0

        records.append(
            {
                "model_id": model_id,
                "task_id": task_id,
                "seed": seed,
                "condition": condition,
                "success": success,
                "utility": utility,
                "risk_events": total_risk,
                "cost_tokens": total_tokens,
                "steps": steps_n,
            }
        )

    df = pd.DataFrame.from_records(records)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path.resolve()}")


if __name__ == "__main__":
    main()
