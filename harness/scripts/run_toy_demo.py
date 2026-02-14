#!/usr/bin/env python3
"""
Toy demo: runs the harness without external services.

Usage:
  python scripts/run_toy_demo.py --out outputs/toy_demo

This verifies:
- JSON schema validation + repair loop
- hormone updates + regime switching
- step-level JSONL logging
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import argparse
from pathlib import Path

from hormone_harness.adapters.dummy import DummyAdapter
from hormone_harness.controller import DigitalHormoneController
from hormone_harness.runner.toy_env import ToyWebEnv
from hormone_harness.runner.webarena_verified import run_task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="outputs/toy_demo")
    ap.add_argument("--task_id", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env = ToyWebEnv()
    model = DummyAdapter()

    # OFF
    off = run_task(
        task_id=args.task_id,
        seed=args.seed,
        env=env,
        model=model,
        model_id="dummy",
        controller=None,
        condition="OFF",
        out_dir=str(out),
        step_budget=10,
        schemas_dir="schemas",
    )

    # Reset env and model for paired trial
    env = ToyWebEnv()
    model = DummyAdapter()
    controller = DigitalHormoneController()
    on = run_task(
        task_id=args.task_id,
        seed=args.seed,
        env=env,
        model=model,
        model_id="dummy",
        controller=controller,
        condition="ON",
        out_dir=str(out),
        step_budget=10,
        schemas_dir="schemas",
    )

    print("OFF:", off)
    print("ON :", on)
    print(f"Logs written to: {out.resolve()}")


if __name__ == "__main__":
    main()
