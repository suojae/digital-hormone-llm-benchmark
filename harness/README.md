
Digital Hormone Harness (v0.2.0)

Goal
----
Run *paired* hormone OFF vs ON experiments for LLM agents in interactive environments,
and log step-level traces to quantify utility–risk–cost trade-offs.

This repository intentionally treats “emotion” as a **functional control metaphor**
(state compression + regime switching), not as sentience or clinical affect.

Why this harness exists
-----------------------
Reviewers typically reject “wrapper ideas” when the evaluation is:
- not reproducible (uncontrolled environment drift),
- not fair (different info / different tools / different prompting),
- not statistically grounded (no paired runs, no CIs),
- or not ablated (could be explained by simple temperature tuning).

This harness bakes in:
- JSON schema enforcement + repair loop,
- deterministic run directory structure,
- step-level JSONL logging (hormones + controls + outcomes + usage),
- paired ON/OFF conventions for bootstrap CI analysis.

Key files
---------
- src/hormone_harness/controller.py
  Tri-axis hormone dynamics + hysteretic regime selection + control mapping.

- src/hormone_harness/runner/webarena_verified.py
  Generic runner using a minimal WebEnv interface + JSON schema validation.

- schemas/agent_step_action_schema.json
  Step-level schema (tool vs final) for provider-agnostic tool calling.

- schemas/webarena_verified_agent_response_schema.json
  Final schema compatible with WebArena-Verified style evaluation.

- scripts/aggregate_results.py
  Convert outputs/*/task_*/... logs into a results.csv.

- scripts/plot_pareto.py, scripts/plot_hormones.py
  Produce paper-ready plots (PNG, 300dpi). Switch to PDF/SVG for camera-ready.

Quick smoke test (no external services)
---------------------------------------
This verifies wiring without any real models/benchmarks:

  python scripts/run_toy_demo.py --out outputs/toy_demo

You should see paired OFF/ON runs under outputs/toy_demo/.

Integrating a real benchmark (WebArena-Verified)
------------------------------------------------
1) Install WebArena/WebArena-Verified and set up self-hosted sites as instructed by their docs.
2) Implement a WebEnv wrapper that:
   - reset(task_id, seed) -> Observation
   - step_tool(tool, args) -> (Observation, Outcome)

3) Make sure you:
   - capture HAR network traces if you plan offline evaluation,
   - keep tool/action space identical across models for fairness,
   - run paired OFF/ON for each task_id and seed.

Provider adapters (GPT / Gemini / Claude)
-----------------------------------------
Implement your SDK calls in:
- src/hormone_harness/adapters/openai.py
- src/hormone_harness/adapters/google.py
- src/hormone_harness/adapters/anthropic.py

The harness uses *plain JSON* (not vendor-specific tool calling) for fairness.

Aggregating + plotting
----------------------
After you run experiments:

  python scripts/aggregate_results.py --root outputs/webarena_verified --out results.csv
  python scripts/plot_pareto.py --csv results.csv --out_dir figures

Ethics note
-----------
Avoid framing the system as “inducing depression”. Prefer:
- defensive / risk-off mode
- low-initiative mode
- budget-constrained mode

If you evaluate in the wild (SNS, real services), you must comply with platform policies
and avoid misuse (spam, manipulation).
