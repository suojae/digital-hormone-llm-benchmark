# Digital Hormone LLM Benchmark

This repository contains:

1. A submission-ready paper draft for:
`Emotion-Inspired State Compression for Foundation-Model Agents: A Tri-Axis Digital Hormone Controller for Risk- and Budget-Aware Behavior`
2. A reproducible experiment harness (`harness/`) for paired ON/OFF evaluation.

## Structure

- `paper/paper.tex`: main manuscript (integrated draft)
- `paper/sections/`: modular section template (v2)
- `paper/refs.bib`: bibliography used by the integrated draft
- `paper/references.bib`: template bibliography variant (v2)
- `paper/figures/architecture.pdf`: architecture figure scaffold
- `harness/`: tri-axis controller + schema enforcement + JSONL logging + plotting scripts

## Paper Build

From repository root:

```bash
cd paper
latexmk -pdf paper.tex
```

If `latexmk` is unavailable:

```bash
cd paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

## Harness Quick Start

```bash
cd harness
python scripts/run_toy_demo.py --out outputs/toy_demo
python scripts/aggregate_results.py --root outputs/toy_demo --out results.csv
python scripts/plot_pareto.py --csv results.csv --out_dir figures
```

## Current Status

- Manuscript text is polished, but `Results` remains placeholder until real experiments are run.
- Harness includes: schema validation/repair, paired ON/OFF run conventions, step-level JSONL logs, and utility-risk-cost aggregation scripts.
- Recommended main benchmark is WebArena-Verified with deterministic evaluation and replay.
