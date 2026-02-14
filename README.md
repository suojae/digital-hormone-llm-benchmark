# Digital Hormone LLM Benchmark

This repository contains a submission-ready paper draft for:

`Emotion-Inspired State Compression for Foundation-Model Agents: A Tri-Axis Digital Hormone Controller for Risk- and Budget-Aware Behavior`

## Structure

- `paper/paper.tex`: main manuscript
- `paper/refs.bib`: bibliography
- `paper/figures/`: figure outputs (placeholders now)

## Build

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

## Current Status

- The manuscript is polished and formatted for submission drafting.
- `Results` and some figures/tables intentionally remain placeholders until experiments are run.
- The protocol emphasizes paired ON/OFF evaluation, deterministic scoring, and reproducibility.
