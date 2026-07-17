---
name: climate-pulse-paper-generation
description: Generate publication-ready artifacts, metrics tables, and the final PDF research paper for Climate Pulse.
---

# Climate Pulse Paper Generation

This skill guides the generation of the final academic paper and all supporting artifacts (plots, metrics tables, comparison figures).

## Prerequisites

All models must be trained before generating paper artifacts. Checkpoints must exist in `runtime/models/checkpoints/`.

## Workflow

### 1. Generate paper artifacts (figures, metrics, LaTeX tables)
```bash
make gen-research-paper-artifacts
```
This runs `src/scripts/generate_paper_artifacts.py` with `PYTHONPATH=src`.  
Outputs are saved to `runtime/paper_artifacts/`.

Artifacts include:
- Evaluation metrics for all models (RMSE, MAE, skill scores)
- Comparison plots (TFT vs LSTM for El Niño, ST-GNN loss curves, etc.)
- LaTeX-ready tables for the research paper

### 2. Compile the PDF research paper
```bash
make gen-research-paper-pdf
```
This runs Pandoc + XeLaTeX on `docs/research_paper/research_paper.md` and saves `climatepulse_research_paper.pdf` to the same directory.

Requirements: `pandoc` and `xelatex` (TeX Live or MikTeX) must be installed on the system.

### 3. Clean generated artifacts
```bash
make clean-research-paper-artifacts   # Removes runtime/paper_artifacts/
```

## Key Source Files

| File | Purpose |
|:---|:---|
| `src/scripts/generate_paper_artifacts.py` | Main artifact generation entry point |
| `src/visualization/charts.py` | Plot generation helpers |
| `src/visualization/themes.py` | Matplotlib/Plotly theme configuration |
| `src/models/el_nino/evaluate.py` | TFT / LSTM evaluation metrics |
| `src/models/pollution/evaluate.py` | ST-GNN evaluation metrics |
| `src/models/climate_feedback/evaluate.py` | Bayesian model evaluation |
| `docs/research_paper/research_paper.md` | Paper source (Markdown + LaTeX) |
