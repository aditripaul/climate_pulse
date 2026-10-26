---
name: climate-pulse-paper-generation
description: Generate publication-ready artifacts, metrics tables, research paper PDF, and patent specification PDF for Climate Pulse.
---

# Climate Pulse Paper & Patent Document Generation

This skill guides the generation of academic research papers, patent specifications, and supporting evaluation artifacts (plots, metrics tables, comparison figures).

## Prerequisites

All models must be trained before generating paper artifacts. Checkpoints must exist in `runtime/models/checkpoints/`.

## Workflow

### 1. Generate paper & patent artifacts (figures, metrics, LaTeX tables)
```bash
make gen-eval-artifacts
```
This runs `src/scripts/generate_eval_artifacts.py` with `PYTHONPATH=src`.  
Outputs are saved to `runtime/eval_artifacts/`.

Artifacts include:
- Evaluation metrics for all models (RMSE, MAE, skill scores)
- Comparison plots (TFT vs LSTM for El Niño, ST-GNN loss curves, etc.)
- LaTeX-ready tables for research publications and patent documentation

### 2. Compile the PDF research paper
```bash
make save-run              # snapshot checkpoints + eval artifacts to runtime/saved/
make sync-paper-artifacts  # point docs/research_paper/artifacts at the newest snapshot
make gen-research-paper-pdf
```
`docs/research_paper/artifacts` is a symlink to `runtime/saved/run_save.latest`, so the paper always reads a saved snapshot; never copy files into it. Runs Pandoc + XeLaTeX on `docs/research_paper/research_paper.md` and compiles `climatepulse_research_paper.pdf`.

### 3. Compile the PDF patent specification
```bash
make gen-patent-application-pdf
```
Runs Pandoc + XeLaTeX on `docs/patent/patent_application.md` and compiles `climatepulse_patent_application.pdf`.

### 4. Clean generated artifacts
```bash
make clean-eval-artifacts   # Removes runtime/eval_artifacts/
```

## Key Source Files

| File | Purpose |
|:---|:---|
| `src/scripts/generate_eval_artifacts.py` | Main artifact generation entry point |
| `src/visualization/charts.py` | Plot generation helpers |
| `src/visualization/themes.py` | Matplotlib/Plotly theme configuration |
| `docs/research_paper/research_paper.md` | Research paper source (Markdown + LaTeX) |
| `docs/patent/patent_application.md` | Consolidated patent specification source |
| `docs/project_proposal.md` | High-level project proposal |
