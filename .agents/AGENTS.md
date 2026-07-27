# ClimatePulse Workspace Rules

This file defines project-specific rules and guidelines for AI coding agents operating on the ClimatePulse repository.

## Project Standards & Environment
- **Python Version:** 3.12 strictly pinned across `.python-version`, `Makefile`, and `pyproject.toml`.
- **Environment & Imports:** All module imports within `src/` must use bare module paths relative to `PYTHONPATH=src` (e.g., `import config`, `from data.pipeline import DataPipeline`). Do not use `src.` as a module prefix.
- **Hardware Acceleration:** Support CUDA, AMD ROCm 7.2 (`HSA_OVERRIDE_GFX_VERSION=10.3.0`), and Intel NPU via OpenVINO.

## Documentation & Publication Workflows
- **Research Paper:** Source markdown is maintained at `docs/research_paper/research_paper.md`. Compile to PDF using `make gen-research-paper-pdf`.
- **Patent Application:** Consolidated single patent specification is maintained at `docs/patent/patent_application.md`. Compile to PDF using `make gen-patent-application-pdf`.
- **Project Proposal:** High-level project proposal is maintained at `docs/project_proposal.md`.

## Data Pipeline & Modeling Integrity
- **Normalization Leakage:** Normalization statistics ($\mu, \sigma$) must be computed strictly on the training split and applied to validation/test splits.
- **Path Resolution:** All file paths must be derived relative to `PROJECT_ROOT` in `src/config.py`.
- **Stateless Orchestration:** `DataPipeline` in `src/data/pipeline.py` is a stateless orchestrator. Storage is handled via `src/data/storage/db.py` using Parquet tables and JSON metadata.
