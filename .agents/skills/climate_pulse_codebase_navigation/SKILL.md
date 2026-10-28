---
name: climate-pulse-codebase-navigation
description: Understand the ClimatePulse repository structure, module layout, and coding conventions. Use this when onboarding, adding new features, debugging, or understanding how the codebase is organized.
---

# Climate Pulse Codebase Navigation

This skill documents the structure and conventions of the Climate Pulse repository.

## Repository Layout

```
climate_pulse.git/
├── .agents/                  # Agent customizations (skills)
├── .env                      # Secrets (gitignored) — copy from .env.example
├── .env.example              # Template for .env
├── .python-version           # Pins Python 3.12
├── configs/
│   ├── runtime_conf.yaml     # Runtime configuration defaults
│   └── dataset_conf.yaml     # Data source URL defaults
├── datasets/
│   ├── raw/                  # Downloaded raw data files
│   ├── processed/            # Parquet / NetCDF processed data
│   └── metadata.json         # Ingestion state tracker (auto-managed)
├── docs/
│   ├── design_doc.md         # Architecture & design decisions
│   ├── requirement_spec.md   # Functional & non-functional requirements
│   └── research_paper/       # Paper source Markdown + compiled PDF
├── logs/                     # Runtime log output (managed by loguru)
├── runtime/
│   ├── logs/                 # Alternative log location (config-driven)
│   ├── models/
│   │   ├── checkpoints/      # Per-model checkpoint directories
│   │   └── saved/            # Final saved model weights
│   └── paper_artifacts/      # Generated figures and metrics tables
├── src/
│   ├── config.py             # Centralized Pydantic-Settings configuration
│   ├── utils.py              # Shared utilities (device selection, ISO codes)
│   ├── data/
│   │   ├── pipeline.py       # DataPipeline orchestrator
│   │   ├── sources/          # Per-source data fetchers
│   │   │   ├── noaa_ersst.py
│   │   │   ├── openaq.py
│   │   │   ├── nasa_gistemp.py
│   │   │   └── climate_trace.py
│   │   ├── processing/       # Feature engineering and graph building
│   │   │   ├── cleaning.py
│   │   │   ├── feature_engineering.py
│   │   │   └── graph_builder.py
│   │   └── storage/
│   │       └── db.py         # Metadata JSON store
│   ├── models/
│   │   ├── el_nino/          # TFT + LSTM for ENSO forecasting
│   │   │   ├── dataset.py    # TFT & LSTM dataset builders
│   │   │   ├── tft_model.py  # TFT model factory
│   │   │   ├── lstm_baseline.py
│   │   │   ├── trainer.py    # Training loops for TFT and LSTM
│   │   │   └── evaluate.py
│   │   ├── pollution/        # ST-GNN for PM2.5 sensor graph
│   │   │   ├── graph_dataset.py
│   │   │   ├── stgnn_model.py
│   │   │   ├── trainer.py
│   │   │   └── evaluate.py
│   │   ├── nwm/              # Neural Weather Model (GraphCast-style)
│   │   │   ├── dataset.py    # NWMDataset (ERA5 xarray)
│   │   │   ├── graph_builder.py  # SphericalGraphBuilder (Fibonacci mesh)
│   │   │   ├── stgnn_model.py    # Encoder-Processor-Decoder (SAGEConv)
│   │   │   └── trainer.py    # NWMTrainer (native PyTorch, no Lightning)
│   │   └── climate_feedback/ # Bayesian hierarchical regression
│   │       ├── bayesian_regression.py
│   │       ├── changepoint.py
│   │       ├── ols_baseline.py
│   │       └── evaluate.py
│   ├── scripts/
│   │   ├── download_data.py          # CLI: download all data sources
│   │   ├── download_nwm_data.py      # CLI: download ERA5 for NWM (separate step)
│   │   ├── train_models.py           # CLI: train any/all models
│   │   ├── generate_eval_artifacts.py  # CLI: generate evaluation figures/tables
│   │   ├── install_rocm_deps.py      # ROCm-specific dependency installer
│   │   └── patch_rocm_elf.py         # Radeon 680M ELF patcher
│   ├── app/              # Streamlit simulation app (make app): main.py, loaders.py, views/
│   └── visualization/
│       ├── charts.py         # Plot generation functions
│       └── themes.py         # Matplotlib/Plotly theme config
├── Makefile                  # All project commands (install, train, paper, clean)
├── pyproject.toml            # Package metadata and runtime dependencies
└── requirements.txt          # Pinned dependency snapshot (reference only)
```

## Key Coding Conventions

### Imports
- All code under `src/` uses **bare module names** (no `src.` prefix):
  ```python
  from config import settings          # ✅ correct
  from src.config import settings      # ❌ wrong
  ```
- Run scripts with `PYTHONPATH=src` to make this work.

### Settings Access
- Always import via the singleton proxy: `from config import settings`
- Settings are immutable post-construction. Use `os.environ` + `get_settings.cache_clear()` to override.

### Timestamps
- Always use timezone-aware timestamps: `datetime.now(timezone.utc)`
- Never use: `datetime.utcnow()` (deprecated, returns naive datetime)

### Logging
- Use `loguru`: `from loguru import logger`
- `logger` is configured by `setup_logging()` in `config.py`.
- Log files are written to `runtime/logs/run.log` with 10MB rotation and 10-day retention.

### Device Selection
- Use `get_device()` from `src/utils.py` to get the correct `torch.device`.
- Respects `MODEL_DEVICE=auto|cpu|cuda|mps` from settings.

### Data File Formats
| Format | Usage |
|:---|:---|
| Parquet | All processed time-series and tabular datasets |
| NetCDF4 | ERA5 raw grids |
| JSON | Metadata store (`datasets/metadata.json`) |
| CSV | Per-epoch training metrics |
| `.pt` | Native PyTorch model checkpoints |
| `.ckpt` | pytorch-lightning TFT checkpoints |

## Model Architecture Quick Reference

| Model | Architecture | Library |
|:---|:---|:---|
| TFT | Temporal Fusion Transformer | `pytorch-forecasting` |
| LSTM | Bidirectional LSTM, native PyTorch | `torch` |
| ST-GNN (Pollution) | A3TGCN (Attention Temporal GCN) | `torch-geometric-temporal` |
| NWM | SAGEConv Encoder-Processor-Decoder + Fibonacci mesh | `torch-geometric` |
| Bayesian | Hierarchical regression with changepoint detection | `pymc` |
