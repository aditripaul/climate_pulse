# ClimatePulse

<div align="center">
  <h3>AI-Powered Global Climate Forecasting and Analysis</h3>
  <p>An academic research project integrating deep learning forecasting for El Niño, global pollution, and downstream weather modeling.</p>

  [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
  [![PyTorch 2.13](https://img.shields.io/badge/PyTorch-2.13.0-ee4c2c.svg)](https://pytorch.org/)
  [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
</div>

---

## Overview

ClimatePulse is an academic deep learning research project exploring the interconnected dynamics of global climate systems. Four model families are integrated in a single unified platform:

| Model | Task | Architecture |
|:---|:---|:---|
| **El Niño Forecaster** | Predict ENSO phase 3–60 months ahead | Temporal Fusion Transformer (TFT) + BiLSTM baseline |
| **Pollution Dispersion** | Spatiotemporal PM2.5 / NO₂ across sensor networks | Attention-based Temporal Graph Conv (A3TGCN) |
| **Climate Feedback** | Detect warming regime changepoints | Bayesian Hierarchical Regression (PyMC) |
| **Neural Weather Model** | Global T2m & precipitation conditioned on ENSO | Spherical Mesh Encoder–Processor–Decoder (ST-GNN) |

The full stack is Python-only, using efficient data formats (Parquet + JSON) and mixed-precision PyTorch pipelines designed to run on consumer hardware (≥15 GB VRAM) as well as cloud environments.

---

## Technical Stack

| Layer | Libraries |
|:---|:---|
| **Deep Learning** | PyTorch 2.13, torch-geometric 2.8, pytorch-forecasting 1.8 |
| **Bayesian Modelling** | PyMC 6, ArviZ 1.2 |
| **Data Engineering** | xarray 2026.7, pandas 3, pyarrow 25, zarr 3, gcsfs |
| **Configuration** | Pydantic-Settings v2, PyYAML |
| **Logging / Metrics** | Loguru, CSV training history |

---

## Data Sources

Data ingestion is constrained to the **1950–present** window for high-fidelity relevance.

| Source | Content | Access |
|:---|:---|:---|
| [NOAA ERSST v5](https://psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html) | Sea surface temperature anomalies & Niño 3.4 Index | OPeNDAP |
| [OpenAQ v3](https://openaq.org/) | Global PM2.5, NO₂, O₃ measurements | REST API (key required) |
| [NASA GISTEMP v4](https://data.giss.nasa.gov/gistemp/) | Long-term global surface temperature anomalies | Direct download |
| [Climate TRACE v7](https://climatetrace.org/) | CO₂, CH₄, N₂O emission inventories by sector | REST API |
| [WeatherBench 2 / ERA5](https://weatherbench2.readthedocs.io/) | 2m temperature & total precipitation (5.625° grid, 6-hourly) | Google Cloud Zarr |

---

## Quick Start

### Prerequisites

- Python **3.12** (set via `.python-version` for pyenv users)
- `make`, `git`
- An OpenAQ v3 API key (free at [openaq.org](https://explore.openaq.org/register))

### Installation

```bash
git clone <repo-url>
cd climate_pulse.git

# Standard CPU / CUDA install
make install

# AMD ROCm (GPU)
make install-rocm

# Intel NPU
make install-npu
```

`make install` creates `.venv/` using `python3.12`, installs `torch 2.13.0`,
`pyg-lib` (the PyG 2.8 accelerator — replaces the deprecated `torch-scatter`/`torch-sparse`),
then runs `pip install -e .` to install the package itself.

### Configuration

Two layers of configuration:

**1. `.env` — private credentials**
```bash
cp .env.example .env
# Fill in DATA_OPENAQ_API_KEY and any overrides
```

**2. `configs/default.yaml` — operational parameters**
```yaml
data:
  dir: "./datasets"
  start_year: 1950
  end_year: 2026
  focus_countries: "IND,USA"

model:
  tft_encoder_length: 120
  lstm_encoder_length: 60
  lstm_hidden_size: 64
  gnn_num_periods: 7
```

All values can also be overridden via environment variables prefixed with `DATA_`, `MODEL_`, `LOG_`, etc.

### Running the Pipeline

```bash
# 1. Ingest core datasets (NOAA, OpenAQ, NASA GISTEMP, Climate TRACE)
make download-data

# 2. Download ERA5 Zarr data for the Neural Weather Model (large download)
python src/scripts/download_nwm_data.py

# 3. Train all models
make train

# Or train individual models:
make train-elnino      # TFT + LSTM El Niño forecasters
make train-pollution   # A3TGCN pollution ST-GNN
make train-nwm         # Spherical Mesh Neural Weather Model
make train-climate     # Bayesian climate feedback model

# 4. Generate paper artifacts (figures, metrics, tables)
make gen-research-paper-artifacts

# 5. Compile research paper to PDF (requires pandoc + xelatex)
make gen-research-paper-pdf
```

---

## Project Structure

```text
climate_pulse/
├── src/                          # All source code (PYTHONPATH=src)
│   ├── config.py                 # Pydantic-Settings configuration
│   ├── utils.py                  # ISO helpers, device detection
│   ├── data/
│   │   ├── pipeline.py           # Orchestrates fetch -> clean -> store
│   │   ├── sources/              # Data connectors (NOAA, OpenAQ, NASA, TRACE)
│   │   ├── processing/           # Feature engineering (Nino 3.4, AQI, ...)
│   │   └── storage/db.py         # JSON metadata + Parquet data store
│   ├── models/
│   │   ├── el_nino/              # TFT (pytorch-forecasting) + BiLSTM
│   │   ├── pollution/            # A3TGCN over OpenAQ sensor graph
│   │   ├── nwm/                  # Neural Weather Model (native PyTorch ST-GNN)
│   │   └── climate_feedback/     # Bayesian hierarchical regression (PyMC)
│   └── scripts/
│       ├── download_data.py
│       ├── download_nwm_data.py
│       ├── train_models.py       # CLI entry point for all training
│       └── generate_paper_artifacts.py
├── configs/
│   ├── default.yaml              # Default runtime configuration
│   └── dataset_conf.yaml         # Dataset-specific metadata
├── datasets/                     # Data root (gitignored)
│   ├── raw/                      # Raw downloads
│   └── processed/                # Cleaned Parquet files
├── runtime/                      # Execution artifacts (gitignored)
│   ├── models/checkpoints/       # Trained model weights
│   ├── paper_artifacts/          # Figures, CSVs for paper
│   └── logs/                     # Training logs
├── docs/
│   ├── research_paper/           # Research paper source
│   ├── requirement_spec.md       # Formal requirements specification
│   └── design_doc.md             # Architecture and design decisions
├── pyproject.toml                # Package metadata and dependencies
├── requirements.txt              # Flat dependency list
├── Makefile                      # Developer workflow commands
└── .env.example                  # Template for environment variables
```

---

## Makefile Reference

```bash
# Setup
make install              # Standard install (torch CPU/CUDA)
make install-rocm         # torch 2.13 with ROCm 7.2 (AMD GPUs)
make install-npu          # torch 2.13 + OpenVINO (Intel NPU)

# Data
make download-data        # Ingest NOAA, OpenAQ, NASA, Climate TRACE

# Training
make train                # Train all four model families
make train-elnino         # TFT + LSTM
make train-pollution      # A3TGCN
make train-nwm            # Spherical Mesh ST-GNN
make train-climate        # Bayesian regression

# Research outputs
make gen-research-paper-artifacts   # Figures, tables, metrics
make gen-research-paper-pdf         # Compile to PDF (needs pandoc + xelatex)

# Cleanup
make clean                # Remove __pycache__, build artifacts
make clean-checkpoints    # Remove trained model weights
make clean-downloaded-datasets  # Remove raw and processed datasets
```

---

## Environment Variables

Key variables (full list in `.env.example`):

| Variable | Default | Description |
|:---|:---|:---|
| `DATA_OPENAQ_API_KEY` | — | **Required.** OpenAQ v3 API key |
| `DATA_DIR` | `./datasets` | Root data directory |
| `DATA_START_YEAR` | `1950` | Historical data start year |
| `DATA_FOCUS_COUNTRIES` | `IND,USA` | ISO-3 country filter for OpenAQ |
| `MODEL_MAX_EPOCHS` | `100` | Training epoch cap |
| `MODEL_BATCH_SIZE` | `32` | Training batch size |
| `MODEL_LEARNING_RATE` | `0.001` | Base learning rate |
| `MODEL_DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `mps` |
| `MODEL_TFT_ENCODER_LENGTH` | `120` | TFT lookback window (months) |
| `MODEL_LSTM_HIDDEN_SIZE` | `64` | LSTM hidden units |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Contributing

1. Fork the repository and create a feature branch.
2. Make your changes and test them.
3. Open a pull request with a clear description of your changes.

---

## License

This project is licensed under the **GNU General Public License v3.0**.
See the [LICENSE](LICENSE) file for the full text.
