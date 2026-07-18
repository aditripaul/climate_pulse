# ClimatePulse

<div align="center">
  <h3>AI-Powered Global Climate Forecasting and Analysis</h3>
  <p>An academic research project integrating deep learning forecasting for El Niño, global pollution, and downstream weather modeling.</p>

  [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
  [![PyTorch 2.12](https://img.shields.io/badge/PyTorch-2.12.1-ee4c2c.svg)](https://pytorch.org/)
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

## Models

ClimatePulse integrates four distinct modeling approaches to capture the complexity of the global climate system:

### 1. El Niño Forecaster (TFT & LSTM)
Predicts the phase and intensity of the El Niño-Southern Oscillation (ENSO) up to 60 months in advance.
- **TFT (Temporal Fusion Transformer):** The primary model, leveraging multi-head self-attention and variable selection networks to capture long-range dependencies and provide probabilistic prediction intervals (e.g., 10th-90th quantiles).
- **LSTM Baseline:** A bidirectional LSTM used as a baseline to benchmark the TFT's skill score and attention mechanisms.

### 2. Pollution Dispersion (A3TGCN)
Models the spatiotemporal dispersion of PM2.5 across global monitoring networks.
- **Architecture:** Uses an Attention-based Spatial-Temporal Graph Convolutional Network (A3TGCN). It constructs a k-nearest-neighbor spatial graph of OpenAQ sensor locations and applies Chebyshev graph convolutions combined with Gated Recurrent Units (GRUs) to capture both spatial diffusion and temporal accumulation of pollutants.

### 3. Neural Weather Model (ST-GNN)
A global, autoregressive weather surrogate model that predicts 2-meter temperature (T2m) and precipitation, conditioned on the ENSO phase.
- **Architecture:** Employs an Encoder-Processor-Decoder Graph Neural Network. It maps the high-resolution ERA5 equirectangular grid (2,048 nodes) to a latent spherical Fibonacci mesh (2,000 nodes) for isotropic message passing, avoiding the spatial distortion at the poles inherent in standard CNN-based weather models.

### 4. Climate Feedback (Bayesian Regression)
Detects structural changes in the global warming trend and quantifies climate sensitivity to greenhouse gas emissions.
- **Architecture:** A Bayesian piecewise hierarchical regression implemented in PyMC. It models NASA GISTEMP temperature anomalies against Climate TRACE emissions data to provide credible intervals for the continuous warming slope and rigorously identify potential regime shifts.

---

## Technical Stack

| Layer | Libraries |
|:---|:---|
| **Deep Learning** | PyTorch 2.12, torch-geometric 2.8, pytorch-forecasting 1.8 |
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

`make install` creates `.venv/` using `python3.12`, installs `torch 2.12.1`,
`pyg-lib` (the PyG 2.8 accelerator — replaces the deprecated `torch-scatter`/`torch-sparse`),
then runs `pip install -e .` to install the package itself.

### Configuration

Two layers of configuration:

**1. `.env` — private credentials**
```bash
cp .env.example .env
# Fill in DATA_OPENAQ_API_KEY and any overrides
```

**2. `configs/runtime_conf.yaml` — operational parameters**
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

Additionally, the configuration file paths themselves can be overridden using the `RUNTIME_CONF` and `DATASET_CONF` environment variables or command-line parameters:
```bash
# Override config paths for Make command
make train RUNTIME_CONF=configs/custom_runtime.yaml DATASET_CONF=configs/custom_dataset.yaml

# Or directly in python
RUNTIME_CONF=configs/custom_runtime.yaml DATASET_CONF=configs/custom_dataset.yaml PYTHONPATH=src .venv/bin/python src/scripts/train_models.py
```

### Running the Pipeline

```bash
# 1. Ingest core datasets (NOAA, OpenAQ, NASA GISTEMP, Climate TRACE)
make download-data

# 2. Download ERA5 Zarr data for the Neural Weather Model (large download)
make download-nwm-data

# 3. Train all models
make train

# Or train individual models:
make train-elnino      # TFT + LSTM El Niño forecasters
make train-pollution   # A3TGCN pollution ST-GNN
make train-nwm         # Spherical Mesh Neural Weather Model
make train-climate     # Bayesian climate feedback model

# 4. Generate paper artifacts (figures, metrics, tables)
make gen-eval-artifacts

# 5. Save the latest run checkpoints and artifacts for future reference
make save-run

# 6. Compile research paper to PDF (requires pandoc + xelatex)
make gen-research-paper-pdf
```

---

## Project Structure

This project follows a modular architecture designed to separate data ingestion, modeling, and execution. The repository is organized into the following primary directories:

- **`src/`**: Contains the core Python source code, structured into submodules for data processing (`src/data/`), model architectures (`src/models/`), and execution scripts (`src/scripts/`).
- **`configs/`**: Stores YAML configuration files that define runtime behavior, hyperparameter values, and dataset parameters.
- **`datasets/`**: The local storage directory for raw downloads and processed Parquet files.
- **`runtime/`**: Directory for artifacts generated during execution, including trained model checkpoints, evaluation figures, saved runs, and training logs.
- **`docs/`**: Contains formal project documentation, requirement specifications, architectural design documents, and the markdown source for the research paper.

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
│       └── generate_eval_artifacts.py
├── configs/
│   ├── runtime_conf.yaml         # Default runtime configuration
│   └── dataset_conf.yaml         # Dataset-specific metadata
├── datasets/                     # Data root (gitignored)
│   ├── raw/                      # Raw downloads
│   └── processed/                # Cleaned Parquet files
├── runtime/                      # Execution artifacts (gitignored)
│   ├── models/checkpoints/       # Trained model weights
│   ├── eval_artifacts/           # Figures, CSVs for paper
│   ├── saved/                    # Snapshot runs saved for future inference
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
make install-rocm         # torch 2.12 with ROCm 6.1 (AMD GPUs)
make install-npu          # torch 2.12 + OpenVINO (Intel NPU)

# Data
make download-data        # Ingest NOAA, OpenAQ, NASA, Climate TRACE

# Training
make train                # Train all four model families
make train-elnino         # TFT + LSTM
make train-pollution      # A3TGCN
make train-nwm            # Spherical Mesh ST-GNN
make train-climate        # Bayesian regression

# Research outputs
make gen-eval-artifacts   # Figures, tables, metrics
make save-run             # Save run checkpoints/artifacts to runtime/saved/
make gen-research-paper-pdf         # Compile to PDF (needs pandoc + xelatex)

# Cleanup
make clean                # Remove __pycache__, build artifacts
make clean-checkpoints    # Remove trained model weights (also cleans runtime/saved/)
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
