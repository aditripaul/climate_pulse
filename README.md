# ClimatePulse

<div align="center">
  <h3>AI-Powered Global Climate Forecasting and Analysis</h3>
  <p>An academic research project integrating deep learning forecasting for El Niño, regional air pollution, and global weather fields, plus statistical analysis of the warming trend.</p>

  [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
  [![PyTorch 2.12](https://img.shields.io/badge/PyTorch-2.12.1-ee4c2c.svg)](https://pytorch.org/)
  [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/archanpaul/climate_pulse/blob/main/run.ipynb)
</div>

---

## Overview

ClimatePulse is an academic deep learning research project exploring the interconnected dynamics of global climate systems. Four model families share one data pipeline, configuration system and evaluation tooling:

| Model | Task | Architecture |
|:---|:---|:---|
| **El Niño Forecaster** | Forecast the Niño 3.4 index 1–12 months ahead | Temporal Fusion Transformer (TFT) + BiLSTM and persistence baselines |
| **Pollution Dispersion** | Next-day PM2.5 across a sensor network | Attention Temporal Graph Convolutional Network (A3TGCN) |
| **Neural Weather Model** | Next 6-hour T2m & precipitation, conditioned on Niño 3.4 | Spherical-mesh Encoder–Processor–Decoder GNN |
| **Climate Feedback** | Detect changes in the global warming rate | Piecewise-linear trend with AR(1) errors, change-points selected by BIC |

The stack is Python-only, stores data as Parquet + JSON, and runs on CPU; CUDA, AMD ROCm and Intel NPU installs are also supported. A full CPU-only training run takes about 2.2 hours, almost all of it the Neural Weather Model. An interactive [Streamlit app](#interactive-simulation-app) (`make app`) runs the trained models on held-out data.

---

## Models

### 1. El Niño Forecaster (TFT & LSTM)
Forecasts the monthly Niño 3.4 sea-surface temperature anomaly 1–12 months ahead from a 60-month history.
- **TFT (Temporal Fusion Transformer):** multi-head self-attention and variable selection networks over Niño 1+2, 3, 3.4 and 4 indices, rolling statistics and seasonal features. Quantile regression (10th/50th/90th percentiles) gives prediction intervals.
- **LSTM baseline:** a two-layer bidirectional LSTM on the Niño 3.4 series alone.
- **Persistence baseline:** the last observed anomaly held constant; skill scores are measured against it.

Data are split chronologically 70/15/15 (train Jan 1950–Aug 2003, validation to Feb 2015, test Mar 2015–Sep 2026). No training target falls in the validation or test periods.

### 2. Pollution Dispersion (A3TGCN)
Predicts next-day PM2.5 at each station of an OpenAQ monitoring network (75 stations in India and the United States in the reference run).
- **Architecture:** a 10-nearest-neighbour station graph with inverse-distance edge weights; A3TGCN combines graph convolutions (GCN) with GRU cells and attention over a 7-day input window.

### 3. Neural Weather Model (ST-GNN)
A global weather surrogate predicting the next 6-hourly 2 m temperature and precipitation fields from the previous two, conditioned on the Niño 3.4 index.
- **Architecture:** an Encoder–Processor–Decoder GNN mapping the 5.625° ERA5 grid (2,048 nodes) to a 2,000-node Fibonacci spherical mesh, four GraphSAGE processor layers with Niño 3.4 injected into each layer's messages, and a decoder back to the grid. Training uses a latitude-cosine-weighted MSE loss.
- **Conditioning input:** the *observed* monthly Niño 3.4 value for each sample time. TFT forecasts are not yet passed to this model.

### 4. Climate Feedback
- **Warming-rate change-points (used in evaluation):** a continuous piecewise-linear trend fitted to annual NASA GISTEMP means (1950–2025) with AR(1) errors. Every placement of 0–3 change-points (segments of at least 10 years) is fitted, and the number of change-points is chosen by BIC.
- **Emissions regression (`make train-climate`):** a Bayesian linear regression (PyMC) of global temperature anomaly on Climate TRACE CO₂e emissions. Only about 6 years of data overlap, so it is not yet informative and is not used in evaluation.

---

## Current Results

Results from the corrected evaluation code (no train/test leakage, no synthetic values). Full tables are written to `runtime/eval_artifacts/metrics_summary.md` by `make gen-eval-artifacts`; the [research paper](docs/research_paper/research_paper.md) discusses them. Single-run results are noted as such.

| Model | Result | Caveats |
|:---|:---|:---|
| El Niño TFT (no year covariate) | Test 2015–2026, leads 1–12, mean ± s.d. over 5 seeds: RMSE 0.78 ± 0.08 °C, ACC 0.65 ± 0.02, skill 0.43 ± 0.11 vs. persistence | LSTM is comparable (RMSE 0.81 ± 0.02 °C, skill 0.39 ± 0.03) and far more stable across seeds; both are biased about −0.3 °C because the test period is warmer than training; the TFT 80 % interval covers 63 ± 11 % of observations |
| Pollution A3TGCN | RMSE 46.6 µg/m³ vs. 52.0 persistence; exceedance F1 0.83 at PM2.5 ≥ 35.5 µg/m³ (AQI > 100) | Only 85 of 1,500 test station-days have readings (36 stations); the model alerts on every one, so F1 equals an always-alert rule and persistence scores higher |
| Neural Weather Model | 6-hour 2 m temperature, area-weighted test RMSE 1.21 K vs. 2.46 K persistence | Ablation (one run each): unconditioned 1.20 K, TFT-forecast-conditioned 1.33 K, unweighted loss 1.16 K, so neither the Niño 3.4 conditioning nor the area-weighted loss helps; trained on 2020–2021 only, a single ENSO phase |
| Change-points | One break in 1976: 0.03 → 0.21 °C/decade | Break year is loosely constrained (1970–1986); a second break in 2012 is nearly as well supported |

---

## Technical Stack

| Layer | Libraries |
|:---|:---|
| **Deep Learning** | PyTorch 2.12, torch-geometric 2.8, torch-geometric-temporal, pytorch-forecasting 1.8 |
| **Statistical Modelling** | NumPy (change-point search), PyMC 6 + ArviZ 1.2 (emissions regression) |
| **Data Engineering** | xarray 2026.7, pandas 3, pyarrow 25, zarr 3, gcsfs |
| **Configuration** | Pydantic-Settings v2, PyYAML |
| **Interactive app** | Streamlit 1.66, Plotly |
| **Logging / Metrics** | Loguru, CSV training history, JSON run metadata |

---

## Data Sources

Data ingestion covers **1950–present** unless a source starts later.

| Source | Content used | Access |
|:---|:---|:---|
| [NOAA ERSST v5](https://psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html) | Monthly SST; Niño 1+2, 3, 3.4 and 4 indices | OPeNDAP |
| [OpenAQ v3](https://openaq.org/) | PM2.5 measurements for the focus countries (default India, USA) | REST API (key required) |
| [NASA GISTEMP v4](https://data.giss.nasa.gov/gistemp/) | Monthly global land–ocean temperature anomalies | Direct download |
| [Climate TRACE v7](https://climatetrace.org/) | CO₂e (100-year GWP) emissions for the focus countries | REST API |
| [WeatherBench 2 / ERA5](https://weatherbench2.readthedocs.io/) | 2 m temperature & total precipitation, 5.625° grid, 6-hourly, 2020–2021 | Google Cloud Zarr |

---

## Quick Start

### Prerequisites

- Python **3.12** (set via `.python-version` for pyenv users)
- `make`, `git`
- An OpenAQ v3 API key (free at [openaq.org](https://explore.openaq.org/register))
- For PDFs: `pandoc` and `xelatex`

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
`pyg-lib` (the PyG 2.8 accelerator, replacing the deprecated `torch-scatter`/`torch-sparse`),
then runs `pip install -e .` to install the package itself.

### Configuration

Two layers of configuration:

**1. `.env` — private credentials**
```bash
cp -n .env.example .env   # -n: never overwrite an existing .env
# Fill in DATA_OPENAQ_API_KEY
```

**2. `configs/runtime_conf.yaml` — operational parameters**
```yaml
data:
  dir: "./datasets"
  start_year: 1950
  end_year: 2026
  focus_countries: "IND,USA"

model:
  max_epochs: 100
  seed: 42
  tft_encoder_length: 60
  tft_use_year: false     # set from the seed study (elnino_seed_selection.json)
  lstm_encoder_length: 60
  lstm_hidden_size: 64
  gnn_num_periods: 7

nwm:
  conditioning: "observed"   # observed | forecast | none
  area_weighted_loss: true
```

Environment variables prefixed with `DATA_`, `MODEL_`, `NWM_`, `LOG_` or `META_` override the YAML values, for example `MODEL_MAX_EPOCHS=5 make train-elnino`. `train_models.py` also accepts `--epochs` and `--seed`.

The configuration file paths themselves can be overridden with the `RUNTIME_CONF` and `DATASET_CONF` environment variables or Make parameters:
```bash
# Override config paths for Make command
make train RUNTIME_CONF=configs/custom_runtime.yaml DATASET_CONF=configs/custom_dataset.yaml

# Or directly in python
RUNTIME_CONF=configs/custom_runtime.yaml DATASET_CONF=configs/custom_dataset.yaml PYTHONPATH=src .venv/bin/python src/scripts/train_models.py
```

### Running the Pipeline

```bash
# 1. Ingest core datasets (NOAA, OpenAQ, NASA GISTEMP, Climate TRACE); cached files are reused
make download-data

# 2. Download ERA5 data for the Neural Weather Model (large; skipped if present)
make download-nwm-data

# 3. Train all models
make train

# Or train individual models:
make train-elnino      # TFT + LSTM El Niño forecasters
make train-pollution   # A3TGCN pollution ST-GNN
make train-nwm         # Spherical-mesh Neural Weather Model
make train-climate     # Bayesian emissions regression

# 4. Robustness studies: multi-seed El Niño, then the NWM ablation
#    (forecast-conditioned, unconditioned and unweighted-loss variants)
make train-elnino-seeds
make export-tft-forecasts
make nwm-ablation

# 5. Evaluate all models; writes figures, metrics and tables to runtime/eval_artifacts/
make gen-eval-artifacts

# 6. Save the run's checkpoints and artifacts to runtime/saved/run_<timestamp>/
make save-run

# 7. Point the paper's artifacts at the newest saved run
make sync-paper-artifacts

# 8. Compile the research paper and patent application PDFs
make gen-research-paper-pdf
make gen-patent-application-pdf
```

Retraining replaces the previous run's checkpoints in `runtime/models/checkpoints/`; run `make save-run` first to keep them.

`make train` trains the four model families in sequence but not the seed study, the forecast export or the ablation; use the sequence below to rebuild everything the paper reports.

---

## Interactive Simulation App

A Streamlit app that runs the **already-trained** models on held-out data and shows the results interactively. Nothing is trained in the app.

```bash
make install        # includes streamlit and plotly
make app            # opens http://localhost:8501
```

Five tabs switch between the pages (only the open tab runs):

| Page | What you can do | Status |
|:---|:---|:---|
| **Overview** | See the reported results and which checkpoints were found | n/a |
| **Weather** | Pick a moment in the test period; compare the model with ERA5 as maps and a rotatable 3D globe; switch between the main model and the three ablation variants; override the Niño 3.4 input; roll the model forward and animate it | Validated for one 6-hour step; roll-forward is a demonstration |
| **El Niño** | Pick a test window and see the TFT median and 10–90 % band against the LSTM, persistence and what happened; animate through 2015–2026; see error by lead time | Validated on held-out years |
| **Air pollution** | See next-day PM2.5 at the stations coloured by EPA band; highlight a station; roll forward a few snapshots | Demonstration only (85 observed test station-days) |
| **Warming rate** | Change the number of breaks and the shortest segment and watch the BIC choice; compare with a straight line | Statistical model, no network |

**What it needs**

- Checkpoints are read from `runtime/models/checkpoints/` first and then from the newest committed snapshot in `runtime/saved/run_*/checkpoints/`, so a fresh clone can run the El Niño, weather and pollution models without retraining.
- Data files are not in the repository. The weather page needs the ERA5 file (`make download-nwm-data`) and `nino_features.parquet` (`make download-data`); the El Niño page needs `nino_features.parquet`; the pollution page needs the cached OpenAQ file; the warming page needs `gistemp_features.parquet`. A page whose files are missing shows what to run instead of failing.
- The pollution station graph is rebuilt offline from the cached OpenAQ file and reproduces the reported test figures (85 observed station-days, RMSE 46.6 µg/m³ against 52.0 for persistence).

**Caveats the app states on screen**

- The weather model was trained and tested only for a single 6-hour step. Rolling it forward feeds it its own output and is not validated: errors grow and it can end up worse than persistence.
- Weather predictions are computed one sample at a time. The model's graph `LayerNorm` pools statistics across a batch, so scores differ slightly with batch size (about 1.217 K one at a time against the reported 1.214 K in batches of 16).

The code is in `src/app/` (`loaders.py` has the model and data access and is tested without Streamlit; `views/` has one module per page; `main.py` is the entry point). The theme is set in `.streamlit/config.toml`.

---

## Reproducing the Results

Everything the paper and patent report can be rebuilt from scratch. A full run takes about 7–8 hours on CPU (about 2 h for the main weather model and 5 h for its comparison versions) and much less on a GPU.

```bash
# 0. Environment and credentials
make install-dev                    # venv, torch, dependencies, pytest
cp -n .env.example .env             # -n keeps an existing .env; set DATA_OPENAQ_API_KEY

# 1. Start clean (the last command also deletes downloaded data; skip it to reuse the cache)
make clean-checkpoints clean-eval-artifacts
make clean-downloaded-datasets

# 2. Data, and a check that the code works
make download-data
make download-nwm-data
make test

# 3. El Niño: five seeds, then the main run
make train-elnino-seeds             # writes runtime/eval_artifacts/run_results/elnino_seeds.csv
#    and elnino_seed_selection.json. Set model.tft_use_year in configs/runtime_conf.yaml to
#    the "tft_use_year" value in that JSON file (the TFT variant with the lower mean validation loss).
make train-elnino

# 4. Other models and the weather-model comparison
make train-pollution
make train-climate                  # optional; not used in any reported result
make train-nwm                      # main weather model
make export-tft-forecasts           # needs the main TFT from step 3
make nwm-ablation                   # trains and scores the 3 comparison versions

# 5. Artifacts, snapshot and documents
make gen-eval-artifacts
make save-run
make sync-paper-artifacts
make gen-research-paper-pdf
make gen-patent-application-pdf
```

Order matters: `nwm-ablation` scores the main weather model, so `train-nwm` comes first; `export-tft-forecasts` needs the main TFT; `clean-eval-artifacts` also removes the seed results, so run it only at the start.

**What is and is not reproduced exactly**
- Seeds are fixed (`model.seed`), but CPU/GPU type and thread counts can change floating-point results slightly, so expect numbers close to, not identical to, the paper's.
- A fresh download uses live APIs (OpenAQ is capped at 10,000 records; NOAA and GISTEMP update monthly), so the data can differ from the data behind the paper. To freeze it, archive `datasets/` and skip the data-clean step.
- The numbers written into `research_paper.md`, `patent_application.md` and the README results table are typed in by hand; the scripts update figures, tables, PDFs and `metrics_summary.md`, not the text.

### Google Colab

[`run.ipynb`](run.ipynb) runs the same sequence on Colab ([open it](https://colab.research.google.com/github/archanpaul/climate_pulse/blob/main/run.ipynb)):
1. Choose a GPU runtime and add the secret `OPENAQ_API_KEY` (and `GITHUB_TOKEN` if the repository is private).
2. Keep `QUICK = True` for a first pass (every step runs 2 epochs on one seed; results are not meaningful), then set `QUICK = False` for the full run.
3. With `USE_DRIVE = True`, datasets and results are saved to Google Drive and restored in the next session, so an interrupted run can resume after the data step.

The notebook applies the seed-study choice of `tft_use_year` automatically. PDF builds are optional (`BUILD_PDFS`) because installing XeLaTeX takes a few minutes.

---

## Documents

| Document | Source | Build |
|:---|:---|:---|
| Research paper | [`docs/research_paper/research_paper.md`](docs/research_paper/research_paper.md) | `make gen-research-paper-pdf` |
| Patent application (draft) | [`docs/patent/patent_application.md`](docs/patent/patent_application.md) | `make gen-patent-application-pdf` |
| Project proposal | [`docs/project_proposal.md`](docs/project_proposal.md) | `make project_proposal_pdf` |
| Project report | [`docs/project_report.md`](docs/project_report.md) | `make gen-project-report-pdf` |
| Requirements specification | [`docs/requirement_spec.md`](docs/requirement_spec.md) | — |
| Design document | [`docs/design_doc.md`](docs/design_doc.md) | — |

`docs/research_paper/artifacts` is a symlink to `runtime/saved/run_save.latest`, which points at a saved run, so the paper always reads a fixed snapshot. Do not copy files into it; use `make save-run` followed by `make sync-paper-artifacts`. The patent has its own copy of the artifacts in `docs/patent/artifacts/`, refreshed by `make gen-patent-application-pdf`.

---

## Project Structure

- **`src/`**: Python source code: data ingestion and processing (`src/data/`), models (`src/models/`), command-line scripts (`src/scripts/`) and the Streamlit app (`src/app/`).
- **`configs/`**: YAML configuration for runtime behaviour, hyperparameters and data sources.
- **`datasets/`**: Raw downloads and processed Parquet files (gitignored).
- **`runtime/`**: Checkpoints, evaluation artifacts, saved runs and logs.
- **`docs/`**: Research paper, patent application, project proposal, requirements and design documents.
- **`.agents/`**: Workspace rules and skills for AI coding agents.

```text
climate_pulse/
|-- src/                          # All source code (PYTHONPATH=src)
|   |-- config.py                 # Pydantic-Settings configuration
|   |-- utils.py                  # ISO helpers, device detection
|   |-- data/
|   |   |-- pipeline.py           # Orchestrates fetch -> clean -> store
|   |   |-- sources/              # Data connectors (NOAA, OpenAQ, NASA, TRACE)
|   |   |-- processing/           # Cleaning, feature engineering, station graphs
|   |   +-- storage/db.py         # JSON metadata + Parquet data store
|   |-- models/
|   |   |-- el_nino/              # TFT (pytorch-forecasting) + BiLSTM
|   |   |-- pollution/            # A3TGCN over OpenAQ sensor graph
|   |   |-- nwm/                  # Neural Weather Model (native PyTorch ST-GNN)
|   |   +-- climate_feedback/     # Change-point detection + Bayesian regression
|   |-- visualization/            # Chart helpers and themes
|   |-- app/                      # Streamlit simulation app (make app)
|   +-- scripts/
|       |-- download_data.py
|       |-- download_nwm_data.py
|       |-- train_models.py       # CLI entry point for all training
|       +-- generate_eval_artifacts.py
|-- configs/
|   |-- runtime_conf.yaml         # Default runtime configuration
|   +-- dataset_conf.yaml         # Data source endpoints and regions
|-- datasets/                     # Data root (gitignored)
|   |-- raw/                      # Raw downloads
|   +-- processed/                # Cleaned Parquet files
|-- runtime/
|   |-- models/checkpoints/       # Latest trained weights (gitignored)
|   |-- eval_artifacts/           # Latest figures, CSVs and metrics
|   |-- saved/                    # Saved runs (tracked in git); run_save.latest -> current paper run
|   +-- logs/                     # Training logs (gitignored)
|-- docs/
|   |-- research_paper/           # Paper source, PDF; artifacts -> runtime/saved/run_save.latest
|   |-- patent/                   # Patent application draft, PDF and artifacts
|   |-- project_proposal.md       # Project proposal
|   |-- requirement_spec.md       # Formal requirements specification
|   +-- design_doc.md             # Architecture and design decisions
|-- .streamlit/config.toml        # Streamlit theme (white, grey, blue)
|-- tests/                        # pytest suite (make test)
|-- run.ipynb                     # Google Colab notebook running the full pipeline
|-- .agents/                      # AI agent workspace rules and skills
|-- pyproject.toml                # Package metadata and dependencies
|-- requirements.txt              # Flat dependency list
|-- Makefile                      # Developer workflow commands
+-- .env.example                  # Template for environment variables
```

---

## Makefile Reference

```bash
# Setup
make install                    # Standard install (torch 2.12.1, CPU/CUDA)
make install-dev                # Install plus test dependencies (pytest)
make test                       # Run the test suite
make app                        # Launch the Streamlit simulation app (needs trained checkpoints)
make install-rocm               # torch 2.6.0 with ROCm 6.1 (AMD GPUs)
make install-npu                # torch 2.12.1 + OpenVINO (Intel NPU)

# Data
make download-data              # Ingest NOAA, OpenAQ, NASA, Climate TRACE
make download-nwm-data          # ERA5 subset for the Neural Weather Model

# Training
make train                      # Train all model families
make train-elnino               # TFT + LSTM
make train-pollution            # A3TGCN
make train-nwm                  # Spherical-mesh ST-GNN
make train-climate              # Bayesian emissions regression

# Robustness studies
make train-elnino-seeds         # TFT (with/without year covariate) + LSTM over 5 seeds
make export-tft-forecasts       # TFT lead-1 forecasts for forecast-conditioned NWM
make nwm-ablation               # NWM: forecast-conditioned, unconditioned, unweighted variants

# Evaluation and documents
make gen-eval-artifacts         # Figures, tables, metrics in runtime/eval_artifacts/
make save-run                   # Save checkpoints/artifacts to runtime/saved/run_<timestamp>/
make sync-paper-artifacts       # Point paper artifacts at the newest saved run
make gen-research-paper-pdf     # Compile paper PDF (needs pandoc + xelatex)
make sync-patent-artifacts      # Copy runtime/eval_artifacts/ into docs/patent/artifacts/
make gen-patent-application-pdf # Sync patent artifacts and compile patent PDF
make project_proposal_pdf       # Compile project proposal PDF
make gen-project-report-pdf         # Compile project report PDF

# Cleanup
make clean                      # Remove __pycache__, build artifacts
make clean-checkpoints          # Remove trained weights (saved runs in runtime/saved/ are kept)
make clean-eval-artifacts       # Remove runtime/eval_artifacts/
make clean-downloaded-datasets  # Remove raw and processed datasets
```

---

## Environment Variables

Credentials go in `.env` (see `.env.example`). Environment variables override `configs/runtime_conf.yaml`; the values below are those the shipped YAML sets.

| Variable | Value | Description |
|:---|:---|:---|
| `DATA_OPENAQ_API_KEY` | — | **Required.** OpenAQ v3 API key |
| `DATA_DIR` | `./datasets` | Root data directory |
| `DATA_START_YEAR` | `1950` | Historical data start year |
| `DATA_FOCUS_COUNTRIES` | `IND,USA` | ISO-3 country filter for OpenAQ and Climate TRACE |
| `MODEL_MAX_EPOCHS` | `100` | Training epoch cap |
| `MODEL_BATCH_SIZE` | `32` | Training batch size |
| `MODEL_LEARNING_RATE` | `0.001` | Base learning rate |
| `MODEL_DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `mps` |
| `MODEL_TFT_ENCODER_LENGTH` | `60` | TFT lookback window (months) |
| `MODEL_LSTM_HIDDEN_SIZE` | `64` | LSTM hidden units |
| `MODEL_SEED` | `42` | Random seed |
| `NWM_CONDITIONING` | `observed` | NWM Niño 3.4 input: `observed`, `forecast` (TFT lead-1) or `none` |
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
