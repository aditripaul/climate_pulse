# ClimatePulse — Requirements Specification

---

## 1. Purpose and Scope

This document defines the functional, data, model, and non-functional requirements for the ClimatePulse research platform. ClimatePulse is an academic deep learning system for integrated climate forecasting, covering El Niño/ENSO dynamics, global air-quality dispersion, climate feedback attribution, and downstream weather prediction.

---

## 2. Stakeholders

| Role | Interest |
|:---|:---|
| **ML Researchers** | Reproducible training pipelines, model comparisons, paper artifacts |
| **Climate Scientists** | Accurate ENSO forecasts, pollution dispersion, warming trend analysis |
| **Engineers / Contributors** | Clean codebase, documented APIs, hardware-portable setup |

---

## 3. Functional Requirements

### 3.1 Data Ingestion

| ID | Requirement |
|:---|:---|
| **D-01** | The system shall fetch NOAA ERSST v5 SST anomaly grids via OPeNDAP for the period 1950–present. |
| **D-02** | The system shall fetch OpenAQ v3 PM2.5 (and optionally NO$_2$, O$_3$) measurements for configurable country sets via the REST API. |
| **D-03** | The system shall fetch NASA GISTEMP v4 global surface temperature anomaly records. |
| **D-04** | The system shall fetch Climate TRACE v7 emission inventories (CO$_2$, CH$_4$, N$_2$O). |
| **D-05** | The system shall download ERA5 6-hourly T2m and total precipitation at 5.625° resolution from Google Cloud Zarr (WeatherBench 2). |
| **D-06** | All raw data shall be stored in `datasets/raw/` and processed artefacts in `datasets/processed/` as Parquet files, with metadata tracked in a JSON store (`db.py`). |
| **D-07** | OpenAQ locations discovered during ingestion shall be persisted to `datasets/processed/openaq/locations.parquet` for reuse by the ST-GNN graph builder. |
| **D-08** | Data ingestion shall be idempotent — re-running shall update rather than duplicate records. |
| **D-09** | The `DATA_START_YEAR` lower bound shall be configurable down to 1854 (ERSST dataset origin); `DATA_END_YEAR` shall be at most 2030. |

### 3.2 Feature Engineering

| ID | Requirement |
|:---|:---|
| **F-01** | The system shall derive the monthly Niño 3.4 index (area-averaged SST anomaly, 5°S–5°N, 170°W–120°W) from ERSST data. |
| **F-02** | The system shall compute AQI from raw PM2.5 measurements using the US EPA breakpoint formula. |
| **F-03** | The system shall compute ISO-3 to ISO-2 country code conversions for OpenAQ location lookups. |
| **F-04** | The NWM dataset shall apply z-score normalisation using statistics computed from the **training split only**, then apply the same parameters to validation and test splits. |

### 3.3 El Niño Model

| ID | Requirement |
|:---|:---|
| **E-01** | The system shall train a Temporal Fusion Transformer (TFT) on Niño 3.4 index time-series with a configurable encoder length (default 60 months) and prediction horizon (default 12 months). |
| **E-02** | The system shall train a bidirectional LSTM baseline for direct comparison with the TFT. |
| **E-03** | Both models shall forecast 1–12 months ahead (configurable via `tft_prediction_months` / `lstm_prediction_months`) on one shared chronological 70/15/15 split in which no training target falls in the validation or test period and no validation target falls in the test period. |
| **E-04** | TFT training shall use a PyTorch-Lightning `Trainer` (via `pytorch-forecasting`) with early stopping (patience=10) and best-checkpoint saving. |
| **E-05** | LSTM training shall use a native PyTorch loop with `ReduceLROnPlateau`, gradient clipping (max norm 1.0), and early stopping (patience=15). |
| **E-06** | Training metrics (epoch, train_loss, val_loss) shall be written to `metrics.csv` in the checkpoint directory. |
| **E-07** | Skill scores shall be computed against a persistence forecast (last observed value held across all leads). |
| **E-08** | Reported El Niño results shall include the mean and standard deviation over multiple training seeds (`make train-elnino-seeds`); model variants shall be selected on validation loss, never on test results. |

### 3.4 Pollution ST-GNN Model

| ID | Requirement |
|:---|:---|
| **P-01** | The system shall construct a spatial graph over OpenAQ sensor locations using k-nearest-neighbour connectivity. |
| **P-02** | The system shall train an A3TGCN (Attention Temporal Graph Convolutional Network) on the PM2.5 sensor graph. |
| **P-03** | The model shall support a sliding-window input size configurable via `MODEL_GNN_NUM_PERIODS` (default 7). |
| **P-04** | The pollution model shall produce per-sensor, per-timestep PM2.5 forecasts. |
| **P-05** | Missing sensor readings shall be excluded from normalisation statistics, the training loss and evaluation; inputs may impute them with the training mean. |
| **P-06** | Normalisation statistics shall be computed on the training split only. |
| **P-07** | Exceedance metrics shall use a threshold in physical units (default PM2.5 ≥ 35.5 µg/m³, AQI > 100) and be reported alongside persistence and always-alert baselines. |

### 3.5 Neural Weather Model (NWM)

| ID | Requirement |
|:---|:---|
| **N-01** | The system shall map ERA5 grid nodes to a latent spherical mesh using a Fibonacci lattice (default 2,000 mesh nodes). |
| **N-02** | Grid-to-mesh, mesh-to-mesh, and mesh-to-grid connectivity shall be derived via KD-tree nearest-neighbour search on 3D Cartesian coordinates. |
| **N-03** | The model architecture shall follow an Encoder–Processor–Decoder pattern using `SAGEConv` layers with `LayerNorm` and GELU activations. |
| **N-04** | The model shall accept the monthly Niño 3.4 index as a scalar conditioning signal injected at each processor layer. The source shall be configurable (`nwm.conditioning`): the observed index, TFT lead-1 forecasts, or none. Each sample shall use the value for the month containing its timestamp. |
| **N-05** | Batch edge-index expansion shall be computed via vectorised tensor operations (no Python loops over batch dimension). |
| **N-06** | The NWM shall be trained using `NWMTrainer`, a native PyTorch class with no `pytorch-lightning` import. |
| **N-07** | The training loop shall use `AdamW` with `ReduceLROnPlateau`, gradient clipping (default max norm 1.0), and configurable early stopping patience (default 10). |
| **N-08** | The dataset split shall be 70% train / 15% validation / 15% test over the temporal axis. |
| **N-09** | An ablation (`make nwm-ablation`) shall compare the main model with forecast-conditioned, unconditioned and unweighted-loss variants on area-weighted test RMSE in physical units. |

### 3.6 Climate Feedback

| ID | Requirement |
|:---|:---|
| **B-01** | The system shall detect changes in the global warming rate by fitting continuous piecewise-linear trends with AR(1) errors to annual NASA GISTEMP means, selecting the number of change-points (0–3) by BIC. |
| **B-01a** | The system shall fit a Bayesian linear regression (PyMC) of temperature anomaly on Climate TRACE emissions. *Status: implemented as a single-level model; the overlapping record (about 6 years) is too short for informative estimates.* |
| **B-02** | The model fitting results shall include posterior trace, summary statistics, and posterior predictive samples. |
| **B-03** | Normalisation parameters (`X_mean`, `X_std`, `predictor_cols`) shall be returned alongside the model object rather than monkey-patched onto `pm.Model`. |

### 3.7 Configuration System

| ID | Requirement |
|:---|:---|
| **C-01** | All configuration shall be managed through Pydantic-Settings `BaseSettings` classes loaded from YAML, with environment variables (and `.env`) taking precedence over YAML values. |
| **C-02** | `PROJECT_ROOT` shall resolve to the repository root (two parents above `src/config.py`). |
| **C-03** | `ModelSettings` shall expose fields for all model hyperparameters (`tft_*`, `lstm_*`, `gnn_*`, `seed`); `NWMModelSettings` shall expose the NWM hyperparameters, conditioning source and loss weighting. |
| **C-04** | Settings shall be immutable at runtime; epoch overrides shall be applied via `os.environ` + `get_settings.cache_clear()`. |

### 3.8 Research Artifact Generation

| ID | Requirement |
|:---|:---|
| **R-01** | The system shall generate reproducible evaluation metrics, comparison plots, and LaTeX-ready tables for all four model families. |
| **R-01a** | The generated evaluation plots shall include: SST anomaly forecasts, CRPS vs lead time, calibration diagrams, and error heatmaps for the El Niño model; RMSE spatial distributions and exceedance confusion matrices for the Pollution model; T2m prediction maps, error maps, and mesh visualizations for the NWM; and warming-rate change-point plots. All plotted and tabulated values shall be computed from model outputs; missing results shall be shown as missing, never replaced by synthetic or default values. |
| **R-01b** | The generated dataset artifacts shall include processed training inputs (`ersst_v5.parquet`, `nino_features.parquet`, `gistemp_features.parquet`, etc.) and output metric summaries (`tft_metrics.csv`, `pollution_metrics.csv`, `nwm_metrics.csv`, `summary_metrics.json`). |
| **R-02** | The `gen-research-paper-pdf` Makefile target shall compile `docs/research_paper/research_paper.md` to PDF using Pandoc + XeLaTeX. |
| **R-02b** | The `gen-patent-application-pdf` Makefile target shall compile `docs/patent/patent_application.md` to PDF using Pandoc + XeLaTeX. |
| **R-03** | The `save-run` Makefile target shall copy and snapshot the latest model checkpoints and evaluation artifacts to a timestamped folder under `runtime/saved/` for future reference and inference. |
| **R-04** | The system shall maintain a consolidated single patent application draft (`docs/patent/patent_application.md`) integrating the Neural Weather Model, Fibonacci spherical mesh, ENSO message conditioning, and multi-modal climate framework for IPINDIA / PCT legal review. |
| **R-05** | The system shall let every reported result be rebuilt from scratch with a documented command sequence (README, "Reproducing the Results") and with a Google Colab notebook (`run.ipynb`) that runs the same sequence, persists data and results to Google Drive, and has a quick mode for checking the workflow. Seeds shall be fixed; text in the paper and patent that quotes results is maintained by hand. |

### 3.9 Interactive Simulation App

| ID | Requirement |
|:---|:---|
| **R-06** | The system shall provide a Streamlit application (`make app`, entry point `src/app/main.py`) that runs the trained El Niño, pollution and weather models and the change-point analysis on held-out data. It shall not train or modify any model. |
| **R-06a** | Navigation shall be a top menu with an overview and one page per simulation; each page shall have a direct URL. |
| **R-06b** | The weather page shall show truth, prediction and error for a chosen test time, as maps and as a 3D globe, for temperature and precipitation, for each available model version, with an optional Niño 3.4 override. Rolling the model forward beyond one 6-hour step shall be available only with an on-screen statement that it is not validated. |
| **R-06c** | The El Niño page shall show TFT median and 10–90 % interval, LSTM and persistence for a chosen test window, an animation across the test period, and error by lead time. |
| **R-06d** | The pollution page shall show station predictions coloured by US EPA PM2.5 band against observations where they exist, and shall state that the test record is too small to show exceedance skill. |
| **R-06e** | The warming page shall let the user change the maximum number of breaks and the shortest segment and shall show the BIC for each number of breaks. |
| **R-07** | Checkpoints shall be resolved from the live checkpoint directory and then from the newest saved run. A page whose checkpoint or data file is missing shall show which command creates it instead of failing. |
| **R-08** | Model and data access shall be separate from rendering (`src/app/loaders.py`) and covered by tests that run without Streamlit. |
| **R-09** | The app shall use the repository theme (white, grey, blue accent) configured in `.streamlit/config.toml` and the Plotly styling in `src/app/style.py`. |

---

## 4. Non-Functional Requirements

### 4.1 Performance

| ID | Requirement |
|:---|:---|
| **NF-01** | All models shall support mixed-precision training (FP16/BF16) where hardware allows. *Status: not yet implemented.* |
| **NF-02** | The system shall run end-to-end (data ingestion + all model training) on a single workstation; a GPU is optional. *Reference: about 2.2 hours CPU-only for the main models.* |
| **NF-03** | NWM batch edge expansion shall be O(E) in tensor operations, not O(batch × E) via Python loops. |

### 4.2 Hardware Compatibility

| ID | Requirement |
|:---|:---|
| **NF-04** | The system shall support CUDA (NVIDIA), ROCm 7.2 (AMD), and Intel NPU via OpenVINO. |
| **NF-05** | Device selection shall auto-detect available hardware when `MODEL_DEVICE=auto`. |

### 4.3 Reliability

| ID | Requirement |
|:---|:---|
| **NF-06** | All file I/O failures shall raise descriptive exceptions; all API calls shall log warnings and return empty DataFrames on non-fatal errors. |
| **NF-07** | `datetime.utcnow()` shall not be used; all timestamps shall be timezone-aware (`timezone.utc`). |
| **NF-08** | xarray datasets opened in `NWMDataset` shall be closed on object destruction. |

### 4.4 Security

| ID | Requirement |
|:---|:---|
| **NF-09** | API keys shall never be committed to the repository; all secrets shall be loaded from `.env` (gitignored). |
| **NF-10** | `pytorch-lightning` shall not be explicitly pinned or installed; it enters the environment only as a transitive dependency of `pytorch-forecasting` to avoid the 2.6.2/2.6.3 supply-chain compromise. |

### 4.5 Reproducibility

| ID | Requirement |
|:---|:---|
| **NF-11** | NWM normalisation statistics shall be computed from the training split only and applied to validation and test. |
| **NF-12** | All training runs shall log a unique `run_id`, configuration string, timestamps, and best metric to the JSON metadata store. |

### 4.6 Maintainability

| ID | Requirement |
|:---|:---|
| **NF-13** | All imports shall use bare module names consistent with `PYTHONPATH=src` (no `src.` prefix anywhere in `src/`). |
| **NF-15** | Python version shall be pinned to **3.12** across `.python-version`, `Makefile`, and `pyproject.toml`. |

---

## 5. External Interfaces

### 5.1 Data APIs

| API | Auth | Rate Limits |
|:---|:---|:---|
| NOAA OPeNDAP | None | None (government open data) |
| OpenAQ v3 REST | API key (`DATA_OPENAQ_API_KEY`) | Tier-dependent (free tier: 60 req/min) |
| NASA GISTEMP | None | None |
| Climate TRACE | None | None |
| Google Cloud Zarr (ERA5) | None (public GCS bucket) | None |

### 5.2 File Formats

| Format | Usage |
|:---|:---|
| **Parquet** | All processed time-series and tabular data |
| **JSON** | Metadata store (source registration, model run logs) |
| **NetCDF4** | ERA5 downloaded grids |
| **CSV** | Per-epoch training metrics history |
| **`.pt`** | Native PyTorch model checkpoints |
| **`.ckpt`** | pytorch-lightning TFT checkpoints |

---

## 6. Constraints

- Data ingestion is constrained to **1950–2030**.
- OpenAQ coverage begins in **2015**.
- The NWM model requires ERA5 data to be downloaded separately (`download_nwm_data.py`) before training.
- The TFT trainer requires `pytorch-forecasting >= 1.8.0` which pulls in `pytorch-lightning` as a transitive dependency.
