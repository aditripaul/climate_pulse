# ClimatePulse — Architecture and Design Document

**Version:** 1.0  
**Date:** July 2026  
**Status:** Active  

---

## 1. System Overview

ClimatePulse is a modular, Python-only research platform for integrated global climate forecasting. It is structured as four independent model subsystems sharing a common data pipeline, configuration layer, and storage backend.

```
+-----------------------------------------------------------------+
|                         ClimatePulse                            |
|                                                                 |
|  +--------------+  +--------------+  +-----------+  +-------+  |
|  |  El Niño     |  |  Pollution   |  |    NWM    |  | Bayes |  |
|  |  TFT / LSTM  |  |   A3TGCN    |  |  ST-GNN   |  | Regr. |  |
|  +------+-------+  +------+-------+  +-----+-----+  +---+---+  |
|         |                 |                |             |      |
|  +------+-----------------+----------------+-------------+----+  |
|  |               Unified Data Pipeline (pipeline.py)          |  |
|  +------+-----------------------------------------------------+  |
|         |                                                        |
|  +------+------------------------------------------------------+  |
|  |  Data Sources: NOAA ERSST · OpenAQ · NASA GISTEMP ·         |  |
|  |                Climate TRACE · Google Cloud ERA5 Zarr        |  |
|  +-------------------------------------------------------------+  |
+-----------------------------------------------------------------+
```

---

## 2. Repository Layout

```
src/                  # PYTHONPATH root
+-- config.py         # Pydantic-Settings: DataSettings, ModelSettings, LogSettings, …
+-- utils.py          # Device detection, ISO code conversion maps + helpers
+-- data/
|   +-- pipeline.py           # DataPipeline orchestrator
|   +-- sources/              # One class per external data source
|   |   +-- noaa_ersst.py     # NOAAERSSTSource
|   |   +-- openaq.py         # OpenAQSource
|   |   +-- nasa_gistemp.py   # NASAGISTEMPSource
|   |   +-- climate_trace.py  # ClimateTraceSource
|   +-- processing/           # Feature engineering functions
|   +-- storage/
|       +-- db.py             # Database: JSON metadata + Parquet bulk store
+-- models/
|   +-- el_nino/
|   |   +-- tft_model.py      # TemporalFusionTransformer (pytorch-forecasting)
|   |   +-- lstm_baseline.py  # LSTMForecaster (native PyTorch)
|   |   +-- dataset.py        # LSTMDataset, TFT TimeSeriesDataSet builder
|   |   +-- trainer.py        # train_tft(), train_lstm()
|   |   +-- evaluate.py       # Metrics, aligned TFT/LSTM/persistence test predictions
|   +-- pollution/
|   |   +-- stgnn_model.py    # A3TGCN wrapper
|   |   +-- graph_dataset.py  # Sensor graph construction
|   |   +-- trainer.py        # Native PyTorch loop
|   |   +-- evaluate.py
|   +-- nwm/
|   |   +-- stgnn_model.py    # NWMModel (Encoder–Processor–Decoder)
|   |   +-- graph_builder.py  # SphericalGraphBuilder (Fibonacci mesh, KD-tree)
|   |   +-- dataset.py        # NWMDataset (ERA5 NetCDF + Niño conditioning)
|   |   +-- trainer.py        # NWMTrainer (native PyTorch, no lightning)
|   |   +-- evaluate.py       # Area-weighted test RMSE in physical units
|   +-- climate_feedback/
|       +-- changepoint.py          # detect_slope_changepoints() (piecewise trend, AR(1), BIC)
|       +-- bayesian_regression.py  # build_bayesian_model(), fit_bayesian_model()
|       +-- ols_baseline.py         # statsmodels OLS baseline
|       +-- evaluate.py
+-- app/
|   +-- main.py               # Streamlit entry point: top navigation, overview
|   +-- loaders.py            # Checkpoint lookup, model loading, rollouts (no Streamlit)
|   +-- style.py, ui.py       # Plotly look, page chrome and CSS
|   +-- views/                # weather.py, elnino.py, pollution.py, warming.py
+-- scripts/
    +-- download_data.py           # CLI: ingest pipeline
    +-- download_nwm_data.py       # CLI: ERA5 Zarr download
    +-- train_models.py            # CLI: train all / individual models
    +-- generate_eval_artifacts.py
    +-- run_elnino_seeds.py        # multi-seed TFT/LSTM evaluation
    +-- export_tft_forecasts.py    # TFT lead-1 forecasts for NWM conditioning
    +-- run_nwm_ablation.py        # NWM conditioning / loss-weighting ablation
```

---

## 3. Configuration Architecture

### 3.1 Settings Hierarchy

```
get_settings()  →  Settings (singleton, lru_cache)
                    +-- DataSettings      (prefix: DATA_)
                    +-- MetadataSettings  (prefix: META_)
                    +-- ModelSettings     (prefix: MODEL_)
                    +-- NWMModelSettings  (prefix: NWM_)
                    +-- LogSettings       (prefix: LOG_)
```

Each section reads from three sources in increasing priority:
1. Default field values in code
2. `configs/runtime_conf.yaml` (loaded explicitly in `get_settings()` and passed as init values)
3. `.env` file and environment variables

pydantic-settings normally ranks init values above environment variables; `_YamlDefaultSettings.settings_customise_sources()` reorders the sources so environment variables win.

### 3.2 Path Resolution

`PROJECT_ROOT` is resolved as `Path(__file__).resolve().parent.parent` — two parents above `src/config.py` — which gives the repository root regardless of the working directory when the interpreter is launched.

All data and runtime paths are expressed as `PROJECT_ROOT / "datasets" / ...` so they remain stable across invocation contexts.

### 3.3 Settings Mutation

Pydantic v2 `BaseSettings` instances are immutable by default. Runtime overrides (e.g., `--epochs` CLI flag) are applied by:

1. Writing the override to `os.environ`
2. Calling `get_settings.cache_clear()` to invalidate the singleton
3. The next `get_settings()` call reconstructs settings from scratch, picking up the env var

---

## 4. Data Pipeline Design

### 4.1 DataPipeline Orchestrator

`DataPipeline` (in `src/data/pipeline.py`) is a stateless orchestrator that wires data sources to storage. It does not hold data in memory beyond a single method call.

```
DataPipeline.run_ersst()
  → NOAAERSSTSource.fetch()        # xarray + OPeNDAP
  → clean_ersst_data()             # processing/
  → compute_nino34_index()         # feature engineering
  → db.save_parquet(nino_df, ...)  # storage
  → db.register_source(...)        # metadata

DataPipeline.run_openaq()
  → OpenAQSource.fetch_locations() # REST API
  → db.save_parquet(locations, ...) # persisted for graph builder reuse
  → OpenAQSource.fetch_measurements()
  → clean_pollution_data()
  → add_aqi_column()
  → db.register_source(...)
```

### 4.2 Storage Layer

`Database` (`src/data/storage/db.py`) is a thin wrapper over:
- A **JSON file** (`metadata.json`) for lightweight metadata: source registration, model run logs with timestamps, status, and metrics
- **Parquet files** for bulk tabular data (via `pyarrow`)

All timestamps use `datetime.now(tz=timezone.utc)` for Python 3.12+ compatibility.

### 4.3 ISO Country Code Handling

`src/utils.py` exposes:
- `_ISO2_TO_ISO3` / `_ISO3_TO_ISO2` — internal dicts
- `ISO2_TO_ISO3` / `ISO3_TO_ISO2` — public aliases
- `iso2_to_iso3(code)` / `iso3_to_iso2(code)` — safe lookup helpers

OpenAQ accepts ISO-2 codes; user config uses ISO-3. The conversion happens inside `OpenAQSource`.

---

## 5. Model Architectures

### 5.1 El Niño — Temporal Fusion Transformer (TFT)

**Library:** `pytorch-forecasting $\ge$ 1.8.0`  
**Training driver:** `pytorch-lightning` (transitive dep; `L.Trainer`)  

```
Input: Niño 3.4 time-series (monthly, encoder_length months)
       + static covariates (month-of-year encoding)
  |
  +- Variable Selection Networks
  +- LSTM Encoder (seq2seq)
  +- Static Enrichment
  +- Temporal Self-Attention (multi-head)
  +- Output: predictions for next [1, prediction_months] steps

Loss: QuantileLoss (0.1, 0.5, 0.9)
```

**Key configuration fields** (`ModelSettings`):
- `tft_encoder_length` (default 60 months)
- `tft_prediction_months` (default 12)
- `tft_hidden_size`, `tft_attention_head_size`, `tft_dropout`
- `tft_use_year` (include calendar year as a known covariate)

**Splits:** `chronological_split_bounds()` defines one 70/15/15 split shared with the LSTM. The training `TimeSeriesDataSet` covers only the training months; validation windows stop before the test period. Training removes the previous run's checkpoints so that only the current run's best checkpoint is evaluated.

### 5.2 El Niño — LSTM Baseline

**Architecture:** Bidirectional 2-layer LSTM → Dropout → FC → FC

```
Input: (batch, seq_len, 1) Niño 3.4 monthly sequence
  → BiLSTM (hidden_size, num_layers, bidirectional)
  → Last hidden state: (batch, hidden_size * 2)
  → Dropout
  → Linear(hidden_size * 2, hidden_size) → ReLU → Dropout
  → Linear(hidden_size, prediction_months)
Output: (batch, prediction_months)

Loss: MSELoss
Optimizer: Adam + ReduceLROnPlateau
Early stopping: patience=15 epochs
```

**Key config:** `lstm_hidden_size`, `lstm_num_layers`, `lstm_bidirectional`, `lstm_prediction_months`

### 5.3 Pollution — A3TGCN

**Library:** `torch-geometric-temporal` (A3TGCN layer)

```
Input: [N_sensors, T_steps, features] PM2.5 time-series on sensor graph
  → A3TGCN: Attention + GCN graph conv (Kipf & Welling) + GRU
  → Linear projection
Output: [N_sensors, forecast_horizon]

Graph: k-NN connectivity over sensor (lat, lon) positions
Training: Native PyTorch loop, MSE over observed station-days only
```

Missing readings are NaN: they are excluded from normalisation statistics (computed on training days only), imputed with the training mean as inputs, and masked out of the loss and evaluation. Exceedance is evaluated at PM2.5 ≥ 35.5 µg/m³ (AQI > 100) against persistence and always-alert baselines.

**Key config:** `gnn_num_periods` (sliding window size, default 7)

### 5.4 Neural Weather Model — Spherical Mesh ST-GNN

**Architecture:** Encoder–Processor–Decoder over a Fibonacci spherical mesh (no lightning)

```
Grid nodes (lat × lon ERA5 grid)
  |  StaticFeatures: [sin(lat), cos(lat), sin(lon), cos(lon)]
  |  DynamicFeatures: z-normalised T2m, Precip (window_size steps)
  |
  v Encoder (BipartiteGNN: Grid → Mesh)
    SAGEConv((hidden, hidden), hidden) + LayerNorm + GELU

Mesh nodes (2,000 Fibonacci lattice points on unit sphere)
  v Processor (N × SAGEConv Mesh → Mesh)
    At each layer: msg = GELU(LN(SAGEConv(mesh_feat) + condition_embed(Niño3.4)))
                   mesh_feat = mesh_feat + msg

  v Decoder (BipartiteGNN: Mesh → Grid)
    SAGEConv((hidden, hidden), hidden) + LayerNorm + GELU
    + residual from grid_embed

  v Output Projection
    Linear(hidden, grid_out_channels)

Output: (batch, N_grid_nodes, prediction_steps × 2)
Loss: cos(latitude)-weighted MSE  |  Optimizer: AdamW + ReduceLROnPlateau
```

**Conditioning source** (`nwm.conditioning`): `observed` (monthly Niño 3.4 for the month containing each sample), `forecast` (TFT lead-1 forecasts from `export_tft_forecasts.py`, i.e. issued before the month starts) or `none`. `run_nwm_ablation.py` compares these, and an unweighted-loss variant, on area-weighted test RMSE.

#### Graph Construction (`SphericalGraphBuilder`)

1. Project all (lat, lon) grid points to 3D Cartesian: `(cos(lat)cos(lon), cos(lat)sin(lon), sin(lat))`
2. Generate 2,000 uniform mesh points using the Fibonacci lattice
3. Build KD-trees over both sets; query k nearest neighbours (k=4 for G<->M, k=8 for M<->M)
4. Edges are stored as `[2, E]` LongTensor in COO format

#### Batched Edge Expansion

A vectorised `_batch_edges()` function expands single-graph edge indices to batched graphs using `repeat` + `repeat_interleave` — avoiding Python loops over the batch dimension:

```python
src_offsets = (torch.arange(B, device=d) * num_src).repeat_interleave(E)
tgt_offsets = (torch.arange(B, device=d) * num_tgt).repeat_interleave(E)
src = edge_idx[0].repeat(B) + src_offsets
tgt = edge_idx[1].repeat(B) + tgt_offsets
```

### 5.5 Climate Feedback — Warming-Rate Change-Points

**Function:** `detect_slope_changepoints()` (NumPy only)

```
Data: annual means of complete years of NASA GISTEMP (1950–2025)
Model: continuous piecewise-linear trend
       mu(t) = b0 + b1 t + sum_k d_k max(0, t - tau_k)
Errors: AR(1), fitted by two-step Prais–Winsten GLS, exact Gaussian likelihood
Search: every set of 0–3 breaks on the annual grid, segments >= 10 years
Selection: minimum BIC with 4 + 2K parameters
```

A mean-shift CUSUM (`detect_changepoints_simple()`) is kept as a diagnostic only: on a trending series it reports spurious level shifts.

### 5.6 Climate Feedback — Bayesian Emissions Regression

**Library:** `pymc $\ge$ 6`, `arviz $\ge$ 1.2`

```
Predictors X: [CO2_anomaly, CH4_anomaly, N2O_anomaly] (z-normalised)
Response y: Global temperature anomaly (NASA GISTEMP)

Priors:
  alpha ~ Normal(0, 1)       # intercept
  beta_i ~ Normal(0, 1)     # coefficients
  sigma ~ HalfNormal(1)      # observation noise

Likelihood:
  y ~ Normal(alpha + X @ beta, sigma)

Sampling: NUTS (default 1000 draws, 2 chains)
```

Climate TRACE and GISTEMP currently overlap for only about 6 years, so this model is not yet informative and is not used in evaluation.

Normalisation metadata (`predictor_cols`, `X_mean`, `X_std`) is returned separately from `build_bayesian_model()` rather than attached to the `pm.Model` object.

---

## 6. Training Loop Design

### 6.1 Native PyTorch Pattern

Both the LSTM and NWM use the same idiomatic native loop:

```
for epoch in range(max_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    if val_loader:
        val_loss = validate(model, val_loader)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), best_path)
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= early_stopping_patience:
            break

torch.save(model.state_dict(), final_path)
csv.DictWriter → metrics.csv
db.complete_model_run(run_id, ...)
```

### 6.2 TFT Training (pytorch-forecasting / Lightning)

TFT requires a `pytorch-lightning` `Trainer` because `TemporalFusionTransformer` is a `LightningModule`. The trainer is configured with:
- `EarlyStopping(monitor="val_loss", patience=10)`
- `ModelCheckpoint(monitor="val_loss", save_top_k=1)`
- A custom `LoguruLoggingCallback` that routes Lightning metrics to the Loguru logger

---

## 7. Key Design Decisions

### 7.1 No Explicit Lightning Dependency

**Decision:** Remove `pytorch-lightning` from `pyproject.toml` and `requirements.txt`.  
**Rationale:** Only the TFT model requires it, and only as a transitive dependency of `pytorch-forecasting`. Pinning an explicit version exposed the project to the supply-chain compromise in `lightning` 2.6.2/2.6.3. Letting `pytorch-forecasting` manage the compatible version is safer and reduces the dependency surface.

### 7.2 PYTHONPATH=src Without Package Namespace Prefix

**Decision:** Use bare module imports (`from config import settings`) rather than absolute imports (`from src.config import settings`).  
**Rationale:** All Makefile targets and scripts set `PYTHONPATH=src`. The package is installed with `pip install -e .` using the `[tool.setuptools.package-dir] "" = "src"` configuration, which makes the same bare imports work in both installed and PYTHONPATH modes.

### 7.3 File-Based Storage (No Database Server)

**Decision:** Use JSON + Parquet files instead of SQLite/PostgreSQL.  
**Rationale:** The project is research-focused with a single-user workflow. File-based storage is zero-infrastructure, fully portable, and trivially inspectable. The `Database` class provides a clean interface that could be swapped for a real DB in a production context.

### 7.4 Vectorised Batch Edges in NWM

**Decision:** Use `repeat` + `repeat_interleave` for edge index batching.  
**Rationale:** The original Python loop over the batch dimension was O(B×E) in Python overhead — a significant bottleneck for batch_size=32 with 8,000 edges per graph type. The vectorised version performs the same operation in a single tensor kernel call.

### 7.5 No Leakage Across Splits

**Decision:** Normalisation statistics (NWM, pollution) come from the training split only; El Niño training and validation targets never fall in a later split.  
**Rationale:** Statistics or targets drawn from validation/test data let evaluation data inform training, inflating reported skill. Each is covered by a test in `tests/`.

### 7.6 Honest Artifacts

**Decision:** Evaluation artifacts never fall back to synthetic or default values.  
**Rationale:** A missing result is shown as "n/a" or an empty panel, so that no invented number or curve can reach the paper or patent.

### 7.7 Inference-Only Streamlit App

**Decision:** `src/app/` loads trained checkpoints and runs inference; it never trains. All model and data access is in `loaders.py` as plain, cached functions with no Streamlit import, and the pages in `views/` only render. Pages are selected with `st.navigation(position="top")`, so only the chosen page runs.  
**Rationale:** Keeping logic out of the UI makes it testable (`tests/test_app_loaders.py`) and keeps one page's heavy model load from slowing the others.

- **Checkpoint resolution:** `find_checkpoint()` searches the live checkpoint directory and then each committed snapshot, newest first, so a fresh clone can run without retraining. Missing data raises `DataMissing` with the command that creates it, and the page shows that message.
- **Reuse:** the weather page builds the model exactly as `evaluate_nwm_checkpoint()` does; the El Niño page calls `elnino_test_predictions()` and `elnino_metrics()`; the warming page calls `detect_slope_changepoints()`.
- **Pollution offline:** `build_pollution_splits()` calls the OpenAQ API for station locations, so the app rebuilds the graph from the cached measurement file (which carries each station's coordinates). A test run reproduces the reported 85 observed station-days and RMSE 46.58 µg/m³.
- **Roll-forward is outside the validated scope:** the NWM was trained and tested for one 6-hour step. The app can feed predictions back in (sliding both input windows), but labels this as a demonstration on screen.
- **Batch dependence:** `torch_geometric.nn.LayerNorm` defaults to graph mode and, without a batch vector, normalises over every node in the batch, so a prediction depends slightly on the other samples in its batch. The reported 1.214 K was measured in batches of 16; the app runs one sample at a time and gets 1.217 K. Fixing it needs per-sample normalisation and retraining.

---

## 8. Hardware Configuration

| Target | Makefile Target | torch index URL | PyG wheel |
|:---|:---|:---|:---|
| CPU / CUDA | `make install` | default (PyPI), torch 2.12.1 | `torch-2.12.1+cpu` |
| AMD ROCm 6.1 | `make install-rocm` | `download.pytorch.org/whl/rocm6.1`, torch 2.6.0 | `torch-2.6.0+cpu` |
| Intel NPU | `make install-npu` | default + OpenVINO, torch 2.12.1 | `torch-2.12.1+cpu` |

`torch-scatter` and `torch-sparse` are deprecated as of torch-geometric 2.8.0 and replaced by `pyg-lib`, installed from `https://data.pyg.org/whl/torch-<version>+cpu.html` for the matching torch version (the ROCm path also builds `torch-scatter`/`torch-sparse` via `install_rocm_deps.py`).

---

## 9. Data Flow Diagram

```
External APIs / Files
        |
        v
+-------------------+     +------------------------------+
| Data Sources      |----->|  DataPipeline.run_*()        |
| NOAA / OpenAQ /   |     |  fetch → clean → engineer    |
| NASA / TRACE /    |     +--------------+---------------+
| ERA5 Zarr         |                    |
+-------------------+                    v
                              +----------------------+
                              |  Database (db.py)     |
                              |  +- metadata.json     |
                              |  +- *.parquet files   |
                              +----------+------------+
                                         |
          +------------------------------+------------------------------+
          v                              v                              v
  +--------------+              +------------------+           +--------------+
  |  El Niño     |              |  Pollution       |           |  NWM         |
  |  Dataset     |              |  GraphDataset    |           |  NWMDataset  |
  |  (Niño 3.4)  |              |  (PM2.5 sensor)  |           |  (ERA5 .nc)  |
  +------+-------+              +-------+----------+           +------+-------+
         |                             |                              |
         v                             v                              v
  +--------------+              +------------------+           +--------------+
  |  TFT / LSTM  |              |  A3TGCN Trainer  |           |  NWMTrainer  |
  |  Trainer     |              |  (native PyTorch)|           |  (native     |
  |  (Lightning) |              +------------------+           |   PyTorch)   |
  +--------------+                                             +--------------+
         |                             |                              |
         +-----------------------------+------------------------------+
                                       |
                                       v
                               +----------------------+
                               |  runtime/             |
                               |  +- models/           |
                               |  +- eval_artifacts/   |
                               |  +- saved/            |
                               |  +- logs/             |
                               +----------------------+
```

---

## 10. Generated Datasets and Evaluation Artifacts

The ClimatePulse pipeline generates two main classes of artifacts: processed datasets (used for training) and evaluation artifacts (plots and metrics used for analysis and the research paper).

### 10.1 Processed Datasets (`datasets/processed/`)
- **`ersst_v5.parquet`**: Cleaned sea surface temperature anomalies.
- **`nino34_index.parquet` / `nino_features.parquet`**: Calculated Niño 3.4 index and engineered features for ENSO forecasting.
- **`gistemp_features.parquet` / `gistemp_global_mean.parquet`**: Global temperature features and anomalies.
- **`regression_dataset.parquet`**: Annual emissions × temperature dataset for the Bayesian emissions regression.
- **`nino34_tft_lead1.parquet`**: TFT lead-1 Niño 3.4 forecasts for forecast-conditioned NWM runs.
- **`openaq/`**: Graph-structured dataset representing the spatial topology of global air quality sensors.
- **`climate_trace/`**: Processed multi-sector greenhouse gas emissions inventories.

### 10.2 Evaluation Artifacts (`runtime/eval_artifacts/`)
Running `make gen-eval-artifacts` executes model inference and generates publication-ready plots and metrics.

**El Niño Forecaster (TFT & LSTM):**
- `elnino_forecast_vs_actual.png`: 12-month ahead SST anomaly forecasts (median and intervals) vs observations.
- `elnino_crps_leadtime.png`: RMSE by forecast lead time for TFT, LSTM and persistence.
- `elnino_calibration_diagram.png`: Observed frequency below each TFT quantile (0.1, 0.5, 0.9) on the test set.
- `elnino_predicted_vs_actual_scatter.png`: Scatter plot comparing TFT and LSTM error clusters.
- `elnino_error_heatmap.png`: Heatmap of absolute forecast errors across lead horizons and chronological windows.
- `elnino_quantile_coverage.png`: Probabilistic 6-month forecast trajectories vs observed anomalies.

**Pollution Dispersion (A3TGCN):**
- `pollution_dispersion_nodes.png`: Distribution of per-station RMSE (µg/m³).
- `pollution_confusion_matrix.png`: Next-day exceedance matrix at PM2.5 ≥ 35.5 µg/m³ (AQI > 100), observed station-days only.

**Neural Weather Model (ST-GNN):**
- `nwm_convergence_lr.png`: MSE loss convergence and ReduceLROnPlateau scheduling curves.
- `nwm_predicted_vs_actual.png`: Visual comparison of predicted global T2m fields against ERA5 ground truth.
- `nwm_spatial_error_map.png`: Time-averaged RMSE per spatial grid node.
- `nwm_fibonacci_mesh.png`: 3D visualization of the bipartite graph mapping 2,048 equirectangular grid nodes to a 2,000-node Fibonacci lattice.

**Climate Feedback (Change-Points):**
- `climate_changepoint_regression.png`: GISTEMP anomalies with the BIC-selected piecewise-linear trend and change-points.

**General Metrics (`run_results/`):**
- `summary_metrics.json`: High-level aggregated performance skill scores across all models.
- `model_training_curves.png`: Multi-panel figure of training/validation loss trajectories for all deep learning models.
- Model-specific performance CSVs (`tft_metrics.csv`, `lstm_metrics.csv`, `nwm_metrics.csv`, `pollution_metrics.csv`).
- `elnino_seeds.csv` and `nwm_ablation.csv`: multi-seed El Niño and NWM ablation results, when those runs have been made.

### 10.3 Documentation & Patent Build Pipeline
- **`make gen-research-paper-pdf`**: Compiles `docs/research_paper/research_paper.md` to `climatepulse_research_paper.pdf` via XeLaTeX.
- **`make gen-patent-application-pdf`**: Compiles `docs/patent/patent_application.md` to `climatepulse_patent_application.pdf` via XeLaTeX.
- **`docs/patent/patent_application.md`**: Consolidated Single Patent Specification draft titled *"Unified Multi-Modal Climate Inference System Operating on a Fibonacci Spherical Latent Mesh with ENSO-Conditioned Neural Weather Modeling"* targeting IPINDIA / PCT filing.

---

## 11. Reproducibility and Execution Environments

**Command sequence.** The full rebuild is: install; clean; `download-data`; `download-nwm-data`; `test`; `train-elnino-seeds`; set `tft_use_year`; `train-elnino`; `train-pollution`; `train-nwm`; `export-tft-forecasts`; `nwm-ablation`; `gen-eval-artifacts`; `save-run`; `sync-paper-artifacts`; and the two PDF targets. Dependencies between steps: `nwm-ablation` scores the main weather model (`train-nwm` first) and the forecast-conditioned variant (`export-tft-forecasts` first, which needs the main TFT); `clean-eval-artifacts` also removes the seed results.

**Seeds.** `model.seed` seeds Python, NumPy and PyTorch (`utils.set_seed()`) before every model is built. `run_elnino_seeds.py` runs one seed per training subprocess with separate checkpoint, metadata and log directories, and chooses the TFT variant on validation loss only, so test data never drives a modelling choice. Residual non-determinism comes from hardware and thread counts.

**Isolation of concurrent runs.** `run_nwm_ablation.py` starts one training process per variant with its own `MODEL_CHECKPOINT_DIR`, `META_METADATA_PATH` and `LOG_DIR`, because the shared metadata JSON and log files are not safe for concurrent writers.

**Google Colab.** `run.ipynb` follows the same sequence. It installs the package on top of Colab's PyTorch, replaces the Makefile's `.venv/bin/python` with a one-line wrapper around the notebook interpreter so that every `make` target works unchanged, reads the OpenAQ key from Colab secrets, and (optionally) mounts Google Drive to restore and save datasets and results across sessions. A quick mode (2 epochs, one seed) checks the workflow end to end.

**Not automated.** Result numbers quoted in the paper, patent and README are updated by hand.

---

## 12. Known Limitations and Future Work

| Area | Current State | Potential Improvement |
|:---|:---|:---|
| NWM mesh resolution | 2,000 Fibonacci nodes | Hierarchical icosahedral mesh (GraphCast-style) |
| NWM variables | T2m, Precip only | Add wind, humidity, pressure level data |
| Pollution graph | Static k-NN graph | Dynamic graph with time-varying edge weights |
| TFT covariates | Niño 3.4 only | Add $\text{CO}_2$ ppm, solar index as global covariates |
| Emissions regression | Single-level, ~6 overlapping years | Longer emissions record (e.g. pre-2015 inventories); hierarchical by region |
| Evaluation | RMSE, MAE, ACC, skill, quantile calibration | Add CRPS for the probabilistic TFT forecasts |
| Mixed precision | Not implemented | FP16/BF16 training on supported GPUs |
| Artifact generator | Monolithic 27 KB script | Split into per-model artifact modules |
| NWM layer norm | Graph-mode `LayerNorm` pools statistics across a batch, so predictions depend slightly on batch composition | Per-sample normalisation (needs retraining) |
| Simulation app | Reads the cached OpenAQ file; edits to model code are not hot-reloaded in cached loaders | Packaged datasets for demos; model registry |
