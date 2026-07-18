# ClimatePulse — Architecture and Design Document

**Version:** 1.0  
**Date:** July 2026  
**Status:** Active  

---

## 1. System Overview

ClimatePulse is a modular, Python-only research platform for integrated global climate forecasting. It is structured as four independent model subsystems sharing a common data pipeline, configuration layer, and storage backend.

```
┌─────────────────────────────────────────────────────────────────┐
│                         ClimatePulse                            │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  ┌───────┐  │
│  │  El Niño     │  │  Pollution   │  │    NWM    │  │ Bayes │  │
│  │  TFT / LSTM  │  │   A3TGCN    │  │  ST-GNN   │  │ Regr. │  │
│  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘  └───┬───┘  │
│         │                 │                │             │      │
│  ┌──────┴─────────────────┴────────────────┴─────────────┴────┐  │
│  │               Unified Data Pipeline (pipeline.py)          │  │
│  └──────┬─────────────────────────────────────────────────────┘  │
│         │                                                        │
│  ┌──────┴──────────────────────────────────────────────────────┐  │
│  │  Data Sources: NOAA ERSST · OpenAQ · NASA GISTEMP ·         │  │
│  │                Climate TRACE · Google Cloud ERA5 Zarr        │  │
│  └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Repository Layout

```
src/                  # PYTHONPATH root
├── config.py         # Pydantic-Settings: DataSettings, ModelSettings, LogSettings, …
├── utils.py          # Device detection, ISO code conversion maps + helpers
├── data/
│   ├── pipeline.py           # DataPipeline orchestrator
│   ├── sources/              # One class per external data source
│   │   ├── noaa_ersst.py     # NOAAERSSTSource
│   │   ├── openaq.py         # OpenAQSource
│   │   ├── nasa_gistemp.py   # NASAGISTEMPSource
│   │   └── climate_trace.py  # ClimateTraceSource
│   ├── processing/           # Feature engineering functions
│   └── storage/
│       └── db.py             # Database: JSON metadata + Parquet bulk store
├── models/
│   ├── el_nino/
│   │   ├── tft_model.py      # TemporalFusionTransformer (pytorch-forecasting)
│   │   ├── lstm_baseline.py  # LSTMForecaster (native PyTorch)
│   │   ├── dataset.py        # LSTMDataset, TFT TimeSeriesDataSet builder
│   │   ├── trainer.py        # train_tft(), train_lstm()
│   │   └── evaluate.py       # Metrics: RMSE, MAE, skill score
│   ├── pollution/
│   │   ├── stgnn_model.py    # A3TGCN wrapper
│   │   ├── graph_dataset.py  # Sensor graph construction
│   │   ├── trainer.py        # Native PyTorch loop
│   │   └── evaluate.py
│   ├── nwm/
│   │   ├── stgnn_model.py    # NWMModel (Encoder–Processor–Decoder)
│   │   ├── graph_builder.py  # SphericalGraphBuilder (Fibonacci mesh, KD-tree)
│   │   ├── dataset.py        # NWMDataset (ERA5 NetCDF + Niño conditioning)
│   │   └── trainer.py        # NWMTrainer (native PyTorch, no lightning)
│   └── climate_feedback/
│       ├── bayesian_regression.py  # build_bayesian_model(), fit_bayesian_model()
│       └── evaluate.py
└── scripts/
    ├── download_data.py           # CLI: ingest pipeline
    ├── download_nwm_data.py       # CLI: ERA5 Zarr download
    ├── train_models.py            # CLI: train all / individual models
    └── generate_eval_artifacts.py
```

---

## 3. Configuration Architecture

### 3.1 Settings Hierarchy

```
get_settings()  →  Settings (singleton, lru_cache)
                    ├── DataSettings      (prefix: DATA_)
                    ├── ModelSettings     (prefix: MODEL_)
                    ├── LogSettings       (prefix: LOG_)
                    ├── TFTModelSettings  (prefix: TFT_)
                    └── GRPCSettings      (prefix: GRPC_)
```

Each `BaseSettings` subclass reads from three sources in increasing priority:
1. Default field values in code
2. `configs/runtime_conf.yaml` (loaded explicitly in `get_settings()`)
3. `.env` file and environment variables (Pydantic-Settings native)

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

**Library:** `pytorch-forecasting ≥ 1.8.0`  
**Training driver:** `pytorch-lightning` (transitive dep; `L.Trainer`)  

```
Input: Niño 3.4 time-series (monthly, encoder_length months)
       + static covariates (month-of-year encoding)
  │
  ├─ Variable Selection Networks
  ├─ LSTM Encoder (seq2seq)
  ├─ Static Enrichment
  ├─ Temporal Self-Attention (multi-head)
  └─ Output: predictions for next [1, prediction_months] steps

Loss: QuantileLoss (0.1, 0.5, 0.9)
```

**Key configuration fields** (`ModelSettings`):
- `tft_encoder_length` (default 120 months)
- `tft_prediction_months` (default 12)
- `tft_hidden_size`, `tft_attention_head_size`, `tft_dropout`

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
  → A3TGCN: Attention + Chebyshev graph conv + GRU
  → Linear projection
Output: [N_sensors, forecast_horizon]

Graph: k-NN connectivity over sensor (lat, lon) positions
Training: Native PyTorch loop
```

**Key config:** `gnn_num_periods` (sliding window size, default 7)

### 5.4 Neural Weather Model — Spherical Mesh ST-GNN

**Architecture:** Encoder–Processor–Decoder over a Fibonacci spherical mesh (no lightning)

```
Grid nodes (lat × lon ERA5 grid)
  │  StaticFeatures: [sin(lat), cos(lat), sin(lon), cos(lon)]
  │  DynamicFeatures: z-normalised T2m, Precip (window_size steps)
  │
  ▼ Encoder (BipartiteGNN: Grid → Mesh)
    SAGEConv((hidden, hidden), hidden) + LayerNorm + GELU

Mesh nodes (2,000 Fibonacci lattice points on unit sphere)
  ▼ Processor (N × SAGEConv Mesh → Mesh)
    At each layer: mesh_feat += condition_embed(Niño3.4)
                   mesh_feat += SAGEConv(mesh_feat) + residual

  ▼ Decoder (BipartiteGNN: Mesh → Grid)
    SAGEConv((hidden, hidden), hidden) + LayerNorm + GELU
    + residual from grid_embed

  ▼ Output Projection
    Linear(hidden, grid_out_channels)

Output: (batch, N_grid_nodes, prediction_steps × 2)
Loss: MSELoss  |  Optimizer: AdamW + ReduceLROnPlateau
```

#### Graph Construction (`SphericalGraphBuilder`)

1. Project all (lat, lon) grid points to 3D Cartesian: `(cos(lat)cos(lon), cos(lat)sin(lon), sin(lat))`
2. Generate 2,000 uniform mesh points using the Fibonacci lattice
3. Build KD-trees over both sets; query k nearest neighbours (k=4 for G↔M, k=8 for M↔M)
4. Edges are stored as `[2, E]` LongTensor in COO format

#### Batched Edge Expansion

A vectorised `_batch_edges()` function expands single-graph edge indices to batched graphs using `repeat` + `repeat_interleave` — avoiding Python loops over the batch dimension:

```python
src_offsets = (torch.arange(B, device=d) * num_src).repeat_interleave(E)
tgt_offsets = (torch.arange(B, device=d) * num_tgt).repeat_interleave(E)
src = edge_idx[0].repeat(B) + src_offsets
tgt = edge_idx[1].repeat(B) + tgt_offsets
```

### 5.5 Climate Feedback — Bayesian Regression

**Library:** `pymc ≥ 6`, `arviz ≥ 1.2`

```
Predictors X: [CO2_anomaly, CH4_anomaly, N2O_anomaly] (z-normalised)
Response y: Global temperature anomaly (NASA GISTEMP)

Priors:
  α ~ Normal(0, 1)       # intercept
  β_i ~ Normal(0, 1)     # coefficients
  σ ~ HalfNormal(1)      # observation noise

Likelihood:
  y ~ Normal(α + X @ β, σ)

Sampling: NUTS (default 1000 draws, 2 chains)
```

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

### 7.5 Normalisation Leak Fix

**Decision:** NWM z-score statistics computed from training split only.  
**Rationale:** Computing statistics over the full dataset before splitting causes val/test targets to inform training normalisation — a subtle but real data leakage. Statistics are now derived from `t2m[:train_end]` and applied uniformly.

---

## 8. Hardware Configuration

| Target | Makefile Target | torch index URL | PyG wheel |
|:---|:---|:---|:---|
| CPU / CUDA | `make install` | default (PyPI) | `torch-2.13.0+cpu` |
| AMD ROCm 7.2 | `make install-rocm` | `download.pytorch.org/whl/rocm7.2` | `torch-2.13.0+rocm7.2` |
| Intel NPU | `make install-npu` | default + OpenVINO | `torch-2.13.0+cpu` |

`torch-scatter` and `torch-sparse` are deprecated as of torch-geometric 2.8.0 and replaced by `pyg-lib`, which is installed from `https://data.pyg.org/whl/torch-2.13.0+{variant}.html`.

---

## 9. Data Flow Diagram

```
External APIs / Files
        │
        ▼
┌───────────────────┐     ┌──────────────────────────────┐
│ Data Sources      │────▶│  DataPipeline.run_*()        │
│ NOAA / OpenAQ /   │     │  fetch → clean → engineer    │
│ NASA / TRACE /    │     └──────────────┬───────────────┘
│ ERA5 Zarr         │                    │
└───────────────────┘                    ▼
                              ┌──────────────────────┐
                              │  Database (db.py)     │
                              │  ├─ metadata.json     │
                              │  └─ *.parquet files   │
                              └──────────┬────────────┘
                                         │
          ┌──────────────────────────────┼──────────────────────────────┐
          ▼                              ▼                              ▼
  ┌──────────────┐              ┌──────────────────┐           ┌──────────────┐
  │  El Niño     │              │  Pollution       │           │  NWM         │
  │  Dataset     │              │  GraphDataset    │           │  NWMDataset  │
  │  (Niño 3.4)  │              │  (PM2.5 sensor)  │           │  (ERA5 .nc)  │
  └──────┬───────┘              └───────┬──────────┘           └──────┬───────┘
         │                             │                              │
         ▼                             ▼                              ▼
  ┌──────────────┐              ┌──────────────────┐           ┌──────────────┐
  │  TFT / LSTM  │              │  A3TGCN Trainer  │           │  NWMTrainer  │
  │  Trainer     │              │  (native PyTorch)│           │  (native     │
  │  (Lightning) │              └──────────────────┘           │   PyTorch)   │
  └──────────────┘                                             └──────────────┘
         │                             │                              │
         └─────────────────────────────┴──────────────────────────────┘
                                       │
                                       ▼
                               ┌──────────────────────┐
                               │  runtime/             │
                               │  ├─ models/           │
                               │  ├─ eval_artifacts/   │
                               │  ├─ saved/            │
                               │  └─ logs/             │
                               └──────────────────────┘
```

---

## 10. Known Limitations and Future Work

| Area | Current State | Potential Improvement |
|:---|:---|:---|
| NWM mesh resolution | 2,000 Fibonacci nodes | Hierarchical icosahedral mesh (GraphCast-style) |
| NWM variables | T2m, Precip only | Add wind, humidity, pressure level data |
| Pollution graph | Static k-NN graph | Dynamic graph with time-varying edge weights |
| TFT covariates | Niño 3.4 only | Add CO₂ ppm, solar index as global covariates |
| Bayesian model | Single-level regression | Hierarchical by region/country |
| Evaluation | RMSE/MAE | Add CRPS (probabilistic), anomaly correlation coefficient |
| Artifact generator | Monolithic 27 KB script | Split into per-model artifact modules |
