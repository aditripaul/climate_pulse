---
header-includes:
  - \usepackage{amsmath}
  - \usepackage{amssymb}
  - \usepackage{unicode-math}
  - \usepackage{fancyhdr}
  - \usepackage[utf8]{inputenc}
  - \usepackage{draftwatermark}
  - \SetWatermarkText{research.aditripaul.org}
  - \SetWatermarkScale{0.25}
  - \SetWatermarkColor[gray]{0.95}
  - \newcommand{\gt}{>}
  - \pagestyle{fancy}
  - '\fancyhead[L]{\small ClimatePulse Project Proposal}'
  - \fancyhead[R]{\small \today}
  - \fancyfoot[L]{pre-print DRAFT}
  - \fancyfoot[C]{\thepage}
  - \fancyfoot[R]{\footnotesize © 2026 research.aditripaul.org}
  - \renewcommand{\headrulewidth}{0.4pt}
  - \setlength{\headheight}{15pt}
---

# PROJECT PROPOSAL

# ClimatePulse: AI-Powered Global Climate Forecasting and Environmental Telemetry Framework

### Aditri Paul (2428010209)
### Manipal University Jaipur

---

## EXECUTIVE SUMMARY

Global climate change presents an urgent need for fast, high-precision, and multi-scale environmental forecasting tools. Traditional Numerical Weather Prediction (NWP) systems are computationally prohibitive, requiring supercomputers running for hours per forecast cycle. Conversely, existing data-driven AI weather models operate in disciplinary isolation—predicting short-term atmospheric states without integrating long-range macro-climatic forcings (such as El Niño-Southern Oscillation, or ENSO) or local urban pollution transport.

**ClimatePulse** addresses this challenge by proposing a unified, modular, open-source Python framework that bridges four core climate modeling dimensions:
1. **El Niño Forecasting:** Probabilistic multi-horizon forecasting of the Niño 3.4 sea surface temperature index 1–12 months ahead.
2. **Pollution Dispersion:** Spatial-temporal forecasting of $\text{PM}_{2.5}$ concentrations across a global sensor network.
3. **Neural Weather Modeling (NWM):** Global weather forecasting operating on a uniform 3D spherical latent mesh conditioned on macro-climate ENSO signals.
4. **Climate Feedback Attribution:** Statistical change-point analysis detecting shifts in the rate of historical global warming.

This proposal outlines the technical vision, system architecture, dataset sources, technical stack, expected evaluation metrics, and system requirements for the ClimatePulse platform.

---

## 1. PROBLEM STATEMENT AND MOTIVATION

### 1.1 Limitations of Classical Numerical Weather Prediction (NWP)
Systems such as ECMWF's Integrated Forecasting System (IFS) rely on numerical integration of primitive fluid dynamic equations over high-resolution grids. While accurate, their extreme computational latency makes real-time scenario simulation, agricultural planning, and civil protection response difficult.

### 1.2 Limitations of Existing Data-Driven AI Weather Models
Recent neural weather models (GraphCast, FourCastNet, Pangu-Weather, FuXi) have reduced forecast latency by orders of magnitude. However, they suffer from key structural limitations:

1. **Equirectangular Polar Distortion:** Models operating on standard latitude-longitude grids apply spatially invariant convolutions across unequal physical surface areas. Equatorial cells subtend up to 8.6$\times$ more physical area than high-latitude cells, causing unweighted models to over-allocate computational capacity to polar regions.
2. **Single-Task Unconditioned Architecture:** Prior models map atmospheric states from time $t \to t+1$ without incorporating large-scale ocean-atmosphere forcing signals (such as ENSO phases) into latent message passing. Consequently, they treat El Niño and La Niña initializations identically.
3. **Rigidity of Icosahedral Meshes:** Prior graph-based models rely on icosahedral subdivision, locking computational mesh resolution to fixed discrete node counts ($20, 80, 320, 1280, 5120$).
4. **Disciplinary Silos:** No unified platform connects global climate oscillations, local air quality dispersion, and global atmospheric field forecasting in a single reproducible framework.

---

## 2. SYSTEM ARCHITECTURE AND MODEL SUBSYSTEMS

ClimatePulse connects four specialized AI modeling components through a single data pipeline. The key innovation is **cross-domain conditioning**: long-range El Niño forecasts feed directly into the daily Neural Weather Model as a background climate signal.

### 2.1 Subsystem Overview

| Model / Subsystem | Core Task | Architecture | Target Data Source |
|:---|:---|:---|:---|
| **El Niño Forecaster** | Predict ENSO phase 1–12 months ahead | Temporal Fusion Transformer (TFT) + BiLSTM baseline | NOAA ERSST v5 |
| **Pollution Dispersion** | Spatiotemporal $\text{PM}_{2.5}$ transport | Attention-based Temporal Graph Conv (A3TGCN) | OpenAQ v3 Telemetry |
| **Climate Feedback** | Detect warming regime changepoints | Piecewise-linear trend with AR(1) errors, BIC model selection; Bayesian emissions regression (PyMC) | NASA GISTEMP v4 & Climate TRACE |
| **Neural Weather Model** | Global $T_{2\text{m}}$ & precipitation conditioned on ENSO | Spherical Fibonacci Mesh Graph Neural Network (ST-GNN) | WeatherBench 2 / ERA5 |

```
+----------------------------------------------------------------------------------+
|                           CLIMATEPULSE DATA PIPELINE                             |
|               Ingests: NOAA Ocean SST * OpenAQ Telemetry * ERA5 Reanalysis       |
+----------------------------------------+-----------------------------------------+
                                         |
     +-----------------------------------+-----------------------------------+
     |                                   |                                   |
     v                                   v                                   v
+------------------------+   +------------------------+   +------------------------+
|  1. EL NIÑO FORECASTER  |   | 2. POLLUTION DISPERSION|   |  4. CLIMATE FEEDBACK   |
|  Models macro ocean    |   | Models local sensor    |   | Models long-term global|
|  oscillations (ENSO)   |   | network air quality    |   | temperature trends     |
|                        |   |                        |   |                        |
| Output: Niño 3.4 Index |   | Output: PM2.5 & AQI    |   | Output: Changepoints   |
+-----------+------------+   +------------------------+   +------------------------+
            |
            | (Macro-Climate Background Conditioning)
            v
+----------------------------------------------------------------------------------+
|                         3. NEURAL WEATHER MODEL (NWM)                            |
| Maps observation grid -> Uniform 3D Spherical Mesh -> Conditioned Latent Processor |
|                                                                                  |
| Output: Predicted Global Temperature (T2m) and Precipitation Fields             |
+----------------------------------------------------------------------------------+
```

### 2.2 Subsystem Details

#### 2.2.1 El Niño Forecaster (Ocean Oscillation Module)
- **Role:** Predicts the phase and intensity of the El Niño-Southern Oscillation (ENSO) 1–12 months in advance using historical ocean temperature records.
- **How it Works:** Uses a Temporal Fusion Transformer (TFT) with self-attention and variable selection networks to process multi-year ocean temperature series.
- **Key Output:** Delivers probabilistic prediction intervals (10th, 50th, and 90th percentiles) for the Niño 3.4 index to capture forecast uncertainty.

#### 2.2.2 Pollution Dispersion (Air Quality Module)
- **Role:** Forecasts local $\text{PM}_{2.5}$ pollutant concentrations across a network of global monitoring stations.
- **How it Works:** Combines spatial graph neural networks (which connect nearby sensor stations based on geographic distance) with temporal recurrent units to model how pollutants diffuse over space and accumulate over time.
- **Key Output:** Generates per-station concentration forecasts and automated air quality exceedance alert warnings.

#### 2.2.3 Neural Weather Model (Global Weather Module)
- **Role:** Predicts global surface temperatures ($T_{2\text{m}}$) and total precipitation fields.
- **How it Works:**
  1. **Uniform 3D Spherical Mesh:** Converts flat latitude-longitude maps into a 2,000-node 3D spherical mesh (using a golden-angle spiral) to eliminate polar surface distortion.
  2. **Macro-Climate Conditioning:** Injects the predicted El Niño state directly into the neural network's internal processor layers so that the global forecast is dynamically shaped by the active climate phase.
  3. **Physical Area Weighting:** Weights the learning objective by the physical surface area of each grid cell to ensure fair global prediction accuracy.
- **Key Output:** Gridded global atmospheric predictions for upcoming forecast windows.

#### 2.2.4 Climate Feedback Attribution (Regime Shift Module)
- **Role:** Analyzes long-term historical global temperature trends (1950–present) to identify structural shifts in global warming rates.
- **How it Works:** Fits continuous piecewise-linear trends with autocorrelated errors to annual surface temperature anomalies and selects the number of regime changes by BIC; a Bayesian regression relates temperature to greenhouse gas emissions inventories.
- **Key Output:** Identifies changes in the warming rate and estimates the rate within each regime.

---

## 3. DATASET SOURCES AND INGESTION PIPELINE

Data ingestion is constrained to the **1950–present** window for high-fidelity historical relevance. All datasets are fetched programmatically via the unified data ingestion pipeline:

| Dataset Source | Content & Variables | Ingestion Access Mechanism | Primary Subsystem Usage |
|:---|:---|:---|:---|
| **[NOAA ERSST v5](https://psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html)** | Monthly sea surface temperature (SST) anomalies & derived Niño 3.4 index | OPeNDAP Protocol | El Niño Forecaster (TFT / LSTM) |
| **[OpenAQ v3](https://openaq.org/)** | Spatiotemporal $\text{PM}_{2.5}$ measurements from monitoring stations (default: India and the United States) | REST API (API key authenticated) | Pollution Dispersion (A3TGCN) |
| **[NASA GISTEMP v4](https://data.giss.nasa.gov/gistemp/)** | Long-term global surface temperature anomaly records (1880–present) | Direct HTTPS Download | Climate Feedback (change-points) |
| **[Climate TRACE v7](https://climatetrace.org/)** | Multi-sector greenhouse gas emissions inventories ($\text{CO}_2$, $\text{CH}_4$, $\text{N}_2\text{O}$) | REST API | Climate Feedback (emissions regression) |
| **[WeatherBench 2 / ERA5](https://weatherbench2.readthedocs.io/)** | 6-hourly 2m temperature ($T_{2\text{m}}$) & total precipitation ($5.625^\circ$ equirectangular grid) | Google Cloud Storage Zarr | Neural Weather Model (ST-GNN) |

All ingested data are processed and stored using high-performance Parquet format for bulk time-series data and JSON for metadata tracking.

---

## 4. TECHNICAL STACK AND SYSTEM REQUIREMENTS

### 4.1 Technical Stack

| System Layer | Libraries & Technologies |
|:---|:---|
| **Deep Learning** | PyTorch 2.12, torch-geometric 2.8, pytorch-forecasting 1.8 |
| **Statistical Modeling** | NumPy, statsmodels, PyMC 6, ArviZ 1.2 |
| **Data Engineering** | xarray 2026.7, pandas 3, pyarrow 25, zarr 3, gcsfs |
| **Configuration** | Pydantic-Settings v2, PyYAML |
| **Logging / Artifacts** | Loguru, CSV metrics logger |

### 4.2 Hardware Compatibility
- **Hardware Acceleration:** Supports NVIDIA CUDA, AMD ROCm 6.1 (`HSA_OVERRIDE_GFX_VERSION=10.3.0`), and Intel NPU via OpenVINO; runs on CPU when no accelerator is present.
- **Resource Footprint:** Trains on a single workstation; the main models take about 2.2 hours CPU-only.

### 4.3 Build and Workflow Automation
The platform includes an automated command-line workflow for environment setup, data ingestion, training, artifact generation, and document compilation:

```bash
make install                        # Install environment and dependencies
make download-data                  # Ingest ERSST, OpenAQ, GISTEMP, and Climate TRACE data
make download-nwm-data              # Ingest ERA5 WeatherBench 2 Zarr data
make train                          # Train all four model families
make gen-eval-artifacts             # Generate evaluation plots and summary metrics
make save-run                       # Snapshot checkpoints and evaluation artifacts
make sync-paper-artifacts           # Point the paper at the newest snapshot
make gen-research-paper-pdf         # Compile research paper PDF
```

---

## 5. EXPECTED OUTCOMES AND EVALUATION STRATEGY

The framework will be evaluated against standardized meteorological benchmarks and open environmental observation datasets:

### 5.1 Evaluation Framework and Target Metrics

| Model / Subsystem | Primary Evaluation Metrics | Target Performance Criteria | Evaluation Dataset |
|:---|:---|:---|:---|
| **El Niño Forecaster (TFT)** | Skill Score, RMSE, Anomaly Correlation Coefficient (ACC) | Skill Score $> 0.85$; RMSE $< 0.40^\circ\text{C}$ over 12-month lead horizon | NOAA ERSST v5 (1950–present) |
| **Pollution ST-GNN (A3TGCN)** | $F_1$ Score, Precision, Recall, Station RMSE | $F_1 > 0.85$ on AQI exceedance alerts (AQI $> 100$, i.e. $\text{PM}_{2.5} \ge 35.5\,\mu\text{g/m}^3$), above persistence and always-alert baselines | OpenAQ Sensor Network Telemetry |
| **Neural Weather Model (ST-GNN)** | Latitude-Weighted Validation MSE, Convergence Rate | $>35\%$ forecast error reduction relative to unweighted initialization | ERA5 WeatherBench 2 Reanalysis ($5.625^\circ$) |
| **Climate Feedback (Change-Points)** | Number and timing of warming-rate change-points, segment warming rates | Change-points selected by BIC with autocorrelated errors; recovery of known breaks in synthetic series | NASA GISTEMP v4 Surface Anomaly Records |

---

## 6. CONCLUSION

ClimatePulse establishes a novel, highly efficient paradigm for multi-modal climate forecasting. By unifying long-horizon ocean-atmosphere forecasting, local sensor-network pollution transport, and global spherical neural weather modeling into a single reproducible open-source platform, the project aims to deliver competitive predictive performance while maintaining a low computational footprint.
