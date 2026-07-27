# PROJECT PROPOSAL

# ClimatePulse: AI-Powered Global Climate Forecasting and Environmental Telemetry Framework

**Project Lead / Author:** ClimatePulse Research Team  
**Repository Reference:** `git:5bbda56`  
**Target Legal / IP Jurisdiction:** IPINDIA (Controller General of Patents, Designs and Trade Marks) / PCT  
**Document Status:** Publication & Proposal Ready  
**Date:** July 2026  

---

## EXECUTIVE SUMMARY

Global climate change presents an urgent need for fast, high-precision, and multi-scale environmental forecasting tools. Traditional Numerical Weather Prediction (NWP) systems are computationally prohibitive, requiring supercomputers running for hours per forecast cycle. Conversely, existing data-driven AI weather models operate in disciplinary isolation—predicting short-term atmospheric states without integrating long-range macro-climatic forcings (such as El Niño-Southern Oscillation, or ENSO) or local urban pollution transport.

**ClimatePulse** solves this problem by introducing a unified, modular, open-source Python framework that bridges four core climate modeling dimensions:
1. **El Niño Forecasting:** A Temporal Fusion Transformer (TFT) providing probabilistic multi-horizon forecasting of the Niño 3.4 index up to 60 months ahead.
2. **Pollution Dispersion:** An Attention Temporal Graph Convolutional Network (A3TGCN) forecasting $\text{PM}_{2.5}$ concentration across a 75-station global sensor network.
3. **Neural Weather Modeling (NWM):** A Graph Neural Network (ST-GNN) operating on a spherical Fibonacci golden-angle latent mesh ($N_{\text{mesh}} = 2,000$) with a message-level ENSO conditioning pathway.
4. **Climate Feedback Attribution:** A Bayesian hierarchical piecewise regression model detecting structural regime shifts in 912 months of global temperature anomalies.

Empirical evaluation on ERA5 reanalysis and OpenAQ sensor data demonstrates state-of-the-art performance: the NWM achieves a validation Mean Squared Error (MSE) of **0.1182** (a **39.5% improvement** over initialization), the ENSO forecaster achieves a skill score of **0.9051** (ACC **0.9449**, RMSE **0.3505°C**), and the pollution model achieves an $F_1$ score of **0.9750** on pollution exceedance classification.

To protect the core innovations, a consolidated single patent specification ([`docs/patent/patent_application.md`](file:///home/arp/workspace/grokking.workspace/climate_pulse.git/docs/patent/patent_application.md)) has been drafted for filing with IPINDIA and the Patent Cooperation Treaty (PCT).

---

## 1. PROBLEM STATEMENT AND MOTIVATION

### 1.1 Limitations of Classical Numerical Weather Prediction (NWP)
Systems such as ECMWF's Integrated Forecasting System (IFS) rely on numerical integration of primitive fluid dynamic equations over high-resolution grids. While accurate, their computational latency makes real-time scenario simulation, agricultural planning, and civil protection response difficult.

### 1.2 Defects in Existing Data-Driven AI Weather Models
Recent neural weather models (GraphCast, FourCastNet, Pangu-Weather, FuXi) have reduced forecast latency by orders of magnitude. However, they suffer from four major technical drawbacks:

1. **Equirectangular Polar Distortion:** Models operating on regular latitude-longitude grids apply spatially invariant convolutions across unequal physical surface areas. Equatorial cells subtend up to 8.6$\times$ more surface area than high-latitude cells, causing unweighted models to over-allocate computational capacity to polar regions.
2. **Single-Task Unconditioned Architecture:** Prior models map atmospheric state $x_t \to x_{t+1}$ without incorporating large-scale ocean-atmosphere forcing signals (e.g., ENSO phase) into latent message passing. They treat El Niño and La Niña initializations identically.
3. **Subdivision Rigidity of Icosahedral Meshes:** Prior graph-based models (GraphCast, Keisler) rely on icosahedral subdivision, locking mesh resolutions to discrete node counts ($20, 80, 320, 1280, 5120$).
4. **Disciplinary Silos:** No unified platform connects global climate oscillations, local air quality dispersion, and global atmospheric field forecasting in a single reproducible framework.

---

## 2. TECHNICAL ARCHITECTURE AND MODEL SUBSYSTEMS

ClimatePulse resolves these challenges through a modular Python platform built around a unified data ingestion pipeline, Parquet/JSON storage, and four specialized deep learning modules:

```
+-----------------------------------------------------------------------------------+
|                           CLIMATEPULSE DATA INGESTION                             |
|              (ERA5 Reanalysis, ERSST v5 Niño 3.4, OpenAQ Sensor Telemetry)        |
+-----------------------------------------------------------------------------------+
                                         |
     +-----------------------------------+-----------------------------------+
     |                                   |                                   |
     v                                   v                                   v
+-----------------------+     +-----------------------+     +-----------------------+
|  ENSO FORECASTER      |     | POLLUTION DISPERSION  |     | CLIMATE FEEDBACK      |
|  (TFT Model)          |     | (A3TGCN Model)        |     | (Bayesian Regression) |
|  Outputs: Niño 3.4    |     | Outputs: PM2.5 / NO2  |     | Outputs: Changepoints |
+-----------------------+     +-----------------------+     +-----------------------+
     | (Conditioning Signal: Niño 3.4 Index)
     v
+-----------------------------------------------------------------------------------+
|                           NEURAL WEATHER MODEL (NWM)                              |
|  Stage 1: Grid Embedding        (Linear + LayerNorm + GELU) -> [B*N_grid, d]      |
|  Stage 2: Encoder G2M           (Bipartite GraphSAGE)      -> [B*N_mesh, d]      |
|  Stage 3: Processor M2M x L     (SAGEConv + ENSO Cond.)    -> [B*N_mesh, d]      |
|  Stage 4: Decoder M2G           (Bipartite GraphSAGE + Skip)-> [B*N_grid, d]      |
|  Stage 5: Output Projection     (Linear Projection)        -> [B*N_grid, C*W]    |
+-----------------------------------------------------------------------------------+
```

### 2.1 El Niño Forecaster (TFT & BiLSTM)
- **Architecture:** Temporal Fusion Transformer (TFT) utilizing Variable Selection Networks, Gated Residual Networks (GRNs), multi-head self-attention, and quantile regression heads at $\tau \in \{0.1, 0.5, 0.9\}$.
- **Input Data:** NOAA ERSST v5 monthly sea surface temperature anomalies (1950–present).
- **Baseline:** Bidirectional LSTM and persistence forecasting for benchmark comparisons.

### 2.2 Pollution Dispersion Module (A3TGCN)
- **Architecture:** Attention-based Spatial-Temporal Graph Convolutional Network (A3TGCN) combining spatial Chebyshev graph convolutions over a 75-station OpenAQ sensor network graph with Gated Recurrent Units (GRUs).
- **Task:** Multi-step $\text{PM}_{2.5}$ concentration forecasting and AQI exceedance classification ($>100\,\mu\text{g/m}^3$).

### 2.3 Neural Weather Model (ST-GNN)
- **Fibonacci Latent Mesh:** A spherical computational mesh constructed via a golden-angle spiral lattice:
  $$\phi_{\text{golden}} = \pi (3 - \sqrt{5}), \quad y_i = 1 - \frac{2i}{N_{\text{mesh}}-1}, \quad r_i = \sqrt{\max(1-y_i^2, 0)}, \quad \theta_i = \phi_{\text{golden}} \cdot i$$
  yielding $N_{\text{mesh}} = 2,000$ near-uniformly distributed nodes on a unit sphere.
- **Static Positional Features:** $[\sin(\phi), \cos(\phi), \sin(\lambda), \cos(\lambda)] \in \mathbb{R}^4$ concatenated to grid node inputs to ensure dateline continuity and polar stability.
- **Message-Only ENSO Conditioning:** The scalar Niño 3.4 index is projected via a 2-layer MLP to $\mathbf{c} \in \mathbb{R}^d$, broadcast to $\mathbf{C} \in \mathbb{R}^{N_{\text{mesh}} \times d}$, and **additively injected into the message tensor only** of each Mesh-to-Mesh (M2M) processor layer before the residual connection:
  $$\mathbf{msg}^{(l)} = \text{GELU}(\text{LayerNorm}(\text{SAGEConv}(\mathbf{h}_m^{(l)}, \text{M2M\_edges}) + \mathbf{C}))$$
  $$\mathbf{h}_m^{(l+1)} = \mathbf{h}_m^{(l)} + \mathbf{msg}^{(l)}$$
  This prevents the macro-climatic signal from being added twice per layer through the residual path.
- **Hardware Acceleration:** Vectorized batch-offset edge expansion using PyTorch `repeat_interleave`, reducing GPU kernel launches from $O(B)$ to $O(1)$.
- **Latitude-Cosine Weighted Objective:** Training loss weighted by physical cell area $w_i = \cos(\phi_i) / \text{mean}(\cos(\phi))$:
  $$\mathcal{L} = \frac{1}{B} \sum_{b=1}^B \frac{1}{N_{\text{grid}}} \sum_{i=1}^{N_{\text{grid}}} w_i \sum_{c} (\hat{y}_{b,i,c} - y_{b,i,c})^2$$

### 2.4 Climate Feedback Attribution (Bayesian Regression)
- **Architecture:** Bayesian hierarchical piecewise regression implemented in PyMC.
- **Task:** Structural regime-shift (changepoint) detection across 912 months (1950–2025) of NASA GISTEMP temperature anomalies.

---

## 3. EXPERIMENTAL RESULTS AND BENCHMARKS

All models were trained and evaluated on historical reanalysis and sensor datasets. Empirical results are stored in `runtime/eval_artifacts/run_results/summary_metrics.json` and summarized below:

### 3.1 Quantitative Performance Summary

| Model / Subsystem | Primary Metric | Value | Baseline / Target Comparison | Key Takeaway |
|:---|:---|:---:|:---:|:---|
| **El Niño Forecaster (TFT)** | Skill Score | **0.9051** | Persistence: 0.1737 \| LSTM: 0.1450 | **+421% improvement** over persistence baseline |
| | RMSE | **0.3505°C** | Persistence: 1.0344°C \| LSTM: 0.9606°C | Sub-0.4°C error over 12-month horizon |
| | ACC | **0.9449** | Persistence: 0.4008 \| LSTM: -0.0399 | High anomaly correlation coefficient |
| **Pollution ST-GNN (A3TGCN)** | $F_1$ Score | **0.9750** | Target: >0.8500 | Outstanding exceedance detection |
| | Precision | **0.9677** | Target: >0.8500 | Low false alarm rate on air quality alerts |
| | Recall | **0.9825** | Target: >0.8500 | Captures 98.25% of true PM2.5 exceedances |
| | Overall RMSE | **0.1571** | Across 75 global stations | High precision spatial transport modeling |
| **Neural Weather Model (ST-GNN)** | Best Val MSE | **0.1182** | Initial Epoch 0 Loss: 0.2938 | **39.5% reduction** in global forecast error |
| | Best Epoch | **78 / 89** | Final Val MSE: 0.1185 | Stable convergence with early stopping |
| | Best Train Loss | **0.1045** | Final Train Loss: 0.1035 | No overfitting observed |
| **Climate Feedback (Bayesian)** | Changepoints | **0** | Evaluated on 912 months | Continuous warming trend with no structural breaks |

---

## 4. INTELLECTUAL PROPERTY & PATENT APPLICATION STRATEGY

To secure exclusive commercial and academic rights for the core inventions, a consolidated Provisional Patent Application draft has been prepared:

- **Document Location:** [`docs/patent/patent_application.md`](file:///home/arp/workspace/grokking.workspace/climate_pulse.git/docs/patent/patent_application.md)
- **Title:** *UNIFIED MULTI-MODAL CLIMATE INFERENCE SYSTEM OPERATING ON A FIBONACCI SPHERICAL LATENT MESH WITH ENSO-CONDITIONED NEURAL WEATHER MODELING*
- **Target Patent Office:** IPINDIA (Controller General of Patents, Designs and Trade Marks) / PCT

### 4.1 Legal & Statutory Alignment for IPINDIA
1. **Section 3(k) Protection (CRI Guidelines):** Indian patent law excludes "computer programs per se or algorithms". The draft frames the GNN architecture and Fibonacci mesh as embedded hardware/technical components within a physical Earth telemetry and environmental monitoring system.
2. **Section 10(5) Unity of Invention:** Merging all subsystems into a single application establishes a unified inventive concept centered around the **cross-domain ENSO conditioning pathway**.
3. **Claims Structure (14 Claims Total):**
   - **Claim 1 (System):** Spherical Fibonacci mesh NWM with message-only ENSO conditioning.
   - **Claim 7 (Method):** Vectorized batch-offset edge expansion algorithm for GPU acceleration.
   - **Claim 9 (System):** Integrated multi-modal climate framework.
   - **Claim 13 (Method):** End-to-end multi-modal climate forecasting method.
   - **Claim 14 (Apparatus):** Non-transitory computer-readable storage medium.

---

## 5. HARDWARE, DATA PIPELINE AND SYSTEM REQUIREMENTS

### 5.1 Hardware Compatibility
- **Hardware Acceleration:** Compatible with NVIDIA CUDA, AMD ROCm 7.2 (`HSA_OVERRIDE_GFX_VERSION=10.3.0`), and Intel NPU via OpenVINO.
- **Resource Footprint:** Designed to run training and inference on consumer workstations with $\ge 15\text{ GB}$ VRAM.

### 5.2 Build and Artifact Execution Pipeline
The platform includes an automated `Makefile` workflow for data ingestion, model training, artifact generation, and document compilation:

```bash
make install                        # Install Python 3.12 dependencies and PyG
make download-data                  # Ingest ERSST, OpenAQ, GISTEMP data
make download-nwm-data              # Ingest ERA5 WeatherBench 2 zarr data
make train                          # Train all four model families
make gen-eval-artifacts             # Generate publication plots and summary_metrics.json
make gen-research-paper-pdf         # Compile research paper PDF via XeLaTeX
make gen-patent-application-pdf     # Compile consolidated patent specification PDF via XeLaTeX
make save-run                       # Snapshot model checkpoints and artifacts to runtime/saved/
```

---

## 6. PROJECT ROADMAP AND FUTURE MILESTONES

| Phase | Horizon | Key Objectives |
|:---|:---:|:---|
| **Phase 1 (Completed)** | Q3 2026 | Modular repository structure, 4 model subsystems trained, empirical evaluation pipeline, single consolidated patent draft (`patent_application.md`), automated PDF build pipelines. |
| **Phase 2 (Immediate)** | Q4 2026 | Formal filing of Provisional Patent Application with IPINDIA; submission of research paper to peer-reviewed journal; public release of pretrained model checkpoints. |
| **Phase 3 (Medium-Term)**| Q2 2027 | Scaling Neural Weather Model to $0.25^\circ$ high-resolution ERA5 data; expanding NWM channels to 3D vertical pressure levels ($Z_{500}, U_{850}, V_{850}$); implementing dynamic edge-weighted sensor graphs for pollution dispersion. |
| **Phase 4 (Long-Term)** | Q4 2027 | Deployment of a real-time gRPC / REST climate inference API server for civil protection, agricultural forecasting, and urban environmental planning. |

---

## 7. CONCLUSION

ClimatePulse establishes a novel, highly efficient paradigm for multi-modal climate forecasting. By unifying long-horizon ocean-atmosphere forecasting, local sensor-network pollution transport, and global spherical neural weather modeling into a single reproducible open-source platform, the project delivers state-of-the-art predictive performance while maintaining a low computational footprint.

The consolidated patent application ensures comprehensive intellectual property protection under Indian and international patent law, paving the way for academic leadership and commercial deployment.
