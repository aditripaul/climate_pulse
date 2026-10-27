# ClimatePulse Model Evaluation Metrics

This document summarizes the performance metrics generated during the evaluation phase of the trained forecasting and regression models. "n/a" marks a result that could not be computed.

---

## 1. El Niño Forecasting (SST Anomaly)

### Main run (figures)

| Model | RMSE (°C) | MAE (°C) | Anomaly Correlation (ACC) | Skill Score | Bias (°C) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Temporal Fusion Transformer (TFT)** | 0.6838 | 0.5237 | 0.7041 | 0.5644 | -0.2066 |
| **LSTM Baseline** | 0.7984 | 0.6133 | 0.5569 | 0.4063 | -0.2364 |
| **Persistence Baseline** | 1.0361 | 0.7510 | 0.3835 | 0.0000 | 0.0617 |

*Skill Score = 1 − MSE(model) / MSE(persistence), where persistence carries the last observed value across all 12 leads. TFT 80 % interval coverage: 0.757.*

### Across seeds

Mean ± standard deviation over 5 seeds (TFT variant selected by mean validation loss).

| Model | Variant | Val loss | RMSE (°C) | ACC | Skill Score | Bias (°C) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| tft | with_year | 0.331 ± 0.036 | 0.838 ± 0.070 | 0.483 ± 0.193 | 0.341 ± 0.106 | -0.242 ± 0.141 |
| tft | without_year | 0.283 ± 0.016 | 0.782 ± 0.075 | 0.645 ± 0.023 | 0.427 ± 0.113 | -0.321 ± 0.146 |
| lstm | - | 0.342 ± 0.006 | 0.809 ± 0.020 | 0.562 ± 0.035 | 0.390 ± 0.029 | -0.268 ± 0.078 |
| persistence | - | n/a | 1.036 ± 0.000 | 0.383 ± 0.000 | 0.000 ± 0.000 | 0.062 ± 0.000 |

---

## 2. Spatio-Temporal Pollution Dispersion (next-day PM2.5)

| Metric | Model | Persistence | Always alert |
| :--- | :---: | :---: | :---: |
| **RMSE (µg/m³)** | 46.58 | 51.97 | — |
| **F1, PM2.5 ≥ 35.5 µg/m³ (AQI > 100)** | 0.8276 | 0.8806 | 0.8276 |
| **Precision / Recall** | 0.7059 / 1.0000 | — | — |

*Observed station-days: 85 of 1500; exceedance rate 0.706. Missing readings are excluded.*

---

## 3. Neural Weather Model (NWM)

| Metric | Value |
| :--- | :---: |
| **Best Validation Loss (MSE, normalized)** | 0.1173 |
| **Final Training Loss (MSE, normalized)** | 0.1070 |
| **Test 2 m Temperature RMSE (K, area-weighted)** | 1.214 |
| **Persistence 2 m Temperature RMSE (K)** | 2.457 |

### Ablation

| Variant | Conditioning | Area-weighted loss | Epochs | Test T2m RMSE (K) | Test precip RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: |
| observed | observed | True | 81 | 1.214 | 6.821e-04 |
| forecast | forecast | True | 85 | 1.335 | 6.828e-04 |
| unconditioned | none | True | 79 | 1.195 | 6.801e-04 |
| unweighted | observed | False | 85 | 1.164 | 6.786e-04 |
| persistence | — | — | — | 2.457 | 9.032e-04 |

---

## 4. Climate Feedback Change-Points

* **Method**: continuous piecewise-linear trend, AR(1) errors, K selected by BIC
* **Detected Breaks**: 1
* **Change-Point Years**: 1976
* **Segment Warming Rates (°C/decade)**: 0.035, 0.211
