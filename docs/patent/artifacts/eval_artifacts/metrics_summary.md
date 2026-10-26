# ClimatePulse Model Evaluation Metrics

This document summarizes the performance metrics generated during the evaluation phase of the trained forecasting and regression models.

---

## 1. El Niño Forecasting (SST Anomaly)

| Model | RMSE (°C) | MAE (°C) | Anomaly Correlation (ACC) | Skill Score |
| :--- | :---: | :---: | :---: | :---: |
| **Temporal Fusion Transformer (TFT)** | 0.9362 | 0.7838 | 0.7268 | 0.1778 |
| **LSTM Baseline** | 0.8705 | 0.6734 | 0.5099 | 0.2890 |
| **Persistence Baseline** | 1.0324 | 0.7511 | 0.3979 | 0.0000 |

*Skill Score = 1 − MSE(model) / MSE(persistence), where persistence carries the last observed value across all 12 leads.*

---

## 2. Spatio-Temporal Pollution Dispersion

| Metric | Value (Normalized PM2.5 AQI) |
| :--- | :---: |
| **Root Mean Square Error (RMSE)** | 0.1596 |
| **Mean Absolute Error (MAE)** | 0.0478 |
| **F1 Exceedance Rate (AQI > 100)** | 0.9727 |
| **Precision** | 0.9624 |
| **Recall** | 0.9832 |

---

## 3. Neural Weather Model (NWM)

| Metric | Value (MSE) |
| :--- | :---: |
| **Best Validation Loss** | 0.1185 |
| **Final Training Loss** | 0.1053 |

---

## 4. Climate Feedback Change-Points

* **Method**: continuous piecewise-linear trend, AR(1) errors, K selected by BIC
* **Detected Breaks**: 1
* **Change-Point Years**: 1976
* **Segment Warming Rates (°C/decade)**: 0.034, 0.212
