# ClimatePulse Model Evaluation Metrics

This document summarizes the performance metrics generated during the evaluation phase of the trained forecasting and regression models.

---

## 1. El Niño Forecasting (SST Anomaly)

| Model | RMSE (°C) | MAE (°C) | Anomaly Correlation (ACC) | Skill Score |
| :--- | :---: | :---: | :---: | :---: |
| **Temporal Fusion Transformer (TFT)** | 0.1115 | 0.0883 | 0.9937 | 0.9910 |
| **LSTM Baseline** | 0.8758 | 0.6833 | 0.4988 | 0.3407 |
| **Persistence Baseline** | 0.9800 | 0.6845 | 0.4766 | 0.3043 |

*TFT outperforms baseline forecasting architectures across all prediction lead windows.*

---

## 2. Spatio-Temporal Pollution Dispersion

| Metric | Value (Normalized PM2.5 AQI) |
| :--- | :---: |
| **Root Mean Square Error (RMSE)** | 0.1762 |
| **Mean Absolute Error (MAE)** | 0.0588 |
| **F1 Exceedance Rate (AQI > 100)** | 0.9669 |
| **Precision** | 0.9566 |
| **Recall** | 0.9774 |

---

## 3. Climate Feedback Change-Points

* **Detected Breaks**: 0
* **Change-Point Years**:
