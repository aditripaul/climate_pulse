# ClimatePulse Model Evaluation Metrics

This document summarizes the performance metrics generated during the evaluation phase of the trained forecasting and regression models.

---

## 1. El Niño Forecasting (SST Anomaly)

| Model | RMSE (°C) | MAE (°C) | Anomaly Correlation (ACC) | Skill Score |
| :--- | :---: | :---: | :---: | :---: |
| **Temporal Fusion Transformer (TFT)** | 0.1066 | 0.0838 | 0.9943 | 0.9918 |
| **LSTM Baseline** | 0.8615 | 0.6656 | 0.5051 | 0.3619 |
| **Persistence Baseline** | 0.9800 | 0.6845 | 0.4766 | 0.3043 |

*TFT outperforms baseline forecasting architectures across all prediction lead windows.*

---

## 2. Spatio-Temporal Pollution Dispersion

| Metric | Value (Normalized PM2.5 AQI) |
| :--- | :---: |
| **Root Mean Square Error (RMSE)** | 0.1571 |
| **Mean Absolute Error (MAE)** | 0.0464 |
| **F1 Exceedance Rate (AQI > 100)** | 0.9750 |
| **Precision** | 0.9677 |
| **Recall** | 0.9825 |

---

## 3. Neural Weather Model (NWM)

| Metric | Value (MSE) |
| :--- | :---: |
| **Best Validation Loss** | 0.1182 |
| **Final Training Loss** | 0.1035 |

---

## 4. Climate Feedback Change-Points

* **Detected Breaks**: 0
* **Change-Point Years**: 
