"""Evaluation for climate feedback models."""

from __future__ import annotations

import numpy as np
from loguru import logger


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "") -> dict:
    """Compute regression evaluation metrics."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    logger.info(f"{model_name} regression: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
    return {"r_squared": float(r2), "rmse": rmse, "mae": mae}
