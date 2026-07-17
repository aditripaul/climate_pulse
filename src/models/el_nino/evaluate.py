"""
Evaluation metrics for El Niño forecasting models.

Computes RMSE, MAE, Anomaly Correlation Coefficient (ACC),
and skill scores against a climatological persistence baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def anomaly_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Anomaly Correlation Coefficient (ACC).

    Measures pattern correlation between predicted and observed anomalies.
    Values near 1 indicate skillful forecasts.
    """
    y_true_anom = y_true - np.mean(y_true)
    y_pred_anom = y_pred - np.mean(y_pred)
    numerator = np.sum(y_true_anom * y_pred_anom)
    denominator = np.sqrt(np.sum(y_true_anom**2) * np.sum(y_pred_anom**2))
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def persistence_baseline(y_true: np.ndarray) -> np.ndarray:
    """
    Climatological persistence baseline.

    Forecasts that the anomaly remains constant at the last observed value.
    """
    return np.full_like(y_true, y_true[0] if len(y_true) > 0 else 0.0)


def skill_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_pred: np.ndarray | None = None,
) -> float:
    """
    Compute skill score relative to a baseline (default: persistence).

    SS = 1 - MSE(model) / MSE(baseline)
    SS > 0 means the model is better than baseline.
    """
    if baseline_pred is None:
        baseline_pred = persistence_baseline(y_true)

    mse_model = np.mean((y_true - y_pred) ** 2)
    mse_baseline = np.mean((y_true - baseline_pred) ** 2)

    if mse_baseline == 0:
        return 0.0

    return float(1 - mse_model / mse_baseline)


def evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model",
) -> dict:
    """
    Compute full evaluation metrics for a forecast.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.
        model_name: Name for logging.

    Returns:
        Dictionary of metric name → value.
    """
    metrics = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "acc": anomaly_correlation(y_true, y_pred),
        "skill_score": skill_score(y_true, y_pred),
    }

    logger.info(
        f"{model_name} evaluation: "
        f"RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, "
        f"ACC={metrics['acc']:.4f}, SS={metrics['skill_score']:.4f}"
    )

    return metrics


def compare_models(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Compare multiple models side-by-side.

    Args:
        y_true: Observed values.
        predictions: Dict of model_name → predicted values.

    Returns:
        DataFrame with metrics per model.
    """
    rows = []
    for name, y_pred in predictions.items():
        metrics = evaluate_forecast(y_true, y_pred, model_name=name)
        metrics["model"] = name
        rows.append(metrics)

    df = pd.DataFrame(rows).set_index("model")
    return df


def reliability_diagram(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute reliability diagram data for probabilistic forecasts.

    Checks whether predicted confidence intervals have correct coverage.

    Args:
        y_true: Observed values.
        lower: Lower bound (e.g., 10th percentile).
        upper: Upper bound (e.g., 90th percentile).
        n_bins: Number of bins for calibration.

    Returns:
        Dictionary with coverage statistics.
    """
    # Overall coverage (should be ~80% for 10th-90th interval)
    in_interval = (y_true >= lower) & (y_true <= upper)
    coverage = float(np.mean(in_interval))

    # Interval width
    avg_width = float(np.mean(upper - lower))

    return {
        "target_coverage": 0.8,
        "actual_coverage": coverage,
        "avg_interval_width": avg_width,
        "is_calibrated": abs(coverage - 0.8) < 0.1,
    }
