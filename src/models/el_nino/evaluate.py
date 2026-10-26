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
    baseline_pred: np.ndarray | None = None,
) -> dict:
    """
    Compute full evaluation metrics for a forecast.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.
        model_name: Name for logging.
        baseline_pred: Reference forecast for the skill score, aligned with y_true.
            Pass the true persistence forecast (last observed value per window);
            the default constant baseline is only meaningful for a single window.

    Returns:
        Dictionary of metric name → value.
    """
    metrics = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "acc": anomaly_correlation(y_true, y_pred),
        "skill_score": skill_score(y_true, y_pred, baseline_pred),
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


def quantile_calibration(
    y_true: np.ndarray,
    quantile_preds: dict[float, np.ndarray],
) -> pd.DataFrame:
    """
    Observed frequency of outcomes falling below each predicted quantile.

    A calibrated forecast has observed frequency equal to the quantile level.
    """
    rows = [
        {"quantile": q, "observed_frequency": float(np.mean(y_true <= pred))}
        for q, pred in sorted(quantile_preds.items())
    ]
    return pd.DataFrame(rows)


def elnino_test_predictions(tft_model, test_ds, lstm_model, lstm_test_data: np.ndarray) -> dict:
    """
    Run the TFT and LSTM over the test period and align them window by window.

    Returns a dict of [n_windows, n_leads] arrays: 'actual', 'tft', 'tft_lower',
    'tft_upper', 'lstm', 'persistence', plus 'first_target_time_idx' [n_windows].
    """
    import torch

    from models.el_nino.dataset import LSTMDataset
    from models.el_nino.tft_model import predict_tft

    tft_res = predict_tft(tft_model, test_ds)

    dataloader = test_ds.to_dataloader(train=False, batch_size=128, num_workers=0)
    actuals, last_obs, first_idx = [], [], []
    for x, y in dataloader:
        actuals.append(y[0])
        last_obs.append(x["encoder_target"][:, -1:])
        first_idx.append(x["decoder_time_idx"][:, 0])
    actual = torch.cat(actuals).numpy()
    last_y = torch.cat(last_obs).numpy()
    first_target = torch.cat(first_idx).numpy()

    lstm_loader = torch.utils.data.DataLoader(
        LSTMDataset(lstm_test_data.reshape(-1, 1)), batch_size=128, shuffle=False
    )
    lstm_model.eval()
    lstm_preds, lstm_actuals = [], []
    with torch.no_grad():
        for x_lstm, y_lstm in lstm_loader:
            lstm_preds.append(lstm_model(x_lstm))
            lstm_actuals.append(y_lstm)
    lstm_pred = torch.cat(lstm_preds).numpy()
    lstm_actual = torch.cat(lstm_actuals).squeeze(-1).numpy()

    n = min(len(actual), len(tft_res["predictions"]), len(lstm_pred))
    n_leads = actual.shape[1]
    if not np.allclose(lstm_actual[:n, :n_leads], actual[:n], atol=1e-4):
        raise ValueError("TFT and LSTM test windows are misaligned; skill comparison invalid")

    return {
        "actual": actual[:n],
        "tft": tft_res["predictions"][:n],
        "tft_lower": tft_res["lower"][:n],
        "tft_upper": tft_res["upper"][:n],
        "lstm": lstm_pred[:n, :n_leads],
        "persistence": np.repeat(last_y[:n, 0:1], n_leads, axis=1),
        "first_target_time_idx": first_target[:n],
    }


def elnino_metrics(p: dict) -> dict:
    """Pooled and per-lead metrics for aligned El Nino test predictions."""
    actual = p["actual"].ravel()
    baseline = p["persistence"].ravel()

    def _model(name: str, pred: np.ndarray) -> dict:
        m = evaluate_forecast(actual, pred.ravel(), name, baseline_pred=baseline)
        m["bias"] = float(np.mean(pred.ravel() - actual))
        return m

    calib = quantile_calibration(
        actual, {0.1: p["tft_lower"].ravel(), 0.5: p["tft"].ravel(), 0.9: p["tft_upper"].ravel()}
    )
    coverage = reliability_diagram(actual, p["tft_lower"].ravel(), p["tft_upper"].ravel())
    coverage["frac_above_upper"] = float(np.mean(actual > p["tft_upper"].ravel()))
    coverage["frac_below_lower"] = float(np.mean(actual < p["tft_lower"].ravel()))

    def _lead_rmse(pred: np.ndarray) -> list[float]:
        return np.sqrt(np.mean((p["actual"] - pred) ** 2, axis=0)).tolist()

    return {
        "tft": _model("TFT", p["tft"]),
        "lstm": _model("LSTM", p["lstm"]),
        "persistence": _model("Persistence", p["persistence"]),
        "calibration": coverage,
        "quantile_calibration": calib.to_dict(orient="records"),
        "lead_rmse": {
            "tft": _lead_rmse(p["tft"]),
            "lstm": _lead_rmse(p["lstm"]),
            "persistence": _lead_rmse(p["persistence"]),
        },
    }
