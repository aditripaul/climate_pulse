"""Evaluation metrics for the Pollution ST-GNN."""

from __future__ import annotations

import warnings

import numpy as np
import torch
from loguru import logger

# US EPA: AQI > 100 ("Unhealthy for Sensitive Groups" and above) starts at 35.5 ug/m3
AQI_100_PM25_UGM3 = 35.5


def _exceedance_scores(pred_exceed: np.ndarray, true_exceed: np.ndarray) -> dict:
    tp = int(np.sum(pred_exceed & true_exceed))
    fp = int(np.sum(pred_exceed & ~true_exceed))
    fn = int(np.sum(~pred_exceed & true_exceed))
    tn = int(np.sum(~pred_exceed & ~true_exceed))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


def evaluate_pollution_model(
    model,
    test_signal,
    norm_stats: dict,
    threshold_ugm3: float = AQI_100_PM25_UGM3,
) -> dict:
    """
    Evaluate the pollution model on a test signal, in physical units.

    Only station-days with a reading count. Predictions are compared with two
    baselines: persistence (the last input day, with the training mean where that
    day is missing) and an always-alert exceedance classifier.

    Args:
        model: Trained PollutionSTGNN.
        test_signal: Windowed test signal; missing targets are NaN.
        norm_stats: {"mean", "std"} used to normalize the data (training split).
        threshold_ugm3: Exceedance threshold in ug/m3 (default: AQI 100).

    Returns:
        Dictionary with RMSE/MAE in ug/m3, exceedance scores, baseline scores,
        per-node RMSE and the denormalized predictions/targets (NaN where missing).
    """
    device = next(model.parameters()).device
    model.eval()

    all_preds, all_true, all_persist = [], [], []

    with torch.no_grad():
        for snapshot in test_signal:
            x = torch.FloatTensor(snapshot.x).to(device)
            edge_index = torch.LongTensor(snapshot.edge_index).to(device)
            edge_weight = (
                torch.FloatTensor(snapshot.edge_attr).to(device)
                if snapshot.edge_attr is not None
                else None
            )
            pred = model(x, edge_index, edge_weight).squeeze().cpu().numpy()
            all_preds.append(pred)
            all_true.append(snapshot.y)
            all_persist.append(snapshot.x[:, 0, -1])  # last input day, first channel

    mean, std = norm_stats["mean"], norm_stats["std"]
    preds = np.stack(all_preds) * std + mean
    trues = np.stack(all_true) * std + mean  # NaN stays NaN
    persist = np.stack(all_persist) * std + mean

    observed = np.isfinite(trues)
    p, t, b = preds[observed], trues[observed], persist[observed]

    def _rmse(a: np.ndarray, c: np.ndarray) -> float:
        return float(np.sqrt(np.mean((a - c) ** 2)))

    true_exceed = t >= threshold_ugm3
    model_scores = _exceedance_scores(p >= threshold_ugm3, true_exceed)
    persist_scores = _exceedance_scores(b >= threshold_ugm3, true_exceed)
    always_scores = _exceedance_scores(np.ones_like(true_exceed), true_exceed)

    # Per-node RMSE over that node's observed days (NaN if never observed)
    sq_err = np.where(observed, (preds - np.nan_to_num(trues)) ** 2, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # nodes never observed -> NaN
        node_rmse = np.sqrt(np.nanmean(sq_err, axis=0))

    metrics = {
        "rmse_ugm3": _rmse(p, t),
        "mae_ugm3": float(np.mean(np.abs(p - t))),
        "persistence_rmse_ugm3": _rmse(b, t),
        "threshold_ugm3": threshold_ugm3,
        "n_observed": int(observed.sum()),
        "n_total": int(observed.size),
        "exceedance_rate": float(true_exceed.mean()),
        "f1": model_scores["f1"],
        "precision": model_scores["precision"],
        "recall": model_scores["recall"],
        "confusion": {k: model_scores[k] for k in ("tp", "fp", "fn", "tn")},
        "persistence_f1": persist_scores["f1"],
        "always_alert_f1": always_scores["f1"],
        "node_rmse": [None if not np.isfinite(v) else float(v) for v in node_rmse],
        "predictions": np.where(observed, preds, np.nan),
        "targets": trues,
    }

    logger.info(
        f"Pollution eval ({metrics['n_observed']}/{metrics['n_total']} observed): "
        f"RMSE={metrics['rmse_ugm3']:.2f} ug/m3 (persistence {metrics['persistence_rmse_ugm3']:.2f}), "
        f"F1={metrics['f1']:.4f} (persistence {metrics['persistence_f1']:.4f}, "
        f"always-alert {metrics['always_alert_f1']:.4f})"
    )
    return metrics
