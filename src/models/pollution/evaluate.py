"""Evaluation metrics for the Pollution ST-GNN."""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger


def evaluate_pollution_model(model, test_signal, threshold: float = 100.0) -> dict:
    """
    Evaluate the pollution model on a test signal.

    Args:
        model: Trained PollutionSTGNN.
        test_signal: Test temporal signal.
        threshold: AQI threshold for classification metrics.

    Returns:
        Dictionary with RMSE, MAE, F1, per-node errors.
    """
    device = next(model.parameters()).device
    model.eval()

    all_preds = []
    all_true = []

    with torch.no_grad():
        for snapshot in test_signal:
            x = torch.FloatTensor(snapshot.x).to(device)
            edge_index = torch.LongTensor(snapshot.edge_index).to(device)
            edge_weight = (
                torch.FloatTensor(snapshot.edge_attr).to(device)
                if snapshot.edge_attr is not None
                else None
            )
            y = snapshot.y

            pred = model(x, edge_index, edge_weight).squeeze().cpu().numpy()
            all_preds.append(pred)
            all_true.append(y)

    preds = np.stack(all_preds)
    trues = np.stack(all_true)

    # Overall metrics
    rmse_val = float(np.sqrt(np.mean((preds - trues) ** 2)))
    mae_val = float(np.mean(np.abs(preds - trues)))

    # Threshold classification (AQI exceedance)
    pred_exceed = preds > threshold
    true_exceed = trues > threshold
    tp = float(np.sum(pred_exceed & true_exceed))
    fp = float(np.sum(pred_exceed & ~true_exceed))
    fn = float(np.sum(~pred_exceed & true_exceed))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)

    # Per-node RMSE
    node_rmse = np.sqrt(np.mean((preds - trues) ** 2, axis=0))

    logger.info(f"Pollution eval: RMSE={rmse_val:.4f}, MAE={mae_val:.4f}, F1={f1:.4f}")

    return {
        "rmse": rmse_val,
        "mae": mae_val,
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "node_rmse": node_rmse.tolist(),
        "predictions": preds,
        "targets": trues,
    }
