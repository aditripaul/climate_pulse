"""Test-set evaluation for the Neural Weather Model (NWM)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from config import settings
from models.nwm.dataset import ERA5_PATH, NWMDataset, nino_conditioning_path
from models.nwm.graph_builder import SphericalGraphBuilder
from models.nwm.trainer import NWMTrainer


def evaluate_nwm_checkpoint(ckpt_path: str | Path, conditioning: str) -> dict:
    """
    Run a trained NWM over the test period and score it in physical units.

    RMSEs are area-weighted by cos(latitude) over all grid nodes and test samples:
    2 m temperature in K, total precipitation in the dataset's units (m per 6 h).
    Persistence (the last input step carried forward) is reported for reference.

    Returns:
        Dictionary of scalar metrics plus 'preds', 'targets' (normalized,
        [N_test, num_grid, 2]) and the 'dataset'.
    """
    nwm = settings.nwm
    test_ds = NWMDataset(
        ERA5_PATH, nino_conditioning_path(conditioning), mode="test", conditioning=conditioning
    )
    graph_builder = SphericalGraphBuilder(test_ds.lat, test_ds.lon, num_mesh_nodes=nwm.num_mesh_nodes)

    trainer = NWMTrainer(
        grid_in_channels=test_ds.window_size * 2 + 4,
        grid_out_channels=test_ds.prediction_steps * 2,
        hidden_dim=nwm.hidden_dim,
        num_processor_layers=nwm.num_processor_layers,
        learning_rate=nwm.learning_rate,
        weight_decay=nwm.weight_decay,
        edge_grid_to_mesh=graph_builder.build_grid_to_mesh(),
        edge_mesh_to_mesh=graph_builder.build_mesh_to_mesh(),
        edge_mesh_to_grid=graph_builder.build_mesh_to_grid(),
        num_mesh_nodes=nwm.num_mesh_nodes,
    )
    trainer.load_checkpoint(ckpt_path)
    trainer.model.eval()

    preds, targets, persist = [], [], []
    loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    with torch.no_grad():
        for x, cond, y in loader:
            preds.append(trainer._forward(x.to(trainer.device), cond.to(trainer.device)).cpu().numpy())
            targets.append(y.numpy())
            w = test_ds.window_size
            # Last input step of each channel: inputs are [T2m x W, precip x W, static x 4]
            persist.append(np.stack([x[..., w - 1].numpy(), x[..., 2 * w - 1].numpy()], axis=-1))
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    persist = np.concatenate(persist)

    weights = test_ds.lat_weights / test_ds.lat_weights.mean()  # [num_grid]
    scales = [test_ds.t2m_std, test_ds.tp_std]

    def _rmse(a: np.ndarray, channel: int) -> float:
        sq = (a[..., channel] - targets[..., channel]) ** 2 * weights[None, :]
        return float(np.sqrt(sq.mean()) * scales[channel])

    metrics = {
        "conditioning": conditioning,
        "checkpoint": str(ckpt_path),
        "n_test": int(len(preds)),
        "t2m_rmse_K": _rmse(preds, 0),
        "tp_rmse": _rmse(preds, 1),
        "persistence_t2m_rmse_K": _rmse(persist, 0),
        "persistence_tp_rmse": _rmse(persist, 1),
        "preds": preds,
        "targets": targets,
        "dataset": test_ds,
    }
    logger.info(
        f"NWM test ({conditioning}): T2m RMSE {metrics['t2m_rmse_K']:.3f} K "
        f"(persistence {metrics['persistence_t2m_rmse_K']:.3f} K), "
        f"precip RMSE {metrics['tp_rmse']:.3e} (persistence {metrics['persistence_tp_rmse']:.3e})"
    )
    return metrics
