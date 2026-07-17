"""
Graph dataset for pollution dispersion ST-GNN.

Wraps temporal snapshots into torch-geometric-temporal signal objects.
"""

from __future__ import annotations

import numpy as np
from loguru import logger


def create_temporal_signal(
    snapshots: list[dict],
    target_offset: int = 1,
):
    """
    Create a StaticGraphTemporalSignal from temporal snapshots.

    Args:
        snapshots: List of dicts with 'x', 'edge_index', 'edge_weight'.
        target_offset: How many steps ahead for the target.

    Returns:
        StaticGraphTemporalSignal instance.
    """
    from torch_geometric_temporal.signal import StaticGraphTemporalSignal

    if len(snapshots) < target_offset + 1:
        raise ValueError(f"Need at least {target_offset + 1} snapshots, got {len(snapshots)}")

    edge_index = snapshots[0]["edge_index"]
    edge_weight = snapshots[0]["edge_weight"]

    features = []
    targets = []

    for i in range(len(snapshots) - target_offset):
        features.append(snapshots[i]["x"])
        # Target: node values at t + offset (first feature channel = target pollutant)
        targets.append(snapshots[i + target_offset]["x"][:, 0])

    dataset = StaticGraphTemporalSignal(
        edge_index=edge_index,
        edge_weight=edge_weight,
        features=features,
        targets=targets,
    )

    logger.info(
        f"Created temporal signal: {len(features)} snapshots, "
        f"{features[0].shape[0]} nodes, {features[0].shape[1]} features"
    )
    return dataset


def split_temporal_signal(
    snapshots: list[dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    """
    Split snapshots into train/val/test by time.

    Returns:
        (train_signal, val_signal, test_signal)
    """
    n = len(snapshots)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_signal = create_temporal_signal(snapshots[:train_end])
    val_signal = create_temporal_signal(snapshots[train_end:val_end])
    test_signal = create_temporal_signal(snapshots[val_end:])

    logger.info(f"Split: train={train_end}, val={val_end - train_end}, test={n - val_end}")
    return train_signal, val_signal, test_signal


def create_windowed_signal(
    snapshots: list[dict],
    window_size: int = 12,
    target_offset: int = 1,
):
    """
    Create windowed temporal signals where each sample contains
    `window_size` consecutive snapshots as input features.

    This stacks temporal windows into the feature dimension,
    compatible with A3TGCN's `periods` parameter.

    Returns:
        StaticGraphTemporalSignal with stacked temporal features.
    """
    from torch_geometric_temporal.signal import StaticGraphTemporalSignal

    edge_index = snapshots[0]["edge_index"]
    edge_weight = snapshots[0]["edge_weight"]

    features = []
    targets = []

    for i in range(len(snapshots) - window_size - target_offset + 1):
        # Stack window_size snapshots: (num_nodes, in_channels, window_size)
        window = np.stack(
            [snapshots[i + j]["x"] for j in range(window_size)],
            axis=-1,
        )
        features.append(window)
        targets.append(snapshots[i + window_size + target_offset - 1]["x"][:, 0])

    dataset = StaticGraphTemporalSignal(
        edge_index=edge_index,
        edge_weight=edge_weight,
        features=features,
        targets=targets,
    )

    logger.info(f"Created windowed signal: {len(features)} samples, " f"window_size={window_size}")
    return dataset
