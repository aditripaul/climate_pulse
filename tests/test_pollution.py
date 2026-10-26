"""Pollution preprocessing and evaluation handle missing readings correctly."""

import numpy as np
import torch

from data.processing.graph_builder import normalize_snapshot_features
from models.pollution.evaluate import evaluate_pollution_model
from models.pollution.graph_dataset import create_windowed_signal

EDGES = {"edge_index": np.array([[0, 1], [1, 0]]), "edge_weight": np.array([1.0, 1.0], dtype=np.float32)}


def _snapshots(values: list[list[float]]) -> list[dict]:
    return [{"x": np.array(v, dtype=np.float32).reshape(-1, 1), **EDGES} for v in values]


def test_normalization_ignores_missing_and_uses_given_stats():
    train = _snapshots([[10.0, np.nan], [30.0, 20.0]])
    _, stats = normalize_snapshot_features(train)
    assert stats["mean"] == 20.0
    assert np.isnan(train[0]["x"][1, 0])

    test = _snapshots([[40.0, np.nan]])
    normalize_snapshot_features(test, stats=stats)
    assert test[0]["x"][0, 0] == (40.0 - 20.0) / stats["std"]


def test_windows_impute_inputs_but_keep_missing_targets():
    signal = create_windowed_signal(_snapshots([[1.0, np.nan], [2.0, 3.0], [4.0, np.nan]]), window_size=2)
    window = next(iter(signal))
    assert not np.isnan(window.x).any()
    assert window.x[1, 0, 0] == 0.0
    assert window.y[0] == 4.0 and np.isnan(window.y[1])


class _ConstantModel(torch.nn.Module):
    """Predicts a fixed normalized value at every node."""

    def __init__(self, value: float):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(value), requires_grad=False)

    def forward(self, x, edge_index, edge_weight=None):
        return self.value.expand(x.shape[0], 1)


def test_evaluation_uses_physical_threshold_and_skips_missing():
    stats = {"mean": 50.0, "std": 10.0}
    # Two windows; node 0 is observed at 60 and 20 ug/m3 (z = 1, -3), node 1 never is
    snaps = _snapshots([[0.0, 0.0], [0.0, 0.0], [1.0, np.nan], [-3.0, np.nan]])
    signal = create_windowed_signal(snaps, window_size=2)

    # Model always predicts 50 ug/m3 (normalized 0): above 35.5, so it alerts everywhere
    metrics = evaluate_pollution_model(_ConstantModel(0.0), signal, stats)
    assert metrics["n_observed"] == 2
    assert metrics["confusion"] == {"tp": 1, "fp": 1, "fn": 0, "tn": 0}
    assert metrics["f1"] == metrics["always_alert_f1"]
