"""The TFT train/val/test split must not let targets cross split boundaries."""

import numpy as np
import pandas as pd
import pytest

from models.el_nino.dataset import (
    chronological_split_bounds,
    create_tft_dataset,
    split_tft_dataset,
)


def _decoder_time_idx(dataset) -> np.ndarray:
    loader = dataset.to_dataloader(train=False, batch_size=256, num_workers=0)
    return np.concatenate([x["decoder_time_idx"].numpy() for x, _ in loader])


@pytest.fixture
def nino_df() -> pd.DataFrame:
    n = 400
    t = np.arange(n)
    time = pd.date_range("1950-01-01", periods=n, freq="MS")
    df = pd.DataFrame(
        {
            "time": time,
            "time_idx": t,
            "series_id": "nino34",
            "nino34": np.sin(t / 6.0),
            "nino3": np.sin(t / 6.0 + 0.1),
            "month": time.month,
            "year": time.year,
            "month_sin": np.sin(2 * np.pi * time.month / 12),
            "month_cos": np.cos(2 * np.pi * time.month / 12),
        }
    )
    return df


def test_split_bounds_are_ordered():
    assert chronological_split_bounds(918) == (642, 780)


def test_targets_never_cross_split_boundaries(nino_df):
    val_start, test_start = chronological_split_bounds(len(nino_df))
    train_ds, val_ds, test_ds = split_tft_dataset(create_tft_dataset(nino_df), nino_df)

    train_targets = _decoder_time_idx(train_ds)
    val_targets = _decoder_time_idx(val_ds)
    test_targets = _decoder_time_idx(test_ds)

    assert train_targets.max() < val_start
    assert val_targets.min() >= val_start and val_targets.max() < test_start
    assert test_targets.min() >= test_start
