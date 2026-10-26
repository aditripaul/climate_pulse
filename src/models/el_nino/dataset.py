"""
Dataset preparation for El Niño forecasting.

Prepares TimeSeriesDataSet for the Temporal Fusion Transformer using
pytorch-forecasting, and standard PyTorch datasets for the LSTM baseline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset

from config import settings


def chronological_split_bounds(
    n: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[int, int]:
    """
    Row indices for a chronological train/val/test split shared by TFT and LSTM.

    Returns:
        (val_start, test_start): train = [0, val_start), val = [val_start, test_start),
        test = [test_start, n).
    """
    return int(n * train_ratio), int(n * (train_ratio + val_ratio))


def prepare_tft_dataframe(
    nino_features_path: str | None = None,
) -> pd.DataFrame:
    """
    Load and format Niño features into the long-format required by
    pytorch-forecasting's TimeSeriesDataSet.

    Returns:
        DataFrame ready for TimeSeriesDataSet with columns:
        time_idx, series_id, target (nino34), and covariates.
    """
    path = nino_features_path or str(settings.data.processed_dir / "nino_features.parquet")
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Create integer time index (months since start)
    df["time_idx"] = np.arange(len(df))

    # Single series ID (global Niño index)
    df["series_id"] = "nino34"

    # Ensure target column exists
    if "nino34" not in df.columns:
        raise ValueError("nino34 column not found in features DataFrame")

    # Fill NaN in feature columns (rolling/lag features have NaN at start)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].bfill().fillna(0)

    logger.info(f"Prepared TFT DataFrame: {len(df)} timesteps, {len(numeric_cols)} features")
    return df


def create_tft_dataset(
    df: pd.DataFrame,
    encoder_length: int | None = None,
    prediction_length: int | None = None,
) -> Any:
    """
    Create a pytorch-forecasting TimeSeriesDataSet for the TFT.

    Args:
        df: Formatted DataFrame from prepare_tft_dataframe().
        encoder_length: Number of historical timesteps (default from config).
        prediction_length: Number of forecast timesteps (default from config).

    Returns:
        TimeSeriesDataSet instance.
    """
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import NaNLabelEncoder

    enc_len = encoder_length or settings.model.tft_encoder_length
    pred_len = prediction_length or settings.model.tft_prediction_months

    # Determine available feature columns
    time_varying_known = ["month_sin", "month_cos", "month", "year"]
    time_varying_unknown = [
        c
        for c in df.columns
        if (c.startswith("nino") and c != "nino34")
        or c.startswith("nino34_roll")
        or c.startswith("nino34_lag")
    ]

    # Filter to columns that actually exist
    time_varying_known = [c for c in time_varying_known if c in df.columns]
    time_varying_unknown = [c for c in time_varying_unknown if c in df.columns]

    # Train only on the training split so that no validation/test month is ever
    # seen as an encoder input or prediction target during fitting.
    val_start, _ = chronological_split_bounds(len(df))

    dataset = TimeSeriesDataSet(
        df[df["time_idx"] < val_start],
        time_idx="time_idx",
        target="nino34",
        group_ids=["series_id"],
        max_encoder_length=enc_len,
        max_prediction_length=pred_len,
        time_varying_known_reals=time_varying_known,
        time_varying_unknown_reals=["nino34"] + time_varying_unknown,
        target_normalizer=None,  # Keep raw anomaly values
        categorical_encoders={"series_id": NaNLabelEncoder(add_nan=True)},
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    logger.info(
        f"Created TFT dataset: encoder={enc_len}, prediction={pred_len}, "
        f"known_reals={len(time_varying_known)}, unknown_reals={len(time_varying_unknown)+1}"
    )
    return dataset


def split_tft_dataset(
    dataset: Any,
    df: pd.DataFrame,
) -> tuple[Any, Any, Any]:
    """
    Split TimeSeriesDataSet into train/val/test by time.

    `dataset` must come from create_tft_dataset(), which already covers only the
    training split. Validation and test prediction targets never overlap each
    other or the training targets.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    from pytorch_forecasting import TimeSeriesDataSet

    n = len(df)
    val_start, test_start = chronological_split_bounds(n)

    enc_len = dataset.max_encoder_length

    # Validation and test DataFrames must include the historical encoder length
    # so that the model has past context to make predictions. Validation stops
    # at test_start so its targets never reach into the test period.
    val_df = df[(df["time_idx"] >= val_start - enc_len) & (df["time_idx"] < test_start)]
    test_df = df[df["time_idx"] >= test_start - enc_len]

    val_dataset = TimeSeriesDataSet.from_dataset(dataset, val_df, stop_randomization=True)
    test_dataset = TimeSeriesDataSet.from_dataset(dataset, test_df, stop_randomization=True)

    logger.info(
        f"Split: train={val_start} months, val={test_start - val_start} months, "
        f"test={n - test_start} months; windows train={len(dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )
    return dataset, val_dataset, test_dataset


# ---------------------------------------------------------------------------
# LSTM Dataset (simpler format)
# ---------------------------------------------------------------------------


class LSTMDataset(Dataset):
    """
    Simple sliding-window dataset for the LSTM baseline.

    Creates (input_sequence, target_sequence) pairs from
    the Niño 3.4 time series.
    """

    def __init__(
        self,
        data: np.ndarray,
        encoder_length: int | None = None,
        prediction_length: int | None = None,
    ):
        self.data = torch.FloatTensor(data)
        self.encoder_length = encoder_length or settings.model.lstm_encoder_length
        self.prediction_length = prediction_length or settings.model.lstm_prediction_months

    def __len__(self) -> int:
        return len(self.data) - self.encoder_length - self.prediction_length + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.encoder_length]
        y = self.data[
            idx + self.encoder_length : idx + self.encoder_length + self.prediction_length
        ]
        return x, y

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        target_col: str = "nino34",
        feature_cols: list[str] | None = None,
    ) -> LSTMDataset:
        """Create dataset from a features DataFrame."""
        if feature_cols:
            cols = [target_col] + [c for c in feature_cols if c in df.columns]
        else:
            cols = [target_col]

        data = df[cols].values.astype(np.float32)
        return cls(data)
