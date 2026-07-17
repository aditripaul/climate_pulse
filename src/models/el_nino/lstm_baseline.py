"""
LSTM baseline model for El Niño forecasting.

A bidirectional LSTM serving as a comparison baseline against the
Temporal Fusion Transformer.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from config import settings


class LSTMForecaster(nn.Module):
    """
    Bidirectional LSTM for multi-step time series forecasting.

    Architecture:
        Input → Bidirectional LSTM (2 layers) → Linear → Prediction
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int | None = None,
        num_layers: int | None = None,
        prediction_length: int | None = None,
        bidirectional: bool | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size or settings.model.lstm_hidden_size
        self.num_layers = num_layers or settings.model.lstm_num_layers
        self.prediction_length = prediction_length or settings.model.lstm_prediction_months
        self.bidirectional = (
            bidirectional if bidirectional is not None else settings.model.lstm_bidirectional
        )
        self.num_directions = 2 if self.bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=dropout if self.num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size * self.num_directions, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.prediction_length),
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"Created LSTM model: {n_params:,} params, "
            f"hidden={self.hidden_size}, layers={self.num_layers}, "
            f"bidirectional={self.bidirectional}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Predictions of shape (batch, prediction_length).
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)

        # Use the last hidden state
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)

        # Project to prediction length
        output = self.fc(last_hidden)
        return cast(torch.Tensor, output)


def create_lstm_model(input_size: int = 1) -> LSTMForecaster:
    """Create an LSTM forecaster with settings from config."""
    return LSTMForecaster(input_size=input_size)


def load_lstm_model(checkpoint_path: str | Path, input_size: int = 1) -> LSTMForecaster:
    """Load a trained LSTM model from checkpoint."""
    model = LSTMForecaster(input_size=input_size)
    state_dict = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)

    # Filter state dict for size mismatches (e.g. if prediction_length changed)
    model_dict = model.state_dict()
    filtered_dict = {}
    for k, v in state_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            filtered_dict[k] = v
        else:
            logger.warning(f"Skipping key {k} from checkpoint due to shape mismatch or missing.")

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

    model.eval()
    logger.info(f"Loaded LSTM model from {checkpoint_path}")
    return model


def predict_lstm(
    model: LSTMForecaster,
    input_sequence: np.ndarray,
) -> np.ndarray:
    """
    Generate predictions from the LSTM model.

    Args:
        model: Trained LSTMForecaster.
        input_sequence: Array of shape (seq_len,) or (seq_len, features).

    Returns:
        Predictions of shape (prediction_length,).
    """
    model.eval()

    if input_sequence.ndim == 1:
        input_sequence = input_sequence.reshape(-1, 1)

    x = torch.FloatTensor(input_sequence).unsqueeze(0)  # (1, seq_len, features)

    with torch.no_grad():
        pred = model(x)

    return cast(np.ndarray, pred.squeeze().numpy())
