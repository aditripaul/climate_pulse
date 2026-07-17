"""
Spatio-Temporal Graph Neural Network for pollution dispersion modeling.

Uses A3TGCN (Attention Temporal Graph Convolutional Network) from
torch-geometric-temporal for joint spatial-temporal forecasting
of air quality across monitoring stations.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from loguru import logger

from config import settings


class PollutionSTGNN(nn.Module):
    """
    Spatio-Temporal GNN for pollution dispersion prediction.

    Architecture:
        Input → A3TGCN (attention temporal graph conv) → Linear → Prediction

    The A3TGCN layer combines:
    - Graph convolution for spatial message passing between stations
    - GRU cells for temporal evolution
    - Attention for adaptive spatial weighting
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int | None = None,
        out_channels: int = 1,
        periods: int | None = None,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels or settings.model.gnn_hidden_channels
        self.periods = periods or settings.model.gnn_num_periods

        # Lazy import to avoid hard dependency at module level
        from torch_geometric_temporal.nn.recurrent import A3TGCN

        self.tgcn = A3TGCN(
            in_channels=in_channels,
            out_channels=self.hidden_channels,
            periods=self.periods,
        )

        self.linear = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_channels // 2, out_channels),
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"Created PollutionSTGNN: {n_params:,} params, "
            f"hidden={self.hidden_channels}, periods={self.periods}"
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features of shape (num_nodes, in_channels, periods).
            edge_index: Graph edge indices of shape (2, num_edges).
            edge_weight: Optional edge weights of shape (num_edges,).

        Returns:
            Predictions of shape (num_nodes, out_channels).
        """
        # A3TGCN expects (num_nodes, in_channels, periods)
        h = self.tgcn(x, edge_index, edge_weight)  # → (num_nodes, hidden_channels)
        out = self.linear(h)  # → (num_nodes, out_channels)
        return cast(torch.Tensor, out)


class PollutionSTGNNMultiStep(nn.Module):
    """
    Multi-step prediction variant of the ST-GNN.

    Unrolls predictions over multiple future time steps.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int | None = None,
        prediction_steps: int = 7,
        periods: int | None = None,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels or settings.model.gnn_hidden_channels
        self.prediction_steps = prediction_steps

        from torch_geometric_temporal.nn.recurrent import A3TGCN

        self.tgcn = A3TGCN(
            in_channels=in_channels,
            out_channels=self.hidden_channels,
            periods=periods or settings.model.gnn_num_periods,
        )

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_channels, prediction_steps),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for multi-step prediction.

        Returns:
            Predictions of shape (num_nodes, prediction_steps).
        """
        h = self.tgcn(x, edge_index, edge_weight)
        return cast(torch.Tensor, self.fc(h))


def create_pollution_model(
    in_channels: int = 1,
    multi_step: bool = False,
    prediction_steps: int = 7,
) -> nn.Module:
    """Create a pollution ST-GNN model."""
    if multi_step:
        return PollutionSTGNNMultiStep(
            in_channels=in_channels,
            prediction_steps=prediction_steps,
        )
    return PollutionSTGNN(in_channels=in_channels)


def load_pollution_model(
    checkpoint_path: str | Path,
    in_channels: int = 1,
) -> PollutionSTGNN:
    """Load a trained pollution model."""
    model = PollutionSTGNN(in_channels=in_channels)
    state_dict = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Loaded PollutionSTGNN from {checkpoint_path}")
    return model
