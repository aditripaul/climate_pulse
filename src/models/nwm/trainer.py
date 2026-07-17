"""
Native PyTorch Trainer for the Neural Weather Model (NWM).

Replaces the PyTorch Lightning wrapper with a plain PyTorch training loop
that mirrors the pattern used by train_lstm() in models/el_nino/trainer.py.
No lightning dependency.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from config import settings
from data.storage.db import db
from models.nwm.stgnn_model import NWMModel
from utils import get_device


class NWMTrainer:
    """
    Self-contained trainer for the Neural Weather Model (ST-GNN).

    Handles the training loop, validation, early stopping, checkpointing,
    and LR scheduling without any third-party training framework dependency.

    Args:
        grid_in_channels:    Features per grid node (channels × window + static).
        grid_out_channels:   Output features per grid node (channels × pred_steps).
        hidden_dim:          Latent dimension inside the mesh.
        num_processor_layers: Message-passing steps in the mesh processor.
        learning_rate:       Initial learning rate for AdamW.
        weight_decay:        L2 regularisation coefficient.
        edge_grid_to_mesh:   [2, E] edge index, grid→mesh.
        edge_mesh_to_mesh:   [2, E] edge index, mesh→mesh.
        edge_mesh_to_grid:   [2, E] edge index, mesh→grid.
        num_mesh_nodes:      Number of latent mesh nodes.
    """

    def __init__(
        self,
        grid_in_channels: int,
        grid_out_channels: int,
        hidden_dim: int,
        num_processor_layers: int,
        learning_rate: float,
        weight_decay: float,
        edge_grid_to_mesh: torch.Tensor,
        edge_mesh_to_mesh: torch.Tensor,
        edge_mesh_to_grid: torch.Tensor,
        num_mesh_nodes: int,
    ) -> None:
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_mesh_nodes = num_mesh_nodes

        self.device = get_device()

        # Move graph topology to the target device once at construction time.
        self.edge_grid_to_mesh = edge_grid_to_mesh.to(self.device)
        self.edge_mesh_to_mesh = edge_mesh_to_mesh.to(self.device)
        self.edge_mesh_to_grid = edge_mesh_to_grid.to(self.device)

        self.model = NWMModel(
            grid_in_channels=grid_in_channels,
            grid_out_channels=grid_out_channels,
            hidden_dim=hidden_dim,
            num_processor_layers=num_processor_layers,
        ).to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        max_epochs: int | None = None,
        checkpoint_dir: Path | None = None,
        early_stopping_patience: int = 10,
        grad_clip: float = 1.0,
    ) -> dict:
        """
        Run the training loop.

        Args:
            train_loader:             DataLoader yielding (x, condition, y) batches.
            val_loader:               Optional validation DataLoader.
            max_epochs:               Number of epochs (falls back to settings).
            checkpoint_dir:           Directory for checkpoints and metrics CSV.
            early_stopping_patience:  Epochs without val improvement before stopping.
            grad_clip:                Max gradient norm for clipping.

        Returns:
            dict with keys: best_checkpoint, best_val_loss, epochs_trained.
        """
        epochs = max_epochs or settings.model.max_epochs
        ckpt_dir = checkpoint_dir or (settings.model.checkpoint_dir / "nwm")
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

        run_id = db.log_model_run(
            "nwm_stgnn",
            config=f"epochs={epochs}, lr={self.learning_rate}, wd={self.weight_decay}",
        )

        best_val_loss = float("inf")
        best_path = ""
        patience_counter = 0
        history: list[dict] = []

        logger.info(
            f"Training NWM ST-GNN: {epochs} epochs, device={self.device}, "
            f"lr={self.learning_rate}, wd={self.weight_decay}"
        )

        for epoch in range(epochs):
            # ---- Training phase ----------------------------------------
            self.model.train()
            train_loss_sum = 0.0
            n_train = 0

            for x, condition, y in train_loader:
                x = x.to(self.device)
                condition = condition.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                y_hat = self._forward(x, condition)
                loss = F.mse_loss(y_hat, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                optimizer.step()

                train_loss_sum += loss.item()
                n_train += 1

            avg_train_loss = train_loss_sum / max(n_train, 1)

            # ---- Validation phase --------------------------------------
            avg_val_loss: float | None = None
            val_loss_str = ""

            if val_loader is not None:
                avg_val_loss = self._validate(val_loader)
                scheduler.step(avg_val_loss)
                val_loss_str = f", val_loss={avg_val_loss:.6f}, val_rmse={avg_val_loss**0.5:.6f}"

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_path = str(ckpt_dir / "nwm_best.pt")
                    torch.save(self.model.state_dict(), best_path)
                    patience_counter = 0
                else:
                    patience_counter += 1

            history.append(
                {"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss}
            )

            if (
                val_loader is not None
                and patience_counter >= early_stopping_patience
            ):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}.")
                break

            if (
                (epoch + 1) % settings.model.log_frequency == 0
                or epoch == 0
                or epoch == epochs - 1
            ):
                logger.info(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"train_loss={avg_train_loss:.6f}{val_loss_str}"
                )

        # ---- Persist final state & metrics ----------------------------
        final_path = str(ckpt_dir / "nwm_last.pt")
        torch.save(self.model.state_dict(), final_path)

        with open(ckpt_dir / "metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
            writer.writeheader()
            writer.writerows(history)

        db.complete_model_run(
            run_id,
            status="completed",
            metrics=f"best_val_loss={best_val_loss:.6f}",
            checkpoint_path=best_path or final_path,
        )

        logger.info(f"NWM training complete. Best val_loss={best_val_loss:.6f}")
        logger.info(f"Best checkpoint: {best_path or final_path}")

        return {
            "best_checkpoint": best_path or final_path,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(history),
        }

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model weights from a checkpoint file."""
        state = torch.load(str(path), map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        logger.info(f"Loaded NWM checkpoint from {path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.model(
            x,
            condition,
            self.edge_grid_to_mesh,
            self.edge_mesh_to_mesh,
            self.edge_mesh_to_grid,
            self.num_mesh_nodes,
        )

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        for x, condition, y in val_loader:
            x = x.to(self.device)
            condition = condition.to(self.device)
            y = y.to(self.device)
            y_hat = self._forward(x, condition)
            total_loss += F.mse_loss(y_hat, y).item()
            n_batches += 1
        self.model.train()
        return total_loss / max(n_batches, 1)
