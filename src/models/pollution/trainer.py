"""Training and evaluation for the Pollution ST-GNN."""

from __future__ import annotations

import torch
from loguru import logger

from config import settings
from data.storage.db import db
from utils import get_device


def train_pollution_model(
    model: torch.nn.Module,
    train_signal,
    val_signal=None,
    max_epochs: int | None = None,
    learning_rate: float | None = None,
) -> dict:
    """Train the Pollution ST-GNN model."""
    epochs = max_epochs or settings.model.max_epochs
    lr = learning_rate or settings.model.learning_rate

    device = get_device()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    checkpoint_dir = settings.model.checkpoint_dir / "pollution"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_id = db.log_model_run("stgnn_pollution", config=f"epochs={epochs}, lr={lr}")

    best_val_loss = float("inf")
    best_path = ""
    history = []

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Training PollutionSTGNN: {epochs} epochs, lr={lr}, "
        f"device={device}, params={n_params:,}"
    )

    nan_batches_total = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        n = 0

        for snapshot in train_signal:
            x = torch.FloatTensor(snapshot.x).to(device)
            edge_index = torch.LongTensor(snapshot.edge_index).to(device)
            edge_weight = (
                torch.FloatTensor(snapshot.edge_attr).to(device)
                if snapshot.edge_attr is not None
                else None
            )
            y = torch.FloatTensor(snapshot.y).to(device)

            # Only stations with a reading on the target day contribute to the loss
            observed = torch.isfinite(y)
            if not observed.any():
                nan_batches_total += 1
                continue

            optimizer.zero_grad()
            pred = model(x, edge_index, edge_weight).squeeze()
            loss = torch.nn.functional.mse_loss(pred[observed], y[observed])

            # Guard against NaN loss (numerical instability)
            if torch.isnan(loss):
                nan_batches_total += 1
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n += 1

        avg_train = train_loss / max(n, 1)

        # Validation
        avg_val = None
        if val_signal:
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for snapshot in val_signal:
                    x = torch.FloatTensor(snapshot.x).to(device)
                    edge_index = torch.LongTensor(snapshot.edge_index).to(device)
                    edge_weight = (
                        torch.FloatTensor(snapshot.edge_attr).to(device)
                        if snapshot.edge_attr is not None
                        else None
                    )
                    y = torch.FloatTensor(snapshot.y).to(device)

                    observed = torch.isfinite(y)
                    if not observed.any():
                        continue

                    pred = model(x, edge_index, edge_weight).squeeze()
                    batch_loss = torch.nn.functional.mse_loss(pred[observed], y[observed])
                    if not torch.isnan(batch_loss):
                        val_loss += batch_loss.item()
                        n_val += 1

            avg_val = val_loss / max(n_val, 1)
            scheduler.step(avg_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_path = str(checkpoint_dir / "stgnn_best.pt")
                torch.save(model.state_dict(), best_path)

            if (epoch + 1) % settings.model.log_frequency == 0 or epoch == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch+1}/{epochs}: train={avg_train:.6f}, val={avg_val:.6f}")
        else:
            if (epoch + 1) % settings.model.log_frequency == 0 or epoch == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch+1}/{epochs}: train={avg_train:.6f}")

        history.append({"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val})

    final_path = str(checkpoint_dir / "stgnn_last.pt")
    torch.save(model.state_dict(), final_path)

    # Save training history to CSV
    import csv

    with open(checkpoint_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)

    if nan_batches_total > 0:
        logger.warning(f"Skipped {nan_batches_total} windows with no observed targets or NaN loss")

    db.complete_model_run(
        run_id, metrics=f"best_val={best_val_loss:.6f}", checkpoint_path=best_path or final_path
    )

    return {"best_checkpoint": best_path or final_path, "best_val_loss": best_val_loss}
