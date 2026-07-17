"""
Training utilities for El Niño forecasting models.

Handles TFT training via PyTorch Lightning and LSTM training
via a standard PyTorch training loop.
"""

from __future__ import annotations

import torch
from loguru import logger
from torch.utils.data import DataLoader

from config import settings
from data.storage.db import db
from utils import get_device


def train_tft(
    model,
    train_dataset,
    val_dataset,
    max_epochs: int | None = None,
    batch_size: int | None = None,
) -> dict:
    """
    Train the Temporal Fusion Transformer using PyTorch Lightning.

    Args:
        model: TFT model instance.
        train_dataset: Training TimeSeriesDataSet.
        val_dataset: Validation TimeSeriesDataSet.
        max_epochs: Override max training epochs.
        batch_size: Override batch size.

    Returns:
        Dictionary with training results and best checkpoint path.
    """
    import pytorch_lightning as L  # noqa: N812
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    epochs = max_epochs or settings.model.max_epochs
    bs = batch_size or settings.model.batch_size

    # Data loaders
    train_loader = train_dataset.to_dataloader(train=True, batch_size=bs, num_workers=0)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=bs, num_workers=0)

    # Callbacks
    checkpoint_dir = settings.model.checkpoint_dir / "el_nino" / "tft"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    class LoguruLoggingCallback(L.Callback):
        def on_validation_epoch_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            epoch = trainer.current_epoch
            if (
                (epoch + 1) % settings.model.log_frequency == 0
                or epoch == 0
                or epoch == trainer.max_epochs - 1
            ):
                metrics = trainer.callback_metrics
                val_loss = metrics.get("val_loss")
                train_loss = (
                    metrics.get("train_loss_epoch")
                    or metrics.get("train_loss")
                    or metrics.get("loss")
                )

                parts = []
                if train_loss is not None:
                    parts.append(f"train_loss={train_loss.item():.6f}")
                if val_loss is not None:
                    parts.append(f"val_loss={val_loss.item():.6f}")

                if parts:
                    logger.info(f"Epoch {epoch + 1}/{trainer.max_epochs}: TFT " + ", ".join(parts))

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="tft-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        LoguruLoggingCallback(),
    ]

    # Trainer
    trainer = L.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        gradient_clip_val=0.1,
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        default_root_dir=str(settings.log.dir),
    )

    # Log run
    run_id = db.log_model_run("tft_el_nino", config=f"epochs={epochs}, bs={bs}")

    logger.info(f"Training TFT: {epochs} epochs, batch_size={bs}")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint_callback = trainer.checkpoint_callback
    if isinstance(checkpoint_callback, ModelCheckpoint):
        best_path = checkpoint_callback.best_model_path
        best_score = (
            float(checkpoint_callback.best_model_score)
            if checkpoint_callback.best_model_score is not None
            else 0.0
        )
    else:
        best_path = ""
        best_score = 0.0

    db.complete_model_run(
        run_id,
        status="completed",
        metrics=f"best_val_loss={best_score:.6f}",
        checkpoint_path=best_path,
    )

    logger.info(f"TFT training complete. Best val_loss={best_score:.6f}")
    logger.info(f"Best checkpoint: {best_path}")

    return {
        "best_checkpoint": best_path,
        "best_val_loss": float(best_score) if best_score else None,
        "epochs_trained": trainer.current_epoch,
    }


def train_lstm(
    model,
    train_dataset,
    val_dataset=None,
    max_epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
) -> dict:
    """
    Train the LSTM baseline using a standard PyTorch loop.

    Args:
        model: LSTMForecaster instance.
        train_dataset: LSTMDataset for training.
        val_dataset: Optional LSTMDataset for validation.
        max_epochs: Override max epochs.
        batch_size: Override batch size.
        learning_rate: Override learning rate.

    Returns:
        Dictionary with training results.
    """
    epochs = max_epochs or settings.model.max_epochs
    bs = batch_size or settings.model.batch_size
    lr = learning_rate or settings.model.learning_rate

    device = get_device()
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = torch.nn.MSELoss()

    checkpoint_dir = settings.model.checkpoint_dir / "el_nino" / "lstm"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_id = db.log_model_run("lstm_el_nino", config=f"epochs={epochs}, bs={bs}, lr={lr}")

    best_val_loss = float("inf")
    best_path = ""
    patience_counter = 0
    max_patience = 15
    history = []

    logger.info(f"Training LSTM: {epochs} epochs, batch_size={bs}, lr={lr}")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        n_batches = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Flatten target if multi-feature
            if y_batch.ndim == 3:
                y_batch = y_batch[:, :, 0]

            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(n_batches, 1)

        # Validation phase
        val_loss_str = ""
        avg_val_loss = None
        if val_loader:
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    if y_batch.ndim == 3:
                        y_batch = y_batch[:, :, 0]
                    predictions = model(x_batch)
                    val_loss += criterion(predictions, y_batch).item()
                    n_val += 1

            avg_val_loss = val_loss / max(n_val, 1)
            scheduler.step(avg_val_loss)
            val_loss_str = f", val_loss={avg_val_loss:.6f}"

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = str(checkpoint_dir / "lstm_best.pt")
                torch.save(model.state_dict(), best_path)
                patience_counter = 0
            else:
                patience_counter += 1

        history.append({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        if val_loader and patience_counter >= max_patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % settings.model.log_frequency == 0 or epoch == 0 or epoch == epochs - 1:
            logger.info(
                f"Epoch {epoch + 1}/{epochs}: train_loss={avg_train_loss:.6f}{val_loss_str}"
            )

    # Save final model
    final_path = str(checkpoint_dir / "lstm_last.pt")
    torch.save(model.state_dict(), final_path)

    # Save training history to CSV
    import csv

    with open(checkpoint_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)

    db.complete_model_run(
        run_id,
        status="completed",
        metrics=f"best_val_loss={best_val_loss:.6f}",
        checkpoint_path=best_path or final_path,
    )

    logger.info(f"LSTM training complete. Best val_loss={best_val_loss:.6f}")

    return {
        "best_checkpoint": best_path or final_path,
        "best_val_loss": best_val_loss,
        "epochs_trained": epoch + 1,
    }
