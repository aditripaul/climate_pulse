"""
Temporal Fusion Transformer model for El Niño forecasting.

Wraps pytorch-forecasting's TFT implementation with ClimatePulse-specific
configuration, hyperparameter tuning, and inference utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from loguru import logger

from config import settings
from utils import get_device


def create_tft_model(dataset: Any) -> Any:
    """
    Instantiate a Temporal Fusion Transformer from a TimeSeriesDataSet.

    Args:
        dataset: A pytorch-forecasting TimeSeriesDataSet.

    Returns:
        TemporalFusionTransformer model instance.
    """
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss

    model = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=settings.model.learning_rate,
        hidden_size=settings.model.tft_hidden_size,
        attention_head_size=settings.model.tft_attention_head_size,
        dropout=settings.model.tft_dropout,
        hidden_continuous_size=settings.model.tft_hidden_size // 2,
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        reduce_on_plateau_patience=5,
        optimizer="adam",
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Created TFT model: {n_params:,} parameters")
    return model


def load_tft_model(checkpoint_path: str | Path) -> Any:
    """Load a trained TFT model from checkpoint."""
    from pytorch_forecasting import TemporalFusionTransformer

    device = get_device()

    model = TemporalFusionTransformer.load_from_checkpoint(
        str(checkpoint_path), map_location=device
    )
    model.eval()
    logger.info(f"Loaded TFT model from {checkpoint_path} on {device}")
    return model


def predict_tft(
    model: Any,
    dataset: Any,
    return_raw: bool = False,
) -> dict:
    """
    Generate El Niño forecasts using the trained TFT.

    Args:
        model: Trained TFT model.
        dataset: Test TimeSeriesDataSet.
        return_raw: If True, return raw prediction tensor.

    Returns:
        Dictionary with:
        - 'predictions': median (50th percentile) forecasts
        - 'lower': 10th percentile (uncertainty lower bound)
        - 'upper': 90th percentile (uncertainty upper bound)
        - 'raw': raw prediction tensor (if return_raw=True)
    """
    # Use pytorch-forecasting's built-in dataloader which has the correct collate function
    dataloader = dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            x, _ = batch
            # Move all tensors to the model's device (e.g., GPU)
            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    x[k] = v.to(model.device)
            output = model(x)
            if hasattr(output, "prediction"):
                preds = output.prediction
            elif isinstance(output, tuple):
                preds = output[0]
            elif isinstance(output, dict):
                preds = output["prediction"]
            else:
                preds = output
            all_preds.append(preds.cpu())

    raw_preds = torch.cat(all_preds, dim=0)

    result = {
        "predictions": raw_preds[:, :, 1].numpy(),  # Median (50th percentile)
        "lower": raw_preds[:, :, 0].numpy(),  # 10th percentile
        "upper": raw_preds[:, :, 2].numpy(),  # 90th percentile
    }

    if return_raw:
        result["raw"] = raw_preds.numpy()

    logger.info(f"Generated {len(raw_preds)} TFT forecasts")
    return result


def get_variable_importance(model: Any, dataset: Any) -> pd.DataFrame:
    """
    Extract variable importance from TFT attention weights.

    Returns:
        DataFrame with columns: variable, importance, type (encoder/decoder).
    """
    # Use pytorch-forecasting's built-in dataloader which has the correct collate function
    dataloader = dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

    # model.predict with return_x=True returns a tuple. The first element is predictions.
    res = model.predict(dataloader, return_x=True, mode="raw")
    raw_preds = res[0] if isinstance(res, tuple) else res
    interpretation = model.interpret_output(raw_preds, reduction="mean")

    # Extract encoder variable importance tensors
    encoder_imp_tensor = interpretation.get("encoder_variables", torch.tensor([]))
    decoder_imp_tensor = interpretation.get("decoder_variables", torch.tensor([]))

    if hasattr(encoder_imp_tensor, "cpu"):
        encoder_imp_tensor = encoder_imp_tensor.cpu().numpy()
    if hasattr(decoder_imp_tensor, "cpu"):
        decoder_imp_tensor = decoder_imp_tensor.cpu().numpy()

    # Match with model variable names
    encoder_vars = getattr(model, "encoder_variables", [])
    decoder_vars = getattr(model, "decoder_variables", [])

    rows = []
    for i, var in enumerate(encoder_vars):
        if i < len(encoder_imp_tensor):
            rows.append(
                {"variable": var, "importance": float(encoder_imp_tensor[i]), "type": "encoder"}
            )

    for i, var in enumerate(decoder_vars):
        if i < len(decoder_imp_tensor):
            rows.append(
                {"variable": var, "importance": float(decoder_imp_tensor[i]), "type": "decoder"}
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("importance", ascending=False)
    return df
