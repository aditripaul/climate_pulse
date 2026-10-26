"""
Train all ClimatePulse models.

Usage:
    python -m scripts.train_models [--model tft|lstm|stgnn|bayesian|nwm|all]
                                   [--epochs N] [--seed N]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os

# Radeon 680M (RDNA2) workaround for PyTorch ROCm (gfx1036 -> gfx1030)
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("HSA_ENABLE_SDMA", "0")
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
# Disable PyTensor C compilation (missing Python.h)
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

import torch
torch.backends.cudnn.enabled = False

from loguru import logger  # noqa: E402

from config import settings  # noqa: E402


def train_tft():
    """Train the Temporal Fusion Transformer."""
    logger.info("Training TFT model...")

    from models.el_nino.dataset import (
        create_tft_dataset,
        prepare_tft_dataframe,
        split_tft_dataset,
    )
    from models.el_nino.tft_model import create_tft_model
    from models.el_nino.trainer import train_tft as _train

    from utils import set_seed

    df = prepare_tft_dataframe()
    dataset = create_tft_dataset(df)
    train_ds, val_ds, _ = split_tft_dataset(dataset, df)
    set_seed()
    model = create_tft_model(train_ds)
    result = _train(model, train_ds, val_ds)

    logger.info(f"TFT result: {result}")
    return result


def train_lstm():
    """Train the LSTM baseline."""
    logger.info("Training LSTM baseline...")

    import numpy as np
    import pandas as pd

    from models.el_nino.dataset import LSTMDataset, chronological_split_bounds
    from models.el_nino.lstm_baseline import create_lstm_model
    from models.el_nino.trainer import train_lstm as _train

    df = pd.read_parquet(settings.data.processed_dir / "nino_features.parquet")
    data = df["nino34"].values.astype(np.float32)

    train_end, val_end = chronological_split_bounds(len(data))

    enc_len = settings.model.lstm_encoder_length
    train_ds = LSTMDataset(data[:train_end].reshape(-1, 1))
    val_ds = LSTMDataset(data[train_end - enc_len : val_end].reshape(-1, 1))

    from utils import set_seed

    set_seed()
    model = create_lstm_model(input_size=1)
    result = _train(model, train_ds, val_ds)

    logger.info(f"LSTM result: {result}")
    return result


def train_bayesian():
    """Train the Bayesian regression model."""
    logger.info("Training Bayesian regression...")

    import pandas as pd

    from data.pipeline import DataPipeline
    from models.climate_feedback.bayesian_regression import (
        build_bayesian_model,
        fit_bayesian_model,
        summarize_posterior,
    )

    pipeline = DataPipeline()
    result = pipeline.build_regression_dataset()

    df = pd.read_parquet(result["path"])
    model = build_bayesian_model(df)
    trace = fit_bayesian_model(model, draws=1000, chains=2)
    summary = summarize_posterior(trace, model)

    logger.info(f"Bayesian regression posterior summary:\n{summary}")
    return {"status": "success"}


def train_stgnn():
    """Train the Spatio-Temporal Graph Neural Network for Pollution."""
    logger.info("Training Pollution ST-GNN...")

    import json

    from models.pollution.graph_dataset import build_pollution_splits
    from models.pollution.stgnn_model import create_pollution_model
    from models.pollution.trainer import train_pollution_model
    from utils import set_seed

    splits = build_pollution_splits(window_size=settings.model.gnn_num_periods)

    # Save normalization stats (computed on the training split) for evaluation/inference
    stats_path = settings.model.checkpoint_dir / "pollution" / "norm_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(splits["norm_stats"], f, indent=2)
    logger.info(f"Saved normalization stats to {stats_path}")

    set_seed()
    model = create_pollution_model(in_channels=1)
    result = train_pollution_model(model, splits["train"], splits["val"])

    logger.info(f"ST-GNN result: {result}")
    return result


def train_nwm():
    """Train the Neural Weather Model (ST-GNN)."""
    logger.info("Training Neural Weather Model (NWM)...")

    import torch
    from torch.utils.data import DataLoader

    from models.nwm.dataset import NWMDataset
    from models.nwm.graph_builder import SphericalGraphBuilder
    from models.nwm.trainer import NWMTrainer
    from config import settings

    from models.nwm.dataset import ERA5_PATH, nino_conditioning_path
    from utils import set_seed

    nwm = settings.nwm
    if not ERA5_PATH.exists():
        logger.error(
            "ERA5 data missing. Please run `python src/scripts/download_nwm_data.py` first."
        )
        return

    nino_path = nino_conditioning_path(nwm.conditioning)
    logger.info(
        f"NWM conditioning={nwm.conditioning} ({nino_path.name}), "
        f"area_weighted_loss={nwm.area_weighted_loss}"
    )
    train_ds = NWMDataset(ERA5_PATH, nino_path, mode="train", conditioning=nwm.conditioning)
    val_ds = NWMDataset(ERA5_PATH, nino_path, mode="val", conditioning=nwm.conditioning)
    set_seed()

    train_loader = DataLoader(
        train_ds, batch_size=settings.model.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=settings.model.batch_size, shuffle=False, num_workers=0
    )

    # Build the spherical graph topology once
    graph_builder = SphericalGraphBuilder(
        train_ds.lat, train_ds.lon, num_mesh_nodes=nwm.num_mesh_nodes
    )
    e_g2m = graph_builder.build_grid_to_mesh()
    e_m2m = graph_builder.build_mesh_to_mesh()
    e_m2g = graph_builder.build_mesh_to_grid()

    # A1: grid_in_channels includes 4 static positional features (sin/cos lat/lon)
    # that are now appended to every input sample inside NWMDataset.__getitem__.
    grid_in_channels = train_ds.window_size * 2 + 4   # T2m_W + Precip_W + static
    grid_out_channels = train_ds.prediction_steps * 2

    trainer = NWMTrainer(
        grid_in_channels=grid_in_channels,
        grid_out_channels=grid_out_channels,
        hidden_dim=nwm.hidden_dim,
        num_processor_layers=nwm.num_processor_layers,
        learning_rate=nwm.learning_rate,
        weight_decay=nwm.weight_decay,
        edge_grid_to_mesh=e_g2m,
        edge_mesh_to_mesh=e_m2m,
        edge_mesh_to_grid=e_m2g,
        num_mesh_nodes=nwm.num_mesh_nodes,
        # C3: cosine-latitude weights for area-correct loss (disabled for the ablation)
        lat_weights=torch.tensor(train_ds.lat_weights) if nwm.area_weighted_loss else None,
    )

    result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=settings.model.max_epochs,
    )

    logger.info("NWM training complete.")
    return result


def main():
    parser = argparse.ArgumentParser(description="Train ClimatePulse models")
    parser.add_argument(
        "--model",
        choices=["tft", "lstm", "stgnn", "bayesian", "nwm", "all"],
        default="all",
        help="Which model to train",
    )
    parser.add_argument("--epochs", type=int, help="Override max epochs")
    parser.add_argument("--seed", type=int, help="Override the random seed")
    args = parser.parse_args()

    # CLI overrides go through environment variables, which take precedence over the
    # YAML config; clearing the cache makes get_settings() pick them up.
    overrides = {"MODEL_MAX_EPOCHS": args.epochs, "MODEL_SEED": args.seed}
    if any(v is not None for v in overrides.values()):
        from config import get_settings

        for key, value in overrides.items():
            if value is not None:
                os.environ[key] = str(value)
        get_settings.cache_clear()

    if args.model in ("tft", "all"):
        train_tft()
    if args.model in ("lstm", "all"):
        train_lstm()
    if args.model in ("stgnn", "all"):
        train_stgnn()
    if args.model in ("bayesian", "all"):
        train_bayesian()
    if args.model in ("nwm", "all"):
        train_nwm()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
