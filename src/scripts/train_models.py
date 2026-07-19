"""
Train all ClimatePulse models.

Usage:
    python -m scripts.train_models [--model tft|lstm|stgnn|bayesian|all]
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

    df = prepare_tft_dataframe()
    dataset = create_tft_dataset(df)
    train_ds, val_ds, _ = split_tft_dataset(dataset, df)
    model = create_tft_model(train_ds)
    result = _train(model, train_ds, val_ds)

    logger.info(f"TFT result: {result}")
    return result


def train_lstm():
    """Train the LSTM baseline."""
    logger.info("Training LSTM baseline...")

    import numpy as np
    import pandas as pd

    from models.el_nino.dataset import LSTMDataset
    from models.el_nino.lstm_baseline import create_lstm_model
    from models.el_nino.trainer import train_lstm as _train

    df = pd.read_parquet(settings.data.processed_dir / "nino_features.parquet")
    data = df["nino34"].values.astype(np.float32)

    n = len(data)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    enc_len = settings.model.lstm_encoder_length
    train_ds = LSTMDataset(data[:train_end].reshape(-1, 1))
    val_ds = LSTMDataset(data[train_end - enc_len : val_end].reshape(-1, 1))

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

    from config import settings
    from data.processing.graph_builder import (
        build_station_graph,
        build_temporal_snapshots,
        filter_locations_to_measurements,
        normalize_snapshot_features,
    )
    from data.sources.openaq import OpenAQSource
    from models.pollution.graph_dataset import (
        create_windowed_signal,
    )
    from models.pollution.stgnn_model import create_pollution_model
    from models.pollution.trainer import train_pollution_model

    openaq = OpenAQSource()
    locations_df = openaq.fetch_locations(
        countries=settings.data.focus_countries,
        limit=settings.data.openaq_max_locations,
    )
    pm25_df = openaq.fetch_measurements(
        parameter="pm25",
        countries=settings.data.focus_countries,
        date_from=f"{settings.data.start_year}-01-01",
        date_to=f"{settings.data.end_year}-12-31",
        limit=settings.data.openaq_limit,
        max_locations=settings.data.openaq_max_locations,
        use_cache=True,
    )

    # Filter to only stations present in measurements (reduces sparsity)
    locations_df = filter_locations_to_measurements(locations_df, pm25_df)

    # Build Graph
    graph = build_station_graph(locations_df)

    # Build Time-Series Snapshots
    snapshots = build_temporal_snapshots(pm25_df, graph, feature_cols=["pm25"], time_freq="D")

    # Normalize features for stable training (z-score on non-zero values)
    import json

    snapshots, norm_stats = normalize_snapshot_features(snapshots)

    # Save normalization stats for inference
    stats_path = settings.model.checkpoint_dir / "pollution" / "norm_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    logger.info(f"Saved normalization stats to {stats_path}")

    # Split & Window Dataset
    n = len(snapshots)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_windowed = create_windowed_signal(
        snapshots[:train_end], window_size=settings.model.gnn_num_periods
    )
    val_windowed = create_windowed_signal(
        snapshots[train_end:val_end], window_size=settings.model.gnn_num_periods
    )

    model = create_pollution_model(
        in_channels=1,
    )
    result = train_pollution_model(model, train_windowed, val_windowed)

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

    nc_path = settings.data.raw_dir / "nwm_era5" / "era5_5deg_6hr_2020_2022.nc"
    nino_path = settings.data.processed_dir / "nino_features.parquet"

    if not nc_path.exists():
        logger.error(
            "ERA5 data missing. Please run `python src/scripts/download_nwm_data.py` first."
        )
        return

    train_ds = NWMDataset(nc_path, nino_path, mode="train")
    val_ds = NWMDataset(nc_path, nino_path, mode="val")

    train_loader = DataLoader(
        train_ds, batch_size=settings.model.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=settings.model.batch_size, shuffle=False, num_workers=0
    )

    # Build the spherical graph topology once
    graph_builder = SphericalGraphBuilder(train_ds.lat, train_ds.lon, num_mesh_nodes=2000)
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
        hidden_dim=128,
        num_processor_layers=4,
        learning_rate=settings.model.learning_rate,
        weight_decay=1e-4,
        edge_grid_to_mesh=e_g2m,
        edge_mesh_to_mesh=e_m2m,
        edge_mesh_to_grid=e_m2g,
        num_mesh_nodes=2000,
        # C3: pass cosine-latitude weights for area-correct loss computation
        lat_weights=torch.tensor(train_ds.lat_weights),
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
    args = parser.parse_args()

    if args.epochs:
        # Re-build the settings singleton with the epoch override applied.
        # Pydantic v2 BaseSettings instances are immutable; we clear the
        # lru_cache so get_settings() re-reads everything from scratch and
        # the overridden value is picked up via the env variable mechanism.
        import os
        from config import get_settings
        os.environ["MODEL_MAX_EPOCHS"] = str(args.epochs)
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
