"""
Model and data access for the ClimatePulse Streamlit app.

Everything here is plain Python (no Streamlit), so it can be unit-tested. The
app only runs inference with the already-trained checkpoints; nothing is trained.
Checkpoints are looked up in ``settings.model.checkpoint_dir`` first and then in
the newest committed snapshot under ``runtime/saved/run_*/checkpoints``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import PROJECT_ROOT, settings

SAVED_DIR = PROJECT_ROOT / "runtime" / "saved"

# label -> (checkpoint folder relative to the checkpoint root, conditioning mode)
NWM_VARIANTS: dict[str, tuple[str, str]] = {
    "Main model (observed Niño 3.4)": ("nwm", "observed"),
    "Unconditioned (Niño input fixed at 0)": ("nwm_ablation/unconditioned/nwm", "none"),
    "Forecast-conditioned (TFT lead-1)": ("nwm_ablation/forecast/nwm", "forecast"),
    "Unweighted loss (observed Niño 3.4)": ("nwm_ablation/unweighted/nwm", "observed"),
}

# US EPA PM2.5 (24 h) breakpoints in ug/m3: (upper bound, label, colour)
AQI_BANDS = [
    (12.0, "Good", "#BFD7F2"),
    (35.4, "Moderate", "#8DB6E3"),
    (55.4, "Unhealthy for sensitive groups", "#4F86C6"),
    (150.4, "Unhealthy", "#2F6DB5"),
    (250.4, "Very unhealthy", "#1E4E8C"),
    (float("inf"), "Hazardous", "#13294B"),
]


class DataMissing(RuntimeError):
    """A file the page needs is absent; the message says how to create it."""


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def checkpoint_roots() -> list[tuple[str, Path]]:
    """Checkpoint roots to search, newest first: live training output, then snapshots."""
    roots: list[tuple[str, Path]] = [("live checkpoints", settings.model.checkpoint_dir)]
    for run in sorted(SAVED_DIR.glob("run_*"), reverse=True):
        roots.append((f"snapshot {run.name}", run / "checkpoints"))
    return roots


def find_checkpoint(relative: str) -> tuple[Path, str]:
    """Locate a checkpoint file or folder; returns (path, source label)."""
    for label, root in checkpoint_roots():
        candidate = root / relative
        if candidate.exists():
            return candidate, label
    raise DataMissing(
        f"No checkpoint found for '{relative}'. Train it (make train) or restore a saved run under runtime/saved/."
    )


def require(path: Path, hint: str) -> Path:
    if not path.exists():
        raise DataMissing(f"Missing {path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path}. {hint}")
    return path


# ---------------------------------------------------------------------------
# Neural Weather Model
# ---------------------------------------------------------------------------


@dataclass
class NWMContext:
    dataset: object  # NWMDataset in test mode (normalised ERA5 cube lives here)
    trainer: object  # NWMTrainer with weights loaded
    conditioning: str
    checkpoint: Path
    source: str
    num_lat: int
    num_lon: int


@lru_cache(maxsize=4)
def load_nwm(variant: str) -> NWMContext:
    from models.nwm.dataset import ERA5_PATH, NWMDataset, nino_conditioning_path
    from models.nwm.graph_builder import SphericalGraphBuilder
    from models.nwm.trainer import NWMTrainer

    relative, conditioning = NWM_VARIANTS[variant]
    ckpt_dir, source = find_checkpoint(relative)
    ckpt = ckpt_dir / "nwm_best.pt"
    require(ckpt, "Train the weather model with make train-nwm.")
    require(ERA5_PATH, "Download ERA5 with make download-nwm-data.")
    nino_path = require(nino_conditioning_path(conditioning), "Run make download-data (and make export-tft-forecasts for the forecast variant).")

    nwm = settings.nwm
    ds = NWMDataset(ERA5_PATH, nino_path, mode="test", conditioning=conditioning)
    builder = SphericalGraphBuilder(ds.lat, ds.lon, num_mesh_nodes=nwm.num_mesh_nodes)
    trainer = NWMTrainer(
        grid_in_channels=ds.window_size * 2 + 4,
        grid_out_channels=ds.prediction_steps * 2,
        hidden_dim=nwm.hidden_dim,
        num_processor_layers=nwm.num_processor_layers,
        learning_rate=nwm.learning_rate,
        weight_decay=nwm.weight_decay,
        edge_grid_to_mesh=builder.build_grid_to_mesh(),
        edge_mesh_to_mesh=builder.build_mesh_to_mesh(),
        edge_mesh_to_grid=builder.build_mesh_to_grid(),
        num_mesh_nodes=nwm.num_mesh_nodes,
    )
    trainer.load_checkpoint(ckpt)
    trainer.model.eval()
    return NWMContext(ds, trainer, conditioning, ckpt, source, len(ds.lat), len(ds.lon))


def nino_value(ctx: NWMContext, when: pd.Timestamp, override: float | None = None) -> float:
    """Niño 3.4 scalar fed to the model at time ``when`` (override wins)."""
    if override is not None:
        return float(override)
    if ctx.conditioning == "none":
        return 0.0
    series = ctx.dataset.nino_series
    idx = series.index.get_indexer([when], method="ffill")[0]
    return float(series.iloc[idx]) if idx >= 0 else 0.0


def test_times(ctx: NWMContext) -> list[pd.Timestamp]:
    """Timestamps of the held-out test samples (the time each sample predicts)."""
    return [ctx.dataset.times[i] for i in ctx.dataset.valid_indices]


@torch.no_grad()
def nwm_rollout(ctx: NWMContext, start_pos: int, n_steps: int, nino_override: float | None = None) -> dict:
    """
    Roll the weather model forward from test sample ``start_pos``.

    Step 0 is the verified one-step forecast (6 h ahead of the two input fields).
    Later steps feed the model its own predictions, which it was never trained or
    tested for. Returns physical units: T2m in K, precipitation in mm per 6 h.
    """
    ds = ctx.dataset
    w = ds.window_size
    total = len(ds.times)
    t0 = ds.valid_indices[start_pos]
    n_steps = max(1, min(n_steps, total - t0))

    t2m = ds.t2m.reshape(total, -1)
    tp = ds.tp.reshape(total, -1)
    win_t2m = [t2m[t0 - w + i] for i in range(w)]
    win_tp = [tp[t0 - w + i] for i in range(w)]
    persistence = t2m[t0 - 1] * ds.t2m_std + ds.t2m_mean

    weights = ds.lat_weights / ds.lat_weights.mean()
    pred_k, truth_k, pred_mm, truth_mm, times, rmse, rmse_persist, nino = [], [], [], [], [], [], [], []
    for k in range(n_steps):
        x = np.concatenate([np.stack(win_t2m, axis=1), np.stack(win_tp, axis=1), ds.static_features], axis=1)
        when = ds.times[t0 + k]
        cond = nino_value(ctx, when, nino_override)
        out = ctx.trainer._forward(
            torch.tensor(x[None], dtype=torch.float32, device=ctx.trainer.device),
            torch.tensor([[cond]], dtype=torch.float32, device=ctx.trainer.device),
        )[0].cpu().numpy()  # [N, 2] normalised T2m, precip

        p_k = out[:, 0] * ds.t2m_std + ds.t2m_mean
        t_k = t2m[t0 + k] * ds.t2m_std + ds.t2m_mean
        pred_k.append(p_k.reshape(ctx.num_lat, ctx.num_lon))
        truth_k.append(t_k.reshape(ctx.num_lat, ctx.num_lon))
        pred_mm.append(np.clip((out[:, 1] * ds.tp_std + ds.tp_mean) * 1000.0, 0, None).reshape(ctx.num_lat, ctx.num_lon))
        truth_mm.append(np.clip((tp[t0 + k] * ds.tp_std + ds.tp_mean) * 1000.0, 0, None).reshape(ctx.num_lat, ctx.num_lon))
        rmse.append(float(np.sqrt(np.mean((p_k - t_k) ** 2 * weights))))
        rmse_persist.append(float(np.sqrt(np.mean((persistence - t_k) ** 2 * weights))))
        times.append(when)
        nino.append(cond)

        win_t2m = win_t2m[1:] + [out[:, 0]]
        win_tp = win_tp[1:] + [out[:, 1]]

    return {
        "times": times,
        "lat": ds.lat,
        "lon": ds.lon,
        "pred_k": np.stack(pred_k),
        "truth_k": np.stack(truth_k),
        "pred_mm": np.stack(pred_mm),
        "truth_mm": np.stack(truth_mm),
        "persistence_k": persistence.reshape(ctx.num_lat, ctx.num_lon),
        "rmse_k": rmse,
        "persistence_rmse_k": rmse_persist,
        "nino": nino,
    }


# ---------------------------------------------------------------------------
# El Niño
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_elnino() -> dict:
    """TFT / LSTM / persistence forecasts for every test window, plus metrics."""
    from models.el_nino.dataset import create_tft_dataset, lstm_test_series, prepare_tft_dataframe, split_tft_dataset
    from models.el_nino.evaluate import elnino_metrics, elnino_test_predictions
    from models.el_nino.lstm_baseline import load_lstm_model
    from models.el_nino.tft_model import find_best_tft_checkpoint, load_tft_model

    require(settings.data.processed_dir / "nino_features.parquet", "Run make download-data.")
    tft_dir, source = find_checkpoint("el_nino/tft")
    lstm_path, _ = find_checkpoint("el_nino/lstm/lstm_best.pt")

    df = prepare_tft_dataframe()
    _, _, test_ds = split_tft_dataset(create_tft_dataset(df), df)
    preds = elnino_test_predictions(
        load_tft_model(find_best_tft_checkpoint(tft_dir)), test_ds, load_lstm_model(lstm_path), lstm_test_series()
    )
    n_leads = preds["actual"].shape[1]
    first = preds["first_target_time_idx"].astype(int)
    times = pd.to_datetime(df["time"].values)
    preds["target_times"] = np.array([[times[i + j] for j in range(n_leads)] for i in first])
    preds["history"] = df[["time", "nino34"]].copy()
    preds["metrics"] = elnino_metrics(preds)
    preds["source"] = source
    return preds


# ---------------------------------------------------------------------------
# Pollution
# ---------------------------------------------------------------------------


def aqi_band(value: float) -> tuple[str, str]:
    """(label, colour) of the PM2.5 band containing ``value`` (ug/m3)."""
    for upper, label, colour in AQI_BANDS:
        if value <= upper:
            return label, colour
    return AQI_BANDS[-1][1], AQI_BANDS[-1][2]


@dataclass
class PollutionContext:
    model: object
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    coords: np.ndarray  # [N, 2] lat, lon
    mean: float
    std: float
    dates: list  # snapshot dates
    windows: np.ndarray  # [n_windows, N, 1, P] normalised inputs (NaN -> 0)
    targets: np.ndarray  # [n_windows, N] normalised next-day values (NaN = missing)
    split: list[str]  # train / val / test per window
    source: str
    window_size: int


@lru_cache(maxsize=1)
def load_pollution() -> PollutionContext:
    """Build the station graph and windows offline from the cached OpenAQ file."""
    from data.processing.graph_builder import build_station_graph, build_temporal_snapshots, normalize_snapshot_features
    from models.pollution.stgnn_model import load_pollution_model

    ckpt, source = find_checkpoint("pollution/stgnn_best.pt")
    stats = json.loads(require(ckpt.parent / "norm_stats.json", "Train the pollution model with make train-pollution.").read_text())
    files = sorted((settings.data.processed_dir / "openaq").glob("pm25_*.parquet"))
    if not files:
        raise DataMissing("No cached OpenAQ PM2.5 file. Run make download-data (needs DATA_OPENAQ_API_KEY).")

    df = pd.read_parquet(files[-1])
    stations = df.drop_duplicates("location_id")[["location_id", "lat", "lon"]].reset_index(drop=True)
    graph = build_station_graph(stations)
    snaps = build_temporal_snapshots(df, graph, feature_cols=["pm25"], time_freq="D")
    normalize_snapshot_features(snaps, stats=stats)

    model = load_pollution_model(str(ckpt), in_channels=1)
    model.eval()
    p = model.periods
    n = len(snaps)
    val_end = int(n * 0.85)
    train_end = int(n * 0.7)

    windows, targets, split = [], [], []
    for i in range(n - p):
        windows.append(np.stack([np.nan_to_num(snaps[i + j]["x"], nan=0.0) for j in range(p)], axis=-1))
        targets.append(snaps[i + p]["x"][:, 0])
        # A window belongs to a split only if all its inputs and its target lie inside
        # that split's snapshots, exactly as in build_pollution_splits.
        if i >= val_end:
            split.append("test")
        elif i >= train_end and i + p < val_end:
            split.append("val")
        elif i + p < train_end:
            split.append("train")
        else:
            split.append("spans splits")

    return PollutionContext(
        model=model,
        edge_index=torch.LongTensor(graph["edge_index"]),
        edge_weight=torch.FloatTensor(graph["edge_weight"]),
        coords=np.asarray(graph["node_coords"], dtype=float),
        mean=stats["mean"],
        std=stats["std"],
        dates=[s["time"] for s in snaps],
        windows=np.stack(windows).astype(np.float32),
        targets=np.stack(targets).astype(np.float32),
        split=split,
        source=source,
        window_size=p,
    )


@torch.no_grad()
def pollution_rollout(ctx: PollutionContext, window_pos: int, n_steps: int) -> dict:
    """Next-day PM2.5 per station; later steps feed predictions back in (not validated)."""
    window = ctx.windows[window_pos].copy()
    preds = []
    for _ in range(max(1, n_steps)):
        out = ctx.model(torch.FloatTensor(window), ctx.edge_index, ctx.edge_weight).squeeze(-1).numpy()
        preds.append(out * ctx.std + ctx.mean)
        window = np.concatenate([window[:, :, 1:], out[:, None, None].astype(np.float32)], axis=-1)
    observed = ctx.targets[window_pos] * ctx.std + ctx.mean
    last_input = ctx.windows[window_pos][:, 0, -1] * ctx.std + ctx.mean
    return {"pred": np.stack(preds), "observed": observed, "last_input": last_input}


# ---------------------------------------------------------------------------
# Warming rate
# ---------------------------------------------------------------------------


def annual_gistemp() -> pd.DataFrame:
    """Annual means of complete calendar years, as used for the reported result."""
    path = require(settings.data.processed_dir / "gistemp_features.parquet", "Run make download-data.")
    monthly = pd.read_parquet(path)
    monthly["year"] = pd.to_datetime(monthly["time"]).dt.year
    annual = monthly.groupby("year")["global_temp_anomaly"].agg(["mean", "count"]).reset_index()
    annual = annual[annual["count"] == 12].rename(columns={"mean": "global_temp_anomaly"})
    return annual[["year", "global_temp_anomaly"]].reset_index(drop=True)


@lru_cache(maxsize=8)
def changepoints(max_changepoints: int, min_segment: int) -> dict:
    from models.climate_feedback.changepoint import detect_slope_changepoints

    annual = annual_gistemp()
    result = detect_slope_changepoints(
        annual, "global_temp_anomaly", "year", max_changepoints=max_changepoints, min_segment=min_segment
    )
    result["annual"] = annual
    return result
