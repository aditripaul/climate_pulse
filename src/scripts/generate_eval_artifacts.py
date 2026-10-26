#!/usr/bin/env python3
"""
Generate publication-quality paper artifacts for ClimatePulse.

Evaluates all models (TFT, LSTM, ST-GNN, Bayesian) on test sets,
saves raw prediction/metric files, and plots standard graphics.
"""

import glob
import json
import os
import shutil
import sys
from pathlib import Path

# Radeon 680M (RDNA2) workaround for PyTorch ROCm (gfx1036 -> gfx1030)
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("HSA_ENABLE_SDMA", "0")
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

torch.backends.cudnn.enabled = False

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import settings  # noqa: E402
from models.el_nino.dataset import (  # noqa: E402
    create_tft_dataset,
    lstm_test_series,
    prepare_tft_dataframe,
    split_tft_dataset,
)
from models.el_nino.evaluate import elnino_metrics, elnino_test_predictions  # noqa: E402
from models.el_nino.lstm_baseline import load_lstm_model  # noqa: E402
from models.el_nino.tft_model import find_best_tft_checkpoint, load_tft_model  # noqa: E402
from models.pollution.evaluate import AQI_100_PM25_UGM3, evaluate_pollution_model  # noqa: E402
from models.pollution.stgnn_model import load_pollution_model  # noqa: E402

# Set seaborn style for paper figures
sns.set_theme(style="whitegrid", context="paper", palette="muted")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 13,
        "legend.fontsize": 9,
    }
)

# Create output directories
ARTIFACTS_DIR = PROJECT_ROOT / "runtime" / "eval_artifacts"
RESULTS_DIR = ARTIFACTS_DIR / "run_results"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_tft_test_data():
    """Load the best TFT checkpoint of the latest run and the test split."""
    best_ckpt = find_best_tft_checkpoint(settings.model.checkpoint_dir / "el_nino" / "tft")
    print(f"Using TFT checkpoint: {best_ckpt}")
    tft_model = load_tft_model(best_ckpt)

    df = prepare_tft_dataframe()
    dataset = create_tft_dataset(df)
    _, _, test_ds = split_tft_dataset(dataset, df)

    return tft_model, test_ds, df


def load_lstm_test_data():
    """Load the LSTM baseline and the Nino 3.4 series covering its test windows."""
    lstm_ckpt = settings.model.checkpoint_dir / "el_nino" / "lstm" / "lstm_best.pt"
    if not lstm_ckpt.exists():
        raise FileNotFoundError("No LSTM baseline checkpoint found.")
    return load_lstm_model(lstm_ckpt), lstm_test_series()


def evaluate_el_nino(tft_model, test_ds, lstm_model, lstm_test_data):
    """Evaluate El Nino TFT vs LSTM and persistence baselines."""
    print("Evaluating El Niño forecasting models...")

    p = elnino_test_predictions(tft_model, test_ds, lstm_model, lstm_test_data)
    m = elnino_metrics(p)

    tft_actuals, tft_preds = p["actual"], p["tft"]
    tft_lower, tft_upper = p["tft_lower"], p["tft_upper"]
    lstm_actuals, lstm_preds = p["actual"], p["lstm"]
    comp_samples = len(tft_actuals)

    # Save raw run results to CSV/JSON
    pd.DataFrame(
        {
            "actual": tft_actuals.flatten(),
            "tft_pred": tft_preds.flatten(),
            "tft_lower": tft_lower.flatten(),
            "tft_upper": tft_upper.flatten(),
            "lstm_pred": lstm_preds.flatten(),
            "persistence_pred": p["persistence"].flatten(),
        }
    ).to_csv(RESULTS_DIR / "elnino_predictions.csv", index=False)

    # Error over forecast lead time
    lead_times = np.arange(1, tft_actuals.shape[1] + 1)
    tft_lead_rmse = m["lead_rmse"]["tft"]
    lstm_lead_rmse = m["lead_rmse"]["lstm"]
    persist_lead_rmse = m["lead_rmse"]["persistence"]
    pd.DataFrame(
        {
            "lead_time_months": lead_times,
            "tft_rmse": tft_lead_rmse,
            "lstm_rmse": lstm_lead_rmse,
            "persistence_rmse": persist_lead_rmse,
        }
    ).to_csv(RESULTS_DIR / "elnino_lead_time_metrics.csv", index=False)

    # Plot 1: Forecast vs Actual sequence (draw a nice random one)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    seq_idx = comp_samples - 10  # recent sequence
    ax.plot(lead_times, tft_actuals[seq_idx, :12], "k-", label="Observed (Niño 3.4)", marker="o")
    ax.plot(lead_times, tft_preds[seq_idx, :12], "b-", label="TFT Forecast (Median)", marker="s")
    ax.fill_between(
        lead_times,
        tft_lower[seq_idx, :12],
        tft_upper[seq_idx, :12],
        color="blue",
        alpha=0.15,
        label="TFT 80% CI",
    )
    ax.plot(lead_times, lstm_preds[seq_idx, :12], "r--", label="LSTM Baseline", marker="^")
    ax.set_xlabel("Forecast Lead Time (Months)")
    ax.set_ylabel("SST Anomaly (°C)")
    ax.set_title("Multi-Horizon El Niño Forecast Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "elnino_forecast_vs_actual.png", dpi=300)
    plt.close(fig)

    # Plot 2: Lead time RMSE
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(lead_times, tft_lead_rmse, "b-o", label="TFT")
    ax.plot(lead_times, lstm_lead_rmse, "r-^", label="LSTM")
    ax.plot(lead_times, persist_lead_rmse, "g--x", label="Persistence Baseline")
    ax.set_xlabel("Lead Time (Months)")
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("Forecast Skill Decay over Lead Horizon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "elnino_crps_leadtime.png", dpi=300)
    plt.close(fig)

    # Plot 3: Quantile calibration (observed frequency below each predicted quantile)
    calib = pd.DataFrame(m["quantile_calibration"])
    calib.to_csv(RESULTS_DIR / "elnino_quantile_calibration.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.plot(calib["quantile"], calib["observed_frequency"], "b-s", label="TFT (test set)")
    for q, f in zip(calib["quantile"], calib["observed_frequency"], strict=True):
        ax.annotate(f"{f:.2f}", (q, f), textcoords="offset points", xytext=(-24, 6), fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted Quantile Level")
    ax.set_ylabel("Observed Frequency Below Quantile")
    ax.set_title("TFT Quantile Calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "elnino_calibration_diagram.png", dpi=300)
    plt.close(fig)

    # Plot 4: Scatter Plot predicted vs actual anomalies
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        tft_actuals[:comp_samples].flatten(),
        tft_preds[:comp_samples].flatten(),
        color="blue",
        alpha=0.3,
        label="TFT",
        s=10,
    )
    ax.scatter(
        lstm_actuals[:comp_samples].flatten(),
        lstm_preds[:comp_samples].flatten(),
        color="red",
        alpha=0.3,
        label="LSTM",
        s=10,
    )
    min_val = float(min(tft_actuals[:comp_samples].min(), lstm_actuals[:comp_samples].min()))
    max_val = float(max(tft_actuals[:comp_samples].max(), lstm_actuals[:comp_samples].max()))
    ax.plot([min_val, max_val], [min_val, max_val], "k--", label="Perfect Forecast")
    ax.set_xlabel("Observed SST Anomaly (°C)")
    ax.set_ylabel("Predicted SST Anomaly (°C)")
    ax.set_title("Predicted vs. Observed Anomaly Scatter")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "elnino_predicted_vs_actual_scatter.png", dpi=300)
    plt.close(fig)
    print(
        "Saved El Nino scatter plot to runtime/eval_artifacts/elnino_predicted_vs_actual_scatter.png"
    )

    # Plot 5: Absolute Error Heatmap
    abs_error = np.abs(tft_actuals[:comp_samples] - tft_preds[:comp_samples])
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.heatmap(
        abs_error.T,
        cmap="Reds",
        yticklabels=np.arange(1, 13),
        xticklabels=False,
        ax=ax,
        cbar_kws={"label": "Absolute Error (°C)"},
    )
    ax.set_ylabel("Forecast Lead Horizon (Months)")
    ax.set_xlabel("Test Set Sequence Chronology")
    ax.set_title("Multi-Horizon Absolute Error Dynamics")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "elnino_error_heatmap.png", dpi=300)
    plt.close(fig)
    print("Saved El Nino error heatmap to runtime/eval_artifacts/elnino_error_heatmap.png")

    # Plot 6: Quantile Coverage over time (6-month lead)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    time_axis = np.arange(comp_samples)
    ax.plot(
        time_axis, tft_actuals[:comp_samples, 5], "k-", label="Observed (6-Month Lead)", alpha=0.8
    )
    ax.plot(time_axis, tft_preds[:comp_samples, 5], "b-", label="TFT Median")
    ax.fill_between(
        time_axis,
        tft_lower[:comp_samples, 5],
        tft_upper[:comp_samples, 5],
        color="blue",
        alpha=0.15,
        label="TFT 80% CI",
    )
    ax.set_xlabel("Test Samples (Chronological Months)")
    ax.set_ylabel("SST Anomaly (°C)")
    ax.set_title("6-Month Ahead Probabilistic Forecast Coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "elnino_quantile_coverage.png", dpi=300)
    plt.close(fig)
    print("Saved El Nino quantile coverage to runtime/eval_artifacts/elnino_quantile_coverage.png")

    return {
        "tft": m["tft"],
        "lstm": m["lstm"],
        "persistence": m["persistence"],
        "calibration": m["calibration"],
        "quantile_calibration": m["quantile_calibration"],
    }


def evaluate_pollution():
    """Evaluate the Pollution ST-GNN on the test split, in physical units."""
    print("Evaluating Pollution ST-GNN model...")

    ckpt_dir = settings.model.checkpoint_dir / "pollution"
    ckpt_path = ckpt_dir / "stgnn_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError("No ST-GNN pollution model checkpoint found.")
    model = load_pollution_model(ckpt_path, in_channels=1)

    from models.pollution.graph_dataset import build_pollution_splits

    splits = build_pollution_splits(window_size=settings.model.gnn_num_periods)
    norm_stats = splits["norm_stats"]

    # The model was trained with the stats saved at training time; they must match
    saved_stats = json.loads((ckpt_dir / "norm_stats.json").read_text())
    if not np.allclose([saved_stats["mean"], saved_stats["std"]], [norm_stats["mean"], norm_stats["std"]]):
        raise ValueError(f"Pollution data changed since training: {saved_stats} vs {norm_stats}")

    metrics = evaluate_pollution_model(model, splits["test"], norm_stats, AQI_100_PM25_UGM3)

    node_rmse_df = pd.DataFrame(
        {"node_id": range(len(metrics["node_rmse"])), "rmse_ugm3": metrics["node_rmse"]}
    )
    node_rmse_df.to_csv(RESULTS_DIR / "pollution_node_rmse.csv", index=False)

    # Plot node error distribution (stations with at least one test reading)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.histplot(node_rmse_df["rmse_ugm3"].dropna(), kde=True, bins=10, ax=ax, color="purple")
    ax.set_xlabel("RMSE (µg/m³)")
    ax.set_ylabel("Station Count")
    ax.set_title("Node-wise PM2.5 Prediction Error Distribution")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "pollution_dispersion_nodes.png", dpi=300)
    plt.close(fig)

    # Plot confusion matrix for the AQI 100 exceedance threshold
    c = metrics["confusion"]
    cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    labels = [f"< {AQI_100_PM25_UGM3} µg/m³\n(AQI ≤ 100)", f"≥ {AQI_100_PM25_UGM3} µg/m³\n(AQI > 100)"]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Purples",
        xticklabels=labels, yticklabels=labels, ax=ax, cbar=False,
    )
    ax.set_ylabel("Observed")
    ax.set_xlabel("Predicted")
    ax.set_title("Next-Day PM2.5 Exceedance (Observed Station-Days)")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "pollution_confusion_matrix.png", dpi=300)
    plt.close(fig)
    print("Saved pollution confusion matrix to runtime/eval_artifacts/pollution_confusion_matrix.png")

    metrics["norm_stats"] = norm_stats
    metrics["snapshot_counts"] = splits["snapshot_counts"]
    return metrics


def evaluate_climate_feedback():
    """Detect warming-rate change-points in GISTEMP with a piecewise-linear trend."""
    print("Evaluating Climate Feedback via piecewise-linear change-point analysis...")

    temp_path = settings.data.processed_dir / "gistemp_features.parquet"
    monthly = pd.read_parquet(temp_path)
    monthly["time"] = pd.to_datetime(monthly["time"])
    monthly["year"] = monthly["time"].dt.year

    # Annual means of complete years only: reduces month-to-month autocorrelation
    # and avoids a biased, partial final year.
    annual = (
        monthly.groupby("year")["global_temp_anomaly"]
        .agg(["mean", "count"])
        .reset_index()
    )
    annual = annual[annual["count"] == 12].rename(columns={"mean": "global_temp_anomaly"})
    annual = annual[["year", "global_temp_anomaly"]].reset_index(drop=True)

    from models.climate_feedback.changepoint import detect_slope_changepoints

    result = detect_slope_changepoints(
        annual, value_col="global_temp_anomaly", time_col="year", max_changepoints=3, min_segment=10
    )
    changepoints = result["changepoints"]
    slopes_per_decade = [10.0 * s for s in result["segment_slopes"]]
    cp_years = [int(cp["time"]) for cp in changepoints]

    # Save changepoints to CSV
    cp_df = pd.DataFrame(
        {
            "year": cp_years,
            "slope_before_c_per_decade": slopes_per_decade[:-1],
            "slope_after_c_per_decade": slopes_per_decade[1:],
        }
    )
    cp_df.to_csv(RESULTS_DIR / "climate_changepoints.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(
        monthly["time"].dt.year + (monthly["time"].dt.month - 0.5) / 12,
        monthly["global_temp_anomaly"],
        "k-",
        alpha=0.15,
        linewidth=0.6,
        label="Monthly anomaly",
    )
    ax.plot(annual["year"] + 0.5, annual["global_temp_anomaly"], "ko", markersize=2.5, label="Annual mean")
    ax.plot(
        annual["year"] + 0.5,
        result["fitted"],
        "r-",
        linewidth=2,
        label="Piecewise trend (" + ", ".join(f"{s:.2f}" for s in slopes_per_decade) + " °C/decade)",
    )
    for year in cp_years:
        ax.axvline(year + 0.5, color="red", linestyle="--", linewidth=1, label=f"Change-point ({year})")

    ax.set_xlabel("Year")
    ax.set_ylabel("Global Temperature Anomaly (°C)")
    ax.set_title("Warming Rate Regime Changes (Piecewise Analysis)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "climate_changepoint_regression.png", dpi=300)
    plt.close(fig)

    # Save metrics to JSON
    feedback_stats = {
        "method": "continuous piecewise-linear trend, AR(1) errors, K selected by BIC",
        "years": [int(annual["year"].min()), int(annual["year"].max())],
        "changepoints_detected": len(changepoints),
        "changepoint_years": cp_years,
        "segment_slopes_c_per_decade": slopes_per_decade,
        "bic_by_k": {str(k): v for k, v in result["bic_by_k"].items()},
        "ar1_phi": result["ar1_phi"],
    }
    with open(RESULTS_DIR / "climate_feedback_results.json", "w") as f:
        json.dump(feedback_stats, f, indent=2)

    return feedback_stats


def evaluate_nwm():
    """Evaluate Neural Weather Model (NWM) results."""
    print("Evaluating Neural Weather Model...")
    
    nwm_csv = settings.model.checkpoint_dir / "nwm" / "metrics.csv"
    if not nwm_csv.exists():
        raise FileNotFoundError("No NWM metrics.csv found.")
        
    df = pd.read_csv(nwm_csv)
    best_epoch = df["val_loss"].idxmin()
    best_val_loss = df.loc[best_epoch, "val_loss"]
    best_train_loss = df.loc[best_epoch, "train_loss"]
    
    metrics = {
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_train_loss),
        "best_val_loss": float(best_val_loss),
        "final_train_loss": float(df["train_loss"].iloc[-1]),
        "final_val_loss": float(df["val_loss"].iloc[-1]),
    }
    return metrics


def generate_training_loss_curves():
    """Generate and plot training and validation loss curves for all models."""
    print("Generating model training curves...")

    # 1. TFT loss
    tft_epochs, tft_train, tft_val = [], [], []
    metric_files = glob.glob(
        str(settings.log.dir / "lightning_logs" / "version_*" / "metrics.csv")
    )
    if metric_files:
        latest_csv = max(metric_files, key=os.path.getctime)
        try:
            df = pd.read_csv(latest_csv)
            tft_train_df = df[["epoch", "train_loss_epoch"]].dropna()
            tft_val_df = df[["epoch", "val_loss"]].dropna()
            tft_epochs = tft_train_df["epoch"].tolist()
            tft_train = tft_train_df["train_loss_epoch"].tolist()
            tft_val = tft_val_df["val_loss"].tolist()
        except Exception:
            pass


    # 2. LSTM loss
    lstm_epochs, lstm_train, lstm_val = [], [], []
    lstm_csv = settings.model.checkpoint_dir / "el_nino" / "lstm" / "metrics.csv"
    if lstm_csv.exists():
        try:
            df = pd.read_csv(lstm_csv)
            lstm_epochs = df["epoch"].tolist()
            lstm_train = df["train_loss"].tolist()
            lstm_val = df["val_loss"].dropna().tolist()
        except Exception:
            pass


    # 3. ST-GNN loss
    gnn_epochs, gnn_train, gnn_val = [], [], []
    gnn_csv = settings.model.checkpoint_dir / "pollution" / "metrics.csv"
    if gnn_csv.exists():
        try:
            df = pd.read_csv(gnn_csv)
            gnn_epochs = df["epoch"].tolist()
            gnn_train = df["train_loss"].tolist()
            gnn_val = df["val_loss"].dropna().tolist()
        except Exception:
            pass


    # 4. NWM loss
    nwm_epochs, nwm_train, nwm_val = [], [], []
    nwm_csv = settings.model.checkpoint_dir / "nwm" / "metrics.csv"
    if nwm_csv.exists():
        try:
            df = pd.read_csv(nwm_csv)
            nwm_epochs = df["epoch"].tolist()
            nwm_train = df["train_loss"].tolist()
            nwm_val = df["val_loss"].dropna().tolist()
        except Exception:
            pass


    # Plot four subplots (2x2 panel). A missing training log leaves its panel empty
    # with a note; never substitute invented curves.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)
    axes = axes.flatten()
    for ax, epochs in zip(axes, [tft_epochs, lstm_epochs, gnn_epochs, nwm_epochs], strict=True):
        if len(epochs) < 2:
            ax.text(0.5, 0.5, "No training log found", ha="center", va="center",
                    transform=ax.transAxes, color="gray")

    # TFT subplot
    axes[0].plot(tft_epochs, tft_train, "b-", label="Train Loss")
    axes[0].plot(tft_epochs[: len(tft_val)], tft_val, "r--", label="Val Loss")
    axes[0].set_title("Temporal Fusion Transformer (TFT)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (Quantile Loss / MSE)")
    axes[0].legend()

    # LSTM subplot
    axes[1].plot(lstm_epochs, lstm_train, "b-", label="Train Loss")
    axes[1].plot(lstm_epochs[: len(lstm_val)], lstm_val, "r--", label="Val Loss")
    axes[1].set_title("LSTM Baseline")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss (MSE)")
    axes[1].legend()

    # ST-GNN subplot
    axes[2].plot(gnn_epochs, gnn_train, "b-", label="Train Loss")
    axes[2].plot(gnn_epochs[: len(gnn_val)], gnn_val, "r--", label="Val Loss")
    axes[2].set_title("Spatio-Temporal GNN (Pollution)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss (MSE)")
    axes[2].legend()
    
    # NWM subplot
    axes[3].plot(nwm_epochs, nwm_train, "b-", label="Train Loss")
    axes[3].plot(nwm_epochs[: len(nwm_val)], nwm_val, "r--", label="Val Loss")
    axes[3].set_title("Neural Weather Model (ST-GNN)")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("Loss (MSE)")
    axes[3].legend()

    fig.suptitle("ClimatePulse Deep Learning Training Dynamics")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "model_training_curves.png", dpi=300)
    plt.close(fig)
    print("Saved training curves to runtime/eval_artifacts/model_training_curves.png")


# ======================================================================
# NWM-Specific Publication Figures
# ======================================================================


def _plot_nwm_convergence():
    """Plot 4: Dedicated NWM loss curve with reconstructed LR schedule."""
    nwm_csv = settings.model.checkpoint_dir / "nwm" / "metrics.csv"
    if not nwm_csv.exists():
        print("  SKIP: NWM metrics.csv not found.")
        return

    df = pd.read_csv(nwm_csv)
    epochs = df["epoch"].values
    train_loss = df["train_loss"].values
    val_loss = df["val_loss"].values

    best_idx = int(df["val_loss"].idxmin())
    best_epoch = int(df.loc[best_idx, "epoch"])
    best_val = float(df.loc[best_idx, "val_loss"])

    # Reconstruct ReduceLROnPlateau schedule (factor=0.5, patience=3)
    lr = float(settings.model.learning_rate)
    lr_schedule = []
    best_so_far = float("inf")
    patience_ctr = 0

    for v in val_loss:
        if v < best_so_far - 1e-8:
            best_so_far = v
            patience_ctr = 0
        else:
            patience_ctr += 1

        lr_schedule.append(lr)

        if patience_ctr >= 3:
            lr *= 0.5
            patience_ctr = 0

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    ax1.plot(epochs, train_loss, color="#1f77b4", linewidth=1.5,
             label="Training Loss", alpha=0.9)
    ax1.plot(epochs, val_loss, color="#d62728", linewidth=1.5,
             linestyle="--", label="Validation Loss", alpha=0.9)
    ax1.axvline(best_epoch, color="#2ca02c", linestyle=":", alpha=0.7,
                label=f"Best Epoch ({best_epoch})")
    ax1.scatter([best_epoch], [best_val], color="#2ca02c", zorder=5, s=60,
                edgecolors="white", linewidth=0.5)

    # Annotate the best point
    ax1.annotate(
        f"Val MSE = {best_val:.4f}",
        xy=(best_epoch, best_val),
        xytext=(best_epoch + 3, best_val + 0.015),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=0.8),
        color="#2ca02c",
    )

    # Early stopping window annotation
    last_epoch = int(epochs[-1])
    if last_epoch > best_epoch:
        ax1.axvspan(best_epoch, last_epoch, alpha=0.06, color="red")
        ax1.text(
            best_epoch + (last_epoch - best_epoch) / 2,
            ax1.get_ylim()[1] * 0.95,
            "Early\nStopping\nWindow",
            ha="center", va="top", fontsize=7, color="#d62728", alpha=0.7,
        )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE)")
    ax1.set_title("Neural Weather Model: Training Convergence & Learning Rate Schedule")

    # LR on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(epochs, lr_schedule, color="#9467bd", linewidth=1.0, alpha=0.5,
             label="Learning Rate")
    ax2.set_ylabel("Learning Rate", color="#9467bd")
    ax2.tick_params(axis="y", labelcolor="#9467bd")
    ax2.set_yscale("log")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "nwm_convergence_lr.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved nwm_convergence_lr.png")


def _plot_fibonacci_mesh(graph_builder):
    """Plot 3: Fibonacci mesh topology — 3D sphere + 2D equirectangular projection."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    mesh = graph_builder.mesh_coords       # [2000, 3]
    grid = graph_builder.grid_coords       # [num_grid, 3]
    m2m = graph_builder.build_mesh_to_mesh(k=8)

    fig = plt.figure(figsize=(12, 5))

    # ---- Left panel: 3D sphere ----
    ax1 = fig.add_subplot(121, projection="3d")

    # Semi-transparent wireframe sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v_s = np.linspace(0, np.pi, 20)
    xs = 0.98 * np.outer(np.cos(u), np.sin(v_s))
    ys = 0.98 * np.outer(np.sin(u), np.sin(v_s))
    zs = 0.98 * np.outer(np.ones_like(u), np.cos(v_s))
    ax1.plot_wireframe(xs, ys, zs, color="lightgray", alpha=0.12, linewidth=0.3)

    # Mesh nodes
    ax1.scatter(mesh[:, 0], mesh[:, 1], mesh[:, 2],
                c="royalblue", s=4, alpha=0.8, label=f"Mesh ({len(mesh)})")

    # Draw a subset of mesh-to-mesh edges for visual clarity
    esrc = m2m[0].numpy()
    etgt = m2m[1].numpy()
    step = max(1, len(esrc) // 3000)
    for i in range(0, len(esrc), step):
        s, t = esrc[i], etgt[i]
        ax1.plot(
            [mesh[s, 0], mesh[t, 0]],
            [mesh[s, 1], mesh[t, 1]],
            [mesh[s, 2], mesh[t, 2]],
            "b-", alpha=0.04, linewidth=0.3,
        )

    ax1.set_title("3D Fibonacci Lattice Mesh", fontsize=10)
    ax1.set_xlim([-1.1, 1.1])
    ax1.set_ylim([-1.1, 1.1])
    ax1.set_zlim([-1.1, 1.1])
    ax1.set_box_aspect([1, 1, 1])
    ax1.axis("off")

    # ---- Right panel: 2D equirectangular ----
    ax2 = fig.add_subplot(122)

    # Convert 3D → geographic lat/lon using the grid convention (z = sin(lat))
    grid_lat = np.degrees(np.arcsin(np.clip(grid[:, 2], -1, 1)))
    grid_lon = np.degrees(np.arctan2(grid[:, 1], grid[:, 0]))

    # For Fibonacci mesh: y is the polar axis, so lat = arcsin(y)
    mesh_lat = np.degrees(np.arcsin(np.clip(mesh[:, 1], -1, 1)))
    mesh_lon = np.degrees(np.arctan2(mesh[:, 2], mesh[:, 0]))

    ax2.scatter(grid_lon, grid_lat, c="lightcoral", s=2, alpha=0.3,
                label=f"Grid nodes ({len(grid)})", zorder=1)
    ax2.scatter(mesh_lon, mesh_lat, c="royalblue", s=4, alpha=0.6,
                label=f"Mesh nodes ({len(mesh)})", zorder=2)

    ax2.set_xlabel("Longitude (°)")
    ax2.set_ylabel("Latitude (°)")
    ax2.set_title("Equirectangular Projection", fontsize=10)
    ax2.legend(fontsize=7, loc="lower left")
    ax2.set_xlim([-180, 180])
    ax2.set_ylim([-90, 90])
    ax2.set_aspect("equal")

    fig.suptitle(
        "NWM Encoder–Processor–Decoder: Fibonacci Spherical Mesh Topology\n"
        f"({len(mesh)} mesh nodes, k=8 connectivity, {len(grid)} grid nodes)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "nwm_fibonacci_mesh.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved nwm_fibonacci_mesh.png")


def _plot_predicted_vs_actual(preds, targets, test_ds, num_lat, num_lon):
    """Plot 1: Side-by-side predicted vs actual T2m field with error map."""
    # Pick a sample near the middle of the test set
    mid = len(preds) // 2

    # out_channels layout: [T2m_step0, ..., T2m_stepN, Precip_step0, ..., Precip_stepN]
    # For prediction_steps=1: channel 0 = T2m, channel 1 = Precip
    pred_t2m = preds[mid, :, 0].reshape(num_lat, num_lon)
    true_t2m = targets[mid, :, 0].reshape(num_lat, num_lon)

    # Denormalize back to Kelvin
    pred_K = pred_t2m * test_ds.t2m_std + test_ds.t2m_mean
    true_K = true_t2m * test_ds.t2m_std + test_ds.t2m_mean
    error_K = pred_K - true_K

    lat = test_ds.lat
    lon = test_ds.lon
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    vmin = min(true_K.min(), pred_K.min())
    vmax = max(true_K.max(), pred_K.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    im0 = axes[0].imshow(true_K, aspect="auto", origin="upper",
                          extent=extent, cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
    axes[0].set_title("ERA5 Ground Truth", fontsize=10)
    axes[0].set_xlabel("Longitude (°)")
    axes[0].set_ylabel("Latitude (°)")
    plt.colorbar(im0, ax=axes[0], shrink=0.8, label="T2m (K)")

    im1 = axes[1].imshow(pred_K, aspect="auto", origin="upper",
                          extent=extent, cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
    axes[1].set_title("NWM Prediction", fontsize=10)
    axes[1].set_xlabel("Longitude (°)")
    axes[1].set_ylabel("Latitude (°)")
    plt.colorbar(im1, ax=axes[1], shrink=0.8, label="T2m (K)")

    # Diverging error map centred on zero
    eabs = max(abs(error_K.min()), abs(error_K.max()), 1e-6)
    im2 = axes[2].imshow(error_K, aspect="auto", origin="upper",
                          extent=extent, cmap="RdBu_r", vmin=-eabs, vmax=eabs)
    axes[2].set_title("Prediction Error", fontsize=10)
    axes[2].set_xlabel("Longitude (°)")
    axes[2].set_ylabel("Latitude (°)")
    plt.colorbar(im2, ax=axes[2], shrink=0.8, label="Error (K)")

    # Add equator line
    for ax in axes:
        ax.axhline(0, color="black", linestyle="--", alpha=0.3, linewidth=0.5)

    rmse_sample = float(np.sqrt(np.mean(error_K ** 2)))
    fig.suptitle(
        f"NWM: 2m Temperature — Predicted vs. ERA5 Ground Truth "
        f"(Test sample RMSE = {rmse_sample:.2f} K)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "nwm_predicted_vs_actual.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved nwm_predicted_vs_actual.png  (sample RMSE = {rmse_sample:.2f} K)")


def _plot_spatial_error_map(preds, targets, test_ds, num_lat, num_lon):
    """Plot 2: Per-grid-node RMSE heatmap across all test samples."""
    # Compute per-node RMSE for T2m across all test samples
    sq_err = (preds[:, :, 0] - targets[:, :, 0]) ** 2   # [N_test, num_grid_nodes]
    rmse_per_node = np.sqrt(sq_err.mean(axis=0))          # [num_grid_nodes]

    # Denormalize to Kelvin scale
    rmse_map_K = rmse_per_node.reshape(num_lat, num_lon) * test_ds.t2m_std

    lat = test_ds.lat
    lon = test_ds.lon
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    fig, ax = plt.subplots(figsize=(9, 4.5))

    im = ax.imshow(rmse_map_K, aspect="auto", origin="upper",
                    extent=extent, cmap="YlOrRd")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")

    # Equator and tropics
    for line_lat, label in [(0, "Equator"), (23.5, "Tropic of Cancer"),
                             (-23.5, "Tropic of Capricorn")]:
        ax.axhline(line_lat, color="white", linestyle="--", alpha=0.5, linewidth=0.5)
        ax.text(lon.max() + 2, line_lat, label, fontsize=6, va="center",
                alpha=0.7, color="gray")

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, label="RMSE (K)")

    # Area-weighted (cos latitude) global RMSE, matching the NWM evaluation metric
    weights = test_ds.lat_weights / test_ds.lat_weights.mean()
    global_rmse = float(np.sqrt((sq_err * weights[None, :]).mean()) * test_ds.t2m_std)
    equatorial = rmse_map_K[num_lat // 3 : 2 * num_lat // 3, :].mean()
    polar = np.concatenate([rmse_map_K[:num_lat // 6, :].flatten(),
                            rmse_map_K[-num_lat // 6:, :].flatten()]).mean()

    ax.set_title(
        f"NWM: Spatial RMSE — 2m Temperature (Test Set)\n"
        f"Global = {global_rmse:.2f} K | Tropical = {equatorial:.2f} K | Polar = {polar:.2f} K",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "nwm_spatial_error_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved nwm_spatial_error_map.png  (global RMSE = {global_rmse:.2f} K)")


def generate_nwm_plots():
    """Generate the NWM figures and return test-set metrics of the main model."""
    print("Generating NWM-specific plots...")

    # Plot 4 (convergence) only needs the metrics CSV
    _plot_nwm_convergence()

    from models.nwm.dataset import ERA5_PATH
    from models.nwm.evaluate import evaluate_nwm_checkpoint
    from models.nwm.graph_builder import SphericalGraphBuilder

    ckpt_path = settings.model.checkpoint_dir / "nwm" / "nwm_best.pt"
    if not ERA5_PATH.exists():
        print("  SKIP: ERA5 NetCDF not found — skipping spatial NWM plots.")
        return {}
    if not ckpt_path.exists():
        print("  SKIP: nwm_best.pt not found — skipping prediction plots.")
        return {}

    m = evaluate_nwm_checkpoint(ckpt_path, settings.nwm.conditioning)
    test_ds, preds, targets = m["dataset"], m["preds"], m["targets"]

    # Plot 3: Fibonacci mesh topology
    _plot_fibonacci_mesh(
        SphericalGraphBuilder(test_ds.lat, test_ds.lon, num_mesh_nodes=settings.nwm.num_mesh_nodes)
    )

    num_lat = len(test_ds.lat)
    num_lon = len(test_ds.lon)
    print(f"  NWM inference: {len(preds)} test samples, grid {num_lat}×{num_lon}")

    # Plot 1: Predicted vs Actual T2m (single sample)
    _plot_predicted_vs_actual(preds, targets, test_ds, num_lat, num_lon)

    # Plot 2: Spatial RMSE heatmap (all test samples)
    _plot_spatial_error_map(preds, targets, test_ds, num_lat, num_lon)

    print("NWM plot generation complete.")
    return {k: v for k, v in m.items() if isinstance(v, (int, float, str))}


def copy_runtime_data():
    """Copy all key runtime-generated data into eval_artifacts/run_results/.

    Collects:
    - Per-model training metrics CSVs (NWM, LSTM, Pollution ST-GNN)
    - TFT Lightning metrics CSV (latest version)
    - NWM checkpoint metadata (file size + path)
    """
    print("Copying runtime data into eval_artifacts/run_results/...")

    # ---- NWM metrics.csv -----------------------------------------------
    nwm_src = settings.model.checkpoint_dir / "nwm" / "metrics.csv"
    if nwm_src.exists():
        dst = RESULTS_DIR / "nwm_metrics.csv"
        shutil.copy2(nwm_src, dst)
        print(f"  Copied {nwm_src.name} -> {dst}")
    else:
        print(f"  WARNING: {nwm_src} not found – skipping.")

    # ---- LSTM metrics.csv -----------------------------------------------
    lstm_src = settings.model.checkpoint_dir / "el_nino" / "lstm" / "metrics.csv"
    if lstm_src.exists():
        dst = RESULTS_DIR / "lstm_metrics.csv"
        shutil.copy2(lstm_src, dst)
        print(f"  Copied {lstm_src.name} -> {dst}")
    else:
        print(f"  WARNING: {lstm_src} not found – skipping.")

    # ---- Pollution ST-GNN metrics.csv -----------------------------------
    pollution_src = settings.model.checkpoint_dir / "pollution" / "metrics.csv"
    if pollution_src.exists():
        dst = RESULTS_DIR / "pollution_metrics.csv"
        shutil.copy2(pollution_src, dst)
        print(f"  Copied {pollution_src.name} -> {dst}")
    else:
        print(f"  WARNING: {pollution_src} not found – skipping.")

    # ---- TFT Lightning metrics.csv (latest version) ---------------------
    metric_files = glob.glob(
        str(settings.log.dir / "lightning_logs" / "version_*" / "metrics.csv")
    )
    if metric_files:
        latest_csv = Path(max(metric_files, key=os.path.getctime))
        dst = RESULTS_DIR / "tft_metrics.csv"
        shutil.copy2(latest_csv, dst)
        print(f"  Copied TFT {latest_csv.parent.name}/metrics.csv -> {dst}")
    else:
        print("  WARNING: No TFT Lightning metrics.csv found – skipping.")

    # ---- NWM checkpoint info (metadata only, not the weight file) -------
    nwm_ckpt = settings.model.checkpoint_dir / "nwm" / "nwm_best.pt"
    if nwm_ckpt.exists():
        size_mb = nwm_ckpt.stat().st_size / 1_048_576
        ckpt_info = {
            "checkpoint_path": str(nwm_ckpt),
            "size_mb": round(size_mb, 3),
        }
        info_path = RESULTS_DIR / "nwm_checkpoint_info.json"
        with open(info_path, "w") as f:
            json.dump(ckpt_info, f, indent=2)
        print(f"  Wrote NWM checkpoint metadata -> {info_path.name}")
    else:
        print("  WARNING: nwm_best.pt not found – skipping checkpoint info.")

    print("Runtime data copy complete.")


def _fmt(summary: dict, *keys, digits: int = 4) -> str:
    """Format a nested summary value, or 'n/a' if missing. Never substitutes numbers."""
    value = summary
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return "n/a"
        value = value[key]
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _seed_summary_md() -> str:
    path = RESULTS_DIR / "elnino_seeds.csv"
    if not path.exists():
        return "*Multi-seed results not available (run `make train-elnino-seeds`).*\n"
    df = pd.read_csv(path)
    lines = [
        f"Mean ± standard deviation over {df['seed'].nunique()} seeds "
        "(TFT variant selected by mean validation loss).\n",
        "| Model | Variant | Val loss | RMSE (°C) | ACC | Skill Score | Bias (°C) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    order = {"tft": 0, "lstm": 1, "persistence": 2}
    df = df.sort_values(["model", "variant"], key=lambda c: c.map(order) if c.name == "model" else c)
    for (model, variant), g in df.groupby(["model", "variant"], sort=False):
        cells = [
            "n/a" if g[c].isna().all() else f"{g[c].mean():.3f} ± {g[c].std():.3f}"
            for c in ["val_loss", "rmse", "acc", "skill_score", "bias"]
        ]
        lines.append(f"| {model} | {variant} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _ablation_md() -> str:
    path = RESULTS_DIR / "nwm_ablation.csv"
    if not path.exists():
        return "*Ablation results not available (run `make nwm-ablation`).*\n"
    df = pd.read_csv(path)
    lines = [
        "| Variant | Conditioning | Area-weighted loss | Epochs | Test T2m RMSE (K) | Test precip RMSE |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]
    for r in df.itertuples():
        lines.append(
            f"| {r.variant} | {r.conditioning} | {r.area_weighted_loss} | {r.epochs} | "
            f"{r.t2m_rmse_K:.3f} | {r.tp_rmse:.3e} |"
        )
    p = df.iloc[0]
    lines.append(
        f"| persistence | — | — | — | {p.persistence_t2m_rmse_K:.3f} | {p.persistence_tp_rmse:.3e} |"
    )
    return "\n".join(lines) + "\n"


def main():
    print("=" * 60)
    print("ClimatePulse Paper Artifacts Generator")
    print("=" * 60)

    summary = {}

    try:
        tft_model, test_ds, df = load_tft_test_data()
        lstm_model, lstm_test_data = load_lstm_test_data()
        summary["elnino"] = evaluate_el_nino(tft_model, test_ds, lstm_model, lstm_test_data)
    except Exception as e:
        print(f"Failed to generate El Nino artifacts: {e}")
        summary["elnino"] = {"status": "failed", "error": str(e)}

    try:
        summary["pollution"] = evaluate_pollution()
    except Exception as e:
        print(f"Failed to generate Pollution artifacts: {e}")
        summary["pollution"] = {"status": "failed", "error": str(e)}

    try:
        summary["climate_feedback"] = evaluate_climate_feedback()
    except Exception as e:
        print(f"Failed to generate Climate Feedback artifacts: {e}")
        summary["climate_feedback"] = {"status": "failed", "error": str(e)}

    try:
        summary["nwm"] = evaluate_nwm()
    except Exception as e:
        print(f"Failed to generate NWM artifacts: {e}")
        summary["nwm"] = {"status": "failed", "error": str(e)}

    try:
        generate_training_loss_curves()
    except Exception as e:
        print(f"Failed to generate training loss curves: {e}")

    try:
        summary["nwm"].update(generate_nwm_plots())
    except Exception as e:
        print(f"Failed to generate NWM-specific plots: {e}")

    # Save overall summary metrics to JSON (scalars, lists and dicts only)
    clean_summary = {
        domain: {k: v for k, v in res.items() if isinstance(v, int | float | str | bool | list | dict)}
        for domain, res in summary.items()
    }
    with open(RESULTS_DIR / "summary_metrics.json", "w") as f:
        json.dump(clean_summary, f, indent=2)

    print("Writing markdown summary metrics table...")
    cs = clean_summary
    rows = "\n".join(
        f"| **{label}** | {_fmt(cs, 'elnino', key, 'rmse')} | {_fmt(cs, 'elnino', key, 'mae')} | "
        f"{_fmt(cs, 'elnino', key, 'acc')} | {_fmt(cs, 'elnino', key, 'skill_score')} | "
        f"{_fmt(cs, 'elnino', key, 'bias')} |"
        for label, key in [
            ("Temporal Fusion Transformer (TFT)", "tft"),
            ("LSTM Baseline", "lstm"),
            ("Persistence Baseline", "persistence"),
        ]
    )
    slopes = cs.get("climate_feedback", {}).get("segment_slopes_c_per_decade", [])
    md_content = f"""# ClimatePulse Model Evaluation Metrics

This document summarizes the performance metrics generated during the evaluation phase of the trained forecasting and regression models. "n/a" marks a result that could not be computed.

---

## 1. El Niño Forecasting (SST Anomaly)

### Main run (figures)

| Model | RMSE (°C) | MAE (°C) | Anomaly Correlation (ACC) | Skill Score | Bias (°C) |
| :--- | :---: | :---: | :---: | :---: | :---: |
{rows}

*Skill Score = 1 − MSE(model) / MSE(persistence), where persistence carries the last observed value across all 12 leads. TFT 80 % interval coverage: {_fmt(cs, 'elnino', 'calibration', 'actual_coverage', digits=3)}.*

### Across seeds

{_seed_summary_md()}
---

## 2. Spatio-Temporal Pollution Dispersion (next-day PM2.5)

| Metric | Model | Persistence | Always alert |
| :--- | :---: | :---: | :---: |
| **RMSE (µg/m³)** | {_fmt(cs, 'pollution', 'rmse_ugm3', digits=2)} | {_fmt(cs, 'pollution', 'persistence_rmse_ugm3', digits=2)} | — |
| **F1, PM2.5 ≥ {AQI_100_PM25_UGM3} µg/m³ (AQI > 100)** | {_fmt(cs, 'pollution', 'f1')} | {_fmt(cs, 'pollution', 'persistence_f1')} | {_fmt(cs, 'pollution', 'always_alert_f1')} |
| **Precision / Recall** | {_fmt(cs, 'pollution', 'precision')} / {_fmt(cs, 'pollution', 'recall')} | — | — |

*Observed station-days: {_fmt(cs, 'pollution', 'n_observed')} of {_fmt(cs, 'pollution', 'n_total')}; exceedance rate {_fmt(cs, 'pollution', 'exceedance_rate', digits=3)}. Missing readings are excluded.*

---

## 3. Neural Weather Model (NWM)

| Metric | Value |
| :--- | :---: |
| **Best Validation Loss (MSE, normalized)** | {_fmt(cs, 'nwm', 'best_val_loss')} |
| **Final Training Loss (MSE, normalized)** | {_fmt(cs, 'nwm', 'final_train_loss')} |
| **Test 2 m Temperature RMSE (K, area-weighted)** | {_fmt(cs, 'nwm', 't2m_rmse_K', digits=3)} |
| **Persistence 2 m Temperature RMSE (K)** | {_fmt(cs, 'nwm', 'persistence_t2m_rmse_K', digits=3)} |

### Ablation

{_ablation_md()}
---

## 4. Climate Feedback Change-Points

* **Method**: {_fmt(cs, 'climate_feedback', 'method')}
* **Detected Breaks**: {_fmt(cs, 'climate_feedback', 'changepoints_detected')}
* **Change-Point Years**: {", ".join(map(str, cs.get('climate_feedback', {}).get('changepoint_years', []))) or 'none'}
* **Segment Warming Rates (°C/decade)**: {", ".join(f"{v:.3f}" for v in slopes) or 'n/a'}
"""
    with open(ARTIFACTS_DIR / "metrics_summary.md", "w") as f:
        f.write(md_content)

    try:
        copy_runtime_data()
    except Exception as e:
        print(f"Failed to copy runtime data: {e}")

    print("=" * 60)
    print(f"Artifact generation completed! Output directory: {ARTIFACTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
