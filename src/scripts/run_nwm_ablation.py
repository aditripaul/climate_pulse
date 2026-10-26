"""
Neural Weather Model ablation: conditioning source and area-weighted loss.

Variants (each differs from the main model in one respect):
    observed      main model: observed Nino 3.4, area-weighted loss (<checkpoint_dir>/nwm)
    forecast      conditioned on TFT lead-1 forecasts (needs export_tft_forecasts.py)
    unconditioned Nino 3.4 input fixed at zero
    unweighted    observed Nino 3.4, unweighted MSE loss

Training runs in parallel subprocesses; evaluation scores every variant on the
test period and writes runtime/eval_artifacts/run_results/nwm_ablation.csv.

Usage:
    PYTHONPATH=src python src/scripts/run_nwm_ablation.py [--train] [--include-main] [--threads N]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from config import PROJECT_ROOT, settings  # noqa: E402

VARIANTS = {
    "observed": {"NWM_CONDITIONING": "observed", "NWM_AREA_WEIGHTED_LOSS": "true"},
    "forecast": {"NWM_CONDITIONING": "forecast", "NWM_AREA_WEIGHTED_LOSS": "true"},
    "unconditioned": {"NWM_CONDITIONING": "none", "NWM_AREA_WEIGHTED_LOSS": "true"},
    "unweighted": {"NWM_CONDITIONING": "observed", "NWM_AREA_WEIGHTED_LOSS": "false"},
}
RESULTS_PATH = PROJECT_ROOT / "runtime" / "eval_artifacts" / "run_results" / "nwm_ablation.csv"


def _checkpoint_root(variant: str) -> Path:
    if variant == "observed":
        return settings.model.checkpoint_dir
    return settings.model.checkpoint_dir / "nwm_ablation" / variant


def _train_all(variants: list[str], threads: int) -> None:
    script = PROJECT_ROOT / "src" / "scripts" / "train_models.py"
    procs = []
    for variant in variants:
        root = _checkpoint_root(variant)
        root.mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            **VARIANTS[variant],
            "MODEL_CHECKPOINT_DIR": str(root),
            # Separate metadata and logs: concurrent writers would corrupt shared files
            "META_METADATA_PATH": str(root / "nwm_run_metadata.json"),
            "LOG_DIR": str(root / "logs"),
            "OMP_NUM_THREADS": str(threads),
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
        }
        log = open(root / f"train_nwm_{variant}.log", "w")
        procs.append(
            (variant, log, subprocess.Popen(
                [sys.executable, str(script), "--model", "nwm"], env=env, stdout=log, stderr=log
            ))
        )
        print(f"Started {variant} (pid {procs[-1][2].pid}), log: {log.name}")

    failed = []
    for variant, log, proc in procs:
        if proc.wait() != 0:
            failed.append(variant)
        log.close()
    if failed:
        raise SystemExit(f"NWM training failed for: {', '.join(failed)}")


def _evaluate(variant: str) -> dict:
    from models.nwm.evaluate import evaluate_nwm_checkpoint

    ckpt_dir = _checkpoint_root(variant) / "nwm"
    history = pd.read_csv(ckpt_dir / "metrics.csv")
    conditioning = VARIANTS[variant]["NWM_CONDITIONING"]
    m = evaluate_nwm_checkpoint(ckpt_dir / "nwm_best.pt", conditioning)
    return {
        "variant": variant,
        "conditioning": conditioning,
        "area_weighted_loss": VARIANTS[variant]["NWM_AREA_WEIGHTED_LOSS"] == "true",
        "epochs": len(history),
        "best_val_loss": float(history["val_loss"].min()),
        **{k: m[k] for k in ("t2m_rmse_K", "tp_rmse", "persistence_t2m_rmse_K", "persistence_tp_rmse")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train", action="store_true", help="Train the ablation variants first")
    parser.add_argument("--include-main", action="store_true", help="Also (re)train the main model")
    parser.add_argument("--threads", type=int, default=4, help="Torch threads per training process")
    args = parser.parse_args()

    if args.train:
        to_train = [v for v in VARIANTS if v != "observed" or args.include_main]
        _train_all(to_train, args.threads)

    results = pd.DataFrame([_evaluate(v) for v in VARIANTS])
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    print(results.round(4).to_string(index=False))
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
