"""
Multi-seed El Nino evaluation, with and without the calendar-year covariate.

Trains the TFT (per variant) and the LSTM over several seeds into
<checkpoint_dir>/el_nino_seeds/, evaluates every run on the test period, and
writes runtime/eval_artifacts/run_results/elnino_seeds.csv. The TFT variant to
use is chosen by mean *validation* loss, never by test results.

Usage:
    PYTHONPATH=src python src/scripts/run_elnino_seeds.py [--seeds 0 1 2 3 4] [--skip-training]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import PROJECT_ROOT, get_settings, settings  # noqa: E402

TFT_VARIANTS = {"with_year": "true", "without_year": "false"}
RESULTS_PATH = PROJECT_ROOT / "runtime" / "eval_artifacts" / "run_results" / "elnino_seeds.csv"
SELECTION_PATH = RESULTS_PATH.with_name("elnino_seed_selection.json")


def _run_dir(name: str, seed: int) -> Path:
    return settings.model.checkpoint_dir / "el_nino_seeds" / name / f"seed_{seed}"


def _train(model: str, run_dir: Path, seed: int, extra_env: dict[str, str]) -> None:
    env = {
        **os.environ,
        **extra_env,
        "MODEL_CHECKPOINT_DIR": str(run_dir),
        "MODEL_SEED": str(seed),
        "LOG_DIR": str(run_dir / "logs"),
        "PYTHONPATH": str(PROJECT_ROOT / "src"),
    }
    script = PROJECT_ROOT / "src" / "scripts" / "train_models.py"
    subprocess.run([sys.executable, str(script), "--model", model], env=env, check=True)


def _evaluate(variant: str, use_year: str, seed: int) -> list[dict]:
    from models.el_nino.dataset import (
        create_tft_dataset,
        lstm_test_series,
        prepare_tft_dataframe,
        split_tft_dataset,
    )
    from models.el_nino.evaluate import elnino_metrics, elnino_test_predictions
    from models.el_nino.lstm_baseline import load_lstm_model
    from models.el_nino.tft_model import find_best_tft_checkpoint, load_tft_model

    os.environ["MODEL_TFT_USE_YEAR"] = use_year
    get_settings.cache_clear()

    tft_ckpt = find_best_tft_checkpoint(_run_dir(variant, seed) / "el_nino" / "tft")
    tft_val = float(tft_ckpt.stem.split("val_loss=")[1].split("-v")[0])
    lstm_dir = _run_dir("lstm", seed) / "el_nino" / "lstm"
    lstm_val = float(pd.read_csv(lstm_dir / "metrics.csv")["val_loss"].min())

    df = prepare_tft_dataframe()
    _, _, test_ds = split_tft_dataset(create_tft_dataset(df), df)
    preds = elnino_test_predictions(
        load_tft_model(tft_ckpt), test_ds, load_lstm_model(lstm_dir / "lstm_best.pt"), lstm_test_series()
    )
    m = elnino_metrics(preds)

    rows = []
    for model, val in (("tft", tft_val), ("lstm", lstm_val), ("persistence", np.nan)):
        r = {"variant": variant, "seed": seed, "model": model, "val_loss": val, **m[model]}
        if model == "tft":
            r["interval_coverage"] = m["calibration"]["actual_coverage"]
        rows.append(r)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    if not args.skip_training:
        for seed in args.seeds:
            _train("lstm", _run_dir("lstm", seed), seed, {})
            for variant, use_year in TFT_VARIANTS.items():
                _train("tft", _run_dir(variant, seed), seed, {"MODEL_TFT_USE_YEAR": use_year})

    rows = []
    for seed in args.seeds:
        for variant, use_year in TFT_VARIANTS.items():
            rows += _evaluate(variant, use_year, seed)
    results = pd.DataFrame(rows)
    # The LSTM and persistence do not depend on the TFT variant; keep one copy
    results = results[(results["model"] == "tft") | (results["variant"] == "with_year")]
    results.loc[results["model"] != "tft", "variant"] = "-"

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)

    cols = ["val_loss", "rmse", "acc", "skill_score", "bias", "interval_coverage"]
    summary = results.groupby(["model", "variant"])[cols].agg(["mean", "std"]).round(4)
    print(summary.to_string())

    tft = results[results["model"] == "tft"].groupby("variant")["val_loss"].mean()
    selected = tft.idxmin()
    SELECTION_PATH.write_text(json.dumps({
        "selected_variant": selected,
        "tft_use_year": TFT_VARIANTS[selected] == "true",
        "mean_val_loss": tft.to_dict(),
        "seeds": args.seeds,
    }, indent=2))
    print(f"\nSelected TFT variant by mean validation loss: {selected} ({tft.to_dict()})")
    print(f"Wrote {RESULTS_PATH} and {SELECTION_PATH.name}")


if __name__ == "__main__":
    main()
