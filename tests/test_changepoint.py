"""Change-point detection on synthetic series with known structure."""

import numpy as np
import pandas as pd

from models.climate_feedback.changepoint import detect_changepoints_simple, detect_slope_changepoints

YEARS = np.arange(1950, 2026, dtype=float)


def _series(trend: np.ndarray, noise_sd: float = 0.02, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"year": YEARS, "v": trend + rng.normal(0, noise_sd, len(YEARS))})


def test_straight_line_has_no_slope_break():
    result = detect_slope_changepoints(_series(0.015 * (YEARS - 1950)), value_col="v")
    assert result["n_changepoints"] == 0


def test_recovers_slope_break_and_rates():
    trend = np.where(YEARS < 1975, 0.0, 0.02 * (YEARS - 1975))
    result = detect_slope_changepoints(_series(trend), value_col="v")
    assert result["n_changepoints"] == 1
    assert abs(result["changepoints"][0]["time"] - 1975) <= 2
    before, after = result["segment_slopes"]
    assert abs(before) < 0.003 and abs(after - 0.02) < 0.003


def test_cusum_finds_interior_mean_shift():
    # Regression test: the peak location used to land on the series endpoint,
    # silently returning no change-points.
    y = np.r_[np.zeros(100), np.ones(100)] + np.random.default_rng(0).normal(0, 0.1, 200)
    cps = detect_changepoints_simple(pd.DataFrame({"global_temp_anomaly": y}), window=10)
    assert any(abs(cp["index"] - 100) <= 2 for cp in cps)
