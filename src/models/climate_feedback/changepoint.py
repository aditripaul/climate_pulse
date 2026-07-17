"""
Change-point detection for climate feedback analysis.

Identifies structural breaks (regime shifts) in global warming rates
using Bayesian change-point detection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def detect_changepoints_bayesian(
    df: pd.DataFrame,
    value_col: str = "global_temp_anomaly",
    n_changepoints: int = 2,
    draws: int = 2000,
) -> dict:
    """
    Bayesian change-point detection using PyMC.

    Models the time series as piecewise linear segments with
    unknown breakpoints.

    Args:
        df: DataFrame sorted by time with temperature anomaly.
        value_col: Column containing the values.
        n_changepoints: Number of change-points to detect.
        draws: MCMC draws.

    Returns:
        Dictionary with detected changepoint locations and posterior info.
    """
    import pymc as pm

    y = df[value_col].values.astype(np.float64)
    t = np.arange(len(y), dtype=np.float64)
    n = len(y)

    logger.info(f"Detecting {n_changepoints} changepoints in {n} observations")

    with pm.Model() as cp_model:  # type: ignore[bad-context-manager]
        # Change-point locations (as continuous indices)
        tau = pm.Uniform("tau", lower=0, upper=n - 1, shape=n_changepoints)
        tau_sorted = pm.Deterministic("tau_sorted", pm.math.sort(tau))

        # Segment-specific slopes and intercepts
        slopes = pm.Normal("slopes", mu=0, sigma=0.01, shape=n_changepoints + 1)
        intercepts = pm.Normal("intercepts", mu=0, sigma=1, shape=n_changepoints + 1)
        sigma = pm.HalfNormal("sigma", sigma=0.5)

        # Build piecewise linear function
        mu = intercepts[0] + slopes[0] * t
        for k in range(n_changepoints):
            segment_contrib = (slopes[k + 1] - slopes[k]) * pm.math.maximum(0, t - tau_sorted[k])
            mu = mu + segment_contrib

        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(
            draws=draws,
            tune=1000,
            chains=2,
            random_seed=42,
            return_inferencedata=True,
            progressbar=True,
        )

    # Extract changepoint estimates
    tau_posterior = trace.posterior["tau_sorted"].values.reshape(-1, n_changepoints)
    tau_mean = tau_posterior.mean(axis=0)
    tau_std = tau_posterior.std(axis=0)

    # Map indices to dates
    changepoints = []
    for i in range(n_changepoints):
        idx = int(round(tau_mean[i]))
        idx = max(0, min(idx, len(df) - 1))
        changepoints.append(
            {
                "index": idx,
                "time": str(df.iloc[idx].get("time", idx)),
                "mean_idx": float(tau_mean[i]),
                "std_idx": float(tau_std[i]),
            }
        )

    logger.info(f"Detected changepoints: {[cp['time'] for cp in changepoints]}")

    return {
        "changepoints": changepoints,
        "trace": trace,
        "n_changepoints": n_changepoints,
    }


def detect_changepoints_simple(
    df: pd.DataFrame,
    value_col: str = "global_temp_anomaly",
    window: int = 10,
) -> list[dict]:
    """
    Simple CUSUM-based change-point detection (non-Bayesian fallback).

    Uses cumulative sum of deviations from mean to find structural breaks.

    Args:
        df: DataFrame with time-ordered values.
        value_col: Column to analyze.
        window: Minimum segment length.

    Returns:
        List of detected changepoints with indices and significance.
    """
    y = np.asarray(df[value_col].values, dtype=np.float64)
    n = len(y)

    if n < 2 * window:
        return []

    # CUSUM
    mean_val = np.mean(y)
    cusum = np.cumsum(y - mean_val)

    # Find points of maximum deviation
    changepoints = []

    # Recursive binary segmentation
    def find_cp(start: int, end: int, depth: int = 0):
        if end - start < 2 * window or depth > 3:
            return

        segment = cusum[start:end] - cusum[start]
        diff = np.max(segment) - np.min(segment)

        # Significance threshold (heuristic)
        threshold = 2 * np.std(y[start:end]) * np.sqrt(end - start)

        if diff > threshold:
            cp_idx = int(start + np.argmax(np.abs(segment - segment.mean())))

            if window < cp_idx - start and cp_idx < end - window:
                changepoints.append(
                    {
                        "index": cp_idx,
                        "time": str(df.iloc[cp_idx].get("time", cp_idx)),
                        "significance": float(diff / threshold),
                    }
                )

                find_cp(start, cp_idx, depth + 1)
                find_cp(cp_idx, end, depth + 1)

    find_cp(0, n)
    changepoints.sort(key=lambda x: int(str(x["index"])))

    logger.info(f"Simple changepoint detection found {len(changepoints)} breaks")
    return changepoints
