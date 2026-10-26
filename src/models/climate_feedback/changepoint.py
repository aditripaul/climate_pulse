"""
Change-point detection for climate feedback analysis.

Identifies structural breaks (regime shifts) in global warming rates
using Bayesian change-point detection.
"""

from __future__ import annotations

from itertools import combinations

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


def _fit_ar1(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, float]:
    """
    Fit y ~ X with AR(1) errors by two-step Prais-Winsten GLS.

    Returns:
        (exact Gaussian log-likelihood of y, coefficients, AR(1) coefficient phi)
    """
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    phi = float(np.clip(np.sum(resid[1:] * resid[:-1]) / np.sum(resid**2), -0.99, 0.99))

    # Prais-Winsten transform whitens the errors; its Jacobian is sqrt(1 - phi^2).
    s = np.sqrt(1.0 - phi**2)
    Xs = np.vstack([s * X[:1], X[1:] - phi * X[:-1]])
    ys = np.concatenate([s * y[:1], y[1:] - phi * y[:-1]])
    beta = np.linalg.lstsq(Xs, ys, rcond=None)[0]
    sigma2 = float(np.mean((ys - Xs @ beta) ** 2))

    n = len(y)
    loglik = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1.0) + np.log(s)
    return float(loglik), beta, phi


def _hinge_design(t: np.ndarray, breaks: list[float]) -> np.ndarray:
    """Design matrix for a continuous piecewise-linear trend with the given breaks."""
    cols = [np.ones_like(t), t] + [np.maximum(0.0, t - b) for b in breaks]
    return np.column_stack(cols)


def detect_slope_changepoints(
    df: pd.DataFrame,
    value_col: str = "global_temp_anomaly",
    time_col: str = "year",
    max_changepoints: int = 3,
    min_segment: int = 10,
) -> dict:
    """
    Detect changes in the warming *rate* with a continuous piecewise-linear trend.

    For each K = 0..max_changepoints, every admissible set of K breakpoints is
    fitted exhaustively (segments at least `min_segment` observations long) with
    AR(1) errors, and K is chosen by BIC. Autocorrelated errors matter: an
    i.i.d. likelihood on climate series overstates the evidence for extra breaks.

    Args:
        df: Time-ordered DataFrame, typically annual means.
        value_col: Column containing the values.
        time_col: Numeric time column (e.g. year); slopes are per unit of it.
        max_changepoints: Largest number of breakpoints considered.
        min_segment: Minimum number of observations per segment.

    Returns:
        Dictionary with the selected changepoints, per-segment slopes, the BIC
        for every K, the AR(1) coefficient and the fitted trend.
    """
    y = np.asarray(df[value_col].values, dtype=np.float64)
    t = np.asarray(df[time_col].values, dtype=np.float64)
    n = len(y)

    candidates = range(min_segment, n - min_segment + 1)
    best_by_k: dict[int, tuple[float, tuple[int, ...], np.ndarray, float]] = {}

    for k in range(max_changepoints + 1):
        best = None
        for idx in combinations(candidates, k):
            if any(b - a < min_segment for a, b in zip(idx, idx[1:], strict=False)):
                continue
            X = _hinge_design(t, [t[i] for i in idx])
            loglik, beta, phi = _fit_ar1(X, y)
            # Parameters: intercept, slope, K slope changes, K locations, sigma, phi
            bic = -2.0 * loglik + (4 + 2 * k) * np.log(n)
            if best is None or bic < best[0]:
                best = (bic, idx, beta, phi)
        if best is not None:
            best_by_k[k] = best

    k_best = min(best_by_k, key=lambda k: best_by_k[k][0])
    bic, idx, beta, phi = best_by_k[k_best]
    breaks = [t[i] for i in idx]
    slopes = np.cumsum(beta[1:]).tolist()  # slope of each segment

    changepoints = [
        {"index": int(i), "time": float(t[i]), "slope_before": slopes[j], "slope_after": slopes[j + 1]}
        for j, i in enumerate(idx)
    ]

    logger.info(
        f"Slope changepoints (K={k_best} by BIC, phi={phi:.2f}): "
        f"{[cp['time'] for cp in changepoints]}, segment slopes={np.round(slopes, 4).tolist()}"
    )

    return {
        "changepoints": changepoints,
        "n_changepoints": k_best,
        "segment_slopes": slopes,
        "bic_by_k": {k: float(v[0]) for k, v in best_by_k.items()},
        "ar1_phi": phi,
        "fitted": _hinge_design(t, breaks) @ beta,
    }


def detect_changepoints_simple(
    df: pd.DataFrame,
    value_col: str = "global_temp_anomaly",
    window: int = 10,
) -> list[dict]:
    """
    Simple CUSUM-based change-point detection (non-Bayesian fallback).

    Detects shifts in the *mean level* via recursive binary segmentation. It is
    not suited to trending series such as warming temperatures: a steady trend
    looks like a mean shift. Use detect_slope_changepoints() for rate changes.

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

    # Find points of maximum deviation
    changepoints = []

    # Recursive binary segmentation
    def find_cp(start: int, end: int, depth: int = 0):
        if end - start < 2 * window or depth > 3:
            return

        # CUSUM of deviations from the segment's own mean
        segment = np.cumsum(y[start:end] - y[start:end].mean())
        diff = np.max(segment) - np.min(segment)

        # Significance threshold (heuristic)
        threshold = 2 * np.std(y[start:end]) * np.sqrt(end - start)

        if diff > threshold:
            # The mean shifts after the observation where |CUSUM| peaks
            cp_idx = int(start + np.argmax(np.abs(segment)) + 1)

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
