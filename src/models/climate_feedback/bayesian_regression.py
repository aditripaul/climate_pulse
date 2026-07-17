"""
Bayesian Linear Regression for climate feedback analysis.

Models the relationship: Temperature ~ β₀ + β₁·CO₂ + β₂·CH₄ + β₃·N₂O + ε
using PyMC for posterior inference and uncertainty quantification.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


def build_bayesian_model(df: pd.DataFrame, target_col: str = "global_temp_anomaly"):
    """
    Build a Bayesian linear regression model for emissions–temperature.

    Args:
        df: Aligned DataFrame with emissions columns and temperature target.
        target_col: Column name for the temperature anomaly target.

    Returns:
        PyMC model object.
    """
    import pymc as pm

    # Identify predictor columns (emissions gases)
    predictor_cols = [c for c in df.columns if c not in [target_col, "year", "time"]]
    X = df[predictor_cols].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)

    # Standardize predictors for better sampling
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_scaled = (X - X_mean) / X_std

    logger.info(
        f"Building Bayesian model: {len(predictor_cols)} predictors, " f"{len(y)} observations"
    )

    with pm.Model() as model:  # type: ignore[bad-context-manager]
        # Priors
        intercept = pm.Normal("intercept", mu=0, sigma=1)
        betas = pm.Normal("betas", mu=0, sigma=1, shape=len(predictor_cols))
        sigma = pm.HalfNormal("sigma", sigma=0.5)

        # Linear model
        mu = intercept + pm.math.dot(X_scaled, betas)

        # Likelihood
        pm.Normal("temperature", mu=mu, sigma=sigma, observed=y)

        # Store metadata for interpretation
        model._predictor_cols = predictor_cols
        model._X_mean = X_mean
        model._X_std = X_std

    return model


def fit_bayesian_model(
    model,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    random_seed: int = 42,
) -> Any:
    """
    Sample from the posterior using MCMC (NUTS sampler).

    Returns:
        InferenceData (arviz) trace object.
    """
    import pymc as pm

    logger.info(f"Sampling: {draws} draws, {tune} tuning, {chains} chains")

    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=True,
        )

    logger.info("MCMC sampling complete")
    return trace


def summarize_posterior(trace, model) -> pd.DataFrame:
    """
    Summarize posterior distributions for regression coefficients.

    Returns:
        DataFrame with mean, sd, hdi_3%, hdi_97% for each parameter.
    """
    import arviz as az

    summary = az.summary(trace, var_names=["intercept", "betas", "sigma"])

    # Map beta indices to predictor names
    predictor_cols = getattr(model, "_predictor_cols", [])
    if predictor_cols:
        new_index = ["intercept"]
        new_index += [f"β_{col}" for col in predictor_cols]
        new_index += ["σ"]
        if len(new_index) == len(summary):
            summary.index = new_index

    logger.info("Posterior summary:\n" + summary.to_string())
    return summary


def posterior_predictive_check(model, trace, n_samples: int = 500) -> dict:
    """
    Run posterior predictive checks.

    Returns:
        Dictionary with observed vs predicted statistics.
    """
    import pymc as pm

    with model:
        ppc = pm.sample_posterior_predictive(trace, random_seed=42)

    predicted = ppc.posterior_predictive["temperature"].values
    observed = model.observed_RVs[0].eval()

    pred_mean = predicted.mean(axis=(0, 1))
    pred_std = predicted.std(axis=(0, 1))

    return {
        "observed": observed,
        "predicted_mean": pred_mean,
        "predicted_std": pred_std,
        "r_squared": 1
        - np.sum((observed - pred_mean) ** 2) / np.sum((observed - observed.mean()) ** 2),
    }
