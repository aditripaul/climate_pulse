"""
OLS Baseline regression for climate feedback analysis.

Simple statsmodels OLS as an interpretable comparison baseline
for the Bayesian model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger


def fit_ols(
    df: pd.DataFrame,
    target_col: str = "global_temp_anomaly",
) -> sm.OLS:
    """
    Fit an OLS regression: Temperature ~ Emissions.

    Returns:
        Fitted OLS results object.
    """
    predictor_cols = [c for c in df.columns if c not in [target_col, "year", "time"]]

    X = df[predictor_cols].values
    y = df[target_col].values

    # Add constant (intercept)
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    logger.info(f"OLS fit: R²={model.rsquared:.4f}, Adj R²={model.rsquared_adj:.4f}")
    logger.info(f"F-stat={model.fvalue:.2f}, p={model.f_pvalue:.6f}")

    return model


def ols_summary(model: sm.OLS) -> dict:
    """Extract key OLS metrics."""
    return {
        "r_squared": float(model.rsquared),
        "r_squared_adj": float(model.rsquared_adj),
        "f_statistic": float(model.fvalue),
        "f_pvalue": float(model.f_pvalue),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "coefficients": dict(
            zip(
                model.params.index if hasattr(model.params, "index") else range(len(model.params)),
                model.params.tolist(),
                strict=False,
            )
        ),
        "pvalues": model.pvalues.tolist(),
    }


def ols_diagnostics(model: sm.OLS) -> dict:
    """Compute diagnostic statistics for the OLS model."""
    residuals = model.resid
    fitted = model.fittedvalues

    return {
        "residuals": residuals.tolist(),
        "fitted_values": fitted.tolist(),
        "durbin_watson": float(sm.stats.stattools.durbin_watson(residuals)),
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
    }
