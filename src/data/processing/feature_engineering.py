"""
Feature engineering for ClimatePulse models.

Generates derived features from raw data for the TFT, ST-GNN,
and regression models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# El Niño / ENSO features
# ---------------------------------------------------------------------------


def compute_nino_indices(sst_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all four Niño indices from gridded SST data.

    Regions:
    - Niño 1+2: 0-10°S, 90°W-80°W  (270-280°E)
    - Niño 3:   5°N-5°S, 150°W-90°W  (210-270°E)
    - Niño 3.4: 5°N-5°S, 170°W-120°W (190-240°E)
    - Niño 4:   5°N-5°S, 160°E-150°W  (160-210°E)

    Args:
        sst_df: DataFrame with columns [time, lat, lon, sst].

    Returns:
        DataFrame with columns [time, nino12, nino3, nino34, nino4].
    """
    regions = {
        "nino12": {"lat": (-10, 0), "lon": (270, 280)},
        "nino3": {"lat": (-5, 5), "lon": (210, 270)},
        "nino34": {"lat": (-5, 5), "lon": (190, 240)},
        "nino4": {"lat": (-5, 5), "lon": (160, 210)},
    }

    result = None

    for name, bounds in regions.items():
        lat_min, lat_max = bounds["lat"]
        lon_min, lon_max = bounds["lon"]

        mask = (
            (sst_df["lat"] >= lat_min)
            & (sst_df["lat"] <= lat_max)
            & (sst_df["lon"] >= lon_min)
            & (sst_df["lon"] <= lon_max)
        )
        region_df = sst_df[mask]

        # Area-weighted mean
        weights = np.cos(np.deg2rad(region_df["lat"]))
        monthly = region_df.copy()
        monthly["weighted_sst"] = monthly["sst"] * weights.values
        monthly["weight"] = weights.values

        grouped = monthly.groupby("time").agg(
            weighted_sum=("weighted_sst", "sum"),
            weight_sum=("weight", "sum"),
        )
        grouped[name] = grouped["weighted_sum"] / grouped["weight_sum"]

        result = grouped[[name]] if result is None else result.join(grouped[[name]], how="outer")  # type: ignore

    assert result is not None, "No Niño indices computed"
    result = result.reset_index()
    result["time"] = pd.to_datetime(result["time"])

    logger.info(f"Computed Niño indices for {len(result)} months")
    return result


def add_rolling_features(
    df: pd.DataFrame,
    value_col: str,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add rolling window statistics (mean, std) to a time series.

    Args:
        df: DataFrame sorted by time with a value column.
        value_col: Column to compute rolling stats for.
        windows: Window sizes in rows (e.g., [3, 6, 12] for months).

    Returns:
        DataFrame with additional rolling feature columns.
    """
    if windows is None:
        windows = [3, 6, 12]

    df = df.copy().sort_values("time")

    for w in windows:
        df[f"{value_col}_roll_mean_{w}"] = df[value_col].rolling(w, min_periods=1).mean()
        df[f"{value_col}_roll_std_{w}"] = df[value_col].rolling(w, min_periods=1).std()

    return df


def add_lag_features(
    df: pd.DataFrame,
    value_col: str,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add lagged values as features.

    Args:
        df: DataFrame sorted by time.
        value_col: Column to lag.
        lags: Lag periods (e.g., [1, 3, 6, 12]).

    Returns:
        DataFrame with lag columns.
    """
    if lags is None:
        lags = [1, 3, 6, 12]

    df = df.copy().sort_values("time")

    for lag in lags:
        df[f"{value_col}_lag_{lag}"] = df[value_col].shift(lag)

    return df


def add_seasonal_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """
    Add cyclical seasonal features (sin/cos encoded month).

    Args:
        df: DataFrame with a time column.
        time_col: Name of the datetime column.

    Returns:
        DataFrame with month_sin, month_cos columns.
    """
    df = df.copy()
    month = pd.to_datetime(df[time_col]).dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["month"] = month
    df["year"] = pd.to_datetime(df[time_col]).dt.year
    return df


# ---------------------------------------------------------------------------
# Pollution features
# ---------------------------------------------------------------------------


def compute_aqi(pm25: float) -> int:
    """
    Compute US EPA Air Quality Index (AQI) from PM2.5 concentration (μg/m³).

    Uses linear interpolation within EPA breakpoint ranges.
    """
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]

    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= pm25 <= bp_hi:
            return round((aqi_hi - aqi_lo) / (bp_hi - bp_lo) * (pm25 - bp_lo) + aqi_lo)

    return 500 if pm25 > 500.4 else 0


def add_aqi_column(df: pd.DataFrame, pm25_col: str = "value") -> pd.DataFrame:
    """Add AQI column computed from PM2.5 values."""
    df = df.copy()
    df["aqi"] = df[pm25_col].apply(compute_aqi)
    return df


# ---------------------------------------------------------------------------
# Climate feedback features
# ---------------------------------------------------------------------------


def align_emissions_temperature(
    emissions_df: pd.DataFrame,
    temperature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align emissions and temperature data by year for regression analysis.

    Args:
        emissions_df: DataFrame with columns [year, gas, emissions_quantity].
        temperature_df: DataFrame with columns [time, global_temp_anomaly].

    Returns:
        Merged DataFrame with annual emissions and temperature anomaly.
    """
    # Aggregate emissions by year and gas
    emissions_annual = (
        emissions_df.groupby(["year", "gas"])["emissions_quantity"].sum().reset_index()
    )
    emissions_pivot = emissions_annual.pivot_table(
        index="year", columns="gas", values="emissions_quantity", aggfunc="sum"
    ).reset_index()

    # Compute annual mean temperature anomaly
    temp_annual = temperature_df.copy()
    temp_annual["year"] = pd.to_datetime(temp_annual["time"]).dt.year
    temp_annual = temp_annual.groupby("year")["global_temp_anomaly"].mean().reset_index()

    # Merge
    merged = pd.merge(temp_annual, emissions_pivot, on="year", how="inner")

    logger.info(f"Aligned {len(merged)} years of emissions × temperature data")
    return merged
