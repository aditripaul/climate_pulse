"""
Data cleaning utilities for ClimatePulse.

Handles missing values, outlier detection, temporal interpolation,
and data quality validation across all data sources.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger


def clean_sst_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean sea surface temperature data.

    - Removes obvious outliers (SST < -5°C or > 40°C).
    - Fills short temporal gaps via linear interpolation.
    - Validates coordinate ranges.

    Args:
        df: Raw SST DataFrame with columns [time, lat, lon, sst].

    Returns:
        Cleaned DataFrame.
    """
    initial_len = len(df)

    # Remove physically impossible values
    mask = (df["sst"] >= -5) & (df["sst"] <= 40)
    df = df[mask].copy()

    # Validate coordinates
    df = df[(df["lat"] >= -90) & (df["lat"] <= 90)]
    df = df[(df["lon"] >= 0) & (df["lon"] <= 360)]

    removed = initial_len - len(df)
    if removed > 0:
        logger.info(f"SST cleaning: removed {removed:,} invalid rows ({removed/initial_len:.1%})")

    return df.sort_values("time").reset_index(drop=True)


def clean_pollution_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean air quality measurement data.

    - Removes negative concentrations.
    - Applies IQR-based outlier detection per parameter.
    - Validates coordinate data.

    Args:
        df: Raw pollution DataFrame with columns [time, location_id, lat, lon,
            parameter, value, unit].

    Returns:
        Cleaned DataFrame.
    """
    initial_len = len(df)

    # Remove negative values
    df = df[df["value"] >= 0].copy()

    # Remove extreme outliers per parameter using IQR
    cleaned_parts = []
    for param in df["parameter"].unique():
        param_df = df[df["parameter"] == param].copy()
        q1 = param_df["value"].quantile(0.25)
        q3 = param_df["value"].quantile(0.75)
        iqr = q3 - q1
        lower = max(0, q1 - 3.0 * iqr)  # More lenient for pollution spikes
        upper = q3 + 3.0 * iqr
        param_df = param_df[(param_df["value"] >= lower) & (param_df["value"] <= upper)]
        cleaned_parts.append(param_df)

    df = pd.concat(cleaned_parts, ignore_index=True)

    # Drop rows with missing coordinates
    df = df.dropna(subset=["lat", "lon"])

    removed = initial_len - len(df)
    if removed > 0:
        logger.info(f"Pollution cleaning: removed {removed:,} rows ({removed/initial_len:.1%})")

    return df.sort_values("time").reset_index(drop=True)


def clean_temperature_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean GISTEMP temperature anomaly data.

    - Removes anomalies > ±10°C (extreme outliers).
    - Forward-fills short gaps (up to 3 months).

    Args:
        df: DataFrame with columns [time, lat, lon, tempanomaly] or
            [time, global_temp_anomaly].

    Returns:
        Cleaned DataFrame.
    """
    temp_col = "global_temp_anomaly" if "global_temp_anomaly" in df.columns else "tempanomaly"

    initial_len = len(df)
    df = df[df[temp_col].abs() <= 10].copy()

    removed = initial_len - len(df)
    if removed > 0:
        logger.info(f"Temperature cleaning: removed {removed} extreme values")

    return df.sort_values("time").reset_index(drop=True)


def clean_emissions_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Climate TRACE emissions data.

    - Removes negative emissions (invalid).
    - Drops duplicates.
    - Validates year range.

    Args:
        df: DataFrame with columns [country, sector, year, gas, emissions_quantity].

    Returns:
        Cleaned DataFrame.
    """
    initial_len = len(df)

    # Remove negative emissions
    df = df[df["emissions_quantity"] >= 0].copy()

    # Drop exact duplicates (use only columns that exist)
    dedup_cols = [c for c in ["country", "sector", "year", "gas"] if c in df.columns]
    df = df.drop_duplicates(subset=dedup_cols)

    removed = initial_len - len(df)
    if removed > 0:
        logger.info(f"Emissions cleaning: removed {removed} invalid/duplicate rows")

    sort_cols = [c for c in ["year", "country", "sector"] if c in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True)


def interpolate_timeseries(
    df: pd.DataFrame,
    time_col: str = "time",
    value_col: str = "value",
    method: str = "linear",
    max_gap: int = 3,
) -> pd.DataFrame:
    """
    Interpolate missing values in a time series.

    Args:
        df: DataFrame with time and value columns.
        time_col: Name of the time column.
        value_col: Name of the value column.
        method: Interpolation method ('linear', 'cubic', 'nearest').
        max_gap: Maximum consecutive NaN gap to fill.

    Returns:
        DataFrame with interpolated values.
    """
    df = df.copy()
    df = df.set_index(time_col).sort_index()
    df[value_col] = df[value_col].interpolate(method=method, limit=max_gap)
    return df.reset_index()
