"""
NASA GISTEMP v4 — Global Surface Temperature Anomaly data connector.

Downloads and parses GISTEMP v4 gridded temperature anomaly data
(GISS Surface Temperature Analysis based on GHCNv4 and ERSSTv5).
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
from loguru import logger

from config import settings


class NASAGISTEMPSource:
    """
    Connector for NASA GISTEMP v4 global temperature anomaly data.

    Data is provided as netCDF on a 2°×2° grid with monthly resolution.
    Anomalies are relative to the 1951-1980 base period.
    """

    def __init__(
        self,
        url: str | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
    ):
        self.url = url or settings.data.nasa_gistemp_url
        self.start_year = start_year or settings.data.start_year
        self.end_year = end_year or settings.data.end_year
        self._raw_path = settings.data.raw_dir / "gistemp1200_GHCNv4_ERSSTv5.nc"
        self._cache_path = settings.data.processed_dir / "gistemp_v4.parquet"

    def fetch(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch GISTEMP v4 gridded temperature anomalies.

        Returns:
            DataFrame with columns: time, lat, lon, tempanomaly
        """
        if use_cache and self._cache_path.exists():
            logger.info("Loading cached GISTEMP data")
            return pd.read_parquet(self._cache_path)

        # Download if needed
        self._download_if_needed()

        logger.info(f"Parsing GISTEMP v4 data ({self.start_year}–{self.end_year})")
        ds = xr.open_dataset(self._raw_path, engine="netcdf4")
        ds = ds.sel(time=slice(f"{self.start_year}-01", f"{self.end_year}-12"))

        df = ds["tempanomaly"].to_dataframe().reset_index()
        df = df.dropna(subset=["tempanomaly"])
        df["time"] = pd.to_datetime(df["time"])
        df = df[["time", "lat", "lon", "tempanomaly"]]

        # Cache
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self._cache_path, index=False)
        logger.info(f"Cached {len(df):,} rows of GISTEMP data")

        return df

    def fetch_global_mean(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Compute global mean temperature anomaly time series.

        Returns:
            DataFrame with columns: time, global_temp_anomaly
        """
        cache_path = settings.data.processed_dir / "gistemp_global_mean.parquet"
        if use_cache and cache_path.exists():
            logger.info("Loading cached GISTEMP global mean")
            return pd.read_parquet(cache_path)

        self._download_if_needed()

        ds = xr.open_dataset(self._raw_path, engine="netcdf4")
        ds = ds.sel(time=slice(f"{self.start_year}-01", f"{self.end_year}-12"))

        temp = ds["tempanomaly"]

        # Area-weighted global mean (cosine of latitude)
        weights = np.cos(np.deg2rad(temp.lat))
        global_mean = temp.weighted(weights).mean(dim=["lat", "lon"])

        df = pd.DataFrame(
            {
                "time": global_mean.time.values,
                "global_temp_anomaly": global_mean.values,
            }
        )
        df["time"] = pd.to_datetime(df["time"])

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        logger.info(f"Computed global mean anomaly ({len(df)} months)")

        return df

    def _download_if_needed(self) -> None:
        """Download the GISTEMP netCDF file if not already cached."""
        if self._raw_path.exists():
            return

        self._raw_path.parent.mkdir(parents=True, exist_ok=True)
        gz_path = Path(str(self._raw_path) + ".gz")

        logger.info("Downloading GISTEMP v4 from NASA GISS...")
        response = requests.get(self.url, stream=True, timeout=120)
        response.raise_for_status()

        with open(gz_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Decompress
        logger.info("Decompressing GISTEMP netCDF...")
        with gzip.open(gz_path, "rb") as f_in, open(self._raw_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        gz_path.unlink()
        logger.info(f"GISTEMP data saved to {self._raw_path}")
