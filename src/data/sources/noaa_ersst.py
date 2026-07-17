"""
NOAA ERSST v5 — Sea Surface Temperature data connector.

Streams Extended Reconstructed Sea Surface Temperature (ERSST) v5 data
from NOAA's Physical Sciences Laboratory via OPeNDAP using xarray.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from config import settings


class NOAAERSSTSource:
    """
    Connector for NOAA ERSST v5 monthly mean SST data.

    Data is accessed via OPeNDAP (no download required) or cached locally
    as Parquet for fast repeated access.
    """

    def __init__(
        self,
        url: str | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
    ):
        self.url = url or settings.data.noaa_ersst_url
        self.start_year = start_year if start_year is not None else settings.data.start_year
        self.end_year = end_year if end_year is not None else settings.data.end_year
        self._cache_path = settings.data.processed_dir / "ersst_v5.parquet"

        if self.start_year > 1991:
            logger.warning(
                f"NOAAERSSTSource start_year ({self.start_year}) is after 1991. "
                "The 1991-2020 climatology reference window for Niño 3.4 will be loaded automatically to prevent biased anomalies."
            )

    def fetch(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch ERSST v5 SST data.

        Args:
            use_cache: If True, return cached Parquet if available.

        Returns:
            DataFrame with columns: time, lat, lon, sst
        """
        if use_cache and self._cache_path.exists():
            logger.info(f"Loading cached ERSST data from {self._cache_path}")
            return pd.read_parquet(self._cache_path)

        logger.info(f"Fetching ERSST v5 data from OPeNDAP ({self.start_year}–{self.end_year})")
        ds = self._open_dataset()
        df = self._to_dataframe(ds)

        # Cache to Parquet
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self._cache_path, index=False)
        logger.info(f"Cached {len(df):,} rows to {self._cache_path}")

        return df

    def fetch_nino34(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Compute the Niño 3.4 index (area-averaged SST anomaly in 5°N-5°S, 170°W-120°W).

        Returns:
            DataFrame with columns: time, nino34_sst, nino34_anomaly
        """
        cache_path = settings.data.processed_dir / "nino34_index.parquet"
        if use_cache and cache_path.exists():
            logger.info("Loading cached Niño 3.4 index")
            return pd.read_parquet(cache_path)

        logger.info("Computing Niño 3.4 index from ERSST v5")
        ds = self._open_dataset(include_climatology=True)

        # Extract Niño 3.4 region
        cfg = settings.data
        nino34 = ds["sst"].sel(
            lat=slice(cfg.nino34_lat_max, cfg.nino34_lat_min),
            lon=slice(cfg.nino34_lon_min, cfg.nino34_lon_max),
        )

        # Area-weighted mean (weight by cosine of latitude)
        weights = np.cos(np.deg2rad(nino34.lat))
        nino34_mean = nino34.weighted(weights).mean(dim=["lat", "lon"])

        # Compute anomaly relative to 1991-2020 climatology
        climatology = nino34_mean.sel(time=slice("1991-01", "2020-12")).groupby("time.month").mean()
        anomaly = nino34_mean.groupby("time.month") - climatology

        df = pd.DataFrame(
            {
                "time": nino34_mean.time.values,
                "nino34_sst": nino34_mean.values,
                "nino34_anomaly": anomaly.values,
            }
        )

        # Filter back to self.start_year for the returned dataset
        df = df[df["time"] >= pd.to_datetime(f"{self.start_year}-01-01")].reset_index(drop=True)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        logger.info(f"Cached Niño 3.4 index ({len(df)} months starting at {self.start_year})")

        return df

    def _open_dataset(self, include_climatology: bool = False) -> xr.Dataset:
        """Open ERSST dataset via OPeNDAP with time subsetting."""
        ds = xr.open_dataset(
            self.url,
            drop_variables=["time_bnds"],
            engine="netcdf4",
        )
        # Subset to configured time range
        start = min(self.start_year, 1991) if include_climatology else self.start_year
        ds = ds.sel(time=slice(f"{start}-01", f"{self.end_year}-12"))
        return ds

    def _to_dataframe(self, ds: xr.Dataset) -> pd.DataFrame:
        """Convert xarray Dataset to a tidy pandas DataFrame."""
        df = ds["sst"].to_dataframe().reset_index()
        # Drop rows where SST is NaN (land cells)
        df = df.dropna(subset=["sst"])
        df["time"] = pd.to_datetime(df["time"])
        return df[["time", "lat", "lon", "sst"]]
