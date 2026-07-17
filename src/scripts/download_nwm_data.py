#!/usr/bin/env python3
"""
Downloads a lightweight subset of global ERA5 data for the Neural Weather Model (NWM).
Accesses the public WeatherBench2 Zarr archive (5.625 degree resolution, 6-hourly).
Slices to a recent subset (2020-2022) and specific variables (2m temperature, total precipitation)
to keep the download under 2GB, perfect for Colab training.
"""

import os
import sys
from pathlib import Path
from loguru import logger
import xarray as xr

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings

def download_nwm_era5():
    import gcsfs
    import pandas as pd
    from tqdm import tqdm
    
    logger.info("Connecting to WeatherBench2 Google Cloud Storage Zarr store...")
    zarr_url = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"
    
    try:
        # Open Zarr store directly from GCP without authentication using gcsfs anon mapper
        fs = gcsfs.GCSFileSystem(token='anon')
        mapper = fs.get_mapper(zarr_url)
        ds = xr.open_zarr(mapper, consolidated=True)
    except Exception as e:
        logger.error(f"Failed to connect to GCP Zarr store: {e}")
        sys.exit(1)
        
    logger.info(f"Successfully opened remote Zarr store. Selecting variables and time slice...")
    
    # Subset to manageable variables for NWM prototype
    # '2m_temperature' -> 't2m'
    # 'total_precipitation_6hr' -> 'total_precipitation'
    target_vars = ['2m_temperature', 'total_precipitation_6hr']
    
    # Check if variables exist in this dataset (WeatherBench uses these names)
    avail_vars = list(ds.data_vars)
    for v in target_vars:
        if v not in avail_vars:
            logger.warning(f"Variable '{v}' not found in dataset. Available: {avail_vars[:10]}...")
            
    # Subset the dataset and rename variables for NWM dataset compatibility
    subset = ds[target_vars].rename({'total_precipitation_6hr': 'total_precipitation'})
    
    # Slice time to 2020 - 2022 to keep it small (~3 years of 6-hourly data)
    subset = subset.sel(time=slice('2020-01-01', '2022-12-31'))
    
    output_dir = settings.data.raw_dir / "nwm_era5"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "era5_5deg_6hr_2020_2022.nc"
    
    if output_file.exists():
        logger.info(f"Dataset already exists at {output_file}. Skipping download.")
        return
        
    logger.info(f"Downloading data to {output_file}...")
    logger.info("This will stream the data from GCP in monthly chunks to show progress.")
    
    # Generate list of months
    months = pd.date_range(start="2020-01-01", end="2022-12-31", freq="MS")
    datasets = []
    
    for start_date in tqdm(months, desc="Downloading ERA5 data (monthly chunks)"):
        end_date = start_date + pd.offsets.MonthEnd(0)
        # slice time and eagerly load
        month_ds = subset.sel(time=slice(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))).load()
        datasets.append(month_ds)
        
    logger.info("Merging monthly datasets...")
    final_ds = xr.concat(datasets, dim='time')
    logger.info(f"Saving merged dataset to {output_file}...")
    final_ds.to_netcdf(output_file, engine='netcdf4')
    
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"Successfully saved {file_size_mb:.2f} MB to {output_file}")
    
if __name__ == "__main__":
    download_nwm_era5()
