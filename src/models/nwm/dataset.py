"""
PyTorch Dataset for Neural Weather Model (ST-GNN).
Yields grid states (T2m, Precip) and El Nino conditioning indices.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import xarray as xr
from pathlib import Path
from loguru import logger
from config import settings


class NWMDataset(Dataset):
    def __init__(
        self,
        nc_path: Path,
        nino_path: Path,
        window_size: int = 2,
        prediction_steps: int = 1,
        mode: str = "train",
    ):
        """
        Args:
            nc_path: Path to ERA5 NetCDF file.
            nino_path: Path to Nino 3.4 parquet file.
            window_size: Number of past time steps to use as input.
            prediction_steps: Number of future time steps to predict.
            mode: 'train', 'val', or 'test'.
        """
        logger.info(f"Loading NWM Dataset from {nc_path}")
        if not nc_path.exists():
            raise FileNotFoundError(f"Missing ERA5 data: {nc_path}. Run download_nwm_data.py first.")
            
        self.ds = xr.open_dataset(nc_path)
        self.window_size = window_size
        self.prediction_steps = prediction_steps
        
        # Load Nino 3.4 index
        nino_df = pd.read_parquet(nino_path)
        nino_df['time'] = pd.to_datetime(nino_df['time'])
        nino_df.set_index('time', inplace=True)
        self.nino_series = nino_df['nino34']
        
        # Extract variables
        self.times = pd.to_datetime(self.ds.time.values)
        self.lat = self.ds.latitude.values
        self.lon = self.ds.longitude.values
        
        # Explicitly transpose to [Time, Lat, Lon] to ensure correct spatial alignment
        self.t2m = self.ds['2m_temperature'].transpose('time', 'latitude', 'longitude').values
        self.tp = self.ds['total_precipitation'].transpose('time', 'latitude', 'longitude').values
        
        # --- B3: Latitude-cosine-weighted normalization ---
        # Grid cells near the equator represent larger physical areas (cos(lat) → 1)
        # than polar cells (cos(lat) → 0). Weighting prevents polar nodes from
        # distorting the global normalization statistics.
        lat_weights = np.cos(np.radians(self.lat))  # [Lat]
        lat_weights = lat_weights / lat_weights.mean()  # unit mean
        # Expand to [Time, Lat, Lon] for broadcasting
        w = lat_weights[np.newaxis, :, np.newaxis]

        # Per-grid-node weights: repeat each lat weight across all longitudes
        node_weights = np.repeat(lat_weights, len(self.lon))  # [Lat*Lon]

        t2m_spatial = self.t2m.mean(axis=0).flatten()  # [Lat*Lon]
        tp_spatial  = self.tp.mean(axis=0).flatten()    # [Lat*Lon]

        self.t2m_mean = float(np.average(t2m_spatial, weights=node_weights))
        self.t2m_std  = float(np.sqrt(((self.t2m - self.t2m_mean) ** 2 * w).mean()) + 1e-6)
        self.tp_mean  = float(np.average(tp_spatial,  weights=node_weights))
        self.tp_std   = float(np.sqrt(((self.tp  - self.tp_mean)  ** 2 * w).mean()) + 1e-6)

        self.t2m = (self.t2m - self.t2m_mean) / self.t2m_std
        self.tp  = (self.tp  - self.tp_mean)  / self.tp_std

        # --- A1: Static positional features ---
        # Precompute [sin(lat), cos(lat), sin(lon), cos(lon)] for each grid node.
        # These are constant across all time steps and are concatenated to every
        # input sample so the model has explicit spatial awareness.
        lons_2d, lats_2d = np.meshgrid(self.lon, self.lat)   # [Lat, Lon]
        lat_rad = np.radians(lats_2d.flatten())
        lon_rad = np.radians(lons_2d.flatten())
        self.static_features = np.stack(
            [np.sin(lat_rad), np.cos(lat_rad), np.sin(lon_rad), np.cos(lon_rad)],
            axis=1,
        ).astype(np.float32)  # [num_grid_nodes, 4]

        # Cosine-latitude weights per grid node, used by the trainer for weighted loss.
        self.lat_weights = np.cos(np.radians(lats_2d.flatten())).astype(np.float32)  # [num_grid_nodes]
        
        # Time indices
        total_steps = len(self.times)
        
        # Split logic (e.g., train on first 70%, val on next 15%, test on last 15%)
        train_end = int(total_steps * 0.7)
        val_end = int(total_steps * 0.85)
        
        self.valid_indices = []
        if mode == "train":
            start_idx, end_idx = 0, train_end
        elif mode == "val":
            start_idx, end_idx = train_end, val_end
        else:
            start_idx, end_idx = val_end, total_steps
            
        # Ensure we have enough room for window and prediction
        start_idx = max(start_idx, self.window_size)
        end_idx = min(end_idx, total_steps - self.prediction_steps)
        
        self.valid_indices = list(range(start_idx, end_idx))
        logger.info(f"Initialized {mode} dataset with {len(self.valid_indices)} samples.")

    def __del__(self) -> None:
        """Close the xarray Dataset to release the file handle."""
        if hasattr(self, "ds"):
            try:
                self.ds.close()
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        time_idx = self.valid_indices[idx]
        
        # 1. Get input window [Time, Lat, Lon, Channels]
        x_t2m = self.t2m[time_idx - self.window_size : time_idx]
        x_tp = self.tp[time_idx - self.window_size : time_idx]
        
        # Flatten temporal window into channels [Lat, Lon, Channels * Window]
        x_t2m_flat = np.transpose(x_t2m, (1, 2, 0)) # [Lat, Lon, Window]
        x_tp_flat = np.transpose(x_tp, (1, 2, 0))   # [Lat, Lon, Window]
        
        x = np.concatenate([x_t2m_flat, x_tp_flat], axis=-1)
        x_flat = x.reshape(-1, x.shape[-1])  # [num_grid_nodes, Channels*Window]

        # A1: Append static positional features so the model knows where each node is.
        x_flat = np.concatenate([x_flat, self.static_features], axis=-1)  # [num_grid_nodes, C*W + 4]
        
        # 2. Get target prediction [Lat, Lon, Channels * PredictionSteps]
        y_t2m = self.t2m[time_idx : time_idx + self.prediction_steps]
        y_tp = self.tp[time_idx : time_idx + self.prediction_steps]
        
        y_t2m_flat = np.transpose(y_t2m, (1, 2, 0))
        y_tp_flat = np.transpose(y_tp, (1, 2, 0))
        
        y = np.concatenate([y_t2m_flat, y_tp_flat], axis=-1)
        y_flat = y.reshape(-1, y.shape[-1]) # [Lat*Lon, Channels*PredictionSteps]
        
        # 3. Get Nino 3.4 conditioning scalar for the current time
        current_time = self.times[time_idx]
        
        # Nearest valid Nino month (nino is usually monthly)
        try:
            nino_idx = self.nino_series.index.get_indexer([current_time], method='nearest')[0]
            nino_val = self.nino_series.iloc[nino_idx]
        except Exception:
            nino_val = 0.0 # Fallback if missing
            
        condition = np.array([nino_val], dtype=np.float32)
        
        return (
            torch.tensor(x_flat, dtype=torch.float32), 
            torch.tensor(condition, dtype=torch.float32), 
            torch.tensor(y_flat, dtype=torch.float32)
        )
