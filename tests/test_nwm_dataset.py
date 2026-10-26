"""NWM dataset: conditioning lookup and leakage-free normalization."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from models.nwm.dataset import NWMDataset


@pytest.fixture
def era5_file(tmp_path):
    times = pd.date_range("2020-01-01", "2020-03-31 18:00", freq="6h")
    lat, lon = np.array([-45.0, 0.0, 45.0]), np.array([0.0, 90.0, 180.0, 270.0])
    rng = np.random.default_rng(0)
    # Warming trend so that train-only statistics differ from full-period ones
    trend = np.linspace(0, 10, len(times))[:, None, None]
    ds = xr.Dataset(
        {
            "2m_temperature": (("time", "latitude", "longitude"),
                               280 + trend + rng.normal(0, 1, (len(times), 3, 4))),
            "total_precipitation": (("time", "latitude", "longitude"),
                                    rng.random((len(times), 3, 4))),
        },
        coords={"time": times, "latitude": lat, "longitude": lon},
    )
    path = tmp_path / "era5.nc"
    ds.to_netcdf(path)
    return path, times


@pytest.fixture
def nino_file(tmp_path):
    path = tmp_path / "nino.parquet"
    pd.DataFrame(
        {"time": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]), "nino34": [1.0, 2.0, 3.0]}
    ).to_parquet(path)
    return path


def _condition_at(ds: NWMDataset, when: str) -> float:
    idx = ds.valid_indices.index(int(np.searchsorted(ds.times, pd.Timestamp(when))))
    return float(ds[idx][1][0])


def test_condition_uses_month_containing_sample(era5_file, nino_file):
    path, _ = era5_file
    ds = NWMDataset(path, nino_file, mode="train")
    # Late January must use January's value, not the nearer February stamp
    assert _condition_at(ds, "2020-01-28") == 1.0
    assert _condition_at(ds, "2020-02-02") == 2.0


def test_no_conditioning_is_zero(era5_file, nino_file):
    path, _ = era5_file
    ds = NWMDataset(path, nino_file, mode="train", conditioning="none")
    assert _condition_at(ds, "2020-01-28") == 0.0


def test_normalization_uses_training_period_only(era5_file, nino_file):
    path, times = era5_file
    train_end = int(len(times) * 0.7)
    raw = xr.open_dataset(path)["2m_temperature"].values
    ds = NWMDataset(path, nino_file, mode="test")
    w = np.cos(np.radians([-45.0, 0.0, 45.0]))
    expected = np.average(raw[:train_end].mean(axis=(0, 2)), weights=w)
    assert ds.t2m_mean == pytest.approx(expected, rel=1e-6)
