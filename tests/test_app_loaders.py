"""Streamlit app helpers: checkpoint lookup, AQI bands and the weather rollout."""

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from app import loaders as L
from models.nwm.dataset import NWMDataset
from models.nwm.graph_builder import SphericalGraphBuilder
from models.nwm.trainer import NWMTrainer


@pytest.fixture
def tiny_context(tmp_path):
    times = pd.date_range("2020-01-01", "2020-03-31 18:00", freq="6h")
    lat, lon = np.array([-45.0, 0.0, 45.0]), np.array([0.0, 90.0, 180.0, 270.0])
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {
            "2m_temperature": (("time", "latitude", "longitude"), 280 + rng.normal(0, 2, (len(times), 3, 4))),
            "total_precipitation": (("time", "latitude", "longitude"), rng.random((len(times), 3, 4)) * 1e-3),
        },
        coords={"time": times, "latitude": lat, "longitude": lon},
    )
    era5 = tmp_path / "era5.nc"
    ds.to_netcdf(era5)
    nino = tmp_path / "nino.parquet"
    pd.DataFrame({"time": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]), "nino34": [1.0, 2.0, 3.0]}).to_parquet(nino)

    data = NWMDataset(era5, nino, mode="test")
    builder = SphericalGraphBuilder(data.lat, data.lon, num_mesh_nodes=12)
    torch.manual_seed(0)
    trainer = NWMTrainer(
        grid_in_channels=data.window_size * 2 + 4, grid_out_channels=2, hidden_dim=8, num_processor_layers=2,
        learning_rate=1e-3, weight_decay=0.0, edge_grid_to_mesh=builder.build_grid_to_mesh(),
        edge_mesh_to_mesh=builder.build_mesh_to_mesh(), edge_mesh_to_grid=builder.build_mesh_to_grid(), num_mesh_nodes=12,
    )
    trainer.model.eval()
    return L.NWMContext(data, trainer, "observed", tmp_path / "x.pt", "test", 3, 4)


def test_first_rollout_step_is_the_one_step_forecast(tiny_context):
    ctx = tiny_context
    r = L.nwm_rollout(ctx, 0, 3)
    x, cond, _ = ctx.dataset[0]
    with torch.no_grad():
        direct = ctx.trainer._forward(x[None], cond[None])[0, :, 0].numpy()
    expected = direct * ctx.dataset.t2m_std + ctx.dataset.t2m_mean
    np.testing.assert_allclose(r["pred_k"][0].ravel(), expected, rtol=1e-4, atol=1e-3)
    assert r["pred_k"].shape == (3, 3, 4)
    assert len(r["times"]) == len(r["rmse_k"]) == len(r["persistence_rmse_k"]) == 3


def test_rollout_feeds_predictions_back(tiny_context):
    r = L.nwm_rollout(tiny_context, 0, 2)
    # Step 2 truth is the next ERA5 field, but its input window is [truth, prediction]
    assert not np.allclose(r["pred_k"][0], r["pred_k"][1])
    assert np.isfinite(r["pred_k"]).all() and (r["pred_mm"] >= 0).all()


def test_rollout_is_capped_at_the_end_of_the_record(tiny_context):
    last = len(tiny_context.dataset.valid_indices) - 1
    r = L.nwm_rollout(tiny_context, last, 50)
    assert len(r["times"]) <= 50
    assert len(r["times"]) == len(tiny_context.dataset.times) - tiny_context.dataset.valid_indices[last]


def test_nino_value_override_and_none(tiny_context):
    when = pd.Timestamp("2020-02-10")
    assert L.nino_value(tiny_context, when) == 2.0
    assert L.nino_value(tiny_context, when, override=-1.5) == -1.5
    tiny_context.conditioning = "none"
    assert L.nino_value(tiny_context, when) == 0.0


def test_aqi_band_edges():
    assert L.aqi_band(5.0)[0] == "Good"
    assert L.aqi_band(35.4)[0] == "Moderate"
    assert L.aqi_band(35.5)[0] == "Unhealthy for sensitive groups"
    assert L.aqi_band(500.0)[0] == "Hazardous"


def test_missing_checkpoint_gives_a_helpful_error(monkeypatch):
    monkeypatch.setattr(L, "checkpoint_roots", lambda: [])
    with pytest.raises(L.DataMissing, match="No checkpoint found"):
        L.find_checkpoint("nwm/nwm_best.pt")


def test_overview_page_starts_without_error():
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    main = Path(__file__).resolve().parent.parent / "src" / "app" / "main.py"
    at = AppTest.from_file(str(main), default_timeout=60).run()
    assert not at.exception
    assert any("Reported results" in m.value for m in at.markdown)
