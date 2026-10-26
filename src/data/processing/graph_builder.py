"""
Graph construction for the Pollution Dispersion GNN.

Builds spatial graphs from air quality monitoring station locations,
with edges weighted by inverse distance and optionally by wind direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.neighbors import BallTree


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in km between two points."""
    r = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(r * 2 * np.arcsin(np.sqrt(a)))


def build_station_graph(
    locations_df: pd.DataFrame,
    max_distance_km: float = 500.0,
    k_nearest: int | None = 10,
) -> dict:
    """
    Build a spatial graph from monitoring station locations.

    Args:
        locations_df: DataFrame with columns [location_id, lat, lon].
        max_distance_km: Maximum edge distance in km.
        k_nearest: If set, connect each node to at most k nearest neighbors.

    Returns:
        Dictionary with:
        - 'edge_index': np.ndarray of shape (2, num_edges) — source and target node indices.
        - 'edge_weight': np.ndarray of shape (num_edges,) — inverse distance weights.
        - 'node_ids': list of location IDs corresponding to node indices.
        - 'node_coords': np.ndarray of shape (num_nodes, 2) — [lat, lon] per node.
    """
    locations = locations_df.drop_duplicates(subset=["location_id"]).reset_index(drop=True)
    n = len(locations)
    node_ids = locations["location_id"].tolist()
    node_coords = locations[["lat", "lon"]].values

    logger.info(f"Building station graph: {n} nodes, max_dist={max_distance_km}km")

    # Compute pairwise distances using BallTree
    sources = []
    targets = []
    weights = []

    if n > 0:
        # BallTree haversine metric requires radians
        coords_rad = np.radians(node_coords)
        tree = BallTree(coords_rad, metric="haversine")
        r_earth_km = 6371.0
        max_dist_rad = max_distance_km / r_earth_km

        if k_nearest is not None:
            # Query up to k_nearest + 1 (since a node is its own nearest neighbor)
            k = min(k_nearest + 1, n)
            dist_rad, ind = tree.query(coords_rad, k=k)
            for i in range(n):
                for j_idx in range(1, k):  # Skip self (index 0 is the node itself)
                    d_km = dist_rad[i, j_idx] * r_earth_km
                    j = ind[i, j_idx]
                    if d_km <= max_distance_km:
                        sources.append(i)
                        targets.append(j)
                        weights.append(1.0 / (d_km + 1.0))
        else:
            # Find all nodes within max_distance_km
            ind, dist_rad = tree.query_radius(coords_rad, r=max_dist_rad, return_distance=True)
            for i in range(n):
                # Sort neighbors by distance to match old behavior
                sort_idx = np.argsort(dist_rad[i])
                neighbors = ind[i][sort_idx]
                dists = dist_rad[i][sort_idx] * r_earth_km
                for j, d_km in zip(neighbors, dists, strict=False):
                    if i != j:
                        sources.append(i)
                        targets.append(j)
                        weights.append(1.0 / (d_km + 1.0))

    edge_index = np.array([sources, targets], dtype=np.int64)
    edge_weight = np.array(weights, dtype=np.float32)

    # Normalize weights to [0, 1]
    if len(edge_weight) > 0:
        edge_weight = edge_weight / edge_weight.max()

    logger.info(f"Built graph: {n} nodes, {len(sources)} edges")

    return {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "node_ids": node_ids,
        "node_coords": node_coords,
    }


def build_temporal_snapshots(
    measurements_df: pd.DataFrame,
    graph: dict,
    feature_cols: list[str] | None = None,
    time_freq: str = "D",
) -> list[dict]:
    """
    Create temporal snapshots for the Spatio-Temporal GNN.

    Each snapshot contains node feature matrices for one time step,
    along with the shared graph topology.

    Args:
        measurements_df: DataFrame with [time, location_id, parameter, value].
        graph: Graph dict from build_station_graph().
        feature_cols: Parameters to use as node features (e.g., ['pm25', 'no2']).
        time_freq: Temporal aggregation frequency ('D' for daily, 'W' for weekly).

    Returns:
        List of dicts, each with:
        - 'time': timestamp
        - 'x': np.ndarray of shape (num_nodes, num_features); NaN where a station
          has no reading for that period
        - 'edge_index': shared graph topology
        - 'edge_weight': shared edge weights
    """
    if feature_cols is None:
        feature_cols = ["pm25"]

    node_ids = graph["node_ids"]
    node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    n_nodes = len(node_ids)
    n_features = len(feature_cols)

    # Aggregate measurements by time period and location
    df = measurements_df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.dropna(subset=["value"])
    df = (
        df.set_index("time")
        .groupby([pd.Grouper(freq=time_freq), "location_id", "parameter"])["value"]
        .mean()
        .reset_index()
    )

    snapshots = []
    for time_step, group in df.groupby("time"):
        x = np.full((n_nodes, n_features), np.nan, dtype=np.float32)

        for _, row in group.iterrows():
            loc_id = row["location_id"]
            param = row["parameter"]
            if loc_id in node_id_to_idx and param in feature_cols:
                node_idx = node_id_to_idx[loc_id]
                feat_idx = feature_cols.index(param)
                x[node_idx, feat_idx] = row["value"]

        snapshots.append(
            {
                "time": time_step,
                "x": x,
                "edge_index": graph["edge_index"],
                "edge_weight": graph["edge_weight"],
            }
        )

    logger.info(f"Created {len(snapshots)} temporal snapshots ({time_freq} resolution)")
    return snapshots


def filter_locations_to_measurements(
    locations_df: pd.DataFrame,
    measurements_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Filter locations to only those with actual measurements.

    Reduces graph sparsity by excluding nodes that have
    no sensor data in the measurement dataset.
    """
    measured_ids = set(measurements_df["location_id"].unique())
    filtered = locations_df[locations_df["location_id"].isin(measured_ids)].reset_index(drop=True)
    logger.info(
        f"Filtered locations: {len(locations_df)} \u2192 {len(filtered)} "
        f"(keeping only stations with measurements)"
    )
    return filtered


def normalize_snapshot_features(
    snapshots: list[dict],
    stats: dict | None = None,
) -> tuple[list[dict], dict]:
    """
    Z-score normalize node features across snapshots.

    Missing readings are NaN and are excluded from the statistics and left as NaN.
    To avoid leaking test data, compute `stats` on the training snapshots only and
    pass them in when normalizing validation and test snapshots.

    Args:
        snapshots: List of temporal snapshot dicts with 'x' arrays.
        stats: Pre-computed {"mean": float, "std": float}. If None, computed from data.

    Returns:
        (snapshots, stats) — snapshots are modified in-place.
    """
    if stats is None:
        values = np.concatenate([s["x"].ravel() for s in snapshots]) if snapshots else np.array([])
        values = values[np.isfinite(values)]
        mean = float(values.mean()) if len(values) > 0 else 0.0
        std = float(values.std()) if len(values) > 0 else 1.0
        stats = {"mean": mean, "std": max(std, 1e-8)}

    mean_val, std_val = stats["mean"], stats["std"]
    for s in snapshots:
        s["x"] = (s["x"] - mean_val) / std_val

    logger.info(f"Normalized snapshot features: mean={mean_val:.2f}, std={std_val:.2f}")
    return snapshots, stats
