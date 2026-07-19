"""
Graph builder for the Neural Weather Model (ST-GNN).
Maps 2D lat/lon weather grids to a 3D spherical mesh for efficient global message passing.
"""

import math

import numpy as np
import torch
from scipy.spatial import KDTree


def latlon_to_cartesian(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Convert arrays of latitude and longitude (in degrees) to 3D Cartesian coordinates on a unit sphere."""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.stack([x, y, z], axis=1)


def generate_fibonacci_sphere(num_samples: int) -> np.ndarray:
    """Generate roughly uniform points on a sphere using a vectorised Fibonacci lattice."""
    i = np.arange(num_samples, dtype=np.float64)
    phi = math.pi * (3.0 - math.sqrt(5.0))   # golden angle in radians
    y = 1.0 - (i / float(num_samples - 1)) * 2.0  # y in [-1, 1]
    radius = np.sqrt(np.clip(1.0 - y * y, 0.0, None))
    theta = phi * i
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    return np.stack([x, y, z], axis=1)  # [num_samples, 3]


class SphericalGraphBuilder:
    def __init__(self, lat_grid: np.ndarray, lon_grid: np.ndarray, num_mesh_nodes: int = 2000):
        """
        Args:
            lat_grid: 1D array of latitude coordinates.
            lon_grid: 1D array of longitude coordinates.
            num_mesh_nodes: Number of nodes in the latent spherical mesh.
        """
        # Create full grid
        self.num_lat = len(lat_grid)
        self.num_lon = len(lon_grid)
        self.num_grid_nodes = self.num_lat * self.num_lon
        
        lons, lats = np.meshgrid(lon_grid, lat_grid)
        self.grid_lat = lats.flatten()
        self.grid_lon = lons.flatten()
        
        self.grid_coords = latlon_to_cartesian(self.grid_lat, self.grid_lon)
        self.mesh_coords = generate_fibonacci_sphere(num_mesh_nodes)
        
        self.grid_tree = KDTree(self.grid_coords)
        self.mesh_tree = KDTree(self.mesh_coords)

    def build_grid_to_mesh(self, k: int = 4) -> torch.Tensor:
        """Connect each grid node to its k nearest mesh nodes (vectorised)."""
        _, indices = self.mesh_tree.query(self.grid_coords, k=k)  # [num_grid, k]
        src = np.repeat(np.arange(self.num_grid_nodes), k)       # [num_grid * k]
        tgt = indices.flatten()                                    # [num_grid * k]
        return torch.tensor(np.stack([src, tgt]), dtype=torch.long)

    def build_mesh_to_grid(self, k: int = 4) -> torch.Tensor:
        """Connect each mesh node to its k nearest grid nodes (vectorised)."""
        num_mesh = len(self.mesh_coords)
        _, indices = self.grid_tree.query(self.mesh_coords, k=k)  # [num_mesh, k]
        src = np.repeat(np.arange(num_mesh), k)   # [num_mesh * k]
        tgt = indices.flatten()                    # [num_mesh * k]
        return torch.tensor(np.stack([src, tgt]), dtype=torch.long)

    def build_mesh_to_mesh(self, k: int = 8) -> torch.Tensor:
        """Connect each mesh node to its k nearest mesh nodes, excluding self (vectorised)."""
        num_mesh = len(self.mesh_coords)
        _, indices = self.mesh_tree.query(self.mesh_coords, k=k + 1)  # +1: first hit is self
        indices = indices[:, 1:]                           # drop self-loop column → [num_mesh, k]
        src = np.repeat(np.arange(num_mesh), k)           # [num_mesh * k]
        tgt = indices.flatten()                            # [num_mesh * k]
        return torch.tensor(np.stack([src, tgt]), dtype=torch.long)

    def get_grid_features(self) -> torch.Tensor:
        """Returns static grid features: [sin(lat), cos(lat), sin(lon), cos(lon)]."""
        lat_rad = np.radians(self.grid_lat)
        lon_rad = np.radians(self.grid_lon)
        
        features = np.stack([
            np.sin(lat_rad),
            np.cos(lat_rad),
            np.sin(lon_rad),
            np.cos(lon_rad)
        ], axis=1)
        
        return torch.tensor(features, dtype=torch.float32)
