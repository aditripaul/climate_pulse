"""
Hybrid storage layer for ClimatePulse.

JSON for metadata and small reference records.
Parquet for large time-series data via PyArrow.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from config import settings


class Database:
    """
    Unified storage interface combining JSON and Parquet backends.

    JSON handles metadata.
    Parquet handles bulk time-series storage for performance.
    """

    def __init__(self, metadata_path: Path | None = None):
        self.metadata_path = metadata_path or settings.metadata.metadata_path
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        """Create JSON file if it doesn't exist."""
        if not self.metadata_path.exists():
            with open(self.metadata_path, "w") as f:
                json.dump({
                    "data_sources": {},
                    "model_runs": {},
                    "next_run_id": 1
                }, f, indent=4)

    def _read_metadata(self) -> dict[str, Any]:
        with open(self.metadata_path, "r") as f:
            return json.load(f)
            
    def _write_metadata(self, data: dict[str, Any]) -> None:
        with open(self.metadata_path, "w") as f:
            json.dump(data, f, indent=4)

    # ----- Data source metadata -----

    def register_source(
        self,
        name: str,
        description: str,
        record_count: int,
        cache_path: str,
    ) -> None:
        """Register or update a data source entry."""
        data = self._read_metadata()
        data["data_sources"][name] = {
            "name": name,
            "description": description,
            "last_updated": datetime.now(tz=timezone.utc).isoformat(),
            "record_count": record_count,
            "cache_path": cache_path
        }
        self._write_metadata(data)

    def get_sources(self) -> list[dict[str, Any]]:
        """List all registered data sources."""
        data = self._read_metadata()
        return list(data.get("data_sources", {}).values())

    # ----- Model runs -----

    def log_model_run(
        self,
        model_type: str,
        config: str = "",
    ) -> int:
        """Start logging a model training run. Returns the run ID."""
        data = self._read_metadata()
        run_id = data.get("next_run_id", 1)
        data["next_run_id"] = run_id + 1
        
        data["model_runs"][str(run_id)] = {
            "id": run_id,
            "model_type": model_type,
            "config": config,
            "started_at": datetime.now(tz=timezone.utc).isoformat(),
            "completed_at": None,
            "status": "running",
            "metrics": "",
            "checkpoint_path": ""
        }
        self._write_metadata(data)
        return run_id

    def complete_model_run(
        self,
        run_id: int,
        status: str = "completed",
        metrics: str = "",
        checkpoint_path: str = "",
    ) -> None:
        """Mark a model run as completed."""
        data = self._read_metadata()
        run_key = str(run_id)
        if run_key in data["model_runs"]:
            data["model_runs"][run_key].update({
                "completed_at": datetime.now(tz=timezone.utc).isoformat(),
                "status": status,
                "metrics": metrics,
                "checkpoint_path": checkpoint_path
            })
            self._write_metadata(data)

    def get_model_runs(self, model_type: str | None = None) -> list[dict[str, Any]]:
        """Get model run history."""
        data = self._read_metadata()
        runs = list(data.get("model_runs", {}).values())
        if model_type:
            runs = [r for r in runs if r.get("model_type") == model_type]
        # Sort by started_at descending
        runs.sort(key=lambda x: x.get("started_at", ""), reverse=True)
        return runs

    # ----- Parquet helpers -----

    @staticmethod
    def save_parquet(df: pd.DataFrame, path: Path) -> None:
        """Save DataFrame to Parquet with compression."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, compression="snappy")
        logger.debug(f"Saved {len(df):,} rows to {path}")

    @staticmethod
    def load_parquet(path: Path) -> pd.DataFrame:
        """Load DataFrame from Parquet."""
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")
        return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
db = Database()
