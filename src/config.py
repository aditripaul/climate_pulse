"""
Centralized configuration for ClimatePulse.

Uses pydantic-settings to load configuration from environment variables
and .env files with validation and type safety.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Project root: src/config.py → src/ → repo root (two parents)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load dataset defaults from dataset_conf.yaml
DATASET_CONF_PATH = PROJECT_ROOT / "configs" / "dataset_conf.yaml"


def _load_dataset_defaults() -> dict[str, Any]:
    try:
        if DATASET_CONF_PATH.exists():
            with open(DATASET_CONF_PATH) as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}


_DEFAULTS = _load_dataset_defaults()


class DataSettings(BaseSettings):
    """Configuration for data sources and storage."""

    model_config = SettingsConfigDict(
        env_prefix="DATA_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    dir: Path = Field(default=PROJECT_ROOT / "datasets", description="Root data directory")
    start_year: int = Field(default=2000, ge=1854, le=2030)
    end_year: int = Field(default=2026, ge=1854, le=2030)
    focus_countries: list[str] | None = Field(
        default=None,
        description="Comma-separated list of ISO country codes (e.g., 'US,IN'). None means global.",
    )

    # NOAA ERSST v5
    noaa_ersst_url: str = Field(
        default=_DEFAULTS.get("noaa_ersst", {}).get(
            "url", "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.ersst.v5/sst.mnmean.nc"
        )
    )

    # If 0, no limit is applied. If > 0, returns the last N months.
    historical_stream_limit: int = Field(default=0, description="Max historical months to stream")

    # OpenAQ v3
    openaq_base_url: str = Field(
        default=_DEFAULTS.get("openaq", {}).get("base_url", "https://api.openaq.org/v3")
    )
    openaq_api_key: str = Field(default="", description="Free API key from explore.openaq.org")
    openaq_max_locations: int = Field(
        default=_DEFAULTS.get("openaq", {}).get("max_locations", 50),
        description="Max stations to sample globally",
    )
    openaq_limit: int = Field(
        default=_DEFAULTS.get("openaq", {}).get("limit", 10000),
        description="Max measurements to fetch",
    )

    # NASA GISTEMP v4
    nasa_gistemp_url: str = Field(
        default=_DEFAULTS.get("nasa_gistemp", {}).get(
            "url", "https://data.giss.nasa.gov/pub/gistemp/gistemp1200_GHCNv4_ERSSTv5.nc.gz"
        )
    )

    # Climate TRACE
    climate_trace_base_url: str = Field(
        default=_DEFAULTS.get("climate_trace", {}).get(
            "base_url", "https://api.climatetrace.org/v7"
        )
    )

    # Niño 3.4 region bounds (5°N-5°S, 170°W-120°W)
    nino34_lat_min: float = Field(
        default=_DEFAULTS.get("noaa_ersst", {}).get("nino34_bounds", {}).get("lat_min", -5.0)
    )
    nino34_lat_max: float = Field(
        default=_DEFAULTS.get("noaa_ersst", {}).get("nino34_bounds", {}).get("lat_max", 5.0)
    )
    nino34_lon_min: float = Field(
        default=_DEFAULTS.get("noaa_ersst", {}).get("nino34_bounds", {}).get("lon_min", 190.0)
    )
    nino34_lon_max: float = Field(
        default=_DEFAULTS.get("noaa_ersst", {}).get("nino34_bounds", {}).get("lon_max", 240.0)
    )

    @property
    def raw_dir(self) -> Path:
        return self.dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.dir / "processed"

    @field_validator("dir", mode="before")
    @classmethod
    def resolve_data_dir(cls, v: str | Path) -> Path:
        p = Path(v)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p

    @field_validator("focus_countries", mode="before")
    @classmethod
    def parse_focus_countries(cls, v: str | list[str] | None) -> list[str] | None:
        if not v:
            return None
        if isinstance(v, str):
            return [code.strip().upper() for code in v.split(",") if code.strip()]
        return v


class MetadataSettings(BaseSettings):
    """Configuration for metadata storage."""

    model_config = SettingsConfigDict(
        env_prefix="META_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    metadata_path: Path = Field(
        default=PROJECT_ROOT / "datasets" / "metadata.json",
        description="Path to metadata JSON file",
    )



class TFTModelSettings(BaseSettings):
    hidden_size: int = 32
    lstm_layers: int = 1
    num_attention_heads: int = 4
    dropout: float = 0.1
    hidden_continuous_size: int = 16


class NWMModelSettings(BaseSettings):
    mesh_resolution: int = 5  # Icosahedron subdivision level (e.g. 5 is roughly 5.625 deg equivalent)
    hidden_dim: int = 128
    num_processor_layers: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5


class ModelSettings(BaseSettings):
    """Configuration for ML/DL model training."""

    model_config = SettingsConfigDict(
        env_prefix="MODEL_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    checkpoint_dir: Path = Field(default=PROJECT_ROOT / "runtime" / "models" / "checkpoints")
    device: str = Field(default="auto", description="auto | cpu | cuda | mps")
    max_epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=0.001, gt=0)
    log_frequency: int = Field(
        default=1, ge=1, description="Interval of training epochs to log progress"
    )

    # TFT (Temporal Fusion Transformer) hyperparameters
    tft_hidden_size: int = Field(default=32)
    tft_attention_head_size: int = Field(default=4)
    tft_dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    tft_encoder_length: int = Field(default=60, ge=1)
    tft_prediction_months: int = Field(default=12, ge=1)

    # LSTM baseline hyperparameters
    lstm_hidden_size: int = Field(default=64)
    lstm_num_layers: int = Field(default=2, ge=1)
    lstm_prediction_months: int = Field(default=12, ge=1)
    lstm_encoder_length: int = Field(default=60, ge=1)
    lstm_bidirectional: bool = Field(default=True)

    # Pollution ST-GNN hyperparameters
    gnn_num_periods: int = Field(default=7, ge=1, description="Sliding window size for ST-GNN")

    @field_validator("checkpoint_dir", mode="before")
    @classmethod
    def resolve_checkpoint_dir(cls, v: str | Path) -> Path:
        p = Path(v)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p


class LogSettings(BaseSettings):
    """Configuration for logging."""

    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    dir: Path = Field(default=PROJECT_ROOT / "runtime" / "logs")


class Settings(BaseSettings):
    """
    Root configuration aggregating all sub-settings.

    Usage:
        from config import settings
        print(settings.data.noaa_ersst_url)
        print(settings.grpc.address)
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data: DataSettings = Field(default_factory=DataSettings)
    metadata: MetadataSettings = Field(default_factory=MetadataSettings)
    tft: TFTModelSettings = Field(default_factory=TFTModelSettings)
    nwm: NWMModelSettings = Field(default_factory=NWMModelSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    log: LogSettings = Field(default_factory=LogSettings)

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for d in [
            self.data.dir,
            self.data.raw_dir,
            self.data.processed_dir,
            self.model.checkpoint_dir,
            self.log.dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Lazy Singleton Settings
# ---------------------------------------------------------------------------


@lru_cache
def get_settings() -> Settings:
    yaml_path = PROJECT_ROOT / "configs" / "default.yaml"
    config_dict: dict[str, Any] = {}
    if yaml_path.exists():
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f) or {}

    s = Settings(
        data=DataSettings(**config_dict.get("data", {})),
        metadata=MetadataSettings(**config_dict.get("metadata", {})),
        tft=TFTModelSettings(**config_dict.get("tft", {})),
        nwm=NWMModelSettings(**config_dict.get("nwm", {})),
        model=ModelSettings(**config_dict.get("model", {})),
        log=LogSettings(**config_dict.get("log", {})),
    )
    setup_logging(s)
    return s


class _SettingsProxy:
    def __getattr__(self, name):
        return getattr(get_settings(), name)


settings = _SettingsProxy()


def setup_logging(s: Settings) -> None:
    """Configure loguru based on Settings."""
    import sys

    from loguru import logger

    # Remove default loguru handler to prevent duplicate logs
    logger.remove()

    # Add console handler with configured level
    logger.add(
        sys.stderr,
        level=s.log.level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    # Add file handler if directory can be resolved/created
    try:
        s.log.dir.mkdir(parents=True, exist_ok=True)
        log_file = s.log.dir / "log"
        logger.add(
            str(log_file),
            level=s.log.level,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="10 days",
        )
    except Exception:
        pass
