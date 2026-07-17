---
name: climate-pulse-configuration
description: Understand and modify the ClimatePulse configuration system (Pydantic-Settings, YAML files, environment variables). Use when changing hyperparameters, data paths, or runtime behavior.
---

# Climate Pulse Configuration

The project uses a **Pydantic-Settings** configuration system with a layered override hierarchy.

## Configuration Hierarchy (highest → lowest priority)

1. **Environment variables** (e.g., `export MODEL_MAX_EPOCHS=200`)
2. **`.env` file** at the project root (gitignored, loaded automatically)
3. **`configs/default.yaml`** — default runtime values
4. **`configs/dataset_conf.yaml`** — dataset-specific URL defaults
5. **Field defaults** in `src/config.py`

## Settings Classes

All settings are defined in [`src/config.py`](file:///home/arp/workspace/grokking.workspace/climate_pulse.git/src/config.py) and aggregated into a root `Settings` object.

| Class | Env Prefix | Purpose |
|:---|:---|:---|
| `DataSettings` | `DATA_` | Data paths, API keys, date ranges, country filters |
| `MetadataSettings` | `META_` | Path to `datasets/metadata.json` |
| `ModelSettings` | `MODEL_` | All training hyperparameters (TFT, LSTM, ST-GNN) |
| `TFTModelSettings` | *(nested)* | TFT architecture params |
| `NWMModelSettings` | *(nested)* | NWM architecture params |
| `LogSettings` | `LOG_` | Log level and log output directory |

## Accessing Settings in Code

```python
from config import settings

# Access nested settings
print(settings.data.openaq_api_key)
print(settings.model.tft_encoder_length)
print(settings.model.device)
```

The `settings` object is a lazy singleton proxy. It calls `get_settings()` on first access, which is cached with `@lru_cache`.

## Overriding Settings at Runtime

Because the settings object is immutable after creation, override via environment variables **before** the first access:

```python
import os
from config import get_settings

os.environ["MODEL_MAX_EPOCHS"] = "50"
get_settings.cache_clear()   # Forces re-initialization
```

This is the exact pattern used by `src/scripts/train_models.py` for the `--epochs` CLI flag.

## Key Environment Variables Reference

### Data
| Variable | Default | Description |
|:---|:---|:---|
| `DATA_START_YEAR` | `1950` | Historical data window start |
| `DATA_END_YEAR` | `2026` | Historical data window end |
| `DATA_FOCUS_COUNTRIES` | `IND,USA` | ISO-2 comma-separated country filter |
| `DATA_OPENAQ_API_KEY` | *(required)* | OpenAQ v3 API key |
| `DATA_OPENAQ_MAX_LOCATIONS` | `200` | Max sensor stations |
| `DATA_OPENAQ_LIMIT` | `10000` | Max measurements |
| `DATA_HISTORICAL_STREAM_LIMIT` | `0` | Max historical months (0 = no limit) |

### Model Training
| Variable | Default | Description |
|:---|:---|:---|
| `MODEL_DEVICE` | `auto` | `auto`, `cpu`, `cuda`, `mps` |
| `MODEL_MAX_EPOCHS` | `100` | Training epoch cap |
| `MODEL_BATCH_SIZE` | `32` | Batch size |
| `MODEL_LEARNING_RATE` | `0.001` | Initial learning rate |
| `MODEL_LOG_FREQUENCY` | `1` | Log every N epochs |
| `MODEL_TFT_ENCODER_LENGTH` | `120` | TFT lookback window (months) |
| `MODEL_TFT_PREDICTION_MONTHS` | `12` | TFT forecast horizon |
| `MODEL_LSTM_ENCODER_LENGTH` | `60` | LSTM lookback window (months) |
| `MODEL_LSTM_PREDICTION_MONTHS` | `12` | LSTM forecast horizon |
| `MODEL_GNN_NUM_PERIODS` | `7` | ST-GNN sliding window days |

### Logging
| Variable | Default | Description |
|:---|:---|:---|
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LOG_DIR` | `runtime/logs/` | Log file output directory |

## YAML Config Files

### `configs/default.yaml`
Provides runtime defaults for `data`, `tft`, `model` (and nested `nwm`) sections. Values here override Python field defaults.

### `configs/dataset_conf.yaml`
Provides data source URL defaults (NOAA ERSST URL, OpenAQ base URL, etc.). Loaded into `DataSettings` at startup.

## Important Rules

- Settings are **immutable** at runtime after first access.
- `PROJECT_ROOT` is resolved as two parents above `src/config.py` (i.e., the repo root).
- Use `timezone.utc`-aware timestamps; never `datetime.utcnow()`.
- All imports use bare module names (`from config import settings`), never `src.config`.
