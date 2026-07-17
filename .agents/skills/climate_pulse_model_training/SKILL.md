---
name: climate-pulse-model-training
description: Train deep learning and Bayesian models for the Climate Pulse project (TFT, LSTM, ST-GNN, NWM).
---

# Climate Pulse Model Training

This skill guides you through training all Climate Pulse models. Each model has its own trainer, dataset loader, and checkpoint location.

## Prerequisites

1. Virtual environment installed: `make install`
2. Datasets downloaded: `make download-data`
3. ERA5 data downloaded (NWM only): `make download-nwm-data`

## Model Overview

| Model | Makefile Target | Script Flag | Task |
|:---|:---|:---|:---|
| TFT (Temporal Fusion Transformer) | `make train-elnino` | `--model tft` | ENSO / El Niño 12-month forecast |
| LSTM Baseline | `make train-elnino` | `--model lstm` | ENSO baseline comparison |
| ST-GNN (Pollution) | `make train-pollution` | `--model stgnn` | Per-sensor PM2.5 forecasting |
| NWM (Neural Weather Model) | `make train-nwm` | `--model nwm` | Global T2m + precipitation forecast |
| Bayesian Regression | `make train-climate` | `--model bayesian` | Climate sensitivity / feedback |

## Training Commands

### Train all models at once
```bash
make train
```

### Train individual models
```bash
make train-elnino       # TFT + LSTM
make train-pollution    # ST-GNN
make train-nwm          # Neural Weather Model
make train-climate      # Bayesian regression
```

### Override max epochs
```bash
PYTHONPATH=src .venv/bin/python src/scripts/train_models.py --model tft --epochs 50
```
The `--epochs` flag temporarily overrides `MODEL_MAX_EPOCHS` via environment variable and clears the settings cache.

## Key Source Files

| Model | Trainer | Dataset |
|:---|:---|:---|
| TFT | `src/models/el_nino/trainer.py` | `src/models/el_nino/dataset.py` |
| LSTM | `src/models/el_nino/trainer.py` | `src/models/el_nino/dataset.py` (`LSTMDataset`) |
| ST-GNN | `src/models/pollution/trainer.py` | `src/models/pollution/graph_dataset.py` |
| NWM | `src/models/nwm/trainer.py` (`NWMTrainer`) | `src/models/nwm/dataset.py` (`NWMDataset`) |
| Bayesian | `src/models/climate_feedback/bayesian_regression.py` | Built via `DataPipeline.build_regression_dataset()` |

## Configuration

All model hyperparameters are controlled via environment variables with the `MODEL_` prefix, loaded from `.env`, or overridden in `configs/runtime_conf.yaml`.

Key variables:
| Variable | Default | Description |
|:---|:---|:---|
| `MODEL_DEVICE` | `auto` | `auto`, `cpu`, `cuda`, `mps` |
| `MODEL_MAX_EPOCHS` | `100` | Training epoch cap |
| `MODEL_BATCH_SIZE` | `32` | Batch size |
| `MODEL_LEARNING_RATE` | `0.001` | Learning rate |
| `MODEL_TFT_ENCODER_LENGTH` | `120` | TFT encoder window (months) |
| `MODEL_TFT_PREDICTION_MONTHS` | `12` | TFT forecast horizon |
| `MODEL_LSTM_ENCODER_LENGTH` | `60` | LSTM lookback (months) |
| `MODEL_LSTM_PREDICTION_MONTHS` | `12` | LSTM forecast horizon |
| `MODEL_GNN_NUM_PERIODS` | `7` | ST-GNN sliding window |

## Hardware Acceleration

The `Makefile` automatically injects:
- `HSA_OVERRIDE_GFX_VERSION=10.3.0` for AMD Radeon RDNA2 GPUs (ROCm workaround)
- `OPENVINO_DEVICE=NPU` for Intel NPU if detected

For AMD GPUs, use `make install-rocm` instead of `make install`. For Intel NPU, use `make install-npu`.

## Checkpoints & Artifacts

- Checkpoints: `runtime/models/checkpoints/<model_name>/`
- Per-epoch metrics CSV: `runtime/models/checkpoints/<model_name>/metrics.csv`
- ST-GNN normalization stats: `runtime/models/checkpoints/pollution/norm_stats.json`

```bash
make clean-checkpoints   # Removes runtime/models/checkpoints/ and runtime/models/saved/
```

## Important Constraints

- The NWM (`NWMTrainer`) does **not** use `pytorch-lightning`. It is a native PyTorch training loop.
- TFT uses `pytorch-forecasting` ≥ 1.8.0, which pulls in `pytorch-lightning` as a **transitive** dependency only.
- Do **not** pin or explicitly install `pytorch-lightning` (see NF-10 in `docs/requirement_spec.md`).
