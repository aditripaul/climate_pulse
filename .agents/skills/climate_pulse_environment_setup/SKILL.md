---
name: climate-pulse-environment-setup
description: Set up the ClimatePulse Python virtual environment and install all dependencies (CPU, AMD ROCm, or Intel NPU). Use this when onboarding, after dependency changes, or when resolving installation errors.
---

# Climate Pulse Environment Setup

This skill covers creating and managing the Python virtual environment for the Climate Pulse project.

## Python Version

The project requires **Python 3.12** exactly (pinned in `.python-version`).  
Verify with: `python3.12 --version`

## Installation (CPU / Default)

```bash
make install
```

This:
1. Creates `.venv/` using Python 3.12 (`python3.12 -m venv .venv`)
2. Upgrades pip
3. Installs `torch==2.12.1` (CPU build)
4. Installs `pyg-lib` from the PyG wheel index for `torch-2.12.1+cpu`
5. Installs the project in editable mode: `pip install -e . --no-build-isolation`

## Installation (AMD ROCm GPU)

```bash
make install-rocm
```

Uses ROCm 6.1 PyTorch index. Also runs `src/scripts/patch_rocm_elf.py` (Radeon 680M ELF patcher) and `src/scripts/install_rocm_deps.py`.

## Installation (Intel NPU)

```bash
make install-npu
```

Installs `openvino` in addition to the standard CPU dependencies.

## Build System Notes

- Build backend: `setuptools.build_meta` (PEP 517), declared in `pyproject.toml`.
- Always use `--no-build-isolation` with `-e .` to avoid reinstalling torch inside an isolated env.
- The `TMPDIR` is set to `.venv/tmp/` inside the Makefile to avoid `/tmp` quota issues.

## PyTorch & PyG Compatibility

- PyTorch is pinned to `2.12.1` across `Makefile`, `pyproject.toml`, and `README.md`.
- `pyg-lib` wheels are fetched from `https://data.pyg.org/whl/torch-2.12.1+cpu.html`.
- Do **not** upgrade PyTorch without verifying pre-built `pyg-lib` wheels exist on the PyG server for the new version and Python/OS combination.

## Environment Variables

The project loads all settings from `.env` at the repo root (gitignored).  
Copy `.env.example` to `.env` and populate values before running any scripts:

```bash
cp .env.example .env
```

Key secrets:
| Variable | Description |
|:---|:---|
| `DATA_OPENAQ_API_KEY` | Required for OpenAQ data download |

## PYTHONPATH Convention

All scripts under `src/` must be invoked with `PYTHONPATH=src`:
```bash
PYTHONPATH=src .venv/bin/python src/scripts/download_data.py
```

Modules import each other with bare names (e.g., `from config import settings`), never with a `src.` prefix.

## Cleanup

```bash
make clean    # Removes __pycache__, dist/, build/, *.egg-info/
rm -rf .venv  # Full venv reset
```
