.PHONY: install install-rocm install-npu download-data train train-elnino train-pollution train-nwm train-climate clean clean-checkpoints clean-research-paper-artifacts clean-downloaded-datasets gen-research-paper-artifacts gen-research-paper-pdf help

# Use a local temporary directory inside the virtual environment to prevent
# "Disk quota exceeded" errors on systems with small/restricted tmpfs /tmp partitions.
export TMPDIR := $(shell mkdir -p .venv/tmp && pwd)/.venv/tmp

# Check for AMD GPU
HAS_AMD_GPU := $(shell lspci 2>/dev/null | grep -i -E 'vga|3d|display' | grep -i -E 'amd|radeon' >/dev/null 2>&1 && echo 1 || echo 0)
ifeq ($(HAS_AMD_GPU),1)
    ROCM_ENV := HSA_OVERRIDE_GFX_VERSION=10.3.0
else
    ROCM_ENV :=
endif

# Check for Intel NPU
HAS_INTEL_NPU := $(shell lspci 2>/dev/null | grep -i -E 'neural|npu' | grep -i intel >/dev/null 2>&1 && echo 1 || echo 0)
ifeq ($(HAS_INTEL_NPU),1)
    NPU_ENV := OPENVINO_DEVICE=NPU
else
    NPU_ENV :=
endif

# Sentinel file so Make only re-creates the venv when it doesn't exist.
.venv/.created:
	python3.12 -m venv .venv
	touch .venv/.created

venv: .venv/.created

# === PyG accelerator note ===
# torch-geometric 2.8.0+ replaced torch-scatter / torch-sparse with pyg-lib.
# pyg-lib is installed from the official PyG wheel index below.
# See: https://pyg.org/whl/

install: venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install torch==2.12.1
	.venv/bin/pip install pyg-lib --find-links https://data.pyg.org/whl/torch-2.12.1+cpu.html
	.venv/bin/pip install -e . --no-build-isolation

# === Setup (AMD ROCm) ===
# pytorch-lightning 2.6.5 is the safe version (2.6.2-2.6.3 had supply-chain compromise)
install-rocm: venv
	.venv/bin/pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/rocm6.1
	.venv/bin/python src/scripts/patch_rocm_elf.py
	.venv/bin/python src/scripts/install_rocm_deps.py
	.venv/bin/pip install pyg-lib --find-links https://data.pyg.org/whl/torch-2.12.1+cpu.html
	.venv/bin/pip install -e . --no-build-isolation

# === Setup (Intel NPU) ===
install-npu: venv
	.venv/bin/pip install openvino
	.venv/bin/pip install torch==2.12.1
	.venv/bin/pip install pyg-lib --find-links https://data.pyg.org/whl/torch-2.12.1+cpu.html
	.venv/bin/pip install -e . --no-build-isolation

# === Data ===
download-data:
	PYTHONPATH=src .venv/bin/python src/scripts/download_data.py

# === Model Training ===
train:
	PYTHONPATH=src $(ROCM_ENV) $(NPU_ENV) .venv/bin/python src/scripts/train_models.py --model all

train-elnino:
	PYTHONPATH=src $(ROCM_ENV) $(NPU_ENV) .venv/bin/python src/scripts/train_models.py --model tft
	PYTHONPATH=src $(ROCM_ENV) $(NPU_ENV) .venv/bin/python src/scripts/train_models.py --model lstm

train-pollution:
	PYTHONPATH=src $(ROCM_ENV) $(NPU_ENV) .venv/bin/python src/scripts/train_models.py --model stgnn

train-climate:
	PYTHONPATH=src $(ROCM_ENV) $(NPU_ENV) .venv/bin/python src/scripts/train_models.py --model bayesian

train-nwm:
	PYTHONPATH=src $(ROCM_ENV) $(NPU_ENV) .venv/bin/python src/scripts/train_models.py --model nwm

# === Paper Artifacts ===
gen-research-paper-artifacts:
	PYTHONPATH=src .venv/bin/python src/scripts/generate_paper_artifacts.py

gen-research-paper-pdf:
	cd docs/research_paper/ && pandoc research_paper.md -o climatepulse_research_paper.pdf \
    --pdf-engine=xelatex \
    --metadata=smart:false \
    -V geometry:"a4paper, top=2.5cm, bottom=2.5cm, includehead, includefoot" \
    -s

# === Cleanup ===
clean-checkpoints:
	find runtime/models/checkpoints/ -mindepth 1 ! -name ".gitignore" -exec rm -rf {} + 2>/dev/null || true
	find runtime/models/saved/ -mindepth 1 ! -name ".gitignore" -exec rm -rf {} + 2>/dev/null || true

clean-research-paper-artifacts:
	rm -rf runtime/paper_artifacts/

clean-downloaded-datasets:
	rm -rf datasets/raw/ datasets/processed/ datasets/metadata.json

# clean: clean-checkpoints clean-research-paper-artifacts clean-downloaded-datasets
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/ htmlcov/

# === Help ===
help:
	@echo "ClimatePulse - Available Commands:"
	@echo "  make install : Install the package"
	@echo "  make install-rocm : Install the package with AMD ROCm GPU support"
	@echo "  make install-npu : Install the package with Intel NPU support"
	@echo "  make download-data : Download all data sources"
	@echo "  make train : Train all models"
	@echo "  make train-elnino : Train El Niño model only"
	@echo "  make train-pollution : Train pollution model only"
	@echo "  make train-climate : Train climate feedback model only"
	@echo "  make train-nwm : Train Neural Weather Model (ST-GNN) only"
	@echo "  make gen-research-paper-artifacts : Generate publication-quality paper figures/results"
	@echo "  make gen-research-paper-pdf : Generate publication-quality paper pdf"
	@echo "  make clean : Remove build and test cache artifacts"
	@echo "  make clean-checkpoints : Remove trained model checkpoints"
	@echo "  make clean-research-paper-artifacts : Remove generated paper figures/results"
	@echo "  make clean-downloaded-datasets : Remove raw and processed datasets and JSON metadata"
