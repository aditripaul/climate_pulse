.PHONY: install install-dev install-rocm install-npu download-data download-nwm-data train train-elnino train-pollution train-nwm train-climate clean clean-checkpoints clean-eval-artifacts clean-downloaded-datasets gen-eval-artifacts sync-paper-artifacts gen-research-paper-pdf sync-patent-artifacts gen-patent-application-pdf project_proposal_pdf help

.DEFAULT_GOAL := help

# Use a local temporary directory inside the virtual environment to prevent
# "Disk quota exceeded" errors on systems with small/restricted tmpfs /tmp partitions.
export TMPDIR := $(shell mkdir -p .venv/tmp && pwd)/.venv/tmp

# Config file overrides (can be overridden via command line parameters, e.g. RUNTIME_CONF=custom.yaml)
export RUNTIME_CONF ?= configs/runtime_conf.yaml
export DATASET_CONF ?= configs/dataset_conf.yaml


# Check for AMD GPU
HAS_AMD_GPU := $(shell lspci 2>/dev/null | grep -i -E 'vga|3d|display' | grep -i -E 'amd|radeon' >/dev/null 2>&1 && echo 1 || echo 0)
ifeq ($(HAS_AMD_GPU),1)
    ROCM_ENV := HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_ENABLE_SDMA=0 TORCH_BLAS_PREFER_HIPBLASLT=0
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

install-dev: install

# === Setup (AMD ROCm) ===
# pytorch-lightning 2.6.5 is the safe version (2.6.2-2.6.3 had supply-chain compromise)
install-rocm: venv
	.venv/bin/pip install torch==2.6.0+rocm6.1 --index-url https://download.pytorch.org/whl/rocm6.1
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

download-nwm-data:
	PYTHONPATH=src .venv/bin/python src/scripts/download_nwm_data.py

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

# === Evaluation Artifacts ===
gen-eval-artifacts:
	PYTHONPATH=src $(ROCM_ENV) $(NPU_ENV) .venv/bin/python src/scripts/generate_eval_artifacts.py

sync-paper-artifacts:
	mkdir -p docs/research_paper/artifacts/eval_artifacts/run_results
	cp -r runtime/eval_artifacts/* docs/research_paper/artifacts/eval_artifacts/
	cp runtime/eval_artifacts/*.png docs/research_paper/artifacts/ 2>/dev/null || true

gen-research-paper-pdf: sync-paper-artifacts
	cd docs/research_paper/ && sed '/^```mermaid$/,/^```$/d' research_paper.md | pandoc research_paper.md -o climatepulse_research_paper.pdf \
    --pdf-engine=xelatex \
    --metadata=smart:false \
    -V geometry:"a4paper, top=2.5cm, bottom=2.5cm, includehead, includefoot" \
    -s

gen-research-paper_2-pdf: sync-paper-artifacts
	cd docs/research_paper/ && sed '/^```mermaid$$/,/^```$$/d' research_paper_2.md | pandoc -f markdown -o climatepulse_research_paper_2.pdf \
     --pdf-engine=xelatex \
     --metadata=smart:false \
     -V geometry:"a4paper, top=2.5cm, bottom=2.5cm, includehead, includefoot" \
     -s

sync-patent-artifacts:
	mkdir -p docs/patent/artifacts/eval_artifacts/run_results
	cp -r runtime/eval_artifacts/* docs/patent/artifacts/eval_artifacts/ 2>/dev/null || true
	cp runtime/eval_artifacts/*.png docs/patent/artifacts/ 2>/dev/null || true

gen-patent-application-pdf: sync-patent-artifacts
	cd docs/patent/ && pandoc patent_application.md -o climatepulse_patent_application.pdf \
    --pdf-engine=xelatex \
    --metadata=smart:false \
    -V geometry:"a4paper, top=2.5cm, bottom=2.5cm, includehead, includefoot" \
    -s

project_proposal_pdf:
	cd docs/ && pandoc project_proposal.md -o project_proposal.pdf \
    --pdf-engine=xelatex \
    --metadata=smart:false \
    -V geometry:"a4paper, top=2.5cm, bottom=2.5cm, includehead, includefoot" \
    -s

# === Save Run ===
save-run:
	@echo "Saving latest run artifacts and checkpoints for future inference..."
	@TIMESTAMP=$$(date +"%Y%m%d_%H%M%S"); \
	SAVE_DIR="runtime/saved/run_$$TIMESTAMP"; \
	mkdir -p "$$SAVE_DIR"; \
	if [ -d "runtime/models/checkpoints" ]; then cp -r runtime/models/checkpoints "$$SAVE_DIR/"; fi; \
	if [ -d "runtime/eval_artifacts" ]; then cp -r runtime/eval_artifacts "$$SAVE_DIR/"; fi; \
	echo "Run successfully saved to: $$SAVE_DIR"

# === Cleanup ===
clean-checkpoints:
	find runtime/models/checkpoints/ -mindepth 1 ! -name ".gitignore" -exec rm -rf {} + 2>/dev/null || true
	find runtime/saved/ -mindepth 1 ! -name ".gitignore" -exec rm -rf {} + 2>/dev/null || true

clean-eval-artifacts:
	rm -rf runtime/eval_artifacts/

clean-downloaded-datasets:
	rm -rf datasets/raw/ datasets/processed/ datasets/metadata.json

# clean: clean-checkpoints clean-eval-artifacts clean-downloaded-datasets
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
	@echo "  make download-nwm-data : Download ERA5 data for the Neural Weather Model (NWM)"
	@echo "  make train : Train all models"
	@echo "  make train-elnino : Train El Niño model only"
	@echo "  make train-pollution : Train pollution model only"
	@echo "  make train-climate : Train climate feedback model only"
	@echo "  make train-nwm : Train Neural Weather Model (ST-GNN) only"
	@echo "  make gen-eval-artifacts : Generate publication-quality paper figures/results"
	@echo "  make sync-paper-artifacts : Sync evaluation artifacts to research paper docs"
	@echo "  make sync-patent-artifacts : Sync evaluation artifacts to patent docs"
	@echo "  make gen-research-paper-pdf : Generate publication-quality paper pdf"
	@echo "  make gen-patent-application-pdf : Generate publication-quality patent application pdf"
	@echo "  make project_proposal_pdf : Generate project proposal pdf"
	@echo "  make save-run : Save the latest trained checkpoints and evaluation artifacts to a timestamped folder for future inference"
	@echo "  make clean : Remove build and test cache artifacts"
	@echo "  make clean-checkpoints : Remove trained model checkpoints"
	@echo "  make clean-eval-artifacts : Remove generated paper figures/results"
	@echo "  make clean-downloaded-datasets : Remove raw and processed datasets and JSON metadata"
