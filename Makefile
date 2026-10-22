# Convenience targets. Run `make help` for a summary.
PY ?= .venv/bin/python
FV ?= .venv/bin/fundusvision

.PHONY: help setup lint format test download train eval figures notebooks clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup: ## create .venv and install the package with dev extras
	python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -e ".[dev,download]"

lint: ## ruff lint + format check
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

format: ## auto-format with ruff
	.venv/bin/ruff format . && .venv/bin/ruff check --fix .

test: ## unit + CPU smoke tests on synthetic data
	.venv/bin/pytest

download: ## fetch both datasets from Kaggle into data/
	$(FV) download odir && $(FV) download vessel

train: ## train both models (MPS/CUDA if available)
	$(FV) train-segmentation --no-progress
	$(FV) train-classifier --no-progress

eval: ## evaluate both runs and regenerate docs/figures
	$(FV) eval-segmentation
	$(FV) eval-classifier
	$(FV) explain

notebooks: ## execute the walkthrough notebooks in place
	.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

clean: ## remove caches and build artefacts
	rm -rf build dist .pytest_cache .ruff_cache src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
