# Root Makefile. Each agent owns one make/<agent>.mk; shared targets live here.
SHELL := /bin/bash
UV ?= $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
# The project venv is venv/ (PyCharm's interpreter); CI uses the same path.
export UV_PROJECT_ENVIRONMENT ?= venv
RUN := $(UV) run --inexact
PY := $(RUN) python

.DEFAULT_GOAL := help
.PHONY: help sync test test-gpu test-slow lint format contracts contracts-check env-check

help:  ## List targets
	@grep -hE '^[a-zA-Z0-9_%-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

sync:  ## Install base + dev deps into venv/ (add EXTRAS="--extra train" etc.)
	$(UV) sync --inexact $(EXTRAS)

test:  ## CPU tests: no GPU, no network (hard rule 8)
	$(RUN) pytest -m "not gpu and not net and not slow" -q

test-gpu:  ## GPU tests, on the VM
	$(RUN) pytest -m gpu -q

test-slow:  ## Slow tests (soak, imminent check, parity)
	$(RUN) pytest -m "slow and not gpu and not net" -q

lint:  ## ruff + contract snapshot
	$(RUN) ruff check src tests scripts
	$(PY) -m intensity.schema check contracts

format:  ## ruff autofix and format
	$(RUN) ruff check --fix src tests scripts
	$(RUN) ruff format src tests scripts

contracts:  ## Re-export contracts/*.schema.json (needs ML lead approval + CHANGELOG entry)
	$(PY) -m intensity.schema export contracts

env-check:  ## GPU, NVDEC, PyNvVideoCodec, faster-whisper and LightGBM checks; snapshot to run/env_check.json
	$(PY) scripts/env_check.py

include make/engine.mk make/training.mk make/ingest.mk make/data.mk
