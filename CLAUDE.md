# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`sports-intensity`, built from `docs/Sports_Intensity___Implementation_Plan.md` (read it before larger
changes). One package, `intensity`, under `src/`. So far the foundation (F0, F1) and the engine stream
(S1 to S4, P4) exist; training has T0, S2 and the T1 pipeline (`make build-<sport>`). GPU
ingest/perception is not started. Progress and open asks
are in `status/<agent>.md`; decisions and contract changes are in `CHANGELOG.md`.

## Environment

- Python 3.12.7 in the project venv at `venv/`, managed by uv (`UV_PROJECT_ENVIRONMENT=venv`, set by
  the Makefile). uv lives at `~/.local/bin/uv`, which is not on PATH; the Makefile finds it.
- `make sync` installs base + dev deps (`EXTRAS="--extra train"` for LightGBM etc.).
- LightGBM on macOS needs `brew install libomp`; without it the engine serves heuristic forecasts and
  the bundle test skips.
- `venv/` and `.idea/` are IDE/environment artifacts; don't edit or search them.

## Commands

- `make test`: CPU tests, no GPU, no network (must stay green: hard rule 8). `make test-slow` runs the
  30-minute simulated soak.
- Single test: `venv/bin/pytest tests/unit/engine/test_imminent.py::test_weightlifting_phase_machine -q`
- `make lint`: ruff + contract snapshot check. `make mock`: engine on the mock at :8000/debug.
- `make eval-imminent SPORT=football SET=mock|tuning|demo`; `make samples` regenerates
  `contracts/samples/`.
- macOS has no `timeout`; to guard hanging tests use `perl -e 'alarm shift; exec @ARGV' 60 <cmd>`.

## Architecture (engine side)

- `schema.py` is the contract source of truth. Changing it means `make contracts`, a `CHANGELOG.md`
  entry and ML lead approval; `tests/contract` fails otherwise.
- `engine/core.py` (`Engine`) is shared by the live service (`engine/service.py`, FastAPI) and offline
  replay (`engine/simulate.py`). Time is always passed in as `now`, so the same code runs live or
  simulated. Events are applied on arrival, so rules only read the past (leakage test in `test_core.py`).
- Per tick and feed: adapter parts (`engine/adapters/`, stubs) + trackers (`engine/trackers/`) +
  forecast (`forecast.py`/`registry.py`) + imminent (`engine/imminent/`) → `scorer.combine` →
  reasons (`engine/explain/`, every number guarded against facts).
- `engine/mock.py` is deterministic per (seed, feed, step). The tests, `contracts/samples/` and
  `SET=mock` evaluations depend on it, so changing it changes those too.
- Every tunable comes from `config/` (hard rule 4); `config/default/engine.yaml` is validated by
  `intensity.config.EngineConfig` (`extra="forbid"`).
