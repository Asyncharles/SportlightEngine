# Engine agent status

Updated 2026-11-07. Built on a CPU laptop (macOS arm64, Python 3.12.7).

## Milestones

| Milestone | State | Evidence (command) |
| --- | --- | --- |
| F0 scaffold | Done, except the image build and env-check, which need Docker and the VM | `make lint test` green; `make env-check` written, unrun (no GPU here) |
| F1 contracts and mock | Done | `pytest tests/contract`; 4 feeds × all 6 sources in `test_mock.py`; samples in `contracts/samples/` |
| S1 scorer core | Done | `test_scorer.py`, `test_core.py`; tick p95 0.19 ms (simulated), 1.1 ms (live service, 4 mock feeds) |
| S2 replay and sync sheet | Done (adapters are F1 stubs; real ones are the training agent's) | `test_replay.py`, `test_syncsheet.py` on a synthetic mp4 |
| S3 reasons and service | Done | `test_explain.py`, `test_service.py`; `make mock`; guard rejects invented numbers and names |
| S4 trackers | Done on mock and fixture events; awaits real events at the hour-12 gate | `test_trackers.py` |
| P4 imminent rules | Rules, unit tests and evaluation done; tuning waits for tapes (I6) | `test_imminent.py`, `test_imminent_eval.py`; `make eval-imminent SPORT=<s> SET=mock` |

Mock imminent check (20 minutes per sport, `SET=mock`). Mock footage only tests the pipeline; it
says nothing about real broadcasts.

| Sport | Precision | Recall | Median lead | False alarms / 10 min |
| --- | --- | --- | --- | --- |
| football | 1.00 | 0.95 | 5.3 s | 0 |
| f1 | 1.00 | 0.94 | 6.5 s | 0 |
| nba | 1.00 | 0.91 | 5.5 s | 0 |
| weightlifting | 1.00 | 0.24 | 2.8 s | 0 |

Weightlifting recall is low by design: the context `1 − |2p − 1|` lets only near coin-flip lifts
cross the threshold.

## Measured

- `make engine-report` (run `20261108-095200`, `gs://…/reports/engine/`): tick per feed p95 0.19 ms; engine CPU
  0.6 ms per feed per second, which is about 1,600 feeds per core on the engine side alone (the GPU ingest is
  the real limit, E3); restart pre-roll 120 to 280 ms per feed. The probes run on a private engine, so the
  live service is never touched.

- Out-of-cycle imminent update reaches a `/ws/scores` client in under 200 ms (`test_service.py`,
  TestClient).
- Ingest to engine over the real WebSocket transport: p95 19.9 ms (mock streaming for 8 s; the mock
  sends in 0.25 s batches).
- 30 simulated minutes of 4 feeds through 3 clip loops: no stale updates, traced memory flat
  (`make test-slow`).

## Blockers and asks

- **ML lead**: approve contracts v1 and the placeholder weights in CHANGELOG.md; set the demo
  `match_id`s and sync tables in `config/feeds.yaml`.
- **Training agent**: replace the stub adapters (`engine/adapters/`). The engine also calls
  `facts()`, `player_names()` and `FORECAST_SOURCE` (see CHANGELOG). Write `models/intensity_ref.json`.
- **Ingest agent**: emit facts keyed by payload field name, write `calibration.json` in the format in
  CHANGELOG, and record tapes with virtual `t_emit_ns`. `config/prompts/` was seeded by the engine
  agent so the mock and video tracker had prompt names; it is yours.
- **Charles**: LightGBM on macOS needs `brew install libomp` (the bundle test is skipped without
  it). Docker is not installed on this laptop, so the image is unbuilt.
- **Plan conflict**: `training/imminent_eval.py` sits in the training agent's directory, but P4
  evaluation is engine work. I followed the layout; move it if you prefer.
