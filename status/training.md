# Training agent status

Updated 2026-11-07. Laptop (macOS arm64, Python 3.12.7), home network. Owns `src/intensity/training/`
(except `imminent_eval.py`, written by the engine agent for P4), `src/intensity/engine/adapters/`
(F1 stubs in place), `models/`, `data/highlights/`, `data/weightlifting/`,
`config/default/training.yaml`.

## Milestones

| Milestone | State | Evidence (command) |
| --- | --- | --- |
| T0 data fetchers | Done; NBA and F1 downloads still running | `pytest tests/unit/training/test_sources.py`; `INTENSITY_NET=1 pytest -m net tests/unit/training`; `make fetch-<sport>` |
| S2 adapters | Done; part definitions await ML lead review | `pytest tests/unit/training/test_adapters.py tests/unit/training/test_convert.py`; `make convert-<sport>` |
| T1 to T4, parity | Not started; T1 (football) is next, as its data is complete | |

T0 details:

- `python -m intensity.training.cli fetch --sport football|f1|nba|all` (or `make fetch-<sport>`)
  writes one JSON file per match, race or game to `data/raw/<sport>/`, with the item list cached in
  `_index.json`. Writes are atomic and a rerun skips what is on disk. Checked by killing a live NBA
  run with SIGTERM mid-download: no partial files, and the rerun resumed at the next game.
  Retries back off exponentially and honour `Retry-After`. Items that still fail go to `_failed.json`.
- Sources and rates come from `training.sources` in `config/default/training.yaml`:
  - football: StatsBomb open data, 2022 World Cup, 64 matches, about 130 MB.
  - nba: `nba_api` PlayByPlayV3, 2024-25 and 2025-26 regular season plus playoffs, 2,629 games,
    about 470 MB, roughly 40 minutes at 0.6 s per request.
  - f1: Jolpica, 2019 to 2024, 128 races, about 10 MB, roughly 3.5 hours (19 requests per race at
    Jolpica's 500 requests an hour).
- Full downloads were started on 2026-11-07 with logs in `run/fetch/<sport>.log`. Football has
  finished.

S2 details:

- `make convert-<sport>` turns `data/raw/<sport>/` into `data/events/<sport>/<item>.jsonl` match
  records, the shape the engine's replayer serves. It skips up-to-date files, so rerun it as
  downloads land. After a converter change, run `make convert-all ARGS=--force`.
- Converted on 2026-11-07, all without failures: 64 football matches, 2,077 NBA games and 11 F1 races
  so far.
- Adapters: football (31 features), F1 (14), NBA (20), weightlifting (3; full version in T4). Their
  record kinds are in each adapter's docstring, and tunables are in `config/adapters/<sport>.yaml`.
  Per-tick cost on real matches is 7 to 15 µs.
- Tests on hand-built fixtures for every sport check:
  - every part fact passes the guard, and phrases contain no numbers;
  - reading state (features, parts, facts) never changes a later tick;
  - reset forgets everything;
  - sport-specific state: scores, windows, the shoot-out, battles, pit swaps, the safety car,
    possession, the bonus and the overtime clock.
  The same checks also run on the first three real matches per sport when `data/events/` exists.
- The engine suite passes unchanged, and the mock imminent check still meets every target.

## Blockers and asks

- F1 at lap granularity from Jolpica only was approved by the ML lead on 2026-11-07 (FastF1 now
  needs an F1 account).
- **ML lead**: review the part definitions in `config/adapters/<sport>.yaml` (closeness, momentum
  and stakes per sport, with the football stage weights).
- **ML lead**: demo `match_id`s in `config/feeds.yaml` (the split needs them), highlight lists in
  `data/highlights/`, and the weightlifting session CSV (T4).
- **Charles**: `brew install libomp` before T1 (LightGBM). No bucket exists yet, so raw data stays
  in `data/raw/` and is not synced.
