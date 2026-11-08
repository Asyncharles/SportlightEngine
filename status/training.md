# Training agent status

Updated 2026-11-08. Laptop (macOS arm64, Python 3.12.7), home network. Owns `src/intensity/training/`
(except `imminent_eval.py`, written by the engine agent for P4), `src/intensity/engine/adapters/`
(F1 stubs in place), `models/`, `data/highlights/`, `data/weightlifting/`,
`config/default/training.yaml`.

## Milestones

| Milestone | State | Evidence (command) |
| --- | --- | --- |
| T0 data fetchers | Done; football and NBA complete, F1 at 99 of 128 races | `pytest tests/unit/training/test_sources.py`; `INTENSITY_NET=1 pytest -m net tests/unit/training`; `make fetch-<sport>` |
| S2 adapters | Done; part definitions await ML lead review | `pytest tests/unit/training/test_adapters.py tests/unit/training/test_convert.py`; `make convert-<sport>` |
| T1 football | Pipeline done; first model misses both targets, so football keeps the heuristic | `make build-football`; `pytest tests/unit/training/test_pipeline.py`; report in `reports/training/football/<version>/report.html` |
| T2 to T4, parity | Not started; T2 (F1) waits for the last F1 races | |

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

T1 details:

- `make build-football` runs source, featurize, label, split, train (with calibration), evaluate and
  export in about 40 s from cached raw data. Each step is also its own CLI command and reuses its
  output when the inputs are unchanged. `build` exits 2 and exports nothing when a target is
  missed. `export` never touches `CURRENT`.
- Featurize replays each match through the live adapter once per game second: 400k ticks over 64
  matches. Label: a shot with xG >= 0.1 or a goal within 30 s, base rate 3.2%. The shoot-out is
  skipped. Split (seed 1234): train 45 matches, validation 9, test 10 including the final.
- Leakage guards are tests: features at t are unchanged when every later record is shuffled, moved
  or dropped; labels look strictly after t; demo matches can't reach train or validation; the
  engine's own loader checks the exported bundle.
- Latest model (staged as `run/training/football/20261108-e9a20d`, not exported):
  - Test PR-AUC is 1.70x the base rate (target 2x); validation was 2.8x. The heuristic manages
    1.14x on the same ticks. Lift in the top 5% is 2.9.
  - Highlight check: 4 of 10 peaks precede a highlight (target 6). Hits: Messi's penalty,
    Mbappé's two penalties and Lloris's save from Messi.
  - The test set is only 10 matches, so it's noisy: near-identical models have scored between
    1.35x and 2.07x on it. Grouped 5-fold CV on the 54 non-test matches puts the
    current features at about 2.1x, against 1.9x before the attack features were added.
- Changes made along the way:
  - Football gained 9 possession-relative attack features over the last 30 s (40 in total),
    because the 2 and 5 minute windows are too slow for a 30 s horizon.
  - Calibration interpolates between isotonic block centres; flat steps tied most ticks and broke
    peak picking.
- Football's highest unmatched peak is 84 s before Messi's penalty, at the foul that won it. The
  highlight list was drafted before any model ran and has not been edited to fit (hard rule 6).

## Blockers and asks

- F1 at lap granularity from Jolpica only was approved by the ML lead on 2026-11-07 (FastF1 now
  needs an F1 account).
- **ML lead**: review the part definitions in `config/adapters/<sport>.yaml` (closeness, momentum
  and stakes per sport, with the football stage weights).
- **ML lead**: confirm the football demo `match_id` 3869685 (the 2022 final) set in
  `config/feeds.yaml`, and review the draft highlight list in `data/highlights/3869685.yaml`
  (9 moments: the six goals, plus saves and chances). F1 and NBA demo ids are still needed before
  T2 and T3, since the split refuses to run without them. The weightlifting session CSV is still
  needed for T4.
- **ML lead**: decide whether football may add StatsBomb competitions (for example the 2018 World
  Cup, Euro 2020 and Euro 2024, about 165 more matches). With 45 training matches the model is
  data-starved, and the 10-match test set can't settle a 2x target either way.
- libomp is installed, so LightGBM and the bundle test now run. No bucket exists yet, so raw data
  stays in `data/raw/` and is not synced.
