# sports-intensity

Ranks simulated live sports feeds by intensity, once a second per feed, with plain-word reasons and a
warning *before* an intense moment. One Python 3.12 package (`intensity`), two processes at run time:

- **ingest** (GPU, `python -m intensity.ingest.run`): decodes each feed's RTSP video and audio once
  and turns them into `Event`s, `ClockTick`s and `Control` messages. *Not built yet (ingest agent).*
- **engine** (CPU, `python -m intensity.engine.service`): scores every feed and serves `ScoreUpdate`s
  on `/ws/scores`. Built, with a deterministic mock in place of the ingest.

The full design is in `docs/Sports_Intensity___Implementation_Plan.md`; progress per agent is in
`status/`.

## Quick start (laptop, no GPU)

```bash
make sync            # uv sync into venv/ (base + dev); EXTRAS="--extra train" for LightGBM etc.
make test            # CPU tests, no network
make mock            # engine driven by the mock: open http://127.0.0.1:8000/debug
```

uv is the environment manager. If it is not on your `PATH`, the Makefile also looks in
`~/.local/bin/uv`.

On macOS, LightGBM needs OpenMP (`brew install libomp`). Until a trained bundle exists, the engine
serves each sport's heuristic forecast, so this only matters for training and the bundle test.

## Running the engine against a separate ingest

```bash
make engine                      # engine alone on :8000, waiting for /ingest
make mock-ingest                 # in another shell: the mock streams over WebSocket, as the ingest will
```

Overlays: `--overlay local` (laptop) or `--overlay demo` (VM); any setting can be overridden with
`--set engine.imminent.hold_s=2.5`. Every run saves its resolved config under `run/engine/<time>/`.

## Consuming `/ws/scores` (front end)

Connect a WebSocket to `ws://<engine>:8000/ws/scores`. Each message is one `ScoreUpdate` JSON
object (`contracts/ScoreUpdate.schema.json`; examples in `contracts/samples/`). You get the latest
update per feed on connect, then one per feed per second, plus an extra one the moment
`imminent.heating_up` turns on. A slow client skips stale updates and never builds a backlog.

- Rank on `intensity_pct` (0 to 100 within the sport).
- Show the "heating up" badge and auto-switch on `heating_up`; `heating_up_by` says whether the
  forecast, the imminent signal, or both lit it.
- `reasons` holds at most three sentences. Every number and name in them comes from their `facts`.
- `t_video` is the position in the source video, so the player can hold an update until playback
  reaches it (decision D3). `stale` means the feed's clock stopped.

Other endpoints: `/feeds`, `/health`, `/metrics`, `POST /admin/reload`,
`POST /admin/switch {"key": "engine.imminent.enabled", "value": false}`, and `/debug`.

## Adding a feed

1. Add an entry to `config/feeds.yaml` (`feed_id`, `sport`, `sample`, `rtsp_path`, `match_ref`, `sync`,
   `asr`). Demo matches are kept out of training automatically (`demo: true` is the default).
2. Commit the sync table at `data/sync/<clip_id>.csv` and check it with `make sync-sheet CLIP=<clip_id>`.
3. Put the match records at `data/events/<sport>/<match_id>.jsonl` (the training agent's step 1).

The engine also accepts feeds it has never heard of (clones for E3): it learns the sport from the
first event.

## Imminent rules

`config/imminent/<sport>.yaml` holds every weight and threshold. To check a rule:

```bash
make eval-imminent SPORT=football SET=tuning   # replays data/tapes/ for the tuning clips
make eval-imminent SPORT=football SET=mock     # mock footage, checks the pipeline end to end
```

Reports land in `reports/imminent/<sport>/<rule_version>/<set>/report.html`. Tune only on
`SET=tuning`; `SET=demo` is scored once per rule version (hard rule 6).

## Not built yet

Training (`make build-/train-/eval-<sport>`, owned by the training agent), ingest and perception
(`make feeds-up`, `ingest`, `record-tapes`, `video-calibrate`, `transcripts`, `clock-check`, owned
by the ingest agent), the demo stack (`make demo`) and E3 (`make bench-e3`, `make report`). Those
targets exist and say who owns them. `make env-check` is written but can only pass on the L4 VM.

## Data sources

Match data will come from [StatsBomb open data](https://github.com/statsbomb/open-data) (credit
StatsBomb under its user agreement), [FastF1](https://github.com/theOehrly/Fast-F1), the
[Jolpica F1 API](https://github.com/jolpica/jolpica-f1) and
[nba_api](https://github.com/swar/nba_api). Broadcast footage lives only in the private bucket and
is never committed.
