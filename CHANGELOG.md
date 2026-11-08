# Changelog

Contract, label, clip-setting and weight changes need the ML lead's approval and a line here
(hard rule 9).

## Spike score for the demo: the UI switches on moments (Charles, 2026-11-08)

**What happened:** live on sportlight.uk, the main window switched once in 4 minutes and then stayed on F1. F1's intensity climbs lap by lap (stakes), so its level percentile against its own history stays high for most of its loop.

**Changes:**
- New `engine.percentile.mode: spike` (default `level`, so the plan's behaviour is unchanged elsewhere).
  - It ranks the intensity minus the feed's own baseline, an average over about `spike_baseline_s: 60`.
  - The baseline resets on a loop.
  - The pre-roll fills the window with spike values and hands its EMA and baseline to the live feed.
- `config/demo.yaml` (the deployed engine) turns it on, with `tie_tolerance: 0.005` to suit the smaller values.

**Validation:** the four demo clips looping for 78 min at 1 Hz, through the UI's hysteresis (20 s minimum, +10 for 3 s). A labelled moment is caught if the main window shows its feed from 10 s before to 30 s after it. Highlight lists: football, F1 and NBA from `data/highlights`; weightlifting from the decisions on attempts 2 and 3.

| Score (real engine) | Switches | Moments caught | Chance (same sequence, shifted times) |
| --- | --- | --- | --- |
| level | 35 | 29/75 (39%) | 30% |
| spike, 60 s | 82 | 46/75 (61%) | 39% |

The scratch variants with 120 s and 300 s baselines caught 53% and 51%, each against a chance rate of about 35%.

**Side effect on the front end:** it records a highlight when the score enters 78 or more. In spike mode that happens about twice as often (102 entries against 45 in 78 min).

## Restarts take seconds, not 20 (deploy agent, 2026-11-08)

**What happened:** the `69e3742` deploy left sportlight.uk without fresh scores for 19.3 s. The brief's target is 10 s.
- The engine took 7.5 s to stop, because uvicorn waited for its open WebSockets.
- The ingest took 10.5 s, Docker's full stop timeout: Python as PID 1 ignores SIGTERM, and the ingest also kept running after its health server stopped.

**Changes:**
- Compose: `init: true` and `stop_grace_period: 5s` on both services, plus `start_interval: 1s` on the engine's health check, so the ingest starts within a second of the engine being ready.
- The engine and the ingest's health server use `timeout_graceful_shutdown` (2 s and 1 s).
- The ingest cancels its clock lane and closes the engine link as soon as its server stops.
- Locally, both exit 0.17 s after SIGTERM.
- `deploy dev` gains a `restart` input: `FORCE_RECREATE=1` makes `run_demo.sh` recreate the containers on an unchanged release, giving a clean restart for the recovery test.

## intensity_pct: tie tolerance and per-feed pre-roll (Charles, 2026-11-08)

**What happened:** on the live `si-dev` run, F1 and weightlifting sat at `intensity_pct` 100 for the first ~10 minutes, so the UI's auto-switch would never have left F1.
- **Cause:** each demo sport has one feed, and there is no `intensity_ref.json`, so a feed is ranked against its own seconds-old history. F1's stakes rise lap by lap, and the weightlifting EMA converges from below. A value that creeps up is always "the highest so far".
- **Offline check:** a 1 Hz replay of the four demo clips reproduced it: F1 averaged 89 in the first 10 minutes, then the feeds rotated. Ranking against a reference distribution instead made NBA lead 85% of the time, so that wasn't adopted.

**Changes (the meaning of `intensity_pct` is unchanged):**
- `engine.percentile.tie_tolerance: 0.01`: intensity is ranked rounded to 0.01, so drift by noise is a tie, not a new high.
- `engine.percentile.preroll_s: 1800`: on a feed's first clock tick, a shadow engine replays its match data at 1 Hz over the preceding clip time (up to 30 min, within the loop). It puts those intensities into the sport window at the engine times they would have had. Every (re)start then ranks against realistic history.
  - Cost: 0.16–0.3 s per demo feed on the laptop. Feeds without match data (the mock, clones) are skipped.
  - First live percentiles after a start with 30 min of history: football 74, F1 21, NBA 53, weightlifting 70 (F1 was 100).
- Either setting can be switched off with 0.

## Sync check: the match data lines up with the video (deploy agent, 2026-11-08)

- `make sync-check CLIP=<clip_id>|demo [FROM_BUCKET=1]` (`engine/synccheck.py`) picks about 10 key moments spread over a clip, maps them through the sync table (weightlifting sessions are already in video time), and grabs frames 5 s before, at and after each.
  - Each row is labelled with what the broadcast should show: football minute, NBA quarter clock, F1 lap, the lifter and weight.
  - `FROM_BUCKET=1` streams the clip from the private bucket over authenticated HTTPS, so no footage is copied to a laptop. Only 320 px thumbnails are written, to `data/reports/sync/` (the bucket's `reports/`).
  - `synccheck shift --clip … --by S [--row N]` corrects a measured offset.
- `intensity.data.bucket()` falls back to `GCS_BUCKET` in `.env`, reading that key only.
- **Results on the four demo clips (spot-checked frames; full sheets in `gs://…/reports/sync/`):**
  - football: the broadcast clock matched the prediction exactly (79:24 at Mbappé's penalty, 107:58 at Messi's extra-time goal);
  - NBA: matched to within 1 s (Q4 11:34 exact, 5:38 against 5:37);
  - F1: matched at lap level (the lap tower reads the next lap after each crossing, as the lap data implies);
  - weightlifting: matched at the decision (the lift 5 s before, the result graphic at the decision time).
  - No offset needed.

## MVP engine on a CPU VM (Charles, 2026-11-08)

- L4 GPUs were out of stock (`ZONE_RESOURCE_POOL_EXHAUSTED`) for `g2-standard-8` and `g2-standard-4` in nine zones: europe-west4 a/b/c, west1 b/c, west2 a/b, west3 a/b.
- The clock lane needs no GPU, so `si-dev` runs on an `e2-standard-4` CPU VM (`GPU=0 infra/gcp/create_gpu_vm.sh`; `setup_vm.sh GPU=0` skips the NVIDIA driver and toolkit). It has the same image, workflow, service account and tag `si`, with label `role=engine`.
- `docker/compose.gpu.yaml` now holds the ingest's NVIDIA runtime and L4 reservation. `deploy-demo` layers it; `deploy-dev` (CPU `si-dev`) doesn't. `run_demo.sh` checks the GPU only when that overlay is in use.
- `docs/mvp-use-case.md`: what F1, NBA and weightlifting need, and how the model reaches the viewer's feed (written before implementation).

## Data package in the bucket; 2018 matches approved (Charles, 2026-11-08)

- The package is uploaded and `make data-verify` passes:
  - 17 clips (4 demo, 13 tuning, 3 audio-only) under the labels' exact names;
  - 592 stills;
  - the five NBA clip games, converted to `events/nba/`. The NBA demo feed now replays Game 7;
  - private: anonymous GET returns 403.
- The ML lead approved the 2018 World Cup StatsBomb files (`raw/football/7560.json`, `7567.json`). They are kept in the bucket.
  - The sync guard stays (`ALLOW_RAW_FOOTBALL_2018=1` to push).
  - `verify_bucket.py --allow-raw-football` now passes.
  - They are not converted or trained on yet.
- The football forecast stays on the heuristic: `models/football/CURRENT` is not set for the in-sample bundle.

## Deploy wiring, Milestone 3: FeedEvents to the front end over RabbitMQ (deploy agent, 2026-11-08)

New base dependency **aio-pika 9.6** (approved by Charles).

- `engine/amqp.py` maps every `ScoreUpdate` to the front end's versioned `FeedEvent`s (Asyncharles/sportlight `packages/shared/src/events.ts`, `schemas.ts`), on exchange `sportlight.feeds` (topic, durable, declared with the backend's arguments; queues are never declared).
  - `feed.started` is sent first, on every reconnect and every 10 s.
  - `feed.intensity` carries `score = round(intensity_pct)` and `streamTime`, for non-stale updates.
  - `feed.updated` carries `availability` when a feed goes stale or recovers, and `source` when the clip clock moves.
  - `feed.ended` is never sent on a restart.
  - Routing key: `feed.<type>.<sportId>.<feedId>`.
  - Sports map `f1` to `motorsport` and `nba` to `basketball`. Titles and media URLs come from `config/publish.yaml`, copied from the front end's `media/local-sources.json`.
- **Timing:** `mediaOffsetSec` and `streamTime` come from `run/feeds_state.json`, so the browser's loop position equals the engine's `t_video`.
- **Not sent:** forecast, `heating_up`, trend, parts and reasons. The front end's schema has no field for them, and no front-end code uses `heating_up`.
- The publisher:
  - is active only when `AMQP_URL` is set (`engine.publish`);
  - uses publisher confirms, reconnects with 0.5 to 30 s backoff, keeps a latest-wins slot per feed and a bounded lifecycle queue;
  - `offer()` never awaits, so scoring never blocks on the broker;
  - its status is in `/health`, with `degraded` when it's disconnected;
  - `/ws/scores` is unchanged.
- Tests:
  - `tests/contract/test_feed_events.py`: a strict mirror of the zod schema, browser/engine time alignment, and the publisher's announce, availability and reconnect behaviour against a fake broker.
  - `tests/e2e` and `.github/workflows/e2e.yml`: RabbitMQ 4.3.7 and the real front-end backend in CI, with no GCP. It checks that the dead-letter queue stays empty, that the backend has a fresh run within 10 s of an engine restart, and that its queue doesn't grow.

## Deploy wiring: image build and deploy workflows (deploy agent, 2026-11-08)

- `image.yml`: on a merge to `master` (not `main`; the repo's default branch is `master`) or by hand, `infra/gcp/cloudbuild.yaml` builds `$AR_REPO/si:<sha>` on Cloud Build as `si-build`, authenticated as `si-web-ci` through WIF.
  - Layers are cached from `:cache`, and a smoke step imports both services.
  - The build is submitted async and polled.
- `deploy-reusable.yml` is the handoff's, with inputs unchanged.
- `deploy-dev.yml` (si-dev) and `deploy-demo.yml` (si-demo, environment `demo`) resolve the handoff's TODOs:
  - `COMPOSE_FILE=docker/compose.yaml:docker/compose.gcp.yaml`;
  - engine `http://127.0.0.1:8000/health` (body `status` must be ok or degraded, because the route is always 200);
  - ingest `http://127.0.0.1:8001/health` (200 only when healthy);
  - release env with `SI_DATA`, `SI_RUN`, `GCS_BUCKET` and the AMQP host, vhost, user and the secret's *name*.
- `deploy/remote-up.sh` reads `rabbitmq-app-password` on the VM with the VM's identity into `/opt/si/secrets/engine.env` (0600, percent-encoded into `AMQP_URL`). It is never in the repo, image, release files or logs.
- `infra/gcp/run_demo.sh` starts the stack; `infra/gcp/RUNBOOK.md` covers create, deploy, the rollback proof and logs.

## Deploy wiring: the clock-lane ingest, weightlifting replay, image-based Compose (deploy agent, 2026-11-08)

Charles chose the clock lane (plan D1), because ingest and perception are not built.

- `python -m intensity.ingest.run` (`make ingest-clock`; `ingest.mode: clock_only`) runs the clock lane.
  - Each feed loops its clip from a start time kept in `run/feeds_state.json`, the file the plan's pushers write and the front end's `feed-clock.ts` reads.
  - The start time survives restarts while the clip is unchanged.
  - It sends a `ClockTick` per feed at 4 Hz and a `Control(loop)` on wrap, and serves `GET /health` on :8001 (503 until the engine link is up).
  - Engine scores then come from match data replayed in clip time. Crowd, commentary and video parts are null until perception exists.
- Weightlifting session CSVs convert to match records with `convert --sport weightlifting`. This is a minimal T4 converter for the training agent to review: `attempt_called` at walk-on, `attempt_result` at the decision, and no `eta_game_s`, because the decision time would leak the future.
  - `Engine.add_feed` replays `session_csv` feeds with the identity sync, since their records are already in video time.
  - `feeds.yaml` is unchanged.
- `docker/compose.yaml` is image-based (`${IMAGE_SI}`), keeping `build:` for laptops.
  - Services: `engine` and `ingest`, with MediaMTX behind `--profile video`.
  - Bucket folders are mounted one by one from `${SI_DATA}`, and `models/` is no longer mounted over the image's.
  - `docker/compose.gcp.yaml` puts the ingest on the NVIDIA runtime with the L4 reserved, reads the engine's secrets from `/opt/si/secrets/engine.env`, and rotates logs.
- `.dockerignore` / `.gcloudignore`: no footage, data or secrets in the build context.

## Training on the full dataset: `training.split.mode: in_sample` (Charles, 2026-11-08)

Charles decided there is no validation split: train on the full dataset and run on that same data. That suspends hard rule 6 (demo matches never train) and the T1 held-out evaluation. **ML lead to note.**

- One switch, `training.split.mode` in `config/default/training.yaml`: `in_sample` (default) or `holdout` (the old grouped split, unchanged).
  - In `in_sample`, `split.json` puts every match, demo matches included, in `all`; train, calibration and evaluation all see the same rows.
  - There is no search, because choosing settings by in-sample score would pick the most overfit one. One fixed setting (`training.in_sample`, 400 rounds) trains without early stopping.
- `assert_no_demo_leak` runs only in `holdout`. The demo-leak test is now mode-aware:
  - `holdout` still forbids demo matches in training;
  - `in_sample` asserts they are included.
- Every bundle `meta.json` carries `"split_mode"` and `"evaluation": "in-sample" | "held-out"`. `report.json` carries `metrics.evaluation`. `report.html` shows an in-sample banner and labels its numbers "in-sample", not "test".
- New serve-parity test (`tests/unit/training/test_serve_parity.py`): live engine forecasts equal the offline transform within 1e-6 (at the contract's 5 decimals).
  - The fast version, on a synthetic match, runs in `make test`.
  - The `slow` version runs on every demo match that has a CURRENT bundle and its event file.
- **Engine fix found by serve parity:** `Replayer` sorted records released at the same `t_video` (everything before the clip starts) by game time alone. That interleaved first-half stoppage time with the second half: 123 period switches before the 2022 final's clip.
  - The adapter reset its period state each time, so the first ~20 s of served forecasts differed from training's.
  - It now sorts by `(t_video, period, game_time)`. Parity holds on all 4,525 ticks of `fb-arg-fra`.
- `make train-all` (`scripts/train_all.sh`) pulls `raw/`, `events/`, `tables/` and `highlights/` from the bucket, converts offline, then builds and exports football, f1 and nba to `models/`.
  - It never fetches unless `FETCH=1`, and refuses NBA fetching on GCE.
  - `PUSH=1` pushes results back.
  - No video head exists to train.
  - CURRENT is not moved.
- Training reports move to `data/reports/training/`, so they sync to the bucket's `reports/` like every other data folder.

## Deploy wiring, Milestone 1: bucket and data layer (deploy agent, 2026-11-08)

Plan: `docs/deploy-wiring-plan.md`.

- Data stays in the existing private bucket `gs://sportlight-514020-si-private`, by Charles's decision. No new bucket was created.
  - `infra/gcp/bucket.sh` sets the labels `app=sports-intensity` and `owner=charles` and re-asserts the settings: uniform access, public access prevention enforced, versioning off.
  - It also adds the team and GPU-VM bindings. si-web's `si-vm-sa` access is left as it was (plan D8).
- `intensity.data` is the one data layer. Every `data/...` config path resolves under `INTENSITY_DATA_DIR`, defaulting to `<repo>/data`, through `config.resolve_path`, so the laptop and the VM read the same relative paths.
  - The bucket mirrors `data/` without the prefix.
- `scripts/sync_bucket.sh` (`make data-pull` / `make data-push`) never deletes at the destination without `--delete`, and never pushes the two 2018 StatsBomb files (`raw/football/7560.json`, `7567.json`) without `ALLOW_RAW_FOOTBALL_2018=1`.
- `make data-verify` checks the uploaded package: counts and sizes, every labelled clip at its path, and anonymous access refused.
- `infra/gcp/create_gpu_vm.sh` (si-dev: g2-standard-8, 1 L4, standard, 12 h max run, tag `si`, SA `si-gpu-sa`) and `infra/gcp/setup_vm.sh` (driver, Docker, NVIDIA toolkit, bucket pull, idle watchdog).
  - Human-only steps are in `infra/gcp/PREREQS.md`.

## v1, 2026-11-07: contracts (awaiting ML lead approval at the hour-0 gate)

`src/intensity/schema.py`, exported to `contracts/*.schema.json`. Shapes are exactly as in the plan's
Contracts section:

- `Event`, `ClockTick`, `Control`, `Envelope` into the engine; `ScoreUpdate` (with `PartValue`,
  `Forecast`, `Imminent`, `Reason`) out of it.
- Seven differences from the Global Plan: `loop_epoch` everywhere; `t_video` and `stale` on
  `ScoreUpdate`; the `ClockTick` and `Control` messages; `payload.score` null for the v0 video
  model; `forecast` as the seventh part; `imminent` as the eighth part and a field; the `voice`
  and `ocr` sources.
- Per-source payloads are pydantic models (`VideoPayload`, `CrowdPayload`, `VoicePayload`,
  `OcrPayload`, `CommentaryPayload`, `MatchPayload`) that forbid unknown fields. `Event` validates
  its payload against the model for its source.
- Validation the plan implies but does not spell out, added without changing any shape:
  `Envelope.kind` must match the type of `data`; `ScoreUpdate.parts` must have exactly the
  `PART_NAMES` keys; `ScoreUpdate.reasons` holds at most 3.
- The model bundle layout is unchanged from the plan (`model.txt`, `calibrator.json` with `x`/`y`
  isotonic points, `meta.json`, `../CURRENT`).

## Engine defaults and decisions to review (engine agent, 2026-11-07)

Each item is a default the engine builds on until its owner changes it.

- **Weights** (`config/weights.yaml`): video 0.15, forecast 0.2, imminent 0.15 per D6. Closeness
  0.20, momentum 0.15, stakes 0.15, crowd 0.15 and commentary 0.10 are **placeholders**; the
  Global Plan's starting weights were not available. ML lead, D2.
- **Imminent thresholds** are untuned defaults: football and NBA 0.45, F1 0.45, weightlifting
  0.72. The weightlifting threshold sits above the setup cue (0.7), so only `setup_hush` or the
  lift can fire. At 0.6, setup alone fired about 13 s early on mock footage. P4 tunes all of them
  on tuning footage.
- **Weightlifting lift spike**: the plan says "a pan or motion spike". A percentile alone fires on
  noise when the platform is still, so a spike also needs the value to be at least
  `lift_spike_ratio` (2x) the feed's rolling median. The lift enters on the spike itself rather than
  after the 1 s phase hold, because the spike is brief.
- **Weightlifting ETA prior**: read as "once game time passes ETA + 30 s while still idle, the cue
  holds 0.2".
- **Forecast `heating_up` for rule-based p** (weightlifting): always false. A high p(success) means
  a safe lift, not an intense one. The weightlifting badge comes from imminent.
- **Commentary part** drops out after 60 s with no ASR segment (`asr: off`, or the link is down);
  otherwise silence decays it to 0 over 10 s.
- **Adapter interface** (owner: training agent): the engine also uses `facts(game_time)` (state
  facts keyed by name, which SHAP reasons and imminent rules read: `min_gap_s`, `p_success`,
  `attempt_eta_game_s`), `player_names()` and `FORECAST_SOURCE`. The stub adapters from milestone F1 implement all
  of them.
- **`calibration.json` format** proposed to the ingest agent: `{sport: {signal: [deciles]}}`, with
  signals `motion`, `pan_speed`, `cuts_per_10s`, `prompt:<text>`, `loudness_rel_db`, `rise_db_s`,
  `energy_rel_db`, `pitch_rel_st`, `pace_rel`. Until it exists, trackers use the placeholder deciles in
  `engine.trackers.default_seeds`.
- **`intensity_ref.json` format** proposed to the training agent: `{sport: [intensity values]}` at
  `models/intensity_ref.json`.
- **Event tapes**: `t_emit_ns` must be the virtual wall time at which the live ingest would have sent
  each message; offline replay releases envelopes in that order.
- **Event facts**: trackers prefer the facts an event carried, keyed by its payload field name
  (`loudness_rel_db`, `rise_db_s`, `pan_speed`, `shot_type`, `energy_rel_db`, ...). Without one, they
  build a fact from that payload field, so every number still comes from the event.
- **Reason polish** (D10, off): model from config, default `claude-opus-5-5` at `effort: low` so it
  fits the 1.5 s timeout. It also sends Anthropic's server-side refusal fallback
  (`fallbacks: "default"`, beta `server-side-fallback-2026-07-01`). Output is used only if it passes
  the fact guard.
- **Feed registry**: `Feed.demo` (default true) marks demo matches for the split and the imminent
  tuning guard. It is a new optional field in the shared `config/feeds.yaml`.

## Engine fixes from code review (2026-11-07)

- The feed clock runs on at most `engine.max_extrapolate_s` (0.5 s) past its last tick, instead
  of up to `stale_after_s`, and never steps backwards within a loop. Before, a stalled stream
  released match records early and then made `t_video` jump back.
- A feed is scored at most once per engine step, so an out-of-cycle update and the 1 Hz tick at the
  same instant no longer double-step the EMA, trend and percentile.
- An unregistered feed first seen in a later loop keeps its first clock tick.
- `/admin/reload` also swaps `rule_version` and reloads lexicons. State-reading endpoints run on
  the event loop.
- New config keys: `engine.trackers.voice.drop_after_s` (3 s) and
  `engine.trackers.commentary.drop_after_s` (60 s), which were hard-coded before.
- Imminent evaluation: an NBA lead change is the go-ahead basket after the other team led, directly
  or after a tie; ties alone are not moments.
- Mock: weightlifting phases and hush follow the loop that is playing.

## Training data sources (training agent, 2026-11-07; F1 approved by the ML lead)

- F1 is sourced from Jolpica only, because FastF1's live-timing archive now needs an F1 account.
  The F1 features and the `on_track_pass` label are therefore built at lap granularity. With an F1
  account, FastF1 can be added later without changing the raw Jolpica files. The ML lead approved
  lap-granularity F1 on 2026-11-07.
- NBA uses the two most recent complete seasons, 2024-25 and 2025-26, regular season plus playoffs.

## S2 adapters and match records (training agent, 2026-11-07; part definitions await ML lead review)

- Real football, F1 and NBA adapters replace the F1 stubs; weightlifting keeps its rules but now
  returns real features (the full T4 version follows). Each adapter's docstring lists its record
  kinds and attrs, which `training/sources/<sport>.py` writes to `data/events/<sport>/*.jsonl`.
  Every kind and attr the mock sends is still accepted, so the mock and the engine tests are unchanged.
- Part definitions, windows, phrase thresholds and heuristic forecasts live in
  `config/adapters/<sport>.yaml` (hard rule 4) for the ML lead to review. `Adapter(cfg=None)` reads
  that file; `config_hash(cfg)` is for T1 bundles to store next to `feature_hash`.
- Feature lists grew: football 31, F1 14, NBA 20, weightlifting 3. No bundle exists yet, so nothing
  needs retraining.
- Football game time is the broadcast clock (second half from 45:00), so records order by
  `(period, game_time_s)`. Shoot-out goals are counted separately and never change the score.
- F1 records are per car per lap (`crossing`), each at that car's own line crossing. Safety cars
  are inferred when a lap's median time is 1.2 times the green median
  (`training.convert.f1.sc_slow_ratio`). On-track passes exclude pit swaps and lap-one changes.
- NBA possession is inferred from scores, last free throws, rebounds and turnovers; the game clock
  goes past regulation only when the score is tied.
- `contracts/samples/` were regenerated: the same shapes, with the real adapters' facts. On the mock
  the imminent check still meets every target. Recall is now football 1.00 (was 0.95) and NBA 0.955
  (was 0.91); F1 and weightlifting are unchanged.

## T1 football pipeline (training agent, 2026-11-08; awaiting ML lead review)

- New CLI steps: `featurize`, `label`, `split`, `train` (which calibrates), `evaluate`, `export`
  and `build`, with make targets `build-`, `train-`, `eval-` and `export-<sport>`. Paths and every
  setting are in `training.paths`, `featurize`, `search`, `calibration`, `evaluate` and
  `highlights` in `config/default/training.yaml`.
- Football label `big_chance`: a shot with xG >= 0.1 or a goal (own goals included, shoot-out kicks
  excluded) within 30 s, in the same period. Shoot-out ticks get no rows
  (`featurize.skip_periods`).
- Bundle `meta.json` adds `adapter_config_hash`, `label_cfg`, `params`, `best_iteration` and
  `rows` to the plan's fields. The layout is unchanged.
- Calibration: `calibrator.json` points sit at each isotonic block's centre, not at its edges, so p
  keeps the model's ranking. It is still monotone, still `x`/`y` interpolated by the engine.
- When no level reaches the precision target, the threshold sits above every validation p, so the
  forecast never heats up on its own; the report says so.
- Football adapter: 9 new attack features (40 in total). `windows_s` gains `instant: 30` and
  `possession_cap: 120`.
- Cached tables also hash the adapter, featurize and label source files, so a code change
  rebuilds them even when the feature names stay the same.
- `config/feeds.yaml`: football demo `match_id` set to 3869685 (the 2022 final, which the plan holds
  out). ML lead to confirm.
- `data/highlights/3869685.yaml`: draft highlight list for the ML lead to review or replace.
- The football model misses both targets (test PR-AUC 1.70x, highlights 4 of 10), so it is not
  exported and football keeps the heuristic.

## Clip labels from the data-collection kit (2026-11-08; awaiting ML lead review)

Labels for the 17 broadcast clips (4 demo, 13 tuning), made with the data-collection kit and checked
by eye on the video. Footage stays in the bucket (hard rule 10): demo clips go to `samples/`, tuning
clips to `samples/tuning/<sport>/<clip_id>.mp4`.

- **`config/feeds.yaml`**: the four demo feeds now point at the real clips. The feed ids are
  unchanged, so `contracts/samples/` still match.
  - Football: `fb_arg_fra_2022` (58' to the end of extra time), match 3869685.
  - F1: `f1_2019_brazilian_gp` (lap 45 to the flag), item `2019-20`.
  - NBA: `nba_20160619_cle_gsw` (Q4 to the buzzer), game `0041500407`.
  - Weightlifting: `wl_worlds2025_m79` (the whole session, `session_csv`, no sync table).
- **`data/sync/`**: 13 sync tables, one per clip, in each sport's existing game time:
  - football: the broadcast clock from 0', 45', 90' and 105';
  - NBA: elapsed game-clock seconds;
  - F1: seconds since lights out, from the leader's Jolpica lap times.

  Every table loads with `SyncTable`. NBA tables have one row per stretch of running clock (16 to 48
  rows). Baku 2021, Britain 2022 and Austria 2024 are left out: their only footage has stripes over
  the picture.
- **`data/highlights/`**: 12 lists, one per clip's match.
  - They hold only moments confirmed on the video. Unconfirmed F1 passes are left out: they are timed
    at the end of the lap, up to about 90 s late.
  - `clock` is game seconds, so NBA and F1 parse with the existing `parse_clock`.
  - Each list covers only its clip's span, not the whole match. On the NBA and F1 demos, the forecast
    check's top-10 peaks can land outside it.
  - The NBA demo's seven times all land exactly on a record from `sources.nba.to_records`.
  - `3869685.yaml`, the training agent's draft for the final, is unchanged. The kit has ten confirmed
    moments from 58' onwards. Six are also in the draft, and they agree within 1 s (the draft uses
    whole seconds). The kit adds four chances (59:51, 105:41, 107:57 and 122:59). The draft adds
    Lloris's save at 96:37 and the two first-half goals.
- **`data/weightlifting/`**: the four Worlds 2025 session CSVs: 207 attempts, with `t_video_walk_on`
  and `t_video_decision` read from the broadcast graphic, and order and result from the IWF results
  book. T4's converter should read these.

Match data still missing for some clips (to fetch on a laptop, since `data/raw/` is not shared):

- **NBA**: all five games are outside the configured seasons. The fetcher already takes explicit ids:
  `python -m intensity.training.cli source --sport nba --items 0041500407,0041600206,0041200224,0042100111,0041200231`.
  `featurize` reads every file in `data/events/nba/`, so the four tuning games would join training.
  The demo game is held out by `feeds.yaml`.
- **Football**: `7560` (Spain v Morocco) and `7567` (South Korea v Germany) are 2018 World Cup matches
  (competition 43, season 3). The config fetches only 2022, and fetching them by id would lose the
  match row (team names). This waits on the ML lead's decision on adding competitions.
