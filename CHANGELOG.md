# Changelog

Contract, label, clip-setting and weight changes need the ML lead's approval and a line here
(hard rule 9).

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
