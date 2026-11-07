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
