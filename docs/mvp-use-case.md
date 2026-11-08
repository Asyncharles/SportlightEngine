# MVP use case: four sports scored by the engine, the model shaping each viewer's feed

Written 8 Nov 2026 for Charles, before any implementation. It builds on `docs/deploy-wiring-plan.md`, which is merged: the engine publishes FeedEvents to the front end over RabbitMQ.

## 1. The use case

**Actor:** a viewer on https://sportlight.uk, with optional follows (sports, competitions, teams) and votes.

**Main flow:**
1. The viewer opens the site and sees the four demo feeds live:
   - football: Argentina v France, the 2022 final;
   - F1: Brazil 2019;
   - NBA: Game 7 of the 2016 Finals;
   - weightlifting: Worlds 2025, men's 79 kg.

   Each feed plays its clip, and the engine scores the same moment of the same file once a second.
2. Each feed's intensity comes from the engine and includes that sport's **forecast**: the trained model for football, F1 and NBA, and the rule for weightlifting.
3. The feeds are ranked by intensity, adjusted by the viewer's votes, and only the sports and teams the viewer follows are eligible.
4. When the forecast says a feed is **heating up** (a big chance, an on-track pass, a lead change or a decisive lift is likely soon), the viewer sees it. The main window can switch to that feed before the moment happens, within the viewer's pace and lock settings.
5. When the moment happens, the score peaks and the backend records a highlight.

**Done when:**
- All four feeds are live from `sportlight-engine`, each with a served forecast of `source: model` (football, F1, NBA) or `source: rule` (weightlifting).
- A "heating up" signal reaches the browser and is visible.
- Auto-switch can act on it.
- On each demo match, the ranking changes ahead of the labelled highlights.

## 2. Where things stand

| Area | Today | Gap to the MVP |
| --- | --- | --- |
| Football | Label (T1), training, in-sample bundle `20261108-7d7cb7`, serve parity proven | Choose to serve it (`CURRENT`); optionally retrain with the two approved 2018 matches |
| F1 | Adapter (S2), 101 races of events, demo race replays, highlight list `2019-20` | **Label is unmerged** (`feat/training-t2-f1`: on-track passes, P5 or better, awaiting ML lead); then train, export, serve |
| NBA | Adapter (S2), 2,634 games of events (seasons 2024–26 plus the five clip games), demo game replays, highlight list `0041500407` | **No label function** (`lead_change` is configured but not written: T3); then train, export, serve |
| Weightlifting | Rule forecast (logistic over the athlete's best, by design no model); session CSVs convert and replay in video time | Minimal converter: no `standings_gap_kg`, no `eta_game_s`. The imminent phase rule is tuned only on mock data (recall 0.24) |
| Model reaching the score | Wired: the forecast is one weighted part of `intensity_pct` (0.20 nominal, about 23% of the score while crowd, commentary and video are absent), which becomes the front end's `score` | Only active per sport once `models/<sport>/CURRENT` is set. Football is on the heuristic by Charles's decision |
| Model reaching the UI | Not possible: the front end's `IntensityPayload` is `{score, streamTime}`; zod strips anything else; no "heating up" anywhere in the UI | Contract, backend, selection and UI changes in `Asyncharles/sportlight`, plus the engine sending the fields |
| Running it | Image builds by WIF; workflows ready; front-end PR #4 ready | **No GPU VM**: L4 exhausted in 9 zones across 4 regions on 8 Nov (retry running) |

## 3. What each part needs

### A. F1

1. ML lead approves the T2 label (`training.labels.f1.max_position: 5`, base rate 13.9%).
2. Rebase `feat/training-t2-f1` onto `master` (conflicts in `CHANGELOG.md` and `status/training.md`), with a test. Merge.
3. `make train-all`, in-sample: export `models/f1/<version>/` and check the report. The highlight check runs on `2019-20`.
4. Serve parity on `f1-demo-race` (the slow test exists).
5. Set `models/f1/CURRENT` (ML lead), merge, build, deploy.

**Accept when:** `/health` shows `f1-demo-race` with `forecast_source: model`, and parity passes.

### B. NBA

1. Write the T3 label `lead_change` with a 60 s horizon: positive when the leading team changes in the next 60 s, from the running margin of `score` records. Add unit tests on synthetic games.
2. ML lead approves the definition and checks the base rate in the report.
3. The data is in place. Training must run on a laptop, because stats.nba.com blocks cloud IPs, but `train-all` only reads the bucket and never fetches.
4. Train, export, run parity on `nba-demo-game`, set `CURRENT`, deploy.

**Accept when:** the same checks as F1 pass.

### C. Weightlifting (a rule, not a model)

1. Training agent reviews the minimal session converter and completes T4:
   - `standings_gap_kg`, from the running totals and the best snatch and clean-and-jerk per athlete;
   - `best_kg` semantics;
   - whether a non-leaking `eta_game_s` is possible, for example a typical walk-on-to-lift time rather than the real decision time.
2. Tune the weightlifting imminent rule on the three tuning sessions (m71, m88, m94). Without event tapes (ingest I6), the session CSVs' walk-on and decision times can stand in as ground truth.
3. Decide whether the rule's `p(success)` should count as "heating up". Today `source: rule` never heats up on the forecast alone, only through imminent.

**Accept when:** the weightlifting feed's score rises at walk-ons for heavy attempts, and imminent precision and recall on the tuning sessions meet a target the ML lead sets.

### D. The model's result in the viewer's feed

Three levels; each builds on the one before.

**D1. Through the score (engine only, no front-end change).**
- Serve the bundles by setting `CURRENT` for football, F1 and NBA.
- The forecast then moves `intensity_pct`, and so the ranking and auto-switch, as one of the parts.
- Effort: a decision and a deploy per sport.
- Risk: the bundles are in-sample, so their quality on these same demo matches is optimistic by construction.

**D2. Heating up visible and acted on (front end and engine; contract change, ML lead and front-end approval).**
- **Contract** (`packages/shared/src/events.ts`, `schemas.ts`): add an optional `forecast` to `IntensityPayload`, `{p, heatingUp, label, horizonSec, source}`, and optional `reasons` (up to 3 short strings).
  - Being optional keeps `schemaVersion: 1`. Old producers, such as the mock, stay valid.
  - The engine already computes all of it (`ScoreUpdate.forecast`, `heating_up`, `reasons`).
- **Backend:** keep the fields on `IntensityReading` (`domain.ts`), so they reach the browser in the `feeds` upserts and `/api/feeds`.
- **Selection** (`packages/shared/src/selection.ts`): a heating-up feed gets a configurable boost in `rankOf`, and/or its challenge period is shortened. Either way it respects follows, votes, locks and pinned windows. All values go in `config.ts`, with tests.
- **UI:** a "Heating up" badge, optionally with the top reason, on cards and in the main window. Plus a setting to turn heating-up switching off.
- **Engine** (`engine/amqp.py`): send the new fields only once the front end has deployed the schema. Until then the backend would strip them, which is harmless but useless. Add a contract test against the new zod schema.
- **Delivery:** two PRs, front end first and then the engine.

**D3. Personalised forecasts.** Out of scope for the MVP.
- Today personalisation is follows and votes on top of one shared score.
- A per-viewer model would need viewer history, which doesn't exist yet.

### E. Platform (prerequisite for showing any of this)

1. **A VM to run the engine.** With only the clock lane, nothing uses the GPU yet; the L4 is for the future perception stream.
   - **Decision:** run the MVP engine on a CPU VM (e.g. `e2-standard-4`, about $0.15/h, tag `si`), avoiding the L4 stockout entirely, and move to the L4 when perception lands?
   - It's the same image and workflow. Only `compose.gcp.yaml`'s GPU reservation needs a CPU variant.
2. Merge front-end #4 and deploy it, once the engine publishes.
3. `SPORTLIGHT_REPO_TOKEN`, so the end-to-end test really runs. The D2 contract change makes this more valuable.

## 4. Decisions needed

**Decided 2026-11-08:** #7, run the MVP engine on a CPU VM. `si-dev` is an `e2-standard-4` in `europe-west4-a`.

| # | Decision | Owner |
| --- | --- | --- |
| 1 | Approve the F1 label (P5 or better) | ML lead |
| 2 | Approve the NBA `lead_change` definition once written | ML lead |
| 3 | Serve the in-sample bundles (`CURRENT`) for football, F1, NBA, accepting optimistic quality | Charles + ML lead |
| 4 | Weightlifting: is the rule forecast enough, and should it ever "heat up"? | ML lead |
| 5 | D2 contract change: optional `forecast` and `reasons` on `feed.intensity` | Charles + front-end owner |
| 6 | Heating-up behaviour in selection: boost size, challenge shortening, viewer opt-out | Front-end owner |
| 7 | Run the MVP engine on a CPU VM instead of waiting for an L4 | Charles |

## 5. Proposed order

1. **E1 (the VM decision)**, so everything below can be seen live.
2. **D1 for football:** the bundle already exists, so this is just the decision.
3. **A (F1)** and **B (NBA)** in parallel. Each is train → parity → serve. B is the larger, because its label is new.
4. **D2:** front-end PR (contract, backend, selection, badge), then the engine PR.
5. **C (weightlifting):** converter and rule tuning, alongside 3–4.

Each step is one small PR with CI green, as before, and contract changes get a `CHANGELOG.md` line.

## 6. Risks

- **In-sample models:** every metric is measured on the training data, demo matches included. The forecast will look good on the demo by construction, and heating-up thresholds are calibrated on data the model has seen.
- **Labels gate F1 and NBA:** T2 waits on the ML lead, and T3 isn't written.
- **D2 touches both repos:** it needs the front end to deploy first, and the end-to-end CI job needs the token to protect it.
- **GPU capacity:** if E1 stays on the L4, the whole MVP waits on Google's capacity.
