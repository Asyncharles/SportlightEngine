# Deploy wiring plan: engine on GCP, wired to sportlight.uk, with bucket-backed data

Written 8 Nov 2026 for Charles, after a read-only analysis of this repo, the front end
(`Asyncharles/sportlight`) and the GCP project `sportlight-514020`. Nothing has been changed yet.
Phase 2 starts only after Charles's go-ahead on the decisions in section 6.

## 1. Where the brief and reality differ

| Brief says | Actually | Consequence |
|---|---|---|
| Repos are `Morgs27/SportlightEngine` and `Morgs27/sportlight` | Both are `Asyncharles/*`; the engine's default branch is `master` | Workflows trigger on `master`, not `main` |
| Engine workflows (`image.yml`, `deploy-reusable.yml`, `deploy-dev/demo.yml`, `deploy/remote-*.sh`) exist with TODOs | Only in `~/Downloads/sportlight-actions/engine/`, never committed. `infra/gcp/cloudbuild.yaml`, `setup_vm.sh`, `run_demo.sh` don't exist anywhere | We commit them, fixed (section 5, PR 4) |
| The front end's `deploy.yml` calls `deploy-reusable.yml@main` | It is self-contained (`sportlight/.github/workflows/deploy.yml`) | No cross-repo interface to preserve; we still keep `deploy-reusable.yml`'s inputs as written |
| CI identity `si-ci` | Doesn't exist. Charles: use `si-web-ci` for the engine too | `si-web-ci` needs extra bindings (section 7) |
| Create `gs://sportlight-private` | `gs://sportlight-514020-si-private` already exists. Charles: use it | No new bucket; add labels and access only |
| Engine serves real scores on `/ws/scores` | It scores only what arrives on `/ingest` (or `--mock`). **Ingest and perception are empty modules**; every `make/ingest.mk` target exits "not implemented" | Blocker for "scores the four demo clips"; see section 3 |
| The front end ranks on `intensity_pct`; `forecast.heating_up` drives a badge and auto-switch | The backend consumes a different envelope (`FeedEvent`), not `ScoreUpdate`. The browser ranks on `score` (+ votes) with hysteresis. **`heating_up` is not used anywhere in the front end** | We map `ScoreUpdate` to `FeedEvent` on the engine side. No ranking or routing message is needed |
| Footage stays private | Playing a clip in a browser on https://sportlight.uk serves it publicly; the prod media image deliberately has no engine footage | Needs a decision (section 6, D5) |
| Serve-parity test (live = offline within 1e-6) exists, keep it | Not implemented (`status/training.md:15`, "Not started") | We write it (PR 2) |
| Demo GPU VMs `si-dev` / `si-demo` | No GPU VM exists. `GPUS_ALL_REGIONS` = 1. `PREEMPTIBLE_CPUS` = 0 in europe-west4, so Spot g2 may be refused | `create_gpu_vm.sh`, standard provisioning, run only after Charles approves |

## 2. Inventory

**Engine (this repo).**
- `engine/service.py`: port 8000. Routes: `WS /ingest`, `WS /ws/scores`, `GET /health`, `/feeds`, `/metrics`, `/debug`.
- `/health` always returns 200, with `status: ok|degraded` in the body (`engine/core.py:378`).
- The four demo feeds are in `config/feeds.yaml` (`fb-arg-fra`, `f1-demo-race`, `nba-demo-game`, `wl-demo-session`).
- A feed is scored only once it is `seen`, which needs envelopes on `/ingest`. Its clock is driven by `ClockTick`s (`engine/core.py:233`, `feedstate.py:30-68`).
- The replayer releases match events as the feed clock passes them (`engine/core.py:85-95`). That needs `data/sync/<clip>.csv` plus `data/events/<sport>/<match_id>.jsonl`.
- Event files exist for football (3869685) and F1 (2019-20), so those two feeds are covered.
- NBA's `0041500407.jsonl` is missing. The raw JSON for it is in the data package (`raw/nba/`).
- Weightlifting has `sync: null`, and no code reads `data/weightlifting/*.csv` at serve time, so that feed gets no match data today.
- `docker/compose.yaml` has an `engine` service with `build:` and no `image:`. Its mounts `../models` and `../data/events` don't exist in a release directory. `compose.demo.yaml` is missing.
- Nothing listens on 8001.
- `GCS_BUCKET` is read nowhere. Data paths are config strings resolved against `REPO_ROOT` (`config.py:248`). `scripts/sync_bucket.sh` doesn't exist.
- Splits: `training/split.py` puts demo matches in test. `assert_no_demo_leak` runs on write and on load (`split.py:56-83`).
  - The split is used in `training/model.py:74` (train/val) and `training/evaluate.py:166` (test).
  - Tests that pin it: `test_pipeline.py:202` and `test_feeds.py:27`.
  - Config keys: `training.split.{val_frac,test_frac}`.
  - `imminent_eval.py:278` keeps demo clips out of `SET=tuning`. That is imminent tuning, not training, and stays as it is.
- No AMQP library is in `pyproject.toml` or `uv.lock`.
- The unmerged `feat/training-t2-f1` branch (F1 label change) is untouched by this work.

**Front end.**
- The backend (`apps/backend/src/amqp-consumer.ts`, amqplib) declares its own topology:
  - topic exchange `sportlight.feeds` (durable)
  - fanout DLX `sportlight.feeds.dlx`
  - DLQ `sportlight.feeds.dead`
  - queue `sportlight.backend` (x-max-length 100000), bound to `feed.#`
- Message handling: JSON body, checked with zod (`packages/shared/src/schemas.ts`). Invalid messages are nacked to the DLQ. Headers and the routing key are ignored.
- `/readyz` is 200 once the topology is declared and consuming has started (`apps/backend/src/app.ts:210`).
- Feeds exist only after a `feed.started` message.
- Timeouts: intensity goes stale after 5 s without an update, a feed is `unavailable` after 15 s of silence, and it is `ended` after 120 s.
- After a backend restart its state rebuilds only from re-announced `feed.started` messages.
- The mock publisher re-announces every 10 s and sends intensity at 1 Hz. It already has an engine bridge (`apps/publisher/src/engine-scores.ts`, `feed-clock.ts`) that defines the clock alignment we copy.
- `media` is plain nginx serving MP4 with byte ranges. The browser loops the clip itself at `mod(mediaOffsetSec + (serverNow − runStartedAt), durationSec)` (`packages/shared/src/timeline.ts:16`).
- Engine feed ids and files are in `media/local-sources.json`, served at `/media/local/videos/<file>`.
- Sports are a fixed list (`catalogue.ts:4`): `f1` is `motorsport`, `nba` is `basketball`.

**GCP.**
- VMs: `si-mq` (10.164.0.2, tag `si-mq`) and `si-web` (tag `si`, SA `si-vm-sa`). Both are RUNNING, no GPU.
- Artifact Registry repo `si`: front-end images only.
- Bucket `sportlight-514020-si-private`:
  - europe-west4, Standard, uniform access, public access prevention enforced, versioning off.
  - **No labels; empty.**
  - Object admin for `si-deployer` and **`si-vm-sa` (si-web's identity)**.
- Secrets: `rabbitmq-admin-password`, `rabbitmq-app-password`, `sportlight-web-env`.
- No Cloud Build history.
- The engine repo has no GitHub variables, secrets or environments.
- Blocked by the local gcloud permission deny list (Charles to check): firewall rules, WIF provider conditions, service-account list and IAM bindings, static addresses.

## 3. The ingest blocker and the proposed minimum

Milestone 2's done-when needs the engine to score the four clips on the VM at 1 Hz with non-null
`intensity_pct`. The plan's ingest stream (I1 to I6, P1 to P3) does that with NVDEC decode, audio
and perception. It is many hours of GPU work owned by the ingest agent, and none of it exists.

Proposal: build the **clock lane only** now, as `python -m intensity.ingest.run --clock-only`.
- It does what the plan's `start_feeds.py` + `clock.py` do, minus the RTSP decode:
  - Probe each clip's duration with PyAV.
  - Write `run/feeds_state.json` (`started_at`, `duration_s` per feed; kept across restarts so the timeline doesn't jump).
  - Send `ClockTick`s at 4 Hz and `Control(loop)` on wrap to the engine's `/ingest`.
  - Serve `/health` on 8001.
- **What the scores then contain.** Scores come from match data replayed in clip time (closeness, momentum, stakes, forecast, imminent where its cues are match-based). Crowd, commentary and video parts are `null`, and `scorer.combine` renormalises over the parts that are present.
- **Fill the two feeds with no match data:**
  - convert NBA 0041500407 from `raw/nba/` (data package);
  - add a small session-CSV event writer for weightlifting, so its demo session replays in video time.
- **GPU.** The ingest service runs with the NVIDIA runtime (`nvidia-smi` in its health check) so real perception drops in later, but the clock lane itself needs no GPU.

This is honest about what "scores the clip" means: the clip file is the timeline, not yet an input
signal. Building the full perception stack instead would be a separate, much larger task.

## 4. Message mapping: `ScoreUpdate` to the backend's `FeedEvent`

The engine publishes to exchange `sportlight.feeds` (asserted as topic and durable with the backend's exact arguments). It never declares queues.

Every message carries:
- **Routing key:** `feed.<type>.<sportId>.<feedId>`.
- **Properties:** `content_type=application/json`, `message_id=eventId`, `timestamp`. Lifecycle events are persistent; intensity is not.
- **Envelope fields:**
  - `schemaVersion: 1`
  - `eventId`: a UUID
  - `provider: "sportlight-engine"`
  - `feedId`: the engine's `feed_id`, unchanged
  - `runId`: `<feedId>.<base36 run-start ms>`
  - `runStartedAt`: the publisher's process start, so each restart gives a newer run, which the backend accepts
  - `seq`: increasing per run
  - `occurredAt`: when the message was emitted

| Front-end event | When | Payload, from what |
|---|---|---|
| `feed.started` | Startup, every reconnect, every 10 s (as the mock does) | `sportId`: `football`, `f1` becomes `motorsport`, `nba` becomes `basketball`, `weightlifting`. `event`: titles from a new `config/publish.yaml` (copied from `media/local-sources.json`). `source`: `{kind: "looping-mp4", url: "/media/local/videos/<file>", durationSec, mediaOffsetSec, hasAudio: true, credits, attributionNotice}` |
| `feed.intensity` | Each non-stale `ScoreUpdate` (1 Hz per feed, plus out-of-cycle imminent ones) | `score = round(intensity_pct)` clamped to 0 to 100. `streamTime = clockStart + (loop_epoch × durationSec + t_video) − runStartedAt` in seconds; negative values are skipped |
| `feed.updated {availability}` | The feed goes stale for over 5 s, or recovers | `unavailable` / `available` |
| `feed.updated {source}` | The clock start moves (the ingest restarted without its state file) | New `mediaOffsetSec` |
| `feed.ended {completed}` | A feed is removed from config | Not sent on an ordinary restart, so the UI keeps the feed while the engine comes back |

**Clock alignment.**
- `clockStart` and `durationSec` come from `run/feeds_state.json` (the same file and keys the front end's `feed-clock.ts` already parses).
- The fallback, used if the file is missing, is `now − (loop_epoch × dur + t_video)`.
- `mediaOffsetSec = mod(runStartedAt − clockStart, dur)`. With that offset, the browser's position equals the engine's `t_video`.
- Both VMs use Google's NTP; the browser uses the backend's clock via ping/pong.

**Dropped fields.** `forecast`, `heating_up`, `trend`, `parts` and `reasons` have nowhere to go: `IntensityPayload` is `{score, streamTime}`. Showing heating-up in the UI would be a front-end schema change, which is not proposed here.

**Publisher behaviour.**
- aio-pika robust connection, with backoff of 0.5 to 30 s, and publisher confirms.
- A latest-wins slot per feed for intensity, plus a bounded (1000) queue for lifecycle events.
- Publishing runs as a background task fed from the scoring loop through a non-blocking hand-off, so scoring never waits on the broker.
- `/ws/scores` is unchanged.
- The AMQP connection is configured by env (`AMQP_URL`) and is off when unset (local dev, tests).
- Its status is added to `/health` as a part, without changing the 200.

## 5. Implementation order (one PR each, CI green before asking to merge)

1. **PR 1: data layer and bucket (Milestone 1).**
   - `intensity.data`: one resolver. `INTENSITY_DATA_DIR` defaults to `<repo>/data`, and every `data/...` config path goes through it.
   - `scripts/sync_bucket.sh push|pull <folder…> [--delete] [--dry-run]` using `gcloud storage rsync -r`, never deleting unless `--delete`.
   - `make data-pull` / `make data-push`.
   - `infra/gcp/bucket.sh`: idempotent labels plus the IAM bindings in section 7.
   - `scripts/verify_bucket.py` (`make data-verify`): file counts and sizes per folder against `bucket-plan.md`, the two 2018 files absent from `raw/football/`, anonymous GET returns 403.
   - `.env` gets `GCS_BUCKET=sportlight-514020-si-private`.
   - `.gcloudignore` keeps `data/` (except `data/sync`, `data/weightlifting`) out of Cloud Build uploads.
2. **PR 2: training (Milestone 2).**
   - `training.split.mode: in_sample` (default) | `holdout`. This is one switch.
   - `in_sample` puts every match in train and evaluates on the same rows.
   - Without a validation set, `model.py` trains for a fixed number of rounds (`training.model.num_boost_round`).
   - `assert_no_demo_leak` runs only in `holdout`.
   - The demo-leak tests become mode-aware: `holdout` still asserts no demo match in training, `in_sample` asserts demo matches *are* included.
   - Every report and `report.json` carries `"metrics": "in-sample"` and a banner.
   - New serve-parity test: the live `Engine` and offline featurize-and-predict give the same forecast within 1e-6.
   - `make train-all`: data-pull of `raw/`, `events/`, `tables/`, then `build-<sport>` for football, f1, nba, then export to `models/`. It refuses NBA fetching on GCE (metadata-server check).
   - Video head: no code exists, so `train-all` skips it with a message.
3. **PR 3: clock-lane ingest and demo data (Milestone 2).**
   - Everything in section 3.
   - NBA demo event conversion and the weightlifting session events.
   - `docker/compose.yaml` becomes image-based (`${IMAGE_SI}`). A `local` profile keeps `build:` for laptops.
   - New `docker/compose.gcp.yaml`: GPU device reservation for `ingest`, `/opt/si/data` mounted at `/app/data`, the env file from Secret Manager, health checks.
4. **PR 4: build and deploy (Milestone 2).**
   - Commit the handoff workflows, fixed:
     - `master` triggers;
     - real compose files (`docker/compose.yaml:docker/compose.gcp.yaml`);
     - health URLs (engine `127.0.0.1:8000/health`, ingest `127.0.0.1:8001/health`, both must return `status != error`);
     - `GCS_BUCKET` and `AMQP_HOST` in `extra_env`.
   - `infra/gcp/cloudbuild.yaml` (layer cache from the last `master` image, 3600 s timeout, Cloud Logging only).
   - `deploy/remote-up.sh` reads `rabbitmq-app-password` on the VM into `/opt/si/secrets/amqp.env` (0600, never echoed, never in `release.env`).
   - `infra/gcp/create_gpu_vm.sh`, `setup_vm.sh` (driver, Docker, NVIDIA Container Toolkit, bucket pull of `samples/`, `events/`, `calibration/`, `transcripts/`, `video_head/`), `run_demo.sh`, and an idle watchdog unit.
   - Rollback proof: a throwaway commit whose image exits at start, deployed to `si-dev`.
5. **PR 5: AMQP publisher (Milestone 3).**
   - Section 4, with unit tests against a fake channel.
   - Contract tests against JSON fixtures copied from the front end's zod schemas.
   - CI job `e2e`:
     - `rabbitmq:4.3.7-management` service container.
     - The front-end backend checked out and run from source.
     - The engine and clock lane on a synthetic clip generated in CI.
     - Assert `/readyz` and `/api/feeds` show live intensity, then restart the engine and assert recovery within 10 s.
6. **PR 6 (front-end repo, separate and small): production wiring.**
   - `publisher` gets `profiles: [mock]` and is dropped from `remote-up.sh`. Local dev uses `--profile mock`.
   - The media service serves the four demo clips from `/opt/sportlight-web/footage`, synced from `gs://…/samples/` at deploy, mounted read-only at `/media/local/videos/`, behind whatever D5 decides.

Then: deploy to `si-dev`, switch production, and prove each done-when item.

## 6. Decisions needed from Charles

**Answered 8 Nov 2026:**
- D1: clock lane now.
- D2 and D4: aio-pika, `rabbitmq:4.3.7-management` and Node 22 in CI, all approved.
- D3: `si-dev`, `g2-standard-8`, approved to write and run.
- D5: accept public playback for the demo window (no allowlist).

**Still open:** D6 (PR 6, implied by D5 but to be confirmed on the PR), D7 (emails), D8 (left unchanged until Charles asks).

- **D1 Ingest scope:** clock lane now (section 3), or wait for or build the full ingest and perception stream?
- **D2 Dependency:** add `aio-pika` (9.x, pure Python, pulls `aiormq` and `pamqp`) to base deps? The alternative, `pika`, is blocking and would need a thread.
- **D3 GPU VM:**
  - Proposed: one `si-dev`, `g2-standard-8` + 1 L4, standard provisioning (Spot looks refused: `PREEMPTIBLE_CPUS` = 0), 100 GB pd-balanced, tag `si`, SA `si-gpu-sa` (new), labels `app=sports-intensity,owner=charles,role=gpu`.
  - Limits: `--max-run-duration=12h --instance-termination-action=STOP`, and a watchdog that stops it after 60 min with no running stack and no SSH session.
  - Cost: roughly a dollar an hour while running. The plan's demo size is `g2-standard-16`.
  - The 12 h cap also stops the live site's scores, so production may want a longer cap on `si-demo` later.
- **D4 Base images:** `rabbitmq:4.3.7-management` (CI only, matching si-mq) and Node 22 in CI for the front-end backend. Cross-repo CI also needs a read-only fine-grained token for `Asyncharles/sportlight` stored as secret `SPORTLIGHT_REPO_TOKEN`.
- **D5 Footage on the public site:** the browser must download the clip to play it, which conflicts with "footage never goes anywhere public". Options:
  - (a) Caddy allows `/media/local/*` only from the team and venue CIDRs.
  - (b) Basic auth on `/media/local/*`.
  - (c) Accept public playback for the demo window.

  Recommended: (a), since scores and the rest of the site stay public.
- **D6 Front-end PR 6:** approve that change to `Asyncharles/sportlight`.
- **D7 Team emails:** for bucket read/write.
- **D8 si-web's bucket access:** `si-vm-sa` has object admin on the whole bucket today. Narrow it to read-only on `samples/` (an IAM condition), as the brief asks? Nothing on si-web writes to the bucket now.

## 7. Commands only Charles can run (also in `infra/gcp/PREREQS.md`)

Local agent settings deny `gcloud iam`, `gcloud projects`, firewall reads and writes; these need Charles.

```bash
P=sportlight-514020; PN=251387512246; B=sportlight-514020-si-private
CI=si-web-ci@$P.iam.gserviceaccount.com
# 1. Check the WIF provider lets the engine repo in, then bind it to si-web-ci
gcloud iam workload-identity-pools providers describe github-sportlight --location=global \
  --workload-identity-pool=github --format='value(attributeCondition)'
gcloud iam service-accounts add-iam-policy-binding $CI --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/$PN/locations/global/workloadIdentityPools/github/attribute.repository/Asyncharles/SportlightEngine"
# 2. Let si-web-ci run Cloud Build as si-build (the engine image is built there)
gcloud projects add-iam-policy-binding $P --member=serviceAccount:$CI --role=roles/cloudbuild.builds.editor --condition=None
gcloud iam service-accounts add-iam-policy-binding si-build@$P.iam.gserviceaccount.com --member=serviceAccount:$CI --role=roles/iam.serviceAccountUser
gcloud storage buckets add-iam-policy-binding gs://${P}_cloudbuild --member=serviceAccount:$CI --role=roles/storage.objectAdmin
# 3. GPU VM identity (D3)
gcloud iam service-accounts create si-gpu-sa --display-name="Sportlight GPU VM"
G=si-gpu-sa@$P.iam.gserviceaccount.com; sleep 30
for r in artifactregistry.reader logging.logWriter monitoring.metricWriter; do
  gcloud projects add-iam-policy-binding $P --member=serviceAccount:$G --role=roles/$r --condition=None; done
gcloud secrets add-iam-policy-binding rabbitmq-app-password --member=serviceAccount:$G --role=roles/secretmanager.secretAccessor
for who in $CI si-deployer@$P.iam.gserviceaccount.com; do   # OS Login SSH into a VM that runs as si-gpu-sa
  gcloud iam service-accounts add-iam-policy-binding $G --member=serviceAccount:$who --role=roles/iam.serviceAccountUser; done
# 4. Confirm firewall: si-mq-internal (tcp:5672 from tag si) and si-iap-ssh (tcp:22 from IAP to tag si)
gcloud compute firewall-rules list --format="table(name,sourceRanges.list(),sourceTags.list(),targetTags.list(),allowed[].map().firewall_rule().list())"
```

The agent then runs the bucket bindings itself (`infra/gcp/bucket.sh`, within the brief's bucket access):
- the team's accounts get `objectAdmin`;
- `si-gpu-sa` gets `objectViewer`, plus `objectCreator` conditioned on `reports/`, `tapes/`, `events/`;
- `si-vm-sa` is narrowed per D8.

GitHub (agent, with `gh`, after go-ahead):
- engine repo variables `GCP_PROJECT`, `GCP_REGION`, `GCP_ZONE`, `GCP_WIF_PROVIDER` (`…/providers/github-sportlight`), `GCP_CI_SA` (`si-web-ci`), `GCP_BUILD_SA`, `AR_REPO`;
- environments `dev` and `demo` (demo: Charles as required reviewer, `master` only).

Still Charles's step: upload `sportlight-data.zip` to the bucket (not `raw/football/` until the ML lead approves). The agent then runs `verify_bucket.sh`.
