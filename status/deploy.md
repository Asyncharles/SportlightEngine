# Deploy agent status

Updated 2026-11-08. Plan and decisions: `docs/deploy-wiring-plan.md`. Human-only commands: `infra/gcp/PREREQS.md`. Runbook: `infra/gcp/RUNBOOK.md`.

## Pull requests (stacked: merge in order; each later PR retargets to `master` as the one before merges)

| PR | Branch | What |
| --- | --- | --- |
| #5 | `feat/bucket-data-layer` | M1: data layer, `sync_bucket.sh`, `data-pull`/`push`/`verify`, `bucket.sh`, `create_gpu_vm.sh`, `setup_vm.sh`, PREREQS |
| #6 | `feat/training-in-sample` | M2: `training.split.mode: in_sample`, serve-parity test (and the replay fix it found), `make train-all`, football bundle |
| #7 | `feat/clock-lane` | M2: clock-lane ingest, weightlifting session replay, image-based Compose with the GPU overlay |
| #8 | `feat/deploy-workflows` | M2: Cloud Build image, `deploy-dev`/`deploy-demo`, `remote-*.sh`, secrets from Secret Manager on the VM |
| #9 | `feat/amqp-publisher` | M3: FeedEvents to RabbitMQ (aio-pika), contract tests, CI e2e with the real front-end backend |
| sportlight #4 | `deploy/engine-feeds-prod` | Front end: serve the demo clips from the bucket, mock publisher off in production (approved by Charles 2026-11-08) |

## Milestones

| Milestone | State | Evidence (command) |
| --- | --- | --- |
| M1 storage | Bucket labelled and checked. Laptop push/pull and anonymous 403 proven. `events/` and `raw/` (2022 only) pushed. Waiting on the data upload, team emails, and si-gpu-sa for the VM pull | `infra/gcp/bucket.sh`; `make data-push FOLDERS=reports/sync-test`; `make data-pull …`; `curl` → 403 |
| M2 engine on the GPU | Code, image build and workflows done; first Cloud Build ran. Blocked on PREREQS 1 to 3 (WIF, Cloud Build roles, si-gpu-sa), then VM creation | `make test`; slow parity on `fb-arg-fra`; `tests/integration/test_clock_lane_engine.py` |
| M3 RabbitMQ to the front end | Engine side done and tested against the zod contract. The e2e job skips until `SPORTLIGHT_REPO_TOKEN` exists. Front-end PR prepared locally | `tests/contract/test_feed_events.py`; `.github/workflows/e2e.yml` |

## Decisions (Charles, 2026-11-08)

- Bucket: the existing `gs://sportlight-514020-si-private`.
- CI identity: `si-web-ci` (there is no `si-ci`).
- Data package: Charles uploads it himself.
- Ingest: the clock lane now (plan section 3).
- Dependencies: aio-pika, plus `rabbitmq:4.3.7-management` and Node in CI. Node 20 matches the front end's Dockerfiles; 22 was approved.
- GPU VM: `si-dev`, g2-standard-8, approved.
- Footage on sportlight.uk: public playback accepted for the demo window.
- 2018 StatsBomb files (`raw/football/7560.json`, `7567.json`): ML lead approved; kept in the bucket (Charles, 2026-11-08). Pushing them needs `ALLOW_RAW_FOOTBALL_2018=1`; verify with `--allow-raw-football`.
- Football forecast: keep the heuristic. `models/football/CURRENT` stays unset.
- Front-end PR: approved and opened as sportlight #4.

## Asks

- **Charles:** run `infra/gcp/PREREQS.md` steps 1, 2 and 4 (step 3, si-gpu-sa, is done; `bucket.sh` gave it read access, plus writes to `reports/`, `tapes/`, `events/`), and send the team's emails.
- **Charles:** create `SPORTLIGHT_REPO_TOKEN` (step 5), and say whether the agent may set the repo variables and environments.
- Data package: uploaded and verified 2026-11-08 (17 clips, 592 stills, five NBA games, 2018 files approved).
- **ML lead:** note that `in_sample` suspends hard rule 6.
- **Training agent:** review the minimal weightlifting session converter (T4). F1 (T2, on `feat/training-t2-f1`) and NBA (T3) labels are needed before `train-all` can build them.
