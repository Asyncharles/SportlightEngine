# Deploy agent status

Updated 2026-11-08. Plan and decisions: `docs/deploy-wiring-plan.md`. Human-only commands: `infra/gcp/PREREQS.md`.

## Milestones

| Milestone | State | Evidence (command) |
| --- | --- | --- |
| M1 storage | Bucket labelled and checked; sync, verify, VM scripts written. Waiting on the data upload, team emails and si-gpu-sa (for the VM pull) | `infra/gcp/bucket.sh`; `make data-push FOLDERS=reports/sync-test` then `make data-pull …`; anonymous GET 403 |
| M2 engine on the GPU | Not started | |
| M3 RabbitMQ to the front end | Not started | |

## Decisions (Charles, 2026-11-08)

- Bucket: the existing `gs://sportlight-514020-si-private`.
- CI identity: `si-web-ci` (there is no `si-ci`).
- Data package: Charles uploads it himself.
- Ingest: the clock lane now (plan section 3).
- Dependencies: aio-pika, plus `rabbitmq:4.3.7-management` and Node 22 in CI.
- GPU VM: `si-dev`, g2-standard-8, approved.
- Footage on sportlight.uk: public playback accepted for the demo window.

## Asks

- **Charles:** run `infra/gcp/PREREQS.md` steps 1 to 4, and send the team's emails.
- **Charles:** upload the data package (without `raw/football/`), then `make data-verify PACKAGE=…`.
- **ML lead:** approve `raw/football/` (2018 StatsBomb) before it is pushed.
