# GPU VM runbook

Context: `docs/deploy-wiring-plan.md`. One GPU in the project, so only one of `si-dev` / `si-demo` exists at a time. Workflows never create, start or stop a VM.

## Create and set up (once, Charles approves)

**MVP: `si-dev` is a CPU VM** (`e2-standard-4`, Charles's decision of 2026-11-08). L4s were out of stock in every approved zone, and the clock lane needs no GPU. Create it with `GPU=0 infra/gcp/create_gpu_vm.sh`, set it up with `sudo ROLE=dev GPU=0 bash /tmp/setup_vm.sh`, and `deploy dev` leaves out `compose.gpu.yaml`. The GPU steps below are for the L4 VM, when perception needs it.

Needs `PREREQS.md` steps 1 to 4.

```bash
DRY_RUN=1 infra/gcp/create_gpu_vm.sh      # review
infra/gcp/create_gpu_vm.sh                # si-dev, g2-standard-8, 1 L4, 12 h max run
# L4s run out (STOCKOUT / ZONE_RESOURCE_POOL_EXHAUSTED; on 2026-11-08 in all three zones): try the
# other zones with ZONE=europe-west4-b or -c, then set the engine repo's GCP_ZONE variable to the
# zone the VM landed in, because the deploy workflows look it up there.
gcloud compute scp infra/gcp/setup_vm.sh si-dev:/tmp/ --zone=europe-west4-a --tunnel-through-iap
gcloud compute ssh si-dev --zone=europe-west4-a --tunnel-through-iap --command='sudo ROLE=dev bash /tmp/setup_vm.sh'
# The driver install reboots the VM once: run the ssh line again after it comes back.
infra/gcp/bucket.sh                       # si-gpu-sa's bucket access, once the SA exists
```

## Start, stop

```bash
gcloud compute instances start si-dev --zone=europe-west4-a      # then re-run deploy dev
gcloud compute instances stop si-dev --zone=europe-west4-a
```

The idle watchdog stops the VM after 60 min with no `si` containers running and nobody logged in. The 12 h max run duration stops it regardless.

## Build and deploy

- A merge to `master` that touches `src/`, `docker/`, `config/`, `models/` and similar runs `image.yml`. That builds `$AR_REPO/si:<sha>` on Cloud Build.
- Then run **Actions → deploy dev → Run workflow** with `image_tag=<full sha>`, or with no tag to deploy the commit the run is on.
- **deploy demo** waits for Charles's approval in the `demo` environment.
- On the VM a release lives in `/opt/si/releases/<sha>`, and `/opt/si/current` points at the live one. Health is checked for up to 5 minutes:
  - engine `/health` status `ok` or `degraded`;
  - ingest `/health` returns 200.
- A failing release is replaced by the previous one automatically.

## Prove the automatic rollback

```bash
git checkout -b test/broken-release master
# Make the engine exit at start in the release's compose file (the image itself is unchanged):
sed -i '' 's/"--overlay", "demo"\]/"--overlay", "no-such-overlay"]/' docker/compose.yaml
git commit -am "Deliberately broken release (rollback test; never merge)" && git push -u origin test/broken-release
gh workflow run image.yml --ref test/broken-release        # builds si:<that sha> (cached layers: minutes)
gh workflow run deploy-dev.yml --ref master -f image_tag=$(git rev-parse HEAD)
```

The deploy fails after the health timeout, and the job log shows `Rolled back to <previous sha>`. Delete the branch afterwards.

## Look at the stack

```bash
gcloud compute ssh si-dev --zone=europe-west4-a --tunnel-through-iap --command='cd /opt/si/current && sudo bash deploy/remote-logs.sh'
gcloud compute ssh si-dev --zone=europe-west4-a --tunnel-through-iap --command='curl -s 127.0.0.1:8000/health; curl -s 127.0.0.1:8001/health'
gcloud compute ssh si-dev --zone=europe-west4-a --tunnel-through-iap -- -L 8000:127.0.0.1:8000   # then http://localhost:8000/debug
```

## Data

The VM pulls `samples/` (on the demo VM, only the four demo clips), `events/`, `calibration/`, `transcripts/` and `video_head/` into `/opt/si/data` at setup. To refresh, run `setup_vm.sh` again, or:

```bash
sudo gcloud storage rsync --recursive gs://sportlight-514020-si-private/events /opt/si/data/events
```
