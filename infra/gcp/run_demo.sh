#!/usr/bin/env bash
# Start (or restart onto a new image) the stack on the GPU VM; run from a release dir by deploy/remote-up.sh.
# The VM itself is never started or stopped here.
set -euo pipefail
export COMPOSE_PROJECT_NAME=${COMPOSE_PROJECT_NAME:-si}
docker compose up -d --no-build --remove-orphans engine ingest
docker compose ps
# GPU check (informational: the clock lane runs without it, perception will not).
docker compose exec -T ingest nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null \
  || echo "WARNING: the ingest container sees no GPU" >&2
