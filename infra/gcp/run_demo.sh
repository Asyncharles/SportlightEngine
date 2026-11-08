#!/usr/bin/env bash
# Start (or restart onto a new image) the stack on the GPU VM; run from a release dir by deploy/remote-up.sh.
# The VM itself is never started or stopped here.
set -euo pipefail
export COMPOSE_PROJECT_NAME=${COMPOSE_PROJECT_NAME:-si}
# FORCE_RECREATE=1 (deploy dev's `restart` input) restarts the stack even when nothing changed.
docker compose up -d --no-build --remove-orphans ${FORCE_RECREATE:+--force-recreate} engine ingest
docker compose ps
# GPU check on GPU VMs only (informational: the clock lane runs without it, perception will not).
if [[ "${COMPOSE_FILE:-}" == *compose.gpu.yaml* ]]; then
  docker compose exec -T ingest nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null \
    || echo "WARNING: the ingest container sees no GPU" >&2
fi
