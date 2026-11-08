#!/usr/bin/env bash
# Exit 0 when the stack is healthy (deploy-reusable.yml retries until its timeout, then rolls back).
# The engine's /health is always HTTP 200 with status ok|degraded|error in the body, so read the body;
# the ingest's /health is 200 only when its clocks run and the engine link is up.
set -euo pipefail
set -a; . ./release.env; set +a
curl -fsS --max-time 5 "$ENGINE_HEALTH_URL" \
  | python3 -c 'import json, sys; s = json.load(sys.stdin).get("status"); sys.exit(0 if s in ("ok", "degraded") else 1)'
curl -fsS --max-time 5 "$INGEST_HEALTH_URL" >/dev/null
