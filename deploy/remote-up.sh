#!/usr/bin/env bash
# On the GPU VM, as root, from /opt/si/releases/<sha>. Called by .github/workflows/deploy-reusable.yml.
# Writes the engine's secrets from Secret Manager, pulls the pinned image and starts the stack.
set -euo pipefail
set -a; . ./release.env; set +a          # IMAGE_SI, GIT_SHA, COMPOSE_FILE, SI_*, GCS_BUCKET, AMQP_* (no secrets)
export COMPOSE_PROJECT_NAME=si            # fixed, so containers and volumes don't follow the release dir
PATH=$PATH:/snap/bin
export SI_SECRETS=${SI_SECRETS:-/opt/si/secrets}
mkdir -p "$SI_DATA" "$SI_RUN" "$SI_SECRETS"
chmod 700 "$SI_SECRETS"

# Secrets: read with the VM's own identity at deploy time, written 0600 for compose's env_file, never
# echoed. Without access the engine still starts; it just doesn't publish to RabbitMQ.
umask 077
env_file=$SI_SECRETS/engine.env
tmp=$(mktemp "$SI_SECRETS/.engine.env.XXXXXX")
if [[ -n "${AMQP_PASSWORD_SECRET:-}" ]] \
   && pw=$(gcloud secrets versions access latest --secret="$AMQP_PASSWORD_SECRET" 2>/dev/null); then
  enc=$(PW="$pw" python3 -c 'import os, urllib.parse; print(urllib.parse.quote(os.environ["PW"], safe=""))')
  printf 'AMQP_URL=amqp://%s:%s@%s:%s/%s\n' "$AMQP_USER" "$enc" "$AMQP_HOST" "${AMQP_PORT:-5672}" "${AMQP_VHOST:-sportlight}" > "$tmp"
  unset pw enc
  echo "engine secrets: AMQP_URL written from Secret Manager ($AMQP_PASSWORD_SECRET)"
else
  : > "$tmp"
  echo "WARNING: cannot read Secret Manager secret ${AMQP_PASSWORD_SECRET:-<unset>}; the engine will not publish to RabbitMQ" >&2
fi
mv -f "$tmp" "$env_file"
umask 022

gcloud auth configure-docker "${IMAGE_SI%%/*}" --quiet >/dev/null 2>&1
docker compose pull --quiet engine ingest
bash infra/gcp/run_demo.sh
