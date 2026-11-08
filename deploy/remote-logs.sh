#!/usr/bin/env bash
# Printed once by deploy-reusable.yml when a release fails. Logs never contain secret values.
set -a; . ./release.env; set +a
export COMPOSE_PROJECT_NAME=si
docker compose ps
docker compose logs --tail 80 --no-color engine ingest
