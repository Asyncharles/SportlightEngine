#!/usr/bin/env bash
# Train every sport's forecast bundle from the bucket data (make train-all).
#
#   scripts/train_all.sh [sport ...]        default: football f1 nba
#   PULL=0 scripts/train_all.sh             use data/ as it is, no bucket pull
#   PUSH=1 scripts/train_all.sh             also push events/, tables/ and reports/training/ back
#
# Per sport: convert raw/ to events/ (offline), then build --no-source (featurize, label, split, train,
# calibrate, evaluate, export to models/<sport>/<version>/). Nothing is fetched from the internet here:
# fetching is `make fetch-<sport>`, and NBA fetching must run on a laptop because stats.nba.com blocks
# cloud IPs, so this script refuses `FETCH=1` for nba on a GCE VM. With training.split.mode in_sample
# (the default) every reported metric is in-sample. CURRENT is never moved (the ML lead does that).
# Only sports with a training label (training.labels.LABEL_EVENTS) can build; the others are reported
# as "no label yet" (F1 is T2, NBA is T3, both the training stream's). Weightlifting has no learned model
# (its forecast is a rule), and there is no video-head training code yet, so neither is trained here.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
UV=${UV:-$(command -v uv 2>/dev/null || echo "$HOME/.local/bin/uv")}
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-venv}
PY=("$UV" run --inexact python)
SPORTS=("$@")
[[ ${#SPORTS[@]} -gt 0 ]] || SPORTS=(football f1 nba)

on_gce() { curl -fs -m 1 -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/ >/dev/null 2>&1; }

if [[ "${PULL:-1}" == 1 ]]; then
  bash scripts/sync_bucket.sh pull raw events tables highlights || { echo "bucket pull failed" >&2; exit 1; }
fi

TRAINABLE=" $("${PY[@]}" -c 'from intensity.training.labels import LABEL_EVENTS; print(" ".join(sorted(LABEL_EVENTS)))') "
result=()   # one line per sport, in SPORTS order (bash 3.2 on macOS: no associative arrays)
for sport in "${SPORTS[@]}"; do
  echo "== $sport"
  if [[ "$TRAINABLE" != *" $sport "* ]]; then
    echo "no training label for $sport yet (trainable:$TRAINABLE); skipped"
    result+=("$sport: no label yet, not trained"); continue
  fi
  if [[ "${FETCH:-0}" == 1 ]]; then
    if [[ $sport == nba ]] && on_gce; then
      echo "refusing to fetch nba on GCP: stats.nba.com blocks cloud IPs; fetch on a laptop" >&2
      result+=("$sport: refused"); continue
    fi
    "${PY[@]}" -m intensity.training.cli fetch --sport "$sport" || { result+=("$sport: fetch failed"); continue; }
  fi
  "${PY[@]}" -m intensity.training.cli convert --sport "$sport" || { result+=("$sport: convert failed"); continue; }
  "${PY[@]}" -m intensity.training.cli build --sport "$sport" --no-source
  case $? in
    0) result+=("$sport: exported $(cat "run/training/$sport/LATEST" 2>/dev/null)") ;;
    2) result+=("$sport: targets missed, not exported (see data/reports/training/$sport/)") ;;
    *) result+=("$sport: build failed") ;;
  esac
done

echo "video head: skipped (no training code yet; plan T5)"
if [[ "${PUSH:-0}" == 1 ]]; then
  bash scripts/sync_bucket.sh push events tables reports/training
fi

echo "== summary (metrics are in-sample when training.split.mode is in_sample)"
rc=0
for line in "${result[@]}"; do
  echo "$line"
  [[ $line == *": exported "* || $line == *"no label yet"* ]] || rc=1
done
exit $rc
