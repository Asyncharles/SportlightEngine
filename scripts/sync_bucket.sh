#!/usr/bin/env bash
# Sync data/ with the private bucket. The bucket mirrors data/ without the prefix:
# gs://$GCS_BUCKET/samples/x.mp4 is data/samples/x.mp4 (see src/intensity/data.py).
#
#   scripts/sync_bucket.sh pull [--delete] [--dry-run] [folder ...]
#   scripts/sync_bucket.sh push [--delete] [--dry-run] [folder ...]
#
# Folders are bucket folders or sub-folders (samples, samples/tuning/f1, events/nba, ...);
# none means every known folder. Nothing is ever deleted at the destination unless --delete.
# raw/football/ (2018 StatsBomb files) is never pushed unless ALLOW_RAW_FOOTBALL=1 (ML lead approval).
# GCS_BUCKET comes from the environment or, failing that, from .env. Local root: INTENSITY_DATA_DIR or data/.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
KNOWN=(samples stills raw events tables video_head transcripts tapes calibration reports sync weightlifting highlights)

usage() { sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-2}"; }

[[ $# -ge 1 ]] || usage
case $1 in push|pull) DIR=$1 ;; -h|--help) usage 0 ;; *) usage ;; esac
shift

DELETE=() DRY=() FOLDERS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --delete) DELETE=(--delete-unmatched-destination-objects) ;;
    --dry-run) DRY=(--dry-run) ;;
    -h|--help) usage 0 ;;
    -*) echo "unknown option $1" >&2; usage ;;
    *) FOLDERS+=("${1%/}") ;;
  esac
  shift
done
[[ ${#FOLDERS[@]} -gt 0 ]] || FOLDERS=("${KNOWN[@]}")

if [[ -z "${GCS_BUCKET:-}" && -f "$ROOT/.env" ]]; then
  # Read only this key; .env may hold secrets that must not be exported here.
  GCS_BUCKET=$(sed -n 's/^GCS_BUCKET=//p' "$ROOT/.env" | tail -n 1 | tr -d '"'"'"' \r')
fi
BUCKET=${GCS_BUCKET:-}
BUCKET=${BUCKET#gs://}; BUCKET=${BUCKET%/}
[[ -n "$BUCKET" ]] || { echo "GCS_BUCKET is not set (environment or .env)" >&2; exit 2; }
LOCAL=${INTENSITY_DATA_DIR:-$ROOT/data}

command -v gcloud >/dev/null || { echo "gcloud not found" >&2; exit 2; }

for f in "${FOLDERS[@]}"; do
  top=${f%%/*}
  if [[ ! " ${KNOWN[*]} " == *" $top "* || "$f" == *..* || "$f" == /* ]]; then
    echo "unknown bucket folder: $f (known: ${KNOWN[*]})" >&2; exit 2
  fi
done

EXCLUDE='(^|/)\.DS_Store$|(^|/)\.gitkeep$'
synced=0
for f in "${FOLDERS[@]}"; do
  local_dir="$LOCAL/$f" remote="gs://$BUCKET/$f"
  excl=$EXCLUDE
  if [[ "$DIR" == push && "${ALLOW_RAW_FOOTBALL:-0}" != 1 ]]; then
    if [[ "$f" == raw/football || "$f" == raw/football/* ]]; then
      echo "skip $f: raw/football needs ML lead approval (ALLOW_RAW_FOOTBALL=1)" >&2; continue
    fi
    [[ "$f" == raw ]] && excl="$excl|^football/"
  fi

  if [[ "$DIR" == push ]]; then
    [[ -d "$local_dir" ]] || { echo "skip $f: no $local_dir"; continue; }
    src=$local_dir dst=$remote
  else
    if ! gcloud storage ls "$remote/" >/dev/null 2>&1; then echo "skip $f: nothing at $remote/"; continue; fi
    mkdir -p "$local_dir"
    src=$remote dst=$local_dir
  fi
  echo "$DIR $f: $src -> $dst${DELETE:+ (with --delete)}${DRY:+ (dry run)}"
  gcloud storage rsync --recursive --exclude="$excl" ${DELETE[@]+"${DELETE[@]}"} ${DRY[@]+"${DRY[@]}"} "$src" "$dst"
  synced=$((synced + 1))
done
echo "$DIR done: $synced folder(s)"
