#!/usr/bin/env bash
# Settings, labels and access for the private data bucket (docs/deploy-wiring-plan.md, Milestone 1).
# The bucket already exists (infra/gcp/env BUCKET); this never creates or deletes it. Idempotent.
#
#   DRY_RUN=1 infra/gcp/bucket.sh            print the commands only
#   TEAM_MEMBERS=a@x.com,b@y.com infra/gcp/bucket.sh
#
# Access:
#   TEAM_MEMBERS      read and write everything (roles/storage.objectAdmin)
#   GPU_VM_SA         read everything; write reports/, tapes/, events/ (objectUser, IAM condition)
# si-web's identity (si-vm-sa) keeps its current access until Charles decides (plan D8).
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
[[ -f "$HERE/env" ]] && source "$HERE/env"
PROJECT=${PROJECT:?set PROJECT or create infra/gcp/env}
BUCKET=${BUCKET:?set BUCKET or create infra/gcp/env}
GPU_VM_SA=${GPU_VM_SA:-si-gpu-sa@$PROJECT.iam.gserviceaccount.com}
TEAM_MEMBERS=${TEAM_MEMBERS:-}
WRITE_PREFIXES=(reports tapes events)   # keep in sync with intensity.data.VM_WRITE_FOLDERS

run() {
  if [[ "${DRY_RUN:-0}" == 1 ]]; then { printf '+'; printf ' %q' "$@"; echo; } >&2; else "$@"; fi
}
failed=0
bind() {  # bind <label> <gcloud args...>: add one binding, report, keep going on failure
  local label=$1; shift
  if run gcloud storage buckets add-iam-policy-binding "$B" --project="$PROJECT" "$@" >/dev/null; then
    echo "ok $label"
  else
    echo "FAILED $label (does the principal exist? see infra/gcp/PREREQS.md)" >&2; failed=1
  fi
}

B=gs://$BUCKET
echo "== $B settings and labels"
run gcloud storage buckets update "$B" --project="$PROJECT" \
  --uniform-bucket-level-access --public-access-prevention --no-versioning \
  --update-labels=app=sports-intensity,owner=charles

echo "== team read/write"
if [[ -z "$TEAM_MEMBERS" ]]; then
  echo "TEAM_MEMBERS not set; skipping (ask Charles for the team's Google accounts)"
else
  IFS=, read -r -a members <<<"$TEAM_MEMBERS"
  for m in "${members[@]}"; do
    m=$(echo "$m" | tr -d ' ')
    [[ -n "$m" ]] || continue
    bind "user:$m objectAdmin" --member="user:$m" --role=roles/storage.objectAdmin
  done
fi

echo "== GPU VM identity ($GPU_VM_SA)"
bind "objectViewer" --member="serviceAccount:$GPU_VM_SA" --role=roles/storage.objectViewer
expr=""
for p in "${WRITE_PREFIXES[@]}"; do
  expr+="${expr:+ || }resource.name.startsWith(\"projects/_/buckets/$BUCKET/objects/$p/\")"
done
bind "objectUser on ${WRITE_PREFIXES[*]}" --member="serviceAccount:$GPU_VM_SA" --role=roles/storage.objectUser \
  --condition="title=vm-writes,description=GPU VM writes ${WRITE_PREFIXES[*]},expression=$expr"

echo "== resulting policy"
run gcloud storage buckets get-iam-policy "$B" --project="$PROJECT" \
  --flatten=bindings --format="table(bindings.role,bindings.members.list(),bindings.condition.title)"
exit $failed
