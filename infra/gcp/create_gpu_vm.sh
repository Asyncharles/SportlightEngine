#!/usr/bin/env bash
# Create the one GPU VM (docs/deploy-wiring-plan.md, D3). Charles approves every run.
#
#   DRY_RUN=1 infra/gcp/create_gpu_vm.sh            print the command only
#   infra/gcp/create_gpu_vm.sh                       si-dev, g2-standard-8, 1 L4
#   VM=si-demo MAX_RUN=24h infra/gcp/create_gpu_vm.sh
#   ZONE=europe-west4-b infra/gcp/create_gpu_vm.sh   L4 stockout in the default zone
#
# Standard provisioning (never Spot: the demo must not be preempted, and PREEMPTIBLE_CPUS is 0 in
# europe-west4). A max run duration stops the VM even if the idle watchdog (setup_vm.sh) misses.
# Tag `si` lets it reach si-mq:5672 (firewall si-mq-internal) and IAP SSH (si-iap-ssh).
# Needs infra/gcp/PREREQS.md step 3 (service account si-gpu-sa) first.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CALLER_ZONE=${ZONE:-}   # a zone on the command line beats infra/gcp/env (e.g. ZONE=$ZONE_FALLBACK on a stockout)
# shellcheck source=/dev/null
[[ -f "$HERE/env" ]] && source "$HERE/env"
PROJECT=${PROJECT:?set PROJECT or create infra/gcp/env}
ZONE=${CALLER_ZONE:-${ZONE:-europe-west4-a}}
VM=${VM:-si-dev}
MACHINE=${MACHINE:-g2-standard-8}
DISK_GB=${DISK_GB:-100}
MAX_RUN=${MAX_RUN:-12h}
GPU_VM_SA=${GPU_VM_SA:-si-gpu-sa@$PROJECT.iam.gserviceaccount.com}
ROLE_ENV=${VM#si-}

case $VM in si-dev|si-demo) ;; *) echo "VM must be si-dev or si-demo" >&2; exit 2 ;; esac

# One GPU in the whole project (GPUS_ALL_REGIONS=1): refuse while another GPU VM exists.
others=$(gcloud compute instances list --project="$PROJECT" \
  --format='value(name,status)' --filter="guestAccelerators:* AND name!=$VM" 2>/dev/null || true)
if [[ -n "$others" ]]; then
  echo "Another GPU VM exists (quota is one GPU):"; echo "$others"; exit 1
fi
if gcloud compute instances describe "$VM" --zone="$ZONE" --project="$PROJECT" >/dev/null 2>&1; then
  echo "$VM already exists in $ZONE; nothing to do."; exit 0
fi

cmd=(gcloud compute instances create "$VM" --project="$PROJECT" --zone="$ZONE"
  --machine-type="$MACHINE" --accelerator=type=nvidia-l4,count=1
  --maintenance-policy=TERMINATE --provisioning-model=STANDARD
  --max-run-duration="$MAX_RUN" --instance-termination-action=STOP
  --image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud
  --boot-disk-size="${DISK_GB}GB" --boot-disk-type=pd-balanced
  --no-shielded-secure-boot
  --network=default --tags=si
  --service-account="$GPU_VM_SA" --scopes=cloud-platform
  --labels="app=sports-intensity,owner=charles,role=gpu,env=$ROLE_ENV"
  --metadata=enable-oslogin=TRUE)

if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '+'; printf ' %q' "${cmd[@]}"; echo; exit 0; fi
"${cmd[@]}"
echo "Created $VM. Next: gcloud compute scp infra/gcp/setup_vm.sh $VM:/tmp/ --zone=$ZONE --tunnel-through-iap"
echo "      then: gcloud compute ssh $VM --zone=$ZONE --tunnel-through-iap --command='sudo ROLE=$ROLE_ENV bash /tmp/setup_vm.sh'"
