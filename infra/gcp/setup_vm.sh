#!/usr/bin/env bash
# One-time (idempotent) setup of the GPU VM, run on the VM as root:
#   sudo ROLE=dev|demo GCS_BUCKET=<bucket> bash setup_vm.sh
#
# 1. NVIDIA driver (Google's cuda_installer.pyz; it reboots once: run this script again after).
# 2. Docker, Compose v2 and the NVIDIA Container Toolkit (Compose's GPU service uses it).
# 3. Artifact Registry auth for Docker.
# 4. The data the stack reads, pulled from the bucket into /opt/si/data (mounted at /app/data):
#    samples/ (demo VM: the four demo clips only), events/, calibration/, transcripts/, video_head/.
# 5. The idle watchdog: stops the VM after IDLE_MIN minutes with no running stack and nobody logged in.
set -euo pipefail

ROLE=${ROLE:-dev}
GCS_BUCKET=${GCS_BUCKET:-sportlight-514020-si-private}
AR_HOST=${AR_HOST:-europe-west4-docker.pkg.dev}
IDLE_MIN=${IDLE_MIN:-60}
SI=/opt/si
DATA=$SI/data
export DEBIAN_FRONTEND=noninteractive
PATH=$PATH:/snap/bin

[[ $EUID -eq 0 ]] || { echo "run as root (sudo)" >&2; exit 1; }
case $ROLE in dev|demo) ;; *) echo "ROLE must be dev or demo" >&2; exit 2 ;; esac
mkdir -p "$SI/releases" "$DATA" "$SI/secrets" "$SI/run"
chmod 700 "$SI/secrets"

echo "== 1. NVIDIA driver"
if ! nvidia-smi >/dev/null 2>&1; then
  curl -fsSL https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz \
    -o /opt/cuda_installer.pyz
  echo "Installing the driver; the VM may reboot. Run setup_vm.sh again afterwards."
  python3 /opt/cuda_installer.pyz install_driver --installation-mode=repo --installation-branch=prod
  nvidia-smi >/dev/null 2>&1 || { echo "Driver installed; reboot then re-run setup_vm.sh"; exit 0; }
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "== 2. Docker, Compose, NVIDIA Container Toolkit"
if ! command -v docker >/dev/null || ! docker compose version >/dev/null 2>&1; then
  apt-get update -q
  apt-get install -y -q docker.io docker-compose-v2
fi
if ! command -v nvidia-ctk >/dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -q
  apt-get install -y -q nvidia-container-toolkit
fi
nvidia-ctk runtime configure --runtime=docker >/dev/null
systemctl restart docker
systemctl enable --now docker >/dev/null
docker compose version

echo "== 3. Artifact Registry auth"
gcloud auth configure-docker "$AR_HOST" --quiet >/dev/null
echo "ok $AR_HOST"

echo "== 4. Data from gs://$GCS_BUCKET"
pull() {  # pull <folder> [rsync args...]: skip quietly when the bucket has nothing there yet
  local f=$1; shift
  if gcloud storage ls "gs://$GCS_BUCKET/$f/" >/dev/null 2>&1; then
    mkdir -p "$DATA/$f"
    gcloud storage rsync --recursive "$@" "gs://$GCS_BUCKET/$f" "$DATA/$f"
    echo "ok $f/"
  else
    echo "skip $f/ (empty in the bucket)"
  fi
}
if [[ $ROLE == demo ]]; then
  # The demo VM carries only the four demo clips, never tuning or audio-only footage.
  pull samples --exclude='^(tuning|timing_audio_only)/'
else
  pull samples
fi
for f in events calibration transcripts video_head; do pull "$f"; done
chmod -R a+rX "$DATA"

echo "== 5. Idle watchdog (${IDLE_MIN} min)"
cat > /usr/local/sbin/si-idle-watchdog <<WATCHDOG
#!/usr/bin/env bash
# Stop this VM after ${IDLE_MIN} min with no running si containers, no GPU job and nobody logged in.
set -u
state=/var/lib/si-idle-since
busy=0
[[ -n "\$(docker ps -q --filter label=com.docker.compose.project=si 2>/dev/null)" ]] && busy=1
[[ -n "\$(who)" ]] && busy=1
util=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1)
[[ "\${util:-0}" -gt 5 ]] && busy=1
if (( busy )); then rm -f "\$state"; exit 0; fi
[[ -f "\$state" ]] || date +%s > "\$state"
if (( \$(date +%s) - \$(cat "\$state") >= ${IDLE_MIN} * 60 )); then
  logger -t si-idle-watchdog "idle for ${IDLE_MIN} min; stopping the VM"
  rm -f "\$state"; shutdown -h now
fi
WATCHDOG
chmod 755 /usr/local/sbin/si-idle-watchdog
cat > /etc/systemd/system/si-idle-watchdog.service <<'UNIT'
[Unit]
Description=Stop the GPU VM when idle
[Service]
Type=oneshot
ExecStart=/usr/local/sbin/si-idle-watchdog
UNIT
cat > /etc/systemd/system/si-idle-watchdog.timer <<'UNIT'
[Unit]
Description=Check for an idle GPU VM every 5 minutes
[Timer]
OnBootSec=10min
OnUnitActiveSec=5min
[Install]
WantedBy=timers.target
UNIT
systemctl daemon-reload
systemctl enable --now si-idle-watchdog.timer >/dev/null
echo "ok watchdog"

echo "setup done: role=$ROLE data=$DATA ($(du -sh "$DATA" | cut -f1))"
