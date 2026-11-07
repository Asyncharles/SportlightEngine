# One image, several entry points: ingest (GPU), engine (CPU), feeds, training CLI, bench.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv ffmpeg ca-certificates curl make \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra gpu --extra train --extra polish --no-dev
COPY src ./src
COPY config ./config
COPY contracts ./contracts
COPY models ./models
COPY scripts ./scripts
COPY data/sync ./data/sync
COPY Makefile ./
COPY make ./make
RUN uv sync --frozen --extra gpu --extra train --extra polish --no-dev

# Perception weights are fetched at build time so nothing downloads on stage.
ARG PREFETCH_WEIGHTS=1
RUN if [ "$PREFETCH_WEIGHTS" = "1" ]; then python - <<'PY' || echo "weight prefetch skipped"; fi
import open_clip
from faster_whisper import download_model
open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
download_model("small.en")
PY

EXPOSE 8000
CMD ["python", "-m", "intensity.engine.service", "--overlay", "demo"]
