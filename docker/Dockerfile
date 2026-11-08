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

# Perception weights are fetched at build time so nothing downloads on stage. They depend only on the
# dependencies above, so they sit before any repo file: a code change no longer re-downloads and re-pushes
# them. (One python -c line, not a heredoc: Cloud Build's default Dockerfile frontend has no heredocs.)
ARG PREFETCH_WEIGHTS=1
RUN if [ "$PREFETCH_WEIGHTS" = "1" ]; then \
        python -c 'import open_clip; from faster_whisper import download_model; open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k"); download_model("small.en")' \
        || echo "weight prefetch skipped"; \
    fi

# Repo files, least often changed first, so a code change rebuilds only the last few small layers.
COPY data/sync ./data/sync
COPY Makefile ./
COPY make ./make
COPY contracts ./contracts
COPY scripts ./scripts
COPY models ./models
COPY config ./config
COPY src ./src
RUN uv sync --frozen --extra gpu --extra train --extra polish --no-dev
# Fail the build if the image can't even import the services (inside the build, so Cloud Build never
# has to pull the ~10 GB image back just to check it).
RUN python -c "import intensity.engine.service, intensity.ingest.run; print('imports ok')"

EXPOSE 8000
CMD ["python", "-m", "intensity.engine.service", "--overlay", "demo"]
