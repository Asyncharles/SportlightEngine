"""The data layer: one place that decides where `data/` lives, on a laptop and on the VM.

Every data path in config is written as `data/<folder>/...`. It resolves under `data_dir()`, which is
`<repo>/data` unless `INTENSITY_DATA_DIR` points elsewhere (the VM mounts the bucket copy at
`/app/data`). The private bucket mirrors the same tree without the `data/` prefix, so
`gs://<GCS_BUCKET>/samples/x.mp4` is `data/samples/x.mp4`; `scripts/sync_bucket.sh` moves files
between the two. Training (laptop) and serving (VM) therefore read identical relative paths.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"  # read for GCS_BUCKET only, like scripts/sync_bucket.sh (it may hold secrets)

# Bucket folders, mirroring data/. The VM pulls SERVE_FOLDERS at setup (infra/gcp/setup_vm.sh).
FOLDERS = ("samples", "stills", "raw", "events", "tables", "video_head", "transcripts", "tapes",
           "calibration", "reports", "sync", "weightlifting", "highlights")
SERVE_FOLDERS = ("samples", "events", "calibration", "transcripts", "video_head")
# Folders the GPU VM's identity may write (infra/gcp/bucket.sh).
VM_WRITE_FOLDERS = ("reports", "tapes", "events")


def data_dir() -> Path:
    """The local root of the bucket mirror."""
    env = os.environ.get("INTENSITY_DATA_DIR")
    return Path(env) if env else REPO_ROOT / "data"


def data_path(rel: str | Path) -> Path:
    """`samples/x.mp4` (a bucket key, no `data/` prefix) to its local path."""
    return data_dir() / Path(rel)


def is_data_path(path: str | Path) -> bool:
    parts = PurePosixPath(Path(path).as_posix()).parts
    return bool(parts) and parts[0] == "data"


def resolve_data(path: str | Path) -> Path:
    """A repo-relative `data/...` path to its local location under `data_dir()`."""
    parts = PurePosixPath(Path(path).as_posix()).parts
    if not parts or parts[0] != "data":
        raise ValueError(f"not a data/ path: {path}")
    return data_dir().joinpath(*parts[1:])


def bucket() -> str | None:
    """The bucket name from `GCS_BUCKET` (with or without `gs://`), else from `.env`; None when unset."""
    name = os.environ.get("GCS_BUCKET", "").strip()
    if not name and ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if line.startswith("GCS_BUCKET="):
                name = line.split("=", 1)[1].strip().strip("'\"")
    return name.removeprefix("gs://").rstrip("/") or None


def bucket_uri(rel: str | Path = "") -> str:
    """`samples/x.mp4` to `gs://<bucket>/samples/x.mp4`."""
    name = bucket()
    if name is None:
        raise RuntimeError("GCS_BUCKET is not set (see .env.example)")
    key = PurePosixPath(Path(rel).as_posix()).as_posix().lstrip("./") if str(rel) else ""
    return f"gs://{name}/{key}" if key else f"gs://{name}/"
