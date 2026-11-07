"""`make env-check`: is this machine ready for the GPU ingest?

Checks, in order: nvidia-smi; an FFmpeg NVDEC smoke decode with decoder utilisation above zero;
PyNvVideoCodec decoding one frame into a CUDA tensor through DLPack; faster-whisper loading a tiny
model on CUDA in the same process after PyTorch has initialised CUDA; LightGBM importing. Writes an
environment snapshot (GPU, driver, CUDA, cuDNN, library versions) to run/env_check.json.

PyNvVideoCodec's exact API differs between releases: read the samples shipped in the wheel and
adjust `check_pynvvideocodec` if it fails on the VM (the plan's working rule).
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBS = ["torch", "pynvvideocodec", "faster-whisper", "ctranslate2", "open-clip-torch", "lightgbm", "av",
        "nvidia-ml-py", "fastapi", "pydantic", "numpy"]


def run(cmd: list[str], timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def check_nvidia_smi() -> tuple[bool, str]:
    if not shutil.which("nvidia-smi"):
        return False, "nvidia-smi not found (no NVIDIA driver)"
    r = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    return r.returncode == 0, (r.stdout or r.stderr).strip()


def _test_clip(path: Path) -> None:
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30",
         "-t", "10", "-c:v", "libx264", "-g", "30", "-bf", "0", str(path)], timeout=120)


def check_ffmpeg_nvdec(clip: Path) -> tuple[bool, str]:
    if not shutil.which("ffmpeg"):
        return False, "ffmpeg not found"
    try:
        import pynvml
    except ImportError:
        pynvml = None
    peak = [0]
    stop = threading.Event()

    def sample() -> None:
        if pynvml is None:
            return
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        while not stop.is_set():
            util, _ = pynvml.nvmlDeviceGetDecoderUtilization(handle)
            peak[0] = max(peak[0], util)
            time.sleep(0.05)

    t = threading.Thread(target=sample, daemon=True)
    t.start()
    r = run(["ffmpeg", "-loglevel", "error", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
             "-i", str(clip), "-f", "null", "-"], timeout=120)
    stop.set()
    t.join(timeout=1)
    if r.returncode != 0:
        return False, r.stderr.strip()[-300:]
    if pynvml is None:
        return False, "decoded, but nvidia-ml-py is missing so decoder utilisation is unknown"
    return peak[0] > 0, f"peak NVDEC utilisation {peak[0]}%"


def check_pynvvideocodec(clip: Path) -> tuple[bool, str]:
    try:
        import PyNvVideoCodec as nvc
        import torch
    except ImportError as exc:
        return False, f"import failed: {exc}"
    try:
        demuxer = nvc.CreateDemuxer(filename=str(clip))
        decoder = nvc.CreateDecoder(gpuid=0, codec=demuxer.GetNvCodecId(), cudacontext=0, cudastream=0, usedevicememory=True)
        for packet in demuxer:
            for frame in decoder.Decode(packet):
                tensor = torch.from_dlpack(frame)
                return tensor.is_cuda, f"frame {tuple(tensor.shape)} on {tensor.device}"
        return False, "no frame decoded"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc} (check the wheel's samples for this release's API)"


def check_whisper_after_torch() -> tuple[bool, str]:
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "torch.cuda.is_available() is False"
        torch.zeros(1, device="cuda")  # PyTorch initialises CUDA first
        from faster_whisper import WhisperModel

        WhisperModel("tiny", device="cuda", compute_type="float16")
        return True, f"torch {torch.__version__} (CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()})"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def check_lightgbm() -> tuple[bool, str]:
    try:
        lgb = importlib.import_module("lightgbm")
        return True, lgb.__version__
    except (ImportError, OSError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


def versions() -> dict[str, str | None]:
    out = {}
    for name in LIBS:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


def main() -> int:
    results: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "smoke.mp4"
        if shutil.which("ffmpeg"):
            _test_clip(clip)
        checks = [
            ("nvidia_smi", check_nvidia_smi),
            ("ffmpeg_nvdec", lambda: check_ffmpeg_nvdec(clip)),
            ("pynvvideocodec_dlpack", lambda: check_pynvvideocodec(clip)),
            ("faster_whisper_after_torch", check_whisper_after_torch),
            ("lightgbm", check_lightgbm),
        ]
        for name, fn in checks:
            try:
                ok, detail = fn()
            except Exception as exc:
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            results[name] = {"ok": ok, "detail": detail}
            print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    snapshot = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "checks": results,
        "versions": versions(),
    }
    out = ROOT / "run" / "env_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2))
    print(f"snapshot: {out}")
    return 0 if all(r["ok"] for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
