"""Check the private bucket after the data package upload (docs/deploy-wiring-plan.md, Milestone 1).

    python scripts/verify_bucket.py [--package sportlight-data.zip|dir] [--allow-raw-football]

Checks, each printed as ok / FAIL:
  1. bucket settings: uniform access, public access prevention enforced;
  2. file counts per folder against bucket-plan.md, and sizes: exact per file with --package
     (the zip or its unpacked folder), otherwise within 25% of the plan's rough totals;
  3. every labelled clip is at the path its labels expect (feeds.yaml samples, data/sync tuning clips,
     data/weightlifting sessions), under its exact file name;
  4. the 2018 StatsBomb files (raw/football/7560.json, 7567.json) are absent unless --allow-raw-football;
  5. anonymous requests for an object and for the object listing are refused (403).
Standard library only; talks to GCS through the gcloud CLI. Exit 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# bucket-plan.md, "Uploading now": prefix -> (file suffix, files, rough bytes). Prefixes are direct
# children only unless they end in "**".
GB, MB = 1e9, 1e6
PLAN = {
    "samples/": (".mp4", 4, 2.3 * GB),
    "samples/tuning/football/": (".mp4", 4, 1.2 * GB),
    "samples/tuning/f1/": (".mp4", 2, 0.9 * GB),
    "samples/tuning/nba/": (".mp4", 4, 2.4 * GB),
    "samples/tuning/weightlifting/": (".mp4", 3, 1.7 * GB),
    "samples/timing_audio_only/f1/": (".mp4", 3, 0.6 * GB),
    "stills/**": (".png", 592, 0.4 * GB),
    "raw/nba/": (".json", 5, 1 * MB),
}  # raw/football/'s two 2018 files are checked separately (RAW_2018)
RAW_2018 = {"raw/football/7560.json", "raw/football/7567.json"}  # 2018 World Cup, awaiting the ML lead
SPORT_PREFIX = {"fb": "football", "f1": "f1", "nba": "nba", "wl": "weightlifting"}
IGNORED = (".DS_Store", ".gitkeep")

failures = 0


def check(ok: bool, msg: str) -> None:
    global failures
    print(("ok    " if ok else "FAIL  ") + msg)
    failures += not ok


def gcloud_json(*args: str) -> object:
    out = subprocess.run(["gcloud", *args, "--format=json"], check=True, capture_output=True, text=True).stdout
    return json.loads(out or "[]")


def bucket_name() -> str:
    name = os.environ.get("GCS_BUCKET", "")
    env = ROOT / ".env"
    if not name and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("GCS_BUCKET="):
                name = line.split("=", 1)[1].strip().strip("'\"")
    name = name.removeprefix("gs://").rstrip("/")
    if not name:
        sys.exit("GCS_BUCKET is not set (environment or .env)")
    return name


def list_objects(bucket: str) -> dict[str, int]:
    """Every object key -> size in bytes."""
    rows = gcloud_json("storage", "objects", "list", f"gs://{bucket}/**")
    return {r["name"]: int(r["size"]) for r in rows if not r["name"].endswith("/")}  # type: ignore[index]


def package_files(path: Path) -> dict[str, int]:
    """Bucket key -> size for the data package, keyed from the first known top-level folder."""
    tops = {"samples", "stills", "raw"}
    if path.is_dir():
        entries = [(p.relative_to(path).as_posix(), p.stat().st_size) for p in path.rglob("*") if p.is_file()]
    else:
        with zipfile.ZipFile(path) as z:
            entries = [(i.filename, i.file_size) for i in z.infolist() if not i.is_dir()]
    out = {}
    for name, size in entries:
        parts = name.split("/")
        idx = next((i for i, p in enumerate(parts) if p in tops), None)
        if idx is None or parts[-1] in IGNORED or "__MACOSX" in parts:
            continue
        out["/".join(parts[idx:])] = size
    return out


def in_prefix(key: str, prefix: str) -> bool:
    if prefix.endswith("**"):
        return key.startswith(prefix[:-2])
    return key.startswith(prefix) and "/" not in key[len(prefix):]


def expected_clips() -> list[str]:
    """Bucket keys the repo's labels point at."""
    sys.path.insert(0, str(ROOT / "src"))
    import yaml  # noqa: PLC0415 - pyyaml is a base dependency; only needed here

    feeds = yaml.safe_load((ROOT / "config" / "feeds.yaml").read_text())["feeds"]
    demo = {Path(f["sample"]).stem for f in feeds}
    keys = [f["sample"] for f in feeds]
    for csv in sorted((ROOT / "data" / "sync").glob("*.csv")) + sorted((ROOT / "data" / "weightlifting").glob("*.csv")):
        clip = csv.stem
        if clip not in demo:
            keys.append(f"samples/tuning/{SPORT_PREFIX[clip.split('_')[0]]}/{clip}.mp4")
    return keys


def anonymous_status(url: str) -> int:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=20) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", type=Path, help="sportlight-data.zip or its unpacked folder, for exact sizes")
    ap.add_argument("--allow-raw-football", action="store_true", help="the 2018 raw/football files were approved")
    args = ap.parse_args()
    bucket = bucket_name()
    print(f"bucket gs://{bucket}")

    meta = gcloud_json("storage", "buckets", "describe", f"gs://{bucket}")
    check(meta.get("uniform_bucket_level_access") is True, "uniform bucket-level access")  # type: ignore[union-attr]
    check(meta.get("public_access_prevention") == "enforced", "public access prevention enforced")  # type: ignore[union-attr]

    objects = list_objects(bucket)
    print(f"{len(objects)} objects, {sum(objects.values()) / GB:.2f} GB")

    for prefix, (suffix, count, rough) in PLAN.items():
        files = {k: s for k, s in objects.items() if in_prefix(k, prefix) and k.endswith(suffix)}
        total = sum(files.values())
        check(len(files) == count, f"{prefix:32} {len(files)} {suffix} files (plan {count})")
        if not args.package:
            check(abs(total - rough) <= 0.25 * rough, f"{prefix:32} {total / GB:.2f} GB (plan about {rough / GB:.2f} GB)")

    if args.package:
        pkg = package_files(args.package)
        if not args.allow_raw_football:
            pkg = {k: s for k, s in pkg.items() if k not in RAW_2018}
        missing = sorted(k for k in pkg if k not in objects)
        wrong = sorted(k for k in pkg if k in objects and objects[k] != pkg[k])
        check(not missing, f"every package file is in the bucket ({len(pkg)} files, {len(missing)} missing)")
        for k in missing[:20]:
            print(f"        missing {k}")
        check(not wrong, f"every package file has the same size ({len(wrong)} differ)")
        for k in wrong[:20]:
            print(f"        {k}: bucket {objects[k]} bytes, package {pkg[k]} bytes")

    clips = expected_clips()
    absent = [k for k in clips if k not in objects]
    check(not absent, f"every labelled clip is at its expected path ({len(clips)} clips)")
    for k in absent:
        print(f"        missing {k}")

    raw_2018 = sorted(k for k in objects if k in RAW_2018)
    if args.allow_raw_football:
        print(f"info  2018 StatsBomb files present: {raw_2018} (approved)")
    else:
        check(not raw_2018, "2018 StatsBomb files absent from raw/football/ (need ML lead approval)")

    probe = next(iter(sorted(objects)), None)
    if probe is None:
        check(False, "anonymous access check needs at least one object")
    else:
        url = f"https://storage.googleapis.com/{bucket}/{urllib.request.quote(probe)}"
        code = anonymous_status(url)
        check(code == 403, f"anonymous GET of an object returns {code} (want 403)")
    code = anonymous_status(f"https://storage.googleapis.com/storage/v1/b/{bucket}/o")
    check(code in (401, 403), f"anonymous object listing returns {code} (want 401/403)")

    print("all checks passed" if failures == 0 else f"{failures} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
