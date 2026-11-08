"""Raw files to match records: `data/raw/<sport>/<item>.json` -> `data/events/<sport>/<item>.jsonl`.

The second half of pipeline step 1. Each line is a `MatchPayload`, the shape the engine's replayer
turns into `match` Events, so training and the live service read the same records (hard rule 5).
An item is reconverted when its raw file is newer than its JSONL, or with `force`.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from intensity.schema import MatchPayload
from intensity.telemetry import log_event

log = logging.getLogger(__name__)

ToRecords = Callable[[dict[str, Any], dict[str, Any]], list[dict[str, Any]]]


@dataclass
class ConvertStats:
    converted: int = 0
    skipped: int = 0
    failed: dict[str, str] = field(default_factory=dict)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def convert_dir(
    sport: str,
    to_records: ToRecords,
    raw_dir: Path,
    events_dir: Path,
    cfg: dict[str, Any],
    force: bool = False,
    items: list[str] | None = None,
) -> ConvertStats:
    stats = ConvertStats()
    raw_files = sorted(p for p in raw_dir.glob("*.json") if not p.name.startswith("_"))
    if items is not None:
        raw_files = [p for p in raw_files if p.stem in set(items)]
    for raw_path in raw_files:
        out = events_dir / f"{raw_path.stem}.jsonl"
        if not force and out.exists() and out.stat().st_mtime >= raw_path.stat().st_mtime:
            stats.skipped += 1
            continue
        try:
            records = to_records(json.loads(raw_path.read_text()), cfg)
            for rec in records:
                MatchPayload.model_validate(rec)
        except Exception as exc:  # one bad file must not stop the rest
            stats.failed[raw_path.stem] = f"{type(exc).__name__}: {exc}"
            log_event(log, "convert failed", logging.ERROR, sport=sport, item=raw_path.stem, error=str(exc))
            continue
        write_jsonl(out, records)
        stats.converted += 1
    log_event(
        log,
        "convert done",
        sport=sport,
        converted=stats.converted,
        skipped=stats.skipped,
        failed=len(stats.failed),
    )
    return stats
