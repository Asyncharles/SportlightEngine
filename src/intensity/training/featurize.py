"""Step 2, featurize: replay each match through the live adapter and call `features()` once per second.

The adapter is the one the engine serves (hard rule 5): records are applied once the tick's game time
reaches them, exactly as the replayer releases them, so a row only ever sees the past. Ticks run from
each period's first record to its last; periods in `featurize.skip_periods` still update the adapter
but produce no rows.

Output: `data/tables/<sport>/features.parquet` with `match_id`, `match_period`, `game_time_s`,
`heuristic_p` (the adapter's fallback forecast, for comparison in the report) and the ordered feature
columns. `features.meta.json` beside it records the input hash, so an unchanged rerun is skipped.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from intensity.config import Config, resolve_path
from intensity.engine.adapters import make_adapter
from intensity.engine.adapters.base import Adapter, config_hash, feature_hash
from intensity.engine.replay import load_records
from intensity.schema import MatchPayload
from intensity.telemetry import log_event

log = logging.getLogger(__name__)

KEYS = ["match_id", "match_period", "game_time_s"]  # match_period: a feature may be called period


@dataclass
class MatchRows:
    period: np.ndarray
    game_time_s: np.ndarray
    heuristic_p: np.ndarray
    x: np.ndarray  # (ticks, features)


def ordered(records: list[MatchPayload]) -> list[MatchPayload]:
    return sorted(records, key=lambda r: (r.period, r.game_time_s))


def period_spans(records: list[MatchPayload]) -> list[tuple[int, float, float]]:
    """(period, first record time, last record time) for every period, in order."""
    spans: dict[int, list[float]] = {}
    for r in records:
        span = spans.setdefault(r.period, [r.game_time_s, r.game_time_s])
        span[0] = min(span[0], r.game_time_s)
        span[1] = max(span[1], r.game_time_s)
    return [(p, s, e) for p, (s, e) in sorted(spans.items())]


def tick_times(start: float, end: float, step: float) -> np.ndarray:
    return start + step * np.arange(int(np.floor((end - start) / step)) + 1)


def featurize_match(
    sport: str, records: list[MatchPayload], step_s: float = 1.0, skip_periods: tuple[int, ...] = ()
) -> MatchRows:
    recs = ordered(records)
    adapter = make_adapter(sport)
    periods: list[int] = []
    times: list[float] = []
    heuristic: list[float] = []
    rows: list[np.ndarray] = []
    i = 0
    for period, start, end in period_spans(recs):
        for t in tick_times(start, end, step_s):
            t = float(t)
            while i < len(recs) and (recs[i].period, recs[i].game_time_s) <= (period, t):
                adapter.on_match(recs[i])
                i += 1
            if period in skip_periods:
                continue
            periods.append(period)
            times.append(t)
            heuristic.append(adapter.heuristic_forecast(t))
            rows.append(adapter.features(t))
    n_features = len(adapter.FEATURES)
    return MatchRows(
        period=np.asarray(periods, dtype=np.int16),
        game_time_s=np.asarray(times, dtype=float),
        heuristic_p=np.asarray(heuristic, dtype=float),
        x=np.vstack(rows) if rows else np.empty((0, n_features)),
    )


# --------------------------------------------------------------------------- tables


def events_dir(config: Config, sport: str) -> Path:
    return resolve_path(config.training["paths"]["events"]) / sport


def tables_dir(config: Config, sport: str) -> Path:
    return resolve_path(config.training["paths"]["tables"]) / sport


def match_files(config: Config, sport: str) -> list[Path]:
    return sorted(events_dir(config, sport).glob("*.jsonl"))


def data_hash(paths: list[Path]) -> str:
    """Content hash of the match records, stored in the bundle."""
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode())
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()[:16]


def code_hash(*objects: Any) -> str:
    """Hash of the source files behind these objects, so a code change rebuilds a cached table even
    when the feature names stay the same."""
    h = hashlib.sha256()
    for path in sorted({inspect.getsourcefile(o) or "" for o in objects}):
        h.update(Path(path).read_bytes())
    return h.hexdigest()[:16]


def featurize_settings(config: Config, sport: str) -> tuple[float, tuple[int, ...]]:
    cfg = config.training.get("featurize", {})
    return float(cfg.get("step_s", 1.0)), tuple(int(p) for p in cfg.get("skip_periods", {}).get(sport, []))


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def read_meta(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def featurize(config: Config, sport: str, force: bool = False) -> Path:
    paths = match_files(config, sport)
    if not paths:
        raise FileNotFoundError(
            f"no match records in {events_dir(config, sport)}: run `convert --sport {sport}`"
        )
    adapter = make_adapter(sport)
    step_s, skip = featurize_settings(config, sport)
    inputs = {
        "data_hash": data_hash(paths),
        "feature_hash": feature_hash(adapter.FEATURES),
        "adapter_config_hash": config_hash(adapter.cfg),
        "code_hash": code_hash(type(adapter), Adapter, featurize_match),
        "step_s": step_s,
        "skip_periods": list(skip),
    }
    out = tables_dir(config, sport) / "features.parquet"
    meta_path = out.with_name("features.meta.json")
    if not force and out.exists() and read_meta(meta_path).get("inputs") == inputs:
        log_event(log, "featurize up to date", sport=sport, path=str(out))
        return out
    frames = []
    for path in paths:
        rows = featurize_match(sport, load_records(path), step_s, skip)
        df = pd.DataFrame(rows.x, columns=list(adapter.FEATURES))
        df.insert(0, "heuristic_p", rows.heuristic_p)
        df.insert(0, "game_time_s", rows.game_time_s)
        df.insert(0, "match_period", rows.period)
        df.insert(0, "match_id", path.stem)
        frames.append(df)
    table = pd.concat(frames, ignore_index=True)
    write_parquet(table, out)
    meta_path.write_text(json.dumps({"inputs": inputs, "rows": len(table), "matches": len(paths)}, indent=1))
    log_event(log, "featurize done", sport=sport, rows=len(table), matches=len(paths), path=str(out))
    return out
