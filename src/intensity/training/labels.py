"""Step 3, label: look forward along the same record stream from every featurized tick.

A tick is positive when a label event comes after it (strictly: the tick's own records are already
in its features) and within the horizon, in the same period. A negative tick whose horizon runs past
the end of its period is dropped, since its outcome is unknown. Labels read only records after the
tick; features read only records up to it.

Output: `data/tables/<sport>/labels.parquet` with `match_id`, `match_period`, `game_time_s` and `y`
(dropped ticks are left out).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from intensity.config import Config
from intensity.engine.replay import load_records
from intensity.schema import MatchPayload
from intensity.telemetry import log_event
from intensity.training.featurize import (
    KEYS,
    events_dir,
    featurize,
    period_spans,
    read_meta,
    tables_dir,
    write_parquet,
)

log = logging.getLogger(__name__)


def football_events(records: list[MatchPayload], cfg: dict[str, Any]) -> list[tuple[int, float]]:
    """Big chances: a shot with xG >= min_xg, or a goal (own goals included, shoot-out kicks not)."""
    min_xg = float(cfg.get("min_xg", 0.1))
    return [
        (r.period, r.game_time_s)
        for r in records
        if not r.attrs.get("shootout")
        and (r.kind == "goal" or (r.kind == "shot" and float(r.attrs.get("xg", 0.0)) >= min_xg))
    ]


LABEL_EVENTS: dict[str, Callable[[list[MatchPayload], dict[str, Any]], list[tuple[int, float]]]] = {
    "football": football_events,
}


def label_cfg(config: Config, sport: str) -> dict[str, Any]:
    if sport not in LABEL_EVENTS:
        raise NotImplementedError(f"no label for {sport} yet (football in T1, F1 in T2, NBA in T3)")
    return dict(config.training["labels"][sport])


def label_match(
    records: list[MatchPayload], period: np.ndarray, game_time_s: np.ndarray, sport: str, cfg: dict[str, Any]
) -> np.ndarray:
    """y per tick: 1, 0, or NaN where the horizon runs past the period end with no positive."""
    horizon = float(cfg["horizon_s"])
    ends = {p: e for p, _, e in period_spans(records)}
    by_period: dict[int, list[float]] = {}
    for p, t in LABEL_EVENTS[sport](records, cfg):
        by_period.setdefault(p, []).append(t)
    events = {p: np.sort(np.asarray(ts, dtype=float)) for p, ts in by_period.items()}
    y = np.full(len(game_time_s), np.nan)
    for i, (p, t) in enumerate(zip(period, game_time_s, strict=True)):
        ts = events.get(int(p))
        nxt = None
        if ts is not None:
            j = int(np.searchsorted(ts, t, side="right"))
            nxt = ts[j] if j < len(ts) else None
        if nxt is not None and nxt - t <= horizon:
            y[i] = 1.0
        elif t + horizon <= ends.get(int(p), t):
            y[i] = 0.0
    return y


def label(config: Config, sport: str, force: bool = False) -> Path:
    cfg = label_cfg(config, sport)
    features_path = featurize(config, sport)
    inputs = {
        "features": read_meta(features_path.with_name("features.meta.json")).get("inputs"),
        "label": cfg,
    }
    out = tables_dir(config, sport) / "labels.parquet"
    meta_path = out.with_name("labels.meta.json")
    if not force and out.exists() and read_meta(meta_path).get("inputs") == inputs:
        log_event(log, "label up to date", sport=sport, path=str(out))
        return out
    keys = pd.read_parquet(features_path, columns=KEYS)
    frames = []
    for match_id, ticks in keys.groupby("match_id", sort=True):
        records = load_records(events_dir(config, sport) / f"{match_id}.jsonl")
        y = label_match(
            records, ticks["match_period"].to_numpy(), ticks["game_time_s"].to_numpy(), sport, cfg
        )
        frames.append(ticks.assign(y=y)[~np.isnan(y)])
    table = pd.concat(frames, ignore_index=True)
    table["y"] = table["y"].astype(np.int8)
    write_parquet(table, out)
    stats = {"rows": len(table), "dropped": len(keys) - len(table), "base_rate": float(table["y"].mean())}
    meta_path.write_text(json.dumps({"inputs": inputs, **stats}, indent=1))
    log_event(log, "label done", sport=sport, path=str(out), **stats)
    return out
