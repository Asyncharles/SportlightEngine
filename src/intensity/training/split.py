"""Step 4, split, under one switch: `training.split.mode`.

`in_sample` (the default, Charles's decision of 2026-11-08): no split. Every match, demo matches
included, is in `all`, so the model trains on the full dataset and is evaluated on that same data.
Every metric is in-sample, and the report says so. Hard rule 6 (demo matches never train) is
suspended in this mode.

`holdout`: group by match with the fixed seed. Demo matches from `feeds.yaml` always go to test
(hard rule 6), plus `test_frac` of the other matches; `val_frac` go to validation and the rest to
training. The build fails if a demo match lands in training or validation, or if a demo feed for the
sport has no `match_id` yet (it could leak).

Output: `data/tables/<sport>/split.json` with
`{"mode": ..., "demo": [...], "parts": {match_id: train|val|test|all}}`.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from intensity.config import Config, resolve_path
from intensity.feeds import load_feeds
from intensity.telemetry import log_event
from intensity.training.featurize import match_files, tables_dir

log = logging.getLogger(__name__)


MODES = ("in_sample", "holdout")
ALL = "all"  # the only part in in_sample mode: train, val and test are all of it


class SplitError(RuntimeError):
    pass


def split_mode(config: Config) -> str:
    mode = str(config.training["split"].get("mode", "holdout"))
    if mode not in MODES:
        raise SplitError(f"training.split.mode must be one of {MODES}, not {mode!r}")
    return mode


def members(parts: dict[str, str], name: str) -> list[str]:
    """Matches in part `name` (train, val or test); in in_sample mode that is every match."""
    return sorted(m for m, p in parts.items() if p in (name, ALL))


def demo_match_ids(config: Config, sport: str) -> set[str]:
    feeds = load_feeds(resolve_path(config.training["paths"]["feeds"]))
    unset = [f.feed_id for f in feeds.by_sport(sport) if f.demo and f.match_ref.match_id is None]
    if unset:
        raise SplitError(
            f"demo feed(s) {unset} have no match_ref.match_id in feeds.yaml; the split cannot keep them out"
            " of training (hard rule 6)"
        )
    return feeds.demo_match_ids(sport)


def split_matches(
    match_ids: list[str], demo: set[str], val_frac: float, test_frac: float, seed: int, mode: str = "holdout"
) -> dict[str, str]:
    if mode == "in_sample":
        return {m: ALL for m in sorted(set(match_ids))}
    others = sorted(set(match_ids) - demo)
    order = [others[i] for i in np.random.default_rng(seed).permutation(len(others))]
    n_test = round(test_frac * len(others))
    n_val = round(val_frac * len(others))
    parts = {m: "test" for m in sorted(set(match_ids) & demo)}
    for i, m in enumerate(order):
        parts[m] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    assert_no_demo_leak(parts, demo)
    return parts


def assert_no_demo_leak(parts: dict[str, str], demo: set[str]) -> None:
    leaked = sorted(m for m in demo if parts.get(m) in ("train", "val"))
    if leaked:
        raise SplitError(f"demo matches in training or validation (hard rule 6): {leaked}")


def split(config: Config, sport: str) -> Path:
    mode = split_mode(config)
    demo = demo_match_ids(config, sport)
    match_ids = [p.stem for p in match_files(config, sport)]
    missing = sorted(demo - set(match_ids))
    if missing:
        log.warning("demo matches with no records, so no highlight check for them: %s", missing)
    cfg = config.training["split"]
    parts = split_matches(
        match_ids, demo, float(cfg["val_frac"]), float(cfg["test_frac"]), int(config.training["seed"]), mode
    )
    out = tables_dir(config, sport) / "split.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"mode": mode, "demo": sorted(demo), "parts": parts}, indent=1, sort_keys=True))
    log_event(log, "split done", sport=sport, mode=mode, path=str(out), **Counter(parts.values()))
    return out


def load_split(path: Path) -> tuple[dict[str, str], set[str]]:
    data = json.loads(path.read_text())
    parts, demo = data["parts"], set(data["demo"])
    if data.get("mode", "holdout") == "holdout":
        assert_no_demo_leak(parts, demo)
    return parts, demo


def is_in_sample(parts: dict[str, str]) -> bool:
    return bool(parts) and all(p == ALL for p in parts.values())
