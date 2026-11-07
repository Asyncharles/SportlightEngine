"""Scorer core: weighted parts, EMA, trend and the within-sport percentile."""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path

import numpy as np

from intensity.config import PercentileConfig, TrendConfig, resolve_path
from intensity.engine.adapters.base import AdapterPart
from intensity.schema import PART_NAMES, PartValue

log = logging.getLogger(__name__)


def combine(
    parts: dict[str, AdapterPart | None], weights: dict[str, float]
) -> tuple[float, dict[str, PartValue | None]]:
    """`intensity_raw` = sum(w_i * part_i) / sum(w_i) over present parts with weight > 0.

    Effective weights are renormalised over the present parts, so contributions sum to intensity_raw.
    """
    present = {
        name: part
        for name, part in parts.items()
        if part is not None and name in PART_NAMES and weights.get(name, 0.0) > 0
    }
    total_w = sum(weights[name] for name in present)
    out: dict[str, PartValue | None] = {}
    raw = 0.0
    for name in PART_NAMES:
        part = parts.get(name)
        if part is None:
            out[name] = None
            continue
        value = float(min(1.0, max(0.0, part.value)))
        w = weights.get(name, 0.0) / total_w if name in present and total_w > 0 else 0.0
        contribution = value * w
        raw += contribution
        out[name] = PartValue(
            value=round(value, 4),
            weight=round(w, 4),
            contribution=round(contribution, 4),
            phrase=part.phrase,
            facts=part.facts,
        )
    return float(raw), out


def forecast_part_value(p: float, quantiles: list[float] | None, sport: str) -> float:
    """The forecast's seventh-part value: p through the sport's validation distribution.

    Weightlifting instead uses 1 - |2p - 1|, which peaks when a lift is a coin flip.
    """
    if sport == "weightlifting":
        return float(1.0 - abs(2.0 * p - 1.0))
    if not quantiles:
        return float(min(1.0, max(0.0, p)))
    q = np.asarray(sorted(quantiles), dtype=float)
    return float(np.interp(p, q, np.linspace(0.0, 1.0, len(q))))


class Ema:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value: float | None = None

    def update(self, x: float) -> float:
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


class Trend:
    """Least-squares slope of intensity over the last `window_s` of t_video."""

    def __init__(self, cfg: TrendConfig):
        self.cfg = cfg
        self.buf: deque[tuple[float, float]] = deque(maxlen=512)

    def update(self, t: float, value: float) -> tuple[str, float]:
        self.buf.append((t, value))
        while self.buf and self.buf[0][0] < t - self.cfg.window_s:
            self.buf.popleft()
        if len(self.buf) < 3:
            return "steady", 0.0
        ts = np.fromiter((b[0] for b in self.buf), float)
        vs = np.fromiter((b[1] for b in self.buf), float)
        ts = ts - ts.mean()
        denom = float((ts * ts).sum())
        slope = float((ts * (vs - vs.mean())).sum() / denom) if denom > 0 else 0.0
        if slope >= self.cfg.rising_slope:
            return "rising", slope
        if slope <= self.cfg.falling_slope:
            return "falling", slope
        return "steady", slope

    def reset(self) -> None:
        self.buf.clear()


def load_intensity_ref(path: str | Path) -> dict[str, list[float]]:
    """`intensity_ref.json` (written by training): {sport: [intensity values from held-out matches]}."""
    p = resolve_path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        return {k: [float(x) for x in v] for k, v in data.items()}
    except (OSError, ValueError) as exc:
        log.warning("could not read %s: %s", p, exc)
        return {}


class SportPercentile:
    """`intensity_pct`: the percentile of a feed's intensity within its sport over a rolling 30 min.

    Seeded from intensity_ref.json so it is meaningful from the first tick. Shared by every feed of a
    sport, and kept across loops.
    """

    def __init__(self, cfg: PercentileConfig, seed: list[float] | None = None):
        self.cfg = cfg
        self.seed = np.sort(np.asarray(seed, dtype=float)) if seed else None
        self.samples: deque[tuple[float, float]] = deque(maxlen=max(64, int(cfg.window_s * 64)))

    def add(self, now: float, value: float) -> None:
        self.samples.append((now, value))
        while self.samples and self.samples[0][0] < now - self.cfg.window_s:
            self.samples.popleft()

    def pct(self, value: float) -> float:
        live = np.fromiter((s[1] for s in self.samples), float) if self.samples else np.empty(0)
        n = len(live)
        if n < self.cfg.min_samples and self.seed is not None and len(self.seed):
            ref = np.concatenate([live, self.seed])
        elif n == 0:
            return round(100.0 * value, 2)
        else:
            ref = live
        less = float((ref < value).sum())
        equal = float((ref == value).sum())
        return round(100.0 * (less + 0.5 * equal) / len(ref), 2)
