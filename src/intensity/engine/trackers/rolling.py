"""Per-feed rolling percentiles, seeded from calibration, shared by every tracker."""

from __future__ import annotations

import bisect
import json
import logging
from collections import deque
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from intensity.config import RollingConfig, resolve_path

log = logging.getLogger(__name__)


class RollingPercentile:
    """Percentile rank of a value against the feed's last `window_s` of samples.

    Until `min_samples` live samples exist, the rank blends in the seed distribution (deciles from
    `calibration.json` or config), weighted by how full the live window is. Samples are timed by
    `t_video`, and a sample is only ever added when the engine's clock has reached it.
    """

    def __init__(self, seeds: Sequence[float], cfg: RollingConfig):
        seeds = sorted(float(s) for s in seeds) if seeds else [0.0, 1.0]
        self.seed_values = np.asarray(seeds)
        self.seed_levels = np.linspace(0.0, 1.0, len(seeds))
        self.window_s = cfg.window_s
        self.min_samples = cfg.min_samples
        self._times: deque[tuple[float, float]] = deque(maxlen=cfg.max_samples)
        self._sorted: list[float] = []

    def __len__(self) -> int:
        return len(self._sorted)

    def add(self, t: float, value: float) -> None:
        if len(self._times) == self._times.maxlen:
            self._remove(self._times[0][1])
        self._times.append((t, value))
        bisect.insort(self._sorted, value)
        self.prune(t)

    def prune(self, now: float) -> None:
        while self._times and self._times[0][0] < now - self.window_s:
            _, v = self._times.popleft()
            self._remove(v)

    def _remove(self, value: float) -> None:
        i = bisect.bisect_left(self._sorted, value)
        if i < len(self._sorted) and self._sorted[i] == value:
            del self._sorted[i]

    def seed_rank(self, value: float) -> float:
        return float(np.interp(value, self.seed_values, self.seed_levels))

    def live_rank(self, value: float) -> float:
        n = len(self._sorted)
        if n == 0:
            return 0.5
        lo = bisect.bisect_left(self._sorted, value)
        hi = bisect.bisect_right(self._sorted, value)
        return (lo + 0.5 * (hi - lo)) / n

    def rank(self, value: float) -> float:
        n = len(self._sorted)
        if n >= self.min_samples:
            return self.live_rank(value)
        w = n / self.min_samples
        return w * self.live_rank(value) + (1 - w) * self.seed_rank(value)

    def median(self) -> float:
        """Median of the live window, or of the seeds while it is empty."""
        if self._sorted:
            return self._sorted[len(self._sorted) // 2]
        return float(np.interp(0.5, self.seed_levels, self.seed_values))

    def reset(self) -> None:
        self._times.clear()
        self._sorted.clear()


class Seeds:
    """Seed deciles per sport and signal: `calibration.json` first, then config defaults.

    calibration.json (written by `make video-calibrate`): {sport: {signal: [deciles]}}, where a
    signal is `motion`, `pan_speed`, `prompt:<prompt text>`, `loudness_rel_db`, `rise_db_s`,
    `energy_rel_db`, `pitch_rel_st` or `pace_rel`.
    """

    def __init__(self, sport: str, defaults: dict[str, list[float]], calibration_path: str | Path | None = None):
        self.sport = sport
        self.defaults = defaults
        self.calibration: dict[str, list[float]] = {}
        if calibration_path:
            path = resolve_path(calibration_path)
            if path.exists():
                try:
                    self.calibration = json.loads(path.read_text()).get(sport, {})
                except (OSError, ValueError) as exc:
                    log.warning("could not read %s: %s", path, exc)

    def get(self, signal: str) -> list[float]:
        if signal in self.calibration:
            return self.calibration[signal]
        if signal in self.defaults:
            return self.defaults[signal]
        family = signal.split(":", 1)[0]
        return self.defaults.get(family, [0.0, 1.0])
