"""Structured JSON logging, counters and latency histograms shared by every process."""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections import defaultdict, deque
from typing import Any

import numpy as np


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            out.update(extra)
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def setup_logging(level: str = "INFO", json_format: bool = True) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if json_format else logging.Formatter("%(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())


def log_event(logger: logging.Logger, msg: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, msg, extra={"fields": fields})


class Histogram:
    """Keeps the last `maxlen` samples; enough for p50/p95/p99 on /metrics."""

    def __init__(self, maxlen: int = 4096):
        self._samples: deque[float] = deque(maxlen=maxlen)
        self.count = 0

    def observe(self, value: float) -> None:
        self._samples.append(value)
        self.count += 1

    def summary(self) -> dict[str, float | int]:
        if not self._samples:
            return {"count": self.count}
        arr = np.fromiter(self._samples, dtype=float)
        p50, p95, p99 = np.percentile(arr, [50, 95, 99])
        return {
            "count": self.count,
            "p50": round(float(p50), 4),
            "p95": round(float(p95), 4),
            "p99": round(float(p99), 4),
            "max": round(float(arr.max()), 4),
        }


class Metrics:
    """Thread-safe counters and histograms, rendered as JSON on /metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[str, int] = defaultdict(int)
        self.histograms: dict[str, Histogram] = defaultdict(Histogram)
        self.started = time.time()

    def inc(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.counters[name] += n

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self.histograms[name].observe(value)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "uptime_s": round(time.time() - self.started, 1),
                "counters": dict(sorted(self.counters.items())),
                "latency_ms": {k: h.summary() for k, h in sorted(self.histograms.items())},
            }


class Timer:
    """`with Timer(metrics, "tick_ms"):` records elapsed milliseconds."""

    def __init__(self, metrics: Metrics | None, name: str):
        self.metrics = metrics
        self.name = name
        self.elapsed_ms = 0.0

    def __enter__(self) -> Timer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        if self.metrics is not None:
            self.metrics.observe(self.name, self.elapsed_ms)
