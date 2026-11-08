"""Per-stage timers set on one private engine's instances. The engine code itself is unchanged.

`Probe.instrument(engine)` replaces methods on that engine's own objects (the engine, each feed's adapter,
trackers, forecaster, imminent state, smoothing and the sport percentiles) with timed wrappers, and swaps
the two module functions `_score` calls (`combine`, `build_reasons`) until `close()`. Every call becomes one
sample: stage, feed, context and the engine time it ran at. The context says what the engine was doing:
`ingest` (handling an envelope), `fast` (the imminent fast tick) or `tick` (scoring a feed); a sample's
stage is top-level unless it is listed in NESTED.
"""

from __future__ import annotations

import time
from array import array
from collections import Counter
from collections.abc import Callable
from typing import Any

import pandas as pd

from intensity.engine import core
from intensity.engine.core import Engine
from intensity.engine.feedstate import FeedState
from intensity.schema import Envelope, Event

CONTEXTS = ("ingest", "fast", "tick")
INGEST, FAST, TICK = range(3)

# Stages that run inside another timed stage, so they are left out of per-tick sums.
NESTED = frozenset({"adapter.on_match", "preroll"})

# A tick's top-level stages, grouped for the breakdown chart; what they leave over is "other"
# (combine, EMA and trend, the forecast and imminent parts, building the ScoreUpdate).
TICK_GROUPS: dict[str, tuple[str, ...]] = {
    "replay": ("replay.release",),
    "adapter": ("adapter.parts", "adapter.facts"),
    "forecast": ("forecast",),
    "imminent": ("imminent",),
    "trackers": ("trackers.crowd", "trackers.commentary", "trackers.video"),
    "percentile": ("percentile.add", "percentile.rank"),
    "reasons": ("reasons",),
}


class Probe:
    def __init__(self, scenario: str):
        self.scenario = scenario
        self.now = 0.0  # engine time, set by the driver before each step
        self.ctx = INGEST
        self.feed = ""
        self.paused = False  # during a pre-roll: its shadow engine shares the patched module functions
        self.counts: Counter[str] = Counter()
        self._stages: dict[str, int] = {}
        self._feeds: dict[str, int] = {}
        self._stage = array("H")
        self._feed = array("H")
        self._ctx = array("B")
        self._now = array("d")
        self._ms = array("d")
        self._undo: list[Callable[[], None]] = []
        self._wrapped: set[int] = set()

    # ------------------------------------------------------------------ samples

    def record(self, stage: str, feed: str, ms: float, ctx: int | None = None) -> None:
        self._stage.append(self._stages.setdefault(stage, len(self._stages)))
        self._feed.append(self._feeds.setdefault(feed, len(self._feeds)))
        self._ctx.append(self.ctx if ctx is None else ctx)
        self._now.append(self.now)
        self._ms.append(ms)

    def measure(self, stage: str, feed: str, fn: Callable[..., Any], *args: Any, ctx: int | None = None) -> Any:
        t0 = time.perf_counter_ns()
        try:
            return fn(*args)
        finally:
            self.record(stage, feed, (time.perf_counter_ns() - t0) / 1e6, ctx)

    def __len__(self) -> int:
        return len(self._ms)

    def frame(self) -> pd.DataFrame:
        stages = sorted(self._stages, key=self._stages.get)
        feeds = sorted(self._feeds, key=self._feeds.get)
        return pd.DataFrame({
            "scenario": self.scenario,
            "ctx": pd.Categorical.from_codes(list(self._ctx), categories=list(CONTEXTS)),
            "feed_id": pd.Categorical.from_codes(list(self._feed), categories=feeds),
            "stage": pd.Categorical.from_codes(list(self._stage), categories=stages),
            "now": list(self._now),
            "ms": list(self._ms),
        })

    # ------------------------------------------------------------------ wrapping

    def _set(self, obj: Any, attr: str, fn: Callable[..., Any]) -> None:
        setattr(obj, attr, fn)  # an instance attribute shadows the class method; deleting it restores
        self._undo.append(lambda: delattr(obj, attr))

    def wrap(self, obj: Any, attr: str, stage: str, feed: str | None = None, count: str | None = None) -> None:
        """Time `obj.attr` on this instance only. `feed` None attributes calls to the feed being scored;
        `count` adds the length of each result to that counter (records released, say)."""
        fn = getattr(obj, attr)

        def timed(*args: Any, **kwargs: Any) -> Any:
            if self.paused:
                return fn(*args, **kwargs)
            t0 = time.perf_counter_ns()
            out = fn(*args, **kwargs)
            self.record(stage, self.feed if feed is None else feed, (time.perf_counter_ns() - t0) / 1e6)
            if count is not None:
                self.counts[count] += len(out)
            return out

        self._set(obj, attr, timed)

    def patch(self, module: Any, name: str, stage: str) -> None:
        fn = getattr(module, name)

        def timed(*args: Any, **kwargs: Any) -> Any:
            if self.paused:
                return fn(*args, **kwargs)
            t0 = time.perf_counter_ns()
            try:
                return fn(*args, **kwargs)
            finally:
                self.record(stage, self.feed, (time.perf_counter_ns() - t0) / 1e6)

        setattr(module, name, timed)
        self._undo.append(lambda: setattr(module, name, fn))

    def close(self) -> None:
        """Remove every wrapper and put the module functions back."""
        while self._undo:
            self._undo.pop()()
        self._wrapped.clear()

    def __enter__(self) -> Probe:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ the engine

    def instrument(self, engine: Engine, deep: bool = True) -> None:
        """Time `engine`'s ingest, fast tick and scoring; `deep` also times every stage of each tick."""
        handle, score_feed, tick_imminent, score_all = engine.handle, engine.score_feed, engine.tick_imminent, engine.score_all

        def timed_handle(env: Envelope, now: float) -> None:
            data = env.data
            stage = f"handle.event.{data.source}" if isinstance(data, Event) else f"handle.{env.kind}"
            prev, self.ctx = self.ctx, INGEST
            try:
                self.measure(stage, data.feed_id, handle, env, now)
            finally:
                self.ctx = prev

        def timed_score_feed(fs: FeedState, now: float) -> Any:
            prev_ctx, prev_feed = self.ctx, self.feed
            self.ctx, self.feed = TICK, fs.feed_id
            try:
                return self.measure("tick", fs.feed_id, score_feed, fs, now)
            finally:
                self.ctx, self.feed = prev_ctx, prev_feed

        def timed_tick_imminent(now: float) -> Any:
            self.ctx = FAST
            return self.measure("fast_tick", "", tick_imminent, now)

        def timed_score_all(now: float) -> Any:
            self.ctx = TICK
            return self.measure("score_all", "", score_all, now)

        self._set(engine, "handle", timed_handle)
        self._set(engine, "score_feed", timed_score_feed)
        self._set(engine, "tick_imminent", timed_tick_imminent)
        self._set(engine, "score_all", timed_score_all)
        if not deep:
            return

        preroll = engine._preroll

        def timed_preroll(fs: FeedState, t_now: float, now: float) -> None:
            self.paused = True
            t0 = time.perf_counter_ns()
            try:
                preroll(fs, t_now, now)
            finally:
                self.paused = False
                self.record("preroll", fs.feed_id, (time.perf_counter_ns() - t0) / 1e6)

        self._set(engine, "_preroll", timed_preroll)
        add_feed = engine.add_feed

        def instrumented_add_feed(*args: Any, **kwargs: Any) -> FeedState:
            fs = add_feed(*args, **kwargs)
            self.instrument_feed(engine, fs)
            return fs

        self._set(engine, "add_feed", instrumented_add_feed)
        self.wrap(engine, "_forecast_part", "parts.forecast")
        self.wrap(engine, "_imminent_part", "parts.imminent")
        self.patch(core, "combine", "combine")
        self.patch(core, "build_reasons", "reasons")
        for fs in engine.feeds.values():
            self.instrument_feed(engine, fs)

    def instrument_feed(self, engine: Engine, fs: FeedState) -> None:
        if id(fs) in self._wrapped:
            return
        self._wrapped.add(id(fs))
        fid = fs.feed_id
        if fs.replay is not None:
            self.wrap(fs.replay, "release", "replay.release", fid, count="replay.records")
        self.wrap(fs.adapter, "on_match", "adapter.on_match", fid)
        self.wrap(fs.adapter, "parts", "adapter.parts", fid)
        self.wrap(fs.adapter, "facts", "adapter.facts", fid)
        self.wrap(fs.forecaster, "predict", "forecast", fid)
        self.wrap(fs.imminent, "update", "imminent", fid)
        for name in ("crowd", "commentary", "video"):
            self.wrap(getattr(fs, name), "part", f"trackers.{name}", fid)
        for name in ("crowd", "voice", "commentary", "video"):
            self.wrap(getattr(fs, name), "on_event", f"trackers.{name}.on_event", fid)
        self.wrap(fs.ema, "update", "smoothing.ema", fid)
        self.wrap(fs.trend, "update", "smoothing.trend", fid)
        pct = engine.sport_pct[fs.sport]
        if id(pct) not in self._wrapped:  # one per sport, shared by its feeds: calls go to the feed being scored
            self._wrapped.add(id(pct))
            self.wrap(pct, "add", "percentile.add")
            self.wrap(pct, "pct", "percentile.rank")


def overhead_us(calls: int = 200_000) -> float:
    """What one timed wrapper adds to a call, in microseconds (median of five runs)."""

    class Target:
        def f(self, x: int) -> int:
            return x

    raw, probed = Target(), Target()
    probe = Probe("overhead")
    probe.wrap(probed, "f", "f", "")
    runs = []
    for _ in range(5):
        t0 = time.perf_counter_ns()
        for i in range(calls):
            raw.f(i)
        t1 = time.perf_counter_ns()
        for i in range(calls):
            probed.f(i)
        t2 = time.perf_counter_ns()
        runs.append(((t2 - t1) - (t1 - t0)) / calls / 1e3)
    probe.close()
    return float(sorted(runs)[2])
