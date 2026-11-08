"""The scenarios behind the engine performance report. Each runs a private engine; none touches the live one.

* `startup`: config, model bundles, match records and engine construction, then a restart mid-clip (the
  pre-roll and the first score after it).
* `demo`: the four demo feeds as the VM serves them (clock lane plus replayed match data), on a simulated
  clock, with every stage of every tick timed. Ingest is timed as the service does it: the clock lane's
  JSON, `parse_envelope`, `Engine.handle`; publishing as `Broadcaster.publish` plus the AMQP FeedEvent.
* `perception`: mock footage with all six sources, a stand-in for the trackers' load until perception exists.
* `scaling`: the demo feeds cloned up to N feeds, each at its own clip offset: engine CPU per second and
  `score_all` latency against the feed count.
* `transport`: real time over loopback WebSockets, a private engine service on a free port: ingest to
  engine lag, and ScoreUpdate to `/ws/scores` client latency.
* `live`: only with a URL; one read of a running engine's `/metrics` and `/health`.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import math
import os
import socket
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from intensity.config import Config, load_config, read_yaml, resolve_path
from intensity.data import data_dir
from intensity.engine.amqp import ClipClock, FeedEvents
from intensity.engine.core import Engine, default_feeds
from intensity.engine.perf.probe import FAST, INGEST, TICK, Probe
from intensity.engine.registry import Registry
from intensity.engine.replay import Replayer, load_records
from intensity.engine.service import Broadcaster
from intensity.feeds import Feed
from intensity.ingest.clock_lane import ClockLane, FeedClock, probe_duration
from intensity.schema import ClockTick, Envelope, Event, ScoreUpdate, make_envelope, parse_envelope
from intensity.sync import SyncTable
from intensity.telemetry import Metrics

log = logging.getLogger(__name__)

PERF_CONFIG = "config/experiments/engine_perf.yaml"
# Never publish to the front end or call the LLM from a benchmark, whatever the overlay says.
FORCED = ("engine.publish.enabled=false", "engine.polish.enabled=false")
T0 = 1_800_000_000.0  # virtual wall clock (epoch s) of simulated runs, so they are reproducible
GOLDEN = 0.6180339887  # spreads clone start offsets over the clip


def load_perf(path: str | Path = PERF_CONFIG) -> dict[str, Any]:
    return read_yaml(resolve_path(path))["perf"]


def harness_config(overlays: list[str], overrides: list[str] = ()) -> Config:
    return load_config(overlays, [*overrides, *FORCED])


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 2**20


def pct(values: Iterable[float], qs: tuple[float, ...] = (50, 95, 99)) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if not arr.size:
        return {}
    out = {f"p{q:g}": round(float(v), 4) for q, v in zip(qs, np.percentile(arr, qs), strict=True)}
    out.update(mean=round(float(arr.mean()), 4), max=round(float(arr.max()), 4), count=int(arr.size))
    return out


# --------------------------------------------------------------------------- clips


@dataclass
class Clip:
    feed: Feed
    duration_s: float
    duration_from: str  # mp4 | sync | records | default


def clip_duration(config: Config, feed: Feed) -> tuple[float, str]:
    """The clip's length: from the mp4 if it is here, else the sync table's end, else the last record."""
    mp4 = resolve_path("data") / feed.sample
    if mp4.exists():
        try:
            return round(probe_duration(mp4), 3), "mp4"
        except Exception:  # fall back to the data
            log.warning("cannot probe %s; using its data for the duration", mp4)
    if feed.sync and resolve_path(feed.sync).exists():
        return max(r.video_end for r in SyncTable.load(resolve_path(feed.sync)).rows), "sync"
    path = resolve_path(config.engine.replay.events_dir) / feed.sport / f"{feed.match_ref.match_id}.jsonl"
    if path.exists():
        records = load_records(path)
        if records:
            return max(r.game_time_s for r in records) + 60.0, "records"
    return 600.0, "default"


def demo_clips(config: Config) -> list[Clip]:
    return [Clip(f, *clip_duration(config, f)) for f in default_feeds(config)]


def lane_arrivals(clocks: list[FeedClock], tick_hz: float, until_s: float) -> Iterator[tuple[float, Envelope]]:
    """The clock lane's envelopes at engine time t (wall time T0 + t), as the ingest would send them."""
    lane = ClockLane(clocks)
    k = 0
    while (t := k / tick_hz) <= until_s + 1e-9:
        for env in lane.tick(T0 + t):
            yield t, env
        k += 1


def clocks_for(clips: list[Clip], offsets: list[float] | None = None, feeds: list[Feed] | None = None) -> list[FeedClock]:
    feeds = feeds or [c.feed for c in clips]
    offsets = offsets or [0.0] * len(feeds)
    by_sample = {c.feed.sample: c.duration_s for c in clips}
    return [FeedClock(f.feed_id, f.sport, f.sample, T0 - off, by_sample[f.sample]) for f, off in zip(feeds, offsets, strict=True)]


# --------------------------------------------------------------------------- the simulated driver


@dataclass
class Volume:
    """What crossed the engine's edges: messages and bytes in by kind, ScoreUpdates and bytes out."""

    msgs_in: Counter[str] = field(default_factory=Counter)
    bytes_in: Counter[str] = field(default_factory=Counter)
    updates: Counter[str] = field(default_factory=Counter)  # per feed
    out_of_cycle: Counter[str] = field(default_factory=Counter)
    stale: int = 0
    bytes_ws: int = 0
    amqp_events: int = 0
    bytes_amqp: int = 0

    def as_dict(self, seconds: float) -> dict[str, Any]:
        n = sum(self.updates.values())
        return {
            "simulated_s": round(seconds, 1),
            "msgs_in": dict(sorted(self.msgs_in.items())),
            "bytes_in": dict(sorted(self.bytes_in.items())),
            "msgs_in_per_s": {k: round(v / seconds, 3) for k, v in sorted(self.msgs_in.items())},
            "bytes_in_per_s": round(sum(self.bytes_in.values()) / seconds, 1),
            "updates": n,
            "updates_per_feed": dict(sorted(self.updates.items())),
            "out_of_cycle": sum(self.out_of_cycle.values()),
            "stale_updates": self.stale,
            "bytes_ws": self.bytes_ws,
            "bytes_ws_per_update": round(self.bytes_ws / n, 1) if n else None,
            "bytes_ws_per_s": round(self.bytes_ws / seconds, 1),
            "amqp_events": self.amqp_events,
            "bytes_amqp_per_event": round(self.bytes_amqp / self.amqp_events, 1) if self.amqp_events else None,
        }


class Publisher:
    """Does per update what the service does: `Broadcaster.publish` (JSON for /ws/scores) and the AMQP
    FeedEvent for registered feeds (built and serialised, never sent)."""

    def __init__(self, config: Config, feeds: list[Feed], clocks: list[FeedClock], probe: Probe, volume: Volume):
        self.broadcaster = Broadcaster(Metrics())
        pub = config.engine.publish
        catalog = read_yaml(resolve_path(pub.catalog_path)) if resolve_path(pub.catalog_path).exists() else {}
        self.events = FeedEvents(catalog, feeds, pub.provider, int(T0 * 1000))
        self.clocks = {c.feed_id: ClipClock(c.started_at * 1000.0, c.duration_s) for c in clocks}
        self.probe = probe
        self.volume = volume

    def __call__(self, update: ScoreUpdate, now: float, out_of_cycle: bool) -> None:
        v, fid = self.volume, update.feed_id
        v.updates[fid] += 1
        v.out_of_cycle[fid] += out_of_cycle
        v.stale += update.stale
        self.probe.measure("publish.ws", fid, self.broadcaster.publish, update, ctx=TICK)
        v.bytes_ws += len(self.broadcaster.latest[fid])
        clock = self.clocks.get(fid)
        if clock is None or fid not in self.events.feeds:
            return  # the AMQP publisher skips unregistered feeds (clones, mock ids)
        t0 = time.perf_counter_ns()
        event = self.events.intensity(update, clock, (T0 + now) * 1000.0)
        body = json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode() if event else b""
        self.probe.record("publish.amqp", fid, (time.perf_counter_ns() - t0) / 1e6, TICK)
        if event:
            v.amqp_events += 1
            v.bytes_amqp += len(body)


def drive(engine: Engine, probe: Probe, arrivals: Iterable[tuple[float, Envelope]], duration_s: float,
          volume: Volume, publish: Callable[[ScoreUpdate, float, bool], None] | None = None) -> None:
    """Replay `arrivals` (time-ordered) through `engine` on a simulated clock, with the service's fast and
    score ticks. Each envelope goes through JSON both ways, as it does between the ingest and `/ingest`."""
    fast = engine.cfg.service.fast_tick_s
    per_score = max(1, round(engine.cfg.service.score_interval_s / fast))
    it = iter(arrivals)
    nxt = next(it, None)
    step = 0
    while (now := step * fast) <= duration_s + 1e-9:
        while nxt is not None and nxt[0] <= now + 1e-9:
            t, env = nxt
            probe.now = t
            fid = env.data.feed_id
            key = f"event.{env.data.source}" if isinstance(env.data, Event) else (
                f"control.{env.data.kind}" if env.kind == "control" else env.kind)
            raw = probe.measure("ingest.serialize", fid, env.model_dump_json, ctx=INGEST)
            volume.msgs_in[key] += 1
            volume.bytes_in[key] += len(raw)
            engine.handle(probe.measure("ingest.parse", fid, parse_envelope, raw, ctx=INGEST), t)
            nxt = next(it, None)
        probe.now = now
        t0 = time.perf_counter_ns()
        for update in engine.tick_imminent(now):
            if publish:
                publish(update, now, True)
            # What the service reports as imminent_to_ws_ms: from the fast tick to the update handed out.
            probe.record("imminent_to_publish", update.feed_id, (time.perf_counter_ns() - t0) / 1e6, FAST)
        if step % per_score == 0:
            for update in engine.score_all(now):
                if publish:
                    publish(update, now, False)
        step += 1


def cpu_s() -> float:
    t = psutil.Process().cpu_times()
    return t.user + t.system


# --------------------------------------------------------------------------- scenarios


def run_startup(config: Config, overlays: list[str], clips: list[Clip], restart_at_s: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    t0 = time.perf_counter()
    harness_config(overlays)
    out["config_load_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    t0 = time.perf_counter()
    registry = Registry(config.engine.forecast.models_dir, config.engine.forecast.shadow)
    versions = registry.reload()
    out["registry_reload_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    out["bundles"] = versions

    feeds = []
    for c in clips:
        f = c.feed
        path = resolve_path(config.engine.replay.events_dir) / f.sport / f"{f.match_ref.match_id}.jsonl"
        row: dict[str, Any] = {"feed_id": f.feed_id, "sport": f.sport, "clip_s": c.duration_s,
                               "clip_s_from": c.duration_from, "events_file": str(path.relative_to(data_dir()))
                               if path.is_relative_to(data_dir()) else str(path)}
        if path.exists():
            row["file_bytes"] = path.stat().st_size
            t0 = time.perf_counter()
            records = load_records(path)
            row["load_records_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            row["records"] = len(records)
            sync = SyncTable.load(resolve_path(f.sync)) if f.sync and resolve_path(f.sync).exists() else SyncTable.identity(1e7)
            t0 = time.perf_counter()
            replay = Replayer(f.feed_id, f.sport, records, sync)
            row["build_replay_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            row["records_in_clip"] = sum(1 for t, _ in replay.timeline if 0 < t <= c.duration_s)
        feeds.append(row)

    rss0 = rss_mb()
    t0 = time.perf_counter()
    engine = Engine(config, [c.feed for c in clips])
    out["engine_init_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    out["engine_init_rss_mb"] = round(rss_mb() - rss0, 1)

    # A restart mid-clip: the first ClockTick pre-rolls the sport's percentile window, the first score
    # then replays every record before that point.
    pcfg = config.engine.percentile
    for c, row in zip(clips, feeds, strict=True):
        t_restart = round(min(restart_at_s, 0.6 * c.duration_s), 1)
        env = make_envelope(ClockTick(feed_id=c.feed.feed_id, t_video=t_restart, loop_epoch=0, t_emit_ns=0))
        t0 = time.perf_counter()
        engine.handle(env, 0.0)
        row["restart_at_s"] = t_restart
        row["preroll_s"] = round(min(pcfg.preroll_s, pcfg.window_s, t_restart), 1)
        row["first_clock_tick_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        fs = engine.feeds[c.feed.feed_id]
        t0 = time.perf_counter()
        engine.score_feed(fs, 0.0)
        row["first_score_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    out["feeds"] = feeds
    t0 = time.perf_counter()
    engine.score_all(1.0)
    out["second_score_all_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return out


def run_demo(config: Config, clips: list[Clip], duration_s: float | None) -> tuple[dict[str, Any], pd.DataFrame]:
    duration = duration_s or max(c.duration_s for c in clips)
    engine = Engine(config, [c.feed for c in clips])
    clocks = clocks_for(clips)
    volume = Volume()
    with Probe("demo") as probe:
        probe.instrument(engine)
        publish = Publisher(config, [c.feed for c in clips], clocks, probe, volume)
        tick_hz = float(config.ingest["clock"]["tick_hz"])
        cpu0, wall0, rss0 = cpu_s(), time.perf_counter(), rss_mb()
        drive(engine, probe, lane_arrivals(clocks, tick_hz, duration), duration, volume, publish)
        cpu, wall = cpu_s() - cpu0, time.perf_counter() - wall0
        frame = probe.frame()
        counts = dict(probe.counts)
    summary = {
        "simulated_s": duration,
        "wall_s": round(wall, 2),
        "cpu_s": round(cpu, 2),
        "speedup": round(duration / wall, 1),
        "rss_growth_mb": round(rss_mb() - rss0, 1),
        "loops": {fid: fs.loops for fid, fs in engine.feeds.items()},
        "replayed_records": counts.get("replay.records", 0),
        "volume": volume.as_dict(duration),
        "clips": [{"feed_id": c.feed.feed_id, "sport": c.feed.sport, "clip_s": c.duration_s, "from": c.duration_from} for c in clips],
    }
    return summary, frame


def run_perception(config: Config, clips: list[Clip], minutes: float) -> tuple[dict[str, Any], pd.DataFrame]:
    from intensity.engine.mock import MockGenerator

    duration = minutes * 60.0
    feeds = [(f"mock-{c.feed.sport}", c.feed.sport) for c in clips]
    gen = MockGenerator(feeds, duration_s=duration + 60.0, t0_ns=0)
    envs = gen.envelopes(0.0, duration)
    engine = Engine(config, [])
    volume = Volume()
    with Probe("perception") as probe:
        probe.instrument(engine)
        publish = Publisher(config, [], [], probe, volume)
        cpu0, wall0 = cpu_s(), time.perf_counter()
        drive(engine, probe, ((e.data.t_emit_ns / 1e9, e) for e in envs), duration, volume, publish)
        cpu, wall = cpu_s() - cpu0, time.perf_counter() - wall0
        frame = probe.frame()
    return {"simulated_s": duration, "wall_s": round(wall, 2), "cpu_s": round(cpu, 2),
            "speedup": round(duration / wall, 1), "volume": volume.as_dict(duration)}, frame


def run_scaling(config: Config, clips: list[Clip], counts: list[int], warmup_s: float,
                duration_s: float) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    rows, frames = [], []
    tick_hz = float(config.ingest["clock"]["tick_hz"])
    base = [c.feed for c in clips]
    for n in counts:
        feeds, offsets = [], []
        for i in range(n):
            f, k = base[i % len(base)], i // len(base)
            feeds.append(f if k == 0 else f.model_copy(update={"feed_id": f"{f.feed_id}~{k}"}))
            dur = next(c.duration_s for c in clips if c.feed.sample == f.sample)
            offsets.append(round(((k * GOLDEN + 0.1) % 1.0) * max(0.0, dur - warmup_s - duration_s - 5.0), 1))
        gc.collect()
        rss0 = rss_mb()
        t0 = time.perf_counter()
        engine = Engine(config, feeds)
        init_ms = (time.perf_counter() - t0) * 1000
        engine.allow_preroll = False  # the restart cost is measured once, in `startup`
        clocks = clocks_for(clips, offsets, feeds)
        until = warmup_s + duration_s
        with Probe(f"scaling-{n}") as probe:
            probe.instrument(engine, deep=False)
            publish = Publisher(config, feeds, clocks, probe, Volume())
            cpu0, wall0 = cpu_s(), time.perf_counter()
            drive(engine, probe, lane_arrivals(clocks, tick_hz, until), until, Volume(), publish)
            wall, cpu = time.perf_counter() - wall0, cpu_s() - cpu0
            df = probe.frame()
        df = df[df["now"] >= warmup_s]
        # The engine process's work: parse and handle each envelope, the fast and score ticks, publishing.
        stage = df["stage"].astype(str)
        busy = df[stage.isin(["ingest.parse", "score_all", "fast_tick", "publish.ws", "publish.amqp"]) | stage.str.startswith("handle.")]
        score_all = df.loc[df["stage"] == "score_all", "ms"]
        ticks = df.loc[df["stage"] == "tick", "ms"]
        rows.append({
            "feeds": n,
            "engine_init_ms": round(init_ms, 1),
            "busy_ms_per_s": round(float(busy["ms"].sum()) / duration_s, 2),
            "score_all": pct(score_all),
            "tick": pct(ticks),
            "fast_tick": pct(df.loc[df["stage"] == "fast_tick", "ms"]),
            "rss_mb": round(rss_mb(), 1),
            "rss_growth_mb": round(rss_mb() - rss0, 1),
            "wall_s": round(wall, 2),
            "cpu_s": round(cpu, 2),
        })
        per_step = df[df["stage"] == "score_all"][["now", "ms"]].assign(feeds=n)
        frames.append(per_step)
        del engine
    return rows, pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def max_feeds(rows: list[dict[str, Any]], budget_ms_per_s: float = 1000.0) -> dict[str, float] | None:
    """Least-squares engine busy time per second against feeds: how many feeds fill one core."""
    if len(rows) < 2:
        return None
    x = np.array([r["feeds"] for r in rows], dtype=float)
    y = np.array([r["busy_ms_per_s"] for r in rows], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    if slope <= 0:
        return None
    return {"ms_per_feed_per_s": round(float(slope), 3), "fixed_ms_per_s": round(float(intercept), 3),
            "feeds_per_core": int((budget_ms_per_s - intercept) / slope),
            "feeds_at_half_core": int((budget_ms_per_s / 2 - intercept) / slope)}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_transport(config: Config, clips: list[Clip], seconds: float) -> tuple[dict[str, Any], pd.DataFrame]:
    """A private engine service on a free 127.0.0.1 port, fed by the clock lane over the real transport."""
    import uvicorn
    from websockets.sync.client import connect

    from intensity.engine.service import create_app
    from intensity.transport import IngestClient

    feeds = [c.feed for c in clips]
    app = create_app(config, feeds=feeds, run_loops=True)
    service = app.state.service
    samples: list[tuple[str, str, float, float]] = []  # (kind, feed, t, ms)
    handle = service.engine.handle

    def timed_handle(env: Envelope, now: float) -> None:
        samples.append(("ingest_to_engine", env.data.feed_id, time.time(), (time.time_ns() - env.data.t_emit_ns) / 1e6))
        handle(env, now)

    service.engine.handle = timed_handle  # the private service only
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True, name="perf-engine").start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("private engine service did not start")

    client = IngestClient(f"ws://127.0.0.1:{port}/ingest", maxsize=int(config.ingest["clock"].get("send_queue", 64)))
    lane = ClockLane([FeedClock(c.feed.feed_id, c.feed.sport, c.feed.sample, time.time(), c.duration_s) for c in clips])
    tick_hz = float(config.ingest["clock"]["tick_hz"])
    stop = threading.Event()

    async def lane_loop() -> None:
        sender = asyncio.create_task(client.run())
        period, next_at = 1.0 / tick_hz, time.monotonic()
        while not stop.is_set():
            for env in lane.tick(time.time()):
                client.put(env)
            next_at += period
            await asyncio.sleep(max(0.0, next_at - time.monotonic()))
        client.stop()
        await sender

    lane_thread = threading.Thread(target=lambda: asyncio.run(lane_loop()), daemon=True, name="perf-lane")
    lane_thread.start()
    bytes_ws = 0
    try:
        with connect(f"ws://127.0.0.1:{port}/ws/scores", open_timeout=5) as ws:
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                try:
                    raw = ws.recv(timeout=0.5)
                except TimeoutError:
                    continue
                got = time.time_ns()
                msg = json.loads(raw)
                bytes_ws += len(raw)
                samples.append(("score_to_ws_client", msg["feed_id"], got / 1e9, (got - msg["t_emit_ns"]) / 1e6))
    finally:
        stop.set()
        lane_thread.join(timeout=5)
        server.should_exit = True
        snap = service.metrics.snapshot()
    df = pd.DataFrame(samples, columns=["kind", "feed_id", "t", "ms"])
    first_ws = df.loc[df["kind"] == "score_to_ws_client", "t"].min() if len(df) else math.nan
    # The first updates queue behind the WebSocket handshake; keep the steady state.
    df = df[(df["kind"] != "score_to_ws_client") | (df["t"] >= first_ws + 1.0)]
    summary = {
        "seconds": seconds,
        "feeds": len(feeds),
        "ingest_to_engine_ms": pct(df.loc[df["kind"] == "ingest_to_engine", "ms"]),
        "score_to_ws_client_ms": pct(df.loc[df["kind"] == "score_to_ws_client", "ms"]),
        "sent": client.sent,
        "dropped": client.queue.dropped,
        "bytes_ws": bytes_ws,
        "service_metrics": snap,
        "note": "engine, ingest client and score client share one process here; on the VM they are separate",
    }
    return summary, df


def read_live(url: str, timeout_s: float = 3.0) -> dict[str, Any]:
    """One GET each of a running engine's /metrics and /health: counters only, nothing written."""
    import httpx

    out: dict[str, Any] = {"url": url}
    for name in ("metrics", "health"):
        try:
            r = httpx.get(f"{url.rstrip('/')}/{name}", timeout=timeout_s)
            r.raise_for_status()
            out[name] = r.json()
        except Exception as exc:  # unreachable is a result, not a failure of the report
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def inventory(config: Config) -> dict[str, Any]:
    """Data on disk the engine can serve from: event files per sport, bundles per sport."""
    events = resolve_path(config.engine.replay.events_dir)
    sports = {}
    for d in sorted(p for p in events.iterdir() if p.is_dir()) if events.exists() else []:
        files = list(d.glob("*.jsonl"))
        sports[d.name] = {"files": len(files), "bytes": sum(f.stat().st_size for f in files)}
    models = resolve_path(config.engine.forecast.models_dir)
    bundles = {}
    for d in sorted(p for p in models.iterdir() if p.is_dir()) if models.exists() else []:
        current = d / "CURRENT"
        bundles[d.name] = {"versions": sorted(p.name for p in d.iterdir() if p.is_dir()),
                           "current": current.read_text().strip() if current.exists() else None}
    return {"events": sports, "bundles": bundles, "data_dir": str(data_dir())}


def environment() -> dict[str, Any]:
    import platform
    import subprocess

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=resolve_path("."), timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        sha = None
    freq = psutil.cpu_freq()
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpus_logical": psutil.cpu_count(),
        "cpus_physical": psutil.cpu_count(logical=False),
        "cpu_mhz_max": round(freq.max) if freq and freq.max else None,
        "memory_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "python": platform.python_version(),
        "git_sha": sha,
        "niceness": os.nice(0),
    }
