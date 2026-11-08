"""ScoreUpdates to the front end over RabbitMQ (docs/deploy-wiring-plan.md, section 4).

The front-end backend consumes versioned `FeedEvent`s, not `ScoreUpdate`s, from the topic exchange
`sportlight.feeds` (Asyncharles/sportlight, `packages/shared/src/events.ts` and `schemas.ts`):

- `feed.started` announces a feed (sport, titles, the looping clip): sent first, after every reconnect
  and every `reannounce_s`, because the backend only knows feeds it has seen started recently;
- `feed.intensity` carries `score` (0 to 100, from `intensity_pct`) and `streamTime` for each non-stale
  ScoreUpdate (1 Hz per feed, plus out-of-cycle imminent updates);
- `feed.updated` flips `availability` when a feed goes stale or recovers, and carries a new `source`
  when the clip clock moves;
- `feed.ended` is never sent on an ordinary restart, so the browser keeps the feed while we come back.

Every message has the envelope (`schemaVersion` 1, `eventId`, `provider`, `feedId`, `runId`,
`runStartedAt`, `seq`, `occurredAt`) and the routing key `feed.<type>.<sportId>.<feedId>`. A run starts
when this process starts, so a restart is a newer run, which the backend accepts on `feed.started`.

Timing: the browser plays the clip at `mod(mediaOffsetSec + (now - runStartedAt), durationSec)`. With the
clip clock from `run/feeds_state.json` (`started_at`, `duration_s`, written by the ingest), setting
`mediaOffsetSec = mod(runStartedAt - started_at, duration)` makes that position equal the engine's
`t_video`, and `streamTime = started_at + loop_epoch * duration + t_video - runStartedAt`.

The forecast, heating-up flag, trend, parts and reasons have no field in the front end's schema, so they
are not sent. Scoring never waits on the broker: `offer()` only stores the latest update per feed, and
the publishing task drains it with publisher confirms, reconnecting with backoff when the broker is down.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from intensity.config import PublishConfig, read_yaml, resolve_path
from intensity.feeds import Feed
from intensity.schema import ScoreUpdate
from intensity.telemetry import Metrics

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
CLOCK_REFRESH_S = 2.0  # how often feeds_state.json is re-read


def iso(ms: float) -> str:
    """Epoch milliseconds to the ISO form the front end's zod schema accepts (UTC, with Z)."""
    return datetime.fromtimestamp(ms / 1000.0, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def base36(n: int) -> str:
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = ""
    while True:
        n, r = divmod(n, 36)
        out = digits[r] + out
        if n == 0:
            return out


@dataclass(frozen=True)
class ClipClock:
    started_at_ms: float  # wall time at which the clip was at t_video 0 in loop 0
    duration_s: float


def read_clocks(path: Path) -> dict[str, ClipClock]:
    """Each feed's clip clock from the ingest's feeds_state.json; empty if it isn't there yet."""
    try:
        feeds = json.loads(path.read_text()).get("feeds", {})
    except (OSError, ValueError, AttributeError):
        return {}
    out = {}
    for feed_id, v in feeds.items():
        try:
            if float(v["duration_s"]) > 0:
                out[feed_id] = ClipClock(float(v["started_at"]) * 1000.0, float(v["duration_s"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


class FeedEvents:
    """Builds FeedEvents for one engine run. Pure: no I/O, the time is passed in."""

    def __init__(self, catalog: dict[str, Any], feeds: list[Feed], provider: str, run_started_ms: int):
        self.catalog = catalog
        self.feeds = {f.feed_id: f for f in feeds}
        self.provider = provider
        self.run_started_ms = run_started_ms
        self.seq: dict[str, int] = {}

    def sport_id(self, feed_id: str) -> str:
        sport = self.feeds[feed_id].sport
        return str(self.catalog.get("sports", {}).get(sport, sport))

    def run_id(self, feed_id: str) -> str:
        return f"{feed_id}.{base36(self.run_started_ms)}"

    def routing_key(self, event: dict[str, Any]) -> str:
        return f"feed.{event['type'].removeprefix('feed.')}.{self.sport_id(event['feedId'])}.{event['feedId']}"

    def _envelope(self, kind: str, feed_id: str, payload: dict[str, Any], now_ms: float) -> dict[str, Any]:
        self.seq[feed_id] = self.seq.get(feed_id, 0) + 1
        return {
            "schemaVersion": SCHEMA_VERSION,
            "type": kind,
            "eventId": str(uuid.uuid4()),
            "provider": self.provider,
            "feedId": feed_id,
            "runId": self.run_id(feed_id),
            "runStartedAt": iso(self.run_started_ms),
            "seq": self.seq[feed_id],
            "occurredAt": iso(now_ms),
            "payload": payload,
        }

    def media_offset(self, clock: ClipClock) -> float:
        return round(((self.run_started_ms - clock.started_at_ms) / 1000.0) % clock.duration_s, 3)

    def source(self, feed_id: str, clock: ClipClock) -> dict[str, Any]:
        meta = self.catalog.get("feeds", {}).get(feed_id, {})
        event = self.event_info(feed_id)
        source: dict[str, Any] = {
            "kind": "looping-mp4",
            "url": f"{self.catalog.get('media_base', '/media/local/videos').rstrip('/')}/{PurePosixPath(self.feeds[feed_id].sample).name}",
            "durationSec": round(clock.duration_s, 3),
            "mediaOffsetSec": self.media_offset(clock),
            "hasAudio": bool(self.catalog.get("has_audio", True)),
        }
        if meta.get("source_url"):
            source["credits"] = [{
                "title": event["title"],
                "author": event.get("broadcaster", {}).get("name", self.provider),
                "license": str(self.catalog.get("license", "")),
                "licenseUrl": None,
                "sourceUrl": meta["source_url"],
            }]
        if self.catalog.get("attribution_notice"):
            source["attributionNotice"] = str(self.catalog["attribution_notice"])
        return source

    def event_info(self, feed_id: str) -> dict[str, Any]:
        meta = self.catalog.get("feeds", {}).get(feed_id, {})
        event = dict(meta.get("event") or {"title": feed_id})
        if meta.get("competition_id"):
            event["competitionId"] = meta["competition_id"]
        return event

    def started(self, feed_id: str, clock: ClipClock, now_ms: float) -> dict[str, Any]:
        payload = {"sportId": self.sport_id(feed_id), "event": self.event_info(feed_id), "source": self.source(feed_id, clock)}
        return self._envelope("feed.started", feed_id, payload, now_ms)

    def stream_time(self, update: ScoreUpdate, clock: ClipClock) -> float:
        moment_ms = clock.started_at_ms + (update.loop_epoch * clock.duration_s + update.t_video) * 1000.0
        return round((moment_ms - self.run_started_ms) / 1000.0, 3)

    def intensity(self, update: ScoreUpdate, clock: ClipClock, now_ms: float) -> dict[str, Any] | None:
        """None for a stale update or a moment before the run started (nothing the browser can show)."""
        if update.stale:
            return None
        stream_time = self.stream_time(update, clock)
        if stream_time < 0:
            return None
        score = min(100.0, max(0.0, float(update.intensity_pct)))
        return self._envelope("feed.intensity", update.feed_id, {"score": round(score), "streamTime": stream_time}, now_ms)

    def availability(self, feed_id: str, available: bool, now_ms: float) -> dict[str, Any]:
        return self._envelope("feed.updated", feed_id, {"availability": "available" if available else "unavailable"}, now_ms)

    def source_changed(self, feed_id: str, clock: ClipClock, now_ms: float) -> dict[str, Any]:
        return self._envelope("feed.updated", feed_id, {"source": self.source(feed_id, clock)}, now_ms)


class AmqpPublisher:
    """Publishes FeedEvents with confirms; `offer()` is safe to call from the scoring loop at any time."""

    def __init__(self, cfg: PublishConfig, url: str, feeds: list[Feed], metrics: Metrics | None = None,
                 wall_ms=lambda: time.time() * 1000.0):
        self.cfg = cfg
        self.url = url
        self.metrics = metrics or Metrics()
        self.wall_ms = wall_ms
        catalog = read_yaml(resolve_path(cfg.catalog_path)) if resolve_path(cfg.catalog_path).exists() else {}
        self.events = FeedEvents(catalog, feeds, cfg.provider, int(wall_ms()))
        self.state_path = resolve_path(cfg.feeds_state_path)
        self.latest: dict[str, ScoreUpdate] = {}  # latest-wins slot per feed: old scores are worthless
        self.lifecycle: deque[dict[str, Any]] = deque(maxlen=cfg.lifecycle_buffer)
        self.clocks: dict[str, ClipClock] = {}
        self.announced: set[str] = set()
        self.available: dict[str, bool] = {}
        self.last_live_ms: dict[str, float] = {}
        self.connected = False
        self.published = 0
        self.failures = 0
        self.last_error: str | None = None
        self._clocks_read_ms = -1e18
        self._next_announce_ms = 0.0
        self._wake: asyncio.Event | None = None
        self._missing_logged: set[str] = set()

    # ------------------------------------------------------------------ called by the scoring loop

    def offer(self, update: ScoreUpdate) -> None:
        if update.feed_id not in self.events.feeds:
            return  # unregistered feeds (clones) have no catalogue entry or clip for the browser
        self.latest[update.feed_id] = update
        if not update.stale:
            self.last_live_ms[update.feed_id] = self.wall_ms()
        if self._wake is not None:
            self._wake.set()

    def status(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "published": self.published,
            "failures": self.failures,
            "buffered_lifecycle": len(self.lifecycle),
            "announced": sorted(self.announced),
            "feeds_without_clock": sorted(set(self.events.feeds) - set(self.clocks)),
            "last_error": self.last_error,
        }

    # ------------------------------------------------------------------ what to send next (no I/O)

    def refresh_clocks(self, now_ms: float) -> None:
        if now_ms - self._clocks_read_ms < CLOCK_REFRESH_S * 1000.0:
            return
        self._clocks_read_ms = now_ms
        clocks = read_clocks(self.state_path)
        for feed_id, clock in clocks.items():
            if feed_id in self.announced and self.clocks.get(feed_id) not in (None, clock):
                self.lifecycle.append(self.events.source_changed(feed_id, clock, now_ms))
        self.clocks = {k: v for k, v in clocks.items() if k in self.events.feeds}
        for feed_id in set(self.events.feeds) - set(self.clocks) - self._missing_logged:
            log.warning("no clip clock for %s in %s yet; not publishing it", feed_id, self.state_path)
            self._missing_logged.add(feed_id)

    def plan_lifecycle(self, now_ms: float) -> None:
        """Queue announcements and availability changes due at `now_ms`."""
        if now_ms >= self._next_announce_ms:
            for feed_id, clock in sorted(self.clocks.items()):
                self.lifecycle.append(self.events.started(feed_id, clock, now_ms))
                self.announced.add(feed_id)
                self.available.setdefault(feed_id, True)
                self.last_live_ms.setdefault(feed_id, now_ms)  # a new feed gets the grace period too
            self._next_announce_ms = now_ms + self.cfg.reannounce_s * 1000.0
        limit = self.cfg.unavailable_after_s * 1000.0
        for feed_id in sorted(self.announced):
            live = now_ms - self.last_live_ms.get(feed_id, -1e18) <= limit
            if live != self.available.get(feed_id, True):
                self.available[feed_id] = live
                self.lifecycle.append(self.events.availability(feed_id, live, now_ms))

    def take_intensity(self, now_ms: float) -> list[dict[str, Any]]:
        pending, self.latest = self.latest, {}
        out = []
        for feed_id, update in pending.items():
            clock = self.clocks.get(feed_id)
            if clock is None or feed_id not in self.announced:
                continue
            event = self.events.intensity(update, clock, now_ms)
            if event is not None:
                out.append(event)
        return out

    def reset_run_state(self) -> None:
        """After a (re)connect: announce everything again before any intensity."""
        self.announced.clear()
        self._next_announce_ms = 0.0

    # ------------------------------------------------------------------ the publishing task

    async def run(self) -> None:
        import aio_pika

        self._wake = asyncio.Event()
        backoff, backoff_max = self.cfg.backoff_s
        delay = backoff
        while True:
            try:
                connection = await aio_pika.connect(self.url, timeout=10)
                async with connection:
                    channel = await connection.channel(publisher_confirms=True)
                    exchange = await channel.declare_exchange(self.cfg.exchange, aio_pika.ExchangeType.TOPIC, durable=True)
                    self.connected, delay = True, backoff
                    log.info("AMQP connected; publishing FeedEvents to %s", self.cfg.exchange)
                    self.reset_run_state()
                    await self._pump(connection, exchange)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # broker down, refused, timed out: keep scoring, retry later
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.metrics.inc("amqp.disconnects")
                log.warning("AMQP link down (%s); retrying in %.1fs", self.last_error, delay)
            finally:
                self.connected = False
            await asyncio.sleep(delay)
            delay = min(delay * 2, backoff_max)

    async def _pump(self, connection: Any, exchange: Any) -> None:
        assert self._wake is not None
        while not connection.is_closed:
            now_ms = self.wall_ms()
            self.refresh_clocks(now_ms)
            self.plan_lifecycle(now_ms)
            while self.lifecycle:  # in order, and kept until confirmed
                await self._send(exchange, self.lifecycle[0], persistent=True)
                self.lifecycle.popleft()
            for event in self.take_intensity(now_ms):
                await self._send(exchange, event, persistent=False)
            self._wake.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=1.0)

    async def _send(self, exchange: Any, event: dict[str, Any], persistent: bool) -> None:
        import aio_pika

        message = aio_pika.Message(
            body=json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode(),
            content_type="application/json",
            message_id=event["eventId"],
            timestamp=datetime.now(UTC),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT if persistent else aio_pika.DeliveryMode.NOT_PERSISTENT,
        )
        try:
            await exchange.publish(message, routing_key=self.events.routing_key(event), timeout=self.cfg.confirm_timeout_s)
        except Exception:
            self.failures += 1
            raise
        self.published += 1
        self.metrics.inc(f"amqp.{event['type']}")
