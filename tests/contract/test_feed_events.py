"""The engine's FeedEvents against the front end's contract (Asyncharles/sportlight, packages/shared/src).

`FrontEndEvent` mirrors `schemas.ts`'s zod `FeedEventSchema`, stricter in one way: unknown fields are
rejected (zod would strip them), so the engine never sends something the backend silently drops.
The end-to-end CI job checks the same messages against the real backend.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from intensity.config import PublishConfig, read_yaml, resolve_path
from intensity.engine.amqp import AmqpPublisher, ClipClock, FeedEvents, base36, iso
from intensity.feeds import load_feeds
from intensity.schema import ScoreUpdate

ISO = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?(Z|[+-]\d\d:\d\d)$")  # z.string().datetime({offset: true})
Id = Annotated[str, Field(min_length=1, max_length=128)]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Broadcaster(_M):
    name: Annotated[str, Field(min_length=1, max_length=120)]
    channel: Annotated[str, Field(max_length=120)] | None = None


class EventInfo(_M):
    title: Annotated[str, Field(min_length=1, max_length=200)]
    subtitle: Annotated[str, Field(max_length=200)] | None = None
    competitionId: Id | None = None
    teamIds: list[Id] | None = None
    venue: str | None = None
    broadcaster: Broadcaster | None = None


class Credit(_M):
    title: str
    author: str
    license: Annotated[str, Field(max_length=200)]
    licenseUrl: str | None
    sourceUrl: str


class MediaSource(_M):
    kind: Literal["looping-mp4"]
    url: Annotated[str, Field(min_length=1, max_length=2048)]
    durationSec: Annotated[float, Field(gt=0)]
    mediaOffsetSec: Annotated[float, Field(ge=0)]
    hasAudio: bool
    posterUrl: str | None = None
    credits: list[Credit] | None = None
    attributionNotice: Annotated[str, Field(max_length=2000)] | None = None


class _Envelope(_M):
    schemaVersion: Literal[1]
    eventId: Id
    provider: Id
    feedId: Id
    runId: Id
    runStartedAt: Annotated[str, Field(pattern=ISO.pattern)]
    seq: Annotated[int, Field(ge=0)]
    occurredAt: Annotated[str, Field(pattern=ISO.pattern)]


class StartedPayload(_M):
    sportId: Id
    event: EventInfo
    source: MediaSource


class Started(_Envelope):
    type: Literal["feed.started"]
    payload: StartedPayload


class IntensityPayload(_M):
    score: Annotated[float, Field(ge=0, le=100)]
    streamTime: Annotated[float, Field(ge=0)]


class Intensity(_Envelope):
    type: Literal["feed.intensity"]
    payload: IntensityPayload


class UpdatedPayload(_M):
    sportId: Id | None = None
    event: EventInfo | None = None
    source: MediaSource | None = None
    availability: Literal["available", "unavailable"] | None = None


class Updated(_Envelope):
    type: Literal["feed.updated"]
    payload: UpdatedPayload


FrontEndEvent = TypeAdapter(Annotated[Started | Intensity | Updated, Field(discriminator="type")])
SPORT_IDS = {"football", "basketball", "tennis", "motorsport", "ice-hockey", "cycling", "rugby", "athletics",
             "chess", "esports", "weightlifting"}  # packages/shared/src/catalogue.ts


def update(feed_id: str, t_video: float, epoch: int = 0, pct: float = 63.4, stale: bool = False,
           sport: str = "football") -> ScoreUpdate:
    sample = json.loads((resolve_path("contracts/samples") / "score_update_football.json").read_text())
    return ScoreUpdate.model_validate({**sample, "feed_id": feed_id, "sport": sport, "t_video": t_video,
                                       "loop_epoch": epoch, "intensity_pct": pct, "stale": stale})


@pytest.fixture
def events() -> FeedEvents:
    feeds = load_feeds(resolve_path("config/feeds.yaml")).feeds
    return FeedEvents(read_yaml(resolve_path("config/publish.yaml")), feeds, "sportlight-engine", 1_794_000_000_000)


def test_every_demo_feed_announces_in_the_front_end_format(events: FeedEvents) -> None:
    clock = ClipClock(started_at_ms=1_793_999_000_000, duration_s=4660.22)
    for feed_id in events.feeds:
        ev = events.started(feed_id, clock, 1_794_000_000_500)
        FrontEndEvent.validate_python(ev)
        assert ev["payload"]["sportId"] in SPORT_IDS
        assert ev["payload"]["source"]["url"].startswith("/media/local/videos/")
        assert events.routing_key(ev) == f"feed.started.{ev['payload']['sportId']}.{feed_id}"
    assert {events.sport_id(f) for f in events.feeds} == {"football", "motorsport", "basketball", "weightlifting"}


def test_catalog_matches_the_front_ends_local_sources() -> None:
    """config/publish.yaml must name the same files the front end's media service serves."""
    feeds = load_feeds(resolve_path("config/feeds.yaml")).feeds
    catalog = read_yaml(resolve_path("config/publish.yaml"))
    assert set(catalog["feeds"]) == {f.feed_id for f in feeds}
    expected = {  # sportlight media/local-sources.json: id -> file, sportId
        "fb-arg-fra": ("fb_arg_fra_2022.mp4", "football"),
        "f1-demo-race": ("f1_2019_brazilian_gp.mp4", "motorsport"),
        "nba-demo-game": ("nba_20160619_cle_gsw.mp4", "basketball"),
        "wl-demo-session": ("wl_worlds2025_m79.mp4", "weightlifting"),
    }
    for f in feeds:
        assert (f.sample.split("/")[-1], catalog["sports"][f.sport]) == expected[f.feed_id]


def test_browser_position_equals_engine_t_video(events: FeedEvents) -> None:
    """The browser plays mod(mediaOffsetSec + (now - runStartedAt), durationSec): it must be t_video."""
    clock = ClipClock(started_at_ms=1_793_998_765_432, duration_s=600.0)
    offset = events.started("fb-arg-fra", clock, 1_794_000_000_000)["payload"]["source"]["mediaOffsetSec"]
    for now_ms in (1_794_000_000_000, 1_794_000_123_400, 1_794_000_999_999):
        elapsed = (now_ms - clock.started_at_ms) / 1000.0
        epoch, t_video = divmod(elapsed, clock.duration_s)
        browser = (offset + (now_ms - events.run_started_ms) / 1000.0) % clock.duration_s
        assert browser == pytest.approx(t_video, abs=2e-3)
        ev = events.intensity(update("fb-arg-fra", t_video, int(epoch)), clock, now_ms)
        assert ev["payload"]["streamTime"] == pytest.approx((now_ms - events.run_started_ms) / 1000.0, abs=2e-3)


def test_intensity_envelope_seq_and_skips(events: FeedEvents) -> None:
    clock = ClipClock(started_at_ms=events.run_started_ms - 10_000, duration_s=100.0)
    first = events.intensity(update("fb-arg-fra", 12.0, pct=63.4), clock, events.run_started_ms + 2_000)
    second = events.intensity(update("fb-arg-fra", 13.0, pct=101.0), clock, events.run_started_ms + 3_000)
    for ev in (first, second):
        FrontEndEvent.validate_python(ev)
    assert first["payload"] == {"score": 63, "streamTime": 2.0} and second["payload"]["score"] == 100
    assert second["seq"] == first["seq"] + 1 and first["runId"] == f"fb-arg-fra.{base36(events.run_started_ms)}"
    assert events.routing_key(first) == "feed.intensity.football.fb-arg-fra"
    assert events.intensity(update("fb-arg-fra", 12.0, stale=True), clock, events.run_started_ms) is None
    assert events.intensity(update("fb-arg-fra", 5.0), clock, events.run_started_ms) is None  # before the run
    assert iso(0) == "1970-01-01T00:00:00.000Z"
    FrontEndEvent.validate_python(events.availability("fb-arg-fra", False, events.run_started_ms))
    FrontEndEvent.validate_python(events.source_changed("fb-arg-fra", clock, events.run_started_ms))


def test_publisher_announces_first_flags_staleness_and_reannounces(tmp_path) -> None:
    state = tmp_path / "feeds_state.json"
    state.write_text(json.dumps({"feeds": {"fb-arg-fra": {"started_at": 1000.0, "duration_s": 600.0},
                                           "nba-demo-game": {"started_at": 1000.0, "duration_s": 900.0}}}))
    now = {"ms": 1_000_000.0}
    cfg = PublishConfig(feeds_state_path=str(state), reannounce_s=10.0, unavailable_after_s=5.0, lifecycle_buffer=50)
    pub = AmqpPublisher(cfg, "amqp://unused", load_feeds(resolve_path("config/feeds.yaml")).feeds, wall_ms=lambda: now["ms"])

    pub.offer(update("fb-arg-fra", 10.0))
    pub.refresh_clocks(now["ms"])
    assert pub.status()["feeds_without_clock"] == ["f1-demo-race", "wl-demo-session"]
    pub.plan_lifecycle(now["ms"])
    assert [e["type"] for e in pub.lifecycle] == ["feed.started", "feed.started"]  # before any intensity
    intensity = pub.take_intensity(now["ms"])
    assert [e["feedId"] for e in intensity] == ["fb-arg-fra"] and pub.latest == {}
    seqs = [e["seq"] for e in pub.lifecycle if e["feedId"] == "fb-arg-fra"] + [intensity[0]["seq"]]
    assert seqs == sorted(seqs)
    pub.lifecycle.clear()

    now["ms"] += 6_000  # no live score for 6 s: both feeds unavailable
    pub.plan_lifecycle(now["ms"])
    assert sorted((e["feedId"], e["payload"]["availability"]) for e in pub.lifecycle) == [
        ("fb-arg-fra", "unavailable"), ("nba-demo-game", "unavailable")]
    pub.lifecycle.clear()
    pub.offer(update("fb-arg-fra", 16.0))
    now["ms"] += 5_000  # and 11 s after the first announcement: announce again
    pub.plan_lifecycle(now["ms"])
    kinds = [(e["type"], e["feedId"], e["payload"].get("availability")) for e in pub.lifecycle]
    assert ("feed.updated", "fb-arg-fra", "available") in kinds and kinds.count(("feed.started", "fb-arg-fra", None)) == 1
    pub.lifecycle.clear()

    state.write_text(json.dumps({"feeds": {"fb-arg-fra": {"started_at": 2000.0, "duration_s": 600.0},
                                           "nba-demo-game": {"started_at": 1000.0, "duration_s": 900.0}}}))
    now["ms"] += 3_000
    pub.refresh_clocks(now["ms"])  # the ingest restarted without its state: the browser must re-seek
    assert [(e["type"], "source" in e["payload"]) for e in pub.lifecycle] == [("feed.updated", True)]

    pub.reset_run_state()  # a reconnect: everything is announced again before intensity
    pub.offer(update("fb-arg-fra", 30.0))
    assert pub.take_intensity(now["ms"]) == []
    for e in [*pub.lifecycle]:
        FrontEndEvent.validate_python(e)


def test_unregistered_feeds_and_a_full_buffer_never_raise(tmp_path) -> None:
    cfg = PublishConfig(feeds_state_path=str(tmp_path / "missing.json"), lifecycle_buffer=3)
    pub = AmqpPublisher(cfg, "amqp://unused", load_feeds(resolve_path("config/feeds.yaml")).feeds)
    pub.offer(update("clone-7", 1.0))
    assert pub.latest == {}
    for _ in range(10):
        pub.lifecycle.append({"type": "feed.updated"})
    assert len(pub.lifecycle) == 3


class _FakeExchange:
    def __init__(self, sent: list, fail_on: set[int]):
        self.sent, self.fail_on = sent, fail_on

    async def publish(self, message, routing_key: str, timeout: float):
        if len(self.sent) in self.fail_on:
            self.fail_on.discard(len(self.sent))
            raise ConnectionError("broker went away")
        self.sent.append((routing_key, message))


class _FakeConnection:
    def __init__(self, exchange: _FakeExchange):
        self.exchange, self.is_closed = exchange, False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.is_closed = True

    async def channel(self, publisher_confirms: bool):
        assert publisher_confirms
        return self

    async def declare_exchange(self, name: str, kind, durable: bool):
        assert (name, str(kind.value), durable) == ("sportlight.feeds", "topic", True)
        return self.exchange


def test_publisher_task_confirms_in_order_and_recovers(tmp_path, monkeypatch) -> None:
    import asyncio

    import aio_pika

    state = tmp_path / "feeds_state.json"
    state.write_text(json.dumps({"feeds": {"fb-arg-fra": {"started_at": 1000.0, "duration_s": 600.0}}}))
    sent: list = []
    exchange = _FakeExchange(sent, fail_on={2})  # the 3rd publish fails: the link drops once
    attempts = {"n": 0}

    async def fake_connect(url: str, timeout: float):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ConnectionRefusedError("broker not up yet")
        return _FakeConnection(exchange)

    monkeypatch.setattr(aio_pika, "connect", fake_connect)
    cfg = PublishConfig(feeds_state_path=str(state), backoff_s=(0.01, 0.02))
    pub = AmqpPublisher(cfg, "amqp://unused", load_feeds(resolve_path("config/feeds.yaml")).feeds,
                        wall_ms=lambda: 1_020_000.0)

    async def scenario() -> None:
        task = asyncio.create_task(pub.run())
        for t in range(6):
            pub.offer(update("fb-arg-fra", 20.0 + t))  # never awaits: scoring can't block on the broker
            await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert attempts["n"] >= 3  # refused, connected, dropped by the failed publish, reconnected
    kinds = [(key.split(".")[1], json.loads(m.body)["runId"]) for key, m in sent]
    assert kinds[0][0] == "started" and "started" in [k for k, _ in kinds[1:]]  # re-announced after the drop
    assert len({run for _, run in kinds}) == 1  # one run per process, across reconnects
    first_intensity = next(i for i, (k, _) in enumerate(kinds) if k == "intensity")
    assert any(k == "started" for k, _ in kinds[:first_intensity])
    for key, m in sent:
        ev = json.loads(m.body)
        FrontEndEvent.validate_python(ev)
        assert m.content_type == "application/json" and m.message_id == ev["eventId"]
        persistent = m.delivery_mode == aio_pika.DeliveryMode.PERSISTENT
        assert persistent == (ev["type"] != "feed.intensity")
        assert key == f"feed.{ev['type'][5:]}.football.fb-arg-fra"
    assert pub.status()["published"] == len(sent) and pub.status()["failures"] == 1
