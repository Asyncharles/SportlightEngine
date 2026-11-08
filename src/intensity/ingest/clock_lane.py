"""The clock lane: each feed's looping-clip clock, without decoding video (docs/deploy-wiring-plan.md, D1).

This is the part of the ingest (I1 and I2 in the plan) that the engine and the front end need to agree
on time; perception (I3 to I6, P1 to P3) is not built yet. Each feed loops its clip from a start time
kept in `run/feeds_state.json`, the file the plan's pushers write and the front end's `feed-clock.ts`
already reads:

    {"feeds": {"<feed_id>": {"started_at": <epoch s>, "duration_s": <s>, "sample": "samples/x.mp4"}}}

so `t_video = (now - started_at) mod duration_s` and `loop_epoch = floor((now - started_at) / duration_s)`.
A start time survives restarts as long as the clip is unchanged, so neither the engine nor the browser
seeks when the ingest restarts. The lane sends a `ClockTick` per feed at `ingest.clock.tick_hz`, and a
`Control(loop)` when a feed wraps.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from intensity.config import resolve_path
from intensity.feeds import Feed
from intensity.schema import ClockTick, Control, Envelope, make_envelope

log = logging.getLogger(__name__)

DURATION_TOLERANCE_S = 0.05  # a clip whose duration changed by more is a different clip: restart its clock


def probe_duration(path: Path) -> float:
    """The clip's duration in seconds, from its container (PyAV, CPU; no frame is decoded)."""
    import av

    with av.open(str(path)) as container:
        if container.duration:
            return container.duration / av.time_base
        stream = container.streams.video[0]
        if stream.duration and stream.time_base:
            return float(stream.duration * stream.time_base)
    raise ValueError(f"cannot read the duration of {path}")


@dataclass
class FeedClock:
    feed_id: str
    sport: str
    sample: str
    started_at: float  # epoch seconds at which the clip was at t_video 0 in loop 0
    duration_s: float

    def position(self, now: float) -> tuple[float, int]:
        """(t_video, loop_epoch) at wall time `now`."""
        elapsed = max(0.0, now - self.started_at)
        epoch = math.floor(elapsed / self.duration_s)
        return elapsed - epoch * self.duration_s, epoch


def load_state(path: Path) -> dict[str, dict]:
    try:
        return dict(json.loads(path.read_text()).get("feeds", {}))
    except (OSError, ValueError, AttributeError):
        return {}


def save_state(path: Path, clocks: list[FeedClock]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    feeds = {c.feed_id: {k: v for k, v in asdict(c).items() if k != "feed_id"} for c in clocks}
    tmp.write_text(json.dumps({"feeds": feeds, "written_at": time.time()}, indent=1, sort_keys=True))
    os.replace(tmp, path)


def build_clocks(feeds: list[Feed], state_path: Path, now: float, probe=probe_duration) -> tuple[list[FeedClock], dict[str, str]]:
    """A clock per feed whose clip is on disk, keeping known start times; returns (clocks, problems)."""
    previous = load_state(state_path)
    clocks: list[FeedClock] = []
    problems: dict[str, str] = {}
    for feed in feeds:
        clip = resolve_path("data") / feed.sample
        if not clip.exists():
            problems[feed.feed_id] = f"clip missing: data/{feed.sample} (make data-pull FOLDERS=samples)"
            continue
        try:
            duration = probe(clip)
        except Exception as exc:  # a broken clip must not stop the other feeds
            problems[feed.feed_id] = f"cannot probe data/{feed.sample}: {exc}"
            continue
        old = previous.get(feed.feed_id, {})
        same_clip = old.get("sample") == feed.sample and abs(float(old.get("duration_s", -1)) - duration) <= DURATION_TOLERANCE_S
        started_at = float(old["started_at"]) if same_clip and old.get("started_at") else now
        clocks.append(FeedClock(feed.feed_id, feed.sport, feed.sample, started_at, round(duration, 3)))
    save_state(state_path, clocks)
    return clocks, problems


class ClockLane:
    """Envelopes for each tick: a ClockTick per feed, preceded by Control(loop) when a feed wraps."""

    def __init__(self, clocks: list[FeedClock]):
        self.clocks = clocks
        self.epochs: dict[str, int] = {}

    def tick(self, now: float) -> list[Envelope]:
        out: list[Envelope] = []
        t_emit_ns = int(now * 1e9)
        for c in self.clocks:
            t_video, epoch = c.position(now)
            last = self.epochs.get(c.feed_id)
            if last is not None and epoch > last:
                out.append(make_envelope(Control(feed_id=c.feed_id, kind="loop", loop_epoch=epoch, t_emit_ns=t_emit_ns)))
            self.epochs[c.feed_id] = epoch
            out.append(make_envelope(ClockTick(feed_id=c.feed_id, t_video=round(t_video, 4), loop_epoch=epoch, t_emit_ns=t_emit_ns)))
        return out
