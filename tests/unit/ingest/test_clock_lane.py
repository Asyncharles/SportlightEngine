from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import av
import pytest

from intensity.feeds import Feed
from intensity.ingest.clock_lane import ClockLane, FeedClock, build_clocks, load_state, probe_duration
from intensity.schema import ClockTick, Control


def _feed(feed_id: str, sample: str, sport: str = "football") -> Feed:
    return Feed(feed_id=feed_id, sport=sport, sample=sample, rtsp_path=feed_id,
                match_ref={"source": "statsbomb", "match_id": 1})


def synthetic_mp4(path: Path, seconds: float = 3.0, fps: int = 10) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as out:
        stream = out.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        stream.time_base = Fraction(1, fps)
        for i in range(int(seconds * fps)):
            frame = av.VideoFrame(32, 32, "yuv420p")
            frame.pts = i
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return path


def test_position_loops_the_clip():
    c = FeedClock("f", "football", "samples/f.mp4", started_at=1000.0, duration_s=100.0)
    assert c.position(1000.0) == (0.0, 0)
    assert c.position(1050.5) == (50.5, 0)
    t, epoch = c.position(1250.0)
    assert epoch == 2 and t == pytest.approx(50.0)
    assert c.position(900.0) == (0.0, 0)  # a clock set in the future never goes negative


def test_tick_sends_a_clock_per_feed_and_a_loop_control_on_wrap():
    lane = ClockLane([FeedClock("a", "football", "s/a.mp4", 0.0, 10.0), FeedClock("b", "nba", "s/b.mp4", 5.0, 10.0)])
    first = lane.tick(9.9)
    assert [type(e.data) for e in first] == [ClockTick, ClockTick]
    wrapped = lane.tick(10.1)
    assert [type(e.data) for e in wrapped] == [Control, ClockTick, ClockTick]
    assert wrapped[0].data.feed_id == "a" and wrapped[0].data.kind == "loop" and wrapped[0].data.loop_epoch == 1
    assert wrapped[1].data.loop_epoch == 1 and wrapped[1].data.t_video == pytest.approx(0.1)


def test_start_times_survive_restarts_unless_the_clip_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("INTENSITY_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "data" / "samples").mkdir(parents=True)
    for name in ("a.mp4", "b.mp4"):
        (tmp_path / "data" / "samples" / name).write_bytes(b"x")
    feeds = [_feed("a", "samples/a.mp4"), _feed("b", "samples/b.mp4"), _feed("c", "samples/missing.mp4")]
    state = tmp_path / "run" / "feeds_state.json"
    durations = {"a.mp4": 100.0, "b.mp4": 200.0}

    clocks, problems = build_clocks(feeds, state, now=1000.0, probe=lambda p: durations[p.name])
    assert [c.feed_id for c in clocks] == ["a", "b"] and "c" in problems
    assert json.loads(state.read_text())["feeds"]["a"] == {"started_at": 1000.0, "duration_s": 100.0,
                                                           "sample": "samples/a.mp4", "sport": "football"}

    durations["b.mp4"] = 201.0  # b's clip was replaced: its clock restarts; a keeps its timeline
    clocks, _ = build_clocks(feeds, state, now=5000.0, probe=lambda p: durations[p.name])
    assert {c.feed_id: c.started_at for c in clocks} == {"a": 1000.0, "b": 5000.0}
    assert load_state(state)["b"]["duration_s"] == 201.0


def test_probe_reads_the_container_duration(tmp_path):
    clip = synthetic_mp4(tmp_path / "clip.mp4", seconds=3.0)
    assert probe_duration(clip) == pytest.approx(3.0, abs=0.15)
