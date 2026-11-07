from __future__ import annotations

from collections import Counter

from intensity.engine.mock import MockGenerator, read_tape, write_tape
from intensity.schema import Envelope, parse_envelope

FEEDS = [("a", "football"), ("b", "f1"), ("c", "nba"), ("d", "weightlifting")]


def test_deterministic_and_chunk_independent():
    g1 = MockGenerator(FEEDS, seed=3, t0_ns=0)
    g2 = MockGenerator(FEEDS, seed=3, t0_ns=0)
    whole = [e.model_dump_json() for e in g1.envelopes(0, 60)]
    chunks = [e.model_dump_json() for t in range(0, 60, 7) for e in g2.envelopes(t, min(t + 7, 60))]
    assert whole == chunks
    other = [e.model_dump_json() for e in MockGenerator(FEEDS, seed=4, t0_ns=0).envelopes(0, 60)]
    assert whole != other


def test_valid_envelopes_for_four_feeds_and_every_source():
    envs = MockGenerator(FEEDS, t0_ns=0).envelopes(0, 120)
    for env in envs:
        parse_envelope(env.model_dump_json())  # survives the wire format
    by_feed = Counter(e.data.feed_id for e in envs)
    assert set(by_feed) == {"a", "b", "c", "d"}
    for fid, _ in FEEDS:
        sources = {e.data.source for e in envs if e.kind == "event" and e.data.feed_id == fid}
        assert sources == {"match", "crowd", "voice", "video", "ocr", "commentary"}, fid
    ticks = [e for e in envs if e.kind == "clock" and e.data.feed_id == "a"]
    assert len(ticks) == 480  # 4 Hz


def test_loops_send_control_and_bump_epoch():
    g = MockGenerator([("a", "football")], duration_s=30, t0_ns=0)
    envs = g.envelopes(0, 65)
    loops = [e for e in envs if e.kind == "control"]
    assert [e.data.loop_epoch for e in loops] == [1, 2]
    assert max(e.data.t_video for e in envs if e.kind == "clock") < 30


def test_tape_round_trip(tmp_path):
    envs = MockGenerator(FEEDS[:1], t0_ns=0).envelopes(0, 5)
    write_tape(tmp_path / "t.jsonl", envs)
    back = read_tape(tmp_path / "t.jsonl")
    assert [e.model_dump_json() for e in back] == [e.model_dump_json() for e in envs]
    assert all(isinstance(e, Envelope) for e in back)
