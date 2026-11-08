from __future__ import annotations

import pytest

from intensity.engine.replay import Replayer, load_records
from intensity.schema import MatchPayload
from intensity.sync import SyncRow, SyncTable


def _rec(gt: float, kind="shot", period=1) -> MatchPayload:
    return MatchPayload(game_time_s=gt, period=period, kind=kind)


def test_releases_records_as_the_clock_passes_them():
    sync = SyncTable([SyncRow("c", 0, 100, 600, 1), SyncRow("c", 110, 200, 705, 1)])
    recs = [_rec(590), _rec(610), _rec(702), _rec(720), _rec(900)]
    rp = Replayer("f", "football", recs, sync)
    assert [t for t, _ in rp.timeline] == pytest.approx([0.0, 10.0, 100.0, 125.0, 200.0])
    assert [e.payload["game_time_s"] for e in rp.release(5.0, 0)] == [590]  # before the clip: released at once
    assert [e.payload["game_time_s"] for e in rp.release(50.0, 0)] == [610]
    assert rp.release(99.0, 0) == []
    out = rp.release(105.0, 0)  # 702 happened during the gap: released at the gap's start, then the replayer holds
    assert [e.payload["game_time_s"] for e in out] == [702]
    assert [e.t_video for e in rp.release(130.0, 0)] == [125.0]
    rp.reset()
    assert len(rp.release(1e9, 1)) == 5


def test_records_before_the_clip_keep_match_order_period_first():
    """First-half stoppage time (game time past 2700 s) must not interleave with the second half,
    or the adapter sees period changes back and forth and resets its state (serve parity)."""
    sync = SyncTable([SyncRow("c", 0, 100, 3000, 2)])  # the clip starts in the second half
    recs = [_rec(2700, period=2), _rec(2750, period=1), _rec(2800, period=2), _rec(2720, period=1)]
    rp = Replayer("f", "football", recs, sync)
    released = [(e.payload["period"], e.payload["game_time_s"]) for e in rp.release(0.0, 0)]
    assert released == [(1, 2720), (1, 2750), (2, 2700), (2, 2800)]


def test_events_carry_feed_and_epoch():
    rp = Replayer("fb", "football", [_rec(1.0)], SyncTable.identity(10))
    (ev,) = rp.release(2.0, 3)
    assert ev.feed_id == "fb" and ev.loop_epoch == 3 and ev.source == "match"


def test_load_records(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text('{"game_time_s": 1, "period": 1, "kind": "goal", "team": "A"}\n\n')
    assert load_records(p)[0].kind == "goal"
