from __future__ import annotations

from pathlib import Path

import pytest

from intensity.engine.synccheck import build, clock_label, grab, pick_moments, shift
from intensity.feeds import Feed
from intensity.schema import MatchPayload
from intensity.sync import SyncRow, SyncTable
from tests.unit.ingest.test_clock_lane import synthetic_mp4


def _feed(sport: str, source: str = "statsbomb", sync: str | None = "data/sync/x.csv") -> Feed:
    return Feed(feed_id="f", sport=sport, sample="samples/x.mp4", rtsp_path="f",
                match_ref={"source": source, "match_id": "m"}, sync=sync)


def _rec(t: float, kind: str, period: int = 2, **attrs) -> MatchPayload:
    return MatchPayload(game_time_s=t, period=period, kind=kind, team="T", player="P", attrs=attrs)


def test_moments_are_spread_prefer_goals_and_map_through_sync():
    sync = SyncTable([SyncRow("x", 0.0, 600.0, 3000.0, 2)])
    recs = [_rec(3010, "shot"), _rec(3020, "goal"), _rec(3300, "shot"), _rec(3590, "shot"), _rec(9999, "goal")]
    moments = pick_moments(_feed("football"), recs, sync, n=3)
    assert [m.record.kind for m in moments][0] == "goal"  # the goal beats the shot in the same slice
    assert [round(m.t_video) for m in moments] == [20, 300, 590]  # 9999 s is outside the clip
    assert moments[0].clock == "50:20 (P2)"


def test_f1_lap_one_swaps_are_skipped_and_weightlifting_is_already_in_video_time():
    sync = SyncTable([SyncRow("x", 0.0, 100.0, 0.0, 1)])
    recs = [_rec(5, "pass", period=1, lap=1, start=True), _rec(50, "pass", period=1, lap=12)]
    assert [m.clock for m in pick_moments(_feed("f1"), recs, sync, n=5)] == ["lap 12"]
    wl = [_rec(106.1, "attempt_called", period=1, weight_kg=145, attempt=1), _rec(147.4, "attempt_result", period=1, good=True)]
    moments = pick_moments(_feed("weightlifting", "session_csv", None), wl, None, n=2)
    assert [m.t_video for m in moments] == [106.1, 147.4]
    assert moments[0].clock == "P: 145 kg, attempt 1"


def test_nba_clock_counts_down_the_quarter():
    assert clock_label("nba", _rec(2160 + 19, "score", period=4)) == "Q4 11:41"
    assert clock_label("nba", _rec(2880 + 10, "score", period=5)) == "OT1 4:50"


def test_shift_moves_records_later_in_the_video(tmp_path: Path):
    path = tmp_path / "x.csv"
    path.write_text("clip_id,video_start,video_end,game_start,period\nx,0.000,100.000,3000.000,2\nx,110.000,200.000,3100.000,2\n")
    before = SyncTable.load(path).t_video(3050.0, 2)
    assert shift(path, 3.5, row=0) == ["row 0: game_start 2996.500"]
    assert SyncTable.load(path).t_video(3050.0, 2) == pytest.approx(before + 3.5)
    assert SyncTable.load(path).rows[1].game_start == 3100.0  # only the chosen row


def test_grab_and_build_write_three_frames_per_moment(tmp_path: Path):
    clip = synthetic_mp4(tmp_path / "x.mp4", seconds=30.0, fps=5)
    sync = SyncTable([SyncRow("x", 0.0, 30.0, 3000.0, 2)])
    moments = pick_moments(_feed("football"), [_rec(3010, "goal"), _rec(3020, "shot")], sync, n=2)
    frames = grab(str(clip), None, [t for m in moments for t in (m.t_video - 5, m.t_video, m.t_video + 5)], 160)
    page = build(_feed("football"), moments, frames, 5.0, tmp_path / "out")
    assert len(list((tmp_path / "out").glob("*.jpg"))) == 6
    assert "Should show <b>50:10 (P2)</b>" in page.read_text()
