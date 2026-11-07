from __future__ import annotations

import pytest

from intensity.sync import SyncRow, SyncTable


def _table() -> SyncTable:
    return SyncTable([
        SyncRow("c", 0.0, 100.0, 600.0, 1),    # first segment: game 600-700 s
        SyncRow("c", 110.0, 200.0, 705.0, 1),  # 10 s replay gap, play resumes at 705
        SyncRow("c", 200.0, 260.0, 2700.0, 2),  # second half
    ])


def test_game_time_inside_segments():
    t = _table()
    assert t.game_time(10.0) == (610.0, 1)
    assert t.game_time(150.0) == (745.0, 1)
    assert t.game_time(230.0) == (2730.0, 2)


def test_gaps_have_no_game_time_but_last_holds():
    t = _table()
    assert t.game_time(105.0) is None
    assert t.last_game_time(105.0) == (700.0, 1)
    assert t.last_game_time(-1.0) is None


def test_inverse_mapping():
    t = _table()
    assert t.t_video(610.0) == 10.0
    assert t.t_video(702.0) is None  # happened during the replay gap
    assert t.t_video(2730.0, period=2) == 230.0
    assert t.t_video(2730.0, period=1) is None


def test_load_csv(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("clip_id,video_start,video_end,game_start,period\nc,0,100,600,1\nc,110,200,705,1\n")
    assert SyncTable.load(p).game_time(120.0) == (715.0, 1)


def test_overlapping_rows_rejected():
    with pytest.raises(ValueError):
        SyncTable([SyncRow("c", 0, 100, 0, 1), SyncRow("c", 50, 150, 200, 1)])
