from __future__ import annotations

from pathlib import Path

from intensity.schema import MatchPayload
from intensity.training.sources.weightlifting import to_records

CSV = """session_id,athlete,nation,lift,attempt,weight_kg,result,t_video_walk_on,t_video_decision
s,B,LAT,snatch,1,146,good,169.0,204.5
s,A,KAZ,snatch,1,145,good,106.1,147.4
s,A,KAZ,snatch,2,151,no_lift,491.5,531.8
s,A,KAZ,snatch,3,151,,600.0,
s,C,USA,snatch,1,150,good,,
"""


def test_session_csv_becomes_attempts_in_video_time(tmp_path: Path) -> None:
    path = tmp_path / "s.csv"
    path.write_text(CSV)
    recs = [MatchPayload.model_validate(r) for r in to_records(path)]
    assert [r.game_time_s for r in recs] == sorted(r.game_time_s for r in recs)
    assert [(r.kind, r.player, r.game_time_s) for r in recs][:2] == [
        ("attempt_called", "A", 106.1),
        ("attempt_result", "A", 147.4),
    ]
    a2 = next(r for r in recs if r.kind == "attempt_called" and r.player == "A" and r.attrs["attempt"] == 2)
    assert a2.attrs["best_kg"] == 145.0 and a2.attrs["weight_kg"] == 151.0  # best good lift so far
    assert "eta_game_s" not in a2.attrs  # the decision time is the future at walk-on
    no_lift = next(r for r in recs if r.kind == "attempt_result" and r.game_time_s == 531.8)
    assert no_lift.attrs["good"] is False
    # No decision time: the call is kept, no result is invented; no walk-on time: the row is skipped.
    assert sum(1 for r in recs if r.game_time_s == 600.0) == 1
    assert all(r.player != "C" for r in recs)
