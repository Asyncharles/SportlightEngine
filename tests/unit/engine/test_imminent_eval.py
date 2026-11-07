from __future__ import annotations

import pytest

from intensity.training import imminent_eval as ie

from .helpers import crowd, match, video


def test_match_moments_per_sport():
    fb = [match(10, "shot", xg=0.05), match(20, "shot", xg=0.2), match(30, "goal")]
    assert ie.match_moments(fb, "football") == [20, 30]
    nba = [match(t, "score", sport="nba", home_score=h, away_score=a) for t, h, a in
           [(1, 2, 0), (2, 2, 3), (3, 5, 3), (4, 7, 3)]]
    assert ie.match_moments(nba, "nba") == [2, 3]  # two lead changes
    f1 = [match(5, "pass", sport="f1"), match(6, "pass", sport="f1", pit_swap=True)]
    assert ie.match_moments(f1, "f1") == [5]


def test_cue_moments_roar_replay_celebration():
    events = [crowd(t / 2, loud=0.0) for t in range(100)]  # the roar must sit above the 95th percentile
    events += [crowd(50 + t / 2, loud=15.0) for t in range(4)]
    events += [video(60.0, prompts={"goal celebration": 0.5, "x": 0.1}), video(61.0, shot="replay"), video(61.5, shot="replay")]
    kinds = {src for _, src in ie.cue_moments(events, ["goal celebration"])}
    assert kinds == {"roar", "replay", "celebration"}


def test_merge_keeps_earliest_within_10s():
    assert ie.merge([(30, "a"), (25, "b"), (50, "c"), (34.9, "d")]) == [(25, "b"), (50, "c")]


def test_hits_need_2_to_10s_lead():
    hits = ie.score_hits([10.0, 40.0, 70.0], [15.0, 41.0, 90.0])
    assert hits == [(10.0, 15.0)]


def test_summary_and_targets():
    r = ie.ClipResult("c", 600.0, [15.0, 50.0], ["match", "match"], [10.0, 30.0, 45.0], [(10.0, 15.0), (45.0, 50.0)])
    s = ie.summarise([r])
    assert s["precision"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["recall"] == 1.0 and s["median_lead_s"] == 5.0 and s["false_alarms_per_10min"] == 1.0
    assert s["meets_targets"]


def test_tuning_refuses_demo_clips(tmp_path, monkeypatch):
    tuning = tmp_path / "data" / "samples" / "tuning" / "football"
    tuning.mkdir(parents=True)
    (tuning / "arg_fra_2022_final.mp4").write_bytes(b"")
    monkeypatch.setattr(ie, "resolve_path", lambda p: tmp_path / p)
    with pytest.raises(SystemExit, match="hard rule 6"):
        ie.find_tapes("football", "tuning", tmp_path)


def test_mock_report_end_to_end(tmp_path):
    ie.main(["--sport", "nba", "--set", "mock", "--minutes", "3", "--out", str(tmp_path)])
    page = next(tmp_path.rglob("report.html")).read_text()
    assert "Imminent: nba" in page and "<svg" in page
    assert next(tmp_path.rglob("config.resolved.yaml")).exists()
