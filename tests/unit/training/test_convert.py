"""S2 converters: raw StatsBomb, Jolpica and nba_api shapes to match records, on small fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from intensity.schema import MatchPayload
from intensity.training.sources import convert_dir, f1, football, nba


def _sb_event(
    etype: str, period: int, ts: str, team: str = "Argentina", player: tuple[int, str] | None = None, **extra
):
    ev = {"type": {"name": etype}, "period": period, "timestamp": ts, "team": {"name": team}, **extra}
    if player:
        ev["player"] = {"id": player[0], "name": player[1]}
    return ev


def football_raw() -> dict:
    messi = (1, "Lionel Andrés Messi Cuccittini")
    return {
        "match": {
            "home_team": {"home_team_name": "Argentina"},
            "away_team": {"away_team_name": "France"},
            "competition_stage": {"name": "Final"},
            "home_score": 3,  # the result must never reach the records
        },
        "lineups": [
            {
                "team_name": "Argentina",
                "lineup": [{"player_id": 1, "player_name": messi[1], "player_nickname": "Lionel Messi"}],
            }
        ],
        "events": [
            _sb_event("Half Start", 1, "00:00:00.000"),
            _sb_event("Half Start", 1, "00:00:00.000", team="France"),
            _sb_event(
                "Pass",
                1,
                "00:01:00.500",
                player=messi,
                location=[70, 40],
                **{"pass": {"end_location": [110, 40], "type": {"name": "Corner"}}},
            ),
            _sb_event(
                "Shot",
                1,
                "00:01:02.000",
                player=messi,
                location=[108, 40],
                shot={"statsbomb_xg": 0.78, "outcome": {"name": "Goal"}, "type": {"name": "Penalty"}},
            ),
            _sb_event(
                "Foul Committed",
                1,
                "00:10:00.000",
                team="France",
                location=[50, 30],
                foul_committed={"card": {"name": "Yellow Card"}},
            ),
            _sb_event("Half Start", 2, "00:00:00.000"),
            _sb_event("Own Goal For", 2, "00:05:00.000", team="France"),
            _sb_event(
                "Shot",
                5,
                "00:01:00.000",
                player=messi,
                location=[108, 40],
                shot={"statsbomb_xg": 0.78, "outcome": {"name": "Goal"}, "type": {"name": "Penalty"}},
            ),
        ],
    }


def test_football_records() -> None:
    recs = football.to_records(football_raw())
    kinds = [r["kind"] for r in recs]
    assert kinds == [
        "kickoff",
        "period_start",
        "pass",
        "goal",
        "foul",
        "card",
        "period_start",
        "goal",
        "goal",
    ]
    assert recs[0]["attrs"] == {"home": "Argentina", "away": "France", "stage": "Final"}
    goal = recs[3]
    assert goal["player"] == "Lionel Messi" and goal["game_time_s"] == 62.0 and goal["attrs"]["xg"] == 0.78
    assert recs[2]["attrs"]["pass_type"] == "Corner" and recs[2]["attrs"]["completed"] is True
    assert recs[5]["attrs"] == {"color": "yellow"}
    own = recs[7]
    assert own["game_time_s"] == 2700 + 300 and own["team"] == "France" and own["attrs"]["own_goal"] is True
    assert recs[8]["attrs"]["shootout"] is True and recs[8]["period"] == 5
    assert all("home_score" not in r["attrs"] for r in recs)
    for r in recs:
        MatchPayload.model_validate(r)


def _results(*drivers: tuple[str, str, int]) -> list[dict]:
    return [
        {
            "Driver": {"driverId": d, "familyName": name},
            "Constructor": {"name": f"Team {name}"},
            "grid": str(grid),
            "laps": "3",
        }
        for d, name, grid in drivers
    ]


def f1_raw() -> dict:
    def lap(n: int, *rows: tuple[str, str, int]) -> dict:
        return {"number": n, "Timings": [{"driverId": d, "time": t, "position": str(p)} for d, t, p in rows]}

    return {
        "race": {"season": "2023", "round": "1"},
        "results": _results(("ver", "Verstappen", 1), ("ham", "Hamilton", 2), ("lec", "Leclerc", 3)),
        "laps": [
            lap(1, ("ver", "1:40.000", 1), ("ham", "1:40.500", 2), ("lec", "1:41.000", 3)),
            # Leclerc passes Hamilton on track; Hamilton did not pit.
            lap(2, ("ver", "1:30.000", 1), ("lec", "1:29.200", 2), ("ham", "1:30.200", 3)),
            # Everyone slow: a safety car. Verstappen pits and drops behind Leclerc: a pit swap.
            lap(3, ("lec", "2:00.000", 1), ("ver", "2:00.900", 2), ("ham", "2:00.800", 3)),
        ],
        "pitstops": [{"driverId": "ver", "lap": "3"}],
    }


def test_f1_records() -> None:
    recs = f1.to_records(f1_raw(), {"sc_slow_ratio": 1.2})
    assert recs[0]["kind"] == "race_start" and recs[0]["attrs"]["total_laps"] == 3
    assert recs[0]["attrs"]["grid"] == ["Verstappen", "Hamilton", "Leclerc"]
    crossings = [r for r in recs if r["kind"] == "crossing"]
    assert len(crossings) == 9
    ham1 = next(r for r in crossings if r["player"] == "Hamilton" and r["attrs"]["lap"] == 1)
    assert (
        ham1["game_time_s"] == 100.5
        and ham1["attrs"]["gap_ahead_s"] == 0.5
        and ham1["team"] == "Team Hamilton"
    )
    passes = [(r["player"], r["attrs"]) for r in recs if r["kind"] == "pass"]
    assert ("Leclerc", {"lap": 2, "position": 2, "passed": "Hamilton"}) in passes
    assert any(p == "Leclerc" and a.get("pit_swap") and a["passed"] == "Verstappen" for p, a in passes)
    sc = [r for r in recs if r["kind"] == "safety_car"]
    assert [r["attrs"] for r in sc] == [{"active": True, "lap": 3}]
    # The safety-car record waits for the last car on the lap.
    assert sc[0]["game_time_s"] == max(r["game_time_s"] for r in crossings if r["attrs"]["lap"] == 3)
    ver3 = next(r for r in crossings if r["player"] == "Verstappen" and r["attrs"]["lap"] == 3)
    assert ver3["attrs"]["pitted"] is True and ver3["attrs"]["tyre_age"] == 0
    laps = [r["attrs"]["lap"] for r in recs if r["kind"] == "lap"]
    assert laps == [2, 3, 3]
    assert [r["game_time_s"] for r in recs] == sorted(r["game_time_s"] for r in recs)
    for r in recs:
        MatchPayload.model_validate(r)


def _act(n: int, period: int, clock: str, kind: str, team: str = "", loc: str = "", **kw) -> dict:
    base = {
        "actionNumber": n,
        "period": period,
        "clock": clock,
        "actionType": kind,
        "subType": "",
        "teamTricode": team,
        "playerName": "",
        "scoreHome": "",
        "scoreAway": "",
        "shotValue": 0,
        "location": loc,
        "description": "",
    }
    base.update(kw)
    return base


def nba_raw() -> dict:
    return {
        "game": {
            "actions": [
                _act(1, 1, "PT12M00.00S", "period", subType="start", scoreHome="0", scoreAway="0"),
                _act(
                    2,
                    1,
                    "PT11M30.00S",
                    "Made Shot",
                    "BOS",
                    "h",
                    playerName="Tatum",
                    scoreHome="3",
                    scoreAway="0",
                    shotValue=3,
                ),
                _act(3, 1, "PT11M10.00S", "Missed Shot", "ATL", "v", playerName="Young", shotValue=2),
                _act(4, 1, "PT11M09.00S", "Rebound", "ATL", "v", playerName="Capela"),
                _act(5, 1, "PT11M00.00S", "Foul", "BOS", "h", subType="Shooting"),
                _act(
                    6,
                    1,
                    "PT11M00.00S",
                    "Free Throw",
                    "ATL",
                    "v",
                    playerName="Capela",
                    subType="Free Throw 1 of 2",
                    description="MISS Capela Free Throw 1 of 2",
                ),
                _act(
                    7,
                    1,
                    "PT11M00.00S",
                    "Free Throw",
                    "ATL",
                    "v",
                    playerName="Capela",
                    subType="Free Throw 2 of 2",
                    scoreHome="3",
                    scoreAway="1",
                ),
                _act(8, 1, "PT10M00.00S", "Timeout", "", "h", subType="Regular"),
                _act(
                    9,
                    5,
                    "PT04M00.00S",
                    "Made Shot",
                    "ATL",
                    "v",
                    playerName="Young",
                    scoreHome="3",
                    scoreAway="3",
                    shotValue=2,
                ),
            ]
        }
    }


def test_nba_records() -> None:
    recs = nba.to_records(nba_raw())
    assert [r["kind"] for r in recs] == [
        "game_start",
        "period_start",
        "score",
        "miss",
        "rebound",
        "foul",
        "miss",
        "score",
        "timeout",
        "score",
    ]
    assert recs[0]["attrs"] == {"home": "BOS", "away": "ATL"}
    assert recs[2]["attrs"] == {"points": 3, "home_score": 3, "away_score": 0, "home": "BOS", "away": "ATL"}
    assert recs[2]["game_time_s"] == 30.0
    assert recs[4]["attrs"] == {"offensive": True}
    assert recs[6]["attrs"]["free_throw"] is True and recs[7]["attrs"]["ft_last"] is True
    assert recs[8]["team"] == "BOS"  # timeouts carry no tricode, only the side
    assert recs[9]["game_time_s"] == 2880 + 60 and recs[9]["period"] == 5
    assert nba.game_time(4, "PT00M00.00S") == 2880
    for r in recs:
        MatchPayload.model_validate(r)


def test_convert_dir_skips_up_to_date_and_isolates_failures(tmp_path: Path) -> None:
    raw, events = tmp_path / "raw", tmp_path / "events"
    raw.mkdir()
    (raw / "_index.json").write_text("{}")
    (raw / "good.json").write_text(json.dumps(nba_raw()))
    (raw / "bad.json").write_text(json.dumps({"game": {}}))
    stats = convert_dir("nba", nba.to_records, raw, events, {})
    assert stats.converted == 1 and set(stats.failed) == {"bad"}
    lines = (events / "good.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["kind"] == "game_start"
    again = convert_dir("nba", nba.to_records, raw, events, {})
    assert again.converted == 0 and again.skipped == 1
    assert convert_dir("nba", nba.to_records, raw, events, {}, force=True, items=["good"]).converted == 1
