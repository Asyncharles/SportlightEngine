"""S2 adapters on fixture matches: state, features, parts, facts and the guard, for every sport."""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import pytest

from intensity.config import REPO_ROOT
from intensity.engine.adapters import ADAPTERS, F1Adapter, FootballAdapter, NbaAdapter, make_adapter
from intensity.engine.adapters.base import Adapter
from intensity.engine.explain.guard import check
from intensity.engine.explain.templates import render
from intensity.schema import MatchPayload
from intensity.training.sources import f1, football, nba
from tests.unit.training.test_convert import f1_raw, football_raw, nba_raw

SPORTS = ("football", "f1", "nba", "weightlifting")


def records(raw_records: list[dict]) -> list[MatchPayload]:
    return [MatchPayload.model_validate(r) for r in raw_records]


def weightlifting_records() -> list[MatchPayload]:
    return records(
        [
            {
                "game_time_s": 10.0,
                "period": 1,
                "kind": "attempt_called",
                "team": None,
                "player": "Li Fabin",
                "attrs": {
                    "weight_kg": 140,
                    "attempt": 2,
                    "best_kg": 138,
                    "eta_game_s": 40.0,
                    "standings_gap_kg": -2,
                },
            },
            {
                "game_time_s": 45.0,
                "period": 1,
                "kind": "attempt_result",
                "team": None,
                "player": "Li Fabin",
                "attrs": {"good": True},
            },
        ]
    )


FIXTURES = {
    "football": lambda: records(football.to_records(football_raw())),
    "f1": lambda: records(f1.to_records(f1_raw(), {"sc_slow_ratio": 1.2})),
    "nba": lambda: records(nba.to_records(nba_raw())),
    "weightlifting": weightlifting_records,
}


def feed(adapter: Adapter, recs: list[MatchPayload]) -> Adapter:
    for r in recs:
        adapter.on_match(r)
    return adapter


def check_tick(adapter: Adapter, t: float) -> None:
    row = adapter.features(t)
    assert row.shape == (len(adapter.FEATURES),) and np.isfinite(row).all()
    assert 0.0 <= adapter.heuristic_forecast(t) <= 1.0
    parts = adapter.parts(t)
    assert set(parts) == {"closeness", "momentum", "stakes"}
    for name, part in parts.items():
        assert 0.0 <= part.value <= 1.0, name
        assert not re.search(r"\d", part.phrase), part.phrase  # numbers only ever come from facts
        result = check(render(part.phrase, part.facts), part.facts)
        assert result.ok, (name, result.problems)
    facts = adapter.facts(t, parts)
    assert all(k == f.key for k, f in facts.items() if k != "game_clock")


@pytest.mark.parametrize("sport", SPORTS)
def test_adapter_contract_on_fixture(sport: str) -> None:
    adapter = make_adapter(sport)
    assert len(set(adapter.FEATURES)) == len(adapter.FEATURES)
    recs = FIXTURES[sport]()
    check_tick(adapter, 0.0)  # before any record
    for r in recs:
        adapter.on_match(r)
        check_tick(adapter, r.game_time_s)
        check_tick(adapter, r.game_time_s + 5.0)


@pytest.mark.parametrize("sport", SPORTS)
def test_reading_state_does_not_change_it(sport: str) -> None:
    """Featurize calls features() every second; the live engine calls parts() and facts() too. Neither
    may change what a later tick sees."""
    recs = FIXTURES[sport]()
    quiet = feed(make_adapter(sport), recs)
    busy = make_adapter(sport)
    for r in recs:
        busy.on_match(r)
        for dt in (0.0, 1.0, 3.0):
            busy.features(r.game_time_s + dt)
            busy.facts(r.game_time_s + dt, busy.parts(r.game_time_s + dt))
    end = recs[-1].game_time_s + 2.0
    np.testing.assert_array_equal(quiet.features(end), busy.features(end))


@pytest.mark.parametrize("sport", SPORTS)
def test_reset_forgets_everything(sport: str) -> None:
    fresh = make_adapter(sport)
    used = feed(make_adapter(sport), FIXTURES[sport]())
    used.reset()
    np.testing.assert_array_equal(fresh.features(100.0), used.features(100.0))


def test_football_score_and_windows() -> None:
    a = FootballAdapter()
    recs = FIXTURES["football"]()
    feed(a, [r for r in recs if r.period == 1])
    v = dict(zip(a.FEATURES, a.features(70.0), strict=True))
    assert v["score_diff"] == 1 and v["shots_2min_home"] == 1 and v["xg_2min_home"] == pytest.approx(0.78)
    assert v["box_touches_2min_home"] == 2  # the corner into the box and the penalty
    assert v["corners_5min_home"] == 1 and v["yellow_cards_away"] == 1
    assert v["stage_stakes"] == 0.75
    feed(a, [r for r in recs if r.period == 2])
    v = dict(zip(a.FEATURES, a.features(3010.0), strict=True))
    assert v["score_diff"] == 0 and v["shots_2min_home"] == 0  # own goal levels; windows reset at half-time
    assert v["yellow_cards_away"] == 1
    feed(a, [r for r in recs if r.period == 5])
    assert a.goals == {"Argentina": 1, "France": 1} and a.shootout == {"Argentina": 1}
    assert a.game_clock() == "Penalties"
    assert "Lionel Messi" in a.player_names()
    assert "shootout" in a.facts(7300.0)


def test_f1_battles_passes_and_safety_car() -> None:
    a = F1Adapter()
    recs = FIXTURES["f1"]()
    lap2_end = max(r.game_time_s for r in recs if r.kind == "crossing" and r.attrs["lap"] == 2)
    feed(a, [r for r in recs if r.game_time_s <= lap2_end])
    v = dict(zip(a.FEATURES, a.features(lap2_end), strict=True))
    assert a.game_clock() == "Lap 3/3"
    assert v["passes_3laps"] == 1 and v["safety_car"] == 0
    assert v["gap_leader_s"] == pytest.approx(0.2)  # Leclerc behind Verstappen after lap 2
    assert v["min_gap_s"] == pytest.approx(0.2) and v["battles_1s"] == 2
    assert a.facts(lap2_end)["min_gap_s"].text == "Leclerc 0.2 s behind Verstappen"
    feed(a, [r for r in recs if r.game_time_s > lap2_end])
    v = dict(zip(a.FEATURES, a.features(1e4), strict=True))
    assert v["safety_car"] == 1 and v["laps_since_sc"] == 0
    assert v["passes_3laps"] == 1  # the pit swap is not an overtake
    assert v["pit_stops_1lap"] == 1
    assert a.facts(1e4)["safety_car"].text == "safety car out"


def test_f1_mock_lap_records_still_drive_the_gap() -> None:
    """The mock sends aggregate `lap` records with gaps_s instead of crossings; the imminent rule
    reads the `min_gap_s` fact either way."""
    a = F1Adapter()
    a.on_match(
        MatchPayload(
            game_time_s=10,
            period=1,
            kind="lap",
            attrs={"lap": 5, "total_laps": 58, "gaps_s": [2.2, 0.4, 3.1]},
        )
    )
    assert a.facts(10.0)["min_gap_s"].value == 0.4
    assert dict(zip(a.FEATURES, a.features(10.0), strict=True))["battles_1s"] == 1


def test_nba_state() -> None:
    a = NbaAdapter()
    recs = FIXTURES["nba"]()
    feed(a, [r for r in recs if r.period == 1])
    v = dict(zip(a.FEATURES, a.features(130.0), strict=True))
    assert (v["margin"], v["team_fouls_home"], v["timeouts_left_home"], v["timeouts_left_away"]) == (
        2,
        1,
        6,
        7,
    )
    assert v["possession_home"] == 1.0  # ATL's made last free throw hands the ball to BOS
    assert v["run_2min"] == 2 and v["lead_changes"] == 0
    assert a.game_clock() == "Q1 10:00"
    feed(a, [r for r in recs if r.period == 5])
    assert a.ties == 1 and a.game_clock() == "OT1 4:00"
    assert a.facts(2940.0)["timeouts_left_home"].value == 2  # overtime allowance


def test_nba_clock_stops_at_regulation_unless_tied() -> None:
    a = NbaAdapter()
    a.on_match(
        MatchPayload(
            game_time_s=2870,
            period=4,
            kind="score",
            team="BOS",
            attrs={"points": 2, "home_score": 100, "away_score": 98, "home": "BOS", "away": "ATL"},
        )
    )
    a.parts(2900.0)
    assert a.game_clock() == "Q4 0:00"
    b = NbaAdapter()
    b.on_match(
        MatchPayload(
            game_time_s=2870,
            period=4,
            kind="score",
            team="BOS",
            attrs={"points": 2, "home_score": 100, "away_score": 100, "home": "BOS", "away": "ATL"},
        )
    )
    b.parts(2900.0)
    assert b.game_clock() == "OT1 4:40"


def test_every_sport_has_an_adapter_config() -> None:
    for sport in SPORTS:
        assert ADAPTERS[sport]().cfg, sport


# --------------------------------------------------------------------------- real data, when downloaded


def _real(sport: str, n: int = 3) -> list[Path]:
    return [Path(p) for p in sorted(glob.glob(str(REPO_ROOT / "data" / "events" / sport / "*.jsonl")))[:n]]


@pytest.mark.parametrize("sport", ["football", "f1", "nba"])
def test_adapter_on_real_matches(sport: str) -> None:
    paths = _real(sport)
    if not paths:
        pytest.skip(f"no data/events/{sport}/ yet: make fetch-{sport} convert-{sport}")
    for path in paths:
        recs = [MatchPayload.model_validate(json.loads(line)) for line in path.read_text().splitlines()]
        recs.sort(key=lambda r: (r.period, r.game_time_s))
        adapter = make_adapter(sport)
        for i, r in enumerate(recs):
            adapter.on_match(r)
            if i % 50 == 0:
                check_tick(adapter, r.game_time_s)
        check_tick(adapter, recs[-1].game_time_s)
