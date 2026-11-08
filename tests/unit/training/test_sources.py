"""T0 fetchers: resume after an interruption, retries, paging and failures, all offline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from intensity.training.sources import make_fetcher
from intensity.training.sources.base import FAILED, INDEX, Fetcher, FetchError
from intensity.training.sources.f1 import JolpicaFetcher
from intensity.training.sources.football import StatsBombFetcher
from intensity.training.sources.nba import NbaFetcher

FETCH_CFG = {"timeout_s": 5, "retries": 3, "backoff_s": [1.0, 8.0]}


class Waits(list):
    def __call__(self, s: float) -> None:
        self.append(s)


class FakeFetcher(Fetcher):
    sport = "fake"

    def __init__(self, raw_dir: Path, interrupt_on: str | None = None, fail_on: str | None = None):
        super().__init__({}, FETCH_CFG, raw_dir, client=httpx.Client(), sleep=Waits())
        self.interrupt_on = interrupt_on
        self.fail_on = fail_on
        self.calls: list[str] = []

    def list_items(self) -> tuple[list[str], dict[str, Any]]:
        return ["a", "b", "c"], {}

    def fetch_item(self, item: str) -> Any:
        self.calls.append(item)
        if item == self.interrupt_on:
            raise KeyboardInterrupt
        if item == self.fail_on:
            raise FetchError("gone")
        return {"item": item}


def mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_resumes_after_interruption(tmp_path: Path) -> None:
    first = FakeFetcher(tmp_path, interrupt_on="b")
    with pytest.raises(KeyboardInterrupt):
        first.run()
    assert sorted(p.name for p in tmp_path.iterdir()) == [INDEX, "a.json"]  # no partial b

    second = FakeFetcher(tmp_path)
    stats = second.run()
    assert second.calls == ["b", "c"]
    assert (stats.fetched, stats.skipped, stats.failed) == (2, 1, [])
    assert json.loads((tmp_path / "c.json").read_text()) == {"item": "c"}


def test_limit_counts_new_items_only(tmp_path: Path) -> None:
    FakeFetcher(tmp_path).run(limit=1)
    fetcher = FakeFetcher(tmp_path)
    stats = fetcher.run(limit=1)
    assert fetcher.calls == ["b"] and stats.skipped == 1


def test_failure_is_recorded_then_cleared(tmp_path: Path) -> None:
    stats = FakeFetcher(tmp_path, fail_on="b").run()
    assert stats.failed == ["b"] and stats.fetched == 2
    assert "b" in json.loads((tmp_path / FAILED).read_text())

    stats = FakeFetcher(tmp_path).run()
    assert stats.fetched == 1 and not (tmp_path / FAILED).exists()


def test_index_is_cached(tmp_path: Path) -> None:
    FakeFetcher(tmp_path).index()
    (tmp_path / INDEX).write_text(json.dumps({"sport": "fake", "items": ["z"], "meta": {}}))
    assert FakeFetcher(tmp_path).index()["items"] == ["z"]
    assert FakeFetcher(tmp_path).index(refresh=True)["items"] == ["a", "b", "c"]


def test_get_json_retries_and_honours_retry_after(tmp_path: Path) -> None:
    replies = iter(
        [
            httpx.Response(429, headers={"Retry-After": "5"}),
            httpx.Response(503),
            httpx.Response(200, json=[1]),
        ]
    )
    fetcher = FakeFetcher(tmp_path)
    fetcher.client = mock_client(lambda request: next(replies))
    assert fetcher.get_json("https://x/y") == [1]
    assert fetcher.sleep == [5.0, 2.0]  # Retry-After, then the doubled backoff


def test_get_json_gives_up(tmp_path: Path) -> None:
    fetcher = FakeFetcher(tmp_path)
    fetcher.client = mock_client(lambda request: httpx.Response(500))
    with pytest.raises(FetchError, match="HTTP 500"):
        fetcher.get_json("https://x/y")
    assert len(fetcher.sleep) == FETCH_CFG["retries"]

    fetcher.client = mock_client(lambda request: httpx.Response(404))
    with pytest.raises(FetchError, match="HTTP 404"):
        fetcher.get_json("https://x/y")


def test_statsbomb(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/matches/43/106.json"):
            return httpx.Response(
                200,
                json=[
                    {"match_id": 2, "match_date": "2022-12-18", "home_team": {"home_team_name": "Argentina"}},
                    {"match_id": 1, "match_date": "2022-11-20", "home_team": {"home_team_name": "Qatar"}},
                ],
            )
        if request.url.path.endswith("/events/2.json"):
            return httpx.Response(200, json=[{"type": {"name": "Shot"}}])
        if request.url.path.endswith("/lineups/2.json"):
            return httpx.Response(200, json=[{"team_name": "Argentina", "lineup": []}])
        return httpx.Response(404)

    cfg = {"base_url": "https://sb/data", "competitions": [{"competition_id": 43, "season_id": 106}]}
    fetcher = StatsBombFetcher(cfg, FETCH_CFG, tmp_path, client=mock_client(handler), sleep=Waits())
    stats = fetcher.run(items=["2"])
    assert fetcher.index()["items"] == ["1", "2"]  # by date
    saved = json.loads((tmp_path / "2.json").read_text())
    assert stats.fetched == 1 and saved["match"]["home_team"]["home_team_name"] == "Argentina"
    assert saved["events"] == [{"type": {"name": "Shot"}}]
    assert saved["lineups"][0]["team_name"] == "Argentina"


def _mr(table: str, key: str, rows: list, limit: int, offset: int, total: int) -> dict:
    return {"MRData": {"limit": str(limit), "offset": str(offset), "total": str(total), table: {key: rows}}}


def test_jolpica_pages_and_merges_laps(tmp_path: Path) -> None:
    def timing(driver: str) -> dict:
        return {"driverId": driver, "position": "1", "time": "1:30.000"}

    def handler(request: httpx.Request) -> httpx.Response:
        path, offset = request.url.path, int(request.url.params["offset"])
        if path.endswith("/2023.json"):
            races = [{"season": "2023", "round": "1", "raceName": "Bahrain Grand Prix", "date": "2023-03-05"}]
            return httpx.Response(200, json=_mr("RaceTable", "Races", races, 2, 0, 1))
        if path.endswith("/2023/1/results.json"):
            races = [{"season": "2023", "round": "1", "Results": [{"position": "1"}]}]
            return httpx.Response(200, json=_mr("RaceTable", "Races", races, 2, 0, 1))
        if path.endswith("/2023/1/laps.json"):  # 3 timings, 2 per page; lap 1 spans both pages
            page = (
                [{"Laps": [{"number": "1", "Timings": [timing("ver"), timing("per")]}]}]
                if offset == 0
                else [{"Laps": [{"number": "1", "Timings": [timing("alo")]}]}]
            )
            return httpx.Response(200, json=_mr("RaceTable", "Races", page, 2, offset, 3))
        if path.endswith("/2023/1/pitstops.json"):
            races = [{"PitStops": [{"driverId": "ver", "lap": "14"}]}]
            return httpx.Response(200, json=_mr("RaceTable", "Races", races, 2, 0, 1))
        return httpx.Response(404)

    cfg = {"base_url": "https://j/ergast/f1", "seasons": [2023], "page_limit": 2}
    fetcher = JolpicaFetcher(cfg, FETCH_CFG, tmp_path, client=mock_client(handler), sleep=Waits())
    assert fetcher.run().fetched == 1
    saved = json.loads((tmp_path / "2023-01.json").read_text())
    assert [t["driverId"] for t in saved["laps"][0]["Timings"]] == ["ver", "per", "alo"]
    assert saved["pitstops"] == [{"driverId": "ver", "lap": "14"}]
    assert saved["race"]["round"] == "1" and saved["results"] == [{"position": "1"}]


def test_nba(tmp_path: Path) -> None:
    class Fake(NbaFetcher):
        def _game_rows(self, season: str, season_type: str) -> list[dict[str, Any]]:
            if season_type == "Playoffs":
                return []
            return [
                {
                    "GAME_ID": "002",
                    "GAME_DATE": "2024-10-23",
                    "TEAM_ABBREVIATION": "LAL",
                    "MATCHUP": "LAL @ PHX",
                },
                {
                    "GAME_ID": "001",
                    "GAME_DATE": "2024-10-22",
                    "TEAM_ABBREVIATION": "BOS",
                    "MATCHUP": "BOS vs. NYK",
                },
                {
                    "GAME_ID": "001",
                    "GAME_DATE": "2024-10-22",
                    "TEAM_ABBREVIATION": "NYK",
                    "MATCHUP": "NYK @ BOS",
                },
            ]

        def _pbp(self, game_id: str) -> dict[str, Any]:
            if game_id == "002":
                raise ConnectionError("reset")
            return {"game": {"gameId": game_id, "actions": [{"actionNumber": 1}]}}

    cfg = {"seasons": ["2024-25"], "season_types": ["Regular Season", "Playoffs"]}
    fetcher = Fake(cfg, FETCH_CFG, tmp_path, sleep=Waits())
    stats = fetcher.run()
    index = fetcher.index()
    assert index["items"] == ["001", "002"]
    assert (
        index["meta"]["001"]["teams"] == ["BOS", "NYK"] and index["meta"]["001"]["matchup"] == "BOS vs. NYK"
    )
    assert stats.fetched == 1 and stats.failed == ["002"]  # retried, then recorded
    assert len(fetcher.sleep) == FETCH_CFG["retries"]


def test_config_wires_every_fetcher(config, tmp_path: Path) -> None:
    config.training["paths"]["raw"] = str(tmp_path)
    for sport, cls in (("football", StatsBombFetcher), ("f1", JolpicaFetcher), ("nba", NbaFetcher)):
        fetcher = make_fetcher(sport, config)
        assert isinstance(fetcher, cls) and fetcher.raw_dir == tmp_path / sport
    with pytest.raises(ValueError):
        make_fetcher("weightlifting", config)


@pytest.mark.net
@pytest.mark.parametrize("sport", ["football", "nba", "f1"])
def test_fetch_one_item_live(config, tmp_path: Path, sport: str) -> None:
    config.training["paths"]["raw"] = str(tmp_path)
    fetcher = make_fetcher(sport, config)
    first = fetcher.index()["items"][0]
    assert fetcher.run(items=[first]).fetched == 1
    assert make_fetcher(sport, config).run(items=[first]).skipped == 1
