"""NBA raw data: `nba_api` PlayByPlayV3 for every game in the configured seasons.

stats.nba.com refuses most cloud IP ranges, which is why this runs on a laptop. One item per game:
`data/raw/nba/<game_id>.json` holds the endpoint's raw response.

`to_records` turns one raw game into match records. Game time is elapsed game-clock seconds:
12-minute quarters, then 5-minute overtimes. Kinds and attrs are documented in
`intensity.engine.adapters.nba`.
"""

from __future__ import annotations

import re
from typing import Any

from intensity.training.sources.base import Fetcher, FetchError, Retry


class NbaFetcher(Fetcher):
    sport = "nba"

    def list_items(self) -> tuple[list[str], dict[str, Any]]:
        meta: dict[str, Any] = {}
        for season in self.cfg["seasons"]:
            for season_type in self.cfg["season_types"]:
                for row in self._game_rows(season, season_type):
                    game = meta.setdefault(
                        row["GAME_ID"],
                        {"season": season, "season_type": season_type, "date": row["GAME_DATE"], "teams": []},
                    )
                    if row["TEAM_ABBREVIATION"] not in game["teams"]:
                        game["teams"].append(row["TEAM_ABBREVIATION"])
                    if " vs. " in row["MATCHUP"]:
                        game["matchup"] = row["MATCHUP"]  # the home team's row
        items = sorted(meta, key=lambda g: (meta[g]["date"], g))
        return items, meta

    def fetch_item(self, item: str) -> Any:
        data = self._call(f"pbp {item}", lambda: self._pbp(item))
        if not data.get("game", {}).get("actions"):
            raise FetchError(f"{item}: no play-by-play actions")
        return data

    # The two calls into nba_api; tests replace them.

    def _game_rows(self, season: str, season_type: str) -> list[dict[str, Any]]:
        from nba_api.stats.endpoints import leaguegamefinder

        def call() -> list[dict[str, Any]]:
            finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable=season_type,
                league_id_nullable="00",
                timeout=self.timeout_s,
            )
            return finder.get_data_frames()[0].to_dict("records")

        return self._call(f"games {season} {season_type}", call)

    def _pbp(self, game_id: str) -> dict[str, Any]:
        from nba_api.stats.endpoints import playbyplayv3

        return playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=self.timeout_s).get_dict()

    def _call(self, what: str, fn: Any) -> Any:
        def call() -> Any:
            try:
                return fn()
            except FetchError:
                raise
            except Exception as exc:  # requests timeouts, throttling, half-sent JSON
                raise Retry(f"{type(exc).__name__}: {exc}") from None

        return self.with_retries(what, call)


# --------------------------------------------------------------------------- conversion

QUARTER_S = 720.0
OT_S = 300.0
_CLOCK = re.compile(r"PT(\d+)M([\d.]+)S")
_FT_OF = re.compile(r"(\d+) of (\d+)")


def last_free_throw(sub_type: str) -> bool:
    """ "Free Throw 2 of 2" ends the trip and hands the ball over when made; technicals do not."""
    m = _FT_OF.search(sub_type or "")
    return bool(m) and m.group(1) == m.group(2)


def period_start_s(period: int) -> float:
    return (period - 1) * QUARTER_S if period <= 4 else 4 * QUARTER_S + (period - 5) * OT_S


def period_length_s(period: int) -> float:
    return QUARTER_S if period <= 4 else OT_S


def game_time(period: int, clock: str) -> float:
    m = _CLOCK.fullmatch(clock)
    remaining = int(m.group(1)) * 60 + float(m.group(2)) if m else period_length_s(period)
    return round(period_start_s(period) + period_length_s(period) - remaining, 2)


def to_records(raw: dict[str, Any], cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    actions = raw["game"]["actions"]
    side: dict[str, str] = {}
    for a in actions:
        if a.get("teamTricode") and a.get("location") in ("h", "v"):
            side.setdefault(a["location"], a["teamTricode"])
    home, away = side.get("h", "HOME"), side.get("v", "AWAY")
    out: list[dict[str, Any]] = []

    def rec(
        kind: str, a: dict[str, Any], team: str | None = None, player: str | None = None, **attrs: Any
    ) -> None:
        out.append(
            {
                "game_time_s": game_time(a["period"], a["clock"]),
                "period": a["period"],
                "kind": kind,
                "team": team,
                "player": player or None,
                "attrs": {k: v for k, v in attrs.items() if v is not None},
            }
        )

    if actions:
        rec("game_start", {"period": 1, "clock": "PT12M00.00S"}, home=home, away=away)
    h = a_ = 0
    last_miss_team: str | None = None
    for a in actions:
        kind, sub = a.get("actionType", ""), a.get("subType", "")
        team = a.get("teamTricode") or {"h": home, "v": away}.get(a.get("location", ""))
        player = a.get("playerName") or None
        if kind == "period":
            rec(f"period_{sub}", a)
            continue
        if a.get("scoreHome") not in ("", None) and a.get("scoreAway") not in ("", None):
            nh, na = int(a["scoreHome"]), int(a["scoreAway"])
            if (nh, na) != (h, a_):
                points = (nh - h) + (na - a_)
                scorer = home if nh > h else away
                h, a_ = nh, na
                rec(
                    "score",
                    a,
                    scorer,
                    player,
                    points=points,
                    home_score=h,
                    away_score=a_,
                    home=home,
                    away=away,
                    free_throw=kind == "Free Throw" or None,
                    ft_last=(kind == "Free Throw" and last_free_throw(sub)) or None,
                )
                last_miss_team = None
                continue
        if kind == "Missed Shot" or (kind == "Free Throw" and "MISS" in a.get("description", "")):
            last_miss_team = team
            rec(
                "miss",
                a,
                team,
                player,
                shot_value=int(a.get("shotValue") or 0) or None,
                free_throw=kind == "Free Throw" or None,
            )
        elif kind == "Rebound":
            rec("rebound", a, team, player, offensive=team == last_miss_team)
            last_miss_team = None
        elif kind == "Turnover":
            rec("turnover", a, team, player, sub_type=sub or None)
        elif kind == "Foul":
            rec("foul", a, team, player, sub_type=sub or None)
        elif kind == "Timeout":
            rec("timeout", a, team, None, sub_type=sub or None)
    out.sort(key=lambda r: r["game_time_s"])
    return out
