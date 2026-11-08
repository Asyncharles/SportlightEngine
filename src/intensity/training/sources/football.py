"""Football raw data: StatsBomb open data (the 2022 World Cup by default).

Reads the open-data repository's JSON directly. One item per match:
`data/raw/football/<match_id>.json` holds the match row from the competition index, its events and
its lineups (for players' common names).

`to_records` turns one raw match into match records, ordered by period and game time. Game time is
the broadcast clock in seconds: the second half starts at 45:00 and first-half stoppage time runs
past it, so records are ordered by `(period, game_time_s)`. Kinds and attrs are documented in
`intensity.engine.adapters.football`.
"""

from __future__ import annotations

from typing import Any

from intensity.training.sources.base import Fetcher

PERIOD_START_MIN = {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}
CARDS = {"Yellow Card": "yellow", "Second Yellow": "second_yellow", "Red Card": "red"}


class StatsBombFetcher(Fetcher):
    sport = "football"

    @property
    def base_url(self) -> str:
        return str(self.cfg["base_url"]).rstrip("/")

    def list_items(self) -> tuple[list[str], dict[str, Any]]:
        items: list[str] = []
        meta: dict[str, Any] = {}
        for comp in self.cfg["competitions"]:
            url = f"{self.base_url}/matches/{comp['competition_id']}/{comp['season_id']}.json"
            for match in sorted(self.get_json(url), key=lambda m: (m["match_date"], m["match_id"])):
                match_id = str(match["match_id"])
                items.append(match_id)
                meta[match_id] = match
        return items, meta

    def fetch_item(self, item: str) -> Any:
        match = self.index()["meta"].get(item)
        events = self.get_json(f"{self.base_url}/events/{item}.json")
        lineups = self.get_json(f"{self.base_url}/lineups/{item}.json")
        return {"match": match, "events": events, "lineups": lineups}


# --------------------------------------------------------------------------- conversion


def _game_time(ev: dict[str, Any]) -> float:
    h, m, s = ev["timestamp"].split(":")
    return round(PERIOD_START_MIN.get(ev["period"], 0) * 60 + int(h) * 3600 + int(m) * 60 + float(s), 3)


def _xy(loc: list[float] | None) -> dict[str, float]:
    return {"x": round(loc[0], 1), "y": round(loc[1], 1)} if loc else {}


def _end_xy(loc: list[float] | None) -> dict[str, float]:
    return {"end_x": round(loc[0], 1), "end_y": round(loc[1], 1)} if loc else {}


def to_records(raw: dict[str, Any], cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    match = raw.get("match") or {}
    home = match.get("home_team", {}).get("home_team_name", "Home")
    away = match.get("away_team", {}).get("away_team_name", "Away")
    stage = (match.get("competition_stage") or {}).get("name", "")
    names = {
        p["player_id"]: p.get("player_nickname") or p["player_name"]
        for team in raw.get("lineups") or []
        for p in team.get("lineup", [])
    }
    out: list[dict[str, Any]] = []

    def rec(
        kind: str, ev: dict[str, Any], team: str | None = None, player: str | None = None, **attrs: Any
    ) -> None:
        out.append(
            {
                "game_time_s": _game_time(ev),
                "period": ev["period"],
                "kind": kind,
                "team": team,
                "player": player,
                "attrs": {k: v for k, v in attrs.items() if v is not None},
            }
        )

    seen_periods: set[tuple[str, int]] = set()
    for ev in raw["events"]:
        etype = ev["type"]["name"]
        team = (ev.get("team") or {}).get("name")
        pl = ev.get("player")
        player = names.get(pl["id"], pl["name"]) if pl else None
        period = ev["period"]
        if etype in ("Half Start", "Half End"):
            key = ("start" if etype == "Half Start" else "end", period)
            if key in seen_periods:
                continue  # StatsBomb logs one per team
            seen_periods.add(key)
            if etype == "Half Start" and period == 1:
                rec("kickoff", ev, team=home, home=home, away=away, stage=stage)
            rec(f"period_{key[0]}", ev)
        elif etype == "Shot":
            shot = ev["shot"]
            outcome = shot["outcome"]["name"]
            rec(
                "goal" if outcome == "Goal" else "shot",
                ev,
                team,
                player,
                xg=round(float(shot.get("statsbomb_xg", 0.0)), 3),
                outcome=outcome,
                shot_type=shot.get("type", {}).get("name"),
                shootout=period == 5 or None,
                **_xy(ev.get("location")),
            )
        elif etype == "Own Goal For":
            rec("goal", ev, team, None, xg=0.0, own_goal=True)
        elif etype == "Pass":
            p = ev["pass"]
            rec(
                "pass",
                ev,
                team,
                player,
                completed="outcome" not in p,
                pass_type=(p.get("type") or {}).get("name"),
                cross=p.get("cross"),
                **_xy(ev.get("location")),
                **_end_xy(p.get("end_location")),
            )
        elif etype == "Carry":
            rec(
                "carry",
                ev,
                team,
                player,
                **_xy(ev.get("location")),
                **_end_xy(ev["carry"].get("end_location")),
            )
        elif etype == "Pressure":
            rec("pressure", ev, team, player, **_xy(ev.get("location")))
        elif etype in ("Foul Committed", "Bad Behaviour"):
            detail = ev.get("foul_committed" if etype == "Foul Committed" else "bad_behaviour") or {}
            if etype == "Foul Committed":
                rec("foul", ev, team, player, **_xy(ev.get("location")))
            card = CARDS.get((detail.get("card") or {}).get("name", ""))
            if card:
                rec("card", ev, team, player, color=card)
        elif etype == "Substitution":
            replacement = ev["substitution"].get("replacement")
            rec(
                "substitution",
                ev,
                team,
                player,
                replacement=names.get(replacement["id"], replacement["name"]),
            )
    out.sort(key=lambda r: (r["period"], r["game_time_s"]))
    return out
