"""F1 raw data from Jolpica (the Ergast successor): results, lap-by-lap timings and pit stops.

One item per race, `<season>-<round>`. FastF1's live-timing archive now needs an F1 account
(`F1_USERNAME`, `F1_PASSWORD`), so without one the F1 features are built at lap granularity from
these files (approved by the ML lead on 2026-11-07).

`to_records` turns one raw race into match records. Game time is seconds since the start, from
cumulative lap times, so each car's `crossing` record lands when that car crosses the line. Kinds
and attrs are documented in `intensity.engine.adapters.f1`.
"""

from __future__ import annotations

import statistics
from typing import Any

from intensity.training.sources.base import Fetcher, FetchError


class JolpicaFetcher(Fetcher):
    sport = "f1"

    @property
    def base_url(self) -> str:
        return str(self.cfg["base_url"]).rstrip("/")

    @property
    def page_limit(self) -> int:
        return int(self.cfg.get("page_limit", 100))

    def list_items(self) -> tuple[list[str], dict[str, Any]]:
        items: list[str] = []
        meta: dict[str, Any] = {}
        for season in self.cfg["seasons"]:
            races = self._paged(f"{season}", "Races")
            for race in races:
                item = f"{season}-{int(race['round']):02d}"
                items.append(item)
                meta[item] = {
                    k: race.get(k) for k in ("season", "round", "raceName", "date", "time", "Circuit")
                }
        return items, meta

    def fetch_item(self, item: str) -> Any:
        season, rnd = item.split("-")
        path = f"{season}/{int(rnd)}"
        results = self._paged(f"{path}/results", "Races")
        if not results:
            raise FetchError(f"{item}: no results yet")
        return {
            "race": {k: v for k, v in results[0].items() if k != "Results"},
            "results": [r for race in results for r in race.get("Results", [])],
            "laps": _merge_laps(self._paged(f"{path}/laps", "Races")),
            "pitstops": [
                p for race in self._paged(f"{path}/pitstops", "Races") for p in race.get("PitStops", [])
            ],
        }

    def _paged(self, path: str, table_key: str) -> list[dict[str, Any]]:
        """Every row of a paged endpoint. Jolpica caps `limit` at 100 and pages by the innermost rows
        (lap timings, pit stops), so one race can span several pages."""
        out: list[dict[str, Any]] = []
        offset = 0
        while True:
            data = self.get_json(f"{self.base_url}/{path}.json", {"limit": self.page_limit, "offset": offset})
            mr = data["MRData"]
            table = next(v for k, v in mr.items() if k.endswith("Table"))
            out.extend(table.get(table_key, []))
            offset += int(mr["limit"])
            if offset >= int(mr["total"]):
                return out


def _merge_laps(races: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join the lap pages: a lap cut across two pages appears twice with part of its timings each."""
    laps: dict[int, list[dict[str, Any]]] = {}
    for race in races:
        for lap in race.get("Laps", []):
            laps.setdefault(int(lap["number"]), []).extend(lap.get("Timings", []))
    return [{"number": n, "Timings": timings} for n, timings in sorted(laps.items())]


# --------------------------------------------------------------------------- conversion


def lap_seconds(text: str) -> float:
    """ "1:33.813" or "93.813" -> 93.813."""
    parts = text.split(":")
    secs = 0.0
    for part in parts:
        secs = secs * 60 + float(part)
    return secs


def to_records(raw: dict[str, Any], cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = cfg or {}
    slow_ratio = float(cfg.get("sc_slow_ratio", 1.25))
    results = raw["results"]
    total_laps = max((int(r["laps"]) for r in results), default=0)
    info = {
        r["Driver"]["driverId"]: {"name": r["Driver"]["familyName"], "team": r["Constructor"]["name"]}
        for r in results
    }
    name = {d: v["name"] for d, v in info.items()}
    team = {d: v["team"] for d, v in info.items()}
    grid = {r["Driver"]["driverId"]: int(r["grid"]) or 99 for r in results}  # grid 0 = pit lane start
    pits: dict[int, set[str]] = {}
    for p in raw["pitstops"]:
        pits.setdefault(int(p["lap"]), set()).add(p["driverId"])

    out: list[dict[str, Any]] = []

    def rec(kind: str, t: float, player: str | None = None, **attrs: Any) -> None:
        out.append(
            {
                "game_time_s": round(t, 3),
                "period": 1,
                "kind": kind,
                "team": team.get(player) if player else None,
                "player": name.get(player, player) if player else None,
                "attrs": {k: v for k, v in attrs.items() if v is not None},
            }
        )

    by_grid = sorted(grid, key=lambda d: grid[d])
    rec("race_start", 0.0, total_laps=total_laps, grid=[name[d] for d in by_grid], teams=team_by_name(info))

    cum: dict[str, float] = {}
    last_pit: dict[str, int] = {d: 0 for d in info}
    prev_pos = {d: i + 1 for i, d in enumerate(by_grid)}
    green: list[float] = []
    sc_active = False
    for lap in raw["laps"]:
        n = int(lap["number"])
        timings = sorted(lap["Timings"], key=lambda t: int(t["position"]))
        times = {t["driverId"]: lap_seconds(t["time"]) for t in timings}
        for d, secs in times.items():
            cum[d] = cum.get(d, 0.0) + secs
        for d in pits.get(n, ()):
            last_pit[d] = n
        pos = {t["driverId"]: int(t["position"]) for t in timings}

        # One crossing per car, at its own cumulative time; the gap is to the car ahead on the same lap.
        for i, t in enumerate(timings):
            d = t["driverId"]
            ahead = timings[i - 1]["driverId"] if i else None
            rec(
                "crossing",
                cum[d],
                d,
                lap=n,
                position=pos[d],
                lap_time_s=round(times[d], 3),
                gap_ahead_s=round(cum[d] - cum[ahead], 3) if ahead else None,
                tyre_age=n - last_pit[d],
                pitted=d in pits.get(n, ()) or None,
            )
        # A pass: d was behind o after the previous lap and is ahead now, both still running.
        boxed = pits.get(n, set()) | pits.get(n - 1, set())
        for d in pos:
            for o in pos:
                if (
                    d != o
                    and d in prev_pos
                    and o in prev_pos
                    and prev_pos[d] > prev_pos[o]
                    and pos[d] < pos[o]
                ):
                    rec(
                        "pass",
                        cum[d],
                        d,
                        lap=n,
                        position=pos[d],
                        passed=name[o],
                        pit_swap=(d in boxed or o in boxed) or None,
                        start=n == 1 or None,
                    )
        prev_pos = pos

        leader = timings[0]["driverId"]
        # Safety car (or VSC): the lap's median time, pit laps excluded, well above the green median so far.
        # Known only once the last car on the lap has crossed, so the record lands then.
        clean = [times[d] for d in times if d not in boxed]
        if n > 1 and clean:
            median = statistics.median(clean)
            reference = statistics.median(green) if green else median
            slow = median > slow_ratio * reference
            if not slow:
                green.append(median)
            if slow != sc_active:
                sc_active = slow
                rec("safety_car", max(cum[d] for d in times), active=slow, lap=n)
        rec("lap", cum[leader], lap=min(n + 1, total_laps), completed=n, total_laps=total_laps)
    out.sort(key=lambda r: r["game_time_s"])
    return out


def team_by_name(info: dict[str, dict[str, str]]) -> dict[str, str]:
    return {v["name"]: v["team"] for v in info.values()}
