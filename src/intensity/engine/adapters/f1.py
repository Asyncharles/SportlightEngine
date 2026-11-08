"""F1 adapter (owner: training agent).

Match records (from `training/sources/f1.py`, at lap granularity; game time is seconds since the
start):

- `race_start` (attrs: total_laps, grid, teams)
- `crossing` (player, team, attrs: lap, position, lap_time_s, gap_ahead_s, tyre_age, pitted), one
  per car per lap, when that car crosses the line
- `lap` (attrs: lap, completed, total_laps, and optionally gaps_s: gaps to the car ahead for the top
  cars, which the mock sends instead of crossings)
- `pass` (player, team, attrs: lap, position, passed, pit_swap, start)
- `safety_car` (attrs: active, lap)

On-track passes exclude pit swaps and first-lap position changes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from intensity.engine.adapters.base import Adapter, AdapterPart, clamp01, plural
from intensity.schema import Fact, MatchPayload


@dataclass
class _Car:
    lap: int = 0
    position: int = 99
    gap_ahead: float | None = None
    tyre_age: int = 0
    pitted_lap: int = -1


class F1Adapter(Adapter):
    sport = "f1"
    FEATURES = (
        "min_gap_s",
        "battles_1s",
        "battles_1s_all",
        "gap_leader_s",
        "laps_remaining",
        "race_progress",
        "safety_car",
        "laps_since_sc",
        "passes_3laps",
        "passes_1lap",
        "pit_stops_1lap",
        "tyre_delta_battles",
        "max_tyre_delta_battle",
        "cars_running",
    )
    LABEL = "on_track_pass"
    HORIZON_S = 180.0

    def reset(self) -> None:
        self.lap = 0
        self.total_laps = 0
        self.cars: dict[str, _Car] = {}
        self.team: dict[str, str] = {}
        self.gaps_override: list[float] | None = None
        self.safety_car = False
        self.sc_end_lap: int | None = None
        self.passes: deque[tuple[int, str, str]] = deque()  # (lap, passer, passed)
        self.game_time = 0.0
        self.names: set[str] = set()

    # ------------------------------------------------------------------ records

    def on_match(self, record: MatchPayload) -> None:
        a, kind = record.attrs, record.kind
        self.game_time = record.game_time_s
        if record.player:
            self.names.add(record.player)
            if record.team:
                self.team[record.player] = record.team
        if kind == "race_start":
            self.total_laps = int(a.get("total_laps", self.total_laps))
            self.team.update(a.get("teams", {}))
            self.names.update(a.get("grid", []))
        elif kind == "lap":
            self.lap = max(self.lap, int(a.get("lap", self.lap)))
            self.total_laps = int(a.get("total_laps", self.total_laps))
            if "gaps_s" in a:
                self.gaps_override = [float(g) for g in a["gaps_s"]]
        elif kind == "crossing" and record.player:
            car = self.cars.setdefault(record.player, _Car())
            car.lap = int(a.get("lap", car.lap))
            car.position = int(a.get("position", car.position))
            gap = a.get("gap_ahead_s")
            car.gap_ahead = float(gap) if gap is not None else None
            car.tyre_age = int(a.get("tyre_age", car.tyre_age))
            if a.get("pitted"):
                car.pitted_lap = car.lap
            if car.position == 1:
                self.lap = max(self.lap, min(car.lap + 1, self.total_laps or car.lap + 1))
        elif kind == "pass" and record.player:
            if not a.get("pit_swap") and not a.get("start"):
                self.passes.append((int(a.get("lap", self.lap)), record.player, str(a.get("passed", ""))))
        elif kind == "safety_car":
            active = bool(a.get("active", True))
            if self.safety_car and not active:
                self.sc_end_lap = int(a.get("lap", self.lap))
            self.safety_car = active

    # ------------------------------------------------------------------ state

    def _order(self) -> list[tuple[str, _Car]]:
        """Running order: most laps first, then position at that lap."""
        return sorted(self.cars.items(), key=lambda kv: (-kv[1].lap, kv[1].position))

    def _battles(self) -> tuple[list[tuple[str, str, float]], list[float], float | None]:
        """(chaser, car ahead, gap) for the top N, every known gap to the car ahead, and the gap
        between the first two cars. Each car's gap is from its latest crossing."""
        top_n = int(self.cfg.get("top_n", 10))
        if self.gaps_override is not None and not self.cars:  # the mock's aggregate `lap` records
            gaps = self.gaps_override
            return [("", "", g) for g in gaps[:top_n]], list(gaps), gaps[0] if gaps else None
        order = self._order()
        top, all_gaps = [], []
        leader_gap = None
        for i, (name, car) in enumerate(order):
            if not i or car.gap_ahead is None or car.lap < self.lap - 2:
                continue  # the leader, no car ahead on the lap, or retired
            all_gaps.append(car.gap_ahead)
            if i == 1:
                leader_gap = car.gap_ahead
            if i < top_n:
                top.append((name, order[i - 1][0], car.gap_ahead))
        return top, all_gaps, leader_gap

    def min_gap(self) -> float | None:
        top, _, _ = self._battles()
        return min(g for _, _, g in top) if top else None

    def _values(self) -> dict[str, float]:
        cfg = self.cfg
        battle_gap = float(cfg.get("battle_gap_s", 1.0))
        no_gap = float(cfg.get("no_gap_s", 99.0))
        top, all_gaps, leader_gap = self._battles()
        battles = [(c, a, g) for c, a, g in top if g < battle_gap]
        deltas = [
            abs(self.cars[c].tyre_age - self.cars[a].tyre_age)
            for c, a, _ in battles
            if c in self.cars and a in self.cars
        ]
        window = int(cfg.get("passes_window_laps", 3))
        while self.passes and self.passes[0][0] < self.lap - window:
            self.passes.popleft()
        completed = self.lap - 1
        cap = int(cfg.get("laps_since_sc_cap", 20))
        since_sc = (
            0
            if self.safety_car
            else min(cap, completed - self.sc_end_lap)
            if self.sc_end_lap is not None
            else cap
        )
        return {
            "min_gap_s": min((g for _, _, g in top), default=no_gap),
            "battles_1s": len(battles),
            "battles_1s_all": sum(1 for g in all_gaps if g < battle_gap),
            "gap_leader_s": leader_gap if leader_gap is not None else no_gap,
            "laps_remaining": max(0, self.total_laps - self.lap),
            "race_progress": clamp01(completed / self.total_laps) if self.total_laps else 0.0,
            "safety_car": float(self.safety_car),
            "laps_since_sc": since_sc,
            "passes_3laps": len(self.passes),
            "passes_1lap": sum(1 for lap, _, _ in self.passes if lap >= completed),
            "pit_stops_1lap": sum(1 for c in self.cars.values() if c.pitted_lap >= completed > 0),
            "tyre_delta_battles": float(np.mean(deltas)) if deltas else 0.0,
            "max_tyre_delta_battle": max(deltas, default=0),
            "cars_running": sum(1 for c in self.cars.values() if c.lap >= completed - 1) if self.cars else 0,
        }

    def features(self, game_time: float) -> np.ndarray:
        return self.row(self._values())

    # ------------------------------------------------------------------ parts and facts

    def _gap_fact(self) -> Fact | None:
        top, _, _ = self._battles()
        if not top:
            return None
        chaser, ahead, gap = min(top, key=lambda b: b[2])
        text = f"{chaser} {gap:.1f} s behind {ahead}" if chaser and ahead else f"gap {gap:.1f} s at the front"
        return Fact(key="min_gap_s", value=round(gap, 1), unit="s", text=text)

    def parts(self, game_time: float) -> dict[str, AdapterPart]:
        self.game_time = max(self.game_time, game_time)
        v = self._values()
        cfg = self.cfg
        battles, passes = int(v["battles_1s"]), int(v["passes_3laps"])
        remaining = int(v["laps_remaining"])
        lap_fact = Fact(key="lap", value=self.lap, text=f"lap {self.lap} of {self.total_laps}")
        battles_fact = Fact(
            key="battles_1s", value=battles, text=f"{plural(battles, 'battle')} within a second"
        )
        passes_fact = Fact(
            key="passes_3laps", value=passes, text=f"{plural(passes, 'pass', 'passes')} in 3 laps"
        )
        remaining_fact = Fact(
            key="laps_remaining",
            value=remaining,
            text=f"{plural(remaining, 'lap')} to go" if remaining else "final lap",
        )

        c = cfg.get("closeness", {})
        gap_fact = self._gap_fact()
        if gap_fact is None:
            closeness, close_facts = 0.3, [lap_fact, battles_fact]
        else:
            full, span = float(c.get("gap_full_s", 0.3)), float(c.get("gap_span_s", 2.0))
            closeness = clamp01(1.0 - (float(gap_fact.value) - full) / span)
            close_facts = [gap_fact, battles_fact]

        m = cfg.get("momentum", {})
        momentum = clamp01(
            0.5 * passes / float(m.get("passes_full", 3)) + 0.5 * battles / float(m.get("battles_full", 3))
        )

        s = cfg.get("stakes", {})
        if self.total_laps:
            leader_close = 0.0
            if v["gap_leader_s"] < float(cfg.get("no_gap_s", 99.0)):
                leader_close = clamp01(1.0 - v["gap_leader_s"] / float(s.get("leader_gap_zero_s", 5.0)))
            stakes = clamp01(
                float(s.get("progress_weight", 0.7)) * (1.0 - remaining / self.total_laps)
                + float(s.get("leader_weight", 0.3)) * leader_close
            )
        else:
            stakes = 0.3
        return {
            "closeness": AdapterPart(
                closeness,
                "cars nose to tail" if closeness >= float(c.get("phrase_at", 0.6)) else "a spread-out field",
                close_facts,
            ),
            "momentum": AdapterPart(
                momentum,
                "lots of overtaking"
                if momentum >= float(m.get("phrase_at", 0.5))
                else "a processional spell",
                [passes_fact, battles_fact],
            ),
            "stakes": AdapterPart(
                stakes,
                "the closing laps" if stakes >= float(s.get("phrase_at", 0.7)) else "a long way to go",
                [remaining_fact, lap_fact],
            ),
        }

    def facts(self, game_time: float, parts: dict[str, AdapterPart] | None = None) -> dict[str, Fact]:
        out = super().facts(game_time, parts)
        v = self._values()
        if self.safety_car:
            out["safety_car"] = Fact(key="safety_car", value="out", text="safety car out")
        if v["max_tyre_delta_battle"] > 0:
            n = int(v["max_tyre_delta_battle"])
            out["max_tyre_delta_battle"] = Fact(
                key="max_tyre_delta_battle", value=n, text=f"{n} laps' tyre difference in a battle"
            )
        if v["pit_stops_1lap"]:
            n = int(v["pit_stops_1lap"])
            out["pit_stops_1lap"] = Fact(
                key="pit_stops_1lap", value=n, text=f"{plural(n, 'pit stop')} last lap"
            )
        return out

    def game_clock(self) -> str | None:
        return f"Lap {self.lap}/{self.total_laps}" if self.total_laps else None

    def heuristic_forecast(self, game_time: float) -> float:
        h = self.cfg.get("heuristic", {})
        if self.safety_car:
            return float(h.get("safety_car", 0.02))
        return clamp01(
            float(h.get("base", 0.05)) + float(h.get("per_battle", 0.15)) * self._values()["battles_1s"]
        )

    def player_names(self) -> set[str]:
        return set(self.names)
