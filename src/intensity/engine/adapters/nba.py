"""NBA adapter, STUB from F1 (owner: training agent, who replaces it in S2).

Match records: `score` (team, player, attrs: points, home_score, away_score, home, away). Game time
is elapsed game-clock seconds; 12-minute quarters.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from intensity.engine.adapters.base import Adapter, AdapterPart, clamp01
from intensity.schema import Fact, MatchPayload

QUARTER_S = 720.0


class NbaAdapter(Adapter):
    sport = "nba"
    FEATURES = ("margin", "time_remaining_s", "period", "run_2min", "lead_changes")
    LABEL = "lead_change"
    HORIZON_S = 60.0

    def reset(self) -> None:
        self.home, self.away = "Home", "Away"
        self.home_score = 0
        self.away_score = 0
        self.period = 1
        self.game_time = 0.0
        self.scoring: deque[tuple[float, str, int]] = deque()
        self.names: set[str] = set()

    def on_match(self, record: MatchPayload) -> None:
        self.game_time = record.game_time_s
        self.period = record.period
        if record.player:
            self.names.add(record.player)
        a = record.attrs
        self.home = a.get("home", self.home)
        self.away = a.get("away", self.away)
        if record.kind == "score":
            self.home_score = int(a.get("home_score", self.home_score))
            self.away_score = int(a.get("away_score", self.away_score))
            self.scoring.append((record.game_time_s, record.team or "", int(a.get("points", 2))))

    def features(self, game_time: float) -> np.ndarray:
        return np.zeros(len(self.FEATURES), dtype=float)

    def _run(self, game_time: float) -> tuple[str | None, int]:
        while self.scoring and self.scoring[0][0] < game_time - 120:
            self.scoring.popleft()
        totals: dict[str, int] = {}
        for _, team, pts in self.scoring:
            totals[team] = totals.get(team, 0) + pts
        if not totals:
            return None, 0
        team = max(totals, key=totals.__getitem__)
        others = sum(v for k, v in totals.items() if k != team)
        return team, totals[team] - others

    def parts(self, game_time: float) -> dict[str, AdapterPart]:
        self.game_time = max(self.game_time, game_time)  # the clock runs between records
        self.period = max(self.period, min(4, 1 + int(self.game_time // QUARTER_S)))
        margin = self.home_score - self.away_score
        score = Fact(
            key="score",
            value=f"{self.home_score}-{self.away_score}",
            text=f"{self.home} {self.home_score}-{self.away_score} {self.away}",
        )
        margin_fact = Fact(key="margin", value=abs(margin), unit="pts", text=f"{abs(margin)}-point game")
        run_team, run = self._run(game_time)
        run_fact = (
            Fact(key="run_2min", value=run, unit="pts", text=f"{run_team} on a {run}-point run")
            if run_team
            else Fact(key="run_2min", value=0, unit="pts", text="no scoring run")
        )
        remaining = max(0.0, QUARTER_S - (game_time - (self.period - 1) * QUARTER_S))
        time_fact = Fact(key="time_remaining_s", value=int(remaining), unit="s", text=self.game_clock() or "")
        closeness = clamp01(1.0 - abs(margin) / 15.0)
        momentum = clamp01(run / 10.0)
        stakes = clamp01((self.period - 1) / 3.0 * 0.6 + 0.4 * closeness * (self.period >= 4))
        return {
            "closeness": AdapterPart(closeness, "a one-possession game" if closeness >= 0.8 else "a tight game" if closeness >= 0.5 else "a blowout", [margin_fact, score]),
            "momentum": AdapterPart(momentum, "a big run" if momentum >= 0.6 else "trading baskets", [run_fact, score]),
            "stakes": AdapterPart(stakes, "crunch time" if stakes >= 0.7 else "early in the game", [time_fact, margin_fact]),
        }

    def game_clock(self) -> str | None:
        remaining = max(0.0, QUARTER_S - (self.game_time - (self.period - 1) * QUARTER_S))
        m, s = divmod(int(remaining), 60)
        return f"Q{self.period} {m}:{s:02d}"

    def heuristic_forecast(self, game_time: float) -> float:
        margin = abs(self.home_score - self.away_score)
        _, run = self._run(game_time)
        return clamp01(0.03 + 0.4 * max(0.0, 1 - margin / 6.0) + 0.02 * run)

    def player_names(self) -> set[str]:
        return set(self.names)
