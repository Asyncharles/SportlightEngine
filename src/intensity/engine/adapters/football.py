"""Football adapter, STUB from F1 (owner: training agent, who replaces it in S2).

Tracks score and recent shots from match records so the parts move plausibly; the feature row is
constant until the real adapter lands.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from intensity.engine.adapters.base import Adapter, AdapterPart, clamp01
from intensity.schema import Fact, MatchPayload

_WORDS = {1: "one", 2: "two", 3: "three"}


class FootballAdapter(Adapter):
    sport = "football"
    FEATURES = (
        "score_diff",
        "minute",
        "period",
        "shots_2min_home",
        "shots_2min_away",
        "xg_5min_home",
        "xg_5min_away",
    )
    LABEL = "big_chance"
    HORIZON_S = 30.0

    def reset(self) -> None:
        self.teams: list[str] = []
        self.goals: dict[str, int] = {}
        self.shots: deque[tuple[float, str, float]] = deque()  # (game_time, team, xg)
        self.period = 1
        self.game_time = 0.0
        self.names: set[str] = set()

    def on_match(self, record: MatchPayload) -> None:
        self.game_time = record.game_time_s
        self.period = record.period
        team = record.team
        if team and team not in self.teams:
            self.teams.append(team)
            self.goals.setdefault(team, 0)
        if record.player:
            self.names.add(record.player)
        if record.kind in ("shot", "goal") and team:
            self.shots.append((record.game_time_s, team, float(record.attrs.get("xg", 0.05))))
        if record.kind == "goal" and team:
            self.goals[team] = self.goals.get(team, 0) + 1

    def _prune(self, game_time: float) -> None:
        while self.shots and self.shots[0][0] < game_time - 300:
            self.shots.popleft()

    def _score(self) -> tuple[str, str, int, int]:
        home, away = (self.teams + ["Home", "Away"])[:2]
        return home, away, self.goals.get(home, 0), self.goals.get(away, 0)

    def features(self, game_time: float) -> np.ndarray:
        return np.zeros(len(self.FEATURES), dtype=float)

    def parts(self, game_time: float) -> dict[str, AdapterPart]:
        self.game_time = max(self.game_time, game_time)  # the clock runs between records
        self._prune(game_time)
        home, away, gh, ga = self._score()
        diff = gh - ga
        minute = int(game_time // 60)
        score_text = f"{home} {gh}-{ga} {away}"
        if diff == 0:
            score_fact = Fact(key="score_diff", value=0, text=f"level at {gh}-{ga}")
        else:
            leader, margin = (home, diff) if diff > 0 else (away, -diff)
            score_fact = Fact(
                key="score_diff", value=diff, text=f"{leader} lead by {_WORDS.get(margin, str(margin))}"
            )
        minute_fact = Fact(key="minute", value=minute, unit="min", text=f"minute {minute}")
        score = Fact(key="score", value=f"{gh}-{ga}", text=score_text)
        closeness = clamp01(1.0 - abs(diff) / 3.0)

        recent = [s for s in self.shots if s[0] >= game_time - 120]
        momentum = clamp01(len(recent) / 4.0)
        shots_fact = Fact(key="shots_2min", value=len(recent), text=f"{len(recent)} shot{'s' if len(recent) != 1 else ''} in 2 minutes")

        late = clamp01((minute - 60) / 30.0)
        stakes = clamp01(0.2 + 0.6 * late * closeness + 0.2 * (self.period >= 3))

        return {
            "closeness": AdapterPart(
                closeness, "a tight scoreline" if closeness >= 0.6 else "a one-sided game", [score_fact, score]
            ),
            "momentum": AdapterPart(
                momentum, "end-to-end attacking" if momentum >= 0.5 else "a slow spell", [shots_fact, minute_fact]
            ),
            "stakes": AdapterPart(stakes, "late drama" if stakes >= 0.6 else "plenty of time left", [minute_fact, score_fact]),
        }

    def game_clock(self) -> str | None:
        m, s = divmod(int(self.game_time), 60)
        return f"{m}:{s:02d}"

    def heuristic_forecast(self, game_time: float) -> float:
        self._prune(game_time)
        recent = [s for s in self.shots if s[0] >= game_time - 120]
        xg = sum(s[2] for s in recent)
        return clamp01(0.04 + 0.08 * len(recent) + 0.5 * xg)

    def player_names(self) -> set[str]:
        return set(self.names)
