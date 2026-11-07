"""F1 adapter, STUB from F1 (owner: training agent, who replaces it in S2).

Match records: `lap` (attrs: lap, total_laps, gaps_s = gaps to the car ahead for the top cars),
`pass` (team, player), `safety_car` (attrs: active).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from intensity.engine.adapters.base import Adapter, AdapterPart, clamp01
from intensity.schema import Fact, MatchPayload


class F1Adapter(Adapter):
    sport = "f1"
    FEATURES = ("min_gap_s", "battles_1s", "laps_remaining", "safety_car", "passes_3laps")
    LABEL = "on_track_pass"
    HORIZON_S = 180.0

    def reset(self) -> None:
        self.lap = 0
        self.total_laps = 0
        self.gaps: list[float] = []
        self.safety_car = False
        self.passes: deque[float] = deque()
        self.game_time = 0.0
        self.names: set[str] = set()

    def on_match(self, record: MatchPayload) -> None:
        self.game_time = record.game_time_s
        if record.player:
            self.names.add(record.player)
        if record.kind == "lap":
            self.lap = int(record.attrs.get("lap", self.lap))
            self.total_laps = int(record.attrs.get("total_laps", self.total_laps))
            self.gaps = [float(g) for g in record.attrs.get("gaps_s", [])]
        elif record.kind == "pass":
            self.passes.append(record.game_time_s)
        elif record.kind == "safety_car":
            self.safety_car = bool(record.attrs.get("active", True))

    def min_gap(self) -> float | None:
        return min(self.gaps) if self.gaps else None

    def features(self, game_time: float) -> np.ndarray:
        return np.zeros(len(self.FEATURES), dtype=float)

    def parts(self, game_time: float) -> dict[str, AdapterPart]:
        while self.passes and self.passes[0] < game_time - 270:
            self.passes.popleft()
        gap = self.min_gap()
        battles = sum(1 for g in self.gaps if g < 1.0)
        remaining = max(0, self.total_laps - self.lap)
        lap_fact = Fact(key="lap", value=self.lap, text=f"lap {self.lap} of {self.total_laps}")
        if gap is None:
            closeness, gap_facts = 0.3, [lap_fact]
        else:
            closeness = clamp01(1.0 - (gap - 0.3) / 2.0)
            gap_facts = [Fact(key="min_gap_s", value=round(gap, 1), unit="s", text=f"gap {gap:.1f} s at the front")]
        battles_fact = Fact(key="battles_1s", value=battles, text=f"{battles} battles within a second")
        passes_fact = Fact(key="passes_3laps", value=len(self.passes), text=f"{len(self.passes)} passes in 3 laps")
        momentum = clamp01(0.5 * len(self.passes) / 3.0 + 0.5 * battles / 3.0)
        stakes = clamp01(1.0 - remaining / max(self.total_laps, 1)) if self.total_laps else 0.3
        remaining_fact = Fact(key="laps_remaining", value=remaining, text=f"{remaining} laps to go")
        return {
            "closeness": AdapterPart(
                closeness, "cars nose to tail" if closeness >= 0.6 else "a spread-out field", gap_facts + [battles_fact]
            ),
            "momentum": AdapterPart(
                momentum, "lots of overtaking" if momentum >= 0.5 else "a processional spell", [passes_fact, battles_fact]
            ),
            "stakes": AdapterPart(
                stakes, "the closing laps" if stakes >= 0.7 else "a long way to go", [remaining_fact, lap_fact]
            ),
        }

    def facts(self, game_time: float) -> dict[str, Fact]:
        out = super().facts(game_time)
        if self.safety_car:
            out["safety_car"] = Fact(key="safety_car", value="out", text="safety car out")
        return out

    def game_clock(self) -> str | None:
        return f"Lap {self.lap}/{self.total_laps}" if self.total_laps else None

    def heuristic_forecast(self, game_time: float) -> float:
        if self.safety_car:
            return 0.02
        battles = sum(1 for g in self.gaps if g < 1.0)
        return clamp01(0.05 + 0.15 * battles)

    def player_names(self) -> set[str]:
        return set(self.names)
