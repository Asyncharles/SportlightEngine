"""Weightlifting adapter (owner: training agent; the session-CSV converter and checks land in T4).

Match records: `attempt_called` (player, attrs: weight_kg, attempt, best_kg, eta_game_s, and
optionally standings_gap_kg), `attempt_result` (player, attrs: good). p(success) is rule-based, a
logistic in kilograms over the athlete's best from `config/adapters/weightlifting.yaml`; there is
no trained model.
"""

from __future__ import annotations

import math

import numpy as np

from intensity.engine.adapters.base import Adapter, AdapterPart, clamp01
from intensity.schema import Fact, MatchPayload


class WeightliftingAdapter(Adapter):
    sport = "weightlifting"
    FEATURES = ("weight_over_best_kg", "attempt", "standings_gap_kg")
    LABEL = "lift_success"
    HORIZON_S = 60.0
    FORECAST_SOURCE = "rule"

    def reset(self) -> None:
        self.athlete: str | None = None
        self.weight = 0.0
        self.attempt = 0
        self.best = 0.0
        self.eta: float | None = None
        self.standings_gap = 0.0
        self.results: list[bool] = []
        self.game_time = 0.0
        self.names: set[str] = set()

    def on_match(self, record: MatchPayload) -> None:
        self.game_time = record.game_time_s
        a = record.attrs
        if record.player:
            self.names.add(record.player)
        if record.kind == "attempt_called":
            self.athlete = record.player
            self.weight = float(a.get("weight_kg", 0))
            self.attempt = int(a.get("attempt", 1))
            self.best = float(a.get("best_kg", self.weight))
            self.eta = float(a["eta_game_s"]) if a.get("eta_game_s") is not None else None
            self.standings_gap = float(a.get("standings_gap_kg", 0.0))
        elif record.kind == "attempt_result":
            self.results.append(bool(a.get("good", False)))
            self.eta = None

    def p_success(self) -> float:
        if not self.weight:
            return 0.5
        p = self.cfg.get("p_success", {})
        z = (self.weight - self.best) / float(p.get("scale_kg", 2.5)) - float(p.get("offset", 0.8))
        return 1.0 / (1.0 + math.exp(z))

    def features(self, game_time: float) -> np.ndarray:
        return self.row(
            {
                "weight_over_best_kg": self.weight - self.best,
                "attempt": self.attempt,
                "standings_gap_kg": self.standings_gap,
            }
        )

    def parts(self, game_time: float) -> dict[str, AdapterPart]:
        p = self.p_success()
        name = self.athlete or "the lifter"
        weight_fact = Fact(
            key="weight_kg", value=int(self.weight), unit="kg", text=f"{name} going for {int(self.weight)} kg"
        )
        attempt_fact = Fact(key="attempt", value=self.attempt, text=f"attempt {self.attempt} of 3")
        over = self.weight - self.best
        over_text = (
            f"{int(abs(over))} kg {'above' if over > 0 else 'below'} best" if over else "matching best"
        )
        over_fact = Fact(key="weight_over_best_kg", value=int(over), unit="kg", text=over_text)
        closeness = clamp01(1 - abs(2 * p - 1))
        recent = self.results[-3:]
        momentum = clamp01(sum(recent) / 3.0)
        good_fact = Fact(
            key="good_lifts", value=sum(recent), text=f"{sum(recent)} of the last {len(recent)} lifts good"
        )
        stakes = clamp01(self.attempt / 3.0)
        return {
            "closeness": AdapterPart(
                closeness,
                "a lift that could go either way" if closeness >= 0.6 else "a safe attempt",
                [weight_fact, over_fact],
            ),
            "momentum": AdapterPart(
                momentum,
                "lifters in form" if momentum >= 0.6 else "a run of misses",
                [good_fact, attempt_fact],
            ),
            "stakes": AdapterPart(
                stakes,
                "a final attempt" if self.attempt >= 3 else "an early attempt",
                [attempt_fact, weight_fact],
            ),
        }

    def facts(self, game_time: float, parts: dict[str, AdapterPart] | None = None) -> dict[str, Fact]:
        out = super().facts(game_time, parts)
        p = self.p_success()
        out["p_success"] = Fact(
            key="p_success", value=round(100 * p), unit="%", text=f"{round(100 * p)}% chance of a good lift"
        )
        if self.eta is not None:
            out["attempt_eta_game_s"] = Fact(
                key="attempt_eta_game_s", value=round(self.eta, 1), unit="s", text="next attempt due"
            )
        return out

    def game_clock(self) -> str | None:
        return f"Attempt {self.attempt}" if self.attempt else None

    def heuristic_forecast(self, game_time: float) -> float:
        return self.p_success()

    def player_names(self) -> set[str]:
        return set(self.names)
