"""Sport adapter interface (owner: training agent; the engine agent wrote the stubs in F1).

An adapter turns the match-record stream into the match-data parts (closeness, momentum, stakes),
the ordered forecast feature row, the game clock and a heuristic forecast. Training featurizes
through the same code (hard rule 5), so an adapter must only ever read records it has been given.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from intensity.schema import Fact, MatchPayload


@dataclass
class AdapterPart:
    value: float  # 0 to 1
    phrase: str
    facts: list[Fact] = field(default_factory=list)


def feature_hash(features: tuple[str, ...] | list[str]) -> str:
    return hashlib.sha256("\n".join(features).encode()).hexdigest()[:16]


def clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


class Adapter(ABC):
    sport: ClassVar[str]
    FEATURES: ClassVar[tuple[str, ...]]
    LABEL: ClassVar[str]
    HORIZON_S: ClassVar[float]
    FORECAST_SOURCE: ClassVar[str] = "heuristic"  # "rule" for sports with no trained model

    def __init__(self) -> None:
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        """Forget everything; called at start and on every loop."""

    @abstractmethod
    def on_match(self, record: MatchPayload) -> None:
        """Consume one match record, in game-time order."""

    @abstractmethod
    def features(self, game_time: float) -> np.ndarray:
        """The forecast feature row at `game_time`, ordered as FEATURES."""

    @abstractmethod
    def parts(self, game_time: float) -> dict[str, AdapterPart]:
        """`closeness`, `momentum` and `stakes`, each with a phrase and facts."""

    @abstractmethod
    def game_clock(self) -> str | None:
        """Human game clock: "88:12", "Lap 41/58", "Q4 2:31"."""

    @abstractmethod
    def heuristic_forecast(self, game_time: float) -> float:
        """p of LABEL within HORIZON_S, used until a trained bundle is approved."""

    def facts(self, game_time: float) -> dict[str, Fact]:
        """Current state facts keyed by name. Keys that match FEATURES let SHAP reasons quote them;
        imminent rules read sport-specific keys (`min_gap_s`, `p_success`, `attempt_eta_game_s`)."""
        out: dict[str, Fact] = {}
        for part in self.parts(game_time).values():
            for fact in part.facts:
                out[fact.key] = fact
        clock = self.game_clock()
        if clock:
            out["game_clock"] = Fact(key="game_clock", value=clock, text=clock)
        return out

    def player_names(self) -> set[str]:
        """Names seen in match records, for the commentary tracker."""
        return set()
