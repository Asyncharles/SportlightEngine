"""Sport adapters, looked up by sport name."""

from __future__ import annotations

import numpy as np

from intensity.engine.adapters.base import Adapter, AdapterPart
from intensity.engine.adapters.f1 import F1Adapter
from intensity.engine.adapters.football import FootballAdapter
from intensity.engine.adapters.nba import NbaAdapter
from intensity.engine.adapters.weightlifting import WeightliftingAdapter
from intensity.schema import MatchPayload


class NullAdapter(Adapter):
    """For sports without an adapter yet (chess): neutral parts, no facts, so no reasons."""

    sport = "chess"
    FEATURES = ()
    LABEL = "swing"
    HORIZON_S = 60.0

    def reset(self) -> None:
        pass

    def on_match(self, record: MatchPayload) -> None:
        pass

    def features(self, game_time: float) -> np.ndarray:
        return np.zeros(0)

    def parts(self, game_time: float) -> dict[str, AdapterPart]:
        return {name: AdapterPart(0.3, "a quiet spell") for name in ("closeness", "momentum", "stakes")}

    def game_clock(self) -> str | None:
        return None

    def heuristic_forecast(self, game_time: float) -> float:
        return 0.05


ADAPTERS: dict[str, type[Adapter]] = {
    "football": FootballAdapter,
    "f1": F1Adapter,
    "nba": NbaAdapter,
    "weightlifting": WeightliftingAdapter,
    "chess": NullAdapter,
}


def make_adapter(sport: str) -> Adapter:
    return ADAPTERS.get(sport, NullAdapter)()


__all__ = ["ADAPTERS", "Adapter", "AdapterPart", "NullAdapter", "make_adapter"]
