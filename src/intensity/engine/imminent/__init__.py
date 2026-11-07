"""Per-sport imminent rules, thresholds and weights from `config/imminent/<sport>.yaml`."""

from __future__ import annotations

from typing import Any

from intensity.engine.imminent.base import CueInputs, CueResult, ImminentRule, ImminentState
from intensity.engine.imminent.f1 import F1Rule
from intensity.engine.imminent.football import FootballRule
from intensity.engine.imminent.nba import NbaRule
from intensity.engine.imminent.weightlifting import WeightliftingRule

RULES: dict[str, type[ImminentRule]] = {
    "football": FootballRule,
    "f1": F1Rule,
    "nba": NbaRule,
    "weightlifting": WeightliftingRule,
    "chess": FootballRule,  # audio-only via config; threshold above 1 keeps it silent
}


def make_rule(sport: str, cfg: dict[str, Any]) -> ImminentRule:
    return RULES.get(sport, FootballRule)(cfg)


__all__ = ["CueInputs", "CueResult", "ImminentRule", "ImminentState", "make_rule"]
