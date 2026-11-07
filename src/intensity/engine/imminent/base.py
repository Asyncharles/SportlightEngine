"""Imminent rules: is an intense moment starting in the next 2 to 10 s?

    cue          = sport rule over building cues, 0 to 1
    ctx          = max(forecast part value, stakes part value)
    context      = context_floor + (1 - context_floor) * ctx
    imminent_raw = cue * context
    imminent     = peak-hold of imminent_raw, decaying to 0 over hold_s

A rule reads only cues and facts at or before the tick time; the engine guarantees that by applying
events only once its clock has reached them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from intensity.config import ImminentConfig
from intensity.engine.adapters.base import clamp01
from intensity.engine.trackers.crowd import CrowdCues
from intensity.engine.trackers.video import VideoCues
from intensity.engine.trackers.voice import VoiceCues
from intensity.schema import Fact, Imminent

AUDIO_ONLY_FACT = Fact(key="imminent_inputs", value="audio only", text="video unavailable, judging by sound alone")


@dataclass
class CueInputs:
    t_video: float
    game_time: float | None
    video: VideoCues | None
    crowd: CrowdCues | None
    voice: VoiceCues | None
    adapter_facts: dict[str, Fact] = field(default_factory=dict)
    forecast_part: float = 0.0  # forecast part value, 0 to 1
    stakes_part: float = 0.0
    forecast_p: float = 0.0  # raw calibrated p (weightlifting context)


@dataclass
class CueResult:
    cue: float
    facts: list[Fact] = field(default_factory=list)
    phase: str | None = None


class ImminentRule(ABC):
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.rule_version: str = cfg.get("rule_version", f"{type(self).__name__}-r0")
        self.threshold: float = float(cfg.get("threshold", 0.5))
        self.reset()

    def reset(self) -> None:  # noqa: B027 - stateless rules need nothing
        pass

    @abstractmethod
    def cue(self, inp: CueInputs) -> CueResult: ...

    def context_value(self, inp: CueInputs) -> float:
        return max(inp.forecast_part, inp.stakes_part)

    # Shared pieces of the football / NBA / F1 rules.
    def weighted(self, terms: dict[str, float | None]) -> tuple[float, bool]:
        """Weighted sum over `cue_weights`; terms that are None are missing. With video missing and
        `audio_only_renormalise`, the remaining weights are renormalised. Returns (cue, audio_only)."""
        weights: dict[str, float] = self.cfg.get("cue_weights", {})
        present = {k: v for k, v in terms.items() if v is not None and k in weights}
        missing = [k for k in weights if terms.get(k) is None]
        if not present:
            return 0.0, bool(missing)
        total = sum(weights[k] * v for k, v in present.items())
        if missing and self.cfg.get("audio_only_renormalise", True):
            norm = sum(weights[k] for k in present)
            return (total / norm if norm else 0.0), True
        return total, bool(missing)


def audio_terms(inp: CueInputs) -> tuple[float | None, float | None, list[Fact]]:
    """crowd_rise and voice_rise terms plus the facts behind them."""
    facts: list[Fact] = []
    crowd = inp.crowd.rise_pct if inp.crowd is not None else None
    if inp.crowd is not None and inp.crowd.rise_db_s >= 0.5 and "rise_db_s" in inp.crowd.facts and crowd and crowd >= 0.7:
        facts.append(inp.crowd.facts["rise_db_s"])
    voice = inp.voice.rise if inp.voice is not None else None
    if inp.voice is not None:
        facts.extend(inp.voice.facts.values())
    return crowd, voice, facts


class ImminentState:
    """Peak-hold, threshold with `min_on_s` and the out-of-cycle trigger, per feed."""

    def __init__(self, rule: ImminentRule, cfg: ImminentConfig):
        self.rule = rule
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.rule.reset()
        self.peak = 0.0
        self.peak_t = -1e9
        self.above_since: float | None = None
        self.heating_up = False
        self.last_trigger_t = -1e9
        self.last_t = -1e9
        self.current = Imminent(value=0.0, cue=0.0, context=self.cfg.context_floor, heating_up=False, phase=None, source="rule", facts=[])

    def held(self, t: float) -> float:
        if self.cfg.hold_s <= 0:
            return 0.0
        return self.peak * max(0.0, 1.0 - (t - self.peak_t) / self.cfg.hold_s)

    def update(self, inp: CueInputs) -> tuple[Imminent, bool]:
        """Advance to `inp.t_video`. Returns the Imminent and whether to send an out-of-cycle update."""
        t = inp.t_video
        if t < self.last_t:  # never step backwards; a loop resets the state first
            return self.current, False
        self.last_t = t
        res = self.rule.cue(inp)
        ctx = clamp01(self.rule.context_value(inp))
        floor = self.cfg.context_floor
        context = floor + (1.0 - floor) * ctx
        raw = res.cue * context
        value = self.held(t)
        if raw >= value:
            self.peak, self.peak_t, value = raw, t, raw

        if value >= self.rule.threshold:
            if self.above_since is None:
                self.above_since = t
        else:
            self.above_since = None
        was = self.heating_up
        self.heating_up = self.above_since is not None and t - self.above_since >= self.cfg.min_on_s
        trigger = self.heating_up and not was and t - self.last_trigger_t >= self.cfg.min_on_s
        if trigger:
            self.last_trigger_t = t
        self.current = Imminent(
            value=round(value, 4),
            cue=round(res.cue, 4),
            context=round(context, 4),
            heating_up=self.heating_up,
            phase=res.phase,
            source="rule",
            facts=res.facts,
        )
        return self.current, trigger
