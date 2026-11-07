"""Voice tracker: commentator energy, pitch and pace percentiles for the imminent rules (not a part)."""

from __future__ import annotations

from dataclasses import dataclass, field

from intensity.config import RollingConfig
from intensity.engine.trackers.base import event_fact
from intensity.engine.trackers.rolling import RollingPercentile, Seeds
from intensity.schema import Event, Fact

_FIELDS = ("energy_rel_db", "pitch_rel_st", "pace_rel")


@dataclass
class VoiceCues:
    speaking: bool
    energy_pct: float | None
    pitch_pct: float | None
    pace_pct: float | None
    facts: dict[str, Fact] = field(default_factory=dict)

    @property
    def rise(self) -> float:
        """`voice_rise`: mean of the energy and pitch percentiles while speaking, else 0."""
        if not self.speaking:
            return 0.0
        vals = [v for v in (self.energy_pct, self.pitch_pct) if v is not None]
        return sum(vals) / len(vals) if vals else 0.0


class VoiceTracker:
    def __init__(self, rolling: RollingConfig, seeds: Seeds, drop_after_s: float = 3.0):
        self.pcts = {f: RollingPercentile(seeds.get(f), rolling) for f in _FIELDS}
        self.drop_after_s = drop_after_s
        self.reset()

    def reset(self) -> None:
        for p in self.pcts.values():
            p.reset()
        self.last: Event | None = None
        self.last_t = -1e9

    def on_event(self, event: Event) -> None:
        if event.payload.get("speaking"):
            for name, pct in self.pcts.items():
                value = event.payload.get(name)
                if value is not None:
                    pct.add(event.t_video, float(value))
        self.last = event
        self.last_t = event.t_video

    def cues(self, now: float) -> VoiceCues | None:
        if self.last is None or now - self.last_t > self.drop_after_s:
            return None
        p = self.last.payload
        if not p.get("speaking"):
            return VoiceCues(False, None, None, None)
        ranks = {f: (self.pcts[f].rank(float(p[f])) if p.get(f) is not None else None) for f in _FIELDS}
        facts: dict[str, Fact] = {}
        texts = {
            "energy_rel_db": ("commentator {value} dB louder than usual", "dB"),
            "pitch_rel_st": ("commentator voice up {value} semitones", "st"),
            "pace_rel": ("commentator talking {value} times faster", None),
        }
        for f, (text, unit) in texts.items():
            if ranks[f] is not None and ranks[f] >= 0.7:
                fact = event_fact(self.last, f, text, unit)
                if fact is not None:
                    facts[f] = fact
        return VoiceCues(True, ranks["energy_rel_db"], ranks["pitch_rel_st"], ranks["pace_rel"], facts)
