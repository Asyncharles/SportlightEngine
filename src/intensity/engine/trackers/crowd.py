"""Crowd tracker: the just-happened `crowd` part, plus rise-rate and hush cues for imminent rules."""

from __future__ import annotations

from dataclasses import dataclass, field

from intensity.config import CrowdTrackerConfig, RollingConfig
from intensity.engine.adapters.base import AdapterPart
from intensity.engine.trackers.base import event_fact
from intensity.engine.trackers.rolling import RollingPercentile, Seeds
from intensity.schema import Event, Fact


@dataclass
class CrowdCues:
    rise_pct: float  # percentile of rise_db_s, non-speech frames only
    rise_db_s: float
    hush: bool
    nonspeech_ok: bool
    facts: dict[str, Fact] = field(default_factory=dict)


class CrowdTracker:
    def __init__(self, cfg: CrowdTrackerConfig, rolling: RollingConfig, seeds: Seeds):
        self.cfg = cfg
        self.loudness = RollingPercentile(seeds.get("loudness_rel_db"), rolling)
        self.rise = RollingPercentile(seeds.get("rise_db_s"), rolling)
        self.reset()

    def reset(self) -> None:
        self.loudness.reset()
        self.rise.reset()
        self.last: Event | None = None
        self.last_t = -1e9
        self.last_hush_t = -1e9

    def on_event(self, event: Event) -> None:
        p = event.payload
        self.loudness.add(event.t_video, float(p["loudness_rel_db"]))
        if p.get("nonspeech_ok", True):
            self.rise.add(event.t_video, float(p["rise_db_s"]))
        if p.get("hush"):
            self.last_hush_t = event.t_video
        self.last = event
        self.last_t = event.t_video

    def available(self, now: float) -> bool:
        return self.last is not None and now - self.last_t <= self.cfg.drop_after_s

    def part(self, now: float) -> AdapterPart | None:
        if not self.available(now):
            return None
        assert self.last is not None
        p = self.last.payload
        value = self.loudness.rank(float(p["loudness_rel_db"]))
        facts = []
        above = float(p["loudness_rel_db"]) >= 0
        loud = event_fact(
            self.last, "loudness_rel_db", "crowd {abs} dB above normal" if above else "crowd {abs} dB below normal", "dB"
        )
        if loud is not None:
            facts.append(loud)
        events = p.get("events") or {}
        scores = {k: v for k, v in events.items() if k in self.cfg.events_classes}
        if scores:
            cls, score = max(scores.items(), key=lambda kv: kv[1])
            value = value + (1 - value) * self.cfg.events_weight * float(score)
            if score >= 0.5:
                facts.append(Fact(key=f"audio_{cls}", value=round(float(score), 2), text=f"{cls} heard"))
        rise = event_fact(self.last, "rise_db_s", "crowd rising {value} dB a second", "dB/s")
        if rise is not None and float(p["rise_db_s"]) >= self.cfg.rise_fact_min_db_s:
            facts.append(rise)
        if value >= 0.8:
            phrase = "a roaring crowd"
        elif value >= 0.55:
            phrase = "a lively crowd"
        else:
            phrase = "a quiet crowd"
        return AdapterPart(min(1.0, value), phrase, facts)

    def cues(self, now: float) -> CrowdCues | None:
        if not self.available(now):
            return None
        assert self.last is not None
        p = self.last.payload
        rise = float(p["rise_db_s"])
        hush = bool(p.get("hush")) or now - self.last_hush_t <= 2.0
        facts: dict[str, Fact] = {}
        rise_fact = event_fact(self.last, "rise_db_s", "crowd rising {value} dB a second", "dB/s")
        if rise_fact is not None:
            facts["rise_db_s"] = rise_fact
        if hush:
            facts["hush"] = Fact(key="hush", value="true", text="the crowd goes quiet")
        ok = bool(p.get("nonspeech_ok", True))
        return CrowdCues(self.rise.rank(rise) if ok else 0.5, rise, hush, ok, facts)
