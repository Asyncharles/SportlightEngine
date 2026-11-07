"""Video tracker: the just-happened `video` part, plus pan and sport-prompt cues for imminent rules."""

from __future__ import annotations

from dataclasses import dataclass, field

from intensity.config import RollingConfig, VideoTrackerConfig
from intensity.engine.adapters.base import AdapterPart, clamp01
from intensity.engine.trackers.base import event_fact
from intensity.engine.trackers.rolling import RollingPercentile, Seeds
from intensity.schema import Event, Fact

REPLAY_FACT = Fact(key="replay", value="showing", text="replay showing")


@dataclass
class VideoCues:
    shot_type: str
    replay: bool
    pan_pct: float  # pan-speed percentile; 0 off wide shots
    pan_speed: float
    pan_sustained_s: float
    motion_pct: float
    motion: float
    motion_median: float
    pan_median: float
    prompt_pcts: dict[str, float]
    prompt_scores: dict[str, float]
    facts: dict[str, Fact] = field(default_factory=dict)


class VideoTracker:
    def __init__(self, sport: str, cfg: VideoTrackerConfig, rolling: RollingConfig, seeds: Seeds):
        self.cfg = cfg
        self.rolling = rolling
        self.seeds = seeds
        self.celebration = cfg.celebration_prompts.get(sport, [])
        self.motion = RollingPercentile(seeds.get("motion"), rolling)
        self.pan = RollingPercentile(seeds.get("pan_speed"), rolling)
        self.cuts = RollingPercentile(seeds.get("cuts_per_10s"), rolling)
        self.prompts: dict[str, RollingPercentile] = {}
        self.reset()

    def reset(self) -> None:
        for p in (self.motion, self.pan, self.cuts, *self.prompts.values()):
            p.reset()
        self.last: Event | None = None
        self.last_t = -1e9
        self.value = 0.0
        self.pre_replay_value: float | None = None
        self.last_facts: list[Fact] = []

    def _prompt_pct(self, name: str) -> RollingPercentile:
        if name not in self.prompts:
            self.prompts[name] = RollingPercentile(self.seeds.get(f"prompt:{name}"), self.rolling)
        return self.prompts[name]

    def on_event(self, event: Event) -> None:
        p = event.payload
        t = event.t_video
        wide = p["shot_type"] == "wide"
        if wide:
            self.motion.add(t, float(p["motion"]))
            self.pan.add(t, float(p["pan_speed"]))
        self.cuts.add(t, float(p["cuts_per_10s"]))
        for name, score in p["prompts"].items():
            self._prompt_pct(name).add(t, float(score))

        if p.get("score") is not None:  # trained head (T5) replaces the v0 combination
            value, facts = clamp01(float(p["score"])), []
        else:
            value, facts = self._combine(event)

        if p["replay"]:
            if self.pre_replay_value is None:
                self.pre_replay_value = self.value
            self.value = self.pre_replay_value
            self.last_facts = [f for f in self.last_facts if f.key != "replay"] + [REPLAY_FACT]
        else:
            self.pre_replay_value = None
            self.value = value
            self.last_facts = facts
        self.last = event
        self.last_t = t

    def _combine(self, event: Event) -> tuple[float, list[Fact]]:
        p = event.payload
        w = self.cfg.weights
        terms: dict[str, float] = {}
        facts: list[Fact] = []
        if p["shot_type"] == "wide":
            terms["motion"] = self.motion.rank(float(p["motion"]))
        cel_scores = {n: float(p["prompts"][n]) for n in self.celebration if n in p["prompts"]}
        if cel_scores:
            name, score = max(cel_scores.items(), key=lambda kv: kv[1])
            terms["celebration"] = self._prompt_pct(name).rank(score)
            if terms["celebration"] >= 0.8:
                facts.append(Fact(key=f"prompt:{name}", value=round(score, 2), text=f"{name} on screen"))
        terms["close_up"] = float(p["shot_conf"]) if p["shot_type"] in self.cfg.close_up_shots else 0.0
        terms["cuts"] = self.cuts.rank(float(p["cuts_per_10s"]))
        used = {k: v for k, v in terms.items() if w.get(k, 0.0) > 0}
        total_w = sum(w[k] for k in used)
        value = sum(w[k] * v for k, v in used.items()) / total_w if total_w else 0.0
        if terms.get("motion", 0.0) >= 0.7:
            motion = event_fact(event, "motion", "fast action on a wide shot")
            if motion is not None:
                facts.append(motion)
        shot = event_fact(event, "shot_type", "{value} shot")
        if shot is not None:
            facts.append(shot)
        if w.get("cuts", 0.0) > 0:
            cuts = event_fact(event, "cuts_per_10s", "{value} cuts in 10 s")
            if cuts is not None:
                facts.append(cuts)
        return clamp01(value), facts

    def available(self, now: float) -> bool:
        return self.last is not None and now - self.last_t <= self.cfg.drop_after_s

    def part(self, now: float) -> AdapterPart | None:
        if not self.available(now):
            return None
        v = self.value
        phrase = "frantic camera action" if v >= 0.75 else "lively pictures" if v >= 0.5 else "calm pictures"
        if self.pre_replay_value is not None:
            phrase = "a replay of the action"
        return AdapterPart(v, phrase, list(self.last_facts))

    def cues(self, now: float) -> VideoCues | None:
        if not self.available(now):
            return None
        assert self.last is not None
        p = self.last.payload
        wide = p["shot_type"] == "wide"
        facts: dict[str, Fact] = {}
        pan_pct = self.pan.rank(float(p["pan_speed"])) if wide else 0.0
        if wide:
            facts["shot_type"] = event_fact(self.last, "shot_type", "{value} shot") or Fact(
                key="shot_type", value="wide", text="wide shot"
            )
            pan = event_fact(self.last, "pan_speed", "camera panning {value} frame widths a second", "fw/s")
            if pan is not None:
                facts["pan_speed"] = pan
        if p["replay"]:
            facts["replay"] = REPLAY_FACT
        prompt_scores = {k: float(v) for k, v in p["prompts"].items()}
        return VideoCues(
            shot_type=p["shot_type"],
            replay=bool(p["replay"]),
            pan_pct=pan_pct,
            pan_speed=float(p["pan_speed"]),
            pan_sustained_s=float(p["pan_sustained_s"]) if wide else 0.0,
            motion_pct=self.motion.rank(float(p["motion"])) if wide else 0.0,
            motion=float(p["motion"]),
            motion_median=self.motion.median(),
            pan_median=self.pan.median(),
            prompt_pcts={k: self._prompt_pct(k).rank(v) for k, v in prompt_scores.items()},
            prompt_scores=prompt_scores,
            facts=facts,
        )
