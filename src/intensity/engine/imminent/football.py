"""Football: 0.5 x pan + 0.25 x crowd_rise + 0.25 x voice_rise; 0 during replays.

`pan` counts only on wide shots with the pan sustained for `pan_min_sustained_s`.
"""

from __future__ import annotations

from intensity.engine.imminent.base import AUDIO_ONLY_FACT, CueInputs, CueResult, ImminentRule, audio_terms


class PanRule(ImminentRule):
    def cue(self, inp: CueInputs) -> CueResult:
        v = inp.video
        if v is not None and v.replay:
            return CueResult(0.0, [v.facts["replay"]] if "replay" in v.facts else [])
        crowd, voice, facts = audio_terms(inp)
        pan: float | None = None
        if v is not None:
            sustained = v.shot_type == "wide" and v.pan_sustained_s >= float(self.cfg.get("pan_min_sustained_s", 1.5))
            pan = v.pan_pct if sustained else 0.0
            if sustained and pan >= 0.7:
                facts = [f for k in ("shot_type", "pan_speed") if (f := v.facts.get(k))] + facts
        cue, audio_only = self.weighted({"pan": pan, "crowd_rise": crowd, "voice_rise": voice})
        if audio_only and v is None:
            facts.append(AUDIO_ONLY_FACT)
        return CueResult(min(1.0, cue), facts)


class FootballRule(PanRule):
    pass
