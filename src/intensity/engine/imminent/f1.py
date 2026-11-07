"""F1: 0.4 x battle_shot + 0.4 x gap_close + 0.2 x voice_rise; 0 during replays.

`battle_shot` is the higher of the battle prompts' percentiles; `gap_close` is 1 when the smallest
gap among the top cars is under `gap_close.full_s`, falling linearly to 0 at `gap_close.zero_s`.
"""

from __future__ import annotations

from intensity.engine.imminent.base import AUDIO_ONLY_FACT, CueInputs, CueResult, ImminentRule
from intensity.schema import Fact


class F1Rule(ImminentRule):
    def cue(self, inp: CueInputs) -> CueResult:
        v = inp.video
        if v is not None and v.replay:
            return CueResult(0.0, [v.facts["replay"]] if "replay" in v.facts else [])
        facts: list[Fact] = []
        battle: float | None = None
        if v is not None:
            prompts = self.cfg.get("battle_prompts", [])
            scores = {p: v.prompt_pcts.get(p, 0.0) for p in prompts}
            name, battle = max(scores.items(), key=lambda kv: kv[1]) if scores else ("", 0.0)
            if battle >= 0.8 and name in v.prompt_scores:
                facts.append(Fact(key=f"prompt:{name}", value=round(v.prompt_scores[name], 2), text=f"{name} on screen"))

        gap_fact = inp.adapter_facts.get("min_gap_s")
        gap_close: float | None = None
        if gap_fact is not None and isinstance(gap_fact.value, int | float):
            g = self.cfg.get("gap_close", {})
            full, zero = float(g.get("full_s", 1.0)), float(g.get("zero_s", 2.0))
            gap = float(gap_fact.value)
            gap_close = 1.0 if gap <= full else max(0.0, (zero - gap) / (zero - full))
            if gap_close > 0:
                facts.append(gap_fact)
        else:
            gap_close = 0.0

        voice = inp.voice.rise if inp.voice is not None else None
        if inp.voice is not None:
            facts.extend(inp.voice.facts.values())
        cue, audio_only = self.weighted({"battle_shot": battle, "gap_close": gap_close, "voice_rise": voice})
        if audio_only and v is None:
            facts.append(AUDIO_ONLY_FACT)
        return CueResult(min(1.0, cue), facts)
