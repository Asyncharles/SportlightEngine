"""Weightlifting: a phase state machine driven by prompt scores.

| phase      | enter when                                              | cue                     |
| idle       | "empty platform" leads                                  | 0                       |
| walk_on    | an athlete prompt leads after idle                      | 0.4                     |
| setup      | "athlete chalking hands" / "standing over the barbell"  | 0.7                     |
| setup_hush | setup and crowd hush                                    | 1.0                     |
| lift       | a pan or motion spike on the platform after setup       | 1.0, decaying over hold |

A spike is a percentile at or above `lift_spike_pct` and a value at least `lift_spike_ratio` times
the feed's rolling median.

A candidate phase must lead for `phase_hold_s` before the phase moves. The lift is the exception:
the spike that marks it is brief, so it enters on the spike itself. The session's ETA stays as a
prior: if the video hasn't reached walk_on `eta_prior.late_after_s` after the ETA, the cue holds
`eta_prior.cue` until it does. Context is 1 - |2p - 1| from the rule-based p(success).
"""

from __future__ import annotations

from intensity.engine.imminent.base import CueInputs, CueResult, ImminentRule
from intensity.schema import Fact

PHASE_TEXT = {
    "idle": "platform empty",
    "walk_on": "athlete walking out",
    "setup": "athlete at the bar",
    "setup_hush": "athlete at the bar, crowd hushed",
    "lift": "the lift is under way",
}


class WeightliftingRule(ImminentRule):
    def reset(self) -> None:
        self.phase = "idle"
        self.candidate: str | None = None
        self.candidate_since = 0.0
        self.lift_t = -1e9

    def context_value(self, inp: CueInputs) -> float:
        return 1.0 - abs(2.0 * inp.forecast_p - 1.0)

    def _leading(self, inp: CueInputs) -> str | None:
        v = inp.video
        if v is None:
            return None
        prompts = self.cfg.get("prompts", {})
        groups = {
            "empty": prompts.get("empty", []),
            "setup": prompts.get("setup", []),
            "athlete": [p for p in prompts.get("athlete", []) if p not in prompts.get("setup", [])],
        }
        best_group, best = None, -1.0
        for group, names in groups.items():
            for name in names:
                score = v.prompt_scores.get(name)
                if score is not None and score > best:
                    best_group, best = group, score
        return best_group

    def _target(self, inp: CueInputs, leading: str | None) -> str:
        hush = inp.crowd is not None and inp.crowd.hush
        if leading == "empty":
            return "idle"
        if leading == "setup":
            return "setup_hush" if hush else "setup"
        if leading == "athlete":
            return "walk_on" if self.phase in ("idle", "walk_on") else self.phase
        return self.phase

    def cue(self, inp: CueInputs) -> CueResult:
        t = inp.t_video
        cues = self.cfg.get("phase_cues", {})
        hold = float(self.cfg.get("phase_hold_s", 1.0))
        v = inp.video

        if self.phase in ("setup", "setup_hush") and v is not None:
            spike = float(self.cfg.get("lift_spike_pct", 0.9))
            ratio = float(self.cfg.get("lift_spike_ratio", 2.0))
            # A high percentile alone is noise on a still platform: the value must also jump.
            motion_spike = v.motion_pct >= spike and v.motion >= ratio * v.motion_median
            pan_spike = v.pan_pct >= spike and v.pan_speed >= ratio * v.pan_median
            if motion_spike or pan_spike:
                self.phase, self.lift_t, self.candidate = "lift", t, None

        if self.phase == "lift":
            if t - self.lift_t >= float(self.cfg.get("lift_reset_s", 15.0)):
                self.phase, self.candidate = "idle", None
        else:
            target = self._target(inp, self._leading(inp))
            if target == self.phase:
                self.candidate = None
            elif target != self.candidate:
                self.candidate, self.candidate_since = target, t
            elif t - self.candidate_since >= hold:
                self.phase, self.candidate = target, None

        if self.phase == "lift":
            from_cfg = float(cues.get("lift", 1.0))
            hold_s = float(self.cfg.get("hold_s", 3.0))  # engine.imminent.hold_s
            cue = from_cfg * max(0.0, 1.0 - (t - self.lift_t) / hold_s)
        else:
            cue = float(cues.get(self.phase, 0.0))

        facts = [Fact(key="phase", value=self.phase, text=PHASE_TEXT[self.phase])]
        eta = inp.adapter_facts.get("attempt_eta_game_s")
        prior = self.cfg.get("eta_prior", {})
        if (
            self.phase == "idle"
            and eta is not None
            and inp.game_time is not None
            and isinstance(eta.value, int | float)
            and inp.game_time >= float(eta.value) + float(prior.get("late_after_s", 30.0))
        ):
            cue = max(cue, float(prior.get("cue", 0.2)))
            facts.append(eta)
        if self.phase in ("setup_hush",) and inp.crowd is not None and "hush" in inp.crowd.facts:
            facts.append(inp.crowd.facts["hush"])
        for key in ("weight_kg", "p_success"):
            if key in inp.adapter_facts and self.phase != "idle":
                facts.append(inp.adapter_facts[key])
        return CueResult(min(1.0, cue), facts, self.phase)
