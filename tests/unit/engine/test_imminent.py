from __future__ import annotations

import pytest

from intensity.config import ImminentConfig, load_sport_yaml
from intensity.engine.imminent import CueInputs, CueResult, ImminentRule, ImminentState, make_rule
from intensity.engine.trackers.crowd import CrowdCues
from intensity.engine.trackers.video import VideoCues
from intensity.engine.trackers.voice import VoiceCues
from intensity.schema import Fact


def vcues(shot="wide", pan_pct=0.0, sustained=0.0, replay=False, prompts=None, motion_pct=0.3, motion=0.03,
          motion_median=0.03, pan_speed=0.05, pan_median=0.05) -> VideoCues:
    prompts = prompts or {}
    facts = {"shot_type": Fact(key="shot_type", value=shot, text=f"{shot} shot")}
    if shot == "wide":
        facts["pan_speed"] = Fact(key="pan_speed", value=0.6, text="camera panning 0.6 frame widths a second")
    if replay:
        facts["replay"] = Fact(key="replay", value="showing", text="replay showing")
    return VideoCues(shot, replay, pan_pct, pan_speed, sustained, motion_pct, motion, motion_median, pan_median,
                     dict(prompts), dict(prompts), facts)


def ccues(rise_pct=0.5, rise=0.0, hush=False) -> CrowdCues:
    facts = {"rise_db_s": Fact(key="rise_db_s", value=rise, text=f"crowd rising {rise:.0f} dB a second")}
    if hush:
        facts["hush"] = Fact(key="hush", value="true", text="the crowd goes quiet")
    return CrowdCues(rise_pct, rise, hush, True, facts)


def inputs(t=0.0, video=None, crowd=None, voice=None, facts=None, fpart=0.0, stakes=0.0, p=0.5, gt=None) -> CueInputs:
    return CueInputs(t, gt if gt is not None else t, video, crowd, voice, facts or {}, fpart, stakes, p)


def rule(sport: str) -> ImminentRule:
    return make_rule(sport, {"hold_s": 3.0, **load_sport_yaml("config/imminent", sport)})


# --------------------------------------------------------------- football / NBA


def test_football_weights_pan_crowd_voice():
    r = rule("football")
    res = r.cue(inputs(video=vcues(pan_pct=1.0, sustained=2.0), crowd=ccues(1.0, 3.0), voice=VoiceCues(True, 1.0, 1.0, 1.0)))
    assert res.cue == pytest.approx(1.0)
    assert {f.key for f in res.facts} >= {"shot_type", "pan_speed", "rise_db_s"}


def test_football_pan_needs_wide_and_sustained():
    r = rule("football")
    short = r.cue(inputs(video=vcues(pan_pct=1.0, sustained=1.0), crowd=ccues(0.0), voice=VoiceCues(True, 0.0, 0.0, 0.0)))
    assert short.cue == 0.0
    close = r.cue(inputs(video=vcues(shot="close_player", pan_pct=1.0, sustained=5.0), crowd=ccues(0.0), voice=VoiceCues(False, None, None, None)))
    assert close.cue == 0.0


def test_nba_accepts_shorter_pan():
    res = rule("nba").cue(inputs(video=vcues(pan_pct=1.0, sustained=1.0), crowd=ccues(0.0), voice=VoiceCues(False, None, None, None)))
    assert res.cue == pytest.approx(0.5)


def test_cue_is_zero_during_replays():
    for sport in ("football", "nba", "f1"):
        res = rule(sport).cue(inputs(video=vcues(shot="replay", replay=True, pan_pct=1.0, sustained=5),
                                     crowd=ccues(1.0, 3.0), voice=VoiceCues(True, 1, 1, 1), facts={"min_gap_s": Fact(key="min_gap_s", value=0.3, text="gap")}))
        assert res.cue == 0.0, sport


def test_audio_only_when_video_is_down():
    res = rule("football").cue(inputs(video=None, crowd=ccues(1.0, 3.0), voice=VoiceCues(True, 1.0, 1.0, 1.0)))
    assert res.cue == pytest.approx(1.0)  # renormalised over the audio cues
    assert any(f.key == "imminent_inputs" for f in res.facts)


# --------------------------------------------------------------- F1


@pytest.mark.parametrize("gap,expected", [(0.5, 1.0), (1.0, 1.0), (1.5, 0.5), (2.0, 0.0), (3.0, 0.0)])
def test_f1_gap_close(gap, expected):
    r = rule("f1")
    res = r.cue(inputs(video=vcues(prompts={}), voice=VoiceCues(False, None, None, None),
                       facts={"min_gap_s": Fact(key="min_gap_s", value=gap, unit="s", text=f"gap {gap} s")}))
    # battle 0, voice 0: only gap_close counts, weight 0.4
    assert res.cue == pytest.approx(0.4 * expected)


def test_f1_battle_shot_is_best_of_prompts():
    r = rule("f1")
    res = r.cue(inputs(video=vcues(prompts={"two racing cars close together": 0.2, "onboard camera": 0.95}),
                       voice=VoiceCues(False, None, None, None)))
    assert res.cue == pytest.approx(0.4 * 0.95)
    assert any(f.key == "prompt:onboard camera" for f in res.facts)


# --------------------------------------------------------------- weightlifting


EMPTY = {"empty platform": 0.4, "athlete walking onto the platform": 0.1, "athlete chalking hands": 0.1, "athlete standing over the barbell": 0.1}
WALK = {**EMPTY, "empty platform": 0.1, "athlete walking onto the platform": 0.45}
CHALK = {**EMPTY, "empty platform": 0.1, "athlete chalking hands": 0.45}


def _wl_step(r, t, prompts, hush=False, motion_pct=0.3, motion=0.02, facts=None, gt=None):
    return r.cue(inputs(t=t, video=vcues(prompts=prompts, motion_pct=motion_pct, motion=motion, motion_median=0.02),
                        crowd=ccues(hush=hush), facts=facts, gt=gt))


def test_weightlifting_phase_machine():
    r = rule("weightlifting")
    assert _wl_step(r, 0.0, EMPTY).phase == "idle"
    assert _wl_step(r, 1.0, WALK).phase == "idle"  # held 1 s before moving
    res = _wl_step(r, 2.0, WALK)
    assert res.phase == "walk_on" and res.cue == pytest.approx(0.4)
    _wl_step(r, 3.0, CHALK)
    assert _wl_step(r, 4.0, CHALK).phase == "setup"
    _wl_step(r, 5.0, CHALK, hush=True)
    res = _wl_step(r, 6.0, CHALK, hush=True)
    assert res.phase == "setup_hush" and res.cue == 1.0
    assert any(f.text == "the crowd goes quiet" for f in res.facts)
    lift = _wl_step(r, 7.0, CHALK, motion_pct=0.97, motion=0.2)
    assert lift.phase == "lift" and lift.cue == 1.0
    assert _wl_step(r, 8.5, CHALK).cue == pytest.approx(0.5)  # decays over hold_s (3 s)
    assert _wl_step(r, 10.0, CHALK).cue == 0.0
    assert _wl_step(r, 22.1, EMPTY).phase == "idle"  # lift_reset_s


def test_weightlifting_spike_needs_a_real_jump():
    r = rule("weightlifting")
    for t in range(3):
        _wl_step(r, float(t), CHALK)
    assert r.phase == "setup"
    # high percentile of a flat signal is noise, not a lift
    assert _wl_step(r, 3.0, CHALK, motion_pct=0.99, motion=0.025).phase == "setup"


def test_weightlifting_eta_prior():
    r = rule("weightlifting")
    eta = {"attempt_eta_game_s": Fact(key="attempt_eta_game_s", value=100.0, unit="s", text="next attempt due")}
    assert _wl_step(r, 0.0, EMPTY, facts=eta, gt=120.0).cue == 0.0
    late = _wl_step(r, 1.0, EMPTY, facts=eta, gt=131.0)
    assert late.phase == "idle" and late.cue == pytest.approx(0.2)


def test_weightlifting_context_is_coin_flip():
    r = rule("weightlifting")
    assert r.context_value(inputs(p=0.5)) == 1.0
    assert r.context_value(inputs(p=0.95)) == pytest.approx(0.1)


# --------------------------------------------------------------- the shared combiner


class FixedRule(ImminentRule):
    def __init__(self, cues: dict[float, float], threshold: float = 0.5):
        super().__init__({"threshold": threshold})
        self.cues = cues

    def cue(self, inp: CueInputs) -> CueResult:
        return CueResult(self.cues.get(inp.t_video, 0.0))


def test_context_floor_multiplier():
    st = ImminentState(FixedRule({0.0: 1.0}), ImminentConfig())
    dead, _ = st.update(inputs(t=0.0, fpart=0.0, stakes=0.0))
    assert dead.value == pytest.approx(0.3) and dead.context == pytest.approx(0.3)
    st = ImminentState(FixedRule({0.0: 1.0}), ImminentConfig())
    late, _ = st.update(inputs(t=0.0, fpart=0.2, stakes=0.9))
    assert late.value == pytest.approx(0.3 + 0.7 * 0.9)


def test_peak_hold_decays_over_hold_s():
    st = ImminentState(FixedRule({0.0: 1.0}, threshold=2), ImminentConfig(context_floor=1.0, hold_s=3.0))
    values = [st.update(inputs(t=t))[0].value for t in (0.0, 1.5, 3.0, 4.0)]
    assert values == pytest.approx([1.0, 0.5, 0.0, 0.0])


def test_min_on_s_and_single_trigger():
    cues = {t / 10: 1.0 for t in range(0, 40)}
    st = ImminentState(FixedRule(cues), ImminentConfig(context_floor=1.0, min_on_s=1.0))
    fired = [(t / 10, st.update(inputs(t=t / 10))) for t in range(0, 40)]
    triggers = [t for t, (_, trig) in fired if trig]
    assert triggers == [1.0]  # on after 1 s above threshold, sent once
    assert fired[5][1][0].heating_up is False and fired[10][1][0].heating_up is True


def test_trigger_rate_limited_to_min_on_s():
    # value flickers: above 1.0 s, below briefly, above again
    cues = {**{t / 10: 1.0 for t in range(0, 11)}, **{t / 10: 1.0 for t in range(12, 40)}}
    st = ImminentState(FixedRule(cues), ImminentConfig(context_floor=1.0, min_on_s=1.0, hold_s=0.0))
    triggers = [t / 10 for t in range(0, 40) if st.update(inputs(t=t / 10))[1]]
    assert triggers == [1.0, 2.2]
    assert triggers[1] - triggers[0] >= 1.0


def test_never_steps_backwards():
    st = ImminentState(FixedRule({5.0: 1.0}), ImminentConfig(context_floor=1.0))
    st.update(inputs(t=5.0))
    same, trig = st.update(inputs(t=1.0))
    assert same.value == 1.0 and not trig
