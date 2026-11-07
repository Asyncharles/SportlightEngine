from __future__ import annotations

from intensity.engine.explain.guard import check
from intensity.engine.explain.templates import build_reasons
from intensity.engine.forecast import ForecastResult
from intensity.schema import Fact, Forecast, Imminent, PartValue

F_SCORE = Fact(key="score_diff", value=-1, text="France trail by one")
F_MIN = Fact(key="minute", value=88, unit="min", text="minute 88")
F_LOUD = Fact(key="loudness_rel_db", value=8.4, unit="dB", text="crowd 8 dB above normal")
F_GAP = Fact(key="min_gap_s", value=0.43, unit="s", text="gap 0.4 s at the front")


def test_guard_accepts_numbers_from_facts():
    assert check("Late drama: France trail by one, minute 88", [F_SCORE, F_MIN]).ok


def test_guard_rejects_invented_number():
    res = check("Late drama: France trail by two goals in minute 89", [F_SCORE, F_MIN])
    assert not res.ok and any("89" in p for p in res.problems)


def test_guard_rounding_to_text_precision():
    assert check("Loud: crowd up 8 dB, gap 0.4 s", [F_LOUD, F_GAP]).ok
    assert not check("Loud: crowd up 9 dB, gap 0.4 s", [F_LOUD, F_GAP]).ok


def test_guard_rejects_invented_name():
    res = check("Late drama: Messi pounces in minute 88", [F_SCORE, F_MIN])
    assert not res.ok and any("Messi" in p for p in res.problems)
    # the first word of a sentence is never treated as a name
    assert check("Huge: minute 88, France trail by one", [F_SCORE, F_MIN]).ok


def test_guard_needs_two_facts():
    assert not check("Loud: crowd 8 dB above normal", [F_LOUD]).ok


def _pv(value, weight, phrase, facts):
    return PartValue(value=value, weight=weight, contribution=value * weight, phrase=phrase, facts=facts)


def _forecast(p=0.3) -> ForecastResult:
    return ForecastResult(Forecast(p=p, label="big_chance", horizon_s=30, source="heuristic", heating_up=False, model_version=None), 0.7,
                          contribs={"minute": 0.4, "score_diff": 0.2, "shots": -0.1})


def _imminent(on: bool) -> Imminent:
    return Imminent(value=0.8 if on else 0.1, cue=0.8, context=1.0, heating_up=on, phase=None, source="rule",
                    facts=[Fact(key="shot_type", value="wide", text="wide shot"), F_LOUD] if on else [])


def _parts():
    return {
        "closeness": _pv(1.0, 0.2, "a tight scoreline", [F_SCORE]),
        "momentum": None,
        "stakes": _pv(0.9, 0.15, "late drama", [F_MIN, F_SCORE]),
        "crowd": _pv(0.8, 0.15, "a roaring crowd", [F_LOUD]),
        "commentary": _pv(0.0, 0.1, "quiet commentary", []),
        "video": _pv(0.2, 0.15, "calm pictures", [Fact(key="shot_type", value="wide", text="wide shot")]),
        "forecast": _pv(0.7, 0.2, "a big chance brewing", []),
        "imminent": _pv(0.1, 0.15, "nothing building yet", []),
    }


def test_reasons_ranked_guarded_and_capped():
    facts = {"minute": F_MIN, "score_diff": F_SCORE, "game_clock": Fact(key="game_clock", value="88:12", text="88:12")}
    reasons = build_reasons("football", _parts(), _forecast(), _imminent(False), facts, 3)
    assert 1 <= len(reasons) <= 3
    for r in reasons:
        assert len(r.facts) >= 2 and check(r.text, r.facts).ok and not r.polished
    assert reasons[0].factors[0] == "closeness"  # highest contribution
    forecast = next(r for r in reasons if r.factors[0] == "forecast")
    assert "30% chance of a big chance within 30 s" in forecast.text
    assert forecast.factors[1:] == ["minute", "score_diff"]  # positive SHAP, in order


def test_single_fact_part_borrows_the_game_clock():
    facts = {"game_clock": Fact(key="game_clock", value="88:12", text="88:12")}
    reasons = build_reasons("football", _parts(), _forecast(), _imminent(False), facts, 3)
    closeness = next(r for r in reasons if r.factors == ["closeness"])
    assert closeness.text == "A tight scoreline: France trail by one, 88:12"


def test_imminent_reason_goes_first_when_heating_up():
    reasons = build_reasons("football", _parts(), _forecast(), _imminent(True), {}, 3)
    assert reasons[0].factors == ["imminent"]
    assert reasons[0].text.startswith("Something is building: wide shot")
