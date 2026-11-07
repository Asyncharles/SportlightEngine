"""Reasons: rank factors by part contribution and the forecast's SHAP, then fill templates from facts."""

from __future__ import annotations

from dataclasses import dataclass

from intensity.engine.explain.guard import check
from intensity.engine.explain.phrases import FORECAST_LEAD, IMMINENT_LEADS, capitalise, label_phrase
from intensity.engine.forecast import ForecastResult
from intensity.schema import Fact, Imminent, PartValue, Reason

MAX_FACTS_PER_REASON = 3


@dataclass
class Candidate:
    factor: str  # part name, "imminent", or "forecast"
    score: float  # ranking weight
    lead: str
    facts: list[Fact]
    factors: list[str]


def _dedupe(facts: list[Fact]) -> list[Fact]:
    seen, out = set(), []
    for f in facts:
        if (f.key, f.text) not in seen:
            seen.add((f.key, f.text))
            out.append(f)
    return out


def render(lead: str, facts: list[Fact]) -> str:
    return f"{capitalise(lead)}: {', '.join(f.text for f in facts)}"


def forecast_facts(fr: ForecastResult, adapter_facts: dict[str, Fact]) -> tuple[list[Fact], list[str]]:
    """The p fact plus facts for the features pushing p up the most."""
    fc = fr.forecast
    pct = round(100 * fc.p)
    p_fact = Fact(
        key="forecast_p",
        value=pct,
        unit="%",
        text=f"{pct}% chance of {label_phrase(fc.label)} within {int(fc.horizon_s)} s",
    )
    facts, features = [p_fact], []
    for name, contrib in sorted(fr.contribs.items(), key=lambda kv: -kv[1]):
        if contrib <= 0:
            break
        if name in adapter_facts:
            facts.append(adapter_facts[name])
            features.append(name)
        if len(facts) >= MAX_FACTS_PER_REASON:
            break
    return facts, features


def build_reasons(
    sport: str,
    parts: dict[str, PartValue | None],
    forecast: ForecastResult,
    imminent: Imminent,
    adapter_facts: dict[str, Fact],
    max_reasons: int = 3,
) -> list[Reason]:
    clock = adapter_facts.get("game_clock")
    cands: list[Candidate] = []

    if imminent.heating_up and imminent.facts:
        lead = IMMINENT_LEADS.get(sport, "Something is building")
        # Imminent goes first while it is on: it is the most time-critical thing to say.
        cands.append(Candidate("imminent", 10.0, lead, list(imminent.facts), ["imminent"]))

    for name, pv in parts.items():
        if pv is None or name == "imminent" or pv.contribution <= 0:
            continue
        if name == "forecast":
            facts, features = forecast_facts(forecast, adapter_facts)
            lead = FORECAST_LEAD.format(label=label_phrase(forecast.forecast.label))
            cands.append(Candidate("forecast", pv.contribution, lead, facts, ["forecast", *features]))
        elif pv.facts:
            cands.append(Candidate(name, pv.contribution, pv.phrase, list(pv.facts), [name]))

    cands.sort(key=lambda c: -c.score)
    reasons: list[Reason] = []
    for c in cands:
        facts = _dedupe(c.facts)[:MAX_FACTS_PER_REASON]
        if len(facts) < 2 and clock is not None and clock not in facts:
            facts.append(clock)
        text = render(c.lead, facts)
        if not check(text, facts).ok:
            continue
        reasons.append(Reason(text=text, facts=facts, factors=c.factors, polished=False))
        if len(reasons) >= max_reasons:
            break
    return reasons
