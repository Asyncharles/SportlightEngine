"""Words for reasons. Phrases never contain numbers or names: those come only from facts."""

from __future__ import annotations

LABEL_PHRASES = {
    "big_chance": "a big chance",
    "on_track_pass": "an overtake",
    "lead_change": "a lead change",
    "lift_success": "a good lift",
    "swing": "a swing",
}

IMMINENT_LEADS = {
    "football": "Something is building",
    "nba": "A fast break is on",
    "f1": "A move is coming",
    "weightlifting": "A lift is about to start",
    "chess": "Something is building",
}

FORECAST_LEAD = "{label} looks likely"


def label_phrase(label: str) -> str:
    return LABEL_PHRASES.get(label, label.replace("_", " "))


def capitalise(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text
