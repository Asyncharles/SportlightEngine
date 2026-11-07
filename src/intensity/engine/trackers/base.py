"""Helpers shared by the trackers."""

from __future__ import annotations

from intensity.schema import Event, Fact


def event_fact(event: Event, key: str, fallback_text: str, unit: str | None = None) -> Fact | None:
    """The fact the event carried under `key`, or one built from the payload field of that name.

    Either way every number comes from the event itself (hard rule 7).
    """
    for fact in event.facts:
        if fact.key == key:
            return fact
    value = event.payload.get(key)
    if value is None:
        return None
    if isinstance(value, float):
        value = round(value, 2)
    return Fact(key=key, value=value, unit=unit, text=fallback_text.format(value=value, abs=abs(value) if isinstance(value, int | float) else value))
