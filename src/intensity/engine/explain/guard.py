"""The fact guard (hard rule 7): every number and proper name in a reason must come from its facts.

A number in the text matches a fact if it appears in any fact's text, or equals a numeric fact value
rounded to the text's precision ("8 dB" matches 8.4, "0.4 s" matches 0.43). A proper name is a
capitalised word anywhere but the start of the sentence; it must appear in a fact's text or value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from intensity.schema import Fact

NUMBER = re.compile(r"\d+(?:\.\d+)?")
WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ][\wÀ-ÖØ-öø-ÿ'’-]*")
MIN_FACTS = 2


@dataclass
class GuardResult:
    ok: bool
    problems: list[str] = field(default_factory=list)


def _fact_numbers(facts: list[Fact]) -> tuple[set[str], list[float]]:
    strings: set[str] = set()
    values: list[float] = []
    for f in facts:
        for s in (f.text, str(f.value)):
            for m in NUMBER.findall(s):
                strings.add(m)
                values.append(float(m))
        if isinstance(f.value, int | float) and not isinstance(f.value, bool):
            values.append(abs(float(f.value)))
    return strings, values


def _number_ok(token: str, strings: set[str], values: list[float]) -> bool:
    if token in strings:
        return True
    decimals = len(token.split(".")[1]) if "." in token else 0
    n = float(token)
    return any(abs(round(v, decimals) - n) < 1e-9 for v in values)


def _fact_words(facts: list[Fact]) -> set[str]:
    words: set[str] = set()
    for f in facts:
        for s in (f.text, str(f.value)):
            words.update(WORD.findall(s))
    return words


def check(text: str, facts: list[Fact], min_facts: int = MIN_FACTS) -> GuardResult:
    problems = []
    if len(facts) < min_facts:
        problems.append(f"needs at least {min_facts} facts, has {len(facts)}")
    strings, values = _fact_numbers(facts)
    for token in NUMBER.findall(text):
        if not _number_ok(token, strings, values):
            problems.append(f"number {token} not in facts")
    known = _fact_words(facts)
    words = WORD.findall(text)
    for i, word in enumerate(words):
        if i == 0 or not word[0].isupper():
            continue
        if word not in known:
            problems.append(f"name {word} not in facts")
    return GuardResult(not problems, problems)
