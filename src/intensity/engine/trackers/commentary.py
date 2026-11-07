"""Commentary tracker: lexicon hits, player names and speech rate, decaying over 10 s.

ASR lags by up to 3 s, so this feeds only the `commentary` part and reasons, never imminent rules.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass

from intensity.config import CommentaryTrackerConfig
from intensity.engine.adapters.base import AdapterPart, clamp01
from intensity.schema import Event, Fact


@dataclass
class _Segment:
    t_end: float
    score: float
    facts: list[Fact]


class CommentaryTracker:
    def __init__(self, cfg: CommentaryTrackerConfig, keywords: dict[str, float]):
        self.cfg = cfg
        self.keywords = {k.lower(): float(v) for k, v in keywords.items()}
        self._patterns = [
            (k, w, re.compile(r"\b" + re.escape(k) + r"\b", re.IGNORECASE)) for k, w in self.keywords.items()
        ]
        self.names: set[str] = set()
        self.reset()

    def reset(self) -> None:
        self.segments: deque[_Segment] = deque(maxlen=64)
        self.last_t = -1e9

    def add_names(self, names: set[str]) -> None:
        for name in names:
            self.names.add(name)
            parts = name.split()
            if len(parts) > 1 and len(parts[-1]) >= 3:
                self.names.add(parts[-1])

    def _quote(self, text: str, around: int | None) -> str:
        words = text.split()
        n = self.cfg.max_quote_words
        if len(words) <= n:
            return " ".join(words)
        start = 0
        if around is not None:
            start = max(0, min(len(words) - n, around - n // 2))
        return " ".join(words[start : start + n])

    def on_event(self, event: Event) -> None:
        p = event.payload
        if float(p.get("no_speech_prob", 0.0)) > self.cfg.min_no_speech_prob:
            return
        text: str = p["text"].strip()
        if not text:
            return
        hits = [(k, w, m.start()) for k, w, pat in self._patterns if (m := pat.search(text))]
        lexical = 0.0
        facts: list[Fact] = []
        first_word_idx = None
        if hits:
            hits.sort(key=lambda h: -h[1])
            lexical = min(1.0, hits[0][1] + 0.1 * (len(hits) - 1))
            first_word_idx = len(text[: hits[0][2]].split())
            facts.append(Fact(key="commentary_keyword", value=hits[0][0], text=f'commentator says "{hits[0][0]}"'))
        names = sorted(n for n in self.names if re.search(r"\b" + re.escape(n) + r"\b", text))
        name_score = self.cfg.name_weight if names else 0.0
        if names:
            facts.append(Fact(key="commentary_player", value=names[0], text=f"{names[0]} mentioned"))
        duration = max(0.5, float(p["t_end"]) - float(p["t_start"]))
        rate = len(text.split()) / duration
        rate_score = self.cfg.rate_weight * clamp01(rate / self.cfg.rate_ref_wps)
        score = clamp01(lexical + name_score + rate_score)
        quote = self._quote(text, first_word_idx)
        facts.insert(0, Fact(key="commentary_quote", value=quote, text=f'"{quote}"'))
        self.segments.append(_Segment(event.t_video, score, facts))
        self.last_t = event.t_video

    def part(self, now: float) -> AdapterPart | None:
        best, best_facts = 0.0, []
        for seg in self.segments:
            age = now - seg.t_end
            if age < 0:
                continue
            decayed = seg.score * max(0.0, 1.0 - age / self.cfg.decay_s)
            if decayed > best:
                best, best_facts = decayed, seg.facts
        if now - self.last_t > 60.0:
            return None  # no ASR for a minute (asr: off, or the link is down): drop out
        phrase = "excited commentary" if best >= 0.6 else "steady commentary" if best >= 0.25 else "quiet commentary"
        return AdapterPart(best, phrase, list(best_facts))
