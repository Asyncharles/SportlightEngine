"""Optional reason polish (D10: off by default).

Sends a template sentence and its facts to Claude and keeps the result only if it passes the fact
guard. It never blocks a tick: `submit()` starts a background request, and `polished()` returns the
latest passing rewrite of the same template, if one is ready. Credentials come from the environment
(`ANTHROPIC_API_KEY` in `.env`, or an `ant auth login` profile).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from intensity.config import PolishConfig
from intensity.engine.explain.guard import check
from intensity.schema import Reason

log = logging.getLogger(__name__)

SYSTEM = (
    "You rewrite one short sports-broadcast reason so it reads naturally to a viewer. "
    "Keep it under 20 words. Use only the facts given: do not add any number, name, team or event "
    "that is not in them. Reply with the sentence only."
)


class Polisher:
    def __init__(self, cfg: PolishConfig):
        self.cfg = cfg
        self._client: Any = None
        self._disabled_reason: str | None = None
        self._ready: dict[tuple[str, str], Reason] = {}  # (feed_id, template text) -> polished
        self._inflight: set[tuple[str, str]] = set()
        self._last_submit: dict[str, float] = {}
        self.stats = {"ok": 0, "rejected": 0, "failed": 0}

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled and self._disabled_reason is None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                self._disabled_reason = "anthropic not installed (uv sync --extra polish)"
                log.warning("polish disabled: %s", self._disabled_reason)
                return None
            self._client = anthropic.AsyncAnthropic(timeout=self.cfg.timeout_s, max_retries=0)
        return self._client

    def polished(self, feed_id: str, reason: Reason) -> Reason:
        """The latest passing polish of this template for this feed, or the template itself."""
        return self._ready.get((feed_id, reason.text), reason)

    def submit(self, feed_id: str, reason: Reason) -> None:
        """Start a background polish of `reason`; at most one per feed every `min_interval_s`."""
        if not self.enabled:
            return
        key = (feed_id, reason.text)
        now = time.monotonic()
        if key in self._ready or key in self._inflight:
            return
        if now - self._last_submit.get(feed_id, -1e9) < self.cfg.min_interval_s:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # offline replay has no event loop: templates only
        self._last_submit[feed_id] = now
        self._inflight.add(key)
        loop.create_task(self._run(key, reason))

    async def _run(self, key: tuple[str, str], reason: Reason) -> None:
        try:
            text = await self._polish(reason)
            if text is None:
                return
            if check(text, reason.facts).ok:
                self._ready[key] = reason.model_copy(update={"text": text, "polished": True})
                self.stats["ok"] += 1
                if len(self._ready) > 512:
                    self._ready.pop(next(iter(self._ready)))
            else:
                self.stats["rejected"] += 1
        except Exception as exc:  # polish is best-effort; the template is always there
            self.stats["failed"] += 1
            log.warning("polish failed: %s", exc)
        finally:
            self._inflight.discard(key)

    async def _polish(self, reason: Reason) -> str | None:
        client = self._get_client()
        if client is None:
            return None
        facts = [{"text": f.text, "value": f.value, "unit": f.unit} for f in reason.facts]
        prompt = f"Reason: {reason.text}\nFacts: {json.dumps(facts)}"
        response = await client.beta.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=SYSTEM,
            output_config={"effort": "low"},  # keeps latency inside timeout_s
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
            timeout=self.cfg.timeout_s,
        )
        if response.stop_reason == "refusal":
            return None
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return text or None
