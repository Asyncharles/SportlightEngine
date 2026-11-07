from __future__ import annotations

import asyncio
from types import SimpleNamespace

from intensity.config import PolishConfig
from intensity.engine.explain.polish import Polisher
from intensity.schema import Fact, Reason

REASON = Reason(text="Late drama: minute 88, France trail by one",
                facts=[Fact(key="minute", value=88, text="minute 88"), Fact(key="score_diff", value=-1, text="France trail by one")],
                factors=["stakes"], polished=False)


class FakeMessages:
    def __init__(self, text: str):
        self.text = text
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=self.text)])


def _polisher(text: str) -> tuple[Polisher, FakeMessages]:
    p = Polisher(PolishConfig(enabled=True, min_interval_s=0))
    msgs = FakeMessages(text)
    p._client = SimpleNamespace(beta=SimpleNamespace(messages=msgs))
    return p, msgs


def _run(p: Polisher) -> Reason:
    async def go():
        p.submit("fb", REASON)
        for _ in range(50):
            await asyncio.sleep(0.001)
        return p.polished("fb", REASON)

    return asyncio.run(go())


def test_passing_polish_is_used():
    p, msgs = _polisher("France trail by one in minute 88")
    out = _run(p)
    assert out.polished and out.text == "France trail by one in minute 88"
    assert msgs.calls[0]["model"] == "claude-opus-5-5"


def test_invented_number_is_rejected():
    p, _ = _polisher("France trail by two in minute 89")
    out = _run(p)
    assert out is REASON and p.stats["rejected"] == 1


def test_disabled_or_no_loop_never_calls():
    p = Polisher(PolishConfig(enabled=False))
    p.submit("fb", REASON)
    assert p.polished("fb", REASON) is REASON
    p2, msgs = _polisher("x")
    p2.submit("fb", REASON)  # no running loop: offline replay keeps templates
    assert msgs.calls == []
