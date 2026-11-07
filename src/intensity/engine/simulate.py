"""Offline replay on a simulated clock: tapes, mock output and the CPU pipeline test.

An envelope arrives at its `t_emit_ns` relative to the first envelope (tapes record the virtual
wall time at which the live ingest would have sent each message). The engine then runs exactly as
live: a fast tick for imminent rules and a score tick per `score_interval_s`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from intensity.engine.core import Engine
from intensity.schema import Envelope, ScoreUpdate


@dataclass
class Emitted:
    now: float  # simulated engine seconds
    update: ScoreUpdate
    out_of_cycle: bool


def run(engine: Engine, envelopes: Iterable[Envelope], until_s: float | None = None,
        base_ns: int | None = None) -> list[Emitted]:
    """Replay `envelopes`. Engine time is seconds since `base_ns` (default: the first envelope), so
    consecutive calls with the same base continue one clock."""
    envs = sorted(envelopes, key=lambda e: e.data.t_emit_ns)
    if not envs:
        return []
    base = envs[0].data.t_emit_ns if base_ns is None else base_ns
    fast = engine.cfg.service.fast_tick_s
    interval = engine.cfg.service.score_interval_s
    last_arrival = (envs[-1].data.t_emit_ns - base) / 1e9
    end = until_s if until_s is not None else last_arrival + fast
    steps_per_score = max(1, round(interval / fast))

    out: list[Emitted] = []
    i = 0
    step = max(0, int((envs[0].data.t_emit_ns - base) / 1e9 / fast + 1e-9))
    while True:
        now = step * fast
        if now > end + 1e-9:
            break
        while i < len(envs) and (envs[i].data.t_emit_ns - base) / 1e9 <= now + 1e-9:
            engine.handle(envs[i], now)
            i += 1
        for upd in engine.tick_imminent(now):
            out.append(Emitted(now, upd, True))
        if step % steps_per_score == 0:
            for upd in engine.score_all(now):
                out.append(Emitted(now, upd, False))
        step += 1
    return out
