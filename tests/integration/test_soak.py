"""30 minutes of four feeds at 1 Hz on a simulated clock: bounded state, no stale feeds."""

from __future__ import annotations

import tracemalloc

import pytest

from intensity.engine import simulate
from intensity.engine.core import Engine
from intensity.engine.mock import MockGenerator

FEEDS = [("a", "football"), ("b", "f1"), ("c", "nba"), ("d", "weightlifting")]


@pytest.mark.slow
def test_thirty_minutes_flat_memory(config):
    eng = Engine(config, [])
    gen = MockGenerator(FEEDS, t0_ns=0, duration_s=600)
    tracemalloc.start()
    sizes = []
    for chunk in range(6):  # 6 x 5 minutes, through three loops of the clip
        envs = gen.envelopes(chunk * 300.0, (chunk + 1) * 300.0)
        out = simulate.run(eng, envs, until_s=(chunk + 1) * 300.0 - 0.05, base_ns=0)
        assert out and not any(e.update.stale for e in out)
        sizes.append(tracemalloc.get_traced_memory()[0])
    tracemalloc.stop()
    assert all(fs.loops >= 2 for fs in eng.feeds.values())  # each feed looped its 10-minute clip
    # after warm-up, memory does not keep growing chunk over chunk
    assert sizes[-1] < sizes[2] * 1.25, sizes
