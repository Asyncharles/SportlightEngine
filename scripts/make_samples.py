"""Write sample ScoreUpdates for the front end to contracts/samples/ (`make samples`).

Runs the deterministic mock through the engine on a simulated clock and keeps, per sport, the
last update plus the first update where each sport's badge is lit by imminent. Wall-clock
timestamps are pinned so the samples only change when the code does.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from intensity.config import load_config
from intensity.engine import simulate
from intensity.engine.core import Engine
from intensity.engine.mock import MockGenerator
from intensity.schema import ScoreUpdate

FEEDS = [("fb-arg-fra", "football"), ("f1-demo-race", "f1"), ("nba-demo-game", "nba"), ("wl-demo-session", "weightlifting")]
PINNED_NS = 1_794_000_000_000_000_000


def dump(update: ScoreUpdate, path: Path) -> None:
    data = update.model_copy(update={"t_emit_ns": PINNED_NS}).model_dump(mode="json")
    path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("contracts/samples")
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(overrides=["engine.forecast.source=heuristic", "engine.polish.enabled=false"])
    engine = Engine(config, [])
    emitted = simulate.run(engine, MockGenerator(FEEDS, seed=7, t0_ns=0).envelopes(0.0, 240.0))
    last: dict[str, ScoreUpdate] = {}
    heating: dict[str, ScoreUpdate] = {}
    for e in emitted:
        u = e.update
        last[u.feed_id] = u
        if "imminent" in u.heating_up_by and u.feed_id not in heating and u.reasons:
            heating[u.feed_id] = u
    written = []
    for feed_id, sport in FEEDS:
        if feed_id in last:
            dump(last[feed_id], out / f"score_update_{sport}.json")
            written.append(f"score_update_{sport}.json")
        if feed_id in heating:
            dump(heating[feed_id], out / f"score_update_{sport}_imminent.json")
            written.append(f"score_update_{sport}_imminent.json")
    print("\n".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
