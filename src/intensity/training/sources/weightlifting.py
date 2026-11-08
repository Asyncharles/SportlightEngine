"""Weightlifting session CSV to match records (a minimal T4 converter, for the training agent to review).

`data/weightlifting/<session>.csv` is entered by hand and already in video time (`t_video_walk_on`,
`t_video_decision`), so the records' game time is the clip's `t_video` and the engine replays them
with an identity sync (`Engine.add_feed`, `match_ref.source: session_csv`).

Per attempt, in walk-on order:
- `attempt_called` at the walk-on: player, team (nation), attrs `lift`, `weight_kg`, `attempt`, and
  `best_kg`, the athlete's best good lift so far in the same lift (the bar weight when none yet);
- `attempt_result` at the referees' decision with attrs `good`, when the decision time is known.

`eta_game_s` is left out on purpose: the only source for it is the decision time, which is the future
at walk-on, so using it would leak into the imminent rule.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def to_records(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if _float(r.get("t_video_walk_on")) is not None]
    rows.sort(key=lambda r: float(r["t_video_walk_on"]))
    best: dict[tuple[str, str], float] = {}
    out: list[dict[str, Any]] = []
    for r in rows:
        athlete, lift = r["athlete"].strip(), r["lift"].strip()
        weight = _float(r.get("weight_kg")) or 0.0
        out.append(
            {
                "game_time_s": float(r["t_video_walk_on"]),
                "period": 1,
                "kind": "attempt_called",
                "team": r.get("nation") or None,
                "player": athlete,
                "attrs": {
                    "lift": lift,
                    "weight_kg": weight,
                    "attempt": int(r.get("attempt") or 1),
                    "best_kg": best.get((athlete, lift), weight),
                },
            }
        )
        decision = _float(r.get("t_video_decision"))
        result = (r.get("result") or "").strip()
        if decision is not None and result:
            good = result == "good"
            out.append(
                {
                    "game_time_s": decision,
                    "period": 1,
                    "kind": "attempt_result",
                    "team": r.get("nation") or None,
                    "player": athlete,
                    "attrs": {"lift": lift, "good": good},
                }
            )
            if good:
                best[(athlete, lift)] = max(best.get((athlete, lift), 0.0), weight)
    out.sort(key=lambda rec: rec["game_time_s"])
    return out
