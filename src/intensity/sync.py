"""Sync tables map `t_video` to game time and back (hard rule 3).

`data/sync/<clip_id>.csv`: `clip_id, video_start, video_end, game_start, period`. Each row is a
segment of continuous play; inside it game time is `game_start + (t_video - video_start)`. Gaps
between rows (replays, breaks, cuts) have no game time.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SyncRow:
    clip_id: str
    video_start: float
    video_end: float
    game_start: float
    period: int

    @property
    def game_end(self) -> float:
        return self.game_start + (self.video_end - self.video_start)


class SyncTable:
    def __init__(self, rows: list[SyncRow]):
        self.rows = sorted(rows, key=lambda r: r.video_start)
        for a, b in zip(self.rows, self.rows[1:], strict=False):
            if b.video_start < a.video_end:
                raise ValueError(f"sync rows overlap at t_video {b.video_start}")
        for row in self.rows:
            if row.video_end <= row.video_start:
                raise ValueError(f"sync row ends before it starts: {row}")

    @classmethod
    def load(cls, path: str | Path) -> SyncTable:
        with open(path, newline="") as fh:
            rows = [
                SyncRow(
                    clip_id=r["clip_id"].strip(),
                    video_start=float(r["video_start"]),
                    video_end=float(r["video_end"]),
                    game_start=float(r["game_start"]),
                    period=int(r["period"]),
                )
                for r in csv.DictReader(fh)
            ]
        return cls(rows)

    @classmethod
    def identity(cls, duration_s: float, clip_id: str = "identity", period: int = 1) -> SyncTable:
        """One segment with game time equal to t_video; used by the mock and tests."""
        return cls([SyncRow(clip_id, 0.0, duration_s, 0.0, period)])

    def game_time(self, t_video: float) -> tuple[float, int] | None:
        """Game time and period at `t_video`, or None in a gap."""
        for row in self.rows:
            if row.video_start <= t_video < row.video_end:
                return row.game_start + (t_video - row.video_start), row.period
        return None

    def last_game_time(self, t_video: float) -> tuple[float, int] | None:
        """Game time at `t_video`, holding the last segment's end through gaps."""
        last = None
        for row in self.rows:
            if t_video < row.video_start:
                break
            if t_video < row.video_end:
                return row.game_start + (t_video - row.video_start), row.period
            last = (row.game_end, row.period)
        return last

    def t_video(self, game_time: float, period: int | None = None) -> float | None:
        """The `t_video` at which `game_time` is shown, or None if the clip never shows it."""
        for row in self.rows:
            if period is not None and row.period != period:
                continue
            if row.game_start <= game_time < row.game_end:
                return row.video_start + (game_time - row.game_start)
        return None
