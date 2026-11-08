"""Match-data replayer: releases each match record once the feed's clock passes its `t_video`.

Records come from `data/events/<sport>/<match_id>.jsonl`, one `match` payload per line. Game time maps
to `t_video` through the feed's sync table; records the clip never shows (gaps, before the clip
starts) are released as soon as the clock passes the last shown moment before them, so the adapter's
state stays right, and the replayer holds in gaps.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from intensity.schema import Event, MatchPayload, Sport
from intensity.sync import SyncTable


def load_records(path: str | Path) -> list[MatchPayload]:
    with open(path) as fh:
        return [MatchPayload.model_validate(json.loads(line)) for line in fh if line.strip()]


class Replayer:
    def __init__(self, feed_id: str, sport: Sport, records: list[MatchPayload], sync: SyncTable):
        self.feed_id = feed_id
        self.sport = sport
        self.sync = sync
        timed = []
        for rec in records:
            t = self._t_video(rec)
            timed.append((t, rec))
        # Records released at the same t_video (everything before the clip starts, say) keep match
        # order, period first, exactly as training's featurize replays them; game time alone would
        # interleave first-half stoppage time with the second half and reset the adapter's period state.
        timed.sort(key=lambda tr: (tr[0], tr[1].period, tr[1].game_time_s))
        self.timeline: list[tuple[float, MatchPayload]] = timed
        self.cursor = 0

    def _t_video(self, rec: MatchPayload) -> float:
        t = self.sync.t_video(rec.game_time_s, rec.period)
        if t is not None:
            return t
        # Not shown: release at the end of the last segment that ends before it in game time.
        best = 0.0
        for row in self.sync.rows:
            if (row.period, row.game_end) <= (rec.period, rec.game_time_s):
                best = max(best, row.video_end)
        return best

    def reset(self) -> None:
        self.cursor = 0

    def release(self, t_video: float, loop_epoch: int) -> list[Event]:
        """Records whose t_video the clock has passed, as `match` Events."""
        out = []
        while self.cursor < len(self.timeline) and self.timeline[self.cursor][0] <= t_video:
            t, rec = self.timeline[self.cursor]
            out.append(
                Event(
                    feed_id=self.feed_id,
                    sport=self.sport,
                    source="match",
                    t_video=t,
                    loop_epoch=loop_epoch,
                    t_emit_ns=time.time_ns(),
                    payload=rec.model_dump(),
                )
            )
            self.cursor += 1
        return out
