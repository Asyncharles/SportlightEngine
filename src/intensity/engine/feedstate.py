"""Per-feed engine state: clock, loop epoch, adapter, trackers, imminent state, EMA and trend."""

from __future__ import annotations

from dataclasses import dataclass, field

from intensity.config import EngineConfig
from intensity.engine.adapters import Adapter
from intensity.engine.forecast import Forecaster, ForecastResult
from intensity.engine.imminent import ImminentState
from intensity.engine.replay import Replayer
from intensity.engine.scorer import Ema, Trend
from intensity.engine.trackers.commentary import CommentaryTracker
from intensity.engine.trackers.crowd import CrowdTracker
from intensity.engine.trackers.video import VideoTracker
from intensity.engine.trackers.voice import VoiceTracker
from intensity.schema import ScoreUpdate
from intensity.sync import SyncTable


@dataclass
class FeedClock:
    """The feed's clock is its last ClockTick plus engine time since that tick arrived."""

    t_video: float | None = None
    arrival: float = 0.0  # engine time the last tick arrived

    last: float = 0.0  # highest t_video handed out this loop, so the clock never steps back

    def now(self, engine_now: float, max_extrapolate_s: float) -> float:
        if self.t_video is None:
            return 0.0
        # Extrapolate only across the gap between ticks: past that the stream may have stalled,
        # and running ahead would release match records for video that has not played.
        t = self.t_video + min(max(0.0, engine_now - self.arrival), max_extrapolate_s)
        self.last = max(self.last, t)
        return self.last


@dataclass
class FeedState:
    feed_id: str
    sport: str
    sync: SyncTable
    adapter: Adapter
    crowd: CrowdTracker
    voice: VoiceTracker
    commentary: CommentaryTracker
    video: VideoTracker
    imminent: ImminentState
    forecaster: Forecaster
    ema: Ema
    trend: Trend
    replay: Replayer | None = None
    clock: FeedClock = field(default_factory=FeedClock)
    loop_epoch: int = 0
    down: bool = False
    seen: bool = False
    last_source_at: dict[str, float] = field(default_factory=dict)  # engine time per source
    last_forecast: ForecastResult | None = None
    last_stakes: float = 0.0
    last_update: ScoreUpdate | None = None
    last_scored_at: float | None = None  # engine time of the last score, to score once per step
    last_ocr: dict[str, str] = field(default_factory=dict)
    loops: int = 0
    prerolled: bool = False  # the sport window got this feed's earlier match history (Engine._preroll)

    def stale(self, engine_now: float, cfg: EngineConfig) -> bool:
        return self.down or self.clock.t_video is None or engine_now - self.clock.arrival > cfg.stale_after_s

    def t_video(self, engine_now: float, cfg: EngineConfig) -> float:
        return self.clock.now(engine_now, cfg.max_extrapolate_s)

    def game_time(self, t_video: float) -> float | None:
        gt = self.sync.last_game_time(t_video)
        return gt[0] if gt else None

    def reset_for_loop(self, loop_epoch: int) -> None:
        """On a loop the adapter, replay cursor, trackers, imminent state, EMA and trend reset.
        The sport's percentile history lives in the engine and is kept."""
        self.loop_epoch = loop_epoch
        self.loops += 1
        self.adapter.reset()
        if self.replay is not None:
            self.replay.reset()
        for tracker in (self.crowd, self.voice, self.commentary, self.video):
            tracker.reset()
        self.imminent.reset()
        self.ema.reset()
        self.trend.reset()
        self.clock = FeedClock()
        self.last_forecast = None
        self.last_stakes = 0.0
