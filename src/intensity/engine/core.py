"""The engine core, shared by the live service and offline replay.

Once a second per feed: read the feed's clock, release its match replay, update trackers, score,
forecast, run the imminent rule, explain. Every `fast_tick_s` the imminent rule also runs on its
own, and a crossing scores that feed at once (the out-of-cycle update). Time is passed in as `now`
(engine seconds), so the same code runs on the wall clock or a simulated one.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from intensity.config import Config, load_sport_yaml, load_weights, resolve_path
from intensity.engine.adapters import make_adapter
from intensity.engine.adapters.base import AdapterPart
from intensity.engine.explain.phrases import label_phrase
from intensity.engine.explain.polish import Polisher
from intensity.engine.explain.templates import build_reasons, forecast_facts
from intensity.engine.feedstate import FeedState
from intensity.engine.forecast import Forecaster, ForecastResult
from intensity.engine.imminent import CueInputs, ImminentState, make_rule
from intensity.engine.imminent.weightlifting import PHASE_TEXT
from intensity.engine.registry import Registry
from intensity.engine.replay import Replayer, load_records
from intensity.engine.scorer import Ema, SportPercentile, Trend, combine, load_intensity_ref
from intensity.engine.trackers.commentary import CommentaryTracker
from intensity.engine.trackers.crowd import CrowdTracker
from intensity.engine.trackers.rolling import Seeds
from intensity.engine.trackers.video import VideoTracker
from intensity.engine.trackers.voice import VoiceTracker
from intensity.feeds import Feed
from intensity.schema import ClockTick, Control, Envelope, Event, Imminent, MatchPayload, ScoreUpdate
from intensity.sync import SyncTable
from intensity.telemetry import Metrics, Timer

log = logging.getLogger(__name__)

IDENTITY_SPAN_S = 10_000_000.0  # mock / unsynced feeds: game time = t_video


class Engine:
    def __init__(
        self,
        config: Config,
        feeds: list[Feed] | None = None,
        registry: Registry | None = None,
        metrics: Metrics | None = None,
    ):
        self.config = config
        self.cfg = config.engine
        self.metrics = metrics or Metrics()
        self.registry = registry or Registry(self.cfg.forecast.models_dir, self.cfg.forecast.shadow)
        if registry is None:
            self.registry.reload()
        self.weights = load_weights(self.cfg.weights_path)
        self.intensity_ref = load_intensity_ref(self.cfg.percentile.ref_path)
        self.sport_pct: dict[str, SportPercentile] = {}
        self.polisher = Polisher(self.cfg.polish)
        self.registry_feeds: dict[str, Feed] = {f.feed_id: f for f in feeds or []}
        self.feeds: dict[str, FeedState] = {}
        self._imminent_cfg: dict[str, dict[str, Any]] = {}
        self._lexicons: dict[str, dict[str, float]] = {}
        self._pending_ticks: dict[str, tuple[ClockTick, float]] = {}
        self.allow_preroll = True  # off in the shadow engine a pre-roll runs
        for feed in feeds or []:
            self.add_feed(feed.feed_id, feed.sport, feed)

    # ------------------------------------------------------------------ feeds

    def _weights_for(self, sport: str) -> dict[str, float]:
        return self.weights.get(sport, self.weights.get("default", {}))

    def _imminent_yaml(self, sport: str) -> dict[str, Any]:
        if sport not in self._imminent_cfg:
            self._imminent_cfg[sport] = {"hold_s": self.cfg.imminent.hold_s, **load_sport_yaml(self.cfg.imminent.config_dir, sport)}
        return self._imminent_cfg[sport]

    def _lexicon(self, sport: str) -> dict[str, float]:
        if sport not in self._lexicons:
            self._lexicons[sport] = load_sport_yaml(self.cfg.trackers.commentary.lexicon_dir, sport).get("keywords", {})
        return self._lexicons[sport]

    def add_feed(self, feed_id: str, sport: str, feed: Feed | None = None) -> FeedState:
        tc = self.cfg.trackers
        seeds = Seeds(sport, tc.default_seeds, tc.calibration_path)
        sync = SyncTable.identity(IDENTITY_SPAN_S, feed_id)
        replay = None
        if feed is not None:
            # Match records replay through the clip's sync table; session CSVs (weightlifting) are
            # already in video time, so they replay with the identity sync and need no table.
            timed = feed.match_ref.source == "session_csv"
            if feed.sync and resolve_path(feed.sync).exists():
                sync = SyncTable.load(resolve_path(feed.sync))
                timed = True
            match_id = feed.match_ref.match_id
            events_path = resolve_path(self.cfg.replay.events_dir) / sport / f"{match_id}.jsonl"
            if timed and match_id is not None and events_path.exists():
                replay = Replayer(feed_id, sport, load_records(events_path), sync)  # type: ignore[arg-type]
        fs = FeedState(
            feed_id=feed_id,
            sport=sport,
            sync=sync,
            adapter=make_adapter(sport),
            crowd=CrowdTracker(tc.crowd, tc.rolling, seeds),
            voice=VoiceTracker(tc.rolling, seeds, tc.voice.drop_after_s),
            commentary=CommentaryTracker(tc.commentary, self._lexicon(sport)),
            video=VideoTracker(sport, tc.video, tc.rolling, seeds),
            imminent=ImminentState(make_rule(sport, self._imminent_yaml(sport)), self.cfg.imminent),
            forecaster=Forecaster(sport, self.registry, self.cfg.forecast),
            ema=Ema(self.cfg.ema_alpha),
            trend=Trend(self.cfg.trend),
            replay=replay,
        )
        self.feeds[feed_id] = fs
        if sport not in self.sport_pct:
            self.sport_pct[sport] = SportPercentile(self.cfg.percentile, self.intensity_ref.get(sport))
        return fs

    # ------------------------------------------------------------------ ingest

    def handle(self, env: Envelope, now: float) -> None:
        """Apply one `/ingest` message. Events are applied as they arrive, which is never before
        the feed's clock reaches them, so rules only ever read the past."""
        data = env.data
        fs = self.feeds.get(data.feed_id)
        if fs is None:
            if not isinstance(data, Event):
                # Unregistered feed (a clone, say): its sport arrives with its first event. Keep the
                # latest tick so the feed has a clock from that moment.
                if isinstance(data, ClockTick):
                    self._pending_ticks[data.feed_id] = (data, now)
                self.metrics.inc("ingest.unknown_feed")
                return
            fs = self.add_feed(data.feed_id, data.sport)
            fs.loop_epoch = data.loop_epoch  # a new feed starts in the loop it arrives in: nothing to reset
            pending = self._pending_ticks.pop(data.feed_id, None)
            if pending is not None and pending[0].loop_epoch == data.loop_epoch:
                fs.clock.t_video, fs.clock.arrival = pending[0].t_video, pending[1]
        fs.seen = True

        if data.loop_epoch < fs.loop_epoch:
            self.metrics.inc("ingest.old_epoch")
            return
        if data.loop_epoch > fs.loop_epoch:
            fs.reset_for_loop(data.loop_epoch)
            self.metrics.inc("engine.loop_reset")

        if isinstance(data, ClockTick):
            fs.clock.t_video = data.t_video
            fs.clock.arrival = now
            fs.down = False
            self.metrics.inc("ingest.clock")
            if not fs.prerolled:
                fs.prerolled = True
                self._preroll(fs, data.t_video, now)
            return
        if isinstance(data, Control):
            self.metrics.inc(f"ingest.control.{data.kind}")
            if data.kind == "feed_down":
                fs.down = True
            elif data.kind == "feed_up":
                fs.down = False
            return

        self.metrics.inc(f"ingest.event.{data.source}")
        lag_ms = (time.time_ns() - data.t_emit_ns) / 1e6
        if 0 <= lag_ms < 60_000:  # tapes and simulated runs carry virtual send times
            self.metrics.observe("ingest_to_engine_ms", lag_ms)
        fs.last_source_at[data.source] = now
        self._apply_event(fs, data)

    def _apply_event(self, fs: FeedState, event: Event) -> None:
        src = event.source
        if src == "match":
            rec = MatchPayload.model_validate(event.payload)
            fs.adapter.on_match(rec)
            if rec.player:
                fs.commentary.add_names({rec.player})
        elif src == "crowd":
            fs.crowd.on_event(event)
        elif src == "voice":
            fs.voice.on_event(event)
        elif src == "video":
            fs.video.on_event(event)
        elif src == "commentary":
            fs.commentary.on_event(event)
        elif src == "ocr":
            fs.last_ocr = dict(event.payload.get("fields", {}))

    def _ranked(self, fs: FeedState, intensity: float, now: float) -> float:
        """What the sport's percentile ranks: the intensity (level), or how far it is above the feed's
        own baseline (spike). The baseline follows intensity with a time constant of spike_baseline_s."""
        pcfg = self.cfg.percentile
        if pcfg.mode == "spike":
            base = intensity if fs.baseline is None else fs.baseline
            ranked = intensity - base
            dt = max(0.0, now - fs.baseline_at) if fs.baseline is not None else 0.0
            fs.baseline = base + (intensity - base) * min(1.0, dt / max(1.0, pcfg.spike_baseline_s))
            fs.baseline_at = now
        else:
            ranked = intensity
        fs.last_ranked = ranked
        return ranked

    def _preroll(self, fs: FeedState, t_now: float, now: float) -> None:
        """Fill the sport's percentile window with this feed's intensity over the clip time before `t_now`.

        A shadow engine replays the feed's match records at 1 Hz from `t_now - preroll_s` (within the
        current loop, at most the window) to just before `t_now`, exactly as live; its intensities go into
        the window at the engine times they would have had. Without it a fresh engine ranks each feed
        against seconds of history, so after every (re)start the feed whose intensity only rises sits at
        100 for minutes. Feeds without match data (the mock, clones) are not pre-rolled.
        """
        pcfg = self.cfg.percentile
        feed = self.registry_feeds.get(fs.feed_id)
        span = min(pcfg.preroll_s, pcfg.window_s, t_now)
        if not self.allow_preroll or feed is None or fs.replay is None or span < 1.0:
            return
        started = time.monotonic()
        shadow = Engine(self.config, feeds=[feed], registry=self.registry, metrics=Metrics())
        shadow.allow_preroll = False
        history: list[tuple[float, float]] = []
        t = t_now - span
        while t < t_now - 0.5:
            shadow.handle(Envelope(kind="clock", data=ClockTick(feed_id=fs.feed_id, t_video=t, loop_epoch=0, t_emit_ns=0)), t)
            if any(u.feed_id == fs.feed_id for u in shadow.score_all(t)):
                history.append((now - (t_now - t), shadow.feeds[fs.feed_id].last_ranked))
            t += 1.0
        if fs.sport not in self.sport_pct:
            self.sport_pct[fs.sport] = SportPercentile(pcfg, self.intensity_ref.get(fs.sport))
        self.sport_pct[fs.sport].backfill(history)
        # Carry the smoothing over, so the live feed continues where the replay left off.
        warm = shadow.feeds[fs.feed_id]
        fs.ema.value = warm.ema.value
        fs.baseline, fs.baseline_at = warm.baseline, now
        self.metrics.inc("engine.preroll")
        log.info("pre-rolled %s: %d s of match history in %.0f ms", fs.feed_id, len(history), (time.monotonic() - started) * 1000)

    # ------------------------------------------------------------------ scoring

    def _release_replay(self, fs: FeedState, t: float, now: float) -> None:
        if fs.replay is None:
            return
        for ev in fs.replay.release(t, fs.loop_epoch):
            fs.last_source_at["match"] = now
            self._apply_event(fs, ev)

    def _cue_inputs(self, fs: FeedState, t: float, gt: float | None, adapter_facts: dict | None = None) -> CueInputs:
        fr = fs.last_forecast
        if adapter_facts is None:
            adapter_facts = fs.adapter.facts(gt if gt is not None else t)
        return CueInputs(
            t_video=t,
            game_time=gt,
            video=fs.video.cues(t),
            crowd=fs.crowd.cues(t),
            voice=fs.voice.cues(t),
            adapter_facts=adapter_facts,
            forecast_part=fr.part_value if fr else 0.0,
            stakes_part=fs.last_stakes,
            forecast_p=fr.forecast.p if fr else 0.5,
        )

    def _update_imminent(self, fs: FeedState, t: float, gt: float | None,
                         adapter_facts: dict | None = None) -> tuple[Imminent, bool]:
        if not self.cfg.imminent.enabled:
            return fs.imminent.current, False
        with Timer(self.metrics, "imminent_ms"):
            return fs.imminent.update(self._cue_inputs(fs, t, gt, adapter_facts))

    def tick_imminent(self, now: float) -> list[ScoreUpdate]:
        """Fast tick: run every imminent rule; a crossing scores that feed now (out of cycle)."""
        out = []
        for fs in self.feeds.values():
            if not fs.seen or fs.stale(now, self.cfg) or fs.last_forecast is None:
                continue
            t = fs.t_video(now, self.cfg)
            self._release_replay(fs, t, now)
            _, trigger = self._update_imminent(fs, t, fs.game_time(t))
            if trigger:
                self.metrics.inc("engine.out_of_cycle")
                out.append(self.score_feed(fs, now))
        return out

    def score_all(self, now: float) -> list[ScoreUpdate]:
        """The regular 1 Hz scores. A feed already scored out of cycle at this `now` is skipped, so
        its EMA, trend and percentile take one step, and clients get one update."""
        return [self.score_feed(fs, now) for fs in self.feeds.values() if fs.seen and fs.last_scored_at != now]

    def score_feed(self, fs: FeedState, now: float) -> ScoreUpdate:
        with Timer(self.metrics, "tick_ms"):
            update = self._score(fs, now)
        fs.last_update = update
        fs.last_scored_at = now
        return update

    def _score(self, fs: FeedState, now: float) -> ScoreUpdate:
        cfg = self.cfg
        t = fs.t_video(now, cfg)
        self._release_replay(fs, t, now)
        gt = fs.game_time(t)
        g = gt if gt is not None else t
        adapter_parts = fs.adapter.parts(g)
        adapter_facts = fs.adapter.facts(g, adapter_parts)

        fr = fs.forecaster.predict(fs.adapter, g)
        fs.last_forecast = fr
        fs.last_stakes = adapter_parts["stakes"].value if "stakes" in adapter_parts else 0.0
        imminent, _ = self._update_imminent(fs, t, gt, adapter_facts)

        parts: dict[str, AdapterPart | None] = {
            "closeness": adapter_parts.get("closeness"),
            "momentum": adapter_parts.get("momentum"),
            "stakes": adapter_parts.get("stakes"),
            "crowd": fs.crowd.part(t),
            "commentary": fs.commentary.part(t),
            "video": fs.video.part(t),
            "forecast": self._forecast_part(fr, adapter_facts),
            "imminent": self._imminent_part(fs, imminent) if cfg.imminent.enabled else None,
        }
        raw, part_values = combine(parts, self._weights_for(fs.sport))
        intensity = fs.ema.update(raw)
        trend, slope = fs.trend.update(t, intensity)
        ranked = self._ranked(fs, intensity, now)
        pct = self.sport_pct[fs.sport]
        pct.add(now, ranked)

        by: list[str] = []
        if fr.forecast.heating_up:
            by.append("forecast")
        if cfg.imminent.enabled and imminent.heating_up:
            by.append("imminent")
        if not cfg.imminent.enabled:
            imminent = Imminent(value=0.0, cue=0.0, context=cfg.imminent.context_floor, heating_up=False, phase=None, source="rule", facts=[])

        reasons = build_reasons(fs.sport, part_values, fr, imminent, adapter_facts, cfg.max_reasons)
        if self.polisher.enabled:
            for r in reasons:
                self.polisher.submit(fs.feed_id, r)
            reasons = [self.polisher.polished(fs.feed_id, r) for r in reasons]

        return ScoreUpdate(
            feed_id=fs.feed_id,
            sport=fs.sport,  # type: ignore[arg-type]
            t_video=round(t, 3),
            loop_epoch=fs.loop_epoch,
            game_clock=fs.adapter.game_clock(),
            t_emit_ns=time.time_ns(),
            stale=fs.stale(now, cfg),
            intensity_raw=round(raw, 4),
            intensity=round(intensity, 4),
            intensity_pct=pct.pct(ranked),
            trend=trend,  # type: ignore[arg-type]
            trend_slope=round(slope, 5),
            parts=part_values,
            forecast=fr.forecast,
            imminent=imminent,
            heating_up=bool(by),
            heating_up_by=by,  # type: ignore[arg-type]
            reasons=reasons,
        )

    def _forecast_part(self, fr: ForecastResult, adapter_facts: dict) -> AdapterPart:
        facts, _ = forecast_facts(fr, adapter_facts)
        label = label_phrase(fr.forecast.label)
        phrase = f"{label} brewing" if fr.part_value >= 0.6 else f"no sign of {label} yet"
        return AdapterPart(fr.part_value, phrase, facts)

    def _imminent_part(self, fs: FeedState, imminent: Imminent) -> AdapterPart:
        if imminent.phase:
            phrase = PHASE_TEXT.get(imminent.phase, imminent.phase)
        else:
            phrase = "something is building" if imminent.value >= fs.imminent.rule.threshold else "nothing building yet"
        return AdapterPart(imminent.value, phrase, list(imminent.facts))

    # ------------------------------------------------------------------ admin

    def reload(self) -> dict[str, Any]:
        versions = self.registry.reload()
        self.weights = load_weights(self.cfg.weights_path)
        self._imminent_cfg.clear()
        self._lexicons.clear()
        for fs in self.feeds.values():
            rule = fs.imminent.rule
            cfg = self._imminent_yaml(fs.sport)
            # Runs on the event loop between ticks, so a tick never sees half a swap.
            rule.cfg = cfg
            rule.threshold = float(cfg.get("threshold", rule.threshold))
            rule.rule_version = cfg.get("rule_version", rule.rule_version)
            fs.commentary.set_keywords(self._lexicon(fs.sport))
        return {
            "bundles": versions,
            "errors": self.registry.status()["errors"],
            "imminent_rules": {sport: cfg.get("rule_version") for sport, cfg in self._imminent_cfg.items()},
        }

    SWITCHES = ("engine.polish.enabled", "engine.forecast.source", "engine.imminent.enabled")

    def set_switch(self, key: str, value: Any) -> None:
        if key == "engine.polish.enabled":
            self.cfg.polish.enabled = bool(value)
        elif key == "engine.forecast.source":
            if value not in ("model", "heuristic"):
                raise ValueError("engine.forecast.source must be model or heuristic")
            self.cfg.forecast.source = value
        elif key == "engine.imminent.enabled":
            self.cfg.imminent.enabled = bool(value)
        else:
            raise KeyError(f"unknown switch {key}; known: {', '.join(self.SWITCHES)}")

    def health(self, now: float) -> dict[str, Any]:
        reg = self.registry.status()
        feeds = {}
        for fid, fs in self.feeds.items():
            fr = fs.last_forecast
            feeds[fid] = {
                "sport": fs.sport,
                "seen": fs.seen,
                "stale": fs.stale(now, self.cfg),
                "down": fs.down,
                "t_video": round(fs.t_video(now, self.cfg), 3),
                "loop_epoch": fs.loop_epoch,
                "last_event_age_s": {k: round(now - v, 2) for k, v in sorted(fs.last_source_at.items())},
                "forecast_source": fr.forecast.source if fr else None,
                "model_version": fr.forecast.model_version if fr else None,
                "forecast_error": (fr.error if fr else None) or reg["errors"].get(fs.sport),
                "imminent_rule": fs.imminent.rule.rule_version,
                "replay": fs.replay is not None,
                "ocr": fs.last_ocr or None,
            }
        ok = all(not f["stale"] for f in feeds.values() if f["seen"])
        return {
            "status": "ok" if ok else "degraded",
            "feeds": feeds,
            "models": reg,
            "switches": {
                "engine.polish.enabled": self.cfg.polish.enabled,
                "engine.forecast.source": self.cfg.forecast.source,
                "engine.imminent.enabled": self.cfg.imminent.enabled,
            },
            "polish": self.polisher.stats if self.cfg.polish.enabled else None,
        }


def default_feeds(config: Config) -> list[Feed]:
    from intensity.feeds import load_feeds

    path = resolve_path(config.engine.feeds_path)
    return load_feeds(path).feeds if Path(path).exists() else []
