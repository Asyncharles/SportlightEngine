"""Serve parity: the live engine's forecasts equal the offline evaluation's within 1e-6.

The engine replays a match on a simulated clock exactly as live (ClockTicks in, match records released
by the replayer, `score_all` at 1 Hz). Each `ScoreUpdate.forecast.p` must equal the offline transform
(the adapter replayed over the records, `featurize`'s features at that game time, `evaluate.predict`)
at the same moment, rounded to the 5 decimals the contract serves. This is the proof that training
equals serving, so it holds in both split modes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from intensity.config import load_config, resolve_path
from intensity.engine.adapters import make_adapter
from intensity.engine.core import Engine
from intensity.engine.registry import Registry
from intensity.engine.replay import Replayer, load_records
from intensity.feeds import load_feeds
from intensity.schema import ClockTick, Envelope, MatchPayload
from intensity.sync import SyncTable
from intensity.training.evaluate import predict
from intensity.training.featurize import featurize_match, ordered
from tests.unit.training.test_pipeline import _lightgbm, synthetic_match

TOL = 1e-6


def offline_p(bundle, sport: str, records: list[MatchPayload], moments: list[tuple[int, float]]) -> np.ndarray:
    """The offline transform at each (period, game time): adapter fed every record up to it, then the bundle."""
    recs = ordered(records)
    order = sorted(range(len(moments)), key=lambda k: moments[k])
    adapter, i = make_adapter(sport), 0
    rows = np.empty((len(moments), len(bundle.features)))
    for k in order:
        period, g = moments[k]
        while i < len(recs) and (recs[i].period, recs[i].game_time_s) <= (period, g):
            adapter.on_match(recs[i])
            i += 1
        rows[k] = adapter.features(g)
    return predict(bundle, rows)


def live_forecasts(engine: Engine, feed_id: str, t_end: float) -> list[tuple[float, float]]:
    """(t_video, served p) for every regular 1 Hz score, ticking the clock like the ingest would."""
    fast = engine.cfg.service.fast_tick_s
    steps_per_score = max(1, round(engine.cfg.service.score_interval_s / fast))
    out, step = [], 0
    while step * fast <= t_end + 1e-9:
        now = step * fast
        engine.handle(Envelope(kind="clock", data=ClockTick(feed_id=feed_id, t_video=now, loop_epoch=0, t_emit_ns=0)), now)
        engine.tick_imminent(now)
        if step % steps_per_score == 0:
            for upd in engine.score_all(now):
                if upd.feed_id == feed_id and upd.forecast.source == "model":
                    out.append((upd.t_video, upd.forecast.p))
        step += 1
    return out


@pytest.mark.skipif(_lightgbm() is None, reason="lightgbm unavailable (on macOS: brew install libomp)")
def test_live_forecasts_equal_offline_on_a_synthetic_match(tmp_path: Path) -> None:
    from intensity.training.cli import main

    events = tmp_path / "events" / "football"
    events.mkdir(parents=True)
    for i in range(8):
        (events / f"m{i:02d}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in synthetic_match(i)))
    (tmp_path / "feeds.yaml").write_text(
        "feeds:\n  - {feed_id: fb, sport: football, sample: samples/fb.mp4, rtsp_path: fb,"
        " match_ref: {source: statsbomb, match_id: m03}}\n"
    )
    highlights = tmp_path / "highlights"
    highlights.mkdir()
    (highlights / "m03.yaml").write_text("highlights:\n  - {period: 1, clock: 100, what: chance}\n")
    overrides = [f"training.paths.{k}={tmp_path / k}" for k in ("tables", "runs", "models", "reports", "highlights")]
    overrides += [
        f"training.paths.events={tmp_path / 'events'}",
        f"training.paths.feeds={tmp_path / 'feeds.yaml'}",
        "training.in_sample.num_boost_round=30",
        "training.evaluate.shap_rows=200",
        "training.evaluate.target_pr_auc_mult=0.0",
        "training.highlights.target_hits=0",
    ]
    assert main([a for o in overrides for a in ("--set", o)] + ["build", "--sport", "football", "--no-source"]) == 0
    version = (tmp_path / "runs" / "football" / "LATEST").read_text().strip()
    (tmp_path / "models" / "football" / "CURRENT").write_text(version)
    registry = Registry(tmp_path / "models")
    registry.reload()
    bundle = registry.get("football")
    assert bundle is not None and bundle.version == version

    config = load_config(overrides=["engine.forecast.source=model", f"engine.forecast.models_dir={tmp_path / 'models'}"])
    engine = Engine(config, registry=registry)
    records = [MatchPayload.model_validate(r) for r in synthetic_match(3)]
    half = 600.0  # first half only: game time equals t_video under the identity sync
    fs = engine.add_feed("fb", "football")
    fs.sync = SyncTable.identity(half, "fb")
    fs.replay = Replayer("fb", "football", records, fs.sync)

    served = live_forecasts(engine, "fb", half - 1)
    assert len(served) >= half - 2
    moments = [(1, t) for t, _ in served]
    expected = np.round(offline_p(bundle, "football", records, moments), 5)
    got = np.array([p for _, p in served])
    assert np.max(np.abs(got - expected)) <= TOL

    # And the offline reference is exactly featurize's rows, the table the model trained on.
    rows = featurize_match("football", records, 1.0, skip_periods=(5,))
    first = rows.period == 1
    whole = [(1, float(g)) for g in rows.game_time_s[first] if g < half - 1]
    np.testing.assert_allclose(
        offline_p(bundle, "football", records, whole), predict(bundle, rows.x[first][: len(whole)]), atol=TOL
    )


def _demo_cases() -> list[tuple[str, str]]:
    feeds = load_feeds(resolve_path("config/feeds.yaml"))
    return [(f.feed_id, f.sport) for f in feeds.feeds if f.sync]


@pytest.mark.slow
@pytest.mark.skipif(_lightgbm() is None, reason="lightgbm unavailable (on macOS: brew install libomp)")
@pytest.mark.parametrize(("feed_id", "sport"), _demo_cases())
def test_live_forecasts_equal_offline_on_each_demo_match(feed_id: str, sport: str) -> None:
    config = load_config(overrides=["engine.forecast.source=model"])
    registry = Registry(config.engine.forecast.models_dir)
    registry.reload()
    bundle = registry.get(sport)
    feed = load_feeds(resolve_path("config/feeds.yaml")).get(feed_id)
    events = resolve_path(config.engine.replay.events_dir) / sport / f"{feed.match_ref.match_id}.jsonl"
    if bundle is None or not events.exists() or not resolve_path(feed.sync).exists():
        pytest.skip(f"{feed_id}: needs models/{sport}/CURRENT, {events.name} and the sync table (make data-pull)")
    engine = Engine(config, feeds=[feed], registry=registry)
    sync = SyncTable.load(resolve_path(feed.sync))
    t_end = max(r.video_end for r in sync.rows) - 1
    served = [(t, p) for t, p in live_forecasts(engine, feed_id, t_end) if sync.game_time(t) is not None]
    assert served, f"{feed_id}: no model forecasts served"
    moments = [tuple(reversed(sync.game_time(t))) for t, _ in served]
    expected = np.round(offline_p(bundle, sport, load_records(events), moments), 5)
    got = np.array([p for _, p in served])
    assert np.max(np.abs(got - expected)) <= TOL, f"{feed_id}: live and offline forecasts differ"
