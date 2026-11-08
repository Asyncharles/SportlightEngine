from __future__ import annotations

import random
import time

import numpy as np
import pytest

from intensity.engine import simulate
from intensity.engine.core import Engine
from intensity.engine.mock import MockGenerator
from intensity.schema import PART_NAMES, ClockTick, Control, Envelope, make_envelope

from .helpers import crowd, match


def tick(feed: str, t: float, epoch: int = 0) -> Envelope:
    return make_envelope(ClockTick(feed_id=feed, t_video=t, loop_epoch=epoch, t_emit_ns=0))


def test_scores_every_part_and_reports_dropouts(config):
    eng = Engine(config, [])
    eng.handle(make_envelope(crowd(0.0, feed="f")), 0.0)
    eng.handle(tick("f", 0.0), 0.0)
    (u,) = eng.score_all(0.5)
    assert set(u.parts) == set(PART_NAMES)
    assert u.parts["crowd"] is not None and u.parts["video"] is None  # no video yet: dropped out
    assert u.t_video == pytest.approx(0.5) and not u.stale
    assert u.forecast.source == "heuristic"


def test_feed_goes_stale_without_clock_ticks(config):
    eng = Engine(config, [])
    eng.handle(make_envelope(crowd(0.0, feed="f")), 0.0)
    eng.handle(tick("f", 10.0), 0.0)
    (u,) = eng.score_all(2.5)
    assert u.stale
    assert u.t_video == pytest.approx(10.5)  # the clock runs on at most max_extrapolate_s past a tick
    eng.handle(tick("f", 12.5), 3.0)
    assert not eng.score_all(3.0)[0].stale


def test_loop_resets_feed_state_but_keeps_sport_percentile(config):
    eng = Engine(config, [])
    eng.handle(make_envelope(match(1.0, "goal", team="A", player="P")), 0.0)
    eng.handle(tick("f", 1.0), 0.0)
    for t in range(5):
        eng.handle(make_envelope(crowd(float(t), loud=9.0, feed="f")), float(t))
        eng.handle(tick("f", float(t)), float(t))
        eng.score_all(float(t))
    fs = eng.feeds["f"]
    assert fs.adapter.goals == {"A": 1} and fs.ema.value is not None
    pct_samples = len(eng.sport_pct["football"].samples)
    eng.handle(make_envelope(Control(feed_id="f", kind="loop", loop_epoch=1, t_emit_ns=0)), 6.0)
    assert fs.loop_epoch == 1 and fs.adapter.goals == {} and fs.ema.value is None
    assert fs.crowd.last is None and len(fs.trend.buf) == 0
    assert len(eng.sport_pct["football"].samples) == pct_samples
    # stragglers from the old loop are ignored
    eng.handle(make_envelope(crowd(4.5, loud=9.0, feed="f")), 6.1)
    assert fs.crowd.last is None


def test_feed_down_marks_stale(config):
    eng = Engine(config, [])
    eng.handle(make_envelope(crowd(0.0, feed="f")), 0.0)
    eng.handle(tick("f", 0.0), 0.0)
    eng.handle(make_envelope(Control(feed_id="f", kind="feed_down", loop_epoch=0, t_emit_ns=0)), 0.1)
    assert eng.score_all(0.2)[0].stale
    eng.handle(tick("f", 0.3), 0.3)
    assert not eng.score_all(0.3)[0].stale


def test_heating_up_is_the_or_of_forecast_and_imminent(config):
    eng = Engine(config, [])
    gen = MockGenerator([("f", "football")], t0_ns=0)
    out = simulate.run(eng, gen.envelopes(0, 150))
    for e in out:
        u = e.update
        assert u.heating_up == (u.forecast.heating_up or u.imminent.heating_up)
        assert ("forecast" in u.heating_up_by) == u.forecast.heating_up
        assert ("imminent" in u.heating_up_by) == u.imminent.heating_up
    assert any(e.out_of_cycle for e in out)


def test_imminent_switch_off(config):
    eng = Engine(config, [])
    eng.set_switch("engine.imminent.enabled", False)
    out = simulate.run(eng, MockGenerator([("f", "football")], t0_ns=0).envelopes(0, 60))
    assert all(e.update.parts["imminent"] is None and not e.update.imminent.heating_up for e in out)
    with pytest.raises(KeyError):
        eng.set_switch("engine.nope", 1)


def test_imminent_leakage_later_cues_do_not_change_the_past(config):
    """imminent at t is unchanged when every cue after t is shuffled."""
    gen = MockGenerator([("f", "football")], t0_ns=0)
    envs = gen.envelopes(0, 120)
    cut_ns = int(70 * 1e9)
    before = [e for e in envs if e.data.t_emit_ns <= cut_ns]
    after = [e for e in envs if e.data.t_emit_ns > cut_ns and e.kind == "event"]
    rng = random.Random(0)
    shuffled_payloads = [e.data.payload for e in after]
    rng.shuffle(shuffled_payloads)
    mixed = [Envelope(kind="event", data=e.data.model_copy(update={"payload": p, "source": e.data.source}))
             for e, p in zip(after, shuffled_payloads, strict=True) if p.keys() == e.data.payload.keys()]

    def imminent_until(envelopes):
        out = simulate.run(Engine(config, []), envelopes, until_s=70.0)
        return [(e.update.t_video, e.update.imminent.value) for e in out]

    assert imminent_until(before) == imminent_until(before + mixed)


def test_mock_build_ups_raise_imminent_before_the_moment(config):
    gen = MockGenerator([("f", "football")], t0_ns=0)
    out = simulate.run(Engine(config, []), gen.envelopes(0, 300))
    moments = [m for m in gen.feed("f").moments(0) if 40 <= m <= 295]
    assert len(moments) >= 2
    by_t = {round(e.update.t_video, 1): e.update.imminent.value for e in out}
    for m in moments:
        build = max(v for t, v in by_t.items() if m - 6 <= t < m)
        calm = np.median([v for t, v in by_t.items() if m - 30 <= t < m - 15])
        assert build > 0.45 > calm


def test_tick_under_5ms_p95_with_mock_input(config):
    feeds = [("a", "football"), ("b", "f1"), ("c", "nba"), ("d", "weightlifting")]
    eng = Engine(config, [])
    simulate.run(eng, MockGenerator(feeds, t0_ns=0).envelopes(0, 120))
    times = []
    for fs in eng.feeds.values():
        for i in range(50):
            t0 = time.perf_counter()
            eng.score_feed(fs, 120.0 + i * 0.01)
            times.append((time.perf_counter() - t0) * 1000)
    assert float(np.percentile(times, 95)) < 5.0


def test_health_reports_per_feed_state(config):
    eng = Engine(config, [])
    simulate.run(eng, MockGenerator([("f", "football")], t0_ns=0).envelopes(0, 10))
    h = eng.health(10.0)
    f = h["feeds"]["f"]
    assert f["imminent_rule"] == "football-r1" and f["forecast_source"] == "heuristic"
    assert {"video", "crowd", "voice"} <= set(f["last_event_age_s"])


def test_unregistered_feed_keeps_its_first_clock_tick(config):
    eng = Engine(config, [])
    eng.handle(tick("clone-1", 42.0), 0.0)  # sport unknown until the first event
    eng.handle(make_envelope(crowd(42.0, feed="clone-1")), 0.0)
    (u,) = eng.score_all(0.0)
    assert not u.stale and u.t_video == 42.0


def test_unregistered_feed_in_a_later_loop_keeps_its_tick(config):
    eng = Engine(config, [])
    eng.handle(tick("clone-2", 42.0, epoch=2), 0.0)
    eng.handle(make_envelope(crowd(42.0, feed="clone-2", epoch=2)), 0.0)
    fs = eng.feeds["clone-2"]
    assert fs.loop_epoch == 2 and fs.loops == 0 and fs.clock.t_video == 42.0
    assert eng.metrics.counters.get("engine.loop_reset", 0) == 0


def test_feed_scored_once_per_step(config):
    eng = Engine(config, [])
    eng.handle(make_envelope(crowd(0.0, feed="f")), 0.0)
    eng.handle(tick("f", 0.0), 0.0)
    fs = eng.feeds["f"]
    eng.score_feed(fs, 1.0)  # as an out-of-cycle update would
    ema = fs.ema.value
    assert eng.score_all(1.0) == []  # the regular tick at the same step skips it
    assert fs.ema.value == ema and len(fs.trend.buf) == 1
    assert len(eng.score_all(2.0)) == 1


def test_clock_never_steps_back_after_a_stall(config):
    eng = Engine(config, [])
    eng.handle(make_envelope(crowd(0.0, feed="f")), 0.0)
    eng.handle(tick("f", 10.0), 0.0)
    fs = eng.feeds["f"]
    assert fs.t_video(1.9, config.engine) == pytest.approx(10.5)  # stalled: no running ahead
    eng.handle(tick("f", 10.25), 2.0)
    assert fs.t_video(2.0, config.engine) == pytest.approx(10.5)  # holds rather than jumping back
    assert fs.t_video(2.5, config.engine) == pytest.approx(10.75)


def test_reload_swaps_rule_version_and_lexicon(config, tmp_path):
    imminent = tmp_path / "imminent"
    lexicon = tmp_path / "lexicon"
    imminent.mkdir()
    lexicon.mkdir()
    (imminent / "football.yaml").write_text("rule_version: football-r1\nthreshold: 0.45\n")
    (lexicon / "football.yaml").write_text("keywords: {goal: 1.0}\n")
    config.engine.imminent.config_dir = str(imminent)
    config.engine.trackers.commentary.lexicon_dir = str(lexicon)
    eng = Engine(config, [])
    eng.handle(make_envelope(crowd(0.0, feed="f")), 0.0)
    (imminent / "football.yaml").write_text("rule_version: football-r2\nthreshold: 0.6\n")
    (lexicon / "football.yaml").write_text("keywords: {screamer: 1.0}\n")
    out = eng.reload()
    fs = eng.feeds["f"]
    assert out["imminent_rules"]["football"] == "football-r2"
    assert eng.health(0.0)["feeds"]["f"]["imminent_rule"] == "football-r2"
    assert fs.imminent.rule.threshold == 0.6
    assert set(fs.commentary.keywords) == {"screamer"}


def test_session_csv_feed_replays_in_video_time(tmp_path):
    """Weightlifting sessions have no sync table: their records are already in video time."""
    from intensity.config import load_config
    from intensity.feeds import Feed

    (tmp_path / "weightlifting").mkdir()
    (tmp_path / "weightlifting" / "s1.jsonl").write_text(
        '{"game_time_s": 10.0, "period": 1, "kind": "attempt_called", "player": "A",'
        ' "attrs": {"weight_kg": 150, "attempt": 2, "best_kg": 145}}\n'
    )
    cfg = load_config(overrides=["engine.forecast.source=heuristic", "engine.polish.enabled=false",
                                 f"engine.replay.events_dir={tmp_path}"])
    feed = Feed(feed_id="wl", sport="weightlifting", sample="samples/s1.mp4", rtsp_path="wl",
                match_ref={"source": "session_csv", "match_id": "s1"}, sync=None)
    eng = Engine(cfg, feeds=[feed])
    fs = eng.feeds["wl"]
    assert fs.replay is not None
    eng.handle(tick("wl", 9.0), 0.0)
    eng.score_all(0.0)
    assert fs.adapter.weight == 0.0  # not yet: the walk-on is at t_video 10
    eng.handle(tick("wl", 11.0), 2.0)
    update = eng.score_all(2.0)[0]
    assert fs.adapter.weight == 150.0 and update.intensity_pct is not None


def test_first_clock_tick_prerolls_the_feeds_match_history(tmp_path):
    from intensity.config import load_config
    from intensity.feeds import Feed

    (tmp_path / "weightlifting").mkdir()
    lines = [
        f'{{"game_time_s": {t}, "period": 1, "kind": "attempt_called", "player": "A{t}",'
        f' "attrs": {{"weight_kg": {140 + t // 60}, "attempt": 1, "best_kg": 140}}}}'
        for t in range(30, 600, 60)
    ]
    (tmp_path / "weightlifting" / "s1.jsonl").write_text("\n".join(lines) + "\n")
    base = ["engine.forecast.source=heuristic", "engine.polish.enabled=false", f"engine.replay.events_dir={tmp_path}"]
    feed = Feed(feed_id="wl", sport="weightlifting", sample="samples/s1.mp4", rtsp_path="wl",
                match_ref={"source": "session_csv", "match_id": "s1"}, sync=None)

    eng = Engine(load_config(overrides=[*base, "engine.percentile.preroll_s=300"]), feeds=[feed])
    eng.handle(tick("wl", 500.0), 1000.0)
    window = eng.sport_pct["weightlifting"].samples
    assert len(window) == 300 and window[0][0] == pytest.approx(700.0) and window[-1][0] < 1000.0
    assert eng.metrics.counters["engine.preroll"] == 1
    eng.handle(tick("wl", 501.0), 1001.0)  # only the first tick pre-rolls
    assert eng.metrics.counters["engine.preroll"] == 1

    off = Engine(load_config(overrides=[*base, "engine.percentile.preroll_s=0"]), feeds=[feed])
    off.handle(tick("wl", 500.0), 1000.0)
    assert len(off.sport_pct["weightlifting"].samples) == 0 and "engine.preroll" not in off.metrics.counters


def test_feeds_without_match_data_are_not_prerolled(config):
    eng = Engine(config)  # an unregistered feed (the mock, a clone): no replay
    eng.handle(make_envelope(match(1.0, "kickoff")), 0.0)  # feed "f" joins with its first event
    eng.handle(tick("f", 900.0), 1.0)
    assert eng.feeds["f"].prerolled and "engine.preroll" not in eng.metrics.counters


def test_spike_mode_ranks_moments_not_slow_climbs():
    from intensity.config import load_config

    cfg = load_config(overrides=["engine.forecast.source=heuristic", "engine.polish.enabled=false",
                                 "engine.percentile.mode=spike", "engine.percentile.spike_baseline_s=60",
                                 "engine.percentile.tie_tolerance=0.005", "engine.percentile.min_samples=3"])
    eng = Engine(cfg)
    fs = eng.add_feed("f1", "f1")
    pct = eng.sport_pct["f1"]
    for t in range(300):  # a slow climb, like F1's stakes lap by lap
        pct.add(float(t), eng._ranked(fs, 0.40 + 0.0004 * t, float(t)))
    climbing = pct.pct(eng._ranked(fs, 0.40 + 0.0004 * 300, 300.0))
    jump = pct.pct(eng._ranked(fs, 0.40 + 0.0004 * 301 + 0.15, 301.0))  # a moment
    assert climbing < 80 and jump == 100.0
    fs.reset_for_loop(1)
    assert fs.baseline is None  # a new loop starts a new baseline


def test_spike_mode_preroll_fills_the_window_with_spikes_and_hands_over(tmp_path):
    from intensity.config import load_config
    from intensity.feeds import Feed

    (tmp_path / "weightlifting").mkdir()
    lines = [
        f'{{"game_time_s": {t}, "period": 1, "kind": "attempt_called", "player": "A{t}",'
        f' "attrs": {{"weight_kg": {140 + t // 60}, "attempt": 1, "best_kg": 140}}}}'
        for t in range(30, 600, 60)
    ]
    (tmp_path / "weightlifting" / "s1.jsonl").write_text("\n".join(lines) + "\n")
    cfg = load_config(overrides=["engine.forecast.source=heuristic", "engine.polish.enabled=false",
                                 f"engine.replay.events_dir={tmp_path}", "engine.percentile.preroll_s=300",
                                 "engine.percentile.mode=spike"])
    feed = Feed(feed_id="wl", sport="weightlifting", sample="samples/s1.mp4", rtsp_path="wl",
                match_ref={"source": "session_csv", "match_id": "s1"}, sync=None)
    eng = Engine(cfg, feeds=[feed])
    eng.handle(tick("wl", 500.0), 1000.0)
    values = [v for _, v in eng.sport_pct["weightlifting"].samples]
    assert len(values) == 300 and min(values) < 0.0 < max(values) + 1e-9  # spikes around the baseline
    fs = eng.feeds["wl"]
    assert fs.baseline is not None and fs.ema.value is not None and fs.baseline_at == 1000.0
