"""The engine performance report: probes leave scoring unchanged, and a small run writes every output."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from intensity.engine import core, scorer, simulate
from intensity.engine.core import Engine
from intensity.engine.explain import templates
from intensity.engine.mock import MockGenerator
from intensity.engine.perf.probe import Probe, overhead_us

FEEDS = [("fb", "football"), ("f1", "f1"), ("nba", "nba"), ("wl", "weightlifting")]


def _scores(config, probe: Probe | None) -> list[dict]:
    engine = Engine(config)
    if probe is not None:
        probe.instrument(engine)
    envs = MockGenerator(FEEDS, seed=3, duration_s=120.0, t0_ns=0).envelopes(0.0, 60.0)
    return [e.update.model_dump(exclude={"t_emit_ns"}) | {"now": e.now, "ooc": e.out_of_cycle}
            for e in simulate.run(engine, envs)]


def test_probed_engine_scores_exactly_like_an_unprobed_one(config) -> None:
    plain = _scores(config, None)
    with Probe("test") as probe:
        probed = _scores(config, probe)
        assert core.combine is not scorer.combine  # patched while the probe is open
        frame = probe.frame()
    assert probed == plain
    assert core.combine is scorer.combine and core.build_reasons is templates.build_reasons
    stages = set(frame["stage"].astype(str))
    assert {"tick", "handle.clock", "handle.event.crowd", "adapter.parts", "forecast", "imminent", "combine",
            "reasons", "percentile.rank", "trackers.video.on_event", "fast_tick", "score_all"} <= stages
    assert (frame["ms"] >= 0).all()
    ticks = frame[frame["stage"] == "tick"]
    assert len(ticks) == len(plain)
    assert set(ticks["feed_id"].astype(str)) == {f for f, _ in FEEDS}


def test_close_removes_every_wrapper(config) -> None:
    engine = Engine(config)
    fs = engine.add_feed("fb", "football")
    probe = Probe("test")
    probe.instrument(engine)
    assert "score_feed" in vars(engine) and "parts" in vars(fs.adapter)
    probe.close()
    assert not {"handle", "score_feed", "tick_imminent", "score_all", "_preroll", "add_feed"} & set(vars(engine))
    assert "parts" not in vars(fs.adapter) and "predict" not in vars(fs.forecaster)


def test_overhead_is_small() -> None:
    assert 0 <= overhead_us(calls=20_000) < 20


def test_small_run_writes_the_report(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("matplotlib")
    from intensity.engine.perf.__main__ import main

    data = tmp_path / "data"
    monkeypatch.setenv("INTENSITY_DATA_DIR", str(data))
    events = data / "events" / "weightlifting"
    events.mkdir(parents=True)
    (events / "wl_test.jsonl").write_text("".join(
        json.dumps({"game_time_s": 5.0 + 20 * i, "period": 1, "kind": "attempt_called", "player": "A",
                    "attrs": {"weight_kg": 150 + i, "attempt": 1 + i % 3, "best_kg": 147}}) + "\n" for i in range(4)
    ))
    feeds = tmp_path / "feeds.yaml"
    feeds.write_text(
        "feeds:\n"
        "  - {feed_id: wl-test, sport: weightlifting, sample: samples/wl_test.mp4, rtsp_path: wl-test,"
        " match_ref: {source: session_csv, match_id: wl_test}, sync: null}\n"
        "  - {feed_id: fb-test, sport: football, sample: samples/fb_test.mp4, rtsp_path: fb-test,"
        " match_ref: {source: statsbomb, match_id: 1}, sync: null}\n"
    )
    perf = tmp_path / "perf.yaml"
    perf.write_text(
        "perf:\n  overlays: []\n  nice: 0\n  out_dir: data/reports/engine\n"
        "  startup: {restart_at_s: 60.0}\n  demo: {duration_s: 40.0}\n  perception: {minutes: 0.5}\n"
        "  scaling: {feeds: [2, 4], warmup_s: 5.0, duration_s: 10.0}\n  transport: {seconds: 2.0}\n"
        "  budgets_ms: {ingest_to_engine: 20, tick_per_feed: 50, tick_per_feed_target: 5, imminent_per_feed: 1,"
        " forecast_per_feed: 2, imminent_to_ws: 200, tick_to_ws_clients: 10}\n"
    )
    code = main(["--perf-config", str(perf), "--no-push", "--set", f"engine.feeds_path={feeds}",
                 "--set", "engine.forecast.source=heuristic"])
    assert code == 0
    (run,) = (data / "reports" / "engine").iterdir()
    for name in ("report.html", "summary.json", "samples.parquet", "config.resolved.yaml", "engine_perf.yaml",
                 "stages_demo.csv", "scaling_steps.parquet", "transport.parquet", "charts/tick_breakdown.png"):
        assert (run / name).exists(), name
    summary = json.loads((run / "summary.json").read_text())
    resolved = yaml.safe_load((run / "config.resolved.yaml").read_text())["engine"]
    assert resolved["publish"]["enabled"] is False and resolved["polish"]["enabled"] is False
    assert summary["demo"]["volume"]["updates_per_feed"] == {"fb-test": 41, "wl-test": 41}
    assert summary["demo"]["volume"]["msgs_in"]["clock"] == 2 * 161  # 4 Hz for 40 s, both feeds
    assert [r["feeds"] for r in summary["scaling"]["rows"]] == [2, 4]
    assert summary["transport"]["ingest_to_engine_ms"]["count"] > 0
    assert {b["stage"] for b in summary["budget"]} >= {"Engine tick per feed, all stages"}
    assert "Engine performance" in (run / "report.html").read_text()
