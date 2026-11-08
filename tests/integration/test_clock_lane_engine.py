"""The clock lane drives the real engine service: a looping clip's clock in, ScoreUpdates at 1 Hz out.

Both processes run in-process on localhost ports, on a synthetic clip and a weightlifting session
(already in video time), so no footage or bucket data is needed. This is the CPU stand-in for the
VM's done-when check: every feed scores at 1 Hz with match parts present.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from pathlib import Path

import httpx
import uvicorn
from websockets.sync.client import connect

from intensity.config import load_config
from intensity.engine.service import create_app
from intensity.feeds import load_feeds
from intensity.ingest.run import ClockIngest, make_app
from tests.unit.ingest.test_clock_lane import synthetic_mp4


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve_in_thread(app, port: int) -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, f"server on :{port} did not start"
    return server


def test_clock_lane_makes_the_engine_score_each_feed_at_1hz(tmp_path: Path, monkeypatch) -> None:
    data = tmp_path / "data"
    monkeypatch.setenv("INTENSITY_DATA_DIR", str(data))
    synthetic_mp4(data / "samples" / "wl_test.mp4", seconds=20.0, fps=5)
    (data / "events" / "weightlifting").mkdir(parents=True)
    (data / "events" / "weightlifting" / "wl_test.jsonl").write_text(
        '{"game_time_s": 0.5, "period": 1, "kind": "attempt_called", "player": "A",'
        ' "attrs": {"weight_kg": 150, "attempt": 2, "best_kg": 147}}\n'
    )
    feeds_yaml = tmp_path / "feeds.yaml"
    feeds_yaml.write_text(
        "feeds:\n  - {feed_id: wl-test, sport: weightlifting, sample: samples/wl_test.mp4, rtsp_path: wl-test,"
        " match_ref: {source: session_csv, match_id: wl_test}, sync: null}\n"
    )
    config = load_config(overrides=["engine.forecast.source=heuristic", "engine.polish.enabled=false"])
    engine_port, health_port = free_port(), free_port()
    engine = serve_in_thread(create_app(config, feeds=load_feeds(feeds_yaml).feeds, run_loops=True), engine_port)

    cfg = {**config.ingest, "feeds_state": str(tmp_path / "run" / "feeds_state.json")}
    ingest = ClockIngest(cfg, f"ws://127.0.0.1:{engine_port}/ingest", str(feeds_yaml))
    assert [c.feed_id for c in ingest.clocks] == ["wl-test"] and not ingest.problems
    health_server = serve_in_thread(make_app(ingest), health_port)
    async def lane() -> None:
        await asyncio.gather(ingest.client.run(), ingest.tick_loop())

    threading.Thread(target=lambda: asyncio.run(lane()), daemon=True).start()
    try:
        updates = []
        with connect(f"ws://127.0.0.1:{engine_port}/ws/scores", open_timeout=5) as ws:
            deadline = time.monotonic() + 4.5
            while time.monotonic() < deadline:
                try:
                    updates.append(json.loads(ws.recv(timeout=0.5)))
                except TimeoutError:
                    continue
        live = [u for u in updates if u["feed_id"] == "wl-test" and not u["stale"]]
        assert len(live) >= 3, f"expected ~1 Hz scores, got {len(live)}"
        gaps = [(b["t_emit_ns"] - a["t_emit_ns"]) / 1e9 for a, b in zip(live, live[1:], strict=False)]
        assert all(0.5 <= g <= 1.5 for g in gaps), gaps
        assert all(u["intensity_pct"] is not None for u in live)
        assert live[-1]["parts"]["stakes"] is not None  # match data replayed in clip time
        assert 0.0 <= live[-1]["t_video"] < 20.0 and live[-1]["loop_epoch"] == 0

        health = httpx.get(f"http://127.0.0.1:{health_port}/health", timeout=5)
        assert health.status_code == 200 and health.json()["status"] == "ok"
        assert health.json()["engine_connected"] is True
    finally:
        ingest.client.stop()
        engine.should_exit = health_server.should_exit = True
