from __future__ import annotations

import json
import threading
import time

from fastapi.testclient import TestClient

from intensity.engine.service import create_app
from intensity.schema import ClockTick, Fact, make_envelope

from .helpers import crowd, video, voice


def test_endpoints_with_mock(config):
    app = create_app(config, feeds=[], mock=False, run_loops=False)
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/feeds").json() == {"feeds": []}
        assert "counters" in client.get("/metrics").json()
        assert client.post("/admin/reload").status_code == 200
        r = client.post("/admin/switch", json={"key": "engine.imminent.enabled", "value": False})
        assert r.json()["engine.imminent.enabled"] is False
        assert client.post("/admin/switch", json={"key": "engine.bogus", "value": 1}).status_code == 400
        assert client.post("/admin/switch", json={"key": "engine.forecast.source", "value": "magic"}).status_code == 400
        assert "Engine debug" in client.get("/debug").text


def test_ingest_validates_and_scores_stream(config):
    app = create_app(config, mock=True, run_loops=True)  # feeds from config/feeds.yaml
    with TestClient(app) as client:
        with client.websocket_connect("/ingest") as ingest:
            ingest.send_text('{"kind": "event", "data": {"nonsense": true}}')
        with client.websocket_connect("/ws/scores") as ws:
            first = json.loads(ws.receive_text())
            assert first["v"] == 1 and set(first["parts"]) >= {"closeness", "imminent"}
        metrics = client.get("/metrics").json()["counters"]
        assert metrics.get("ingest.invalid") == 1
        assert metrics.get("ingest.event.video", 0) > 0
        feeds = client.get("/feeds").json()["feeds"]
        assert {f["feed_id"] for f in feeds} >= {"fb-arg-fra", "wl-demo-session"}


def test_imminent_crossing_reaches_ws_out_of_cycle(config):
    """A build-up late in a level game crosses the threshold; the update is pushed at once."""
    app = create_app(config, feeds=[], mock=False, run_loops=True)
    received: list[tuple[int, dict]] = []
    stop = threading.Event()
    with TestClient(app) as client, client.websocket_connect("/ws/scores") as ws, client.websocket_connect("/ingest") as ingest:

        def reader() -> None:
            while not stop.is_set():
                try:
                    msg = ws.receive_text()
                except Exception:
                    return
                received.append((time.time_ns(), json.loads(msg)))

        threading.Thread(target=reader, daemon=True).start()
        t0 = 5280.0  # minute 88, 0-0: high stakes
        start = time.monotonic()
        for i in range(36):
            t = t0 + i * 0.1
            strong = t >= t0 + 1.0
            ingest.send_text(make_envelope(ClockTick(feed_id="fb", t_video=t, loop_epoch=0, t_emit_ns=time.time_ns())).model_dump_json())
            if i % 2 == 0:
                for e in (
                    video(t, pan=0.9 if strong else 0.05, sustained=3.0 if strong else 0.0, feed="fb",
                          facts=[Fact(key="pan_speed", value=0.9, text="camera panning 0.9 frame widths a second")]),
                    crowd(t, rise=5.0 if strong else 0.0, feed="fb"),
                    voice(t, energy=8.0 if strong else 0.0, pitch=4.0 if strong else 0.0, feed="fb"),
                ):
                    e.t_emit_ns = time.time_ns()
                    ingest.send_text(make_envelope(e).model_dump_json())
            time.sleep(max(0.0, start + (i + 1) * 0.1 - time.monotonic()))
        time.sleep(0.4)
        stop.set()
        metrics = client.get("/metrics").json()
    hot = [(ns, u) for ns, u in received if "imminent" in u["heating_up_by"]]
    assert hot, "imminent never crossed"
    recv_ns, first = hot[0]
    assert (recv_ns - first["t_emit_ns"]) / 1e6 < 200  # crossing scored, then delivered within 200 ms
    assert first["t_video"] - (t0 + 1.0 + config.engine.imminent.min_on_s) < 0.35  # not waiting for the 1 Hz tick
    assert metrics["counters"].get("engine.out_of_cycle", 0) >= 1
    assert first["reasons"][0]["factors"] == ["imminent"]
