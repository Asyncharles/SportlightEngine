"""The engine service: `python -m intensity.engine.service [--mock] [--overlay local]`.

Endpoints:
    WS   /ingest       Envelopes from the ingest (or the mock), validated one by one
    WS   /ws/scores    ScoreUpdates; a slow client gets only the latest per feed, never a queue
    GET  /feeds        the feed registry and which feeds are live
    GET  /health       staleness, last-event age per source, model versions per feed
    GET  /metrics      JSON counters and latency histograms
    POST /admin/reload swap model bundles, weights and imminent configs between ticks
    POST /admin/switch {"key": "engine.imminent.enabled", "value": false}
    GET  /debug        plain HTML page with parts bars, imminent and phase per feed
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import ValidationError

from intensity.config import Config, load_config, resolve_path, save_resolved
from intensity.engine.core import Engine, default_feeds
from intensity.feeds import Feed
from intensity.schema import ScoreUpdate, parse_envelope
from intensity.telemetry import Metrics, log_event, setup_logging

log = logging.getLogger(__name__)
DEBUG_HTML = Path(__file__).with_name("debug.html")


class ScoreClient:
    """Latest-wins per feed: a slow client skips stale updates instead of queueing them."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.pending: dict[str, str] = {}
        self.wake = asyncio.Event()

    def offer(self, feed_id: str, payload: str) -> None:
        self.pending[feed_id] = payload
        self.wake.set()

    async def run(self) -> None:
        while True:
            await self.wake.wait()
            self.wake.clear()
            batch, self.pending = self.pending, {}
            for payload in batch.values():
                await self.ws.send_text(payload)


class Broadcaster:
    def __init__(self, metrics: Metrics):
        self.clients: set[ScoreClient] = set()
        self.metrics = metrics
        self.latest: dict[str, str] = {}

    def publish(self, update: ScoreUpdate) -> None:
        payload = update.model_dump_json()
        self.latest[update.feed_id] = payload
        for client in self.clients:
            client.offer(update.feed_id, payload)
        self.metrics.inc("scores.published")


class EngineService:
    def __init__(self, config: Config, feeds: list[Feed], mock: bool = False, mock_seed: int = 7):
        self.config = config
        self.metrics = Metrics()
        self.engine = Engine(config, feeds, metrics=self.metrics)
        self.broadcaster = Broadcaster(self.metrics)
        self.mock = mock
        self.mock_seed = mock_seed
        self.feeds = feeds
        self._tasks: list[asyncio.Task] = []

    @staticmethod
    def now() -> float:
        return time.monotonic()

    async def tick_loop(self) -> None:
        svc = self.config.engine.service
        start = self.now()
        step = 0
        steps_per_score = max(1, round(svc.score_interval_s / svc.fast_tick_s))
        while True:
            now = self.now()
            try:
                for update in self.engine.tick_imminent(now):
                    self.broadcaster.publish(update)
                    self.metrics.observe("imminent_to_ws_ms", (self.now() - now) * 1000.0)
                if step % steps_per_score == 0:
                    t0 = self.now()
                    for update in self.engine.score_all(now):
                        self.broadcaster.publish(update)
                    self.metrics.observe("score_all_ms", (self.now() - t0) * 1000.0)
            except Exception:  # a bad tick must not stop the service
                log.exception("tick failed")
                self.metrics.inc("engine.tick_errors")
            step += 1
            await asyncio.sleep(max(0.0, start + step * svc.fast_tick_s - self.now()))

    async def mock_loop(self) -> None:
        from intensity.engine.mock import STEP_S, MockGenerator

        gen = MockGenerator([(f.feed_id, f.sport) for f in self.feeds], self.mock_seed)
        start = self.now()
        t = 0.0
        while True:
            for env in gen.envelopes(t, t + STEP_S):
                self.engine.handle(env, self.now())
            t += STEP_S
            await asyncio.sleep(max(0.0, start + t - self.now()))

    async def start(self) -> None:
        self._tasks.append(asyncio.create_task(self.tick_loop(), name="tick"))
        if self.mock:
            self._tasks.append(asyncio.create_task(self.mock_loop(), name="mock"))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()


def create_app(config: Config | None = None, feeds: list[Feed] | None = None, mock: bool = False,
               mock_seed: int = 7, run_loops: bool = True) -> FastAPI:
    config = config or load_config()
    feeds = default_feeds(config) if feeds is None else feeds
    service = EngineService(config, feeds, mock=mock, mock_seed=mock_seed)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if run_loops:
            await service.start()
        loop = asyncio.get_running_loop()
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(signal.SIGHUP, lambda: service.engine.reload())
        yield
        await service.stop()

    # Endpoints that read or change engine state are `async def`: they run on the event loop with
    # the ticks, never in a worker thread racing them.
    app = FastAPI(title="Sports Intensity engine", lifespan=lifespan)
    app.state.service = service

    @app.websocket("/ingest")
    async def ingest(ws: WebSocket) -> None:
        await ws.accept()
        service.metrics.inc("ingest.connections")
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    env = parse_envelope(raw)
                except ValidationError as exc:
                    service.metrics.inc("ingest.invalid")
                    log_event(log, "invalid envelope", logging.WARNING, error=exc.errors()[:3])
                    continue
                service.engine.handle(env, service.now())
        except WebSocketDisconnect:
            service.metrics.inc("ingest.disconnects")

    @app.websocket("/ws/scores")
    async def scores(ws: WebSocket) -> None:
        await ws.accept()
        client = ScoreClient(ws)
        for feed_id, payload in service.broadcaster.latest.items():
            client.offer(feed_id, payload)
        service.broadcaster.clients.add(client)
        sender = asyncio.create_task(client.run())
        try:
            while True:  # we only read to notice the disconnect
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            service.broadcaster.clients.discard(client)
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await sender

    @app.get("/")
    def root() -> RedirectResponse:
        return RedirectResponse("/debug")

    @app.get("/feeds")
    async def list_feeds() -> dict[str, Any]:
        now = service.now()
        live = service.engine.feeds
        registry = [f.model_dump() for f in service.feeds]
        known = {f["feed_id"] for f in registry}
        extra = [{"feed_id": fid, "sport": fs.sport, "registered": False} for fid, fs in live.items() if fid not in known]
        for f in registry:
            fs = live.get(f["feed_id"])
            f["live"] = bool(fs and fs.seen and not fs.stale(now, config.engine))
        return {"feeds": registry + extra}

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return service.engine.health(service.now())

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        snap = service.metrics.snapshot()
        snap["ws_clients"] = len(service.broadcaster.clients)
        return snap

    @app.post("/admin/reload")
    async def reload() -> dict[str, Any]:
        return service.engine.reload()

    @app.post("/admin/switch")
    async def switch(key: str = Body(...), value: Any = Body(...)) -> dict[str, Any]:
        try:
            service.engine.set_switch(key, value)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return service.engine.health(service.now())["switches"]

    @app.get("/debug", response_class=HTMLResponse)
    def debug() -> str:
        return DEBUG_HTML.read_text()

    return app


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    ap = argparse.ArgumentParser(description="Sports Intensity engine service")
    ap.add_argument("--overlay", action="append", default=[], help="config overlay (local, demo, or a path)")
    ap.add_argument("--set", dest="overrides", action="append", default=[], help="override, dotted.key=value")
    ap.add_argument("--mock", action="store_true", help="drive the engine from the in-process mock generator")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    args = ap.parse_args(argv)

    config = load_config(args.overlay, args.overrides)
    setup_logging(json_format=bool(config.log.get("json", True)))
    run_dir = resolve_path("run/engine") / time.strftime("%Y%m%d-%H%M%S")
    save_resolved(config, run_dir)
    app = create_app(config, mock=args.mock, mock_seed=args.seed)
    svc = config.engine.service
    uvicorn.run(app, host=args.host or svc.host, port=args.port or svc.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
