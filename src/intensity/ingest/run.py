"""The ingest process: `python -m intensity.ingest.run`.

Only the clock lane exists (`--mode clock_only`, see `clock_lane.py`): ClockTicks and loop Controls for
every feed in `config/feeds.yaml`, sent to the engine's `/ingest`. Video, audio and perception (plan I3
to I6, P1 to P3) are not built, so the engine scores each feed from its replayed match data in clip time.

`GET /health` on `ingest.health.port` (8001): 200 with `status: ok` once the engine link is up and
at least one feed has a clock; 503 otherwise, so a deploy rolls back when the ingest can't do its job.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import shutil
import subprocess
import time
from typing import Any

from intensity.config import load_config, resolve_path, save_resolved
from intensity.feeds import load_feeds
from intensity.ingest.clock_lane import ClockLane, build_clocks
from intensity.telemetry import setup_logging
from intensity.transport import IngestClient

log = logging.getLogger(__name__)


def gpu_name() -> str | None:
    """The GPU this container sees, if any (the clock lane itself needs none)."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return out.strip().splitlines()[0] if out.strip() else None


class ClockIngest:
    def __init__(self, cfg: dict[str, Any], url: str, feeds_path: str):
        self.cfg = cfg
        self.tick_hz = float(cfg["clock"]["tick_hz"])
        self.client = IngestClient(url, maxsize=int(cfg["clock"].get("send_queue", 64)))
        feeds = load_feeds(resolve_path(feeds_path)).feeds
        self.clocks, self.problems = build_clocks(feeds, resolve_path(cfg["feeds_state"]), time.time())
        for feed_id, problem in self.problems.items():
            log.error("feed %s has no clock: %s", feed_id, problem)
        self.lane = ClockLane(self.clocks)
        self.ticks = 0
        self.last_tick = 0.0
        self.gpu = gpu_name()

    def health(self) -> tuple[int, dict[str, Any]]:
        now = time.time()
        ok = self.client.connected and bool(self.clocks) and now - self.last_tick < 5 / self.tick_hz + 1
        body = {
            "status": "ok" if ok else "degraded" if self.clocks else "error",
            "mode": "clock_only",
            "engine_connected": self.client.connected,
            "sent": self.client.sent,
            "dropped": self.client.queue.dropped,
            "gpu": self.gpu,
            "feeds": {
                c.feed_id: dict(zip(("t_video", "loop_epoch"), c.position(now), strict=True), duration_s=c.duration_s)
                for c in self.clocks
            },
            "problems": self.problems,
        }
        return (200 if ok else 503), body

    async def tick_loop(self) -> None:
        period = 1.0 / self.tick_hz
        next_at = time.monotonic()
        while True:
            for env in self.lane.tick(time.time()):
                self.client.put(env)
            self.ticks += 1
            self.last_tick = time.time()
            next_at += period
            await asyncio.sleep(max(0.0, next_at - time.monotonic()))


def make_app(ingest: ClockIngest):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI(title="intensity ingest (clock lane)")

    @app.get("/health")
    def health() -> JSONResponse:
        code, body = ingest.health()
        return JSONResponse(body, status_code=code)

    return app


async def serve(ingest: ClockIngest, host: str, port: int) -> None:
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(make_app(ingest), host=host, port=port, log_level="warning"))
    await asyncio.gather(ingest.client.run(), ingest.tick_loop(), server.serve())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overlay", action="append", default=[], help="config overlay (repeatable)")
    ap.add_argument("--set", action="append", default=[], help="dotted.key=value override (repeatable)")
    ap.add_argument("--mode", choices=["clock_only", "full"], default=None, help="default: ingest.mode")
    ap.add_argument("--engine-url", default=None, help="default: ingest.engine_url")
    args = ap.parse_args(argv)
    config = load_config(overlays=args.overlay, overrides=args.set)
    cfg = config.ingest
    setup_logging(json_format=bool(config.log.get("json", True)))
    mode = args.mode or cfg.get("mode", "clock_only")
    if mode != "clock_only":
        log.error("ingest mode %r is not built yet (plan I3 to I6, P1 to P3); use clock_only", mode)
        return 2
    save_resolved(config, resolve_path("run/ingest") / time.strftime("%Y%m%d-%H%M%S"))
    feeds_path = config.training.get("paths", {}).get("feeds", "config/feeds.yaml")
    ingest = ClockIngest(cfg, args.engine_url or cfg["engine_url"], feeds_path)
    if not ingest.clocks:
        log.error("no feed has a clock; check data/samples (make data-pull FOLDERS=samples)")
    health = cfg.get("health", {})
    log.info("clock lane: %d feed(s) at %.0f Hz -> %s; health on :%s", len(ingest.clocks), ingest.tick_hz,
             ingest.client.url, health.get("port", 8001))
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve(ingest, str(health.get("host", "0.0.0.0")), int(health.get("port", 8001))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
