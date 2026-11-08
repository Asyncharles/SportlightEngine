"""End to end with the real front-end backend: engine -> RabbitMQ -> sportlight backend.

Runs in CI (`.github/workflows/e2e.yml`): RabbitMQ in a service container, the front-end backend built
from Asyncharles/sportlight, and here the engine and the clock lane as real processes on synthetic
clips named like the four demo samples. No footage, no GCP. Skipped unless E2E_BACKEND_URL and AMQP_URL
are set.

Checks: every demo feed appears in the backend with live intensity from `sportlight-engine`; scores keep
moving; nothing lands in the dead-letter queue (the backend accepted every message); restarting the
engine gives the backend a fresh run within 10 s; the backend's queue doesn't grow.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from intensity.config import REPO_ROOT, resolve_path
from intensity.feeds import load_feeds
from intensity.training.sources import weightlifting
from intensity.training.sources.convert import write_jsonl
from tests.unit.ingest.test_clock_lane import synthetic_mp4

BACKEND = os.environ.get("E2E_BACKEND_URL", "")
AMQP_URL = os.environ.get("AMQP_URL", "")
MGMT = os.environ.get("RABBITMQ_MGMT_URL", "")  # http://user:pass@localhost:15672
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not (BACKEND and AMQP_URL), reason="needs E2E_BACKEND_URL and AMQP_URL (CI e2e job)"),
]
FEEDS = load_feeds(resolve_path("config/feeds.yaml")).feeds
FEED_IDS = {f.feed_id for f in FEEDS}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(check, timeout: float, what: str):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            last = check()
            if last:
                return last
        except (httpx.HTTPError, ValueError, KeyError):
            pass
        time.sleep(0.5)
    raise AssertionError(f"timed out after {timeout:.0f}s waiting for {what}; last: {last}")


def backend_feeds() -> dict[str, dict]:
    feeds = httpx.get(f"{BACKEND}/api/feeds", timeout=5).json()["feeds"]
    return {f["id"]: f for f in feeds if f["id"] in FEED_IDS}


def live(feed: dict) -> bool:
    return feed["provider"] == "sportlight-engine" and feed["intensity"] is not None and not feed["intensityStale"]


def queue(name: str) -> dict:
    return httpx.get(f"{MGMT}/api/queues/sportlight/{name}", timeout=5).json()


@pytest.fixture
def stack(tmp_path: Path):
    data = tmp_path / "data"
    for f in FEEDS:  # synthetic clips under the demo samples' names: the browser and engine agree on the file
        synthetic_mp4(data / f.sample, seconds=120.0, fps=2)
    session = REPO_ROOT / "data" / "weightlifting" / "wl_worlds2025_m79.csv"
    write_jsonl(data / "events" / "weightlifting" / "wl_worlds2025_m79.jsonl", weightlifting.to_records(session))
    state = tmp_path / "run" / "feeds_state.json"
    engine_port, health_port = free_port(), free_port()
    env = {**os.environ, "INTENSITY_DATA_DIR": str(data), "AMQP_URL": AMQP_URL, "PYTHONUNBUFFERED": "1"}
    logs = tmp_path / "logs"
    logs.mkdir()

    def start_engine() -> subprocess.Popen:
        return subprocess.Popen(
            [sys.executable, "-m", "intensity.engine.service", "--port", str(engine_port),
             "--set", "engine.forecast.source=heuristic", "--set", f"engine.publish.feeds_state_path={state}"],
            env=env, stdout=open(logs / "engine.log", "a"), stderr=subprocess.STDOUT)  # noqa: SIM115 (the child owns it)

    ingest_env = {k: v for k, v in env.items() if k != "AMQP_URL"}
    ingest = subprocess.Popen(
        [sys.executable, "-m", "intensity.ingest.run", "--engine-url", f"ws://127.0.0.1:{engine_port}/ingest",
         "--set", f"ingest.feeds_state={state}", "--set", f"ingest.health.port={health_port}",
         "--set", "ingest.health.host=127.0.0.1"],
        env=ingest_env, stdout=open(logs / "ingest.log", "a"), stderr=subprocess.STDOUT)  # noqa: SIM115
    procs = {"engine": start_engine(), "ingest": ingest}
    try:
        yield procs, start_engine, engine_port
    finally:
        for p in procs.values():
            p.terminate()
        for p in procs.values():
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        for name in ("engine.log", "ingest.log"):
            print(f"----- {name} (tail)\n" + "".join((logs / name).read_text().splitlines(True)[-40:]))


def test_engine_scores_reach_the_front_end_backend(stack) -> None:
    procs, start_engine, engine_port = stack
    wait_for(lambda: httpx.get(f"{BACKEND}/readyz", timeout=5).status_code == 200, 30, "backend /readyz")
    wait_for(lambda: httpx.get(f"http://127.0.0.1:{engine_port}/health", timeout=5).json().get("amqp", {}).get("connected"),
             60, "the engine's AMQP link")
    feeds = wait_for(lambda: (f := backend_feeds()) and set(f) == FEED_IDS and all(live(x) for x in f.values()) and f,
                     60, "all four demo feeds live in the backend")
    assert {f["sportId"] for f in feeds.values()} == {"football", "motorsport", "basketball", "weightlifting"}
    assert all(f["source"]["url"].startswith("/media/local/videos/") for f in feeds.values())

    # Scores keep flowing at about 1 Hz.
    first = {k: v["intensity"]["streamTime"] for k, v in feeds.items()}
    time.sleep(4)
    later = backend_feeds()
    assert all(later[k]["intensity"]["streamTime"] >= first[k] + 2 for k in FEED_IDS), (first, later)

    # Restart the engine: the backend must have a fresh, live run within 10 s.
    old_runs = {k: v["runId"] for k, v in later.items()}
    procs["engine"].terminate()
    procs["engine"].wait(timeout=10)
    t0 = time.monotonic()
    procs["engine"] = start_engine()
    wait_for(lambda: (f := backend_feeds()) and all(f[k]["runId"] != old_runs[k] and live(f[k]) for k in FEED_IDS),
             10, "a fresh live run for every feed after the engine restart")
    print(f"engine restart -> fresh runs in the backend after {time.monotonic() - t0:.1f}s")

    if MGMT:
        assert queue("sportlight.feeds.dead")["messages"] == 0, "the backend rejected (dead-lettered) a message"
        depth = queue("sportlight.backend")["messages"]
        time.sleep(5)
        assert queue("sportlight.backend")["messages"] <= max(depth, 10), "the backend's queue is growing"
