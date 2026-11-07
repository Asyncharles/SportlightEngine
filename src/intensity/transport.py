"""WebSocket client from the ingest (or the mock) to the engine's `/ingest`.

Reconnects with backoff and keeps a bounded send queue. Under pressure it drops the oldest
`ClockTick`s first, then the oldest non-match events, and never drops `match` events or `Control`
messages: those are rare and losing one would desync the engine.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque

import websockets
from websockets.asyncio.client import connect

from intensity.schema import Envelope, Event

log = logging.getLogger(__name__)


def _priority(env: Envelope) -> int:
    """0 = drop first, 2 = never drop."""
    if env.kind == "clock":
        return 0
    if env.kind == "event" and isinstance(env.data, Event) and env.data.source != "match":
        return 1
    return 2


class SendQueue:
    """Thread-safe bounded queue with the drop policy above. Counts what it drops."""

    def __init__(self, maxsize: int = 1024):
        self.maxsize = maxsize
        self._items: deque[Envelope] = deque()
        self._lock = threading.Lock()
        self.dropped = {"clock": 0, "event": 0}

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def put(self, env: Envelope) -> None:
        with self._lock:
            self._items.append(env)
            while len(self._items) > self.maxsize and self._drop_one():
                pass

    def _drop_one(self) -> bool:
        for level in (0, 1):
            for i, item in enumerate(self._items):
                if _priority(item) == level:
                    del self._items[i]
                    self.dropped["clock" if level == 0 else "event"] += 1
                    return True
        return False  # only never-drop items left: let the queue grow

    def get_nowait(self) -> Envelope | None:
        with self._lock:
            return self._items.popleft() if self._items else None

    def push_front(self, env: Envelope) -> None:
        """Requeue an envelope whose send failed, so a reconnect resends it first."""
        with self._lock:
            self._items.appendleft(env)


class IngestClient:
    """Async sender. `put()` is safe from any thread; `run()` sends until `stop()`."""

    def __init__(self, url: str, maxsize: int = 1024, backoff_s: tuple[float, float] = (0.2, 5.0)):
        self.url = url
        self.queue = SendQueue(maxsize)
        self.backoff_min, self.backoff_max = backoff_s
        self.connected = False
        self.sent = 0
        self.reconnects = 0
        self._wake: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopping = False

    def put(self, env: Envelope) -> None:
        self.queue.put(env)
        if self._loop is not None and self._wake is not None:
            self._loop.call_soon_threadsafe(self._wake.set)

    def stop(self) -> None:
        self._stopping = True
        if self._loop is not None and self._wake is not None:
            self._loop.call_soon_threadsafe(self._wake.set)

    async def drain(self, timeout_s: float = 5.0) -> bool:
        """Wait until everything queued has been sent."""
        deadline = asyncio.get_running_loop().time() + timeout_s
        while len(self.queue) and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)
        return not len(self.queue)

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._wake = asyncio.Event()
        backoff = self.backoff_min
        while not self._stopping:
            try:
                async with connect(self.url, open_timeout=5, max_queue=None) as ws:
                    self.connected = True
                    backoff = self.backoff_min
                    log.info("connected to %s", self.url)
                    await self._send_loop(ws)
            except (OSError, websockets.exceptions.WebSocketException, TimeoutError) as exc:
                if self._stopping:
                    break
                log.warning("ingest link down (%s); retrying in %.1fs", exc, backoff)
            finally:
                if self.connected:
                    self.reconnects += 1
                self.connected = False
            if self._stopping:
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self.backoff_max)

    async def _send_loop(self, ws: websockets.asyncio.client.ClientConnection) -> None:
        assert self._wake is not None
        while not self._stopping:
            env = self.queue.get_nowait()
            if env is None:
                self._wake.clear()
                await self._wake.wait()
                continue
            try:
                await ws.send(env.model_dump_json())
            except Exception:
                self.queue.push_front(env)
                raise
            self.sent += 1

    def start_in_thread(self) -> threading.Thread:
        """For threaded producers (the ingest): run the sender on its own event loop thread."""
        thread = threading.Thread(target=lambda: asyncio.run(self.run()), name="ingest-client", daemon=True)
        thread.start()
        return thread
