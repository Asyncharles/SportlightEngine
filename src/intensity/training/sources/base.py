"""Resumable raw-data fetchers (T0), the download half of pipeline step 1.

Each fetcher lists its items once (cached in `data/raw/<sport>/_index.json`) and downloads each item
to `data/raw/<sport>/<item>.json`. Files are written atomically, so an interrupted run leaves no
half-written item, and a rerun skips every item already on disk. An item that still fails after the
retries is recorded in `_failed.json` and the run moves on; the next run tries it again.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import httpx

from intensity.telemetry import log_event

log = logging.getLogger(__name__)

INDEX = "_index.json"
FAILED = "_failed.json"
_RETRY_STATUS = {429, 500, 502, 503, 504}


class FetchError(RuntimeError):
    """A request that failed for good (after the retries, or with a non-retryable status)."""


class Retry(Exception):
    """Raised inside a `with_retries` call for a failure worth retrying."""

    def __init__(self, reason: str, retry_after: float | None = None):
        super().__init__(reason)
        self.reason = reason
        self.retry_after = retry_after


@dataclass
class FetchStats:
    fetched: int = 0
    skipped: int = 0
    failed: list[str] = field(default_factory=list)


def write_json(path: Path, data: Any) -> None:
    """Write via a temporary file and rename, so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")))
    os.replace(tmp, path)


def safe_name(item: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", item)


class Fetcher(ABC):
    """Subclasses list items and fetch one item as JSON-serialisable data."""

    sport: ClassVar[str]

    def __init__(
        self,
        cfg: dict[str, Any],
        fetch_cfg: dict[str, Any],
        raw_dir: Path,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.cfg = cfg
        self.raw_dir = Path(raw_dir)
        self.timeout_s = float(fetch_cfg.get("timeout_s", 30.0))
        self.retries = int(fetch_cfg.get("retries", 5))
        self.backoff_min, self.backoff_max = (float(x) for x in fetch_cfg.get("backoff_s", (2.0, 120.0)))
        self.min_interval_s = float(cfg.get("min_interval_s", 0.0))
        self.client = client or httpx.Client(timeout=self.timeout_s, follow_redirects=True)
        self.sleep = sleep
        self._last_request = 0.0

    # ------------------------------------------------------------------ subclass API

    @abstractmethod
    def list_items(self) -> tuple[list[str], dict[str, Any]]:
        """Every item id to fetch, plus metadata kept in the index (teams, dates, ...)."""

    @abstractmethod
    def fetch_item(self, item: str) -> Any:
        """One item's raw data."""

    # ------------------------------------------------------------------ requests

    def throttle(self) -> None:
        wait = self._last_request + self.min_interval_s - time.monotonic()
        if wait > 0:
            self.sleep(wait)
        self._last_request = time.monotonic()

    def with_retries(self, what: str, call: Callable[[], Any]) -> Any:
        """Run `call` with throttling and exponential backoff. `call` raises `Retry` to retry."""
        backoff = self.backoff_min
        for attempt in range(self.retries + 1):
            self.throttle()
            try:
                return call()
            except Retry as exc:
                if attempt == self.retries:
                    raise FetchError(f"{what}: {exc.reason}") from None
                wait = max(backoff, exc.retry_after or 0.0)
                log_event(
                    log, "retrying", logging.WARNING, what=what, reason=exc.reason, wait_s=round(wait, 1)
                )
                self.sleep(min(wait, self.backoff_max))
                backoff = min(backoff * 2, self.backoff_max)
        raise AssertionError("unreachable")

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        def call() -> Any:
            try:
                resp = self.client.get(url, params=params, timeout=self.timeout_s)
            except httpx.TransportError as exc:
                raise Retry(f"{type(exc).__name__}: {exc}") from None
            if resp.status_code in _RETRY_STATUS:
                raise Retry(f"HTTP {resp.status_code}", _retry_after(resp))
            if resp.status_code >= 400:
                raise FetchError(f"GET {resp.url}: HTTP {resp.status_code}")
            return resp.json()

        return self.with_retries(url, call)

    # ------------------------------------------------------------------ the run

    def item_path(self, item: str) -> Path:
        return self.raw_dir / f"{safe_name(item)}.json"

    def index(self, refresh: bool = False) -> dict[str, Any]:
        path = self.raw_dir / INDEX
        if path.exists() and not refresh:
            return json.loads(path.read_text())
        items, meta = self.list_items()
        index = {"sport": self.sport, "items": items, "meta": meta}
        write_json(path, index)
        log_event(log, "index written", sport=self.sport, items=len(items))
        return index

    def run(
        self, limit: int | None = None, items: list[str] | None = None, refresh_index: bool = False
    ) -> FetchStats:
        todo = items if items is not None else self.index(refresh_index)["items"]
        stats = FetchStats()
        failed = self._read_failed()
        for item in todo:
            if limit is not None and stats.fetched >= limit:
                break
            path = self.item_path(item)
            if path.exists():
                stats.skipped += 1
                continue
            started = time.monotonic()
            try:
                data = self.fetch_item(item)
            except FetchError as exc:
                stats.failed.append(item)
                failed[item] = str(exc)
                self._write_failed(failed)
                log_event(log, "item failed", logging.ERROR, sport=self.sport, item=item, error=str(exc))
                continue
            write_json(path, data)
            failed.pop(item, None)
            stats.fetched += 1
            log_event(
                log,
                "item fetched",
                sport=self.sport,
                item=item,
                done=stats.fetched + stats.skipped,
                total=len(todo),
                seconds=round(time.monotonic() - started, 1),
            )
        self._write_failed(failed)
        return stats

    def _read_failed(self) -> dict[str, str]:
        path = self.raw_dir / FAILED
        return json.loads(path.read_text()) if path.exists() else {}

    def _write_failed(self, failed: dict[str, str]) -> None:
        path = self.raw_dir / FAILED
        if failed:
            write_json(path, failed)
        elif path.exists():
            path.unlink()


def _retry_after(resp: httpx.Response) -> float | None:
    try:
        return float(resp.headers["retry-after"])
    except (KeyError, ValueError):
        return None
