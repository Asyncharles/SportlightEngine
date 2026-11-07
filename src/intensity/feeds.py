"""`config/feeds.yaml`, the one feed registry that MediaMTX paths, pushers, ingest and engine all read."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intensity.config import read_yaml, resolve_path
from intensity.schema import Sport


class MatchRef(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: str  # statsbomb | fastf1 | nba_api | session_csv
    match_id: str | int | None = None  # filled by the ML lead


class Feed(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feed_id: str
    sport: Sport
    sample: str  # path inside the bucket and data/samples/
    rtsp_path: str
    match_ref: MatchRef
    sync: str | None = None  # data/sync/<clip_id>.csv
    asr: Literal["live", "precomputed", "off"] = "live"
    ocr_regions: dict[str, list[int]] | None = None  # tier 2: {clock: [x, y, w, h]} at 1080p
    demo: bool = True  # demo matches never train (hard rule 6)

    @property
    def clip_id(self) -> str:
        return Path(self.sample).stem


class FeedRegistry(BaseModel):
    feeds: list[Feed] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique(self) -> FeedRegistry:
        ids = [f.feed_id for f in self.feeds]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate feed_id in feeds.yaml")
        return self

    def get(self, feed_id: str) -> Feed:
        for feed in self.feeds:
            if feed.feed_id == feed_id:
                return feed
        raise KeyError(feed_id)

    def by_sport(self, sport: str) -> list[Feed]:
        return [f for f in self.feeds if f.sport == sport]

    def demo_match_ids(self, sport: str) -> set[str]:
        """Match ids that must stay out of training and validation."""
        return {
            str(f.match_ref.match_id)
            for f in self.feeds
            if f.sport == sport and f.demo and f.match_ref.match_id is not None
        }


def load_feeds(path: str | Path = "config/feeds.yaml") -> FeedRegistry:
    return FeedRegistry.model_validate(read_yaml(resolve_path(path)))
