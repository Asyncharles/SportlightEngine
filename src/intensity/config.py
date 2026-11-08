"""Config loader: `config/default/*.yaml`, then overlays (`local`, `demo`, ...), then CLI overrides.

Every model-facing setting and tunable comes from here (hard rule 4), and every run saves the
config it used with `save_resolved`. Each agent owns one `config/default/<area>.yaml`; the loader
deep-merges them all, so agents rarely touch the same file.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from intensity.data import is_data_path, resolve_data

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(os.environ.get("INTENSITY_CONFIG_DIR", REPO_ROOT / "config"))


# --------------------------------------------------------------------------- engine settings


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ServiceConfig(_Strict):
    host: str = "0.0.0.0"
    port: int = 8000
    score_interval_s: float = 1.0  # one ScoreUpdate per feed per second
    fast_tick_s: float = 0.1  # imminent check cadence for out-of-cycle updates
    ingest_queue: int = 4096


class TrendConfig(_Strict):
    window_s: float = 10.0
    rising_slope: float = 0.01
    falling_slope: float = -0.01


class PercentileConfig(_Strict):
    window_s: float = 1800.0  # intensity_pct within sport over a rolling 30 min
    min_samples: int = 30  # below this, intensity_pct uses the seed reference too
    ref_path: str = "models/intensity_ref.json"
    # Intensity is ranked rounded to this, so a value creeping up by noise ties with its past instead of
    # being a new high (a single feed's percentile otherwise sticks at 100 while its EMA converges). 0 = off.
    tie_tolerance: float = 0.01
    # On a feed's first clock tick, replay its match data over up to this much earlier clip time into the
    # sport's window, so intensity_pct has history from the first second after any (re)start. 0 = off.
    preroll_s: float = 1800.0


class ForecastConfig(_Strict):
    source: Literal["model", "heuristic"] = "model"
    models_dir: str = "models"
    shadow: dict[str, str] = Field(default_factory=dict)  # sport -> version scored and logged, not served
    heuristic_threshold: float = 0.5
    heuristic_quantiles: list[float] = Field(
        default_factory=lambda: [0.02, 0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.32, 0.4, 0.5, 0.7]
    )


class ImminentConfig(_Strict):
    enabled: bool = True
    context_floor: float = 0.3
    hold_s: float = 3.0
    min_on_s: float = 1.0
    config_dir: str = "config/imminent"


class RollingConfig(_Strict):
    window_s: float = 300.0  # per-feed rolling percentile window
    min_samples: int = 20  # below this the seed distribution is mixed in
    max_samples: int = 1200


class CrowdTrackerConfig(_Strict):
    events_weight: float = 0.8  # PANNs cheering/applause raise the part up to this share
    events_classes: list[str] = Field(default_factory=lambda: ["cheering", "applause", "crowd"])
    drop_after_s: float = 3.0
    rise_fact_min_db_s: float = 0.5  # smaller rises are not worth a reason


class CommentaryTrackerConfig(_Strict):
    lexicon_dir: str = "config/lexicon"
    decay_s: float = 10.0
    name_weight: float = 0.15
    rate_ref_wps: float = 3.0  # words/s considered fast commentary
    rate_weight: float = 0.2
    max_quote_words: int = 6
    min_no_speech_prob: float = 0.6  # segments above this are ignored
    drop_after_s: float = 60.0  # no ASR segment for this long: the part drops out


class VoiceTrackerConfig(_Strict):
    drop_after_s: float = 3.0


class VideoTrackerConfig(_Strict):
    drop_after_s: float = 3.0
    weights: dict[str, float] = Field(
        default_factory=lambda: {"motion": 0.4, "celebration": 0.35, "close_up": 0.25, "cuts": 0.0}
    )
    # Per sport: which prompt names count as celebration / close-up (from config/prompts/<sport>.yaml).
    celebration_prompts: dict[str, list[str]] = Field(default_factory=dict)
    close_up_shots: list[str] = Field(default_factory=lambda: ["close_player", "close_crowd"])


def _default_seeds() -> dict[str, list[float]]:
    # Deciles (0th to 100th percentile) per signal, used until models/video_v0/calibration.json
    # exists and the feed's own rolling window fills. Placeholders, replaced by `make video-calibrate`.
    return {
        "loudness_rel_db": [-12, -6, -4, -2.5, -1, 0, 1, 2.5, 4, 7, 15],
        "rise_db_s": [-6, -2, -1, -0.5, -0.2, 0, 0.2, 0.6, 1.2, 2.5, 8],
        "energy_rel_db": [-10, -5, -3, -2, -1, 0, 1, 2, 3.5, 6, 12],
        "pitch_rel_st": [-6, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3.5, 8],
        "pace_rel": [0.4, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 2.0],
        "motion": [0, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.065, 0.08, 0.11, 0.25],
        "cuts_per_10s": [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 15],
        "pan_speed": [0, 0.02, 0.04, 0.06, 0.09, 0.12, 0.16, 0.22, 0.3, 0.45, 1.2],
        "prompt": [0, 0.05, 0.08, 0.1, 0.13, 0.16, 0.19, 0.23, 0.28, 0.35, 0.6],
    }


class TrackersConfig(_Strict):
    rolling: RollingConfig = RollingConfig()
    crowd: CrowdTrackerConfig = CrowdTrackerConfig()
    commentary: CommentaryTrackerConfig = CommentaryTrackerConfig()
    voice: VoiceTrackerConfig = VoiceTrackerConfig()
    video: VideoTrackerConfig = VideoTrackerConfig()
    calibration_path: str = "models/video_v0/calibration.json"
    default_seeds: dict[str, list[float]] = Field(default_factory=_default_seeds)


class PolishConfig(_Strict):
    enabled: bool = False
    model: str = "claude-opus-5-5"
    timeout_s: float = 1.5
    max_tokens: int = 1024
    min_interval_s: float = 5.0  # per feed


class ReplayConfig(_Strict):
    events_dir: str = "data/events"


class PublishConfig(_Strict):
    """ScoreUpdates to the front end's FeedEvents on RabbitMQ (engine/amqp.py). Active only when the
    AMQP_URL environment variable is set, so laptops and tests never need a broker."""

    enabled: bool = True
    exchange: str = "sportlight.feeds"  # topic, durable: declared identically by the front-end backend
    provider: str = "sportlight-engine"
    catalog_path: str = "config/publish.yaml"  # front-end titles, sports and media URLs per feed
    feeds_state_path: str = "run/feeds_state.json"  # each feed's clip clock, written by the ingest
    reannounce_s: float = 10.0  # feed.started again this often (the backend forgets silent feeds)
    unavailable_after_s: float = 5.0  # stale this long -> feed.updated {availability: unavailable}
    lifecycle_buffer: int = 1000  # started/updated events kept while the broker is down
    confirm_timeout_s: float = 5.0
    backoff_s: tuple[float, float] = (0.5, 30.0)


class EngineConfig(_Strict):
    clock: Literal["live", "simulated"] = "live"
    speed: float = 1.0  # simulated clock only
    stale_after_s: float = 2.0
    max_extrapolate_s: float = 0.5  # clock runs on past the last tick by at most this (ticks are 4 Hz)
    ema_alpha: float = 0.3
    weights_path: str = "config/weights.yaml"
    feeds_path: str = "config/feeds.yaml"
    service: ServiceConfig = ServiceConfig()
    trend: TrendConfig = TrendConfig()
    percentile: PercentileConfig = PercentileConfig()
    forecast: ForecastConfig = ForecastConfig()
    imminent: ImminentConfig = ImminentConfig()
    trackers: TrackersConfig = TrackersConfig()
    polish: PolishConfig = PolishConfig()
    replay: ReplayConfig = ReplayConfig()
    publish: PublishConfig = PublishConfig()
    max_reasons: int = 3


class Config(BaseModel):
    """The merged config. Areas other than `engine` are owned by other agents and stay free-form
    until their owners add a typed section here."""

    model_config = ConfigDict(extra="allow")

    engine: EngineConfig = EngineConfig()
    ingest: dict[str, Any] = Field(default_factory=dict)
    perception: dict[str, Any] = Field(default_factory=dict)
    training: dict[str, Any] = Field(default_factory=dict)
    clip: dict[str, Any] = Field(default_factory=dict)
    log: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- loading


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def read_yaml(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def parse_override(item: str) -> dict[str, Any]:
    """`engine.imminent.hold_s=2.5` -> {"engine": {"imminent": {"hold_s": 2.5}}}; values parse as YAML."""
    if "=" not in item:
        raise ValueError(f"override {item!r} must look like dotted.key=value")
    key, raw = item.split("=", 1)
    value: Any = yaml.safe_load(raw) if raw != "" else ""
    for part in reversed(key.strip().split(".")):
        value = {part: value}
    return value


def load_raw(
    overlays: list[str] | tuple[str, ...] = (),
    overrides: list[str] | tuple[str, ...] = (),
    config_dir: Path | None = None,
) -> dict[str, Any]:
    root = Path(config_dir) if config_dir else CONFIG_DIR
    merged: dict[str, Any] = {}
    for path in sorted((root / "default").glob("*.yaml")):
        merged = deep_merge(merged, read_yaml(path))
    for overlay in overlays:
        path = Path(overlay)
        if not path.suffix:
            path = root / f"{overlay}.yaml"
        merged = deep_merge(merged, read_yaml(path))
    for item in overrides:
        merged = deep_merge(merged, parse_override(item))
    return merged


def load_config(
    overlays: list[str] | tuple[str, ...] = (),
    overrides: list[str] | tuple[str, ...] = (),
    config_dir: Path | None = None,
) -> Config:
    """Load and validate. Overlays come from `INTENSITY_OVERLAYS` (comma-separated) when none are given."""
    if not overlays and os.environ.get("INTENSITY_OVERLAYS"):
        overlays = [o for o in os.environ["INTENSITY_OVERLAYS"].split(",") if o]
    return Config.model_validate(load_raw(overlays, overrides, config_dir))


def save_resolved(config: Config, run_dir: Path) -> Path:
    """Write the config a run used next to its results (hard rule 4)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config.resolved.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True))
    return path


def resolve_path(path: str | Path) -> Path:
    """Config paths are relative to the repo root unless absolute; `data/...` goes through the data
    layer (`intensity.data`), so it follows `INTENSITY_DATA_DIR` on the VM."""
    p = Path(path)
    if p.is_absolute():
        return p
    if is_data_path(p):
        return resolve_data(p)
    return REPO_ROOT / p


# --------------------------------------------------------------------------- per-sport side files


def load_weights(path: str | Path) -> dict[str, dict[str, float]]:
    """`config/weights.yaml`: part weights per sport, with an optional `default` block."""
    data = read_yaml(resolve_path(path))
    default = data.get("default", {})
    out = {}
    for sport, weights in data.items():
        if sport == "default":
            continue
        out[sport] = {**default, **(weights or {})}
    out["default"] = dict(default)
    return out


def load_sport_yaml(directory: str | Path, sport: str) -> dict[str, Any]:
    path = resolve_path(directory) / f"{sport}.yaml"
    return read_yaml(path) if path.exists() else {}
