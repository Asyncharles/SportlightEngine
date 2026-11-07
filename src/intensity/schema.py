"""Contracts v1: every message that crosses a process boundary.

Changing anything here needs the ML lead's approval and a CHANGELOG.md entry (hard rule 9).
`python -m intensity.schema export` rewrites contracts/*.json; `check` fails on any undeclared change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

SCHEMA_VERSION = 1

Sport = Literal["football", "f1", "nba", "weightlifting", "chess"]
Source = Literal["match", "crowd", "commentary", "video", "voice", "ocr"]
PART_NAMES = ("closeness", "momentum", "stakes", "crowd", "commentary", "video", "forecast", "imminent")
SPORTS: tuple[str, ...] = ("football", "f1", "nba", "weightlifting", "chess")
SHOT_TYPES = ("wide", "close_player", "close_crowd", "replay", "other")
PHASES = ("idle", "walk_on", "setup", "setup_hush", "lift")


# --------------------------------------------------------------------------- messages into the engine


class Fact(BaseModel):
    key: str  # "score_diff", "cuts_per_10s"
    value: float | int | str  # the exact value a reason may quote
    unit: str | None = None
    text: str  # "France trail by one"


class Event(BaseModel):
    v: Literal[1] = 1
    feed_id: str
    sport: Sport
    source: Source
    t_video: float  # seconds into the source mp4
    loop_epoch: int
    t_emit_ns: int  # sender wall clock; latency metrics only
    payload: dict[str, Any]  # validated per source, below
    facts: list[Fact] = []

    @model_validator(mode="after")
    def _check_payload(self) -> Event:
        PAYLOAD_MODELS[self.source].model_validate(self.payload)
        return self


class ClockTick(BaseModel):  # ingest to engine, 4 Hz per feed
    feed_id: str
    t_video: float
    loop_epoch: int
    t_emit_ns: int


class Control(BaseModel):  # ingest to engine
    feed_id: str
    kind: Literal["loop", "feed_down", "feed_up"]
    loop_epoch: int
    t_emit_ns: int


class Envelope(BaseModel):
    kind: Literal["event", "clock", "control"]
    data: Event | ClockTick | Control

    @model_validator(mode="after")
    def _check_kind(self) -> Envelope:
        expected = {"event": Event, "clock": ClockTick, "control": Control}[self.kind]
        if not isinstance(self.data, expected):
            raise ValueError(f"envelope kind {self.kind!r} carries {type(self.data).__name__}")
        return self


# --------------------------------------------------------------------------- payloads by source


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VideoPayload(_Payload):
    clip_seq: int
    t_video_start: float
    motion: float
    cuts_per_10s: float
    shot_type: Literal["wide", "close_player", "close_crowd", "replay", "other"]
    shot_conf: float
    replay: bool
    replay_conf: float
    prompts: dict[str, float]
    pan_dx: float
    pan_dy: float
    pan_speed: float  # frame widths per second
    pan_sustained_s: float
    score: float | None = None  # null for v0
    model: str


class CrowdPayload(_Payload):
    window_s: float
    loudness_dbfs: float
    loudness_rel_db: float  # against the feed's rolling 60 s median
    peak_dbfs: float
    nonspeech_dbfs: float
    nonspeech_ok: bool
    rise_db_s: float
    hush: bool
    events: dict[str, float] | None = None  # PANNs class to score; null without PANNs


class VoicePayload(_Payload):
    window_s: float
    speech_frac: float
    speaking: bool
    energy_rel_db: float | None = None
    pitch_rel_st: float | None = None
    pace_rel: float | None = None


class OcrPayload(_Payload):
    fields: dict[str, str]
    conf: dict[str, float]


class CommentaryPayload(_Payload):
    text: str
    t_start: float
    t_end: float
    lang: str
    avg_logprob: float
    no_speech_prob: float


class MatchPayload(_Payload):
    game_time_s: float
    period: int
    kind: str
    team: str | None = None
    player: str | None = None
    attrs: dict[str, Any] = Field(default_factory=dict)


PAYLOAD_MODELS: dict[str, type[_Payload]] = {
    "video": VideoPayload,
    "crowd": CrowdPayload,
    "voice": VoicePayload,
    "ocr": OcrPayload,
    "commentary": CommentaryPayload,
    "match": MatchPayload,
}


# --------------------------------------------------------------------------- message out of the engine


class PartValue(BaseModel):
    value: float  # 0 to 1
    weight: float  # effective weight after renormalisation
    contribution: float  # value * weight; contributions sum to intensity_raw
    phrase: str  # "frantic camera action"
    facts: list[Fact]


class Forecast(BaseModel):
    p: float  # calibrated probability of the sport's label
    label: str  # "big_chance", "on_track_pass", "lead_change"
    horizon_s: float
    source: Literal["model", "heuristic", "rule"]
    heating_up: bool  # p at or above the bundle's threshold
    model_version: str | None


class Imminent(BaseModel):
    value: float  # 0 to 1, after peak-hold
    cue: float  # 0 to 1, before context
    context: float  # context_floor to 1
    heating_up: bool  # value >= threshold for min_on_s
    phase: str | None  # weightlifting only: "idle", "walk_on", "setup", "setup_hush", "lift"
    source: Literal["rule"]  # "model" reserved for later
    facts: list[Fact]  # the cues behind it


class Reason(BaseModel):
    text: str
    facts: list[Fact]  # at least 2; every number in text appears here
    factors: list[str]  # parts or model features behind it
    polished: bool


class ScoreUpdate(BaseModel):
    v: Literal[1] = 1
    feed_id: str
    sport: Sport
    t_video: float
    loop_epoch: int
    game_clock: str | None  # "88:12", "Lap 41/58", "Q4 2:31"
    t_emit_ns: int
    stale: bool  # no ClockTick for 2 s
    intensity_raw: float  # sum(w_i * part_i) / sum(w_i) over present parts
    intensity: float  # EMA, alpha = 0.3
    intensity_pct: float  # 0 to 100 within sport; the front end ranks on this
    trend: Literal["rising", "steady", "falling"]
    trend_slope: float  # per second over the last 10 s
    parts: dict[str, PartValue | None]  # every PART_NAMES key; None = dropped out
    forecast: Forecast
    imminent: Imminent
    heating_up: bool  # forecast.heating_up or imminent.heating_up
    heating_up_by: list[Literal["forecast", "imminent"]]
    reasons: list[Reason]  # at most 3

    @model_validator(mode="after")
    def _check_parts(self) -> ScoreUpdate:
        if set(self.parts) != set(PART_NAMES):
            raise ValueError(f"parts must have exactly the keys {PART_NAMES}")
        if len(self.reasons) > 3:
            raise ValueError("at most 3 reasons")
        return self


# --------------------------------------------------------------------------- helpers


def make_envelope(data: Event | ClockTick | Control) -> Envelope:
    kind = "event" if isinstance(data, Event) else "clock" if isinstance(data, ClockTick) else "control"
    return Envelope(kind=kind, data=data)


def parse_envelope(raw: str | bytes) -> Envelope:
    """Parse one `/ingest` frame. Raises pydantic.ValidationError on anything off-contract."""
    return Envelope.model_validate_json(raw)


# Every exported contract, in a stable order. The in-process ClipBatch is a dataclass and lives in
# intensity.ingest.handoff; the model bundle layout is documented in CHANGELOG.md.
EXPORTED: dict[str, type[BaseModel]] = {
    "Envelope": Envelope,
    "Event": Event,
    "ClockTick": ClockTick,
    "Control": Control,
    "ScoreUpdate": ScoreUpdate,
    **{f"{name.capitalize()}Payload": model for name, model in PAYLOAD_MODELS.items()},
}


def json_schemas() -> dict[str, dict[str, Any]]:
    return {name: model.model_json_schema() for name, model in EXPORTED.items()}


def _schema_text(schema: dict[str, Any]) -> str:
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def export(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, schema in json_schemas().items():
        path = out_dir / f"{name}.schema.json"
        path.write_text(_schema_text(schema))
        written.append(path)
    return written


def diff(out_dir: Path) -> list[str]:
    """Names of contracts whose checked-in JSON Schema differs from the code."""
    changed = []
    for name, schema in json_schemas().items():
        path = out_dir / f"{name}.schema.json"
        if not path.exists() or path.read_text() != _schema_text(schema):
            changed.append(name)
    return changed


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out_dir = Path(argv[1]) if len(argv) > 1 else Path("contracts")
    if argv and argv[0] == "export":
        for path in export(out_dir):
            print(path)
        return 0
    if argv and argv[0] == "check":
        changed = diff(out_dir)
        if changed:
            print("contracts differ from schema.py: " + ", ".join(changed), file=sys.stderr)
            return 1
        print("contracts match schema.py")
        return 0
    print("usage: python -m intensity.schema export|check [contracts_dir]", file=sys.stderr)
    return 2


__all__ = [
    "PART_NAMES",
    "SPORTS",
    "ClockTick",
    "Control",
    "Envelope",
    "Event",
    "Fact",
    "Forecast",
    "Imminent",
    "PartValue",
    "Reason",
    "ScoreUpdate",
    "ValidationError",
    "make_envelope",
    "parse_envelope",
]

if __name__ == "__main__":
    raise SystemExit(main())
