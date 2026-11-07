"""Contracts v1: fixtures validate, round trips are lossless, and the schema snapshot is unchanged."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from intensity import schema
from intensity.schema import (
    PART_NAMES,
    ClockTick,
    Control,
    Envelope,
    Event,
    ScoreUpdate,
    make_envelope,
    parse_envelope,
)

ROOT = Path(__file__).resolve().parents[2]


def test_every_source_has_a_fixture(envelope_fixtures):
    sources = {f["data"]["source"] for f in envelope_fixtures.values() if f["kind"] == "event"}
    assert sources == set(schema.PAYLOAD_MODELS)
    assert {"clock", "control"} <= set(envelope_fixtures)


@pytest.mark.parametrize("name", ["clock", "commentary", "control", "crowd", "match", "ocr", "video", "voice", "voice_silent"])
def test_fixture_round_trip(envelope_fixtures, name):
    raw = json.dumps(envelope_fixtures[name])
    env = parse_envelope(raw)
    again = parse_envelope(env.model_dump_json())
    assert again == env
    expected = {"clock": ClockTick, "control": Control}.get(env.kind, Event)
    assert isinstance(env.data, expected)


def test_kind_must_match_data(envelope_fixtures):
    bad = dict(envelope_fixtures["clock"], kind="event")
    with pytest.raises(ValidationError):
        Envelope.model_validate(bad)


def test_payload_is_validated_per_source(envelope_fixtures):
    bad = json.loads(json.dumps(envelope_fixtures["crowd"]))
    del bad["data"]["payload"]["rise_db_s"]
    with pytest.raises(ValidationError):
        Envelope.model_validate(bad)
    extra = json.loads(json.dumps(envelope_fixtures["video"]))
    extra["data"]["payload"]["surprise"] = 1
    with pytest.raises(ValidationError):
        Envelope.model_validate(extra)


def test_unknown_sport_and_version_rejected(envelope_fixtures):
    bad = json.loads(json.dumps(envelope_fixtures["match"]))
    bad["data"]["sport"] = "curling"
    with pytest.raises(ValidationError):
        Envelope.model_validate(bad)
    bad = json.loads(json.dumps(envelope_fixtures["match"]))
    bad["data"]["v"] = 2
    with pytest.raises(ValidationError):
        Envelope.model_validate(bad)


def test_make_envelope_picks_kind():
    tick = ClockTick(feed_id="a", t_video=1.0, loop_epoch=0, t_emit_ns=1)
    ctl = Control(feed_id="a", kind="loop", loop_epoch=1, t_emit_ns=1)
    assert make_envelope(tick).kind == "clock"
    assert make_envelope(ctl).kind == "control"


def test_schema_snapshot_unchanged():
    """Fails on any contract change not exported to contracts/ (and recorded in CHANGELOG.md)."""
    assert schema.diff(ROOT / "contracts") == [], "run `make contracts` after the ML lead approves, and update CHANGELOG.md"


def test_part_names_are_the_eight_parts():
    assert PART_NAMES == ("closeness", "momentum", "stakes", "crowd", "commentary", "video", "forecast", "imminent")


@pytest.mark.parametrize("path", sorted((ROOT / "contracts" / "samples").glob("*.json")), ids=lambda p: p.name)
def test_front_end_samples_validate(path):
    update = ScoreUpdate.model_validate_json(path.read_text())
    assert set(update.parts) == set(PART_NAMES)
    assert len(update.reasons) <= 3
    for reason in update.reasons:
        assert len(reason.facts) >= 2
    assert update.heating_up == bool(update.heating_up_by)


def test_samples_cover_every_sport_and_imminent():
    names = {p.stem for p in (ROOT / "contracts" / "samples").glob("*.json")}
    for sport in ("football", "f1", "nba", "weightlifting"):
        assert f"score_update_{sport}" in names
    assert any(n.endswith("_imminent") for n in names)


def test_score_update_needs_every_part():
    sample = json.loads(next((ROOT / "contracts" / "samples").glob("score_update_football.json")).read_text())
    del sample["parts"]["imminent"]
    with pytest.raises(ValidationError):
        ScoreUpdate.model_validate(sample)
