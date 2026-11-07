from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from intensity.config import deep_merge, load_config, load_raw, load_weights, parse_override, save_resolved
from intensity.schema import PART_NAMES


def _tree(tmp_path: Path) -> Path:
    (tmp_path / "default").mkdir()
    (tmp_path / "default" / "engine.yaml").write_text("engine:\n  ema_alpha: 0.3\n  imminent: {hold_s: 3.0}\n")
    (tmp_path / "default" / "ingest.yaml").write_text("ingest: {decoder: nvdec}\n")
    (tmp_path / "local.yaml").write_text("ingest: {decoder: pyav_cpu}\nengine: {imminent: {min_on_s: 2.0}}\n")
    return tmp_path


def test_defaults_then_overlay_then_override(tmp_path):
    root = _tree(tmp_path)
    raw = load_raw(["local"], ["engine.imminent.hold_s=2.5"], root)
    assert raw["ingest"]["decoder"] == "pyav_cpu"
    assert raw["engine"]["imminent"] == {"hold_s": 2.5, "min_on_s": 2.0}
    assert raw["engine"]["ema_alpha"] == 0.3


def test_override_values_parse_as_yaml():
    assert parse_override("a.b=1.5") == {"a": {"b": 1.5}}
    assert parse_override("a=true") == {"a": True}
    assert parse_override("a=[1, 2]") == {"a": [1, 2]}
    with pytest.raises(ValueError):
        parse_override("no_equals")


def test_deep_merge_does_not_mutate():
    base = {"a": {"b": 1}}
    merged = deep_merge(base, {"a": {"c": 2}})
    assert merged == {"a": {"b": 1, "c": 2}} and base == {"a": {"b": 1}}


def test_engine_section_is_strict(tmp_path):
    root = _tree(tmp_path)
    with pytest.raises(ValidationError):
        load_config(overrides=["engine.not_a_setting=1"], config_dir=root)


def test_repo_config_loads_and_has_every_weight():
    cfg = load_config()
    assert cfg.engine.imminent.context_floor == 0.3  # D12
    assert cfg.engine.imminent.hold_s == 3.0
    assert cfg.engine.imminent.min_on_s == 1.0
    assert cfg.clip["frames"] == 16  # D2 placeholders
    weights = load_weights(cfg.engine.weights_path)
    for sport in ("football", "f1", "nba", "weightlifting"):
        assert set(PART_NAMES) <= set(weights[sport])
        assert weights[sport]["video"] == 0.15 and weights[sport]["forecast"] == 0.2 and weights[sport]["imminent"] == 0.15  # D6


def test_save_resolved_round_trips(tmp_path):
    cfg = load_config()
    path = save_resolved(cfg, tmp_path / "run")
    assert yaml.safe_load(path.read_text())["engine"]["ema_alpha"] == cfg.engine.ema_alpha
