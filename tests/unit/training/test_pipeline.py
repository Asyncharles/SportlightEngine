"""T1 pipeline: featurize, label, split, train, calibrate, evaluate and export, on synthetic matches."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from intensity.engine.adapters import make_adapter
from intensity.engine.registry import Registry
from intensity.schema import MatchPayload
from intensity.training.evaluate import Highlight, find_peaks, parse_clock, score_peaks
from intensity.training.featurize import featurize_match
from intensity.training.labels import label_match
from intensity.training.model import block_centres, calibrate, heating_threshold, search_settings
from intensity.training.sources import football
from intensity.training.split import (
    ALL,
    SplitError,
    assert_no_demo_leak,
    is_in_sample,
    members,
    split_matches,
)
from tests.unit.training.test_convert import football_raw

LABEL = {"name": "big_chance", "horizon_s": 30, "min_xg": 0.1}


def _lightgbm():
    try:
        import lightgbm

        return lightgbm
    except (ImportError, OSError):  # OSError: libomp missing on macOS
        return None


def synthetic_match(seed: int, half_s: float = 600.0) -> list[dict]:
    """Two short halves of passes and carries drifting up and down the pitch, with shots near goal."""
    rng = random.Random(seed)
    teams = ("Home FC", "Away FC")
    out = [
        {
            "game_time_s": 0.0,
            "period": 1,
            "kind": "kickoff",
            "team": teams[0],
            "player": None,
            "attrs": {"home": teams[0], "away": teams[1], "stage": "Group Stage"},
        },
    ]
    for period, start in ((1, 0.0), (2, 2700.0)):
        out.append(
            {
                "game_time_s": start,
                "period": period,
                "kind": "period_start",
                "team": None,
                "player": None,
                "attrs": {},
            }
        )
        t, x, team = start + 1.0, 60.0, teams[period - 1]
        while t < start + half_s - 1:
            t += rng.uniform(0.5, 3.0)
            end_x = min(119.0, max(1.0, x + rng.uniform(-15, 25)))
            y = rng.uniform(5, 75)
            if x > 100 and rng.random() < 0.25:
                xg = round(rng.uniform(0.02, 0.5), 3)
                kind = "goal" if rng.random() < xg else "shot"
                out.append(
                    {
                        "game_time_s": round(t, 3),
                        "period": period,
                        "kind": kind,
                        "team": team,
                        "player": "Striker",
                        "attrs": {"xg": xg, "x": round(x, 1), "y": round(y, 1)},
                    }
                )
                team, x = teams[teams.index(team) ^ 1], 20.0
                continue
            kind = rng.choice(("pass", "pass", "carry"))
            attrs = {"x": round(x, 1), "y": round(y, 1), "end_x": round(end_x, 1), "end_y": round(y, 1)}
            if kind == "pass":
                attrs["completed"] = rng.random() < 0.8
            out.append(
                {
                    "game_time_s": round(t, 3),
                    "period": period,
                    "kind": kind,
                    "team": team,
                    "player": "Mid",
                    "attrs": attrs,
                }
            )
            x = end_x
            if kind == "pass" and not attrs["completed"]:
                team, x = teams[teams.index(team) ^ 1], 120.0 - x
        out.append(
            {
                "game_time_s": start + half_s,
                "period": period,
                "kind": "period_end",
                "team": None,
                "player": None,
                "attrs": {},
            }
        )
    return out


def payloads(raw: list[dict]) -> list[MatchPayload]:
    return [MatchPayload.model_validate(r) for r in raw]


# --------------------------------------------------------------------------- featurize and labels


def test_featurize_replays_the_live_adapter_once_per_second() -> None:
    recs = payloads(football.to_records(football_raw()))
    rows = featurize_match("football", recs, 1.0, skip_periods=(5,))
    assert 5 not in set(rows.period)
    first = rows.period == 1
    assert rows.game_time_s[first][0] == 0.0 and np.all(np.diff(rows.game_time_s[first]) == 1.0)
    # The row at 63 s is what an adapter fed every record up to 63 s returns, as the engine would.
    adapter = make_adapter("football")
    for r in recs:
        if (r.period, r.game_time_s) <= (1, 63.0):
            adapter.on_match(r)
    i = int(np.flatnonzero(first & (rows.game_time_s == 63.0))[0])
    np.testing.assert_array_equal(rows.x[i], adapter.features(63.0))


def test_features_at_t_ignore_every_record_after_t() -> None:
    """The leakage property: shuffle, move and drop everything after t; rows up to t stay identical."""
    recs = payloads(synthetic_match(1))
    cut = (1, 300.0)
    past = [r for r in recs if (r.period, r.game_time_s) <= cut]
    future = [r for r in recs if (r.period, r.game_time_s) > cut]
    rng = random.Random(7)
    attrs = [r.attrs for r in future]
    rng.shuffle(attrs)
    scrambled = [
        r.model_copy(update={"attrs": a, "team": rng.choice(("Home FC", "Away FC"))})
        for r, a in zip(future, attrs, strict=True)
        if rng.random() < 0.6
    ]
    a = featurize_match("football", recs)
    b = featurize_match("football", past + scrambled)
    keep_a = (a.period == 1) & (a.game_time_s <= cut[1])
    keep_b = (b.period == 1) & (b.game_time_s <= cut[1])
    np.testing.assert_array_equal(a.x[keep_a], b.x[keep_b])


def test_featurize_rebuilds_when_the_adapter_code_changes(tmp_path: Path, monkeypatch) -> None:
    from intensity.config import load_config
    from intensity.training import featurize as fz

    events = tmp_path / "events" / "football"
    events.mkdir(parents=True)
    (events / "m0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in synthetic_match(0, half_s=60)))
    config = load_config(
        overrides=[
            f"training.paths.events={tmp_path / 'events'}",
            f"training.paths.tables={tmp_path / 'tables'}",
        ]
    )
    meta = tmp_path / "tables" / "football" / "features.meta.json"
    fz.featurize(config, "football")
    first = json.loads(meta.read_text())["inputs"]["code_hash"]
    assert first == fz.code_hash(fz.make_adapter("football").__class__, fz.Adapter, fz.featurize_match)
    monkeypatch.setattr(fz, "code_hash", lambda *objects: "changed")  # same feature names, new code
    fz.featurize(config, "football")
    assert json.loads(meta.read_text())["inputs"]["code_hash"] == "changed"


def test_label_looks_forward_within_the_period_and_drops_unknowns() -> None:
    recs = payloads(football.to_records(football_raw()))
    rows = featurize_match("football", recs, 1.0, skip_periods=(5,))
    y = label_match(recs, rows.period, rows.game_time_s, "football", LABEL)
    by_t = {(int(p), float(t)): v for p, t, v in zip(rows.period, rows.game_time_s, y, strict=True)}
    assert by_t[(1, 32.0)] == 1.0 and by_t[(1, 61.0)] == 1.0  # the penalty at 62 s, 30 s ahead
    assert by_t[(1, 31.0)] == 0.0
    assert by_t[(1, 62.0)] == 0.0  # the goal's own tick: it is already in the features
    assert np.isnan(by_t[(1, 590.0)])  # within 30 s of the period's last record, nothing follows
    assert by_t[(2, 2990.0)] == 1.0  # the own goal counts


def test_low_xg_shot_is_not_a_big_chance() -> None:
    recs = payloads(
        [
            {"game_time_s": 0.0, "period": 1, "kind": "period_start", "attrs": {}},
            {"game_time_s": 10.0, "period": 1, "kind": "shot", "team": "A", "attrs": {"xg": 0.05}},
            {"game_time_s": 100.0, "period": 1, "kind": "period_end", "attrs": {}},
        ]
    )
    y = label_match(recs, np.ones(3, dtype=int), np.array([0.0, 5.0, 60.0]), "football", LABEL)
    assert list(y) == [0.0, 0.0, 0.0]


# --------------------------------------------------------------------------- split


def test_holdout_split_keeps_demo_matches_in_test_and_is_deterministic() -> None:
    ids = [f"m{i:02d}" for i in range(40)]
    parts = split_matches(ids, {"m07"}, 0.15, 0.15, seed=1234)
    assert parts["m07"] == "test"
    counts = {k: sum(1 for v in parts.values() if v == k) for k in ("train", "val", "test")}
    assert counts == {"train": 27, "val": 6, "test": 7}
    assert parts == split_matches(list(reversed(ids)), {"m07"}, 0.15, 0.15, seed=1234)
    with pytest.raises(SplitError):
        assert_no_demo_leak({**parts, "m07": "train"}, {"m07"})
    assert not is_in_sample(parts)


def test_in_sample_split_trains_on_every_match_demo_included() -> None:
    ids = [f"m{i:02d}" for i in range(40)]
    parts = split_matches(ids, {"m07"}, 0.15, 0.15, seed=1234, mode="in_sample")
    assert set(parts.values()) == {ALL} and is_in_sample(parts)
    assert members(parts, "train") == members(parts, "test") == sorted(ids)
    assert "m07" in members(parts, "train")  # hard rule 6 is suspended in this mode, by decision


# --------------------------------------------------------------------------- calibrate and highlights


def test_calibration_is_monotone_without_ties_and_threshold_follows_the_rules() -> None:
    rng = np.random.default_rng(0)
    raw = rng.uniform(size=5000)
    y = (rng.uniform(size=5000) < 0.02 + 0.2 * raw**4).astype(int)
    cal = calibrate(raw, y, {"precision_mult": 3.0, "max_heating_frac": 0.15, "n_quantiles": 21})
    assert np.all(np.diff(cal.x) > 0) and np.all(np.diff(cal.y) >= 0)
    p = cal.apply(raw)
    assert len(np.unique(p)) > len(cal.x)  # interpolated between block centres, not flat steps
    above = p >= cal.threshold
    assert cal.threshold_reached and above.mean() <= 0.15 and y[above].mean() >= 3 * y.mean()
    assert len(cal.p_quantiles) == 21 and cal.p_quantiles == sorted(cal.p_quantiles)


def test_threshold_unreached_never_heats_up() -> None:
    p = np.linspace(0, 1, 100)
    y = np.zeros(100, dtype=int)
    y[::10] = 1  # no signal: precision is the base rate at every level
    threshold, reached = heating_threshold(p, y, 3.0, 0.15)
    assert not reached and (p >= threshold).sum() == 0


def test_block_centres_drop_flat_steps() -> None:
    xs, ys = block_centres(np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.1, 0.1, 0.5, 0.5]))
    assert xs == [0.5, 2.5] and ys == [0.1, 0.5]


def test_peaks_are_spaced_and_scored_against_highlights() -> None:
    period = np.array([1] * 200 + [2] * 200)
    t = np.r_[np.arange(200.0), 2700 + np.arange(200.0)]
    p = np.zeros(400)
    p[[10, 30, 100, 250]] = [0.9, 0.8, 0.5, 0.7]  # 30 is within 60 s of 10, so it is skipped
    peaks = find_peaks(period, t, p, n=3, gap_s=60)
    assert [(int(period[i]), float(t[i])) for i in peaks] == [(1, 10.0), (1, 100.0), (2, 2750.0)]
    highlights = [Highlight(1, 40.0, "goal"), Highlight(2, 2800.0, "late"), Highlight(2, 50.0, "other half")]
    hits = score_peaks([(int(period[i]), float(t[i])) for i in peaks], highlights, 45.0)
    assert [h.what if h else None for h in hits] == ["goal", None, None]
    assert parse_clock("104:28") == 6268.0 and parse_clock(90) == 90.0


def test_search_is_capped_and_seeded() -> None:
    grid = {"a": [1, 2, 3], "b": [1, 2], "c": [1, 2, 3, 4]}
    picks = search_settings({"grid": grid, "max_settings": 5}, seed=3)
    assert len(picks) == 5 and picks == search_settings({"grid": grid, "max_settings": 5}, seed=3)
    assert len(search_settings({"grid": {"a": [1, 2]}, "max_settings": 20}, seed=3)) == 2


# --------------------------------------------------------------------------- end to end


@pytest.mark.skipif(_lightgbm() is None, reason="lightgbm unavailable (on macOS: brew install libomp)")
@pytest.mark.parametrize("mode", ["in_sample", "holdout"])
def test_build_end_to_end_exports_a_bundle_the_engine_loads(tmp_path: Path, mode: str) -> None:
    from intensity.training.cli import main

    events = tmp_path / "events" / "football"
    events.mkdir(parents=True)
    for i in range(14):
        (events / f"m{i:02d}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in synthetic_match(i)))
    (tmp_path / "feeds.yaml").write_text(
        "feeds:\n  - {feed_id: fb, sport: football, sample: samples/fb.mp4, rtsp_path: fb,"
        " match_ref: {source: statsbomb, match_id: m03}}\n"
    )
    highlights = tmp_path / "highlights"
    highlights.mkdir()
    shots = [r for r in synthetic_match(3) if r["kind"] in ("shot", "goal")]
    (highlights / "m03.yaml").write_text(
        "highlights:\n"
        + "".join(f"  - {{period: {r['period']}, clock: {r['game_time_s']}, what: chance}}\n" for r in shots)
    )
    overrides = [f"training.paths.{k}={tmp_path / k}" for k in ("tables", "runs", "models", "reports")]
    overrides += [
        f"training.paths.events={tmp_path / 'events'}",
        f"training.paths.feeds={tmp_path / 'feeds.yaml'}",
        f"training.paths.highlights={highlights}",
        f"training.split.mode={mode}",
        "training.in_sample.num_boost_round=40",
        "training.search.max_settings=2",
        "training.search.num_boost_round=40",
        "training.search.early_stopping_rounds=10",
        "training.evaluate.shap_rows=500",
        # Synthetic data: the check is that the pipeline runs and exports, not the model's quality.
        "training.evaluate.target_pr_auc_mult=0.0",
        "training.highlights.target_hits=0",
    ]
    args = [a for o in overrides for a in ("--set", o)]
    assert main([*args, "build", "--sport", "football", "--no-source"]) == 0

    version = (tmp_path / "runs" / "football" / "LATEST").read_text().strip()
    bundle_dir = tmp_path / "models" / "football" / version
    meta = json.loads((bundle_dir / "meta.json").read_text())
    report = tmp_path / "reports" / "football" / version
    if mode == "holdout":
        assert "m03" in meta["matches"]["test"] and "m03" not in meta["matches"]["train"] + meta["matches"]["val"]
        assert meta["evaluation"] == "held-out" and "In-sample metrics" not in (report / "report.html").read_text()
    else:
        assert len(meta["matches"]["train"]) == 14 and meta["matches"]["train"] == meta["matches"]["test"]
        assert meta["evaluation"] == "in-sample" and meta["metrics"]["test"]["evaluation"] == "in-sample"
        assert "In-sample metrics" in (report / "report.html").read_text()
        assert json.loads((report / "report.json").read_text())["metrics"]["evaluation"] == "in-sample"
    assert meta["metrics"]["passed"] and {"pr_auc", "brier"} <= set(meta["metrics"]["test"])
    assert not (tmp_path / "models" / "football" / "CURRENT").exists()  # only the ML lead moves it
    assert (tmp_path / "reports" / "football" / version / "report.html").exists()

    (tmp_path / "models" / "football" / "CURRENT").write_text(version)
    registry = Registry(tmp_path / "models")
    assert registry.reload()["football"] == version
    p, contrib = registry.get("football").predict(np.zeros(len(meta["features"])))
    assert 0.0 <= p <= 1.0 and len(contrib) == len(meta["features"])

    # A second build with nothing changed reuses the tables and lands on the same version.
    assert main([*args, "build", "--sport", "football", "--no-source"]) == 0
    assert (tmp_path / "runs" / "football" / "LATEST").read_text().strip() == version
