from __future__ import annotations

import pytest

from intensity.config import PercentileConfig, TrendConfig
from intensity.engine.adapters.base import AdapterPart
from intensity.engine.scorer import Ema, SportPercentile, Trend, combine, forecast_part_value
from intensity.schema import PART_NAMES

W = {"closeness": 0.2, "momentum": 0.15, "stakes": 0.15, "crowd": 0.15, "commentary": 0.1, "video": 0.15, "forecast": 0.2, "imminent": 0.15}


def _parts(**values) -> dict:
    return {name: AdapterPart(values.get(name, 0.5), name) for name in PART_NAMES}


def test_weighted_mean_and_contributions_sum():
    raw, parts = combine(_parts(closeness=1.0, imminent=0.0), W)
    expected = sum(W[n] * (1.0 if n == "closeness" else 0.0 if n == "imminent" else 0.5) for n in PART_NAMES) / sum(W.values())
    assert raw == pytest.approx(expected)
    assert sum(p.contribution for p in parts.values()) == pytest.approx(raw, abs=1e-3)
    assert sum(p.weight for p in parts.values()) == pytest.approx(1.0, abs=1e-3)


def test_renormalises_when_parts_drop_out():
    p = _parts(video=1.0)
    p["crowd"] = None
    p["commentary"] = None
    raw, parts = combine(p, W)
    assert parts["crowd"] is None and parts["commentary"] is None
    present = [n for n in PART_NAMES if n not in ("crowd", "commentary")]
    total = sum(W[n] for n in present)
    assert parts["video"].weight == pytest.approx(W["video"] / total, abs=1e-4)
    assert sum(parts[n].weight for n in present) == pytest.approx(1.0, abs=1e-3)


def test_zero_weight_part_reported_but_not_counted():
    raw, parts = combine(_parts(video=1.0), {**W, "video": 0.0})
    assert parts["video"].weight == 0.0 and parts["video"].contribution == 0.0
    assert raw == pytest.approx(0.5)


def test_values_clamped():
    raw, parts = combine(_parts(closeness=1.7, momentum=-1), W)
    assert parts["closeness"].value == 1.0 and parts["momentum"].value == 0.0


def test_ema():
    ema = Ema(0.3)
    assert ema.update(1.0) == 1.0
    assert ema.update(0.0) == pytest.approx(0.7)
    ema.reset()
    assert ema.update(0.2) == 0.2


@pytest.mark.parametrize("step,label", [(0.05, "rising"), (-0.05, "falling"), (0.0, "steady")])
def test_trend_sign(step, label):
    tr = Trend(TrendConfig())
    out = None
    for i in range(12):
        out = tr.update(float(i), 0.5 + step * i)
    assert out[0] == label
    assert out[1] == pytest.approx(step, abs=1e-9)


def test_trend_window_drops_old_points():
    tr = Trend(TrendConfig(window_s=5))
    for i in range(10):
        tr.update(float(i), 1.0 if i < 5 else 0.0)
    assert len(tr.buf) == 6


def test_forecast_part_through_quantiles():
    q = [0.0, 0.1, 0.2, 0.5, 1.0]
    assert forecast_part_value(0.1, q, "football") == pytest.approx(0.25)
    assert forecast_part_value(0.35, q, "football") == pytest.approx(0.625)
    assert forecast_part_value(0.5, None, "f1") == 0.5


def test_weightlifting_forecast_part_peaks_at_coin_flip():
    assert forecast_part_value(0.5, None, "weightlifting") == 1.0
    assert forecast_part_value(0.9, None, "weightlifting") == pytest.approx(0.2)
    assert forecast_part_value(0.1, None, "weightlifting") == pytest.approx(0.2)


def test_sport_percentile_seeded_then_live():
    cfg = PercentileConfig(window_s=100, min_samples=3)
    pct = SportPercentile(cfg, seed=[0.1, 0.2, 0.3, 0.4])
    assert pct.pct(0.35) == pytest.approx(75.0)  # from the seed alone
    for t, v in enumerate([0.9, 0.9, 0.9]):
        pct.add(float(t), v)
    assert pct.pct(0.35) == 0.0  # three live samples: the seed no longer counts
    pct.add(500.0, 0.1)  # the old samples fall out of the 100 s window
    assert len(pct.samples) == 1
