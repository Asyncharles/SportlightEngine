from __future__ import annotations

import json

import numpy as np
import pytest

from intensity.config import ForecastConfig
from intensity.engine.adapters import FootballAdapter
from intensity.engine.adapters.base import feature_hash
from intensity.engine.forecast import Forecaster
from intensity.engine.registry import Registry


def _lightgbm():
    try:
        import lightgbm

        return lightgbm
    except (ImportError, OSError):  # OSError: libomp missing on macOS
        return None


def test_no_bundle_serves_heuristic(tmp_path):
    reg = Registry(tmp_path)
    assert reg.reload()["football"] is None
    fr = Forecaster("football", reg, ForecastConfig()).predict(FootballAdapter(), 0.0)
    assert fr.forecast.source == "heuristic" and fr.forecast.model_version is None


def test_bad_bundle_falls_back_and_reports(tmp_path):
    d = tmp_path / "football" / "v1"
    d.mkdir(parents=True)
    (tmp_path / "football" / "CURRENT").write_text("v1\n")
    (d / "meta.json").write_text(json.dumps({"features": ["wrong"], "feature_hash": "x", "threshold": 0.3}))
    (d / "calibrator.json").write_text(json.dumps({"x": [0, 1], "y": [0, 1]}))
    reg = Registry(tmp_path)
    reg.reload()
    assert "football" in reg.status()["errors"]
    fr = Forecaster("football", reg, ForecastConfig()).predict(FootballAdapter(), 0.0)
    assert fr.forecast.source == "heuristic"


@pytest.mark.skipif(_lightgbm() is None, reason="lightgbm unavailable (on macOS: brew install libomp)")
def test_bundle_loads_predicts_and_explains(tmp_path):
    lgb = _lightgbm()
    feats = FootballAdapter.FEATURES
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, len(feats)))
    y = (x[:, 0] + 0.3 * rng.normal(size=400) > 0.5).astype(int)
    booster = lgb.train({"objective": "binary", "verbose": -1, "num_leaves": 4}, lgb.Dataset(x, y), 20)
    d = tmp_path / "football" / "v1"
    d.mkdir(parents=True)
    booster.save_model(str(d / "model.txt"))
    (d / "calibrator.json").write_text(json.dumps({"x": [0.0, 1.0], "y": [0.0, 1.0]}))
    (d / "meta.json").write_text(json.dumps({"features": list(feats), "feature_hash": feature_hash(feats), "threshold": 0.4,
                                             "label": "big_chance", "horizon_s": 30, "p_quantiles": [0, 0.2, 0.5, 1]}))
    (tmp_path / "football" / "CURRENT").write_text("v1")
    reg = Registry(tmp_path)
    assert reg.reload()["football"] == "v1"
    fr = Forecaster("football", reg, ForecastConfig()).predict(FootballAdapter(), 0.0)
    assert fr.forecast.source == "model" and fr.forecast.model_version == "v1"
    assert set(fr.contribs) == set(feats)
    p, contrib = reg.get("football").predict(np.zeros(len(feats)))
    assert 0 <= p <= 1 and len(contrib) == len(feats)
