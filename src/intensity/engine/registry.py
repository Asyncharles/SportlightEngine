"""Model bundles: `models/<sport>/<version>/{model.txt, calibrator.json, meta.json}` and `CURRENT`.

Each sport's CURRENT bundle loads at startup and is checked against the adapter's FEATURES and
their hash; on any mismatch the sport serves the heuristic and /health says why. `reload()` swaps
bundles atomically between ticks.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from intensity.config import resolve_path
from intensity.engine.adapters import ADAPTERS
from intensity.engine.adapters.base import feature_hash

log = logging.getLogger(__name__)


@dataclass
class Bundle:
    sport: str
    version: str
    booster: Any  # lightgbm.Booster
    cal_x: np.ndarray
    cal_y: np.ndarray
    meta: dict[str, Any]
    features: tuple[str, ...] = field(default=())

    @property
    def threshold(self) -> float:
        return float(self.meta["threshold"])

    @property
    def quantiles(self) -> list[float]:
        return list(self.meta.get("p_quantiles", []))

    def predict(self, row: np.ndarray) -> tuple[float, np.ndarray]:
        """Calibrated p and per-feature SHAP contributions (exact tree SHAP via pred_contrib)."""
        x = row.reshape(1, -1)
        raw = float(self.booster.predict(x)[0])
        p = float(np.interp(raw, self.cal_x, self.cal_y))
        contrib = np.asarray(self.booster.predict(x, pred_contrib=True))[0][:-1]
        return p, contrib


def load_bundle(sport: str, bundle_dir: Path) -> Bundle:
    try:
        import lightgbm as lgb
    except (ImportError, OSError) as exc:  # OSError: libomp missing on macOS
        raise RuntimeError(f"lightgbm unavailable: {exc}") from exc
    meta = json.loads((bundle_dir / "meta.json").read_text())
    cal = json.loads((bundle_dir / "calibrator.json").read_text())
    features = tuple(meta["features"])
    expected = ADAPTERS[sport].FEATURES
    if features != tuple(expected):
        raise ValueError(f"feature list differs from {sport} adapter ({len(features)} vs {len(expected)} features)")
    if meta.get("feature_hash") != feature_hash(expected):
        raise ValueError(f"feature_hash {meta.get('feature_hash')} != adapter {feature_hash(expected)}")
    booster = lgb.Booster(model_file=str(bundle_dir / "model.txt"))
    if booster.num_feature() != len(features):
        raise ValueError(f"booster expects {booster.num_feature()} features, meta lists {len(features)}")
    return Bundle(
        sport=sport,
        version=bundle_dir.name,
        booster=booster,
        cal_x=np.asarray(cal["x"], dtype=float),
        cal_y=np.asarray(cal["y"], dtype=float),
        meta=meta,
        features=features,
    )


class Registry:
    def __init__(self, models_dir: str | Path, shadow: dict[str, str] | None = None):
        self.models_dir = resolve_path(models_dir)
        self.shadow_versions = shadow or {}
        self._lock = threading.Lock()
        self.bundles: dict[str, Bundle] = {}
        self.shadow: dict[str, Bundle] = {}
        self.errors: dict[str, str] = {}

    def _load_sport(self, sport: str, version: str | None = None) -> Bundle | None:
        sport_dir = self.models_dir / sport
        current = sport_dir / "CURRENT"
        if version is None:
            if not current.exists():
                return None
            version = current.read_text().strip()
        return load_bundle(sport, sport_dir / version)

    def reload(self) -> dict[str, str | None]:
        """Load every sport's CURRENT (and shadow) bundle, then swap them in at once."""
        bundles: dict[str, Bundle] = {}
        shadow: dict[str, Bundle] = {}
        errors: dict[str, str] = {}
        for sport in ADAPTERS:
            try:
                bundle = self._load_sport(sport)
                if bundle is not None:
                    bundles[sport] = bundle
            except Exception as exc:  # any bad bundle falls back to the heuristic
                errors[sport] = str(exc)
                log.error("bundle for %s rejected, serving heuristic: %s", sport, exc)
            if sport in self.shadow_versions:
                try:
                    sb = self._load_sport(sport, self.shadow_versions[sport])
                    if sb is not None:
                        shadow[sport] = sb
                except Exception as exc:
                    errors[f"{sport}:shadow"] = str(exc)
        with self._lock:
            self.bundles, self.shadow, self.errors = bundles, shadow, errors
        return {sport: (b.version if (b := bundles.get(sport)) else None) for sport in ADAPTERS}

    def get(self, sport: str) -> Bundle | None:
        with self._lock:
            return self.bundles.get(sport)

    def get_shadow(self, sport: str) -> Bundle | None:
        with self._lock:
            return self.shadow.get(sport)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "versions": {s: b.version for s, b in self.bundles.items()},
                "shadow": {s: b.version for s, b in self.shadow.items()},
                "errors": dict(self.errors),
            }
