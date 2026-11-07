"""Per-tick forecast: the served bundle, or the adapter's heuristic until one is approved."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from intensity.config import ForecastConfig
from intensity.engine.adapters.base import Adapter
from intensity.engine.registry import Registry
from intensity.engine.scorer import forecast_part_value
from intensity.schema import Forecast

log = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    forecast: Forecast
    part_value: float
    contribs: dict[str, float] = field(default_factory=dict)  # feature -> SHAP contribution
    error: str | None = None


class Forecaster:
    def __init__(self, sport: str, registry: Registry, cfg: ForecastConfig):
        self.sport = sport
        self.registry = registry
        self.cfg = cfg

    def predict(self, adapter: Adapter, game_time: float) -> ForecastResult:
        bundle = self.registry.get(self.sport) if self.cfg.source == "model" else None
        if bundle is not None:
            try:
                p, contrib = bundle.predict(adapter.features(game_time))
                self._shadow(adapter, game_time, p)
                return ForecastResult(
                    forecast=Forecast(
                        p=round(p, 5),
                        label=str(bundle.meta.get("label", adapter.LABEL)),
                        horizon_s=float(bundle.meta.get("horizon_s", adapter.HORIZON_S)),
                        source="model",
                        heating_up=p >= bundle.threshold,
                        model_version=bundle.version,
                    ),
                    part_value=forecast_part_value(p, bundle.quantiles, self.sport),
                    contribs=dict(zip(bundle.features, (float(c) for c in contrib), strict=True)),
                )
            except Exception as exc:  # never let a model take down the tick
                log.exception("forecast failed for %s; using heuristic", self.sport)
                return self._heuristic(adapter, game_time, error=str(exc))
        return self._heuristic(adapter, game_time)

    def _heuristic(self, adapter: Adapter, game_time: float, error: str | None = None) -> ForecastResult:
        p = float(adapter.heuristic_forecast(game_time))
        source = "rule" if adapter.FORECAST_SOURCE == "rule" else "heuristic"
        return ForecastResult(
            forecast=Forecast(
                p=round(p, 5),
                label=adapter.LABEL,
                horizon_s=adapter.HORIZON_S,
                source=source,
                heating_up=source != "rule" and p >= self.cfg.heuristic_threshold,
                model_version=None,
            ),
            part_value=forecast_part_value(p, self.cfg.heuristic_quantiles, self.sport),
            error=error,
        )

    def _shadow(self, adapter: Adapter, game_time: float, served_p: float) -> None:
        shadow = self.registry.get_shadow(self.sport)
        if shadow is None:
            return
        try:
            p, _ = shadow.predict(adapter.features(game_time))
            log.info("shadow forecast", extra={"fields": {"sport": self.sport, "shadow": shadow.version, "p": p, "served_p": served_p}})
        except Exception as exc:
            log.warning("shadow forecast failed: %s", exc)
