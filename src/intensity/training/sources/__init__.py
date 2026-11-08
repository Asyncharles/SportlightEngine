"""Raw-data fetchers per sport (T0). Weightlifting has none: its session CSV is entered by hand."""

from __future__ import annotations

from typing import Any

from intensity.config import Config, resolve_path
from intensity.training.sources import f1, football, nba
from intensity.training.sources.base import Fetcher, FetchStats
from intensity.training.sources.convert import ConvertStats, ToRecords, convert_dir
from intensity.training.sources.f1 import JolpicaFetcher
from intensity.training.sources.football import StatsBombFetcher
from intensity.training.sources.nba import NbaFetcher

FETCHERS: dict[str, type[Fetcher]] = {"football": StatsBombFetcher, "f1": JolpicaFetcher, "nba": NbaFetcher}
CONVERTERS: dict[str, ToRecords] = {
    "football": football.to_records,
    "f1": f1.to_records,
    "nba": nba.to_records,
}


def make_fetcher(sport: str, config: Config, **kwargs: Any) -> Fetcher:
    if sport not in FETCHERS:
        raise ValueError(f"no fetcher for {sport!r}; choose from {sorted(FETCHERS)}")
    training = config.training
    raw_dir = resolve_path(training["paths"]["raw"]) / sport
    return FETCHERS[sport](training["sources"][sport], training["fetch"], raw_dir, **kwargs)


def convert(sport: str, config: Config, force: bool = False, items: list[str] | None = None) -> ConvertStats:
    if sport not in CONVERTERS:
        raise ValueError(f"no converter for {sport!r}; choose from {sorted(CONVERTERS)}")
    training = config.training
    return convert_dir(
        sport,
        CONVERTERS[sport],
        resolve_path(training["paths"]["raw"]) / sport,
        resolve_path(training["paths"]["events"]) / sport,
        training.get("convert", {}).get(sport, {}),
        force=force,
        items=items,
    )


__all__ = ["CONVERTERS", "FETCHERS", "ConvertStats", "FetchStats", "Fetcher", "convert", "make_fetcher"]
