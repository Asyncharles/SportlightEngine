"""Step 8, export: copy the staged, evaluated bundle to `models/<sport>/<version>/`.

The copy is reloaded with the engine's loader before it counts. `CURRENT` is never touched here: it
moves to a new version only in a PR the ML lead approves. A bundle that missed a target is not
exported unless forced, since the sport keeps serving the heuristic.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from intensity.config import Config, resolve_path
from intensity.engine.registry import load_bundle
from intensity.telemetry import log_event
from intensity.training.model import staged

log = logging.getLogger(__name__)

BUNDLE_FILES = ("model.txt", "calibrator.json", "meta.json")


def export(config: Config, sport: str, version: str | None = None, force: bool = False) -> Path:
    stage = staged(config, sport, version)
    meta = json.loads((stage / "meta.json").read_text())
    metrics = meta.get("metrics", {})
    if "test" not in metrics:
        raise RuntimeError(f"{stage} has not been evaluated: run `evaluate --sport {sport}` first")
    if not metrics.get("passed") and not force:
        raise RuntimeError(
            f"{sport} {meta['version']} missed a target ({metrics.get('targets')}); the heuristic stays."
            " Use --force to export it anyway."
        )
    out = resolve_path(config.training["paths"]["models"]) / sport / meta["version"]
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for name in BUNDLE_FILES:
        shutil.copy2(stage / name, tmp / name)
    load_bundle(sport, tmp)  # the engine must accept it before it lands
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    log_event(
        log, "export done", sport=sport, version=meta["version"], path=str(out), passed=metrics.get("passed")
    )
    return out
