"""Steps 5 and 6, train and calibrate, staged in `run/training/<sport>/<version>/`.

Train: LightGBM binary with `scale_pos_weight` = negatives / positives, early stopping on validation
average precision, and a capped search over `training.search.grid` (at most `max_settings` settings,
sampled with the seed). The best setting by validation average precision is kept.

In `training.split.mode: in_sample` there is no validation set: one fixed setting
(`training.in_sample.params`) trains for `training.in_sample.num_boost_round` rounds on every match, and
"validation" below means that same training data. Choosing settings by in-sample score would only
pick the most overfit one, so there is no search.

Calibrate: isotonic regression on the kept model's validation predictions, saved as `calibrator.json`
(`x` raw score, `y` calibrated p; the engine interpolates). The points sit at each isotonic block's
centre, so p keeps the model's ranking instead of tying inside blocks. The "heating up" threshold is the lowest
calibrated p whose validation precision reaches `precision_mult` times the base rate, raised if needed
so at most `max_heating_frac` of validation ticks qualify. `p_quantiles` is the validation distribution
of calibrated p, which maps p to the forecast part's 0 to 1 scale.

The staged directory is the bundle in progress: `evaluate` adds test metrics and `export` copies it to
`models/<sport>/<version>/`. `run/training/<sport>/LATEST` names the newest one.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from intensity.config import REPO_ROOT, Config, resolve_path, save_resolved
from intensity.engine.adapters import ADAPTERS
from intensity.engine.adapters.base import feature_hash
from intensity.telemetry import log_event
from intensity.training.featurize import KEYS, read_meta, tables_dir
from intensity.training.labels import label, label_cfg
from intensity.training.split import is_in_sample, load_split, members, split

log = logging.getLogger(__name__)


def _lightgbm() -> Any:
    try:
        import lightgbm
    except (ImportError, OSError) as exc:  # OSError: libomp missing on macOS
        raise RuntimeError(f"lightgbm unavailable ({exc}); on macOS: brew install libomp") from exc
    return lightgbm


# --------------------------------------------------------------------------- data


@dataclass
class Dataset:
    sport: str
    features: tuple[str, ...]
    table: pd.DataFrame  # keys, heuristic_p, features, y, part
    in_sample: bool = False  # split mode in_sample: every part is the whole table

    def part(self, name: str) -> pd.DataFrame:
        if self.in_sample:
            return self.table
        return self.table[self.table["part"] == name]

    def xy(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        df = self.part(name)
        return df[list(self.features)].to_numpy(dtype=float), df["y"].to_numpy(dtype=int)


def load_dataset(config: Config, sport: str) -> Dataset:
    tables = tables_dir(config, sport)
    features = ADAPTERS[sport].FEATURES
    x = pd.read_parquet(tables / "features.parquet")
    y = pd.read_parquet(tables / "labels.parquet")
    parts, _ = load_split(tables / "split.json")
    table = x.merge(y, on=KEYS, how="inner", validate="one_to_one")
    table["part"] = table["match_id"].map(parts)
    if table["part"].isna().any():
        raise ValueError(
            f"matches missing from split.json: {sorted(table.loc[table['part'].isna(), 'match_id'].unique())}"
        )
    return Dataset(sport=sport, features=features, table=table, in_sample=is_in_sample(parts))


# --------------------------------------------------------------------------- train


def search_settings(cfg: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    grid = cfg["grid"]
    names = sorted(grid)
    combos = [
        dict(zip(names, values, strict=True)) for values in itertools.product(*(grid[n] for n in names))
    ]
    n = min(int(cfg.get("max_settings", 20)), len(combos))
    picks = np.random.default_rng(seed).choice(len(combos), size=n, replace=False)
    return [combos[i] for i in sorted(picks)]


@dataclass
class Trial:
    params: dict[str, Any]
    val_ap: float
    best_iteration: int
    seconds: float


def train_model(
    x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, y_va: np.ndarray, cfg: dict[str, Any], seed: int
) -> tuple[Any, Trial, list[Trial]]:
    lgb = _lightgbm()
    pos = max(int(y_tr.sum()), 1)
    base = {
        "objective": "binary",
        "metric": "average_precision",
        "scale_pos_weight": (len(y_tr) - pos) / pos,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }
    # No pre-filtering: the search changes min_data_in_leaf on the same Dataset.
    train_set = lgb.Dataset(x_tr, y_tr, params={"feature_pre_filter": False}, free_raw_data=False)
    val_set = lgb.Dataset(
        x_va, y_va, reference=train_set, params={"feature_pre_filter": False}, free_raw_data=False
    )
    best: tuple[Any, Trial] | None = None
    trials: list[Trial] = []
    for setting in search_settings(cfg, seed):
        started = time.monotonic()
        booster = lgb.train(
            {**base, **setting},
            train_set,
            num_boost_round=int(cfg.get("num_boost_round", 2000)),
            valid_sets=[val_set],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(int(cfg.get("early_stopping_rounds", 100)), verbose=False)],
        )
        trial = Trial(
            params=setting,
            val_ap=float(booster.best_score["val"]["average_precision"]),
            best_iteration=int(booster.best_iteration),
            seconds=round(time.monotonic() - started, 1),
        )
        trials.append(trial)
        log_event(log, "trial", **setting, val_ap=round(trial.val_ap, 4), iterations=trial.best_iteration)
        if best is None or trial.val_ap > best[1].val_ap:
            best = (booster, trial)
    assert best is not None
    return best[0], best[1], trials


def train_in_sample(x: np.ndarray, y: np.ndarray, cfg: dict[str, Any], seed: int) -> tuple[Any, Trial, list[Trial]]:
    """One fixed setting on all the data, no early stopping (split mode in_sample)."""
    from sklearn.metrics import average_precision_score

    lgb = _lightgbm()
    pos = max(int(y.sum()), 1)
    params = {
        "objective": "binary",
        "scale_pos_weight": (len(y) - pos) / pos,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
        **cfg["params"],
    }
    rounds = int(cfg["num_boost_round"])
    started = time.monotonic()
    booster = lgb.train(params, lgb.Dataset(x, y, free_raw_data=False), num_boost_round=rounds)
    trial = Trial(
        params=dict(cfg["params"]),
        val_ap=float(average_precision_score(y, booster.predict(x))),
        best_iteration=rounds,
        seconds=round(time.monotonic() - started, 1),
    )
    log_event(log, "trial", **trial.params, in_sample_ap=round(trial.val_ap, 4), iterations=rounds)
    return booster, trial, [trial]


# --------------------------------------------------------------------------- calibrate


@dataclass
class Calibration:
    x: list[float]
    y: list[float]
    threshold: float
    p_quantiles: list[float]
    val_base_rate: float
    val_precision_at_threshold: float | None
    val_heating_frac: float
    threshold_reached: bool

    def apply(self, raw: np.ndarray) -> np.ndarray:
        return np.interp(raw, self.x, self.y)


def heating_threshold(
    p: np.ndarray, y: np.ndarray, precision_mult: float, max_frac: float
) -> tuple[float, bool]:
    """Lowest calibrated p with precision >= mult x base rate and at most max_frac of ticks at or above it.

    Returns (threshold, reached). When no level reaches the precision, the threshold sits above every
    validation p, so the forecast never heats up on its own, and `reached` is False.
    """
    base = float(y.mean())
    order = np.argsort(-p, kind="stable")
    p_sorted, y_sorted = p[order], y[order]
    levels = np.unique(p_sorted)[::-1]  # descending
    hits = np.cumsum(y_sorted)
    best: float | None = None
    for level in levels:
        n = int(np.searchsorted(-p_sorted, -level, side="right"))  # ticks with p >= level
        if n / len(p) > max_frac:
            break
        if hits[n - 1] / n >= precision_mult * base:
            best = float(level)
    if best is None:
        return float(p.max()) + 1e-6, False  # may exceed 1: then nothing ever qualifies
    return best, True


def block_centres(raw_sorted: np.ndarray, fitted: np.ndarray) -> tuple[list[float], list[float]]:
    """One point per isotonic block, at the block's mean raw score.

    Plain isotonic output is flat inside each block, so many ticks tie and ranking (PR-AUC, the
    highlight peaks) degrades. Interpolating between block centres keeps the mapping monotone and
    the block levels, and only stays flat beyond the first and last centre.
    """
    starts = np.flatnonzero(np.r_[True, np.diff(fitted) != 0])
    ends = np.r_[starts[1:], len(fitted)]
    xs: list[float] = []
    ys: list[float] = []
    for a, b in zip(starts, ends, strict=True):
        x = round(float(raw_sorted[a:b].mean()), 7)
        if xs and x <= xs[-1]:
            continue  # rounding merged two neighbouring blocks
        xs.append(x)
        ys.append(round(float(fitted[a]), 7))
    return xs, ys


def calibrate(raw_va: np.ndarray, y_va: np.ndarray, cfg: dict[str, Any]) -> Calibration:
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw_va, y_va)
    cal_x, cal_y = block_centres(np.sort(raw_va), iso.predict(np.sort(raw_va)))
    p = np.interp(raw_va, cal_x, cal_y)
    threshold, reached = heating_threshold(
        p, y_va, float(cfg.get("precision_mult", 3.0)), float(cfg.get("max_heating_frac", 0.15))
    )
    above = p >= threshold
    q = np.quantile(p, np.linspace(0.0, 1.0, int(cfg.get("n_quantiles", 21))))
    return Calibration(
        x=cal_x,
        y=cal_y,
        threshold=round(threshold, 7),
        p_quantiles=[round(float(v), 7) for v in q],
        val_base_rate=float(y_va.mean()),
        val_precision_at_threshold=float(y_va[above].mean()) if above.any() else None,
        val_heating_frac=float(above.mean()),
        threshold_reached=reached,
    )


# --------------------------------------------------------------------------- staging


def git_sha() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def runs_dir(config: Config, sport: str) -> Path:
    return resolve_path(config.training["paths"]["runs"]) / sport


def staged(config: Config, sport: str, version: str | None = None) -> Path:
    root = runs_dir(config, sport)
    if version is None:
        latest = root / "LATEST"
        if not latest.exists():
            raise FileNotFoundError(f"no staged {sport} bundle: run `train --sport {sport}`")
        version = latest.read_text().strip()
    return root / version


def train(config: Config, sport: str) -> Path:
    label_path = label(config, sport)
    split_path = split(config, sport)
    data = load_dataset(config, sport)
    seed = int(config.training["seed"])
    x_tr, y_tr = data.xy("train")
    x_va, y_va = data.xy("val")
    if not y_tr.any() or not y_va.any():
        raise ValueError(f"{sport}: no positives in training or validation; check the label")
    if data.in_sample:
        booster, best, trials = train_in_sample(x_tr, y_tr, config.training["in_sample"], seed)
    else:
        booster, best, trials = train_model(x_tr, y_tr, x_va, y_va, config.training["search"], seed)
    cal = calibrate(booster.predict(x_va), y_va, config.training["calibration"])

    parts = json.loads(split_path.read_text())["parts"]
    data_hash = read_meta(label_path.parent / "features.meta.json")["inputs"]["data_hash"]
    sha = git_sha()
    fingerprint = json.dumps(
        [
            data_hash,
            read_meta(label_path.with_name("labels.meta.json"))["inputs"],
            parts,
            config.training["in_sample"] if data.in_sample else config.training["search"],
            config.training["calibration"],
            sha,
        ],
        sort_keys=True,
    )
    version = f"{time.strftime('%Y%m%d')}-{hashlib.sha256(fingerprint.encode()).hexdigest()[:6]}"
    out = runs_dir(config, sport) / version
    out.mkdir(parents=True, exist_ok=True)
    save_resolved(config, out)
    booster.save_model(str(out / "model.txt"), num_iteration=best.best_iteration)
    (out / "calibrator.json").write_text(json.dumps({"x": cal.x, "y": cal.y}))
    lcfg = label_cfg(config, sport)
    meta = {
        "sport": sport,
        "version": version,
        "features": list(data.features),
        "feature_hash": feature_hash(data.features),
        "adapter_config_hash": read_meta(label_path.parent / "features.meta.json")["inputs"][
            "adapter_config_hash"
        ],
        "label": lcfg["name"],
        "label_cfg": lcfg,
        "horizon_s": float(lcfg["horizon_s"]),
        "threshold": cal.threshold,
        "p_quantiles": cal.p_quantiles,
        "split_mode": "in_sample" if data.in_sample else "holdout",
        # In-sample: trained, calibrated and evaluated on the same matches, so no metric is held out.
        "evaluation": "in-sample" if data.in_sample else "held-out",
        "matches": {name: members(parts, name) for name in ("train", "val", "test")},
        "rows": {name: int(len(data.part(name))) for name in ("train", "val", "test")},
        "params": best.params,
        "best_iteration": best.best_iteration,
        "metrics": {
            "val_average_precision": best.val_ap,
            "val_base_rate": cal.val_base_rate,
            "val_precision_at_threshold": cal.val_precision_at_threshold,
            "val_heating_frac": cal.val_heating_frac,
            "threshold_reached": cal.threshold_reached,
        },
        "data_hash": data_hash,
        "git_sha": sha,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / "search.json").write_text(json.dumps([t.__dict__ for t in trials], indent=1))
    (runs_dir(config, sport) / "LATEST").write_text(version + "\n")
    log_event(
        log, "train done", sport=sport, version=version, val_ap=round(best.val_ap, 4), threshold=cal.threshold
    )
    return out
