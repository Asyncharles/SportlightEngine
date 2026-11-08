"""Step 7, evaluate on test, and the report the ML lead signs off.

Checks, all on test matches through the staged bundle exactly as the engine loads it:

- PR-AUC against the base rate (target: `evaluate.target_pr_auc_mult` times) and lift in the top
  `top_frac` of ticks; the adapter's heuristic on the same ticks for comparison;
- a reliability curve and Brier score;
- global SHAP: mean absolute `pred_contrib` for the top features, read for leakage;
- the highlight check: the `peaks` highest forecast peaks in each demo match, `peak_gap_s` apart,
  against `data/highlights/<match_id>.yaml`. A peak is a hit if a listed highlight comes 0 to
  `hit_window_s` after it in the same period; the target is `target_hits`;
- a per-match timeline of p with the highlights marked.

Writes `data/reports/training/<sport>/<version>/report.html` and `report.json`, and adds the test metrics
to the staged `meta.json`.

In `training.split.mode: in_sample`, "test" is the training data itself: every metric is in-sample, both
files say so (`"evaluation": "in-sample"` and a banner), and none of it measures generalisation.
"""

from __future__ import annotations

import html
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from intensity.config import Config, read_yaml, resolve_path
from intensity.engine.registry import Bundle, load_bundle
from intensity.telemetry import log_event
from intensity.training.featurize import tables_dir
from intensity.training.model import load_dataset, staged
from intensity.training.split import load_split

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- highlights


@dataclass
class Highlight:
    period: int
    game_time_s: float
    what: str


def parse_clock(clock: str | int | float) -> float:
    """Broadcast clock "mm:ss" (minutes may pass 90) to game seconds."""
    if isinstance(clock, int | float):
        return float(clock)
    minutes, seconds = str(clock).split(":")
    return int(minutes) * 60 + float(seconds)


def load_highlights(path: Path) -> list[Highlight]:
    data = read_yaml(path)
    return [
        Highlight(period=int(h["period"]), game_time_s=parse_clock(h["clock"]), what=str(h.get("what", "")))
        for h in data.get("highlights", [])
    ]


def find_peaks(period: np.ndarray, t: np.ndarray, p: np.ndarray, n: int, gap_s: float) -> list[int]:
    """Indices of the n highest p, each at least gap_s from every other in the same period."""
    picked: list[int] = []
    for i in np.argsort(-p, kind="stable"):
        if all(period[j] != period[i] or abs(t[j] - t[i]) >= gap_s for j in picked):
            picked.append(int(i))
            if len(picked) == n:
                break
    return sorted(picked, key=lambda i: (period[i], t[i]))


def score_peaks(
    peaks: list[tuple[int, float]], highlights: list[Highlight], window_s: float
) -> list[Highlight | None]:
    """For each (period, time) peak, the first highlight 0 to window_s after it, or None."""
    out: list[Highlight | None] = []
    for period, t in peaks:
        ahead = [h for h in highlights if h.period == period and 0.0 <= h.game_time_s - t <= window_s]
        out.append(min(ahead, key=lambda h: h.game_time_s) if ahead else None)
    return out


# --------------------------------------------------------------------------- metrics


def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, p)) if y.any() else float("nan")


def reliability(y: np.ndarray, p: np.ndarray, bins: int) -> list[dict[str, float]]:
    """Quantile bins of p: mean p against the observed rate."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, max(len(edges) - 2, 0))
    out = []
    for b in range(max(len(edges) - 1, 1)):
        mask = idx == b
        if mask.any():
            out.append({"mean_p": float(p[mask].mean()), "rate": float(y[mask].mean()), "n": int(mask.sum())})
    return out


def lift_top(y: np.ndarray, p: np.ndarray, frac: float) -> float:
    n = max(1, int(round(frac * len(p))))
    top = np.argsort(-p, kind="stable")[:n]
    base = y.mean()
    return float(y[top].mean() / base) if base else float("nan")


def predict(bundle: Bundle, x: np.ndarray) -> np.ndarray:
    """Calibrated p for many rows, the same transform `Bundle.predict` applies to one."""
    return np.interp(bundle.booster.predict(x), bundle.cal_x, bundle.cal_y)


@dataclass
class DemoCheck:
    match_id: str
    highlights_file: str | None
    peaks: list[dict[str, Any]] = field(default_factory=list)
    hits: int = 0
    highlights: list[dict[str, Any]] = field(default_factory=list)
    timeline: list[tuple[int, float, float]] = field(default_factory=list)  # (period, t, p)


def demo_check(
    match_id: str, ticks: pd.DataFrame, p: np.ndarray, highlights_dir: Path, cfg: dict[str, Any]
) -> DemoCheck:
    path = highlights_dir / f"{match_id}.yaml"
    period, t = ticks["match_period"].to_numpy(), ticks["game_time_s"].to_numpy()
    check = DemoCheck(match_id=match_id, highlights_file=str(path) if path.exists() else None)
    check.timeline = [(int(a), float(b), round(float(c), 5)) for a, b, c in zip(period, t, p, strict=True)]
    if not path.exists():
        return check
    highlights = load_highlights(path)
    check.highlights = [asdict(h) for h in highlights]
    idx = find_peaks(period, t, p, int(cfg["peaks"]), float(cfg["peak_gap_s"]))
    matched = score_peaks(
        [(int(period[i]), float(t[i])) for i in idx], highlights, float(cfg["hit_window_s"])
    )
    for i, h in zip(idx, matched, strict=True):
        check.peaks.append(
            {
                "period": int(period[i]),
                "game_time_s": float(t[i]),
                "p": float(p[i]),
                "hit": h.what if h else None,
                "lead_s": round(h.game_time_s - float(t[i]), 1) if h else None,
            }
        )
    check.hits = sum(1 for h in matched if h)
    return check


def evaluate(config: Config, sport: str, version: str | None = None) -> Path:
    stage = staged(config, sport, version)
    bundle = load_bundle(sport, stage)  # the engine's own loader, with its feature checks
    meta = json.loads((stage / "meta.json").read_text())
    data = load_dataset(config, sport)
    ecfg, hcfg = config.training["evaluate"], config.training["highlights"]
    test = data.part("test")
    x, y = test[list(data.features)].to_numpy(dtype=float), test["y"].to_numpy(dtype=int)
    p = predict(bundle, x)
    heur = test["heuristic_p"].to_numpy(dtype=float)
    base = float(y.mean())
    ap = average_precision(y, p)
    rng = np.random.default_rng(int(config.training["seed"]))
    sample = x[rng.choice(len(x), size=min(len(x), int(ecfg["shap_rows"])), replace=False)]
    shap = np.abs(np.asarray(bundle.booster.predict(sample, pred_contrib=True))[:, :-1]).mean(axis=0)
    above = p >= bundle.threshold
    metrics = {
        "evaluation": meta.get("evaluation", "held-out"),
        "test_rows": int(len(y)),
        "test_matches": int(test["match_id"].nunique()),
        "base_rate": base,
        "pr_auc": ap,
        "pr_auc_mult": ap / base if base else float("nan"),
        "heuristic_pr_auc": average_precision(y, heur),
        f"lift_top_{ecfg['top_frac']}": lift_top(y, p, float(ecfg["top_frac"])),
        "brier": float(np.mean((p - y) ** 2)),
        "heating_frac": float(above.mean()),
        "precision_at_threshold": float(y[above].mean()) if above.any() else None,
        "reliability": reliability(y, p, int(ecfg["reliability_bins"])),
        "shap_top": [
            {"feature": data.features[i], "mean_abs": round(float(shap[i]), 5)}
            for i in np.argsort(-shap)[: int(ecfg["shap_top"])]
        ],
    }

    features = pd.read_parquet(tables_dir(config, sport) / "features.parquet")
    highlights_dir = resolve_path(config.training["paths"]["highlights"])
    _, demo_ids = load_split(tables_dir(config, sport) / "split.json")
    demos = []
    for match_id in sorted(demo_ids & set(meta["matches"]["test"])):
        ticks = features[features["match_id"] == match_id]
        p_demo = predict(bundle, ticks[list(data.features)].to_numpy(dtype=float))
        demos.append(demo_check(match_id, ticks, p_demo, highlights_dir, hcfg))

    targets = {
        "pr_auc": {
            "target": float(ecfg["target_pr_auc_mult"]),
            "value": metrics["pr_auc_mult"],
            "met": bool(metrics["pr_auc_mult"] >= float(ecfg["target_pr_auc_mult"])),
        },
        "highlights": {
            "target": int(hcfg["target_hits"]),
            "value": {d.match_id: (d.hits if d.highlights_file else None) for d in demos},
            "met": bool(demos)
            and all(d.highlights_file and d.hits >= int(hcfg["target_hits"]) for d in demos),
        },
    }
    passed = all(t["met"] for t in targets.values())
    meta["metrics"]["test"] = {k: v for k, v in metrics.items() if k not in ("reliability", "shap_top")}
    meta["metrics"]["targets"] = targets
    meta["metrics"]["passed"] = passed
    (stage / "meta.json").write_text(json.dumps(meta, indent=1))

    out_dir = resolve_path(config.training["paths"]["reports"]) / sport / meta["version"]
    out_dir.mkdir(parents=True, exist_ok=True)
    search = json.loads((stage / "search.json").read_text()) if (stage / "search.json").exists() else []
    path = write_report(out_dir, meta, metrics, targets, passed, demos, search, bundle.threshold, hcfg)
    (out_dir / "report.json").write_text(
        json.dumps(
            {
                "meta": meta,
                "metrics": metrics,
                "demos": [{k: v for k, v in asdict(d).items() if k != "timeline"} for d in demos],
            },
            indent=1,
        )
    )
    log_event(
        log,
        "evaluate done",
        sport=sport,
        version=meta["version"],
        pr_auc_mult=round(metrics["pr_auc_mult"], 2),
        highlights=targets["highlights"]["value"],
        passed=passed,
        report=str(path),
    )
    return path


# --------------------------------------------------------------------------- report

CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--muted:#6e6e73;--line:#0a66c2;--moment:#c2410c;--hit:#15803d;--miss:#b91c1c;--grid:#d2d2d7;--bar:#93c5fd}
@media (prefers-color-scheme: dark){:root{--bg:#111;--fg:#f5f5f7;--muted:#a1a1a6;--line:#60a5fa;--moment:#fb923c;--hit:#4ade80;--miss:#f87171;--grid:#3a3a3c;--bar:#1e40af}}
body{font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg);max-width:960px;margin:24px auto;padding:0 16px}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid var(--grid);padding:4px 8px;text-align:right}
td:first-child,th:first-child{text-align:left}svg{width:100%;height:auto}.line{fill:none;stroke:var(--line);stroke-width:1.2}
.moment{stroke:var(--moment);stroke-dasharray:3 3}.thr{stroke:var(--muted);stroke-dasharray:6 4}.period{stroke:var(--grid)}
.hit{fill:var(--hit)}.miss{fill:var(--miss)}.ax{fill:var(--muted);font-size:11px}.note{color:var(--muted)}
.pass{color:var(--hit);font-weight:600}.fail{color:var(--miss);font-weight:600}.bar{fill:var(--bar)}.diag{stroke:var(--grid)}
"""


def _f(v: Any, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.{digits}f}" if isinstance(v, float) else html.escape(str(v))


def _clock(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def _timeline_svg(d: DemoCheck, threshold: float, width: int = 900, height: int = 180) -> str:
    if not d.timeline:
        return "<p>No ticks.</p>"
    n = len(d.timeline)
    p_max = max(max(p for _, _, p in d.timeline), threshold) * 1.1 or 1.0
    index = {(per, t): i for i, (per, t, _) in enumerate(d.timeline)}

    def x(i: float) -> float:
        return 40 + (width - 50) * i / max(n - 1, 1)

    def y(v: float) -> float:
        return height - 20 - (height - 30) * v / p_max

    def nearest(period: int, t: float) -> int | None:
        same = [(abs(tt - t), i) for (per, tt), i in index.items() if per == period]
        return min(same)[1] if same else None

    step = max(1, n // 1500)
    pts = " ".join(f"{x(i):.1f},{y(p):.1f}" for i, (_, _, p) in enumerate(d.timeline) if i % step == 0)
    bounds = "".join(
        f'<line x1="{x(i):.1f}" x2="{x(i):.1f}" y1="10" y2="{height - 20}" class="period"/>'
        for i in range(1, n)
        if d.timeline[i][0] != d.timeline[i - 1][0]
    )
    marks = ""
    for h in d.highlights:
        i = nearest(h["period"], h["game_time_s"])
        if i is not None:
            marks += (
                f'<line x1="{x(i):.1f}" x2="{x(i):.1f}" y1="10" y2="{height - 20}" class="moment">'
                f"<title>{html.escape(h['what'])}</title></line>"
            )
    dots = ""
    for pk in d.peaks:
        i = index.get((pk["period"], pk["game_time_s"]))
        if i is not None:
            dots += f'<circle cx="{x(i):.1f}" cy="{y(pk["p"]):.1f}" r="4" class="{"hit" if pk["hit"] else "miss"}"/>'
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="forecast timeline for {html.escape(d.match_id)}">'
        f'{bounds}<line x1="40" x2="{width - 10}" y1="{y(threshold):.1f}" y2="{y(threshold):.1f}" class="thr"/>'
        f'{marks}<polyline points="{pts}" class="line"/>{dots}'
        f'<text x="4" y="{y(p_max):.0f}" class="ax">{p_max:.2f}</text><text x="4" y="{y(0):.0f}" class="ax">0</text></svg>'
    )


def _reliability_svg(rows: list[dict[str, float]], size: int = 260) -> str:
    top = max([1e-6] + [max(r["mean_p"], r["rate"]) for r in rows]) * 1.1

    def s(v: float) -> float:
        return 30 + (size - 40) * v / top

    pts = "".join(
        f'<circle cx="{s(r["mean_p"]):.1f}" cy="{size - s(r["rate"]):.1f}" r="4" class="hit"/>' for r in rows
    )
    line = " ".join(f"{s(r['mean_p']):.1f},{size - s(r['rate']):.1f}" for r in rows)
    return (
        f'<svg viewBox="0 0 {size} {size}" style="max-width:320px" role="img" aria-label="reliability curve">'
        f'<line x1="{s(0)}" y1="{size - s(0)}" x2="{s(top)}" y2="{size - s(top)}" class="diag"/>'
        f'<polyline points="{line}" class="line"/>{pts}'
        f'<text x="{size - 70}" y="{size - 4}" class="ax">predicted p</text><text x="2" y="12" class="ax">observed</text></svg>'
    )


def write_report(
    out_dir: Path,
    meta: dict[str, Any],
    m: dict[str, Any],
    targets: dict[str, Any],
    passed: bool,
    demos: list[DemoCheck],
    search: list[dict[str, Any]],
    threshold: float,
    hcfg: dict[str, Any],
) -> Path:
    def verdict(ok: bool) -> str:
        return '<span class="pass">met</span>' if ok else '<span class="fail">missed</span>'

    hl = targets["highlights"]
    hl_value = (
        ", ".join(f"{k}: {v if v is not None else 'no list'}" for k, v in hl["value"].items())
        or "no demo match"
    )
    lift_key = next(k for k in m if k.startswith("lift_top_"))
    shap_max = max([s["mean_abs"] for s in m["shap_top"]] + [1e-9])
    shap_rows = "".join(
        f'<tr><td>{html.escape(s["feature"])}</td><td>{s["mean_abs"]:.4f}</td><td style="width:50%">'
        f'<svg viewBox="0 0 100 8" preserveAspectRatio="none" style="height:10px"><rect class="bar" width="{100 * s["mean_abs"] / shap_max:.1f}" height="8"/></svg></td></tr>'
        for s in m["shap_top"]
    )
    demo_html = ""
    for d in demos:
        demo_html += f"<h3>{html.escape(d.match_id)}</h3>"
        if not d.highlights_file:
            demo_html += f'<p class="fail">No highlight list at data/highlights/{html.escape(d.match_id)}.yaml, so the check did not run.</p>'
        else:
            rows = "".join(
                f"<tr><td>{pk['period']}</td><td>{_clock(pk['game_time_s'])}</td><td>{pk['p']:.3f}</td>"
                f"<td>{html.escape(pk['hit']) if pk['hit'] else '<span class=fail>miss</span>'}</td><td>{_f(pk['lead_s'], 0)}</td></tr>"
                for pk in d.peaks
            )
            demo_html += (
                f"<p>{d.hits} of {len(d.peaks)} peaks hit a highlight (target {hcfg['target_hits']}).</p>"
                f"<table><tr><th>Period</th><th>Clock</th><th>p</th><th>Highlight</th><th>Lead s</th></tr>{rows}</table>"
            )
        demo_html += _timeline_svg(d, threshold)
    trials = "".join(
        "<tr><td>"
        + html.escape(", ".join(f"{k}={v}" for k, v in t["params"].items()))
        + f"</td><td>{t['val_ap']:.4f}</td><td>{t['best_iteration']}</td><td>{t['seconds']}</td></tr>"
        for t in sorted(search, key=lambda t: -t["val_ap"])
    )
    val = meta["metrics"]
    in_sample = meta.get("evaluation") == "in-sample"
    where = "in-sample" if in_sample else "test"
    banner = (
        '<p class="fail">In-sample metrics: this model was trained, calibrated and evaluated on the same'
        " matches (training.split.mode: in_sample), demo matches included. No number below is held out,"
        " so none of them says how the model does on unseen matches.</p>"
        if in_sample
        else ""
    )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Forecast report</title><style>{
        CSS
    }</style></head><body>
<h1>Forecast: {html.escape(meta["sport"])} <code>{html.escape(meta["version"])}</code></h1>
<p>Label <code>{html.escape(meta["label"])}</code>, horizon {meta["horizon_s"]:.0f} s. Git <code>{
        html.escape(meta["git_sha"])
    }</code>,
data <code>{html.escape(meta["data_hash"])}</code>, created {html.escape(meta["created_at"])}.</p>
{banner}
<p class="{"pass" if passed else "fail"}">{
        "Every target met: this bundle can be proposed for CURRENT."
        if passed
        else "A target was missed: the sport keeps serving the heuristic until a bundle meets every target."
    }</p>
<h2>Targets</h2><table><tr><th>Check</th><th>Target</th><th>Value</th><th></th></tr>
<tr><td>PR-AUC / base rate ({where})</td><td>&ge; {targets["pr_auc"]["target"]}</td><td>{
        _f(targets["pr_auc"]["value"], 2)
    }</td><td>{verdict(targets["pr_auc"]["met"])}</td></tr>
<tr><td>Highlight hits per demo match</td><td>&ge; {hl["target"]} of {hcfg["peaks"]}</td><td>{
        html.escape(hl_value)
    }</td><td>{verdict(hl["met"])}</td></tr></table>
<h2>{where.capitalize()} metrics</h2><table><tr><th>Metric</th><th>Value</th></tr>
<tr><td>Matches / rows</td><td>{m["test_matches"]} / {m["test_rows"]}</td></tr>
<tr><td>Base rate</td><td>{_f(m["base_rate"], 4)}</td></tr>
<tr><td>PR-AUC (model)</td><td>{_f(m["pr_auc"], 4)}</td></tr>
<tr><td>PR-AUC (heuristic, same ticks)</td><td>{_f(m["heuristic_pr_auc"], 4)}</td></tr>
<tr><td>Lift in the top {float(lift_key.split("_")[-1]):.0%} of ticks</td><td>{_f(m[lift_key], 2)}</td></tr>
<tr><td>Brier</td><td>{_f(m["brier"], 4)}</td></tr>
<tr><td>Heating-up threshold (calibrated p)</td><td>{_f(threshold, 4)}{
        ""
        if val["threshold_reached"]
        else ' <span class="fail">(precision target not reached on {"the training data" if in_sample else "validation"}; never heats up)</span>'
    }</td></tr>
<tr><td>Ticks at or above it / precision there</td><td>{_f(m["heating_frac"], 3)} / {
        _f(m["precision_at_threshold"], 3)
    }</td></tr>
<tr><td>{"Training" if in_sample else "Validation"} PR-AUC / base rate</td><td>{_f(val["val_average_precision"], 4)} / {
        _f(val["val_base_rate"], 4)
    }</td></tr></table>
<h2>Reliability</h2>{_reliability_svg(m["reliability"])}
<h2>Global SHAP (top {
        len(m["shap_top"])
    }, read for leakage)</h2><table><tr><th>Feature</th><th>mean |contrib|</th><th></th></tr>{
        shap_rows
    }</table>
<h2>Highlight check and timelines</h2>
<p class="note">Blue: calibrated p. Dashed grey: threshold. Dashed orange: highlights. Dots: peaks, green if a highlight follows within {
        hcfg["hit_window_s"]
    } s, red if not. Thin lines: period changes.</p>
{demo_html or f'<p class="fail">No demo match in {where}.</p>'}
<h2>Search</h2><table><tr><th>Setting</th><th>{"in-sample AP" if in_sample else "val AP"}</th><th>iterations</th><th>s</th></tr>{trials}</table>
<h2>Split ({html.escape(meta.get("split_mode", "holdout"))})</h2><p>{
        ", ".join(f"{k}: {len(v)} matches, {meta['rows'][k]} rows" for k, v in meta["matches"].items())
    }.</p>
</body></html>"""
    path = out_dir / "report.html"
    path.write_text(page)
    return path
