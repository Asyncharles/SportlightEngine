"""Imminent check (P4): replay event tapes through the engine on CPU and score the warnings.

    make eval-imminent SPORT=football SET=tuning

* Moments come from match data (a football goal or a shot with xG >= 0.1, an F1 on-track pass, an
  NBA lead change, a weightlifting lift start) and from just-happened cues (a crowd roar, a replay
  starting, a celebration prompt leading the others by LEAD_MARGIN). Moments within 10 s merge, at the earliest time.
* A spike is each time `imminent.heating_up` turns true, at least 10 s after the last. A spike is a
  hit if it comes 2 to 10 s before a moment.
* Reported per clip and overall: precision, recall, median lead time, false alarms per 10 minutes.

Tuning only ever reads `SET=tuning`; demo clips are scored once per rule version and never tuned
on (hard rule 6), so `SET=tuning` refuses any tape whose clip is a demo feed.
"""

from __future__ import annotations

import argparse
import html
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from intensity.config import Config, load_config, load_sport_yaml, resolve_path, save_resolved
from intensity.engine import simulate
from intensity.engine.core import Engine
from intensity.feeds import load_feeds
from intensity.schema import Envelope, Event

MERGE_S = 10.0
LEAD_MIN_S, LEAD_MAX_S = 2.0, 10.0
SPIKE_GAP_S = 10.0
ROAR_PCT, ROAR_MIN_S = 95.0, 1.0
LEAD_MARGIN = 0.1  # a celebration prompt "leads" when it beats the next prompt by this much

TARGETS = {"precision": 0.4, "median_lead_s": 2.0, "false_alarms_per_10min": 2.0}  # D14


@dataclass
class ClipResult:
    clip_id: str
    duration_s: float
    moments: list[float]
    moment_sources: list[str]
    spikes: list[float]
    hits: list[tuple[float, float]]  # (spike, moment)
    timeline: list[tuple[float, float]] = field(default_factory=list)  # (t_video, imminent)

    @property
    def precision(self) -> float | None:
        return len(self.hits) / len(self.spikes) if self.spikes else None

    @property
    def recall(self) -> float | None:
        if not self.moments:
            return None
        return len({m for _, m in self.hits}) / len(self.moments)

    @property
    def false_alarms(self) -> int:
        hit_spikes = {s for s, _ in self.hits}
        return sum(1 for s in self.spikes if s not in hit_spikes)

    @property
    def leads(self) -> list[float]:
        return [m - s for s, m in self.hits]


# --------------------------------------------------------------------------- labels


def match_moments(events: list[Event], sport: str) -> list[float]:
    out = []
    home = away = 0
    leader = 0
    for ev in events:
        p = ev.payload
        kind, attrs = p["kind"], p.get("attrs", {})
        if sport == "football" and (kind == "goal" or (kind == "shot" and float(attrs.get("xg", 0)) >= 0.1)) or sport == "f1" and kind == "pass" and not attrs.get("pit_swap"):
            out.append(ev.t_video)
        elif sport == "nba" and kind == "score":
            home, away = int(attrs.get("home_score", home)), int(attrs.get("away_score", away))
            now = (home > away) - (home < away)
            if now != leader and (now == 0 or leader != 0 or home + away > 0):
                if leader != 0:
                    out.append(ev.t_video)
                leader = now
        elif sport == "weightlifting" and kind == "attempt_called" and attrs.get("eta_game_s") is not None:
            out.append(float(attrs["eta_game_s"]))  # identity sync on tapes: game time = t_video
    return out


def cue_moments(events: list[Event], celebration: list[str]) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    crowd = [e for e in events if e.source == "crowd"]
    if crowd:
        levels = np.array([float(e.payload["loudness_rel_db"]) for e in crowd])
        thr = float(np.percentile(levels, ROAR_PCT))
        start = None
        for e in crowd:
            loud = float(e.payload["loudness_rel_db"]) > thr
            ev = (e.payload.get("events") or {}).get("cheering", 0.0) > 0.5
            if loud or ev:
                start = e.t_video if start is None else start
                if e.t_video - start >= ROAR_MIN_S or ev:
                    out.append((start, "roar"))
            else:
                start = None
    was_replay = False
    for e in (e for e in events if e.source == "video"):
        replay = bool(e.payload["replay"])
        if replay and not was_replay:
            out.append((e.t_video, "replay"))
        was_replay = replay
        prompts = e.payload["prompts"]
        if len(prompts) >= 2 and celebration:
            ranked = sorted(prompts.items(), key=lambda kv: -kv[1])
            (top, best), (_, second) = ranked[0], ranked[1]
            if top in celebration and best - second >= LEAD_MARGIN:
                out.append((e.t_video, "celebration"))
    return out


def merge(moments: list[tuple[float, str]]) -> list[tuple[float, str]]:
    merged: list[tuple[float, str]] = []
    for t, src in sorted(moments):
        if merged and t - merged[-1][0] <= MERGE_S:
            continue
        merged.append((t, src))
    return merged


def spikes_and_timeline(emitted: list[simulate.Emitted], feed_id: str) -> tuple[list[float], list[tuple[float, float]]]:
    spikes, timeline = [], []
    was = False
    for e in emitted:
        u = e.update
        if u.feed_id != feed_id or u.loop_epoch != 0:
            continue
        timeline.append((u.t_video, u.imminent.value))
        on = u.imminent.heating_up
        if on and not was and (not spikes or u.t_video - spikes[-1] >= SPIKE_GAP_S):
            spikes.append(u.t_video)
        was = on
    return spikes, timeline


def score_hits(spikes: list[float], moments: list[float]) -> list[tuple[float, float]]:
    hits = []
    for s in spikes:
        cands = [m for m in moments if LEAD_MIN_S <= m - s <= LEAD_MAX_S]
        if cands:
            hits.append((s, min(cands)))
    return hits


# --------------------------------------------------------------------------- run


def evaluate_tape(config: Config, envelopes: list[Envelope], sport: str, clip_id: str) -> ClipResult:
    events = [e.data for e in envelopes if isinstance(e.data, Event) and e.data.loop_epoch == 0]
    feed_ids = sorted({e.feed_id for e in events if e.sport == sport})
    if not feed_ids:
        return ClipResult(clip_id, 0.0, [], [], [], [])
    feed_id = feed_ids[0]
    events = [e for e in events if e.feed_id == feed_id]
    engine = Engine(config, [])
    emitted = simulate.run(engine, [e for e in envelopes if e.data.feed_id == feed_id])
    celebration = config.engine.trackers.video.celebration_prompts.get(sport, [])
    labelled = [(t, "match") for t in match_moments([e for e in events if e.source == "match"], sport)]
    labelled += cue_moments(events, celebration)
    moments = merge(labelled)
    spikes, timeline = spikes_and_timeline(emitted, feed_id)
    times = [t for t, _ in moments]
    duration = max((e.t_video for e in events), default=0.0)
    return ClipResult(clip_id, duration, times, [s for _, s in moments], spikes, score_hits(spikes, times), timeline)


def summarise(results: list[ClipResult]) -> dict:
    spikes = sum(len(r.spikes) for r in results)
    hits = sum(len(r.hits) for r in results)
    moments = sum(len(r.moments) for r in results)
    recalled = sum(len({m for _, m in r.hits}) for r in results)
    leads = [lead for r in results for lead in r.leads]
    minutes = sum(r.duration_s for r in results) / 60.0
    fa = sum(r.false_alarms for r in results)
    out = {
        "clips": len(results),
        "minutes": round(minutes, 1),
        "moments": moments,
        "spikes": spikes,
        "hits": hits,
        "precision": round(hits / spikes, 3) if spikes else None,
        "recall": round(recalled / moments, 3) if moments else None,
        "median_lead_s": round(statistics.median(leads), 2) if leads else None,
        "false_alarms_per_10min": round(fa / minutes * 10, 2) if minutes else None,
    }
    out["meets_targets"] = (
        out["precision"] is not None
        and out["precision"] >= TARGETS["precision"]
        and (out["median_lead_s"] or 0) >= TARGETS["median_lead_s"]
        and (out["false_alarms_per_10min"] or 0) <= TARGETS["false_alarms_per_10min"]
    )
    return out


# --------------------------------------------------------------------------- report


def _svg(r: ClipResult, threshold: float, width: int = 900, height: int = 160) -> str:
    if not r.timeline:
        return "<p>No updates.</p>"
    t_max = max(r.duration_s, r.timeline[-1][0], 1.0)

    def x(t: float) -> float:
        return 40 + (width - 50) * t / t_max

    def y(v: float) -> float:
        return height - 20 - (height - 30) * v

    pts = " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in r.timeline)
    marks = "".join(f'<line x1="{x(m):.1f}" x2="{x(m):.1f}" y1="10" y2="{height - 20}" class="moment"/>' for m in r.moments)
    spikes = "".join(f'<circle cx="{x(s):.1f}" cy="{y(threshold):.1f}" r="4" class="{"hit" if any(s == h for h, _ in r.hits) else "miss"}"/>' for s in r.spikes)
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="imminent timeline for {html.escape(r.clip_id)}">'
        f'<line x1="40" x2="{width - 10}" y1="{y(threshold):.1f}" y2="{y(threshold):.1f}" class="thr"/>'
        f"{marks}<polyline points=\"{pts}\" class=\"line\"/>{spikes}"
        f'<text x="4" y="{y(1):.0f}" class="ax">1</text><text x="4" y="{y(0):.0f}" class="ax">0</text>'
        f'<text x="{width - 60}" y="{height - 4}" class="ax">{t_max:.0f} s</text></svg>'
    )


def _fmt(v: float | None, digits: int = 2) -> str:
    return "" if v is None else f"{v:.{digits}f}"


def _row(r: ClipResult) -> str:
    lead = statistics.median(r.leads) if r.leads else None
    cells = [html.escape(r.clip_id), f"{r.duration_s / 60:.1f}", str(len(r.moments)), str(len(r.spikes)),
             str(len(r.hits)), _fmt(r.precision), _fmt(r.recall), _fmt(lead, 1)]
    return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"


def write_report(out_dir: Path, sport: str, set_name: str, rule_version: str, threshold: float,
                 results: list[ClipResult], summary: dict, note: str | None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = "".join(_row(r) for r in results)
    charts = "".join(f"<h3>{html.escape(r.clip_id)}</h3>{_svg(r, threshold)}" for r in results)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Imminent report</title>
<style>
:root{{--bg:#fff;--fg:#1d1d1f;--muted:#6e6e73;--line:#0a66c2;--moment:#c2410c;--hit:#15803d;--miss:#b91c1c;--grid:#d2d2d7}}
@media (prefers-color-scheme: dark){{:root{{--bg:#111;--fg:#f5f5f7;--muted:#a1a1a6;--line:#60a5fa;--moment:#fb923c;--hit:#4ade80;--miss:#f87171;--grid:#3a3a3c}}}}
body{{font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg);max-width:960px;margin:24px auto;padding:0 16px}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid var(--grid);padding:4px 8px;text-align:right}}td:first-child,th:first-child{{text-align:left}}
svg{{width:100%;height:auto}}.line{{fill:none;stroke:var(--line);stroke-width:1.5}}.moment{{stroke:var(--moment);stroke-dasharray:3 3}}
.thr{{stroke:var(--muted);stroke-dasharray:6 4}}.hit{{fill:var(--hit)}}.miss{{fill:var(--miss)}}.ax{{fill:var(--muted);font-size:11px}}.note{{color:var(--muted)}}
</style></head><body>
<h1>Imminent: {html.escape(sport)}, {html.escape(set_name)} set</h1>
<p>Rule <code>{html.escape(rule_version)}</code>, threshold {threshold}. Targets (D14): precision &ge; {TARGETS['precision']},
median lead &ge; {TARGETS['median_lead_s']} s, at most {TARGETS['false_alarms_per_10min']} false alarms per 10 minutes.</p>
{f'<p class="note">{html.escape(note)}</p>' if note else ''}
<h2>Overall</h2><pre>{html.escape(json.dumps(summary, indent=2))}</pre>
<h2>Per clip</h2><table><tr><th>Clip</th><th>min</th><th>moments</th><th>spikes</th><th>hits</th><th>precision</th><th>recall</th><th>median lead s</th></tr>{rows}</table>
<h2>Timelines</h2><p class="note">Blue: imminent. Dashed orange: moments. Dots: spikes (green hit, red false alarm).</p>{charts}
</body></html>"""
    path = out_dir / "report.html"
    path.write_text(page)
    (out_dir / "report.json").write_text(
        json.dumps({"sport": sport, "set": set_name, "rule_version": rule_version, "summary": summary, "note": note,
                    "clips": [{k: v for k, v in asdict(r).items() if k != "timeline"} for r in results]}, indent=1)
    )
    return path


def find_tapes(sport: str, set_name: str, tapes_dir: Path) -> list[Path]:
    feeds = load_feeds()
    demo_clips = {f.clip_id for f in feeds.feeds if f.demo}
    if set_name == "demo":
        clips = [f.clip_id for f in feeds.by_sport(sport) if f.demo]
    else:
        tuning_dir = resolve_path("data/samples/tuning") / sport
        clips = sorted(p.stem for p in tuning_dir.glob("*.mp4")) if tuning_dir.exists() else []
        leaked = set(clips) & demo_clips
        if leaked:
            raise SystemExit(f"refusing to tune on demo clips (hard rule 6): {sorted(leaked)}")
    return [tapes_dir / f"{c}.jsonl" for c in clips if (tapes_dir / f"{c}.jsonl").exists()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sport", required=True)
    ap.add_argument("--set", dest="set_name", choices=["tuning", "demo", "mock"], default="tuning")
    ap.add_argument("--tape", type=Path, action="append", help="explicit tape(s); skips the set lookup")
    ap.add_argument("--tapes-dir", type=Path, default=Path("data/tapes"))
    ap.add_argument("--out", type=Path, default=Path("reports/imminent"))
    ap.add_argument("--minutes", type=float, default=20.0, help="mock set only: minutes of mock footage")
    ap.add_argument("--override", action="append", default=[], help="config override, dotted.key=value")
    args = ap.parse_args(argv)

    from intensity.engine.mock import MockGenerator

    config = load_config(overrides=args.override)
    rule_cfg = load_sport_yaml(config.engine.imminent.config_dir, args.sport)
    rule_version = rule_cfg.get("rule_version", "untracked")
    threshold = float(rule_cfg.get("threshold", 0.5))
    note = None
    tapes: list[tuple[str, list[Envelope]]] = []
    if args.set_name == "mock":
        duration = args.minutes * 60
        gen = MockGenerator([(f"mock-{args.sport}", args.sport)], duration_s=duration + 60, t0_ns=0)
        tapes.append((f"mock-{args.sport}", gen.envelopes(0.0, duration)))
        note = "Mock footage: checks the pipeline end to end, says nothing about real broadcasts."
    else:
        from intensity.engine.mock import read_tape

        paths = args.tape or find_tapes(args.sport, args.set_name, resolve_path(args.tapes_dir))
        tapes = [(p.stem, read_tape(p)) for p in paths]
        if not tapes:
            note = (f"No {args.set_name} tapes for {args.sport}: the sport keeps its config defaults (untuned)."
                    if args.set_name == "tuning" else f"No demo tape for {args.sport} yet.")
    results = [evaluate_tape(config, envs, args.sport, clip) for clip, envs in tapes]
    summary = summarise(results)
    out_dir = resolve_path(args.out) / args.sport / rule_version / args.set_name
    save_resolved(config, out_dir)
    path = write_report(out_dir, args.sport, args.set_name, rule_version, threshold, results, summary, note)
    print(json.dumps(summary))
    print(f"report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
