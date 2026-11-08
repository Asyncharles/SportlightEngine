"""Charts (matplotlib, PNG) and a self-contained HTML page for the engine performance report."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from intensity.engine.perf.probe import NESTED, TICK_GROUPS

# Reference palette (light chart surface): categorical slots in fixed order, text and grid tokens.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SPORT_COLOR = {"football": SLOTS[0], "f1": SLOTS[1], "nba": SLOTS[2], "weightlifting": SLOTS[3], "chess": SLOTS[4]}
GROUPS = [*TICK_GROUPS, "other"]

STAGE_ORDER = [
    "ingest.serialize", "ingest.parse", "handle.clock", "handle.control", "handle.event.match", "handle.event.crowd",
    "handle.event.voice", "handle.event.video", "handle.event.commentary", "handle.event.ocr", "fast_tick",
    "tick", "replay.release", "adapter.on_match", "adapter.parts", "adapter.facts", "forecast", "imminent",
    "trackers.crowd", "trackers.commentary", "trackers.video", "parts.forecast", "parts.imminent", "combine",
    "smoothing.ema", "smoothing.trend", "percentile.add", "percentile.rank", "reasons", "score_all",
    "publish.ws", "publish.amqp", "imminent_to_publish", "preroll",
]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "font.size": 9, "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
    })
    return plt


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


# --------------------------------------------------------------------------- tables from samples


def stage_table(frame: pd.DataFrame) -> pd.DataFrame:
    """count, mean and percentiles per stage, all contexts together."""
    if frame.empty:
        return pd.DataFrame()
    g = frame.groupby("stage", observed=True)["ms"]
    out = pd.DataFrame({
        "count": g.size(),
        "mean": g.mean(),
        "p50": g.quantile(0.5),
        "p95": g.quantile(0.95),
        "p99": g.quantile(0.99),
        "max": g.max(),
        "total_ms": g.sum(),
    })
    order = [s for s in STAGE_ORDER if s in out.index] + sorted(set(out.index) - set(STAGE_ORDER))
    return out.loc[order]


def tick_breakdown(frame: pd.DataFrame, sports: dict[str, str]) -> pd.DataFrame:
    """Mean ms per tick per feed in each stage group; `other` is the tick minus its timed stages."""
    ticks = frame[(frame["ctx"] == "tick")]
    if ticks.empty:
        return pd.DataFrame()
    stage_to_group = {s: g for g, stages in TICK_GROUPS.items() for s in stages}
    per_feed = {}
    for feed, df in ticks.groupby("feed_id", observed=True):
        n = int((df["stage"] == "tick").sum())
        if not n:
            continue
        row = {g: 0.0 for g in GROUPS}
        sums = df.groupby("stage", observed=True)["ms"].sum()
        timed = 0.0
        for stage, total in sums.items():
            if stage == "tick" or stage in NESTED or str(stage).startswith("publish."):
                continue
            timed += total
            if stage in stage_to_group:
                row[stage_to_group[stage]] += total / n
        row["other"] = (sums.get("tick", 0.0) - sum(row[g] * n for g in TICK_GROUPS)) / n
        row["tick"] = sums.get("tick", 0.0) / n
        row["sport"] = sports.get(str(feed), "")
        per_feed[str(feed)] = row
    return pd.DataFrame.from_dict(per_feed, orient="index")


# --------------------------------------------------------------------------- charts


def chart_breakdown(tables: dict[str, pd.DataFrame], path: Path) -> Path | None:
    rows = [(f"{scenario} · {feed}", r) for scenario, t in tables.items() for feed, r in t.iterrows()]
    if not rows:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.6))
    labels = [label for label, _ in rows][::-1]
    left = np.zeros(len(rows))
    for i, group in enumerate(GROUPS):
        vals = np.array([max(0.0, float(r[group])) * 1000 for _, r in rows][::-1])
        ax.barh(labels, vals, left=left, color=SLOTS[i], edgecolor=SURFACE, linewidth=1.0, height=0.62, label=group)
        left += vals
    for y, total in enumerate(left):
        ax.text(total, y, f"  {total:.0f} µs", va="center", color=INK_2, fontsize=8)
    ax.set_xlabel("mean time per tick per feed (µs)")
    ax.set_title("Where a tick's time goes, per feed")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, left.max() * 1.15)
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.04), handlelength=1, columnspacing=1.2)
    return _save(fig, path)


def chart_stage_latency(table: pd.DataFrame, path: Path) -> Path | None:
    t = table[[s not in ("preroll",) for s in table.index]]
    if t.empty:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 0.26 * len(t) + 1.4))
    y = np.arange(len(t))[::-1]
    us = t[["p50", "p95", "p99"]] * 1000
    ax.hlines(y, us["p50"], us["p99"], color=GRID, linewidth=2)
    for i, (col, marker) in enumerate((("p50", "o"), ("p95", "D"), ("p99", "s"))):
        ax.scatter(us[col], y, s=30, color=SLOTS[i], marker=marker, edgecolor=SURFACE, linewidth=1.5, zorder=3, label=col)
    ax.set_yticks(y, t.index)
    ax.set_xscale("log")
    ax.set_xlabel("latency per call (µs, log scale)")
    ax.set_title("Latency per stage (demo feeds)")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.02))
    return _save(fig, path)


def chart_timeline(frame: pd.DataFrame, sports: dict[str, str], path: Path, bin_s: float = 30.0) -> Path | None:
    ticks = frame[frame["stage"] == "tick"]
    if ticks.empty:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for feed, df in ticks.groupby("feed_id", observed=True):
        b = (df["now"] // bin_s) * bin_s
        p95 = df.groupby(b)["ms"].quantile(0.95) * 1000
        ax.plot(p95.index / 60, p95.values, color=SPORT_COLOR.get(sports.get(str(feed), ""), INK_2), label=str(feed))
    ax.set_yscale("log")
    ax.set_xlabel("simulated minutes")
    ax.set_ylabel(f"tick p95 per {bin_s:.0f} s (µs)")
    ax.set_title("Tick latency over the demo clips", pad=26)
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))
    return _save(fig, path)


def chart_scaling(rows: list[dict[str, Any]], fit: dict[str, float] | None, path: Path) -> Path | None:
    if not rows:
        return None
    plt = _plt()
    x = np.array([r["feeds"] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    ax = axes[0]
    ax.plot(x, [r["busy_ms_per_s"] for r in rows], marker="o", markersize=6, color=SLOTS[0])
    if fit:
        xs = np.linspace(0, x.max(), 50)
        ax.plot(xs, fit["fixed_ms_per_s"] + fit["ms_per_feed_per_s"] * xs, color=INK_2, linewidth=1, linestyle="--")
        ax.text(0.02, 0.95, f"{fit['ms_per_feed_per_s']:.2f} ms/s per feed\n≈ {fit['feeds_per_core']} feeds fill one core",
                transform=ax.transAxes, va="top", color=INK, fontsize=8)
    ax.set_title("Engine busy time")
    ax.set_xlabel("feeds")
    ax.set_ylabel("ms of work per second")
    ax = axes[1]
    ax.plot(x, [r["score_all"].get("p95", np.nan) for r in rows], marker="o", markersize=6, color=SLOTS[0])
    ax.set_title("score_all p95 (all feeds, 1 Hz)")
    ax.set_xlabel("feeds")
    ax.set_ylabel("ms")
    ax = axes[2]
    ax.plot(x, [r["rss_mb"] for r in rows], marker="o", markersize=6, color=SLOTS[0])
    ax.set_title("Process memory after the run")
    ax.set_xlabel("feeds")
    ax.set_ylabel("RSS (MB)")
    for a in axes:
        a.set_xticks(x, [str(v) for v in x])
        a.set_xlim(0, x.max() * 1.05)
        a.set_ylim(bottom=0)
    fig.tight_layout()
    return _save(fig, path)


def chart_startup(startup: dict[str, Any], path: Path) -> Path | None:
    if not startup:
        return None
    items = [("config load", startup.get("config_load_ms")), ("model bundles", startup.get("registry_reload_ms")),
             ("engine init (all feeds)", startup.get("engine_init_ms"))]
    for f in startup.get("feeds", []):
        items.append((f"{f['feed_id']}: load records", f.get("load_records_ms")))
    for f in startup.get("feeds", []):
        items.append((f"{f['feed_id']}: restart pre-roll ({f.get('preroll_s', 0):.0f} s)", f.get("first_clock_tick_ms")))
    for f in startup.get("feeds", []):
        items.append((f"{f['feed_id']}: first score", f.get("first_score_ms")))
    items = [(k, v) for k, v in items if v is not None]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 0.3 * len(items) + 1.2))
    labels = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    ax.barh(labels, vals, color=SLOTS[0], height=0.6, edgecolor=SURFACE, linewidth=1)
    for y, v in enumerate(vals):
        ax.text(v, y, f"  {v:.1f}", va="center", color=INK_2, fontsize=8)
    ax.set_xlabel("ms")
    ax.set_xlim(0, max(vals) * 1.15)
    ax.set_title("Startup and restart")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def chart_transport(df: pd.DataFrame, path: Path) -> Path | None:
    if df is None or df.empty:
        return None
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, (kind, title) in zip(axes, (("ingest_to_engine", "Ingest to engine (ClockTick, WebSocket)"),
                                        ("score_to_ws_client", "ScoreUpdate to /ws/scores client")), strict=True):
        v = df.loc[df["kind"] == kind, "ms"].to_numpy()
        if v.size:
            ax.hist(v, bins=40, color=SLOTS[0], edgecolor=SURFACE, linewidth=0.8)
            p95 = float(np.percentile(v, 95))
            ax.axvline(p95, color=INK_2, linewidth=1, linestyle="--")
            ax.text(p95, ax.get_ylim()[1] * 0.92, f" p95 {p95:.2f} ms", color=INK, fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("ms")
        ax.set_ylabel("messages")
    fig.tight_layout()
    return _save(fig, path)


def chart_volume(volumes: dict[str, dict[str, Any]], path: Path) -> Path | None:
    kinds = sorted({k for v in volumes.values() for k in v.get("msgs_in_per_s", {})})
    if not kinds:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(kinds) * max(1, len(volumes) / 1.5) + 1.2))
    y = np.arange(len(kinds))
    h = 0.8 / len(volumes)
    for i, (name, v) in enumerate(volumes.items()):
        vals = [v["msgs_in_per_s"].get(k, 0.0) for k in kinds]
        ax.barh(y + (i - (len(volumes) - 1) / 2) * h, vals, height=h * 0.9, color=SLOTS[i], label=name,
                edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(y, kinds)
    ax.invert_yaxis()
    ax.set_xlabel("messages per second into the engine (all feeds)")
    ax.set_title("Ingest volume by message kind")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=len(volumes))
    return _save(fig, path)


# --------------------------------------------------------------------------- HTML


def _img(path: Path | None, alt: str) -> str:
    if path is None:
        return '<p class="note">No data for this chart.</p>'
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<figure><img alt="{html.escape(alt)}" src="data:image/png;base64,{data}"></figure>'


def _num(v: Any, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if isinstance(v, float):
        return f"{v:,.{digits}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return html.escape(str(v))


def _table(df: pd.DataFrame, digits: int = 4, index_name: str = "") -> str:
    if df is None or df.empty:
        return '<p class="note">No data.</p>'
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in [index_name or df.index.name or "", *df.columns])
    body = "".join(
        "<tr>" + f"<td>{html.escape(str(i))}</td>" + "".join(f"<td>{_num(v, digits)}</td>" for v in row) + "</tr>"
        for i, row in zip(df.index, df.itertuples(index=False), strict=True)
    )
    return f'<div class="scroll"><table><tr>{head}</tr>{body}</table></div>'


def _tile(label: str, value: str, sub: str = "") -> str:
    return f'<div class="tile"><div class="label">{html.escape(label)}</div><div class="value">{value}</div><div class="sub">{html.escape(sub)}</div></div>'


def _status(ok: bool | None) -> str:
    if ok is None:
        return '<span class="st na">– not measured</span>'
    return '<span class="st ok">✓ within</span>' if ok else '<span class="st bad">✗ over</span>'


def write_report(out_dir: Path, summary: dict[str, Any], frames: dict[str, pd.DataFrame],
                 scaling_steps: pd.DataFrame | None, transport: pd.DataFrame | None) -> Path:
    charts = out_dir / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    sports = summary.get("sports", {})
    breakdowns = {name: tick_breakdown(frames[name], sports) for name in ("demo", "perception") if name in frames}
    stage_tables = {name: stage_table(df) for name, df in frames.items()}
    for name, t in stage_tables.items():
        t.round(5).to_csv(out_dir / f"stages_{name}.csv")
    vols = {n: summary[n]["volume"] for n in ("demo", "perception") if n in summary}

    paths = {
        "breakdown": chart_breakdown({k: v for k, v in breakdowns.items() if not v.empty}, charts / "tick_breakdown.png"),
        "latency": chart_stage_latency(stage_tables["demo"], charts / "stage_latency.png") if "demo" in stage_tables else None,
        "timeline": chart_timeline(frames["demo"], sports, charts / "tick_timeline.png") if "demo" in frames else None,
        "scaling": chart_scaling(summary.get("scaling", {}).get("rows", []), summary.get("scaling", {}).get("fit"),
                                 charts / "scaling.png"),
        "startup": chart_startup(summary.get("startup", {}), charts / "startup.png"),
        "transport": chart_transport(transport, charts / "transport.png"),
        "volume": chart_volume(vols, charts / "volume.png"),
    }

    tiles = []
    h = summary.get("headline", {})
    if "tick_p95_ms" in h:
        tiles.append(_tile("Tick per feed, p95", f"{h['tick_p95_ms'] * 1000:,.0f} µs", f"budget {h['budget_tick_ms']:.0f} ms, target {h['target_tick_ms']:.0f} ms"))
    if "feeds_per_core" in h:
        tiles.append(_tile("Engine CPU ceiling", f"≈ {h['feeds_per_core']:,} feeds", "per core at 1 Hz, engine side only"))
    if "restart_max_ms" in h:
        tiles.append(_tile("Restart, slowest feed", f"{h['restart_max_ms']:,.0f} ms", "pre-roll + first score"))
    if "ingest_to_engine_p95_ms" in h:
        tiles.append(_tile("Ingest → engine, p95", f"{h['ingest_to_engine_p95_ms']:.2f} ms", "loopback WebSocket"))
    if "update_bytes" in h:
        tiles.append(_tile("ScoreUpdate size", f"{h['update_bytes']:,.0f} B", f"{h.get('amqp_bytes') or 0:,.0f} B as a FeedEvent"))
    if "updates_per_s" in h:
        tiles.append(_tile("Updates out", f"{h['updates_per_s']:.2f} /s", f"{h.get('feeds', 0)} feeds, {h.get('msgs_in_per_s', 0):.0f} msgs/s in"))

    budget_rows = "".join(
        f"<tr><td>{html.escape(b['stage'])}</td><td>{_num(b['budget_ms'], 1)}</td><td>{_num(b.get('measured_ms'), 4)}</td>"
        f"<td>{html.escape(b.get('measure', ''))}</td><td>{_status(b.get('ok'))}</td></tr>"
        for b in summary.get("budget", [])
    )
    startup = summary.get("startup", {})
    startup_df = pd.DataFrame(startup.get("feeds", [])).set_index("feed_id") if startup.get("feeds") else pd.DataFrame()
    scaling = summary.get("scaling", {})
    scaling_df = pd.DataFrame([
        {"feeds": r["feeds"], "busy ms/s": r["busy_ms_per_s"], "score_all p50": r["score_all"].get("p50"),
         "score_all p95": r["score_all"].get("p95"), "tick p95": r["tick"].get("p95"), "engine init ms": r["engine_init_ms"],
         "RSS MB": r["rss_mb"]} for r in scaling.get("rows", [])
    ]).set_index("feeds") if scaling.get("rows") else pd.DataFrame()
    vol_df = pd.DataFrame({n: {
        "simulated s": v["simulated_s"], "messages in": sum(v["msgs_in"].values()), "messages in /s": sum(v["msgs_in_per_s"].values()),
        "bytes in /s": v["bytes_in_per_s"], "ScoreUpdates": v["updates"], "out of cycle": v["out_of_cycle"],
        "stale": v["stale_updates"], "bytes per ScoreUpdate": v["bytes_ws_per_update"], "WS bytes /s": v["bytes_ws_per_s"],
        "AMQP events": v["amqp_events"], "bytes per FeedEvent": v["bytes_amqp_per_event"],
    } for n, v in vols.items()})
    tr = summary.get("transport", {})
    tr_df = pd.DataFrame({k: tr[k] for k in ("ingest_to_engine_ms", "score_to_ws_client_ms") if tr.get(k)}).T if tr else pd.DataFrame()
    svc = tr.get("service_metrics", {}).get("latency_ms", {}) if tr else {}
    svc_df = pd.DataFrame(svc).T if svc else pd.DataFrame()

    def section(title: str, body: str) -> str:
        return f"<section><h2>{html.escape(title)}</h2>{body}</section>"

    run = summary["run"]
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Engine performance</title>
<style>
:root{{--bg:#fcfcfb;--fg:#0b0b0b;--muted:#52514e;--grid:#e4e3df;--card:#fcfcfb;--ok:#0ca30c;--bad:#d03b3b}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--grid:#383835}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--grid:#383835}}
body{{font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg);max-width:1040px;margin:24px auto;padding:0 16px}}
h1{{margin-bottom:4px}}h2{{margin-top:36px;border-bottom:1px solid var(--grid);padding-bottom:4px}}
.note{{color:var(--muted)}}code{{font-size:13px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px;margin:20px 0}}
.tile{{border:1px solid var(--grid);border-radius:8px;padding:12px 14px}}.tile .label{{color:var(--muted);font-size:13px}}
.tile .value{{font-size:26px;font-weight:600;font-variant-numeric:tabular-nums}}.tile .sub{{color:var(--muted);font-size:12px}}
figure{{margin:16px 0;background:var(--card);border-radius:8px;padding:8px;border:1px solid var(--grid)}}img{{width:100%;height:auto;display:block}}
.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}}
td,th{{border-bottom:1px solid var(--grid);padding:4px 8px;text-align:right;white-space:nowrap}}td:first-child,th:first-child{{text-align:left}}
.st{{font-weight:600}}.st.ok{{color:var(--ok)}}.st.bad{{color:var(--bad)}}.st.na{{color:var(--muted);font-weight:400}}
pre{{overflow-x:auto;font-size:12px;border:1px solid var(--grid);border-radius:8px;padding:8px}}
</style></head><body>
<h1>Engine performance</h1>
<p class="note">Run <code>{html.escape(run['id'])}</code> · {html.escape(run['created_at'])} · git <code>{html.escape(str(summary['environment'].get('git_sha')))}</code> ·
{html.escape(summary['environment']['host'])} ({html.escape(summary['environment']['machine'])}, {summary['environment']['cpus_logical']} CPUs) ·
overlays <code>{html.escape(','.join(run['overlays']))}</code> · scenarios <code>{html.escape(', '.join(run['scenarios']))}</code></p>
<div class="tiles">{''.join(tiles)}</div>
<p class="note">Measured on this host's CPU; the VM's numbers will differ. All timings are wall-clock per call. Each timed call
adds about {summary.get('probe_overhead_us', 0):.2f} µs of probe overhead, and a tick contains about {summary.get('calls_per_tick', 0):.0f} of them, so tick times include roughly
{summary.get('probe_overhead_us', 0) * summary.get('calls_per_tick', 0):.0f} µs of measurement.</p>

{section("Isolation from live inference", '''<ul>
<li>A separate process with its own private engine. Nothing is sent to a running engine; the transport check starts its own engine service on a free 127.0.0.1 port.</li>
<li>No engine code changes. Timers are wrappers set on the private engine's own objects; the two module functions it times are put back afterwards.</li>
<li>AMQP publishing and reason polish are forced off, so no FeedEvent reaches the front end and no LLM is called. The clock lane runs from computed clip durations and never writes <code>run/feeds_state.json</code>.</li>
<li>The process runs at niceness ''' + str(summary['environment'].get('niceness')) + ''', so on a shared host the live engine gets the CPU first.</li>
<li>The live snapshot, when requested, is a single GET each of <code>/metrics</code> and <code>/health</code>.</li></ul>''')}

{section("Latency budget", f'''<p class="note">The plan's budget (Engine and serving). GPU stages are not built yet and are not listed.</p>
<div class="scroll"><table><tr><th>Stage</th><th>Budget ms</th><th>Measured ms</th><th>Measure</th><th>Status</th></tr>{budget_rows}</table></div>''')}

{section("Processing time per tick", _img(paths['breakdown'], 'Stacked bars of mean time per tick per feed by stage')
 + '<p class="note">Groups: replay (match records released), adapter (parts and facts), forecast (model or heuristic), imminent rule, trackers (crowd, commentary, video parts), percentile (sport window), reasons (templates and guard); other is combine, EMA, trend and building the ScoreUpdate. The perception rows run mock footage with all six sources.</p>'
 + _table(pd.concat(breakdowns).round(5) if breakdowns and any(not b.empty for b in breakdowns.values()) else pd.DataFrame(), 5, "scenario, feed"))}

{section("Latency per stage", _img(paths['latency'], 'Dot plot of p50, p95 and p99 latency per stage') + _table(stage_tables.get('demo', pd.DataFrame()).round(5), 5, 'stage (demo)')
 + ('<h3>Perception (mock, all six sources)</h3>' + _table(stage_tables['perception'].round(5), 5, 'stage (perception)') if 'perception' in stage_tables else ''))}

{section("Tick latency over time", _img(paths['timeline'], 'Line chart of tick p95 per feed over simulated time'))}

{section("Data volume", _img(paths['volume'], 'Bars of messages per second into the engine by kind') + _table(vol_df, 1, 'measure')
 + f'''<p class="note">Demo: the clock lane at {summary.get('demo', {}).get('volume', {}).get('msgs_in_per_s', {}).get('clock', 0) / max(1, h.get('feeds', 1)):.1f} ticks per feed per second, with match data replayed inside the engine ({summary.get('demo', {}).get('replayed_records', 0):,} records over the run). Perception (mock) adds crowd, voice, video, commentary and OCR events, which the real ingest does not send yet.</p>''')}

{section("Startup and restart", _img(paths['startup'], 'Bars of startup and restart times') + _table(startup_df, 2, 'feed')
 + f'''<p class="note">Config load {_num(startup.get('config_load_ms'), 1)} ms, model bundles {_num(startup.get('registry_reload_ms'), 1)} ms
({html.escape(json.dumps(startup.get('bundles', {})))}), engine construction {_num(startup.get('engine_init_ms'), 1)} ms
(+{_num(startup.get('engine_init_rss_mb'), 1)} MB). A restart mid-clip pre-rolls each feed's history into the sport's percentile window on its first ClockTick.</p>''' if startup else '<p class="note">Not run.</p>')}

{section("Capacity", _img(paths['scaling'], 'Engine busy time, score_all p95 and memory against feed count') + _table(scaling_df, 3, 'feeds')
 + ('<p class="note">Demo feeds cloned, each clone at its own clip offset; pre-roll off. Busy time per simulated second is the work the engine process does itself: parsing and handling envelopes, the fast and score ticks, and serialising ScoreUpdates and FeedEvents. The engine scores on one event-loop thread, so one core is its ceiling. The fit leaves out WebSocket and asyncio framing (compare <code>tick_ms</code> of the private service under Transport), and the real limit on feeds is the GPU ingest (experiment E3), not the engine. Memory is the process RSS, including the harness.</p>' if scaling.get('rows') else ''))}

{section("Transport", _img(paths['transport'], 'Histograms of ingest-to-engine and score-to-client latency') + _table(tr_df, 4, 'measure')
 + ('<h3>Private service /metrics</h3>' + _table(svc_df, 4, 'histogram') + f'<p class="note">{html.escape(tr.get("note", ""))}. Sent {tr.get("sent", 0):,}, dropped {html.escape(json.dumps(tr.get("dropped", {})))}.</p>' if tr else ''))}

{section("Live engine snapshot", '<pre>' + html.escape(json.dumps(summary['live'], indent=1)[:20000]) + '</pre>' if summary.get('live') else '<p class="note">Not requested (<code>--live-url</code>).</p>')}

{section("Data on disk", '<pre>' + html.escape(json.dumps(summary.get('inventory', {}), indent=1)) + '</pre>')}
{section("Environment", '<pre>' + html.escape(json.dumps(summary['environment'], indent=1)) + '</pre>')}
<p class="note">Files: <code>summary.json</code>, <code>samples.parquet</code> (every timed call), <code>stages_*.csv</code>, <code>scaling_steps.parquet</code>,
<code>transport.parquet</code>, <code>charts/*.png</code>, <code>config.resolved.yaml</code>, <code>engine_perf.yaml</code>.</p>
</body></html>"""
    path = out_dir / "report.html"
    path.write_text(page)
    if scaling_steps is not None and not scaling_steps.empty:
        scaling_steps.to_parquet(out_dir / "scaling_steps.parquet", index=False)
    if transport is not None and not transport.empty:
        transport.to_parquet(out_dir / "transport.parquet", index=False)
    return path
