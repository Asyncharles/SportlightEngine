"""`python -m intensity.engine.perf`: run the scenarios, write the report, push it to the bucket.

    python -m intensity.engine.perf                          # every scenario, then push
    python -m intensity.engine.perf --scenario demo --no-push
    python -m intensity.engine.perf --live-url http://127.0.0.1:8000   # also read a running engine once

Output: data/reports/engine/<run_id>/ (report.html, summary.json, samples.parquet, charts/), pushed to
gs://<bucket>/reports/engine/<run_id>/ with scripts/sync_bucket.sh.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd

from intensity.config import resolve_path, save_resolved
from intensity.data import REPO_ROOT, data_dir
from intensity.engine.perf import bench
from intensity.engine.perf.probe import overhead_us
from intensity.engine.perf.report import write_report
from intensity.telemetry import setup_logging

log = logging.getLogger("intensity.engine.perf")
SCENARIOS = ("startup", "demo", "perception", "scaling", "transport")


def budget_rows(summary: dict[str, Any], frames: dict[str, pd.DataFrame], budgets: dict[str, float]) -> list[dict[str, Any]]:
    demo = frames.get("demo")

    def p95(stage: str) -> float | None:
        if demo is None:
            return None
        v = demo.loc[(demo["stage"] == stage) & (demo["ctx"] == "tick"), "ms"]
        return round(float(v.quantile(0.95)), 4) if len(v) else None

    tr = summary.get("transport", {})
    imminent_ws = tr.get("service_metrics", {}).get("latency_ms", {}).get("imminent_to_ws_ms", {}).get("p95")
    imminent_how = "p95, transport (private service's imminent_to_ws_ms)"
    if imminent_ws is None and demo is not None:  # no crossing in the short real-time window: use the simulation
        v = demo.loc[demo["stage"] == "imminent_to_publish", "ms"]
        imminent_ws = round(float(v.quantile(0.95)), 4) if len(v) else None
        imminent_how = f"p95 of {len(v)} out-of-cycle updates, demo (engine side: fast tick to publish)"
    rows = [
        ("Ingest to engine message on the same host", "ingest_to_engine", tr.get("ingest_to_engine_ms", {}).get("p95"),
         "p95, transport (loopback WebSocket)"),
        ("Engine tick per feed, all stages", "tick_per_feed", p95("tick"), "p95, demo"),
        ("Imminent rule per feed per tick", "imminent_per_feed", p95("imminent"), "p95, demo"),
        ("Forecast per feed per tick", "forecast_per_feed", p95("forecast"), "p95, demo"),
        ("Imminent crossing to /ws/scores", "imminent_to_ws", imminent_ws, imminent_how),
        ("Tick to all WebSocket clients", "tick_to_ws_clients", tr.get("score_to_ws_client_ms", {}).get("p95"),
         "p95, transport (ScoreUpdate t_emit to client receipt)"),
    ]
    out = []
    for stage, key, measured, how in rows:
        budget = float(budgets[key])
        out.append({"stage": stage, "budget_ms": budget, "measured_ms": measured, "measure": how,
                    "ok": None if measured is None else measured <= budget})
    return out


def headline(summary: dict[str, Any], frames: dict[str, pd.DataFrame], budgets: dict[str, float]) -> dict[str, Any]:
    h: dict[str, Any] = {"budget_tick_ms": budgets["tick_per_feed"], "target_tick_ms": budgets["tick_per_feed_target"]}
    demo = frames.get("demo")
    if demo is not None:
        ticks = demo.loc[demo["stage"] == "tick", "ms"]
        if len(ticks):
            h["tick_p95_ms"] = round(float(ticks.quantile(0.95)), 4)
    if fit := summary.get("scaling", {}).get("fit"):
        h["feeds_per_core"] = fit["feeds_per_core"]
    if feeds := summary.get("startup", {}).get("feeds"):
        h["restart_max_ms"] = max(f.get("first_clock_tick_ms", 0) + f.get("first_score_ms", 0) for f in feeds)
    if p := summary.get("transport", {}).get("ingest_to_engine_ms", {}).get("p95"):
        h["ingest_to_engine_p95_ms"] = p
    if vol := summary.get("demo", {}).get("volume"):
        h["update_bytes"] = vol["bytes_ws_per_update"]
        h["amqp_bytes"] = vol["bytes_amqp_per_event"]
        h["updates_per_s"] = vol["updates"] / vol["simulated_s"]
        h["feeds"] = len(vol["updates_per_feed"])
        h["msgs_in_per_s"] = sum(vol["msgs_in_per_s"].values())
    return h


def calls_per_tick(frame: pd.DataFrame | None) -> float:
    if frame is None or frame.empty:
        return 0.0
    ticks = frame[frame["ctx"] == "tick"]
    n = int((ticks["stage"] == "tick").sum())
    inner = ticks[~ticks["stage"].isin(["tick", "score_all", "publish.ws", "publish.amqp"])]
    return len(inner) / n if n else 0.0


def push(out_dir: Path) -> bool:
    try:
        rel = out_dir.resolve().relative_to(data_dir().resolve())
    except ValueError:
        log.error("%s is outside the data dir; not pushed", out_dir)
        return False
    cmd = ["bash", str(REPO_ROOT / "scripts" / "sync_bucket.sh"), "push", rel.as_posix()]
    log.info("pushing: %s", " ".join(cmd))
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode == 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", action="append", choices=SCENARIOS, help="repeatable; default: all")
    ap.add_argument("--overlay", action="append", help="engine config overlay (default: perf.overlays)")
    ap.add_argument("--set", dest="overrides", action="append", default=[], help="engine config override, dotted.key=value")
    ap.add_argument("--perf-config", default=bench.PERF_CONFIG)
    ap.add_argument("--duration", type=float, help="demo: simulated seconds (default: perf.demo.duration_s)")
    ap.add_argument("--live-url", help="also read this running engine's /metrics and /health, once")
    ap.add_argument("--out", help="output root (default: perf.out_dir)")
    ap.add_argument("--no-push", action="store_true", help="write locally only")
    args = ap.parse_args(argv)

    setup_logging(json_format=False)
    perf = bench.load_perf(args.perf_config)
    os.nice(int(perf.get("nice", 0)))  # yield the CPU to anything live on this host
    overlays = args.overlay if args.overlay is not None else list(perf.get("overlays", []))
    scenarios = args.scenario or list(SCENARIOS)
    config = bench.harness_config(overlays, args.overrides)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = resolve_path(args.out or perf["out_dir"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    save_resolved(config, out_dir)
    shutil.copy(resolve_path(args.perf_config), out_dir / "engine_perf.yaml")

    clips = bench.demo_clips(config)
    if not clips:
        log.error("no feeds in %s", config.engine.feeds_path)
        return 2
    summary: dict[str, Any] = {
        "run": {"id": run_id, "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "overlays": overlays,
                "overrides": args.overrides, "scenarios": scenarios},
        "environment": bench.environment(),
        "inventory": bench.inventory(config),
        "sports": {c.feed.feed_id: c.feed.sport for c in clips} | {f"mock-{c.feed.sport}": c.feed.sport for c in clips},
        "probe_overhead_us": round(overhead_us(), 3),
    }
    frames: dict[str, pd.DataFrame] = {}
    scaling_steps = transport = None
    for name in scenarios:
        t0 = time.perf_counter()
        log.info("scenario %s ...", name)
        if name == "startup":
            summary["startup"] = bench.run_startup(config, overlays, clips, float(perf["startup"]["restart_at_s"]))
        elif name == "demo":
            summary["demo"], frames["demo"] = bench.run_demo(config, clips, args.duration or perf["demo"].get("duration_s"))
        elif name == "perception":
            summary["perception"], frames["perception"] = bench.run_perception(config, clips, float(perf["perception"]["minutes"]))
        elif name == "scaling":
            s = perf["scaling"]
            rows, scaling_steps = bench.run_scaling(config, clips, [int(n) for n in s["feeds"]], float(s["warmup_s"]), float(s["duration_s"]))
            summary["scaling"] = {"rows": rows, "fit": bench.max_feeds(rows, 1000.0 * config.engine.service.score_interval_s)}
        elif name == "transport":
            summary["transport"], transport = bench.run_transport(config, clips, float(perf["transport"]["seconds"]))
        log.info("scenario %s done in %.1f s", name, time.perf_counter() - t0)
    if args.live_url:
        summary["live"] = bench.read_live(args.live_url)

    budgets = perf["budgets_ms"]
    summary["calls_per_tick"] = round(calls_per_tick(frames.get("demo")), 1)
    summary["budget"] = budget_rows(summary, frames, budgets)
    summary["headline"] = headline(summary, frames, budgets)
    if frames:
        pd.concat(frames.values(), ignore_index=True).to_parquet(out_dir / "samples.parquet", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    path = write_report(out_dir, summary, frames, scaling_steps, transport)
    print(json.dumps(summary["headline"]))
    print(f"report: {path}")
    if args.no_push:
        return 0
    if not push(out_dir):
        log.error("push failed; the report is still at %s", out_dir)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
