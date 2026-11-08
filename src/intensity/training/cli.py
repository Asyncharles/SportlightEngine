"""Training pipeline, one command per step: `python -m intensity.training.cli <step> --sport <sport>`.

Steps are cached and resumable. `fetch` (T0) downloads raw data and `convert` (S2) turns it into
match records; `source` runs both. Then `featurize`, `label`, `split`, `train` (which calibrates),
`evaluate` and `export`; each runs the steps before it when their outputs are missing or stale.
`build` runs the whole chain from source to export. Labels exist for football (T1); F1 and NBA land
in T2 and T3.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from intensity.config import load_config, resolve_path, save_resolved
from intensity.telemetry import log_event, setup_logging
from intensity.training.evaluate import evaluate
from intensity.training.export import export
from intensity.training.featurize import featurize
from intensity.training.labels import LABEL_EVENTS, label
from intensity.training.model import staged, train
from intensity.training.sources import CONVERTERS, FETCHERS, convert, make_fetcher
from intensity.training.split import split

log = logging.getLogger("intensity.training")


def cmd_fetch(args: argparse.Namespace) -> int:
    config = load_config(overlays=args.overlay, overrides=args.set)
    sports = sorted(FETCHERS) if args.sport == "all" else [args.sport]
    run_dir = (
        resolve_path(config.training["paths"]["runs"])
        / f"fetch-{args.sport}-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    save_resolved(config, run_dir)
    failures = 0
    for sport in sports:
        fetcher = make_fetcher(sport, config)
        items = args.items.split(",") if args.items else None
        stats = fetcher.run(limit=args.limit, items=items, refresh_index=args.refresh_index)
        failures += len(stats.failed)
        log_event(
            log, "fetch done", sport=sport, fetched=stats.fetched, skipped=stats.skipped, failed=stats.failed
        )
    return 1 if failures else 0


def cmd_convert(args: argparse.Namespace) -> int:
    config = load_config(overlays=args.overlay, overrides=args.set)
    sports = sorted(CONVERTERS) if args.sport == "all" else [args.sport]
    failures = 0
    for sport in sports:
        items = args.items.split(",") if args.items else None
        failures += len(convert(sport, config, force=args.force, items=items).failed)
    return 1 if failures else 0


def cmd_source(args: argparse.Namespace) -> int:
    fetched = cmd_fetch(args)
    return cmd_convert(args) or fetched


def cmd_step(args: argparse.Namespace) -> int:
    config = load_config(overlays=args.overlay, overrides=args.set)
    sport, force = args.sport, getattr(args, "force", False)
    if args.step == "featurize":
        featurize(config, sport, force=force)
    elif args.step == "label":
        label(config, sport, force=force)
    elif args.step == "split":
        split(config, sport)
    elif args.step == "train":
        train(config, sport)
    elif args.step == "evaluate":
        print(f"report: {evaluate(config, sport, args.version)}")
    elif args.step == "export":
        print(f"bundle: {export(config, sport, args.version, force=force)}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """Source, featurize, label, split, train, calibrate, evaluate and export, from a clean cache."""
    config = load_config(overlays=args.overlay, overrides=args.set)
    if not args.no_source:
        args.limit = args.items = None
        args.refresh_index = False
        if cmd_source(args):
            return 1
    featurize(config, args.sport, force=args.force)
    label(config, args.sport, force=args.force)
    train(config, args.sport)
    stage = staged(config, args.sport)
    print(f"report: {evaluate(config, args.sport)}")
    meta = json.loads((stage / "meta.json").read_text())
    if not meta["metrics"]["passed"]:
        log_event(log, "targets missed, not exported", sport=args.sport, targets=meta["metrics"]["targets"])
        return 2
    print(f"bundle: {export(config, args.sport)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m intensity.training.cli")
    parser.add_argument("--overlay", action="append", default=[], help="config overlay (repeatable)")
    parser.add_argument("--set", action="append", default=[], help="dotted.key=value override (repeatable)")
    sub = parser.add_subparsers(dest="step", required=True)

    fetch = sub.add_parser("fetch", help="download raw data to data/raw/<sport>/ (resumable)")
    fetch.add_argument("--sport", required=True, choices=[*sorted(FETCHERS), "all"])
    fetch.add_argument("--limit", type=int, default=None, help="stop after fetching this many new items")
    fetch.add_argument("--items", default=None, help="comma-separated item ids instead of the index")
    fetch.add_argument("--refresh-index", action="store_true", help="re-list the items before fetching")
    fetch.set_defaults(func=cmd_fetch)

    conv = sub.add_parser("convert", help="raw files to data/events/<sport>/<item>.jsonl match records")
    conv.add_argument("--sport", required=True, choices=[*sorted(CONVERTERS), "all"])
    conv.add_argument("--items", default=None, help="comma-separated item ids")
    conv.add_argument("--force", action="store_true", help="reconvert even if the JSONL is up to date")
    conv.set_defaults(func=cmd_convert)

    src = sub.add_parser("source", help="fetch, then convert")
    src.add_argument("--sport", required=True, choices=[*sorted(FETCHERS), "all"])
    src.add_argument("--limit", type=int, default=None)
    src.add_argument("--items", default=None)
    src.add_argument("--refresh-index", action="store_true")
    src.add_argument("--force", action="store_true")
    src.set_defaults(func=cmd_source)

    sports = sorted(LABEL_EVENTS)
    feat = sub.add_parser("featurize", help="match records to data/tables/<sport>/features.parquet")
    feat.add_argument("--sport", required=True, choices=sorted(CONVERTERS))
    feat.add_argument("--force", action="store_true", help="rebuild even if the inputs are unchanged")
    lab = sub.add_parser("label", help="label every featurized tick (data/tables/<sport>/labels.parquet)")
    lab.add_argument("--sport", required=True, choices=sports)
    lab.add_argument("--force", action="store_true")
    spl = sub.add_parser("split", help="train/val/test by match; demo matches always test")
    spl.add_argument("--sport", required=True, choices=sorted(CONVERTERS))
    trn = sub.add_parser("train", help="search, train and calibrate; stages run/training/<sport>/<version>/")
    trn.add_argument("--sport", required=True, choices=sports)
    ev = sub.add_parser(
        "evaluate", help="test metrics, highlight check and reports/training/<sport>/<version>/"
    )
    ev.add_argument("--sport", required=True, choices=sports)
    ev.add_argument("--version", default=None, help="staged version (default: run/training/<sport>/LATEST)")
    exp = sub.add_parser(
        "export", help="copy the evaluated bundle to models/<sport>/<version>/ (never CURRENT)"
    )
    exp.add_argument("--sport", required=True, choices=sports)
    exp.add_argument("--version", default=None)
    exp.add_argument("--force", action="store_true", help="export even if a target was missed")
    for p in (feat, lab, spl, trn, ev, exp):
        p.set_defaults(func=cmd_step)

    build = sub.add_parser("build", help="source through export in one command")
    build.add_argument("--sport", required=True, choices=sports)
    build.add_argument("--no-source", action="store_true", help="skip fetch and convert")
    build.add_argument("--force", action="store_true", help="reconvert and rebuild the tables")
    build.set_defaults(func=cmd_build)

    args = parser.parse_args(argv)
    setup_logging()
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request drowns the progress
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
