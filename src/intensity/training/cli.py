"""Training pipeline, one command per step: `python -m intensity.training.cli <step> --sport <sport>`.

Steps are cached and resumable. `fetch` (T0) downloads raw data and `convert` (S2) turns it into
match records; `source` runs both. Featurize, label, split, train, calibrate, evaluate and export
land with T1 to T3.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from intensity.config import load_config, resolve_path, save_resolved
from intensity.telemetry import log_event, setup_logging
from intensity.training.sources import CONVERTERS, FETCHERS, convert, make_fetcher

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

    args = parser.parse_args(argv)
    setup_logging()
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request drowns the progress
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
