"""Sync check: does the match data line up with the video? (`make sync-check CLIP=<clip_id>|demo`)

For each clip it picks up to `--moments` key match records spread across the clip (goals before shots,
scores, passes, lifts), maps each to `t_video` through the clip's sync table (weightlifting sessions are
already in video time), and grabs three frames: `--bracket` seconds before, at, and after. Each row is
labelled with the game clock the broadcast should show (football minute, NBA quarter clock, F1 lap,
the lifter and weight), so a human confirms in a minute that the event is in the middle frame. A clip is
in sync when every row's event is within the middle frame's +-2 s.

The clip is read where it lives: `data/samples/...` on a VM, or straight from the private bucket over
authenticated HTTPS (`--from-bucket`, seeking by range requests), so no footage is ever copied to a
laptop. Only small thumbnails are written, to `data/reports/sync/<clip_id>/`, which mirrors the bucket's
`reports/` (`make data-push FOLDERS=reports/sync`).

A measured offset is fixed with `shift`: `python -m intensity.engine.synccheck shift --clip <id> --by 3.5
[--row N]`, where `--by` is how many seconds LATER than the middle frame the event actually appears
(negative if it appears earlier). It lowers `game_start` by that much in one or all rows, so every record
in those rows maps to a video time `--by` seconds later. Then run the check again.
"""

from __future__ import annotations

import argparse
import csv
import html
import subprocess
from dataclasses import dataclass
from pathlib import Path

import av

from intensity.config import load_config, resolve_path
from intensity.data import bucket_uri
from intensity.engine.adapters.nba import period_bounds
from intensity.engine.replay import load_records
from intensity.engine.syncsheet import _encode_jpeg
from intensity.feeds import Feed, load_feeds
from intensity.schema import MatchPayload
from intensity.sync import SyncTable

# Per sport: the record kinds worth checking, most recognisable first.
KINDS = {
    "football": ("goal", "shot"),
    "nba": ("score",),
    "f1": ("pass", "safety_car"),
    "weightlifting": ("attempt_result", "attempt_called"),
}


@dataclass
class Moment:
    record: MatchPayload
    t_video: float
    clock: str


def clock_label(sport: str, rec: MatchPayload) -> str:
    """What the broadcast should show at this record."""
    t = rec.game_time_s
    a = rec.attrs
    if sport == "football":
        m, s = divmod(int(t), 60)
        return f"{m}:{s:02d} (P{rec.period})"
    if sport == "nba":
        _, end = period_bounds(rec.period)
        m, s = divmod(int(max(0.0, end - t)), 60)
        label = f"Q{rec.period}" if rec.period <= 4 else f"OT{rec.period - 4}"
        return f"{label} {m}:{s:02d}"
    if sport == "f1":
        return f"lap {a.get('lap', '?')}"
    if sport == "weightlifting":
        what = "result" if rec.kind == "attempt_result" else f"{a.get('weight_kg', '?')} kg, attempt {a.get('attempt', '?')}"
        return f"{rec.player or '?'}: {what}"
    return f"{t:.0f}s"


def video_time(feed: Feed, rec: MatchPayload, sync: SyncTable | None) -> float | None:
    if sync is None:  # session CSVs are in video time
        return rec.game_time_s
    return sync.t_video(rec.game_time_s, rec.period)


def pick_moments(feed: Feed, records: list[MatchPayload], sync: SyncTable | None, n: int) -> list[Moment]:
    """Up to n key records spread evenly over the clip, preferring the most recognisable kinds."""
    kinds = KINDS.get(feed.sport, ())
    cands = []
    for rec in records:
        if rec.kind not in kinds or rec.attrs.get("start"):  # F1 lap-one swaps are unreadable on screen
            continue
        t = video_time(feed, rec, sync)
        if t is not None:
            cands.append(Moment(rec, t, clock_label(feed.sport, rec)))
    if not cands:
        return []
    cands.sort(key=lambda m: m.t_video)
    rank = {k: i for i, k in enumerate(kinds)}
    best = sorted(cands, key=lambda m: (rank[m.record.kind], m.t_video))
    lo, hi = cands[0].t_video, cands[-1].t_video
    picked: list[Moment] = []
    for i in range(min(n, len(cands))):  # one moment per slice of the clip, the best kind in it
        a = lo + (hi - lo) * i / max(1, n)
        b = lo + (hi - lo) * (i + 1) / max(1, n) + 1e-6
        slot = [m for m in best if a <= m.t_video <= b and m not in picked]
        if slot:
            picked.append(slot[0])
    return sorted(picked, key=lambda m: m.t_video)


def bucket_source(sample: str) -> tuple[str, dict[str, str]]:
    """An HTTPS URL and FFmpeg options that stream the clip from the private bucket (no local copy)."""
    uri = bucket_uri(sample)
    token = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True, check=True).stdout.strip()
    url = "https://storage.googleapis.com/" + uri.removeprefix("gs://")
    return url, {"headers": f"Authorization: Bearer {token}\r\n"}


def grab(source: str, options: dict[str, str] | None, times: list[float], width: int) -> dict[float, bytes]:
    """The first frame at or after each time (seeking to the keyframe before it)."""
    out: dict[float, bytes] = {}
    with av.open(source, options=options or {}) as container:
        stream = container.streams.video[0]
        tb = stream.time_base
        for t in sorted({round(max(0.0, x), 3) for x in times}):
            container.seek(int(max(0.0, t - 1.0) / tb), stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts is None or float(frame.pts * tb) + 1e-3 >= t:
                    out[t] = _encode_jpeg(frame, width)
                    break
    return out


def build(feed: Feed, moments: list[Moment], frames: dict[float, bytes], bracket: float, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, m in enumerate(moments):
        cells = []
        for j, (label, t) in enumerate((("before", m.t_video - bracket), ("at", m.t_video), ("after", m.t_video + bracket))):
            key = round(max(0.0, t), 3)
            if key in frames:
                name = f"{i:02d}-{j}.jpg"
                (out_dir / name).write_bytes(frames[key])
                cells.append(f'<figure class="{label}"><img src="{name}" alt="{label}"><figcaption>{label} · {key:.1f}s</figcaption></figure>')
            else:
                cells.append(f'<figure class="{label}"><div class="missing">no frame</div><figcaption>{label}</figcaption></figure>')
        r = m.record
        who = " ".join(x for x in (r.team, r.player) if x)
        rows.append(
            f'<section><h2>{html.escape(r.kind)} · {html.escape(who)}</h2>'
            f'<p>Should show <b>{html.escape(m.clock)}</b> at t_video {m.t_video:.2f}s (game {r.game_time_s:.1f}s, P{r.period}).</p>'
            f'<div class="row">{"".join(cells)}</div></section>'
        )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Sync check</title>
<style>:root{{--bg:#fff;--fg:#111;--muted:#666;--at:#0a66c2}}@media (prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#eee;--muted:#aaa;--at:#60a5fa}}}}
body{{font:14px system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:16px;max-width:1100px}}
.row{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}}figure{{margin:0}}
img{{width:100%;height:auto;border-radius:6px}}.at img{{outline:3px solid var(--at)}}
figcaption,p{{color:var(--muted);font-size:12px}}h2{{font-size:15px;margin:18px 0 2px}}
.missing{{aspect-ratio:16/9;display:grid;place-items:center;border:1px dashed var(--muted)}}</style></head>
<body><h1>Sync check: {html.escape(feed.clip_id)}</h1>
<p>{len(moments)} moments. For each, the event should be in the outlined middle frame and the broadcast clock should
read the label. In sync means every row is within about 2 s. To fix an offset: <code>python -m intensity.engine.synccheck
shift --clip {html.escape(feed.clip_id)} --by SECONDS</code>, then run the check again.</p>{"".join(rows)}</body></html>"""
    path = out_dir / "index.html"
    path.write_text(page)
    return path


def check(feed: Feed, from_bucket: bool, n: int, bracket: float, width: int, out_root: Path) -> Path:
    config = load_config()
    sync = SyncTable.load(resolve_path(feed.sync)) if feed.sync else None
    if sync is None and feed.match_ref.source != "session_csv":
        raise SystemExit(f"{feed.feed_id}: no sync table")
    events = resolve_path(config.engine.replay.events_dir) / feed.sport / f"{feed.match_ref.match_id}.jsonl"
    if not events.exists():
        raise SystemExit(f"{feed.feed_id}: match records missing: {events} (make data-pull FOLDERS=events)")
    moments = pick_moments(feed, load_records(events), sync, n)
    times = [t for m in moments for t in (m.t_video - bracket, m.t_video, m.t_video + bracket)]
    if from_bucket:
        source, options = bucket_source(feed.sample)
    else:
        local = resolve_path("data") / feed.sample
        if not local.exists():
            raise SystemExit(f"{local} missing; on a laptop use --from-bucket (footage is never copied locally)")
        source, options = str(local), None
    frames = grab(source, options, times, width)
    return build(feed, moments, frames, bracket, out_root / feed.clip_id)


def shift(sync_path: Path, by: float, row: int | None) -> list[str]:
    """Records appear `by` seconds later in the video than the table says: lower game_start by `by` in one
    row (0-based) or all rows, so they map `by` seconds later. Returns the changed rows."""
    with open(sync_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{sync_path} has no rows")
    changed = []
    for i, r in enumerate(rows):
        if row is None or i == row:
            r["game_start"] = f"{float(r['game_start']) - by:.3f}"
            changed.append(f"row {i}: game_start {r['game_start']}")
    with open(sync_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    SyncTable.load(sync_path)  # still valid
    return changed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    ap.add_argument("--clip", help="clip_id (sample file stem), or 'demo' for every demo feed")
    ap.add_argument("--from-bucket", action="store_true", help="stream the clip from GCS_BUCKET instead of data/")
    ap.add_argument("--moments", type=int, default=10)
    ap.add_argument("--bracket", type=float, default=5.0, help="seconds before and after each moment")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--out", type=Path, default=Path("data/reports/sync"))
    sh = sub.add_parser("shift", help="move a sync table's game time by a measured offset")
    sh.add_argument("--clip", required=True)
    sh.add_argument("--by", type=float, required=True, help="seconds the event appears LATER than predicted (negative: earlier)")
    sh.add_argument("--row", type=int, default=None, help="0-based row; default every row")
    args = ap.parse_args(argv)

    feeds = load_feeds(load_config().engine.feeds_path).feeds
    if args.cmd == "shift":
        feed = next((f for f in feeds if f.clip_id == args.clip), None)
        if feed is None or not feed.sync:
            raise SystemExit(f"no feed with a sync table has clip {args.clip}")
        for line in shift(resolve_path(feed.sync), args.by, args.row):
            print(line)
        return 0
    if not args.clip:
        ap.error("--clip is required")
    chosen = [f for f in feeds if args.clip in ("demo", f.clip_id)]
    if not chosen:
        raise SystemExit(f"no feed has clip {args.clip}")
    for feed in chosen:
        path = check(feed, args.from_bucket, args.moments, args.bracket, args.width, resolve_path(args.out))
        print(f"{feed.feed_id}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
