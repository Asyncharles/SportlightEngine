"""Sync sheet: `make sync-sheet CLIP=<clip_id>`.

Renders the sample's frame at the `t_video` of every key match record (goal, shot, pass, lead change,
lift), so a human can check the sync table by eye. Writes `reports/sync/<clip_id>/index.html` with
JPEG thumbnails beside it. Decoding is PyAV on CPU and only touches local files.
"""

from __future__ import annotations

import argparse
import html
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av

from intensity.config import load_config, resolve_path
from intensity.engine.replay import load_records
from intensity.feeds import Feed, load_feeds
from intensity.schema import MatchPayload
from intensity.sync import SyncTable

KEY_KINDS = {
    "football": {"goal", "shot"},
    "f1": {"pass", "safety_car"},
    "nba": {"score"},
    "weightlifting": {"attempt_called", "attempt_result"},
}


@dataclass
class Shot:
    record: MatchPayload
    t_video: float
    image: str | None


def key_records(sport: str, records: list[MatchPayload]) -> list[MatchPayload]:
    kinds = KEY_KINDS.get(sport, set())
    return [r for r in records if r.kind in kinds]


def _encode_jpeg(frame: av.VideoFrame, width: int) -> bytes:
    height = max(2, int(frame.height * width / frame.width) // 2 * 2)
    small = frame.reformat(width=width, height=height, format="yuvj420p")
    codec = av.CodecContext.create("mjpeg", "w")
    codec.width, codec.height, codec.pix_fmt = width, height, "yuvj420p"
    codec.time_base = small.time_base or Fraction(1, 25)
    packets = list(codec.encode(small)) + list(codec.encode(None))
    return b"".join(bytes(p) for p in packets)


def grab_frames(video: Path, times: list[float], width: int = 320) -> dict[float, bytes]:
    """Nearest decoded frame at or after each time, by seeking to the keyframe before it."""
    out: dict[float, bytes] = {}
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        tb = stream.time_base
        for t in sorted(set(times)):
            container.seek(int(max(0.0, t - 1.0) / tb), stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts is None or float(frame.pts * tb) + 1e-3 >= t:
                    out[t] = _encode_jpeg(frame, width)
                    break
    return out


def build(feed: Feed, records: list[MatchPayload], sync: SyncTable, video: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    shots: list[Shot] = []
    keyed = [(r, sync.t_video(r.game_time_s, r.period)) for r in key_records(feed.sport, records)]
    frames = grab_frames(video, [t for _, t in keyed if t is not None]) if video.exists() else {}
    for i, (rec, t) in enumerate(keyed):
        if t is None:
            continue
        name = None
        if t in frames:
            name = f"{i:04d}.jpg"
            (out_dir / name).write_bytes(frames[t])
        shots.append(Shot(rec, t, name))
    cells = "".join(
        f'<figure>{f"<img src={s.image!r} alt=frame>" if s.image else "<div class=missing>no frame</div>"}'
        f"<figcaption><b>{html.escape(s.record.kind)}</b> {html.escape(s.record.team or '')} "
        f"{html.escape(s.record.player or '')}<br>game {s.record.game_time_s:.0f}s P{s.record.period} → t_video {s.t_video:.2f}s"
        "</figcaption></figure>"
        for s in shots
    )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Sync sheet</title>
<style>:root{{--bg:#fff;--fg:#111;--muted:#666}}@media (prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#eee;--muted:#aaa}}}}
body{{font:14px system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:16px}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,320px),1fr));gap:12px}}
figure{{margin:0}}img{{width:100%;height:auto;border-radius:6px}}figcaption{{color:var(--muted);font-size:12px}}
.missing{{aspect-ratio:16/9;display:grid;place-items:center;border:1px dashed var(--muted)}}</style></head>
<body><h1>Sync sheet: {html.escape(feed.clip_id)}</h1>
<p>{len(shots)} key records from {html.escape(feed.match_ref.source)} match {html.escape(str(feed.match_ref.match_id))};
{len(keyed) - len(shots)} fall in sync gaps. Check each frame shows the event.</p><main>{cells}</main></body></html>"""
    path = out_dir / "index.html"
    path.write_text(page)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True, help="clip_id (the sample's file stem)")
    ap.add_argument("--out", type=Path, default=Path("reports/sync"))
    args = ap.parse_args(argv)
    config = load_config()
    feed = next((f for f in load_feeds(config.engine.feeds_path).feeds if f.clip_id == args.clip), None)
    if feed is None:
        raise SystemExit(f"no feed in feeds.yaml has clip {args.clip}")
    if not feed.sync or not resolve_path(feed.sync).exists():
        raise SystemExit(f"sync table missing for {args.clip}: {feed.sync}")
    if feed.match_ref.match_id is None:
        raise SystemExit(f"match_ref.match_id is not set for {feed.feed_id} (filled by the ML lead)")
    events = resolve_path(config.engine.replay.events_dir) / feed.sport / f"{feed.match_ref.match_id}.jsonl"
    if not events.exists():
        raise SystemExit(f"match records missing: {events} (training agent, step 1)")
    video = resolve_path("data") / feed.sample
    path = build(feed, load_records(events), SyncTable.load(resolve_path(feed.sync)), video, resolve_path(args.out) / args.clip)
    print(f"sync sheet: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
