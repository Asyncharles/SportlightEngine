from __future__ import annotations

from fractions import Fraction

import av
import numpy as np

from intensity.engine.syncsheet import build, grab_frames, key_records
from intensity.feeds import Feed
from intensity.schema import MatchPayload
from intensity.sync import SyncRow, SyncTable


def _synthetic_mp4(path, seconds=6, fps=10):
    """Each frame's brightness encodes its second, so we can check which frame was grabbed."""
    with av.open(str(path), "w") as out:
        stream = out.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.time_base = Fraction(1, fps)
        for i in range(seconds * fps):
            img = np.full((48, 64, 3), (i // fps) * 40, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def test_grab_frames_returns_jpegs(tmp_path):
    clip = tmp_path / "c.mp4"
    _synthetic_mp4(clip)
    frames = grab_frames(clip, [1.0, 4.5], width=32)
    assert set(frames) == {1.0, 4.5}
    assert all(b[:2] == b"\xff\xd8" for b in frames.values())  # JPEG magic


def test_build_sheet_maps_records_through_sync(tmp_path):
    clip = tmp_path / "arg_fra.mp4"
    _synthetic_mp4(clip)
    feed = Feed(feed_id="fb", sport="football", sample="samples/arg_fra.mp4", rtsp_path="fb",
                match_ref={"source": "statsbomb", "match_id": 1})
    sync = SyncTable([SyncRow("arg_fra", 0, 3, 100, 1), SyncRow("arg_fra", 4, 6, 104, 1)])
    recs = [MatchPayload(game_time_s=101, period=1, kind="shot", team="Argentina", player="Messi"),
            MatchPayload(game_time_s=103.5, period=1, kind="goal", team="France"),  # in the gap
            MatchPayload(game_time_s=105, period=1, kind="goal", team="France"),
            MatchPayload(game_time_s=105, period=1, kind="pass")]
    assert [r.kind for r in key_records("football", recs)] == ["shot", "goal", "goal"]
    page = build(feed, recs, sync, clip, tmp_path / "out")
    html = page.read_text()
    assert "t_video 1.00s" in html and "t_video 5.00s" in html and "1 fall in sync gaps" in html
    assert len(list((tmp_path / "out").glob("*.jpg"))) == 2
