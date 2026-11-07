from __future__ import annotations

import pytest
from pydantic import ValidationError

from intensity.feeds import FeedRegistry, load_feeds


def test_repo_registry_loads():
    reg = load_feeds()
    assert {f.sport for f in reg.feeds} >= {"football", "f1", "nba", "weightlifting"}
    fb = reg.get("fb-arg-fra")
    assert fb.clip_id == "arg_fra_2022_final"
    assert fb.asr == "live"


def _feed(fid: str, match_id=None, demo=True) -> dict:
    return {"feed_id": fid, "sport": "football", "sample": f"samples/{fid}.mp4", "rtsp_path": fid,
            "match_ref": {"source": "statsbomb", "match_id": match_id}, "demo": demo}


def test_duplicate_feed_ids_rejected():
    with pytest.raises(ValidationError):
        FeedRegistry.model_validate({"feeds": [_feed("a"), _feed("a")]})


def test_demo_match_ids_drive_the_split():
    reg = FeedRegistry.model_validate({"feeds": [_feed("a", 3869685), _feed("b", 1, demo=False), _feed("c")]})
    assert reg.demo_match_ids("football") == {"3869685"}
