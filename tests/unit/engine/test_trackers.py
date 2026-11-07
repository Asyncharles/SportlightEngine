from __future__ import annotations

import pytest

from intensity.config import CommentaryTrackerConfig, CrowdTrackerConfig, RollingConfig, VideoTrackerConfig
from intensity.engine.trackers.commentary import CommentaryTracker
from intensity.engine.trackers.crowd import CrowdTracker
from intensity.engine.trackers.rolling import RollingPercentile, Seeds
from intensity.engine.trackers.video import VideoTracker
from intensity.engine.trackers.voice import VoiceTracker
from intensity.schema import Fact

from .helpers import commentary, crowd, video, voice

ROLL = RollingConfig(window_s=300, min_samples=10, max_samples=600)
SEEDS = Seeds("football", {"loudness_rel_db": [-10, 0, 10], "rise_db_s": [-3, 0, 3], "motion": [0, 0.05, 0.1],
                           "pan_speed": [0, 0.2, 0.4], "prompt": [0, 0.2, 0.4], "energy_rel_db": [-5, 0, 5],
                           "pitch_rel_st": [-3, 0, 3], "pace_rel": [0.5, 1, 1.5], "cuts_per_10s": [0, 3, 6]})


def test_rolling_percentile_blends_seed_then_live():
    rp = RollingPercentile([0, 10], RollingConfig(window_s=100, min_samples=4, max_samples=50))
    assert rp.rank(5) == pytest.approx(0.5)
    for i, v in enumerate([1, 2, 3, 4]):
        rp.add(float(i), v)
    assert rp.rank(2.5) == pytest.approx(0.5)  # live only now
    rp.add(200.0, 9)  # window prunes the first four
    assert len(rp) == 1


def test_rolling_percentile_max_samples():
    rp = RollingPercentile([0, 1], RollingConfig(window_s=1e9, min_samples=1, max_samples=5))
    for i in range(20):
        rp.add(float(i), float(i))
    assert len(rp) == 5 and rp.rank(14.5) == 0.0


def test_crowd_part_rises_with_loudness_and_drops_out():
    tr = CrowdTracker(CrowdTrackerConfig(), ROLL, SEEDS)
    tr.on_event(crowd(1.0, loud=0.0))
    quiet = tr.part(1.0).value
    tr.on_event(crowd(1.5, loud=9.0, facts=[Fact(key="loudness_rel_db", value=9.0, text="crowd 9 dB above normal")]))
    loud = tr.part(1.5)
    assert loud.value > quiet
    assert loud.facts[0].text == "crowd 9 dB above normal"
    assert tr.part(5.0) is None  # 3 s with no crowd events


def test_crowd_panns_cheering_raises_part():
    tr = CrowdTracker(CrowdTrackerConfig(), ROLL, SEEDS)
    tr.on_event(crowd(1.0, loud=0.0))
    base = tr.part(1.0).value
    tr.on_event(crowd(1.5, loud=0.0, events={"cheering": 0.9}))
    part = tr.part(1.5)
    assert part.value > base
    assert any(f.key == "audio_cheering" for f in part.facts)


def test_crowd_cues_rise_and_hush():
    tr = CrowdTracker(CrowdTrackerConfig(), ROLL, SEEDS)
    tr.on_event(crowd(1.0, rise=3.0))
    cues = tr.cues(1.0)
    assert cues.rise_pct > 0.9 and not cues.hush
    tr.on_event(crowd(1.5, hush=True))
    tr.on_event(crowd(2.0))
    assert tr.cues(2.0).hush  # hush holds for 2 s
    tr.on_event(crowd(4.0))
    assert not tr.cues(4.0).hush


def test_crowd_rise_ignored_when_commentator_covers_it():
    tr = CrowdTracker(CrowdTrackerConfig(), ROLL, SEEDS)
    tr.on_event(crowd(1.0, rise=3.0, ok=False))
    assert tr.cues(1.0).rise_pct == 0.5


def test_voice_null_when_not_speaking():
    tr = VoiceTracker(ROLL, SEEDS)
    tr.on_event(voice(1.0, speaking=False))
    cues = tr.cues(1.0)
    assert not cues.speaking and cues.rise == 0.0 and cues.energy_pct is None
    tr.on_event(voice(1.5, energy=5.0, pitch=3.0))
    assert tr.cues(1.5).rise > 0.9
    assert tr.cues(10.0) is None


def test_commentary_lexicon_names_decay_and_quote():
    tr = CommentaryTracker(CommentaryTrackerConfig(decay_s=10, max_quote_words=6), {"goal": 1.0, "chance": 0.6})
    tr.add_names({"Lionel Messi"})
    tr.on_event(commentary(10.0, "and there it is a wonderful goal from Messi after a long move down the left"))
    part = tr.part(10.0)
    assert part.value == pytest.approx(1.0)
    quote = next(f for f in part.facts if f.key == "commentary_quote")
    assert len(str(quote.value).split()) <= 6 and "goal" in str(quote.value)
    assert any(f.key == "commentary_player" and f.value == "Messi" for f in part.facts)
    assert tr.part(15.0).value == pytest.approx(0.5)
    assert tr.part(20.5).value == 0.0
    assert tr.part(75.0) is None  # no ASR for a minute: dropped out


def test_commentary_ignores_non_speech():
    tr = CommentaryTracker(CommentaryTrackerConfig(), {"goal": 1.0})
    tr.on_event(commentary(1.0, "goal", no_speech=0.9))
    assert tr.part(1.0) is None


def test_video_replay_holds_pre_replay_value():
    tr = VideoTracker("football", VideoTrackerConfig(celebration_prompts={"football": ["goal celebration"]}), ROLL, SEEDS)
    tr.on_event(video(1.0, motion=0.09, prompts={"goal celebration": 0.35}))
    before = tr.part(1.0).value
    tr.on_event(video(1.5, shot="replay", motion=0.0, prompts={"goal celebration": 0.0}))
    held = tr.part(1.5)
    assert held.value == before
    assert any(f.text == "replay showing" for f in held.facts)
    assert tr.part(5.0) is None  # drops out after 3 s with no events


def test_video_trained_head_score_replaces_v0():
    tr = VideoTracker("football", VideoTrackerConfig(), ROLL, SEEDS)
    tr.on_event(video(1.0, score=0.83))
    assert tr.part(1.0).value == pytest.approx(0.83)


def test_video_cues_pan_only_on_wide_shots():
    tr = VideoTracker("football", VideoTrackerConfig(), ROLL, SEEDS)
    tr.on_event(video(1.0, pan=0.5, sustained=2.0))
    cues = tr.cues(1.0)
    assert cues.pan_pct > 0.9 and cues.pan_sustained_s == 2.0 and "pan_speed" in cues.facts
    tr.on_event(video(1.5, shot="close_player", pan=0.5, sustained=2.0))
    cues = tr.cues(1.5)
    assert cues.pan_pct == 0.0 and cues.pan_sustained_s == 0.0
