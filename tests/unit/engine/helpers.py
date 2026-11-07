"""Builders for engine test inputs."""

from __future__ import annotations

from intensity.schema import Event, Fact


def ev(source: str, t: float, payload: dict, sport: str = "football", feed: str = "f", epoch: int = 0,
       facts: list[Fact] | None = None) -> Event:
    return Event(feed_id=feed, sport=sport, source=source, t_video=t, loop_epoch=epoch, t_emit_ns=0,
                 payload=payload, facts=facts or [])


def crowd(t: float, loud: float = 0.0, rise: float = 0.0, hush: bool = False, ok: bool = True, events=None, **kw) -> Event:
    return ev("crowd", t, {"window_s": 1.0, "loudness_dbfs": -28 + loud, "loudness_rel_db": loud, "peak_dbfs": -18.0,
                           "nonspeech_dbfs": -32 + loud, "nonspeech_ok": ok, "rise_db_s": rise, "hush": hush,
                           "events": events}, **kw)


def voice(t: float, energy: float | None = 0.0, pitch: float | None = 0.0, pace: float | None = 1.0, speaking: bool = True, **kw) -> Event:
    if not speaking:
        energy = pitch = pace = None
    return ev("voice", t, {"window_s": 0.5, "speech_frac": 0.9 if speaking else 0.1, "speaking": speaking,
                           "energy_rel_db": energy, "pitch_rel_st": pitch, "pace_rel": pace}, **kw)


def video(t: float, shot: str = "wide", motion: float = 0.03, pan: float = 0.05, sustained: float = 0.0,
          prompts: dict | None = None, score: float | None = None, conf: float = 0.9, **kw) -> Event:
    return ev("video", t, {"clip_seq": int(t * 2), "t_video_start": max(0.0, t - 2), "motion": motion, "cuts_per_10s": 2.0,
                           "shot_type": shot, "shot_conf": conf, "replay": shot == "replay",
                           "replay_conf": 0.9 if shot == "replay" else 0.05, "prompts": prompts or {}, "pan_dx": pan,
                           "pan_dy": 0.0, "pan_speed": pan, "pan_sustained_s": sustained, "score": score, "model": "test"}, **kw)


def commentary(t: float, text: str, dur: float = 2.0, no_speech: float = 0.0, **kw) -> Event:
    return ev("commentary", t, {"text": text, "t_start": t - dur, "t_end": t, "lang": "en", "avg_logprob": -0.2,
                                "no_speech_prob": no_speech}, **kw)


def match(t: float, kind: str, team=None, player=None, period: int = 1, **attrs) -> Event:
    sport = attrs.pop("sport", "football")
    return ev("match", t, {"game_time_s": t, "period": period, "kind": kind, "team": team, "player": player,
                           "attrs": attrs}, sport=sport)
