"""Deterministic mock ingest: plausible events for every source, for any feed list.

Each feed loops a virtual clip of `duration_s`. Per loop it schedules "moments" (a goal chance, an
overtake, a lead change, a lift). In the 8 s before a moment the building cues rise (sustained wide
pan, crowd rise, commentator energy); at the moment the just-happened cues fire (roar, celebration,
replay, excited commentary) together with the match record. The output is a pure function of
(seed, feed_id, step), so any slice of time can be regenerated exactly.

    python -m intensity.engine.mock --url ws://127.0.0.1:8000/ingest      # stream live to an engine
    python -m intensity.engine.mock --tape data/tapes/mock_fb.jsonl --feed fb-arg-fra:football
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from intensity.config import load_sport_yaml
from intensity.schema import ClockTick, Control, Envelope, Event, Fact, make_envelope

STEP_S = 0.25  # ClockTicks at 4 Hz
BUILD_S = 8.0
AFTER_S = 8.0

PLAYERS = {
    "football": (["Argentina", "France"], ["Lionel Messi", "Kylian Mbappe", "Angel Di Maria", "Antoine Griezmann"]),
    "f1": (["Red Bull", "Ferrari"], ["Max Verstappen", "Charles Leclerc", "Lando Norris", "Lewis Hamilton"]),
    "nba": (["Celtics", "Warriors"], ["Jayson Tatum", "Stephen Curry", "Jaylen Brown", "Draymond Green"]),
    "weightlifting": (["CHN", "USA"], ["Li Wenwen", "Mary Theisen-Lappen", "Emily Campbell", "Park Hye-jeong"]),
    "chess": (["White", "Black"], ["Magnus Carlsen", "Fabiano Caruana"]),
}

EXCITED = {
    "football": ["what a chance for {p}", "{p} shoots and it's a goal", "huge save denies {p}"],
    "f1": ["{p} dives down the inside, that's an overtake", "side by side through the corner with {p}"],
    "nba": ["{p} for three, it's good and that's the lead", "and one for {p}, what a play"],
    "weightlifting": ["good lift from {p}, a personal best", "{p} makes it look easy, good lift"],
    "chess": ["{p} plays a stunning sacrifice"],
}
CALM = ["they knock it about in midfield", "a quiet spell here", "plenty still to play for", "settling into a rhythm"]


def _hash(feed_id: str) -> int:
    return zlib.crc32(feed_id.encode())


@dataclass
class MockFeed:
    feed_id: str
    sport: str
    seed: int = 7
    duration_s: float = 600.0
    offset_s: float = 0.0

    def __post_init__(self) -> None:
        prompts = load_sport_yaml("config/prompts", self.sport)
        self.prompts: list[str] = list(prompts.get("sport", []))
        self.teams, self.players = PLAYERS.get(self.sport, PLAYERS["football"])
        self._moments: dict[int, list[float]] = {}

    # -------------------------------------------------------------- schedule

    def moments(self, epoch: int) -> list[float]:
        """Ground-truth moment times (t_video) for one loop of the clip."""
        if epoch not in self._moments:
            rng = np.random.default_rng([self.seed, _hash(self.feed_id), epoch, 1])
            lo, hi = (60.0, 80.0) if self.sport == "weightlifting" else (40.0, 90.0)
            t, out = 30.0 + rng.uniform(0, 10), []
            while t < self.duration_s - 15:
                out.append(round(float(t), 2))
                t += rng.uniform(lo, hi)
            self._moments[epoch] = out
        return self._moments[epoch]

    def _phase(self, tv: float, epoch: int) -> tuple[float, float, float | None]:
        """(build 0..1, after 0..1, nearest moment) at t_video."""
        build, after, nearest = 0.0, 0.0, None
        for m in self.moments(epoch):
            if m - BUILD_S <= tv < m:
                build = max(build, (tv - (m - BUILD_S)) / BUILD_S)
                nearest = m
            elif m <= tv < m + AFTER_S:
                after = max(after, 1.0 - (tv - m) / AFTER_S)
                nearest = m
        return build, after, nearest

    def _player(self, epoch: int, m: float) -> tuple[str, str]:
        i = int(m * 10 + epoch) % len(self.players)
        return self.teams[i % len(self.teams)], self.players[i]

    # -------------------------------------------------------------- one step

    def step(self, k: int, t0_ns: int) -> list[Envelope]:
        t = k * STEP_S
        abs_tv = self.offset_s + t
        epoch = int(abs_tv // self.duration_s)
        tv = round(abs_tv - epoch * self.duration_s, 4)
        emit = t0_ns + int(t * 1e9)
        rng = np.random.default_rng([self.seed, _hash(self.feed_id), k])
        out: list[Envelope] = []

        if k > 0 and int((self.offset_s + t - STEP_S) // self.duration_s) < epoch:
            out.append(make_envelope(Control(feed_id=self.feed_id, kind="loop", loop_epoch=epoch, t_emit_ns=emit)))
        out.append(make_envelope(ClockTick(feed_id=self.feed_id, t_video=tv, loop_epoch=epoch, t_emit_ns=emit)))

        build, after, nearest = self._phase(tv, epoch)
        for rec in self._match_records(tv, epoch, rng):
            out.append(self._event("match", tv, epoch, emit, rec))
        if k % 2 == 0:
            out.append(self._video(tv, epoch, emit, build, after, nearest, rng))
            out.append(self._crowd(tv, epoch, emit, build, after, nearest, rng))
            out.append(self._voice(tv, epoch, emit, build, after, rng))
        if k % 4 == 0:
            out.append(self._event("ocr", tv, epoch, emit, {"fields": {"clock": f"{int(tv // 60)}:{int(tv % 60):02d}"}, "conf": {"clock": 0.9}}))
        com = self._commentary(tv, epoch, emit, k, nearest, rng)
        if com is not None:
            out.append(com)
        return out

    def _event(self, source: str, tv: float, epoch: int, emit: int, payload: dict, facts: list[Fact] | None = None) -> Envelope:
        return make_envelope(
            Event(
                feed_id=self.feed_id,
                sport=self.sport,  # type: ignore[arg-type]
                source=source,  # type: ignore[arg-type]
                t_video=tv,
                loop_epoch=epoch,
                t_emit_ns=emit,
                payload=payload,
                facts=facts or [],
            )
        )

    def _video(self, tv, epoch, emit, build, after, nearest, rng) -> Envelope:
        replay = nearest is not None and after > 0 and tv - nearest >= 2.0 and self.sport != "weightlifting"
        if replay:
            shot = "replay"
        elif after > 0 and nearest is not None and tv - nearest < 2.0:
            shot = "close_player"
        elif build > 0:
            shot = "wide"
        else:
            shot = "wide" if rng.random() < 0.8 else str(rng.choice(["close_player", "close_crowd"]))
        pan_speed = max(0.0, 0.08 + 0.55 * build + rng.normal(0, 0.03)) if shot == "wide" else abs(rng.normal(0, 0.03))
        sustained = round(build * BUILD_S, 2) if shot == "wide" and build > 0 and pan_speed > 0.15 else 0.0
        motion = max(0.0, 0.03 + 0.05 * build + 0.08 * after + rng.normal(0, 0.008))
        if self.sport == "weightlifting":
            motion, pan_speed, sustained = self._lift_motion(tv, nearest, rng)
        prompts = self._prompts(tv, build, after, nearest, rng)
        payload = {
            "clip_seq": int(round(tv / 0.5)),
            "t_video_start": round(max(0.0, tv - 2.0), 3),
            "motion": round(motion, 4),
            "cuts_per_10s": round(2.0 + 4.0 * after + abs(rng.normal(0, 0.5)), 2),
            "shot_type": shot,
            "shot_conf": round(0.7 + 0.25 * rng.random(), 3),
            "replay": shot == "replay",
            "replay_conf": round(0.9 if shot == "replay" else 0.05, 3),
            "prompts": prompts,
            "pan_dx": round(pan_speed, 4),
            "pan_dy": 0.0,
            "pan_speed": round(pan_speed, 4),
            "pan_sustained_s": sustained,
            "score": None,
            "model": "mock-v0",
        }
        facts = [Fact(key="shot_type", value=shot, text=f"{shot.replace('_', ' ')} shot")]
        if shot == "wide":
            facts.append(Fact(key="pan_speed", value=round(pan_speed, 2), unit="fw/s", text=f"camera panning {pan_speed:.1f} frame widths a second"))
            facts.append(Fact(key="motion", value=round(motion, 3), text="fast action on a wide shot"))
        return self._event("video", tv, epoch, emit, payload, facts)

    def _lift_motion(self, tv, nearest, rng) -> tuple[float, float, float]:
        if nearest is not None and nearest <= tv < nearest + 1.5:
            return 0.2 + 0.05 * rng.random(), 0.6, 0.0  # the lift: motion and pan spike
        return max(0.0, 0.02 + rng.normal(0, 0.005)), abs(rng.normal(0, 0.02)), 0.0

    def _prompts(self, tv, build, after, nearest, rng) -> dict[str, float]:
        scores = {p: round(float(0.1 + 0.05 * rng.random()), 3) for p in self.prompts}

        def bump(name: str, value: float) -> None:
            if name in scores:
                scores[name] = round(min(1.0, value + 0.03 * rng.random()), 3)

        if self.sport == "weightlifting":
            nxt = self._next_moment(tv)
            last = nearest if nearest is not None and nearest <= tv else None
            if last is not None and tv - last < 5.0:
                bump("athlete celebrating a lift", 0.5)
            elif nxt is not None and nxt - 25 <= tv < nxt - 15:
                bump("athlete walking onto the platform", 0.45)
            elif nxt is not None and nxt - 15 <= tv < nxt - 5:
                bump("athlete chalking hands", 0.45)
            elif nxt is not None and nxt - 5 <= tv < nxt + 1.5:
                bump("athlete standing over the barbell", 0.5)
            else:
                bump("empty platform", 0.4)
            return scores
        for name in ("two racing cars close together", "players running on a counter attack", "fast break down the court"):
            if build > 0:
                bump(name, 0.15 + 0.35 * build)
        if after > 0 and nearest is not None and tv - nearest < 4.0:
            for name in ("goal celebration", "players hugging", "players celebrating a basket", "team celebrating in pit lane", "bench celebrating"):
                bump(name, 0.2 + 0.4 * after)
        return scores

    def _next_moment(self, tv: float) -> float | None:
        epoch_moments = self.moments(int((self.offset_s + tv) // self.duration_s))
        for m in epoch_moments:
            if m + 1.5 > tv:
                return m
        return None

    def _crowd(self, tv, epoch, emit, build, after, nearest, rng) -> Envelope:
        hush = False
        rise = 3.0 * build - 2.0 * after + rng.normal(0, 0.3)
        loud = rng.normal(0, 1.5) + 3.0 * build + 12.0 * after
        if self.sport == "weightlifting":
            nxt = self._next_moment(tv)
            hush = nxt is not None and nxt - 5 <= tv < nxt
            loud = rng.normal(0, 1.0) + (-8.0 if hush else 0.0) + 12.0 * after
            rise = rng.normal(0, 0.3) - (3.0 if hush and tv < (nxt or 0) - 4 else 0.0) + 4.0 * after
        speech = float(np.clip(0.5 + rng.normal(0, 0.2), 0, 1))
        payload = {
            "window_s": 1.0,
            "loudness_dbfs": round(-28.0 + loud, 2),
            "loudness_rel_db": round(loud, 2),
            "peak_dbfs": round(-18.0 + loud, 2),
            "nonspeech_dbfs": round(-32.0 + loud, 2),
            "nonspeech_ok": speech <= 0.8,
            "rise_db_s": round(rise, 2),
            "hush": hush,
            "events": None,
        }
        word = "above" if loud >= 0 else "below"
        text = f"crowd {abs(loud):.0f} dB {word} normal" if abs(loud) >= 0.5 else "crowd at its usual level"
        facts = [Fact(key="loudness_rel_db", value=round(loud, 1), unit="dB", text=text)]
        if rise > 0.5:
            facts.append(Fact(key="rise_db_s", value=round(rise, 1), unit="dB/s", text=f"crowd rising {rise:.0f} dB a second"))
        return self._event("crowd", tv, epoch, emit, payload, facts)

    def _voice(self, tv, epoch, emit, build, after, rng) -> Envelope:
        speaking = bool(rng.random() < 0.8 or build > 0.3 or after > 0.3)
        if not speaking:
            payload = {"window_s": 0.5, "speech_frac": round(float(rng.uniform(0, 0.25)), 2), "speaking": False,
                       "energy_rel_db": None, "pitch_rel_st": None, "pace_rel": None}
            return self._event("voice", tv, epoch, emit, payload)
        energy = rng.normal(0, 1.5) + 6.0 * build + 8.0 * after
        pitch = rng.normal(0, 0.8) + 3.0 * build + 4.0 * after
        pace = max(0.3, 1.0 + rng.normal(0, 0.08) + 0.4 * build + 0.3 * after)
        payload = {"window_s": 0.5, "speech_frac": round(float(rng.uniform(0.5, 1.0)), 2), "speaking": True,
                   "energy_rel_db": round(energy, 2), "pitch_rel_st": round(pitch, 2), "pace_rel": round(pace, 3)}
        facts = [
            Fact(key="energy_rel_db", value=round(energy, 1), unit="dB", text=f"commentator {abs(energy):.0f} dB {'louder' if energy >= 0 else 'quieter'} than usual"),
            Fact(key="pitch_rel_st", value=round(pitch, 1), unit="st", text=f"commentator voice {'up' if pitch >= 0 else 'down'} {abs(pitch):.0f} semitones"),
        ]
        return self._event("voice", tv, epoch, emit, payload, facts)

    def _commentary(self, tv, epoch, emit, k, nearest, rng) -> Envelope | None:
        text = None
        if nearest is not None and abs(tv - (nearest + 2.5)) < STEP_S / 2:  # ASR lag after the moment
            _, player = self._player(epoch, nearest)
            lines = EXCITED.get(self.sport, EXCITED["football"])
            text = lines[int(nearest) % len(lines)].format(p=player)
        elif k % 24 == 0 and nearest is None:
            text = CALM[(k // 24) % len(CALM)]
        if text is None:
            return None
        payload = {"text": text, "t_start": round(max(0.0, tv - 2.0), 2), "t_end": tv, "lang": "en",
                   "avg_logprob": round(float(-0.2 - 0.2 * rng.random()), 3), "no_speech_prob": 0.02}
        return self._event("commentary", tv, epoch, emit, payload)

    # -------------------------------------------------------------- match data

    def _match_records(self, tv: float, epoch: int, rng) -> list[dict]:
        """Records whose game time falls in [tv, tv + STEP_S). Game time equals t_video (identity sync)."""
        recs: list[dict] = []
        lo, hi = tv, tv + STEP_S

        def rec(kind: str, gt: float, team=None, player=None, period=1, **attrs) -> dict:
            return {"game_time_s": round(gt, 2), "period": period, "kind": kind, "team": team, "player": player, "attrs": attrs}

        moments = self.moments(epoch)
        if self.sport == "football":
            if epoch == 0 and lo <= 0.0 < hi:
                recs.append(rec("kickoff", 0.0, team=self.teams[0]))
            for m in moments:
                if lo <= m < hi:
                    team, player = self._player(epoch, m)
                    goal = int(m) % 3 == 0
                    recs.append(rec("goal" if goal else "shot", m, team=team, player=player, xg=round(0.15 + (int(m) % 5) * 0.08, 2)))
        elif self.sport == "f1":
            lap_s = 90.0
            if int(lo // lap_s) != int(hi // lap_s) or lo == 0.0:
                lap = int(hi // lap_s) + 1
                recs.append(rec("lap", hi, lap=lap, total_laps=58, gaps_s=[2.2, 1.8, 3.1, 4.5]))
            for m in moments:
                lap = int(m // lap_s) + 1
                if lo <= m - BUILD_S < hi:  # the chasing car closes in
                    recs.append(rec("lap", m - BUILD_S, lap=lap, total_laps=58, gaps_s=[0.4, 1.7, 3.0, 4.4]))
                if lo <= m < hi:
                    team, player = self._player(epoch, m)
                    recs.append(rec("pass", m, team=team, player=player))
                if lo <= m + 2 < hi:  # the pass is done and the gap opens again
                    recs.append(rec("lap", m + 2, lap=lap, total_laps=58, gaps_s=[2.4, 1.6, 3.0, 4.4]))
        elif self.sport == "nba":
            q = lambda gt: min(4, 1 + int(gt // 720))  # noqa: E731
            state = self._nba_score(epoch, lo)
            for i, (gt, team, pts, h, a) in enumerate(state["events"]):
                if lo <= gt < hi:
                    player = self.players[i % len(self.players)]
                    recs.append(rec("score", gt, team=team, player=player, period=q(gt), points=pts, home_score=h,
                                    away_score=a, home=self.teams[0], away=self.teams[1]))
        elif self.sport == "weightlifting":
            for i, m in enumerate(moments):
                _, player = self._player(epoch, m)
                if lo <= m - 25 < hi:
                    best = 120 + (i % 4) * 5
                    recs.append(rec("attempt_called", m - 25, player=player, weight_kg=best + (i % 3) * 2,
                                    attempt=1 + i % 3, best_kg=best, eta_game_s=m))
                if lo <= m + 3 < hi:
                    recs.append(rec("attempt_result", m + 3, player=player, good=(i % 3) != 2))
        return recs

    def _nba_score(self, epoch: int, upto: float) -> dict:
        """A deterministic scoring sequence: a basket every ~24 s, with moments flipping the lead."""
        key = ("nba", epoch)
        cache = getattr(self, "_nba_cache", {})
        if key not in cache:
            rng = np.random.default_rng([self.seed, _hash(self.feed_id), epoch, 2])
            moments = set(self.moments(epoch))
            h = a = 0
            events = []
            t = 10.0
            mlist = sorted(moments)
            while t < self.duration_s:
                nxt = next((m for m in mlist if m >= t), None)
                if nxt is not None and nxt < t + 24:
                    t = nxt
                    trailing = self.teams[1] if h >= a else self.teams[0]
                    pts = 3
                    if trailing == self.teams[0]:
                        h += pts if a - h < 3 else a - h + 1
                    else:
                        a += pts if h - a < 3 else h - a + 1
                    events.append((round(t, 2), trailing, pts, h, a))
                    t += 6
                    mlist.remove(nxt)
                    continue
                team = self.teams[int(rng.integers(0, 2))]
                pts = int(rng.choice([2, 2, 3]))
                if team == self.teams[0]:
                    h += pts
                else:
                    a += pts
                events.append((round(t, 2), team, pts, h, a))
                t += float(rng.uniform(15, 30))
            cache[key] = {"events": events}
            self._nba_cache = cache
        return cache[key]


class MockGenerator:
    def __init__(self, feeds: list[tuple[str, str]], seed: int = 7, duration_s: float = 600.0, t0_ns: int | None = None):
        self.feeds = [MockFeed(fid, sport, seed, duration_s, offset_s=17.0 * i) for i, (fid, sport) in enumerate(feeds)]
        self.t0_ns = time.time_ns() if t0_ns is None else t0_ns

    def envelopes(self, t_from: float, t_to: float) -> list[Envelope]:
        """Everything every feed sends in mock time [t_from, t_to), in send order."""
        out: list[Envelope] = []
        k0 = int(np.ceil(t_from / STEP_S - 1e-9))
        k1 = int(np.ceil(t_to / STEP_S - 1e-9))
        for k in range(k0, k1):
            for feed in self.feeds:
                out.extend(feed.step(k, self.t0_ns))
        return out

    def feed(self, feed_id: str) -> MockFeed:
        return next(f for f in self.feeds if f.feed_id == feed_id)


def write_tape(path: Path, envelopes: list[Envelope]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for env in envelopes:
            fh.write(env.model_dump_json() + "\n")


def read_tape(path: Path) -> list[Envelope]:
    with open(path) as fh:
        return [Envelope.model_validate_json(line) for line in fh if line.strip()]


def _feeds_arg(values: list[str] | None) -> list[tuple[str, str]]:
    if values:
        return [tuple(v.split(":", 1)) for v in values]  # type: ignore[misc]
    from intensity.feeds import load_feeds

    return [(f.feed_id, f.sport) for f in load_feeds().feeds]


async def _stream(url: str, gen: MockGenerator, duration: float | None) -> None:
    from intensity.transport import IngestClient

    client = IngestClient(url)
    sender = asyncio.create_task(client.run())
    start = time.monotonic()
    t = 0.0
    try:
        while duration is None or t < duration:
            for env in gen.envelopes(t, t + STEP_S):
                client.put(env)
            t += STEP_S
            await asyncio.sleep(max(0.0, start + t - time.monotonic()))
    finally:
        client.stop()
        await sender


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feed", action="append", help="feed_id:sport (default: config/feeds.yaml)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--clip-s", type=float, default=600.0, help="virtual clip length before it loops")
    ap.add_argument("--duration", type=float, default=None, help="seconds of mock time to produce")
    ap.add_argument("--url", help="stream in real time to this /ingest URL")
    ap.add_argument("--tape", type=Path, help="write a tape (JSONL of envelopes) instead")
    args = ap.parse_args(argv)
    gen = MockGenerator(_feeds_arg(args.feed), args.seed, args.clip_s, t0_ns=0 if args.tape else None)
    if args.tape:
        envs = gen.envelopes(0.0, args.duration or args.clip_s)
        write_tape(args.tape, envs)
        moments = {f.feed_id: f.moments(0) for f in gen.feeds}
        args.tape.with_suffix(".moments.json").write_text(json.dumps(moments, indent=1))
        print(f"wrote {len(envs)} envelopes to {args.tape}")
        return 0
    if not args.url:
        ap.error("give --url or --tape")
    asyncio.run(_stream(args.url, gen, args.duration))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
