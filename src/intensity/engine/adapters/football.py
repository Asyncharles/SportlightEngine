"""Football adapter (owner: training agent).

Match records (from `training/sources/football.py`; game time is the broadcast clock, so the
second half starts at 2700 s):

- `kickoff` (team, attrs: home, away, stage), `period_start`, `period_end`
- `shot`, `goal` (team, player, attrs: xg, outcome, shot_type, x, y, shootout, own_goal)
- `pass` (attrs: completed, pass_type, cross, x, y, end_x, end_y), `carry` (x, y, end_x, end_y)
- `pressure` (x, y), `foul` (x, y), `card` (attrs: color), `substitution` (attrs: replacement)

Coordinates are StatsBomb's, attacking towards x = 120 for the team on the ball. Windows reset at
each period start. Shoot-out goals count separately and never change the score.

The attack features follow the team in possession (the team of the last pass, carry or shot): where
the ball is in its attacking frame, how long it has had the ball, and what it has done in the last
`windows_s.instant` seconds. A 30 s forecast needs this; the 2 and 5 min windows are too slow.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from intensity.engine.adapters.base import Adapter, AdapterPart, clamp01
from intensity.schema import Fact, MatchPayload

_WORDS = {1: "one", 2: "two", 3: "three"}
LAST_EVENT = ("none", "kickoff", "pass", "carry", "pressure", "shot", "goal", "foul", "card", "substitution")
_ON_BALL = {"pass", "carry", "pressure", "shot", "goal", "foul"}
_WINDOWS = ("shot", "entry", "box", "pass", "pressure", "corner")


class FootballAdapter(Adapter):
    sport = "football"
    FEATURES = (
        "score_diff",
        "abs_score_diff",
        "minute",
        "period",
        "stage_stakes",
        "red_cards_home",
        "red_cards_away",
        "yellow_cards_home",
        "yellow_cards_away",
        "shots_2min_home",
        "shots_2min_away",
        "xg_2min_home",
        "xg_2min_away",
        "shots_5min_home",
        "shots_5min_away",
        "xg_5min_home",
        "xg_5min_away",
        "final_third_entries_2min_home",
        "final_third_entries_2min_away",
        "box_touches_2min_home",
        "box_touches_2min_away",
        "possession_5min_home",
        "pressures_2min_home",
        "pressures_2min_away",
        "corners_5min_home",
        "corners_5min_away",
        "last_event_type",
        "last_event_home",
        "last_event_x",
        "last_event_y",
        "secs_since_shot",
        "attack_home",
        "attack_x",
        "attack_dist_goal",
        "possession_s",
        "possession_passes",
        "shots_30s_attack",
        "entries_30s_attack",
        "box_touches_30s_attack",
        "corners_30s_attack",
    )
    LABEL = "big_chance"
    HORIZON_S = 30.0

    def reset(self) -> None:
        self.home: str | None = None
        self.away: str | None = None
        self.stage = ""
        self.goals: dict[str, int] = {}
        self.shootout: dict[str, int] = {}
        self.yellows: dict[str, int] = {}
        self.reds: dict[str, int] = {}
        self.period = 1
        self.game_time = 0.0
        self.windows: dict[str, deque[tuple[float, str, float]]] = {k: deque() for k in _WINDOWS}
        self.last: tuple[str, str | None, float, float] = ("none", None, 60.0, 40.0)
        self.last_shot_t: float | None = None
        self.poss_team: str | None = None
        self.poss_since = 0.0
        self.poss_passes = 0
        self.ball = (60.0, 40.0)  # in the possession team's attacking frame
        self.names: set[str] = set()
        self._cache: tuple[float, int, dict[str, float]] | None = None
        self._seq = 0  # bumped by every record, so cached values never outlive new state

    # ------------------------------------------------------------------ records

    def _team(self, team: str | None) -> None:
        if not team:
            return
        if self.home is None:
            self.home = team
        elif self.away is None and team != self.home:
            self.away = team
        self.goals.setdefault(team, 0)

    def on_match(self, record: MatchPayload) -> None:
        a, team, kind, t = record.attrs, record.team, record.kind, record.game_time_s
        self._seq += 1
        if record.period != self.period:
            self.period = record.period
            for win in self.windows.values():
                win.clear()
            self.last_shot_t = None
            self.poss_team = None
            self.ball = (60.0, 40.0)
        self.game_time = t
        if kind == "kickoff":
            self.home = a.get("home", self.home)
            self.away = a.get("away", self.away)
            self.stage = a.get("stage", self.stage)
        self._team(team)
        for name in (record.player, a.get("replacement")):
            if name:
                self.names.add(name)
        pitch = self.cfg.get("pitch", {})
        ftx = float(pitch.get("final_third_x", 80))
        x, y = a.get("x"), a.get("y")
        end_x, end_y = a.get("end_x"), a.get("end_y")

        if kind in ("shot", "goal") and team:
            if a.get("shootout"):
                if kind == "goal":
                    self.shootout[team] = self.shootout.get(team, 0) + 1
            elif kind == "goal":
                self.goals[team] = self.goals.get(team, 0) + 1
            if not a.get("own_goal"):
                self.windows["shot"].append((t, team, float(a.get("xg", 0.05))))
                self.last_shot_t = t
                if self._in_box(x, y):
                    self.windows["box"].append((t, team, 1.0))
        elif kind == "pass" and team:
            self.windows["pass"].append((t, team, 1.0))
            if a.get("pass_type") == "Corner":
                self.windows["corner"].append((t, team, 1.0))
            if a.get("completed", True):
                self._progress(t, team, x, end_x, end_y, ftx)
        elif kind == "carry" and team:
            self._progress(t, team, x, end_x, end_y, ftx)
        elif kind == "pressure" and team:
            self.windows["pressure"].append((t, team, 1.0))
        elif kind == "card" and team:
            if a.get("color") == "yellow":
                self.yellows[team] = self.yellows.get(team, 0) + 1
            else:
                self.reds[team] = self.reds.get(team, 0) + 1
        self._possession(record)
        if kind in _ON_BALL or kind in ("kickoff", "card", "substitution"):
            last_x = float(end_x if end_x is not None else x if x is not None else self.last[2])
            last_y = float(end_y if end_y is not None else y if y is not None else self.last[3])
            self.last = (kind, team, last_x, last_y)

    def _possession(self, record: MatchPayload) -> None:
        a, team, kind, t = record.attrs, record.team, record.kind, record.game_time_s
        x, y = a.get("x"), a.get("y")
        if not team or x is None or y is None:
            return
        if kind in ("pass", "carry", "shot", "goal") and not a.get("own_goal"):
            if team != self.poss_team:
                self.poss_team, self.poss_since, self.poss_passes = team, t, 0
            if kind == "pass" and a.get("completed", True):
                self.poss_passes += 1
            end_x, end_y = a.get("end_x"), a.get("end_y")
            self.ball = (float(end_x), float(end_y)) if end_x is not None and end_y is not None else (x, y)
        elif kind in ("pressure", "foul") and team != self.poss_team:
            self.ball = (120.0 - float(x), 80.0 - float(y))  # the defender's frame, flipped

    def _in_box(self, x: float | None, y: float | None) -> bool:
        if x is None or y is None:
            return False
        pitch = self.cfg.get("pitch", {})
        lo, hi = pitch.get("box_y", (18, 62))
        return x >= float(pitch.get("box_x", 102)) and lo <= y <= hi

    def _progress(self, t: float, team: str, x, end_x, end_y, ftx: float) -> None:
        if x is not None and end_x is not None and x < ftx <= end_x:
            self.windows["entry"].append((t, team, 1.0))
        if self._in_box(end_x, end_y):
            self.windows["box"].append((t, team, 1.0))

    # ------------------------------------------------------------------ state at a game time

    def _prune(self, game_time: float) -> None:
        horizon = game_time - float(self.cfg.get("windows_s", {}).get("long", 300))
        for win in self.windows.values():
            while win and win[0][0] < horizon:
                win.popleft()

    def _window(self, kind: str, since: float, team: str | None) -> tuple[int, float]:
        n, total = 0, 0.0
        for t, tm, v in self.windows[kind]:
            if t >= since and tm == team:
                n += 1
                total += v
        return n, total

    def _sides(self) -> tuple[str, str]:
        return self.home or "Home", self.away or "Away"

    def _score(self) -> tuple[str, str, int, int]:
        home, away = self._sides()
        return home, away, self.goals.get(home, 0), self.goals.get(away, 0)

    def _minute(self, game_time: float) -> int:
        return int(game_time // 60)

    def _stage_stakes(self) -> float:
        s = self.cfg.get("stakes", {})
        return float(s.get("stage", {}).get(self.stage, s.get("stage_default", 0.3)))

    def _values(self, game_time: float) -> dict[str, float]:
        if self._cache is not None and self._cache[:2] == (game_time, self._seq):
            return self._cache[2]
        self._prune(game_time)
        w = self.cfg.get("windows_s", {})
        short, long_ = game_time - float(w.get("short", 120)), game_time - float(w.get("long", 300))
        home, away, gh, ga = self._score()
        v: dict[str, float] = {
            "score_diff": gh - ga,
            "abs_score_diff": abs(gh - ga),
            "minute": self._minute(game_time),
            "period": self.period,
            "stage_stakes": self._stage_stakes(),
        }
        for side, team in (("home", home), ("away", away)):
            v[f"red_cards_{side}"] = self.reds.get(team, 0)
            v[f"yellow_cards_{side}"] = self.yellows.get(team, 0)
            v[f"shots_2min_{side}"], v[f"xg_2min_{side}"] = self._window("shot", short, team)
            v[f"shots_5min_{side}"], v[f"xg_5min_{side}"] = self._window("shot", long_, team)
            v[f"final_third_entries_2min_{side}"] = self._window("entry", short, team)[0]
            v[f"box_touches_2min_{side}"] = self._window("box", short, team)[0]
            v[f"pressures_2min_{side}"] = self._window("pressure", short, team)[0]
            v[f"corners_5min_{side}"] = self._window("corner", long_, team)[0]
        passes_home = self._window("pass", long_, home)[0]
        passes_away = self._window("pass", long_, away)[0]
        total = passes_home + passes_away
        v["possession_5min_home"] = passes_home / total if total else 0.5
        kind, team, x, y = self.last
        v["last_event_type"] = LAST_EVENT.index(kind) if kind in LAST_EVENT else 0
        v["last_event_home"] = 1.0 if team == home else 0.0 if team == away else 0.5
        v["last_event_x"], v["last_event_y"] = x, y
        v["secs_since_shot"] = (
            min(600.0, game_time - self.last_shot_t) if self.last_shot_t is not None else 600.0
        )
        home_attack = self.poss_team is not None and self.poss_team == home
        v["attack_home"] = 1.0 if home_attack else 0.0 if self.poss_team == away else 0.5
        bx, by = self.ball
        v["attack_x"] = bx
        v["attack_dist_goal"] = float(np.hypot(120.0 - bx, 40.0 - by))
        cap = float(w.get("possession_cap", 120))
        v["possession_s"] = min(cap, game_time - self.poss_since) if self.poss_team else 0.0
        v["possession_passes"] = self.poss_passes if self.poss_team else 0
        instant = game_time - float(w.get("instant", 30))
        for name, kind in (
            ("shots", "shot"),
            ("entries", "entry"),
            ("box_touches", "box"),
            ("corners", "corner"),
        ):
            v[f"{name}_30s_attack"] = self._window(kind, instant, self.poss_team)[0] if self.poss_team else 0
        self._cache = (game_time, self._seq, v)
        return v

    def features(self, game_time: float) -> np.ndarray:
        return self.row(self._values(game_time))

    # ------------------------------------------------------------------ parts and facts

    def _score_facts(self) -> tuple[Fact, Fact]:
        home, away, gh, ga = self._score()
        diff = gh - ga
        if diff == 0:
            score_fact = Fact(key="score_diff", value=0, text=f"level at {gh}-{ga}")
        else:
            leader, margin = (home, diff) if diff > 0 else (away, -diff)
            score_fact = Fact(
                key="score_diff", value=diff, text=f"{leader} lead by {_WORDS.get(margin, str(margin))}"
            )
        return score_fact, Fact(key="score", value=f"{gh}-{ga}", text=f"{home} {gh}-{ga} {away}")

    def parts(self, game_time: float) -> dict[str, AdapterPart]:
        self.game_time = max(self.game_time, game_time)  # the clock runs between records
        v = self._values(game_time)
        home, away = self._sides()
        score_fact, score = self._score_facts()
        minute = int(v["minute"])
        minute_fact = Fact(key="minute", value=minute, unit="min", text=f"minute {minute}")

        c = self.cfg.get("closeness", {})
        closeness = clamp01(1.0 - abs(v["score_diff"]) / float(c.get("goals_zero", 3)))

        m = self.cfg.get("momentum", {})
        weights = m.get("weights", {"xg": 0.5, "shots": 0.3, "box": 0.2})
        side = (
            "home"
            if (v["xg_5min_home"], v["shots_2min_home"]) >= (v["xg_5min_away"], v["shots_2min_away"])
            else "away"
        )
        team = home if side == "home" else away
        xg, shots, box = (
            v[f"xg_5min_{side}"],
            int(v[f"shots_2min_{side}"]),
            int(v[f"box_touches_2min_{side}"]),
        )
        momentum = clamp01(
            float(weights["xg"]) * xg / float(m.get("xg_full", 1.0))
            + float(weights["shots"]) * shots / float(m.get("shots_full", 4))
            + float(weights["box"]) * box / float(m.get("box_full", 8))
        )
        shots_fact = Fact(
            key=f"shots_2min_{side}",
            value=shots,
            text=f"{team} {shots} shot{'s' if shots != 1 else ''} in 2 minutes",
        )
        xg_fact = Fact(key=f"xg_5min_{side}", value=round(xg, 1), text=f"{team} {xg:.1f} xG in 5 minutes")
        box_fact = Fact(
            key=f"box_touches_2min_{side}", value=box, text=f"{team} {box} touches in the box in 2 minutes"
        )
        momentum_facts = [xg_fact, shots_fact] if xg >= 0.05 else [shots_fact, box_fact]

        s = self.cfg.get("stakes", {})
        base = v["stage_stakes"]
        if self.period >= 3:
            late = 1.0
        else:
            lo, hi = float(s.get("late_from_min", 60)), float(s.get("late_full_min", 90))
            late = clamp01((minute - lo) / (hi - lo))
        stakes = clamp01(base + (1.0 - base) * late * closeness)
        stakes_facts = [minute_fact, score_fact]
        if self.stage:
            stakes_facts.append(Fact(key="stage", value=self.stage, text=f"the {self.stage}"))

        return {
            "closeness": AdapterPart(
                closeness,
                "a tight scoreline" if closeness >= float(c.get("phrase_at", 0.6)) else "a one-sided game",
                [score_fact, score],
            ),
            "momentum": AdapterPart(
                momentum,
                "end-to-end attacking" if momentum >= float(m.get("phrase_at", 0.5)) else "a slow spell",
                momentum_facts,
            ),
            "stakes": AdapterPart(
                stakes,
                self._stakes_phrase(stakes >= float(s.get("phrase_at", 0.6)), late),
                stakes_facts,
            ),
        }

    def _stakes_phrase(self, high: bool, late: float) -> str:
        if not high:
            return "plenty of time left"
        if self.period == 5:
            return "a penalty shoot-out"
        return "late drama" if late >= 0.5 else "a big occasion"

    def facts(self, game_time: float, parts: dict[str, AdapterPart] | None = None) -> dict[str, Fact]:
        out = super().facts(game_time, parts)
        v = self._values(game_time)
        home, away = self._sides()
        for side, team in (("home", home), ("away", away)):
            for key, text in (
                (f"shots_2min_{side}", "{n} shots in 2 minutes"),
                (f"box_touches_2min_{side}", "{n} touches in the box in 2 minutes"),
                (f"final_third_entries_2min_{side}", "{n} final-third entries in 2 minutes"),
                (f"corners_5min_{side}", "{n} corners in 5 minutes"),
                (f"red_cards_{side}", "{n} red cards"),
            ):
                n = int(v[key])
                out.setdefault(key, Fact(key=key, value=n, text=f"{team} {text.format(n=n)}"))
            xg = round(v[f"xg_5min_{side}"], 1)
            out.setdefault(
                f"xg_5min_{side}",
                Fact(key=f"xg_5min_{side}", value=xg, text=f"{team} {xg:.1f} xG in 5 minutes"),
            )
        pct = round(100 * v["possession_5min_home"])
        out["possession_5min_home"] = Fact(
            key="possession_5min_home", value=pct, unit="%", text=f"{home} {pct}% of passes in 5 minutes"
        )
        if self.period == 5:
            sh, sa = self.shootout.get(home, 0), self.shootout.get(away, 0)
            out["shootout"] = Fact(
                key="shootout", value=f"{sh}-{sa}", text=f"shoot-out {home} {sh}-{sa} {away}"
            )
        return out

    def game_clock(self) -> str | None:
        if self.period == 5:
            return "Penalties"
        m, s = divmod(int(self.game_time), 60)
        return f"{m}:{s:02d}"

    def heuristic_forecast(self, game_time: float) -> float:
        v = self._values(game_time)
        h = self.cfg.get("heuristic", {})
        box = v["box_touches_2min_home"] + v["box_touches_2min_away"]
        xg = v["xg_2min_home"] + v["xg_2min_away"]
        attacking = v["last_event_x"] >= float(self.cfg.get("pitch", {}).get("final_third_x", 80))
        p = (
            float(h.get("base", 0.03))
            + float(h.get("per_box_touch", 0.01)) * box
            + float(h.get("per_xg", 0.3)) * xg
            + float(h.get("final_third_bonus", 0.05)) * attacking
        )
        return clamp01(min(p, float(h.get("cap", 0.6))))

    def player_names(self) -> set[str]:
        return set(self.names)
