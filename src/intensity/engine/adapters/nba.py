"""NBA adapter (owner: training agent).

Match records (from `training/sources/nba.py`; game time is elapsed game-clock seconds, 12-minute
quarters then 5-minute overtimes):

- `game_start` (attrs: home, away), `period_start`, `period_end`
- `score` (team, player, attrs: points, home_score, away_score, home, away, free_throw, ft_last)
- `miss` (attrs: shot_value, free_throw), `rebound` (attrs: offensive), `turnover`, `foul`
  (attrs: sub_type), `timeout` (attrs: sub_type)

Possession is inferred: a field goal, a made last free throw or a turnover hands the ball over,
and a rebound gives it to the rebounding team.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from intensity.engine.adapters.base import Adapter, AdapterPart, clamp01, plural
from intensity.schema import Fact, MatchPayload

QUARTER_S = 720.0
OT_S = 300.0
REGULATION_S = 4 * QUARTER_S
LAST_EVENT = ("none", "score", "miss", "rebound", "turnover", "foul", "timeout", "period_start", "period_end")


def period_bounds(period: int) -> tuple[float, float]:
    start = (period - 1) * QUARTER_S if period <= 4 else REGULATION_S + (period - 5) * OT_S
    return start, start + (QUARTER_S if period <= 4 else OT_S)


class NbaAdapter(Adapter):
    sport = "nba"
    FEATURES = (
        "margin",
        "abs_margin",
        "period",
        "time_remaining_s",
        "game_remaining_s",
        "run_2min",
        "run_abs_2min",
        "lead_changes",
        "ties",
        "secs_since_score",
        "team_fouls_home",
        "team_fouls_away",
        "bonus_home",
        "bonus_away",
        "timeouts_left_home",
        "timeouts_left_away",
        "possession_home",
        "trailing_has_ball",
        "last_event_type",
        "last_event_home",
    )
    LABEL = "lead_change"
    HORIZON_S = 60.0

    def reset(self) -> None:
        self.home, self.away = "Home", "Away"
        self.home_score = 0
        self.away_score = 0
        self.period = 1
        self.game_time = 0.0
        self.scoring: deque[tuple[float, str, int]] = deque()
        self.fouls: dict[str, list[float]] = {}  # team fouls this period, by game time
        self.timeouts: dict[str, list[int]] = {}  # periods in which each team called a timeout
        self.possession: str | None = None
        self.leader = 0  # +1 home, -1 away, 0 never led
        self.lead_changes = 0
        self.ties = 0
        self.last_score_t: float | None = None
        self.last: tuple[str, str | None] = ("none", None)
        self.names: set[str] = set()

    # ------------------------------------------------------------------ records

    def _other(self, team: str | None) -> str | None:
        return self.away if team == self.home else self.home if team == self.away else None

    def _team_foul(self, sub_type: str | None) -> bool:
        sub = sub_type or ""
        return "Technical" not in sub and sub not in set(self.cfg.get("non_team_fouls", []))

    def on_match(self, record: MatchPayload) -> None:
        a, kind, team, t = record.attrs, record.kind, record.team, record.game_time_s
        self.game_time = t
        if record.period != self.period:
            self.period = record.period
            self.fouls = {}
        if record.player:
            self.names.add(record.player)
        self.home = a.get("home", self.home)
        self.away = a.get("away", self.away)
        if kind == "score":
            prev = (self.home_score > self.away_score) - (self.home_score < self.away_score)
            self.home_score = int(a.get("home_score", self.home_score))
            self.away_score = int(a.get("away_score", self.away_score))
            now = (self.home_score > self.away_score) - (self.home_score < self.away_score)
            if now == 0 and prev != 0:
                self.ties += 1
            if now != 0:
                if self.leader != 0 and now != self.leader:
                    self.lead_changes += 1
                self.leader = now
            self.scoring.append((t, team or "", int(a.get("points", 2))))
            self.last_score_t = t
            if not a.get("free_throw") or a.get("ft_last"):
                self.possession = self._other(team)
        elif kind == "rebound":
            self.possession = team
        elif kind == "turnover":
            self.possession = self._other(team)
        elif kind == "foul" and team and self._team_foul(a.get("sub_type")):
            self.fouls.setdefault(team, []).append(t)
        elif kind == "timeout" and team and a.get("sub_type", "Regular") == "Regular":
            self.timeouts.setdefault(team, []).append(self.period)
        if kind in LAST_EVENT:
            self.last = (kind, team)

    # ------------------------------------------------------------------ state

    def _clock(self, game_time: float) -> tuple[int, float]:
        """Period and seconds left in it. The clock runs on between records, but past regulation only
        when the score is level, since only a tie means overtime."""
        period = self.period
        while game_time >= period_bounds(period)[1] and (period < 4 or self.home_score == self.away_score):
            period += 1
        return period, max(0.0, period_bounds(period)[1] - game_time)

    def _run(self, game_time: float) -> tuple[str | None, int, int]:
        """The team on top over the window, its net points, and home-minus-away points."""
        window = float(self.cfg.get("run_window_s", 120))
        while self.scoring and self.scoring[0][0] < game_time - window:
            self.scoring.popleft()
        totals: dict[str, int] = {}
        for _, team, pts in self.scoring:
            totals[team] = totals.get(team, 0) + pts
        signed = totals.get(self.home, 0) - totals.get(self.away, 0)
        if not totals:
            return None, 0, 0
        team = max(totals, key=totals.__getitem__)
        return team, totals[team] - sum(v for k, v in totals.items() if k != team), signed

    def _bonus(self, team: str, period: int, remaining: float) -> bool:
        b = self.cfg.get("bonus", {})
        fouls = self.fouls.get(team, [])
        limit = int(b.get("quarter_fouls", 5) if period <= 4 else b.get("ot_fouls", 4))
        last2_start = period_bounds(period)[1] - 120.0
        late = sum(1 for t in fouls if t >= last2_start)
        return len(fouls) >= limit or (remaining <= 120.0 and late >= int(b.get("last2_fouls", 2)))

    def _timeouts_left(self, team: str, period: int) -> int:
        t = self.cfg.get("timeouts", {})
        used = self.timeouts.get(team, [])
        if period <= 4:
            return max(0, int(t.get("per_game", 7)) - sum(1 for p in used if p <= 4))
        return max(0, int(t.get("per_ot", 2)) - sum(1 for p in used if p == period))

    def _values(self, game_time: float) -> dict[str, float]:
        period, remaining = self._clock(game_time)
        margin = self.home_score - self.away_score
        _, run, signed = self._run(game_time)
        game_remaining = remaining + (4 - period) * QUARTER_S if period <= 4 else remaining
        possession = 1.0 if self.possession == self.home else 0.0 if self.possession == self.away else 0.5
        trailing = self.away if margin > 0 else self.home if margin < 0 else None
        close = 0 < abs(margin) <= int(self.cfg.get("close_margin", 3))
        kind, team = self.last
        cap = float(self.cfg.get("secs_since_score_cap", 120))
        return {
            "margin": margin,
            "abs_margin": abs(margin),
            "period": period,
            "time_remaining_s": remaining,
            "game_remaining_s": game_remaining,
            "run_2min": signed,
            "run_abs_2min": run,
            "lead_changes": self.lead_changes,
            "ties": self.ties,
            "secs_since_score": min(cap, game_time - self.last_score_t)
            if self.last_score_t is not None
            else cap,
            "team_fouls_home": len(self.fouls.get(self.home, [])),
            "team_fouls_away": len(self.fouls.get(self.away, [])),
            "bonus_home": float(self._bonus(self.home, period, remaining)),
            "bonus_away": float(self._bonus(self.away, period, remaining)),
            "timeouts_left_home": self._timeouts_left(self.home, period),
            "timeouts_left_away": self._timeouts_left(self.away, period),
            "possession_home": possession,
            "trailing_has_ball": float(close and trailing is not None and self.possession == trailing),
            "last_event_type": LAST_EVENT.index(kind) if kind in LAST_EVENT else 0,
            "last_event_home": 1.0 if team == self.home else 0.0 if team == self.away else 0.5,
        }

    def features(self, game_time: float) -> np.ndarray:
        return self.row(self._values(game_time))

    # ------------------------------------------------------------------ parts and facts

    def parts(self, game_time: float) -> dict[str, AdapterPart]:
        self.game_time = max(self.game_time, game_time)  # the clock runs between records
        v = self._values(game_time)
        self.period = max(self.period, int(v["period"]))
        margin = int(v["margin"])
        score = Fact(
            key="score",
            value=f"{self.home_score}-{self.away_score}",
            text=f"{self.home} {self.home_score}-{self.away_score} {self.away}",
        )
        margin_fact = Fact(
            key="margin",
            value=abs(margin),
            unit="pts",
            text=f"{abs(margin)}-point game" if margin else "tied game",
        )
        run_team, run, _ = self._run(game_time)
        run_fact = (
            Fact(key="run_2min", value=run, unit="pts", text=f"{run_team} on a {run}-point run")
            if run_team and run > 0
            else Fact(key="run_2min", value=0, unit="pts", text="no scoring run")
        )
        remaining = int(v["time_remaining_s"])
        time_fact = Fact(key="time_remaining_s", value=remaining, unit="s", text=self.game_clock() or "")

        c = self.cfg.get("closeness", {})
        closeness = clamp01(1.0 - abs(margin) / float(c.get("margin_zero", 15)))
        phrases = c.get("phrases", {})
        if closeness >= float(phrases.get("one_possession", 0.8)):
            close_phrase = "a one-possession game"
        elif closeness >= float(phrases.get("tight", 0.5)):
            close_phrase = "a tight game"
        else:
            close_phrase = "a blowout"

        m = self.cfg.get("momentum", {})
        momentum = clamp01(run / float(m.get("run_full", 10)))

        s = self.cfg.get("stakes", {})
        elapsed = clamp01(1.0 - v["game_remaining_s"] / REGULATION_S) if v["period"] <= 4 else 1.0
        late = v["period"] > 4 or v["game_remaining_s"] <= float(s.get("late_s", 300))
        stakes = clamp01(
            float(s.get("progress_weight", 0.6)) * elapsed**2
            + float(s.get("late_weight", 0.4)) * closeness * late
        )
        return {
            "closeness": AdapterPart(closeness, close_phrase, [margin_fact, score]),
            "momentum": AdapterPart(
                momentum,
                "a big run" if momentum >= float(m.get("phrase_at", 0.6)) else "trading baskets",
                [run_fact, score],
            ),
            "stakes": AdapterPart(
                stakes,
                ("overtime" if v["period"] > 4 else "crunch time")
                if stakes >= float(s.get("phrase_at", 0.7))
                else "early in the game",
                [time_fact, margin_fact],
            ),
        }

    def facts(self, game_time: float, parts: dict[str, AdapterPart] | None = None) -> dict[str, Fact]:
        out = super().facts(game_time, parts)
        v = self._values(game_time)
        for side, team in (("home", self.home), ("away", self.away)):
            if v[f"bonus_{side}"]:
                out[f"bonus_{side}"] = Fact(key=f"bonus_{side}", value=team, text=f"{team} in the bonus")
            n = int(v[f"timeouts_left_{side}"])
            out[f"timeouts_left_{side}"] = Fact(
                key=f"timeouts_left_{side}", value=n, text=f"{team} {plural(n, 'timeout')} left"
            )
        if v["trailing_has_ball"]:
            team = self.home if v["possession_home"] == 1.0 else self.away
            out["trailing_has_ball"] = Fact(
                key="trailing_has_ball", value=team, text=f"{team} have the ball, trailing"
            )
        n = int(v["lead_changes"])
        out["lead_changes"] = Fact(key="lead_changes", value=n, text=plural(n, "lead change"))
        return out

    def game_clock(self) -> str | None:
        period, remaining = self._clock(self.game_time)
        m, s = divmod(int(remaining), 60)
        label = f"Q{period}" if period <= 4 else f"OT{period - 4}"
        return f"{label} {m}:{s:02d}"

    def heuristic_forecast(self, game_time: float) -> float:
        h = self.cfg.get("heuristic", {})
        v = self._values(game_time)
        p = (
            float(h.get("base", 0.03))
            + float(h.get("close_weight", 0.4))
            * max(0.0, 1 - v["abs_margin"] / float(h.get("close_margin", 6)))
            + float(h.get("per_run_point", 0.02)) * v["run_abs_2min"]
            + float(h.get("trailing_ball_bonus", 0.08)) * v["trailing_has_ball"]
        )
        return clamp01(p)

    def player_names(self) -> set[str]:
        return set(self.names)
