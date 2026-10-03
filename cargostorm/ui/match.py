"""The match controller: turns engine results into an animated sequence, and decides
when input is accepted. It holds no pygame code, so it runs headless in tests.

Two clocks run side by side:
- `state` is the engine's truth, updated the instant a move is played.
- The *view* (crates, view.gravity, shown) catches up as animations play.
Input is only accepted when the view has caught up with the truth.

The controller also emits events ("land", "storm", "impact", ...) as the view reaches
each moment. Sound and visual effects listen to those; the controller never knows they
exist.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from cargostorm.agents import Agent
from cargostorm.engine import P1, P2, Cell, Gravity, Turn, legal_lanes, new_game, step
from cargostorm.ui.layout import outside_entry
from cargostorm.ui.tween import Step, Timeline, ease_in_quad, fall_duration, lerp

AI_THINK_DELAY = 0.45  # seconds, so a human can follow AI moves
LOCK_TIME = 0.16  # pause after a crate lands, while it visibly locks in
STORM_WARNING = 1.25  # lightning + compass spin + the hold starting to roll
SETTLE_TIME = 0.55  # the hold rocking back to level after a cascade


class Mode(Enum):
    VS_AI = "vs_ai"
    HOTSEAT = "hotseat"
    SPECTATE = "spectate"


@dataclass
class Crate:
    player: int
    row: float
    col: float


@dataclass
class Storm:
    old: Gravity
    new: Gravity
    stage: str = "warn"  # "warn" -> "slide" -> "settle"
    progress: float = 0.0  # 0 -> 1 within the current stage


@dataclass
class View:
    crates: dict[int, Crate] = field(default_factory=dict)
    at: dict[Cell, int] = field(default_factory=dict)  # resting cell -> crate id
    gravity: Gravity = Gravity.DOWN
    storm: Storm | None = None
    highlight: dict[int, frozenset[Cell]] = field(default_factory=dict)


def _distance(a, b) -> float:
    return abs(b[0] - a[0]) + abs(b[1] - a[1])


class Match:
    def __init__(self, agents: dict[int, Agent | None], seed: int | None = None):
        """`agents[p]` is None for a human player p."""
        self.agents = agents
        self.rng = np.random.default_rng(seed)
        self._ids = itertools.count()
        self.restart()

    @classmethod
    def for_mode(cls, mode: Mode, make_agent, seed: int | None = None) -> "Match":
        agents = {
            Mode.VS_AI: {P1: None, P2: make_agent()},
            Mode.HOTSEAT: {P1: None, P2: None},
            Mode.SPECTATE: {P1: make_agent(), P2: make_agent()},
        }[mode]
        return cls(agents, seed)

    def restart(self) -> None:
        self.state = new_game()
        self.shown = self.state  # the state the HUD describes; lags behind during animation
        self.view = View(gravity=self.state.gravity)
        self.timeline = Timeline()
        self.events: list[tuple[str, dict]] = [("restart", {})]
        self.thinking = 0.0

    # --- queries ---------------------------------------------------------------------

    @property
    def busy(self) -> bool:
        return self.timeline.busy

    def is_human(self, player: int) -> bool:
        return self.agents.get(player) is None

    @property
    def accepts_input(self) -> bool:
        return not self.busy and not self.state.over and self.is_human(self.state.to_move)

    def drain_events(self) -> list[tuple[str, dict]]:
        events, self.events = self.events, []
        return events

    def _emit(self, name: str, **data) -> None:
        self.events.append((name, data))

    # --- input -----------------------------------------------------------------------

    def request_lane(self, lane: int) -> bool:
        """A human picked a lane. Returns False (and does nothing) if it is ignored:
        mid-animation, game over, AI's turn, or the lane is full."""
        if not self.accepts_input or lane not in legal_lanes(self.state):
            return False
        self._play(lane)
        return True

    # --- per frame -------------------------------------------------------------------

    def update(self, dt: float) -> None:
        self.timeline.update(dt)
        if self.busy or self.state.over or self.is_human(self.state.to_move):
            self.thinking = 0.0
            return
        self.thinking += dt
        if self.thinking >= AI_THINK_DELAY:
            self.thinking = 0.0
            self._play(self.agents[self.state.to_move].choose(self.state))

    # --- turn -> animation -----------------------------------------------------------

    def _play(self, lane: int) -> None:
        turn = step(self.state, lane, self.rng)
        self.state = turn.after
        self._animate_drop(turn)
        if turn.shift:
            self._animate_shift(turn)
        self.timeline.add(Step(0.0, on_end=lambda: self._settle(turn)))

    def _animate_drop(self, turn: Turn) -> None:
        drop, gravity = turn.drop, turn.before.gravity
        start = outside_entry(drop.entry, gravity)
        end = drop.landing
        cid = next(self._ids)
        crate = Crate(drop.player, *start)
        distance = _distance(start, end)
        info = dict(cid=cid, player=drop.player, cell=end, distance=distance, gravity=gravity)

        def begin():
            self.view.crates[cid] = crate
            self._emit("drop", duration=fall_duration(distance), **info)

        def move(p):
            e = ease_in_quad(p)
            crate.row = lerp(start[0], end[0], e)
            crate.col = lerp(start[1], end[1], e)

        def land():
            self.view.at[end] = cid
            self._emit("land", **info)

        self.timeline.add(Step(fall_duration(distance), move, begin, land))
        self.timeline.add(Step(LOCK_TIME))

    def _animate_shift(self, turn: Turn) -> None:
        shift = turn.shift
        storm = Storm(shift.old, shift.new)

        # 1. Warning: lightning, compass spin, the hold starts to roll.
        def warn_start():
            self.view.storm = storm
            self.view.gravity = shift.new
            self._emit("storm", old=shift.old, new=shift.new)

        def warn(p):
            storm.progress = p

        self.timeline.add(Step(STORM_WARNING, warn, warn_start))

        # 2. Cascade: every crate slides at once; each takes time for its own distance.
        moves: list[tuple[int, Cell, Cell, float]] = []
        landed: set[int] = set()
        total = max((fall_duration(_distance(s, d)) for s, d in shift.slides), default=0.0)

        def slide_start():
            storm.stage, storm.progress = "slide", 0.0
            for src, dst in shift.slides:
                moves.append((self.view.at[src], src, dst, fall_duration(_distance(src, dst))))
            # Re-key resting positions all at once: slides may pass through each other's cells.
            moved = {self.view.at.pop(src): dst for src, dst in shift.slides}
            self.view.at.update({dst: cid for cid, dst in moved.items()})
            self._emit("cascade", count=len(moves), duration=total, gravity=shift.new)

        def slide(p):
            storm.progress = p
            elapsed = p * total
            for cid, src, dst, duration in moves:
                local = min(1.0, elapsed / duration)
                e = ease_in_quad(local)
                crate = self.view.crates[cid]
                crate.row = lerp(src[0], dst[0], e)
                crate.col = lerp(src[1], dst[1], e)
                if local >= 1.0 and cid not in landed:
                    landed.add(cid)
                    self._emit(
                        "impact", cid=cid, player=crate.player, cell=dst,
                        distance=_distance(src, dst), gravity=shift.new,
                    )

        self.timeline.add(Step(total, slide, slide_start))

        # 3. Settle: the hold rocks back to level.
        def settle_start():
            storm.stage, storm.progress = "settle", 0.0
            self._emit("settling")

        def settle(p):
            storm.progress = p

        def settle_end():
            self.view.storm = None

        self.timeline.add(Step(SETTLE_TIME, settle, settle_start, settle_end))

    def _settle(self, turn: Turn) -> None:
        self.shown = turn.after
        self.view.gravity = turn.after.gravity
        self.view.highlight = turn.lines
        self._emit("turn", to_move=turn.after.to_move)
        if turn.after.over:
            self._emit("over", end=turn.after.end, winner=turn.after.winner)
