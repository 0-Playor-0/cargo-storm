"""Cargo Storm game engine: the rules from RULES.md and nothing else.

No rendering, no sound, no AI, and no hidden randomness: the storm roll is passed in
explicitly (see `play_turn`), so every function here is deterministic. The playable
game uses this module directly; the fast engines used for training and search
(vecgame.py, bitboard.py) implement the same rules.

Board: a 7x7 int8 numpy array indexed [row, col], row 0 at the top.
Cells hold EMPTY (0), P1 (1) or P2 (2).

The key trick is the *canonical frame*: rotate the board so the current gravity points
down. Then dropping and cascading are always the classic "fall to the bottom" case,
written once. The AI later sees the board in this same frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum

import numpy as np

SIZE = 7
CONNECT = 4
EMPTY, P1, P2 = 0, 1, 2
SHIFT_CHANCE = 0.5

Cell = tuple[int, int]


class Gravity(IntEnum):
    """Direction crates fall. The value is the number of counter-clockwise quarter
    turns (np.rot90's k) that rotate this gravity's wall to the bottom."""

    DOWN = 0
    LEFT = 1
    UP = 2
    RIGHT = 3

    @property
    def lanes_are_columns(self) -> bool:
        return self in (Gravity.DOWN, Gravity.UP)


class EndReason(Enum):
    NONE = "none"
    WIN_DROP = "win_drop"  # rule 3.2
    DRAW_FULL = "draw_full"  # rule 3.3
    WIN_SHIFT = "win_shift"  # rule 3.6, one player has four
    DRAW_DOUBLE_FOUR = "draw_double_four"  # rule 3.6, both players have four


class IllegalMove(ValueError):
    pass


# --------------------------------------------------------------------------------------
# Canonical frame
# --------------------------------------------------------------------------------------

# _CANON_SRC[g][r, c] = flat index of the actual cell that sits at canonical (r, c).
# Rotating an index grid with the board means we never hand-derive rotation formulas.
_CANON_SRC = [np.rot90(np.arange(SIZE * SIZE).reshape(SIZE, SIZE), g) for g in Gravity]


def _lane_reference_cell(lane: int, g: Gravity) -> Cell:
    return (0, lane) if g.lanes_are_columns else (lane, 0)


def _build_lane_to_canon_col() -> list[list[int]]:
    table = []
    for g in Gravity:
        row = []
        for lane in range(SIZE):
            r, c = _lane_reference_cell(lane, g)
            _, canon_col = np.argwhere(_CANON_SRC[g] == r * SIZE + c)[0]
            row.append(int(canon_col))
        table.append(row)
    return table


# _LANE_TO_CANON_COL[g][lane] = which canonical column a player-facing lane becomes.
_LANE_TO_CANON_COL = _build_lane_to_canon_col()


def to_canonical(board: np.ndarray, gravity: Gravity) -> np.ndarray:
    """Rotate so `gravity` points down."""
    return np.rot90(board, int(gravity))


def from_canonical(canon: np.ndarray, gravity: Gravity) -> np.ndarray:
    return np.rot90(canon, -int(gravity))


def canon_to_actual(cell: Cell, gravity: Gravity) -> Cell:
    flat = int(_CANON_SRC[gravity][cell])
    return divmod(flat, SIZE)


def lane_to_canon_col(lane: int, gravity: Gravity) -> int:
    return _LANE_TO_CANON_COL[gravity][lane]


def canon_col_to_lane(col: int, gravity: Gravity) -> int:
    return _LANE_TO_CANON_COL[gravity].index(col)


def lane_cells(lane: int, gravity: Gravity) -> list[Cell]:
    """The lane's cells in actual coordinates, ordered entry wall -> gravity wall."""
    col = lane_to_canon_col(lane, gravity)
    return [canon_to_actual((r, col), gravity) for r in range(SIZE)]


# --------------------------------------------------------------------------------------
# Four-in-a-row detection
# --------------------------------------------------------------------------------------

# (row step, col step) for each line direction.
_DIRECTIONS = {
    "horizontal": (0, 1),
    "vertical": (1, 0),
    "descending": (1, 1),  # top-left -> bottom-right
    "ascending": (-1, 1),  # bottom-left -> top-right
}


def _line_starts(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """Boolean grid: True at (r, c) if CONNECT cells starting there along (dr, dc) are
    all set. Built from shifted slices, so there are no Python loops over cells."""
    span = CONNECT - 1
    rows = slice(span, SIZE) if dr < 0 else slice(0, SIZE - span * dr)
    cols = slice(0, SIZE - span * dc)
    hit = np.ones_like(mask[rows, cols])
    for k in range(CONNECT):
        r0 = rows.start + k * dr
        c0 = cols.start + k * dc
        hit &= mask[r0 : r0 + hit.shape[0], c0 : c0 + hit.shape[1]]
    starts = np.zeros_like(mask)
    starts[rows, cols] = hit
    return starts


def has_four(board: np.ndarray, player: int) -> bool:
    mask = board == player
    return any(_line_starts(mask, dr, dc).any() for dr, dc in _DIRECTIONS.values())


def winning_cells(board: np.ndarray, player: int) -> frozenset[Cell]:
    """Every cell that is part of any four-in-a-row for `player` (for highlighting)."""
    mask = board == player
    cells: set[Cell] = set()
    for dr, dc in _DIRECTIONS.values():
        for r, c in np.argwhere(_line_starts(mask, dr, dc)):
            cells.update((int(r) + k * dr, int(c) + k * dc) for k in range(CONNECT))
    return frozenset(cells)


# --------------------------------------------------------------------------------------
# Game state
# --------------------------------------------------------------------------------------


def _frozen(board: np.ndarray) -> np.ndarray:
    board = np.array(board, dtype=np.int8)
    board.setflags(write=False)
    return board


@dataclass(frozen=True, eq=False)
class GameState:
    board: np.ndarray
    gravity: Gravity = Gravity.DOWN
    to_move: int = P1
    ply: int = 0
    winner: int | None = None
    end: EndReason = EndReason.NONE

    def __post_init__(self):
        object.__setattr__(self, "board", _frozen(self.board))

    @property
    def over(self) -> bool:
        return self.end is not EndReason.NONE


def new_game(gravity: Gravity = Gravity.DOWN) -> GameState:
    return GameState(np.zeros((SIZE, SIZE), dtype=np.int8), gravity)


def other(player: int) -> int:
    return P2 if player == P1 else P1


def legal_lanes(state: GameState) -> list[int]:
    if state.over:
        return []
    canon = to_canonical(state.board, state.gravity)
    return [lane for lane in range(SIZE) if canon[0, lane_to_canon_col(lane, state.gravity)] == EMPTY]


# --------------------------------------------------------------------------------------
# Turn mechanics
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Drop:
    lane: int
    player: int
    entry: Cell  # where the crate enters the board
    landing: Cell  # where it comes to rest


@dataclass(frozen=True)
class Shift:
    old: Gravity
    new: Gravity
    slides: tuple[tuple[Cell, Cell], ...]  # (from, to) for every crate that moved


@dataclass(frozen=True)
class Turn:
    """Everything that happened in one turn, in order: what the UI animates."""

    before: GameState
    drop: Drop
    shift: Shift | None
    after: GameState
    lines: dict[int, frozenset[Cell]] = field(default_factory=dict)  # winning cells per player


def _landing_row(canon: np.ndarray, col: int) -> int | None:
    """Rule 2.3: scan from the gravity wall (bottom) toward the entry wall (top) for
    the first empty index."""
    for r in range(SIZE - 1, -1, -1):
        if canon[r, col] == EMPTY:
            return r
    return None


def _cascade(board: np.ndarray, gravity: Gravity) -> tuple[np.ndarray, tuple[tuple[Cell, Cell], ...]]:
    """Rule 3.5: pack every lane against the gravity wall, keeping crate order."""
    canon = to_canonical(board, gravity)
    packed = np.zeros_like(canon)
    slides = []
    for col in range(SIZE):
        rows = np.flatnonzero(canon[:, col])  # top to bottom
        targets = range(SIZE - len(rows), SIZE)
        for src, dst in zip(rows, targets):
            packed[dst, col] = canon[src, col]
            if src != dst:
                slides.append(
                    (canon_to_actual((int(src), col), gravity), canon_to_actual((dst, col), gravity))
                )
    return from_canonical(packed, gravity), tuple(slides)


def roll_shift(gravity: Gravity, rng: np.random.Generator) -> Gravity | None:
    """Rule 3.4 sampled. Always consumes two draws so games replay identically per seed."""
    happens = rng.random() < SHIFT_CHANCE
    pick = int(rng.integers(len(Gravity) - 1))
    if not happens:
        return None
    return [g for g in Gravity if g != gravity][pick]


def play_turn(state: GameState, lane: int, shift_to: Gravity | None) -> Turn:
    """Play one full turn (rule 3). `shift_to` is the storm roll's result, decided by the
    caller (see `roll_shift` / `step`). It is ignored when the turn ends before the roll."""
    if state.over:
        raise IllegalMove("game is over")
    if not 0 <= lane < SIZE:
        raise IllegalMove(f"lane {lane} out of range")
    if shift_to is not None and Gravity(shift_to) == state.gravity:
        raise ValueError("a shift must change gravity")

    mover, g = state.to_move, state.gravity

    # 1. Drop.
    canon = to_canonical(state.board, g).copy()
    col = lane_to_canon_col(lane, g)
    row = _landing_row(canon, col)
    if row is None:
        raise IllegalMove(f"lane {lane} is full")
    canon[row, col] = mover
    board = from_canonical(canon, g)
    drop = Drop(lane, mover, entry=canon_to_actual((0, col), g), landing=canon_to_actual((row, col), g))

    def finish(board, gravity, shift, end, winner=None, lines=None):
        after = GameState(
            board,
            gravity,
            to_move=mover if end is not EndReason.NONE else other(mover),
            ply=state.ply + 1,
            winner=winner,
            end=end,
        )
        return Turn(state, drop, shift, after, lines or {})

    # 2. Drop check.
    if has_four(board, mover):
        return finish(board, g, None, EndReason.WIN_DROP, mover, {mover: winning_cells(board, mover)})

    # 3. Full check.
    if not (board == EMPTY).any():
        return finish(board, g, None, EndReason.DRAW_FULL)

    # 4. Storm roll (already decided by the caller).
    if shift_to is None:
        return finish(board, g, None, EndReason.NONE)

    # 5. Cascade.
    new_g = Gravity(shift_to)
    board, slides = _cascade(board, new_g)
    shift = Shift(g, new_g, slides)

    # 6. Shift check.
    lines = {p: winning_cells(board, p) for p in (P1, P2) if has_four(board, p)}
    if len(lines) == 2:
        return finish(board, new_g, shift, EndReason.DRAW_DOUBLE_FOUR, None, lines)
    if lines:
        (winner,) = lines
        return finish(board, new_g, shift, EndReason.WIN_SHIFT, winner, lines)
    return finish(board, new_g, shift, EndReason.NONE)


def step(state: GameState, lane: int, rng: np.random.Generator) -> Turn:
    """Play a turn with a real storm roll."""
    return play_turn(state, lane, roll_shift(state.gravity, rng))
