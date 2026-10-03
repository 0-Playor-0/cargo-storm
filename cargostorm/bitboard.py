"""Bitboards: the whole hold as two Python integers, for fast game-tree search.

Everything is in the canonical frame (gravity points down). Each player's crates are
one int; bit (8 * col + row) is set when that player has a crate there, with row 0 at the
bottom. Each column has an always-empty 8th bit (a "sentinel"), so shifting the integer
to look at a neighbouring cell can never wrap from one column into the next. That makes
four-in-a-row detection four shift-and-AND operations.

A storm becomes a rotation plus a compaction. Both are precomputed lookup tables, so a
whole cascade costs about 14 table lookups.
"""

from __future__ import annotations

import math

import numpy as np

from cargostorm.engine import EMPTY, SIZE

H = SIZE + 1  # bits per column, including the sentinel
COL = (1 << SIZE) - 1
BOARD = sum(COL << (H * c) for c in range(SIZE))
CENTER_ORDER = (3, 2, 4, 1, 5, 0, 6)


def bit(col: int, row_from_bottom: int) -> int:
    return 1 << (H * col + row_from_bottom)


# --- conversion from canonical numpy boards (row 0 at the top) --------------------------


def from_canonical(canon: np.ndarray, player: int) -> tuple[int, int]:
    """(player's bits, opponent's bits) for a canonical 7x7 board holding 0/1/2."""
    mine = theirs = 0
    for row in range(SIZE):
        for col in range(SIZE):
            v = canon[row, col]
            if v != EMPTY:
                b = bit(col, SIZE - 1 - row)
                if v == player:
                    mine |= b
                else:
                    theirs |= b
    return mine, theirs


# --- rules ---------------------------------------------------------------------------


def has_four(b: int) -> bool:
    for s in (1, H, H + 1, H - 1):  # vertical, horizontal, two diagonals
        m = b & (b >> s)
        if m & (m >> (2 * s)):
            return True
    return False


def height(occ: int, col: int) -> int:
    """Crates in a column. Columns are always packed, so it's the highest set bit."""
    return ((occ >> (H * col)) & COL).bit_length()


def legal_cols(mine: int, theirs: int) -> list[int]:
    occ = mine | theirs
    return [c for c in range(SIZE) if height(occ, c) < SIZE]


def drop(mine: int, theirs: int, col: int) -> int:
    """`mine` with a crate dropped in `col` (assumed legal)."""
    return mine | bit(col, height(mine | theirs, col))


def is_full(mine: int, theirs: int) -> bool:
    return (mine | theirs) == BOARD


def _build_rotation_tables() -> list[list[list[int]]]:
    """ROT[k][col][v]: the rotated bits contributed by column `col` holding pattern `v`,
    when the canonical board is turned by np.rot90(k) (gravity moves k quarter turns)."""
    tables = []
    for k in range(4):
        mapping = {}
        for col in range(SIZE):
            for r in range(SIZE):
                unit = np.zeros((SIZE, SIZE), dtype=np.int8)
                unit[SIZE - 1 - r, col] = 1
                row2, col2 = np.argwhere(np.rot90(unit, k))[0]
                mapping[(col, r)] = bit(int(col2), SIZE - 1 - int(row2))
        per_col = []
        for col in range(SIZE):
            per_col.append([
                sum(mapping[(col, r)] for r in range(SIZE) if v >> r & 1) for v in range(1 << SIZE)
            ])
        tables.append(per_col)
    return tables


def _build_compact_table() -> list[int]:
    """COMPACT[occ << 7 | mine]: a column's `mine` bits after its crates fall to the
    bottom, keeping their order."""
    table = [0] * (1 << (2 * SIZE))
    for occ in range(1 << SIZE):
        sub = occ
        while True:  # every subset of occ
            packed, i = 0, 0
            for r in range(SIZE):
                if occ >> r & 1:
                    if sub >> r & 1:
                        packed |= 1 << i
                    i += 1
            table[occ << SIZE | sub] = packed
            if sub == 0:
                break
            sub = (sub - 1) & occ
    return table


ROT = _build_rotation_tables()
COMPACT = _build_compact_table()


def rotate(b: int, k: int) -> int:
    table = ROT[k]
    out = 0
    for c in range(SIZE):
        v = (b >> (H * c)) & COL
        if v:
            out |= table[c][v]
    return out


def shift(mine: int, theirs: int, k: int) -> tuple[int, int]:
    """Gravity turns by k quarter turns (1-3): rotate into the new canonical frame, then
    let every column fall."""
    rm, rt = rotate(mine, k), rotate(theirs, k)
    nm = nt = 0
    for c in range(SIZE):
        s = H * c
        occ = ((rm | rt) >> s) & COL
        if occ:
            packed = COMPACT[occ << SIZE | ((rm >> s) & COL)]
            nm |= packed << s
            nt |= (((1 << occ.bit_count()) - 1) ^ packed) << s
    return nm, nt


# --- static evaluation ---------------------------------------------------------------


def _counts(mine: int, theirs: int) -> tuple[int, int]:
    """Windows of 4 cells free of the opponent, holding >= 3 and exactly 2 of mine."""
    free = BOARD & ~theirs
    three = two = 0
    for s in (1, H, H + 1, H - 1):
        window = free & (free >> s) & (free >> 2 * s) & (free >> 3 * s)
        if not window:
            continue
        a, b, c, d = mine, mine >> s, mine >> 2 * s, mine >> 3 * s
        ge3 = (a & b & c) | (a & b & d) | (a & c & d) | (b & c & d)
        ge2 = (a & b) | (a & c) | (a & d) | (b & c) | (b & d) | (c & d)
        three += (window & ge3).bit_count()
        two += (window & ge2 & ~ge3).bit_count()
    return three, two


def evaluate(mine: int, theirs: int) -> float:
    """Heuristic value in (-0.9, 0.9) for the owner of `mine`: open threes and twos.
    Gravity-agnostic on purpose, since any line can be completed after a shift."""
    m3, m2 = _counts(mine, theirs)
    t3, t2 = _counts(theirs, mine)
    score = (m3 - t3) * 1.0 + (m2 - t2) * 0.25
    return 0.9 * math.tanh(score * 0.25)
