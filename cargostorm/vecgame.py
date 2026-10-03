"""N games at once in numpy, for training throughput (~440,000 moves per second).

Boards are stored in the canonical frame: each one rotated so its gravity points down.
So an action is always a canonical column 0-6, and a storm is just "rotate this board by
k quarter turns, then let every column fall". The rules match cargostorm.engine and
RULES.md move for move.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cargostorm.engine import CONNECT, EMPTY, P1, P2, SHIFT_CHANCE, SIZE


def has_four(mask: np.ndarray) -> np.ndarray:
    """mask: (N, 7, 7) bool -> (N,) bool, any four-in-a-row in each board."""
    n = SIZE - CONNECT + 1
    h = mask[:, :, :n].copy()
    v = mask[:, :n, :].copy()
    d = mask[:, :n, :n].copy()  # descending: top-left -> bottom-right
    a = mask[:, CONNECT - 1 :, :n].copy()  # ascending: bottom-left -> top-right
    for k in range(1, CONNECT):
        h &= mask[:, :, k : k + n]
        v &= mask[:, k : k + n, :]
        d &= mask[:, k : k + n, k : k + n]
        a &= mask[:, CONNECT - 1 - k : CONNECT - 1 - k + n, k : k + n]
    return h.any((1, 2)) | v.any((1, 2)) | d.any((1, 2)) | a.any((1, 2))


def compact_down(boards: np.ndarray) -> np.ndarray:
    """Let every column fall, keeping crate order: a stable sort that puts empties first."""
    order = np.argsort(boards != EMPTY, axis=1, kind="stable")
    return np.take_along_axis(boards, order, axis=1)


def encode(boards: np.ndarray, to_move: np.ndarray) -> np.ndarray:
    """The network's input: (N, 3, 7, 7) float32 planes [empty, mine, theirs], from the
    point of view of the player to move. Gravity is always down, so it needs no plane."""
    mine = boards == to_move[:, None, None]
    empty = boards == EMPTY
    theirs = ~(mine | empty)
    return np.stack([empty, mine, theirs], axis=1).astype(np.float32)


@dataclass
class StepResult:
    done: np.ndarray  # (k,) bool
    winner: np.ndarray  # (k,) int8: 0 = none or draw, else P1 / P2


class VecGame:
    def __init__(self, n: int, seed: int | None = None):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.boards = np.zeros((n, SIZE, SIZE), dtype=np.int8)
        self.to_move = np.full(n, P1, dtype=np.int8)

    def reset(self, idx=None) -> None:
        idx = slice(None) if idx is None else idx
        self.boards[idx] = EMPTY
        self.to_move[idx] = P1

    def legal(self, idx=None) -> np.ndarray:
        """(k, 7) bool: a column is legal when its top cell is empty."""
        boards = self.boards if idx is None else self.boards[idx]
        return boards[:, 0, :] == EMPTY

    def observe(self) -> np.ndarray:
        return encode(self.boards, self.to_move)

    def step(self, actions: np.ndarray, idx: np.ndarray | None = None) -> StepResult:
        """Play one turn (RULES.md section 3) in games `idx` (default: all) with canonical
        columns `actions`. Finished games stay finished until `reset`."""
        idx = np.arange(self.n) if idx is None else np.asarray(idx)
        actions = np.asarray(actions)
        k = len(idx)
        boards = self.boards[idx]
        mover = self.to_move[idx]
        rows = np.arange(k)

        # 1. Drop: columns are packed, so the landing row is (empty cells in column) - 1.
        landing = (boards[rows, :, actions] == EMPTY).sum(axis=1) - 1
        if (landing < 0).any():
            raise ValueError("illegal move: column is full")
        boards[rows, landing, actions] = mover

        # 2. Drop check, 3. full-board check.
        won = has_four(boards == mover[:, None, None])
        full = ~won & (boards != EMPTY).all(axis=(1, 2))
        winner = np.where(won, mover, 0).astype(np.int8)

        # 4. Storm roll: always drawn, so results replay identically for a given seed.
        happens = self.rng.random(k) < SHIFT_CHANCE
        turns = self.rng.integers(1, 4, size=k).astype(np.int8)
        shifts = np.where(happens & ~won & ~full, turns, 0)

        # 5. Cascade: rotate into the new frame, then let columns fall.
        for q in (1, 2, 3):
            sel = shifts == q
            if sel.any():
                boards[sel] = compact_down(np.rot90(boards[sel], q, axes=(1, 2)))

        # 6. Shift check: one player with four wins; both is a draw.
        stormy = shifts > 0
        w1 = w2 = np.zeros(k, dtype=bool)
        if stormy.any():
            w1 = stormy & has_four(boards == P1)
            w2 = stormy & has_four(boards == P2)
            winner[w1 & ~w2] = P1
            winner[w2 & ~w1] = P2

        done = won | full | w1 | w2
        self.boards[idx] = boards
        self.to_move[idx] = np.where(done, mover, 3 - mover)
        return StepResult(done, winner)
