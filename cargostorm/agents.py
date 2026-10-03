"""Players that are not humans. Every AI (random, heuristic, search, and the trained
networks) implements the same tiny interface, so the game and the trainers can swap
them freely.

Agents think in the canonical frame (gravity points down) via `choose_canonical`;
`choose` adapts that to engine states and player-facing lanes.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from cargostorm import bitboard as bb
from cargostorm.engine import SHIFT_CHANCE, GameState, canon_col_to_lane, to_canonical


class Agent(Protocol):
    name: str

    def choose(self, state: GameState) -> int:
        """Return a legal lane for `state.to_move`."""
        ...


class CanonicalAgent:
    """Base for agents that work on canonical boards. Subclasses implement
    `choose_canonical(canon, player) -> canonical column`."""

    name = "?"

    def choose(self, state: GameState) -> int:
        col = self.choose_canonical(to_canonical(state.board, state.gravity), state.to_move)
        return canon_col_to_lane(col, state.gravity)

    def choose_canonical(self, canon: np.ndarray, player: int) -> int:
        raise NotImplementedError


class HeuristicAgent(CanonicalAgent):
    """Greedy and storm-blind: win now, else block a drop-win, else take the move with
    the best static evaluation that doesn't set up an immediate opponent win."""

    name = "Heuristic"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def choose_canonical(self, canon, player) -> int:
        mine, theirs = bb.from_canonical(canon, player)
        cols = bb.legal_cols(mine, theirs)
        for c in cols:
            if bb.has_four(bb.drop(mine, theirs, c)):
                return c
        for c in cols:
            if bb.has_four(bb.drop(theirs, mine, c)):
                return c
        scores = []
        for c in cols:
            after = bb.drop(mine, theirs, c)
            v = bb.evaluate(after, theirs)
            if any(bb.has_four(bb.drop(theirs, after, c2)) for c2 in bb.legal_cols(after, theirs)):
                v -= 2.0
            scores.append(v)
        best = max(scores)
        return int(self.rng.choice([c for c, v in zip(cols, scores) if v >= best - 1e-9]))


# --- expectiminimax ------------------------------------------------------------------

WIN, LOSS = 1.0, -1.0
_OUTCOMES = [(0, 1 - SHIFT_CHANCE)] + [(k, SHIFT_CHANCE / 3) for k in (1, 2, 3)]  # (rotation, probability)


class Expectiminimax:
    """Depth-limited search through the storm.

    Decision nodes are negamax: a node's value is from the mover's point of view, and the
    opponent's best reply is the negation of their own best value. After each drop comes
    a chance node averaging the four storm outcomes (calm 1/2, each new gravity 1/6).

    Star1 pruning: values are bounded in [-1, 1], so after some outcomes of a chance node
    are known we can bound its average, and stop early when it cannot affect the parent.
    """

    def __init__(self, depth: int):
        self.depth = depth

    def best_move(self, mine: int, theirs: int) -> tuple[int, float]:
        cols = [c for c in bb.CENTER_ORDER if c in bb.legal_cols(mine, theirs)]
        for c in cols:  # take an immediate win without searching
            if bb.has_four(bb.drop(mine, theirs, c)):
                return c, WIN
        best_c, best_v = cols[0], -2.0
        for c in cols:
            v = self._after_drop(bb.drop(mine, theirs, c), theirs, self.depth, best_v, 2.0)
            if v > best_v:
                best_c, best_v = c, v
        return best_c, best_v

    def value(self, mine: int, theirs: int, depth: int, alpha: float = -2.0, beta: float = 2.0) -> float:
        """Value of a decision node for the player to move (owner of `mine`)."""
        best = -2.0
        for c in bb.CENTER_ORDER:
            if bb.height(mine | theirs, c) >= bb.SIZE:
                continue
            after = bb.drop(mine, theirs, c)
            v = self._after_drop(after, theirs, depth, max(alpha, best), beta)
            if v > best:
                best = v
                if best >= beta:
                    return best
        return best

    def _after_drop(self, mine: int, theirs: int, depth: int, alpha: float, beta: float) -> float:
        """Value for the player who just dropped (owner of `mine`): terminal checks, then
        the storm's chance node."""
        if bb.has_four(mine):
            return WIN
        if bb.is_full(mine, theirs):
            return 0.0
        total, remaining = 0.0, 1.0
        for k, p in _OUTCOMES:
            remaining -= p
            if k:
                a, b = bb.shift(mine, theirs, k)
                wa, wb = bb.has_four(a), bb.has_four(b)
                if wa or wb:
                    v = 0.0 if wa and wb else (WIN if wa else LOSS)
                    total += p * v
                    if total + remaining * WIN <= alpha:
                        return total + remaining * WIN
                    if total + remaining * LOSS >= beta:
                        return total + remaining * LOSS
                    continue
            else:
                a, b = mine, theirs
            if depth <= 1:
                v = bb.evaluate(a, b)
            else:
                # The child's window: the range of v that could still matter here.
                lo = max(LOSS, (alpha - total - remaining * WIN) / p)
                hi = min(WIN, (beta - total - remaining * LOSS) / p)
                v = -self.value(b, a, depth - 1, -hi, -lo)
            total += p * v
            if total + remaining * WIN <= alpha:
                return total + remaining * WIN
            if total + remaining * LOSS >= beta:
                return total + remaining * LOSS
        return total


class ExpectiminimaxAgent(CanonicalAgent):
    def __init__(self, depth: int = 2):
        self.search = Expectiminimax(depth)
        self.name = f"Expectiminimax-{depth}"

    def choose_canonical(self, canon, player) -> int:
        mine, theirs = bb.from_canonical(canon, player)
        return self.search.best_move(mine, theirs)[0]
