"""The training environment: a learner playing many games at once against opponents.

Conventions shared by every algorithm:
- Observation: (3, 7, 7) float32 planes [empty, mine, theirs] in the canonical frame
  (board rotated so gravity points down), from the point of view of the player to move.
- Action: a canonical column 0-6. A boolean mask marks the legal ones.
- Reward: +1 win, -1 loss, 0 draw, 0 for every other move. Always from the learner's
  side, and paid even when the game ends on the opponent's turn (storms included).
- The learner's seat (first or second) is random each game, so it learns both.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from cargostorm.agents import CanonicalAgent
from cargostorm.engine import P1, P2
from cargostorm.vecgame import VecGame

# An opponent sees canonical boards, whose turn it is, and the legal mask; returns columns.
Policy = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def random_policy(seed: int | None = None) -> Policy:
    rng = np.random.default_rng(seed)

    def act(boards, to_move, legal):
        return (rng.random(legal.shape) * legal).argmax(axis=1)

    return act


def agent_policy(agent: CanonicalAgent) -> Policy:
    """Adapt a one-board-at-a-time agent (Heuristic, Expectiminimax, ...) to batches."""

    def act(boards, to_move, legal):
        return np.array([agent.choose_canonical(b, int(p)) for b, p in zip(boards, to_move)], dtype=np.int64)

    return act


class VecVsOpponent:
    """N simultaneous games of a learner against opponents.

    `opponents` is one policy or a list of them. With a list, each new game draws its
    opponent using `weights`, which the trainer may change at any time (for example to
    move from easy opponents to self-play). `seats="alternate"` gives balanced,
    repeatable seating for evaluation; the default is a random seat per game.
    """

    def __init__(self, n: int, opponents: Policy | list[Policy], seed: int | None = None,
                 weights: list[float] | None = None, seats: str = "random"):
        self.n = n
        self.game = VecGame(n, seed)
        self.opponents = list(opponents) if isinstance(opponents, (list, tuple)) else [opponents]
        self.weights = weights
        self.seats = seats
        self.rng = np.random.default_rng(None if seed is None else seed + 1)
        self.seat = np.full(n, P1, dtype=np.int8)
        self.opponent_id = np.zeros(n, dtype=np.int16)

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        self._new_games(np.arange(self.n))
        return self.game.observe(), self.game.legal()

    def step(self, actions: np.ndarray):
        """The learner moves in every game, then the opponent replies where the game goes
        on. Finished games restart automatically. Returns (obs, legal, reward, done, info);
        `info["opponent"]` says which opponent each game was played against."""
        g, n = self.game, self.n
        all_idx = np.arange(n)
        reward = np.zeros(n, dtype=np.float32)
        done = np.zeros(n, dtype=bool)

        def settle(idx, result):
            reward[idx] = np.where(result.winner == self.seat[idx], 1.0, np.where(result.winner == 0, 0.0, -1.0))
            done[idx] = result.done

        settle(all_idx, g.step(np.asarray(actions)))
        reply = all_idx[~done]
        if len(reply):
            settle(reply, g.step(self._opponent_moves(reply), reply))

        info = {"opponent": self.opponent_id.copy()}  # before finished games draw new opponents
        finished = all_idx[done]
        if len(finished):
            self._new_games(finished)
        return g.observe(), g.legal(), reward, done, info

    def _new_games(self, idx: np.ndarray) -> None:
        """Start fresh games; where the learner sits second, the opponent opens."""
        self.game.reset(idx)
        if self.seats == "alternate":
            self.seat[idx] = np.where(idx % 2 == 0, P1, P2)
        else:
            self.seat[idx] = self.rng.choice([P1, P2], size=len(idx))
        if len(self.opponents) > 1:
            p = None if self.weights is None else np.asarray(self.weights, float) / np.sum(self.weights)
            self.opponent_id[idx] = self.rng.choice(len(self.opponents), size=len(idx), p=p)
        opens = idx[self.seat[idx] == P2]
        if len(opens):
            self.game.step(self._opponent_moves(opens), opens)

    def _opponent_moves(self, idx: np.ndarray) -> np.ndarray:
        g = self.game
        actions = np.zeros(len(idx), dtype=np.int64)
        ids = self.opponent_id[idx]
        for k in np.unique(ids):  # one batched call per opponent
            sel = ids == k
            sub = idx[sel]
            actions[sel] = self.opponents[k](g.boards[sub], g.to_move[sub], g.legal(sub))
        return actions
