"""Experience replay with n-step returns.

Rewards in this game arrive only at the end, so a 1-step target (r + gamma * Q(s'))
moves information back just one move per update. An n-step target sums the next n
rewards before bootstrapping, so a win teaches the moves that led to it n times faster.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from cargostorm.engine import SIZE


class ReplayBuffer:
    """A ring buffer of transitions. Observations are 0/1 planes, stored as uint8."""

    def __init__(self, capacity: int, seed: int | None = None):
        self.capacity = capacity
        self.obs = np.zeros((capacity, 3, SIZE, SIZE), np.uint8)
        self.next_obs = np.zeros((capacity, 3, SIZE, SIZE), np.uint8)
        self.next_legal = np.zeros((capacity, SIZE), bool)
        self.action = np.zeros(capacity, np.int64)
        self.reward = np.zeros(capacity, np.float32)
        self.discount = np.zeros(capacity, np.float32)  # gamma^n, or 0 when the game ended
        self.size = 0
        self.pos = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, next_legal, discount) -> None:
        i = self.pos
        self.obs[i], self.action[i], self.reward[i] = obs, action, reward
        self.next_obs[i], self.next_legal[i], self.discount[i] = next_obs, next_legal, discount
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, mirror: bool = True) -> dict[str, np.ndarray]:
        """A random batch. With `mirror`, half the samples are flipped left-right (board
        and action together): a free, exactly-valid second view of each position."""
        idx = self.rng.integers(0, self.size, batch)
        out = {
            "obs": self.obs[idx].astype(np.float32),
            "action": self.action[idx].copy(),
            "reward": self.reward[idx],
            "next_obs": self.next_obs[idx].astype(np.float32),
            "next_legal": self.next_legal[idx].copy(),
            "discount": self.discount[idx],
        }
        if mirror:
            flip = self.rng.random(batch) < 0.5
            out["obs"][flip] = out["obs"][flip][..., ::-1]
            out["next_obs"][flip] = out["next_obs"][flip][..., ::-1]
            out["next_legal"][flip] = out["next_legal"][flip][:, ::-1]
            out["action"][flip] = SIZE - 1 - out["action"][flip]
        return out


class NStepCollector:
    """Turns each game's stream of (obs, action, reward) into n-step transitions.

    For every move it waits until n more learner moves have happened (or the game ended),
    then writes: R = r_t + g*r_{t+1} + ... and the observation to bootstrap from.
    """

    def __init__(self, n_envs: int, n_step: int, gamma: float, buffer: ReplayBuffer):
        self.n, self.gamma, self.buffer = n_step, gamma, buffer
        self.pending = [deque() for _ in range(n_envs)]

    def add(self, obs, actions, rewards, next_obs, next_legal, done) -> None:
        for e in range(len(actions)):
            q = self.pending[e]
            q.append((obs[e], actions[e], rewards[e]))
            if done[e]:
                while q:  # flush: the game is over, no bootstrapping
                    self._emit(q, next_obs[e], next_legal[e], terminal=True)
                    q.popleft()
            elif len(q) == self.n:
                self._emit(q, next_obs[e], next_legal[e], terminal=False)
                q.popleft()

    def _emit(self, q, next_obs, next_legal, terminal: bool) -> None:
        ret = sum(r * self.gamma**i for i, (_, _, r) in enumerate(q))
        discount = 0.0 if terminal else self.gamma ** len(q)
        obs0, a0, _ = q[0]
        self.buffer.add(obs0, a0, ret, next_obs, next_legal, discount)
