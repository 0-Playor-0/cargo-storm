"""The computer opponents you can play against, as difficulty tiers.

Each tier is one of the agents from the final tournament, ranked by its Elo rating
there (Random = 0). The three trained networks ship with the game in
cargostorm/models/. Loading PyTorch takes about a second, so the networks are loaded
on a background thread as soon as the game starts, normally long before you need one.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from cargostorm.agents import Agent, ExpectiminimaxAgent, HeuristicAgent

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@dataclass(frozen=True)
class Tier:
    label: str  # difficulty shown to the player
    ai: str  # which agent plays it
    blurb: str
    elo: int  # tournament rating
    make: Callable[["Opponents"], Agent]


TIERS = [
    Tier("Beginner", "PPO", "Policy-gradient network. Plays fast, commits early.", 345, lambda o: o.network("ppo")),
    Tier("Easy", "Heuristic", "Hand-written: blocks, attacks, ignores storms.", 449, lambda o: HeuristicAgent()),
    Tier("Medium", "SAC", "Soft Actor-Critic network, 3M moves of training.", 481, lambda o: o.network("sac")),
    Tier("Hard", "DQN", "Deep Q-Network, 3M moves of training.", 486, lambda o: o.network("dqn")),
    Tier("Expert", "Expectiminimax", "Searches 3 moves ahead through every storm.", 547, lambda o: ExpectiminimaxAgent(3)),
]


class Opponents:
    """Builds agents for tiers; the trained networks load in the background."""

    def __init__(self):
        self._networks: dict[str, Agent] = {}
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._load, daemon=True)
        self._thread.start()

    def _load(self) -> None:
        try:
            from cargostorm.rl.common import NetAgent  # imports PyTorch

            for name in ("ppo", "sac", "dqn"):
                self._networks[name] = NetAgent(MODELS_DIR / f"{name}.pt")
        except Exception as error:  # surfaced when a network is actually requested
            self._error = error

    def network(self, name: str) -> Agent:
        self._thread.join()  # instant unless picked within the first second
        if self._error is not None:
            raise RuntimeError(f"could not load the trained networks from {MODELS_DIR}") from self._error
        return self._networks[name]

    def make(self, tier: Tier) -> Agent:
        return tier.make(self)
