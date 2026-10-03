"""Neural networks shared by every algorithm, so the comparison is about the learning
method, not the architecture.

Input: (B, 3, 7, 7) planes [empty, mine, theirs] in the canonical frame (gravity down).
The trunk is a small residual CNN; 3x3 convolutions suit four-in-a-row patterns, and
padding keeps the 7x7 board size through every layer. It uses GroupNorm rather than
BatchNorm: BatchNorm behaves differently in train and eval mode and mixes statistics
across a batch, which causes subtle bugs when the same network both acts and learns.
"""

from __future__ import annotations

import torch
from torch import nn

from cargostorm.engine import SIZE

ACTIONS = SIZE


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.GroupNorm(8, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.GroupNorm(8, ch)

    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        return torch.relu(x + self.b2(self.c2(y)))


class Trunk(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1, bias=False), nn.GroupNorm(8, channels), nn.ReLU())
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])

    def forward(self, x):
        return self.blocks(self.stem(x))


class QNet(nn.Module):
    """Dueling Q-network: Q(s, a) = V(s) + A(s, a) - mean_a A(s, a). Splitting 'how good
    is this position' from 'how much better is this move' helps when many moves are
    roughly equal, which is common early in a game."""

    def __init__(self, channels: int = 64, blocks: int = 4):
        super().__init__()
        self.init_kwargs = {"channels": channels, "blocks": blocks}  # saved in checkpoints
        self.trunk = Trunk(channels, blocks)
        self.adv = nn.Sequential(nn.Conv2d(channels, 4, 1), nn.ReLU(), nn.Flatten(), nn.Linear(4 * SIZE * SIZE, 128), nn.ReLU(), nn.Linear(128, ACTIONS))
        self.val = nn.Sequential(nn.Conv2d(channels, 2, 1), nn.ReLU(), nn.Flatten(), nn.Linear(2 * SIZE * SIZE, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        h = self.trunk(x)
        a = self.adv(h)
        return self.val(h) + a - a.mean(dim=1, keepdim=True)


def masked(values: torch.Tensor, legal: torch.Tensor, fill: float = -1e9) -> torch.Tensor:
    """Illegal moves get a huge negative score, so argmax/softmax never pick them."""
    return values.masked_fill(~legal, fill)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class PolicyNet(nn.Module):
    """Actor-critic: one trunk, two heads. The policy head gives a score (logit) per
    column; softmax turns them into move probabilities. The value head estimates the
    final result for the player to move, squashed into [-1, 1] by tanh."""

    def __init__(self, channels: int = 64, blocks: int = 4):
        super().__init__()
        self.init_kwargs = {"channels": channels, "blocks": blocks}
        self.trunk = Trunk(channels, blocks)
        self.policy = nn.Sequential(nn.Conv2d(channels, 4, 1), nn.ReLU(), nn.Flatten(), nn.Linear(4 * SIZE * SIZE, 128), nn.ReLU(), nn.Linear(128, ACTIONS))
        self.value = nn.Sequential(nn.Conv2d(channels, 2, 1), nn.ReLU(), nn.Flatten(), nn.Linear(2 * SIZE * SIZE, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh())
        # Near-uniform initial policy: every move starts equally likely.
        nn.init.zeros_(self.policy[-1].weight)
        nn.init.zeros_(self.policy[-1].bias)

    def forward(self, x):
        h = self.trunk(x)
        return self.policy(h), self.value(h).squeeze(1)


class ActorNet(nn.Module):
    """Policy only (logits per column), for SAC's actor. SAC's critics are QNets."""

    def __init__(self, channels: int = 64, blocks: int = 4):
        super().__init__()
        self.init_kwargs = {"channels": channels, "blocks": blocks}
        self.trunk = Trunk(channels, blocks)
        self.policy = nn.Sequential(nn.Conv2d(channels, 4, 1), nn.ReLU(), nn.Flatten(), nn.Linear(4 * SIZE * SIZE, 128), nn.ReLU(), nn.Linear(128, ACTIONS))
        nn.init.zeros_(self.policy[-1].weight)
        nn.init.zeros_(self.policy[-1].bias)

    def forward(self, x):
        return self.policy(self.trunk(x))
