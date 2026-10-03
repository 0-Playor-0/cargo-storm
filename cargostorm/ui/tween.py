"""Easing curves and a sequential animation timeline.

Animations never decide anything: the engine has already computed every final position.
A tween only moves a sprite from where it was to where the engine says it ends up.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease_in_quad(t: float) -> float:
    """Starts slow, speeds up: reads as 'falling'."""
    return t * t


def ease_out_cubic(t: float) -> float:
    return 1 - (1 - t) ** 3


def ease_in_out_cubic(t: float) -> float:
    return 4 * t**3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def fall_duration(cells: float) -> float:
    """Seconds for a crate to travel `cells` cells. Grows with sqrt(distance), like a real
    fall, without simulating any physics."""
    return 0.10 + 0.11 * cells**0.5 if cells > 0 else 0.0


@dataclass
class Step:
    duration: float
    update: Callable[[float], None] | None = None  # called with progress in [0, 1]
    on_start: Callable[[], None] | None = None
    on_end: Callable[[], None] | None = None


class Timeline:
    """Runs steps one after another. `busy` is True while anything is still playing."""

    def __init__(self):
        self._steps: deque[Step] = deque()
        self._elapsed = 0.0
        self._started = False

    @property
    def busy(self) -> bool:
        return bool(self._steps)

    def add(self, step: Step) -> None:
        self._steps.append(step)

    def clear(self) -> None:
        self._steps.clear()
        self._elapsed = 0.0
        self._started = False

    def update(self, dt: float) -> None:
        # A loop, so one long frame can finish several short steps.
        while self._steps:
            step = self._steps[0]
            if not self._started:
                self._started = True
                if step.on_start:
                    step.on_start()
            self._elapsed += dt
            done = self._elapsed >= step.duration
            if step.update:
                step.update(1.0 if done or step.duration <= 0 else self._elapsed / step.duration)
            if not done:
                return
            dt = self._elapsed - step.duration
            self._steps.popleft()
            self._elapsed = 0.0
            self._started = False
            if step.on_end:
                step.on_end()
