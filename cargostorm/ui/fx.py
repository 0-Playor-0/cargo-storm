"""Fire-and-forget visual effects: dust and splinter particles, lock shockwaves, sparks,
and screen shake. None of these block input or affect the game."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pygame


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    size: float
    color: tuple
    ax: float = 0.0  # particles fall the way the hold's gravity points
    ay: float = 0.0
    drag: float = 2.5
    spark: bool = False


@dataclass
class Ring:
    x: float
    y: float
    color: tuple
    age: float = 0.0
    duration: float = 0.38
    start: float = 64.0
    grow: float = 30.0


class FX:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.particles: list[Particle] = []
        self.rings: list[Ring] = []
        self.shake = 0.0

    def clear(self) -> None:
        self.particles.clear()
        self.rings.clear()
        self.shake = 0.0

    # --- spawners --------------------------------------------------------------------

    def dust(self, x, y, gravity_vec, strength: float, color=(150, 120, 86)):
        """A puff where a crate hits: sprays sideways and back against gravity, then
        falls the way gravity points."""
        gy, gx = gravity_vec  # (d_row, d_col) -> screen (x, y)
        px, py = -gy, gx  # perpendicular
        n = int(6 + 10 * strength)
        for _ in range(n):
            side = self.rng.uniform(-1, 1)
            back = self.rng.uniform(0.1, 0.6)
            speed = self.rng.uniform(60, 190) * (0.5 + strength)
            self.particles.append(Particle(
                x + gx * 28 + px * side * 26, y + gy * 28 + py * side * 26,
                (px * side - gx * back) * speed, (py * side - gy * back) * speed,
                life=0.0, max_life=self.rng.uniform(0.35, 0.7), size=self.rng.uniform(1.5, 3.5),
                color=tuple(int(c * self.rng.uniform(0.75, 1.15)) for c in color),
                ax=gx * 500, ay=gy * 500,
            ))

    def sparks(self, x, y, color, n: int = 18):
        for _ in range(n):
            a = self.rng.uniform(0, math.tau)
            speed = self.rng.uniform(90, 300)
            self.particles.append(Particle(
                x, y, math.cos(a) * speed, math.sin(a) * speed,
                life=0.0, max_life=self.rng.uniform(0.5, 1.1), size=self.rng.uniform(1.5, 3),
                color=color, ay=160, drag=1.6, spark=True,
            ))

    def ring(self, x, y, color, duration: float = 0.38):
        self.rings.append(Ring(x, y, color, duration=duration))

    def kick(self, amount: float):
        self.shake = min(14.0, self.shake + amount)

    # --- per frame -------------------------------------------------------------------

    def update(self, dt: float) -> None:
        for p in self.particles:
            p.life += dt
            p.vx += (p.ax - p.vx * p.drag) * dt
            p.vy += (p.ay - p.vy * p.drag) * dt
            p.x += p.vx * dt
            p.y += p.vy * dt
        self.particles = [p for p in self.particles if p.life < p.max_life]
        for r in self.rings:
            r.age += dt
        self.rings = [r for r in self.rings if r.age < r.duration]
        self.shake *= math.exp(-dt * 9)
        if self.shake < 0.05:
            self.shake = 0.0

    def shake_offset(self) -> tuple[float, float]:
        if not self.shake:
            return 0.0, 0.0
        return self.rng.uniform(-1, 1) * self.shake, self.rng.uniform(-1, 1) * self.shake

    def draw(self, surf: pygame.Surface) -> None:
        for r in self.rings:
            t = r.age / r.duration
            half = (r.start + r.grow * (1 - (1 - t) ** 3)) / 2
            alpha = int(220 * (1 - t) ** 1.5)
            box = pygame.Surface((half * 2 + 6, half * 2 + 6), pygame.SRCALPHA)
            pygame.draw.rect(box, (*r.color, alpha), (3, 3, half * 2, half * 2), max(1, int(4 * (1 - t)) + 1), border_radius=8)
            surf.blit(box, (r.x - half - 3, r.y - half - 3))
        for p in self.particles:
            fade = 1 - p.life / p.max_life
            if p.spark:
                tail = (p.x - p.vx * 0.03, p.y - p.vy * 0.03)
                color = tuple(int(c * (0.5 + 0.5 * fade)) for c in p.color)
                pygame.draw.line(surf, color, tail, (p.x, p.y), max(1, int(p.size * fade + 0.5)))
            else:
                pygame.draw.circle(surf, p.color, (p.x, p.y), max(0.5, p.size * fade))
