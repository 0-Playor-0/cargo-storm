"""The Cargo Storm look: palette, fonts, and procedurally drawn art (no image files).

Static art (wall, board, crates) is drawn once into cached surfaces. Animated pieces
(portholes, lanterns, compass, buttons) are small draw functions called every frame.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pygame

from cargostorm.engine import P1, P2, SIZE, Gravity
from cargostorm.ui.layout import BOARD_PX, CELL, HEIGHT, HUD_H, LAYER_PX, MARGIN, WIDTH

# --- palette -------------------------------------------------------------------------

WALL_A = (38, 27, 20)
WALL_B = (31, 22, 17)
TIMBER = (104, 70, 44)
TIMBER_DARK = (60, 40, 26)
TIMBER_LIGHT = (146, 104, 66)
SLOT = (22, 16, 12)
IRON = (62, 64, 70)
BRASS = (204, 164, 86)
BRASS_DARK = (118, 88, 40)
TEXT = (242, 230, 206)  # parchment
MUTED = (176, 154, 122)
LANTERN = (255, 178, 86)
SEA = (14, 30, 44)

PLAYER_COLOR = {P1: (184, 50, 42), P2: (226, 170, 50)}
PLAYER_GLOW = {P1: (255, 112, 92), P2: (255, 216, 112)}
PLAYER_NAME = {P1: "Player 1", P2: "Player 2"}
GRAVITY_NAME = {Gravity.DOWN: "DOWN", Gravity.UP: "UP", Gravity.LEFT: "LEFT", Gravity.RIGHT: "RIGHT"}
# Screen angle of each gravity in degrees: 0 = right, 90 = down (y grows downward).
GRAVITY_ANGLE = {Gravity.RIGHT: 0.0, Gravity.DOWN: 90.0, Gravity.LEFT: 180.0, Gravity.UP: 270.0}

CRATE_PX = 62


def shade(color, k: float):
    return tuple(max(0, min(255, int(c * k))) for c in color)


def mix(a, b, t: float):
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


# --- fonts ---------------------------------------------------------------------------


def _font(names: list[str], size: int) -> pygame.font.Font:
    for name in names:
        path = pygame.font.match_font(name)
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.Font(None, size)


class Fonts:
    def __init__(self):
        display = ["copperplate", "rockwell", "georgia"]
        body = ["avenirnext", "avenir", "helveticaneue", "dejavusans"]
        self.title = _font(display, 76)
        self.heading = _font(display, 46)
        self.label = _font(display, 28)
        self.number = _font(display, 30)
        self.body = _font(body, 25)
        self.small = _font(body, 17)


def text(surface, font, s: str, color, center=None, topleft=None, shadow=True, alpha=255):
    """Engraved-looking text: a dark offset shadow under the colored glyphs."""
    img = font.render(s, True, color)
    rect = img.get_rect(center=center) if center else img.get_rect(topleft=topleft)
    if shadow:
        sh = font.render(s, True, (0, 0, 0))
        sh.set_alpha(int(alpha * 0.6))
        surface.blit(sh, rect.move(2, 2))
    img.set_alpha(alpha)
    surface.blit(img, rect)
    return rect


# --- numpy-built gradients -----------------------------------------------------------


def _alpha_surface(alpha: np.ndarray, color) -> pygame.Surface:
    """A solid-color surface whose per-pixel alpha comes from a (w, h) array."""
    w, h = alpha.shape
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    surf.fill((*color, 255))
    pygame.surfarray.pixels_alpha(surf)[:] = np.clip(alpha, 0, 255).astype(np.uint8)
    return surf


def radial_light(radius: int, color, strength: float = 1.0) -> pygame.Surface:
    """An RGB glow (black edges) meant for additive blending (BLEND_RGB_ADD)."""
    size = radius * 2
    y, x = np.mgrid[0:size, 0:size]
    d = np.sqrt((x - radius + 0.5) ** 2 + (y - radius + 0.5) ** 2) / radius
    fall = np.clip(1 - d, 0, 1) ** 2.2 * strength
    surf = pygame.Surface((size, size))
    px = pygame.surfarray.pixels3d(surf)
    for i in range(3):
        px[:, :, i] = (fall.T * color[i]).astype(np.uint8)
    del px
    return surf


def vertical_gradient(w: int, h: int, top, bottom) -> pygame.Surface:
    surf = pygame.Surface((w, h))
    t = np.linspace(0, 1, h)
    px = pygame.surfarray.pixels3d(surf)
    for i in range(3):
        px[:, :, i] = (top[i] + (bottom[i] - top[i]) * t)[None, :].astype(np.uint8)
    del px
    return surf


# --- static art ----------------------------------------------------------------------


def _grain(surf, rect, color, rng: random.Random, vertical: bool, density: int = 10):
    """Faint wavy lines that read as wood grain."""
    x0, y0, w, h = rect
    span, length = (w, h) if vertical else (h, w)
    for _ in range(density):
        offset = rng.uniform(0, span)
        tone = shade(color, rng.uniform(0.82, 1.12))
        amp, freq, phase = rng.uniform(0.5, 2.5), rng.uniform(0.01, 0.04), rng.uniform(0, 6)
        pts = []
        for s in range(0, int(length) + 1, 6):
            o = offset + amp * math.sin(s * freq + phase)
            pts.append((x0 + o, y0 + s) if vertical else (x0 + s, y0 + o))
        if len(pts) > 1:
            pygame.draw.lines(surf, tone, False, pts, 1)


def make_background() -> pygame.Surface:
    """The hold's wall: vertical planks, a ceiling beam, a floor beam, nail heads,
    warm lantern light and a dark vignette."""
    rng = random.Random(7)
    surf = pygame.Surface((WIDTH, HEIGHT))
    plank = 58
    for i, x in enumerate(range(0, WIDTH, plank)):
        tone = shade(WALL_A if i % 2 else WALL_B, rng.uniform(0.9, 1.08))
        pygame.draw.rect(surf, tone, (x, 0, plank, HEIGHT))
        _grain(surf, (x + 3, 0, plank - 6, HEIGHT), shade(tone, 0.85), rng, vertical=True, density=7)
        pygame.draw.line(surf, shade(tone, 0.45), (x, 0), (x, HEIGHT), 2)
        for y in range(40 + (i % 3) * 50, HEIGHT, 190):
            pygame.draw.circle(surf, (20, 16, 13), (x + 7, y), 3)
            pygame.draw.circle(surf, (86, 74, 62), (x + 6, y - 1), 1)

    # Ceiling and floor beams.
    for rect in ((0, 0, WIDTH, HUD_H - 14), (0, HEIGHT - 30, WIDTH, 30)):
        pygame.draw.rect(surf, TIMBER_DARK, rect)
        _grain(surf, rect, shade(TIMBER_DARK, 1.15), rng, vertical=False, density=14)
        x, y, w, h = rect
        pygame.draw.line(surf, shade(TIMBER_DARK, 1.5), (0, y + h - 1 if y == 0 else y), (WIDTH, y + h - 1 if y == 0 else y), 2)
        pygame.draw.line(surf, (12, 9, 7), (0, y + h if y == 0 else y - 2), (WIDTH, y + h if y == 0 else y - 2), 3)

    # Lantern light pooled around the side lanterns.
    for cx in (62, WIDTH - 62):
        glow = radial_light(330, LANTERN, 0.22)
        surf.blit(glow, glow.get_rect(center=(cx, 520)), special_flags=pygame.BLEND_RGB_ADD)

    # Vignette.
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    d = np.sqrt(((x - WIDTH / 2) / (WIDTH * 0.62)) ** 2 + ((y - HEIGHT / 2) / (HEIGHT * 0.62)) ** 2)
    alpha = np.clip(d - 0.55, 0, 1) ** 1.3 * 330
    surf.blit(_alpha_surface(alpha.T, (0, 0, 0)), (0, 0))
    return surf


def make_board() -> pygame.Surface:
    """The cargo rack: a timber frame with iron corner brackets, brass rivets and 49
    recessed slots. Drawn into layer coordinates (board inset by MARGIN)."""
    rng = random.Random(3)
    surf = pygame.Surface((LAYER_PX, LAYER_PX), pygame.SRCALPHA)
    frame = 18
    outer = pygame.Rect(MARGIN - frame, MARGIN - frame, BOARD_PX + 2 * frame, BOARD_PX + 2 * frame)

    # Soft drop shadow.
    for i, a in enumerate((60, 45, 30, 18)):
        shadow = pygame.Surface(outer.inflate(8 + i * 8, 8 + i * 8).size, pygame.SRCALPHA)
        pygame.draw.rect(shadow, (0, 0, 0, a), shadow.get_rect(), border_radius=18)
        surf.blit(shadow, shadow.get_rect(center=(outer.centerx + 6, outer.centery + 10)))

    pygame.draw.rect(surf, TIMBER_DARK, outer, border_radius=12)
    _grain(surf, outer, shade(TIMBER_DARK, 1.2), rng, vertical=False, density=26)
    pygame.draw.rect(surf, shade(TIMBER_DARK, 1.55), outer, 2, border_radius=12)
    inner = pygame.Rect(MARGIN, MARGIN, BOARD_PX, BOARD_PX)
    pygame.draw.rect(surf, TIMBER, inner)
    _grain(surf, inner, shade(TIMBER, 0.85), rng, vertical=False, density=40)

    for r in range(SIZE):
        for c in range(SIZE):
            cell = pygame.Rect(MARGIN + c * CELL, MARGIN + r * CELL, CELL, CELL)
            slot = cell.inflate(-8, -8)
            pygame.draw.rect(surf, SLOT, slot, border_radius=6)
            # Recessed look: shadow on the top-left inner edge, light catch bottom-right.
            pygame.draw.line(surf, (8, 6, 5), slot.topleft, slot.topright, 3)
            pygame.draw.line(surf, (8, 6, 5), slot.topleft, slot.bottomleft, 3)
            pygame.draw.line(surf, shade(TIMBER, 1.3), (slot.left + 4, slot.bottom), (slot.right - 4, slot.bottom), 1)

    pygame.draw.rect(surf, (12, 9, 7), inner, 3)

    # Iron corner brackets with brass rivets.
    arm = 54
    for sx, sy, (cx, cy) in ((1, 1, outer.topleft), (-1, 1, outer.topright), (1, -1, outer.bottomleft), (-1, -1, outer.bottomright)):
        cx, cy = cx - (sx < 0), cy - (sy < 0)
        pts = [(cx, cy), (cx + sx * arm, cy), (cx + sx * arm, cy + sy * 12), (cx + sx * 12, cy + sy * 12),
               (cx + sx * 12, cy + sy * arm), (cx, cy + sy * arm)]
        pygame.draw.polygon(surf, IRON, pts)
        pygame.draw.polygon(surf, shade(IRON, 1.5), pts, 1)
        for k in (8, 30, 46):
            rivet(surf, (cx + sx * k, cy + sy * 6))
            rivet(surf, (cx + sx * 6, cy + sy * k))

    # Rivets along the frame, one per cell.
    for i in range(SIZE):
        m = MARGIN + i * CELL + CELL // 2
        for p in ((m, MARGIN - frame // 2), (m, MARGIN + BOARD_PX + frame // 2),
                  (MARGIN - frame // 2, m), (MARGIN + BOARD_PX + frame // 2, m)):
            rivet(surf, p, 2.5)
    return surf


def rivet(surf, center, r: float = 3.0):
    pygame.draw.circle(surf, BRASS_DARK, center, r)
    pygame.draw.circle(surf, BRASS, (center[0] - r * 0.3, center[1] - r * 0.3), r * 0.55)


def make_crate(color, size: int = CRATE_PX) -> pygame.Surface:
    """A painted cargo crate: frame boards, three planks, a diagonal brace, nail heads
    and a soft top highlight."""
    rng = random.Random(hash(color) & 0xFFFF)
    s = size
    surf = pygame.Surface((s, s), pygame.SRCALPHA)
    dark, mid, light = shade(color, 0.5), shade(color, 0.78), shade(color, 1.28)
    f = max(6, s // 9)  # frame board width

    pygame.draw.rect(surf, dark, (0, 0, s, s), border_radius=6)
    pygame.draw.rect(surf, color, (2, 2, s - 4, s - 4), border_radius=5)
    field = pygame.Rect(f, f, s - 2 * f, s - 2 * f)
    pygame.draw.rect(surf, mid, field)
    for k in range(3):
        plank = pygame.Rect(field.x, field.y + k * field.h // 3, field.w, field.h // 3)
        _grain(surf, plank.inflate(-2, -4), shade(mid, 0.82), rng, vertical=False, density=3)
        if k:
            pygame.draw.line(surf, dark, (field.x, plank.y), (field.right - 1, plank.y), 2)
    pygame.draw.rect(surf, dark, field, 1)

    # Diagonal brace with an outline.
    a, b = (field.x + 2, field.bottom - 3), (field.right - 3, field.y + 2)
    pygame.draw.line(surf, dark, a, b, f + 3)
    pygame.draw.line(surf, color, a, b, f)
    pygame.draw.line(surf, light, (a[0] + 1, a[1] - 3), (b[0] - 3, b[1] + 1), 1)

    # Frame bevels.
    pygame.draw.line(surf, light, (3, 3), (s - 4, 3), 1)
    pygame.draw.line(surf, light, (3, 3), (3, s - 4), 1)
    pygame.draw.line(surf, dark, (3, s - 4), (s - 4, s - 4), 2)
    pygame.draw.line(surf, dark, (s - 4, 3), (s - 4, s - 4), 2)

    for p in ((f // 2 + 1, f // 2 + 1), (s - f // 2 - 2, f // 2 + 1), (f // 2 + 1, s - f // 2 - 2), (s - f // 2 - 2, s - f // 2 - 2), a, b):
        pygame.draw.circle(surf, (40, 34, 30), p, 2)
        pygame.draw.circle(surf, (226, 214, 190), (p[0] - 0.5, p[1] - 0.5), 1)

    # Top highlight, kept inside the crate's silhouette.
    t = np.linspace(1, 0, s) ** 2 * 46
    gloss = _alpha_surface(np.tile(t, (s, 1)), (255, 255, 255))
    mask = pygame.mask.from_surface(surf).to_surface(setcolor=(255, 255, 255, 255), unsetcolor=(0, 0, 0, 0))
    gloss.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    surf.blit(gloss, (0, 0))
    return surf


def silhouette(sprite: pygame.Surface, color) -> pygame.Surface:
    """A flat-colored copy of a sprite's shape (for flashes and glows)."""
    return pygame.mask.from_surface(sprite).to_surface(setcolor=(*color, 255), unsetcolor=(0, 0, 0, 0))


class Art:
    """Everything drawn once at startup."""

    def __init__(self):
        self.background = make_background()
        self.board = make_board()
        self.crate = {p: make_crate(PLAYER_COLOR[p]) for p in (P1, P2)}
        self.crate_small = {p: make_crate(PLAYER_COLOR[p], 34) for p in (P1, P2)}
        self.flash = {p: silhouette(self.crate[p], (255, 246, 220)) for p in (P1, P2)}
        self.glow = {p: radial_light(70, PLAYER_GLOW[p], 0.9) for p in (P1, P2)}
        self.lantern_glow = radial_light(120, LANTERN, 0.55)
        self._sky: dict[int, pygame.Surface] = {}

    def sky_for(self, size: int) -> pygame.Surface:
        if size not in self._sky:
            self._sky[size] = vertical_gradient(size, size, (8, 12, 22), SEA)
        return self._sky[size]


# --- animated pieces -----------------------------------------------------------------

_RAIN = [(random.Random(i).random(), random.Random(i * 7 + 1).random(), random.Random(i * 13 + 5).uniform(0.6, 1.4)) for i in range(70)]


def draw_porthole(surf, art: Art, center, r: int, time: float, rain: float, flash: float):
    """A brass porthole looking out on a stormy night: falling rain, rolling sea,
    lit up by lightning."""
    size = 2 * r
    view = pygame.Surface((size, size), pygame.SRCALPHA)
    view.blit(art.sky_for(size), (0, 0))
    # The sea line rocks gently.
    sea_y = r * 1.15 + math.sin(time * 0.9) * r * 0.12
    tilt = math.sin(time * 0.6) * r * 0.10
    pygame.draw.polygon(view, SEA, [(0, sea_y - tilt), (size, sea_y + tilt), (size, size), (0, size)])
    pygame.draw.line(view, (60, 90, 110), (0, sea_y - tilt), (size, sea_y + tilt), 1)
    count = int(len(_RAIN) * min(1.0, 0.35 + rain))
    for i in range(count):
        fx, fy, speed = _RAIN[i]
        x = (fx * size + time * 40 * speed) % size
        y = (fy * size + time * 420 * speed) % size
        pygame.draw.line(view, (120, 150, 175), (x, y), (x - 3, y - 9), 1)
    if flash > 0:
        veil = pygame.Surface((size, size))
        veil.fill((210, 220, 255))
        veil.set_alpha(int(220 * flash))
        view.blit(veil, (0, 0))
    # Clip the square view to a circle.
    circle = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.circle(circle, (255, 255, 255, 255), (r, r), r)
    view.blit(circle, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    rect = view.get_rect(center=center)
    surf.blit(view, rect)
    # Glass glint and brass ring with bolts.
    pygame.draw.arc(surf, (190, 210, 230), rect.inflate(-14, -14), math.radians(110), math.radians(160), 2)
    pygame.draw.circle(surf, BRASS_DARK, center, r + 9, 10)
    pygame.draw.circle(surf, BRASS, center, r + 8, 3)
    pygame.draw.circle(surf, shade(BRASS_DARK, 0.6), center, r + 1, 2)
    for k in range(8):
        a = k * math.pi / 4 + math.pi / 8
        rivet(surf, (center[0] + math.cos(a) * (r + 5), center[1] + math.sin(a) * (r + 5)), 2.4)


def draw_lantern(surf, art: Art, hook, angle_deg: float, flicker: float):
    """A brass lantern swinging from a hook. `angle_deg` is the pendulum angle."""
    length = 46
    a = math.radians(angle_deg)
    bx, by = hook[0] + math.sin(a) * length, hook[1] + math.cos(a) * length
    glow = art.lantern_glow
    g = pygame.transform.smoothscale_by(glow, 0.9 + 0.12 * flicker)
    surf.blit(g, g.get_rect(center=(bx, by + 12)), special_flags=pygame.BLEND_RGB_ADD)
    pygame.draw.line(surf, (26, 22, 20), hook, (bx, by), 2)
    pygame.draw.circle(surf, IRON, hook, 4)
    body = pygame.Surface((26, 38), pygame.SRCALPHA)
    pygame.draw.rect(body, BRASS_DARK, (3, 0, 20, 6), border_radius=3)
    pygame.draw.rect(body, mix(LANTERN, (255, 240, 200), 0.3 + 0.3 * flicker), (4, 6, 18, 24), border_radius=4)
    for x in (4, 12, 21):
        pygame.draw.line(body, BRASS_DARK, (x, 6), (x, 29), 2)
    pygame.draw.rect(body, BRASS_DARK, (2, 29, 22, 6), border_radius=2)
    pygame.draw.circle(body, BRASS, (13, 2), 2)
    rotated = pygame.transform.rotate(body, -angle_deg)
    surf.blit(rotated, rotated.get_rect(center=(bx + math.sin(a) * 16, by + math.cos(a) * 16)))


def draw_compass(surf, center, r: int, angle_deg: float, head: float = 1.0, glow: float = 0.0):
    """A brass compass whose needle shows gravity. With head < 1 the needle is drawn as
    a plain bar (axis only) and the red arrowhead grows in as `head` -> 1."""
    cx, cy = center
    if glow > 0:
        halo = pygame.Surface((r * 4, r * 4), pygame.SRCALPHA)
        pygame.draw.circle(halo, (*LANTERN, int(90 * glow)), (r * 2, r * 2), r + 14)
        surf.blit(halo, halo.get_rect(center=center))
    pygame.draw.circle(surf, (14, 12, 12), (cx + 2, cy + 3), r + 6)
    pygame.draw.circle(surf, BRASS_DARK, center, r + 5)
    pygame.draw.circle(surf, BRASS, center, r + 5, 2)
    pygame.draw.circle(surf, (232, 220, 192), center, r)
    pygame.draw.circle(surf, (190, 172, 140), center, r, 1)
    for k in range(16):
        a = math.radians(k * 22.5)
        inner = r - (8 if k % 4 == 0 else 4)
        pygame.draw.line(surf, (110, 92, 70), (cx + math.cos(a) * inner, cy + math.sin(a) * inner),
                         (cx + math.cos(a) * (r - 1), cy + math.sin(a) * (r - 1)), 2 if k % 4 == 0 else 1)
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    px, py = -sa, ca
    tip = r - 5
    w = 5
    # Tail half (dark) and head half (red), as kite-shaped polygons.
    tail = [(cx - ca * tip, cy - sa * tip), (cx + px * w, cy + py * w), (cx - px * w, cy - py * w)]
    pygame.draw.polygon(surf, (44, 42, 46), tail)
    head_color = mix((44, 42, 46), (196, 46, 36), max(0.0, min(1.0, head)))
    head_len = tip
    front = [(cx + ca * head_len, cy + sa * head_len), (cx + px * w, cy + py * w), (cx - px * w, cy - py * w)]
    pygame.draw.polygon(surf, head_color, front)
    pygame.draw.circle(surf, BRASS, center, 4)
    pygame.draw.circle(surf, BRASS_DARK, center, 4, 1)


def draw_plank_button(surf, fonts: Fonts, rect: pygame.Rect, title: str, subtitle: str | None, hover: float):
    """A wooden sign hung on two ropes. `hover` in [0, 1] lifts and lights it."""
    lift = -4 * hover
    r = rect.move(0, lift)
    for x in (r.left + 30, r.right - 30):
        pygame.draw.line(surf, (150, 118, 76), (x, r.top - 18), (x, r.top + 4), 3)
    pygame.draw.rect(surf, (0, 0, 0), r.move(4, 6 - lift * 0.5), border_radius=8)
    base = mix(TIMBER, TIMBER_LIGHT, 0.35 * hover)
    pygame.draw.rect(surf, base, r, border_radius=8)
    pygame.draw.line(surf, shade(base, 0.7), (r.left + 6, r.centery), (r.right - 6, r.centery), 1)
    pygame.draw.rect(surf, mix(shade(base, 0.55), BRASS, hover), r, 2, border_radius=8)
    for p in ((r.left + 10, r.top + 10), (r.right - 10, r.top + 10), (r.left + 10, r.bottom - 10), (r.right - 10, r.bottom - 10)):
        rivet(surf, p, 2.6)
    if subtitle:
        text(surf, fonts.label, title, TEXT, center=(r.centerx, r.centery - 12))
        text(surf, fonts.small, subtitle, mix(MUTED, TEXT, 0.4), center=(r.centerx, r.centery + 17))
    else:
        text(surf, fonts.label, title, TEXT, center=r.center)
