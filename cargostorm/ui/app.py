"""The pygame front end: a menu and the game, dressed as a ship's hold in a storm.

Rendering only reads the Match's view and reacts to its events. All rules live in the
engine, all timing in the Match; this file decides how things look and sound.
"""

from __future__ import annotations

import math
import random

import pygame

from cargostorm.agents import RandomAgent
from cargostorm.engine import SIZE, EndReason, Gravity, lane_cells, legal_lanes
from cargostorm.ui import theme as T
from cargostorm.ui.fx import FX
from cargostorm.ui.layout import (
    BOARD_PX,
    CELL,
    GRAVITY_VEC,
    HEIGHT,
    LAYER_PX,
    LAYER_X,
    LAYER_Y,
    MARGIN,
    ORIGIN_Y,
    WIDTH,
    lane_at,
    layer_center,
    outside_entry,
)
from cargostorm.ui.match import Match, Mode
from cargostorm.ui.sound import SoundBank
from cargostorm.ui.tween import ease_in_out_cubic, ease_out_cubic, lerp

LANE_KEYS = {getattr(pygame, f"K_{i + 1}"): i for i in range(SIZE)} | {
    getattr(pygame, f"K_KP{i + 1}"): i for i in range(SIZE)
}

MENU_ITEMS = [
    (Mode.VS_AI, "Play vs AI", "You are Player 1"),
    (Mode.HOTSEAT, "Two Players", "Take turns at the helm"),
    (Mode.SPECTATE, "Watch AI vs AI", "Sit back and weather the storm"),
]

FOOTER = "Click a lane or press 1-7    ·    R restart    ·    M sound    ·    Esc menu"


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def ease_out_back(t: float, s: float = 1.7) -> float:
    """Overshoots a little, then settles: the 'snap' in landings and reveals."""
    t -= 1
    return 1 + (s + 1) * t**3 + s * t**2


def blit_alpha(dst, src, pos, alpha: int) -> None:
    src.set_alpha(alpha)
    dst.blit(src, pos)
    src.set_alpha(255)


def text_right(surf, font, s, color, right, centery, alpha=255):
    w, h = font.size(s)
    T.text(surf, font, s, color, topleft=(right - w, centery - h // 2), alpha=alpha)


# --------------------------------------------------------------------------------------
# Scenery shared by both scenes
# --------------------------------------------------------------------------------------


class Scenery:
    """The hold around the board: wall, portholes, swinging lanterns, rain, lightning."""

    PORTHOLES = [(63, ORIGIN_Y + 140), (WIDTH - 63, ORIGIN_Y + 140)]
    HOOKS = [(63, ORIGIN_Y + 290), (WIDTH - 63, ORIGIN_Y + 290)]

    def __init__(self, art: T.Art):
        self.art = art
        self.time = 0.0
        self.flash = 0.0  # whole-screen lightning
        self.window_flash = 0.0  # distant lightning, seen only through the portholes
        self.rain = 0.2
        self.rng = random.Random(5)
        self._next_ambient = 3.0

    def lightning(self) -> None:
        self.flash = 1.0
        self.window_flash = 1.0

    def update(self, dt: float, storm_level: float) -> None:
        self.time += dt
        self.flash *= math.exp(-dt * 5.5)
        self.window_flash *= math.exp(-dt * 4.0)
        self.rain += (0.22 + 0.78 * storm_level - self.rain) * min(1.0, dt * 3)
        self._next_ambient -= dt
        if self._next_ambient <= 0:
            self.window_flash = max(self.window_flash, self.rng.uniform(0.3, 0.6))
            self._next_ambient = self.rng.uniform(5, 11)

    def draw(self, surf, sway: float = 0.0) -> None:
        surf.blit(self.art.background, (0, 0))
        for center in self.PORTHOLES:
            T.draw_porthole(surf, self.art, center, 40, self.time, self.rain, self.window_flash)
        flicker = 0.5 + 0.5 * math.sin(self.time * 13) * math.sin(self.time * 7.3)
        for i, hook in enumerate(self.HOOKS):
            angle = sway * 2.4 + math.sin(self.time * 1.3 + i) * 4
            T.draw_lantern(surf, self.art, hook, angle, flicker)

    def draw_flash(self, surf) -> None:
        if self.flash > 0.02:
            veil = pygame.Surface((WIDTH, HEIGHT))
            veil.fill((200, 214, 255))
            veil.set_alpha(int(150 * self.flash))
            surf.blit(veil, (0, 0))


# --------------------------------------------------------------------------------------
# Menu
# --------------------------------------------------------------------------------------


class MenuScene:
    storm_level = 0.0

    def __init__(self, app: "App"):
        self.app = app
        self.time = 0.0
        self.buttons = [
            (pygame.Rect(WIDTH // 2 - 210, 340 + i * 112, 420, 80), mode, title, sub)
            for i, (mode, title, sub) in enumerate(MENU_ITEMS)
        ]
        self.hover = [0.0] * len(self.buttons)
        self._last_hovered = None

    def handle(self, event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for rect, mode, *_ in self.buttons:
                if rect.collidepoint(event.pos):
                    self.app.sound.play("tick", 0.6)
                    self.app.start(mode)
        elif event.type == pygame.KEYDOWN:
            if pygame.K_1 <= event.key < pygame.K_1 + len(self.buttons):
                self.app.sound.play("tick", 0.6)
                self.app.start(self.buttons[event.key - pygame.K_1][1])
            elif event.key == pygame.K_m:
                self.app.sound.toggle_mute()

    def update(self, dt: float) -> None:
        self.time += dt
        mouse = pygame.mouse.get_pos()
        hovered = None
        for i, (rect, *_) in enumerate(self.buttons):
            target = 1.0 if rect.collidepoint(mouse) else 0.0
            if target:
                hovered = i
            self.hover[i] += (target - self.hover[i]) * min(1.0, dt * 12)
        if hovered is not None and hovered != self._last_hovered:
            self.app.sound.play("tick", 0.25)
        self._last_hovered = hovered

    def draw(self, surf) -> None:
        app, f = self.app, self.app.fonts
        app.scenery.draw(surf, sway=math.sin(self.time * 0.7) * 3)
        bob = math.sin(self.time * 1.1) * 4
        for side, player in ((-1, 1), (1, 2)):
            crate = pygame.transform.rotozoom(app.art.crate[player], math.sin(self.time * 1.3 + side) * 6, 1.0)
            surf.blit(crate, crate.get_rect(center=(WIDTH // 2 + side * 345, 178 - bob * side)))
        T.text(surf, f.title, "CARGO STORM", T.TEXT, center=(WIDTH // 2, 172 + bob))
        T.text(surf, f.body, "Connect four in a ship's hold. Gravity shifts. Everything slides.", T.MUTED, center=(WIDTH // 2, 248))
        for i, (rect, _, title, sub) in enumerate(self.buttons):
            T.draw_plank_button(surf, f, rect, f"{i + 1}.  {title}", sub, self.hover[i])
        T.text(surf, f.small, "Press 1, 2 or 3    ·    M sound", T.MUTED, center=(WIDTH // 2, HEIGHT - 15))
        app.scenery.draw_flash(surf)


# --------------------------------------------------------------------------------------
# Game
# --------------------------------------------------------------------------------------


class GameScene:
    def __init__(self, app: "App", mode: Mode, seed: int | None = None):
        self.app = app
        self.mode = mode
        self.match = Match.for_mode(mode, RandomAgent, seed)
        self.fx = FX(seed or 0)
        self.layer = pygame.Surface((LAYER_PX, LAYER_PX), pygame.SRCALPHA)
        self.time = 0.0
        self.lands: dict[int, tuple[float, float, Gravity, bool]] = {}  # cid -> (t0, amp, gravity, locked)
        self.prev: dict[int, tuple[float, float]] = {}
        self.beams: list[tuple[int, tuple, tuple]] = []
        self.over_t: float | None = None
        self.turn_t = 0.0
        self.restart_rect = pygame.Rect(0, 0, 260, 58)
        self.restart_rect.center = (WIDTH // 2, ORIGIN_Y + BOARD_PX + MARGIN // 2 + 2)
        self.restart_hover = 0.0
        self._last_hover_lane = None

    # --- input -----------------------------------------------------------------------

    def handle(self, event) -> None:
        m = self.match
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.app.menu()
            elif event.key == pygame.K_r:
                self.restart()
            elif event.key == pygame.K_m:
                self.app.sound.toggle_mute()
            elif event.key in LANE_KEYS:
                m.request_lane(LANE_KEYS[event.key])
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.result_ready:
                if self.restart_rect.collidepoint(event.pos):
                    self.app.sound.play("tick", 0.6)
                    self.restart()
                return
            lane = lane_at(*event.pos, m.view.gravity)
            if lane is not None:
                m.request_lane(lane)

    def restart(self) -> None:
        self.match.restart()
        self.fx.clear()
        self.lands.clear()
        self.prev.clear()
        self.beams.clear()
        self.over_t = None

    @property
    def result_ready(self) -> bool:
        return self.match.shown.over and not self.match.busy

    @property
    def storm_level(self) -> float:
        storm = self.match.view.storm
        if not storm:
            return 0.0
        return 1.0 - storm.progress if storm.stage == "settle" else 1.0

    def hover_lane(self) -> int | None:
        if not self.match.accepts_input:
            return None
        return lane_at(*pygame.mouse.get_pos(), self.match.view.gravity)

    # --- update & events -------------------------------------------------------------

    def update(self, dt: float) -> None:
        self.time += dt
        self.match.update(dt)
        for name, data in self.match.drain_events():
            self.on_event(name, data)
        self.fx.update(dt)
        if self.over_t is not None:
            self.over_t += dt
        target = 1.0 if self.result_ready and self.restart_rect.collidepoint(pygame.mouse.get_pos()) else 0.0
        self.restart_hover += (target - self.restart_hover) * min(1.0, dt * 12)
        lane = self.hover_lane()
        if lane is not None and lane != self._last_hover_lane:
            self.app.sound.play("tick", 0.18)
        self._last_hover_lane = lane

    def on_event(self, name: str, data: dict) -> None:
        snd = self.app.sound
        if name == "drop":
            snd.play_distance("fall", data["distance"], 0.35, self._pan(data["cell"]))
        elif name == "land":
            self._impact(data, locked=True)
        elif name == "storm":
            self.app.scenery.lightning()
            snd.play("thunder", 0.9)
            snd.play("creak", 0.5)
            self.fx.kick(5)
        elif name == "cascade":
            snd.play("wave", 0.75)
            if data["count"]:
                snd.play_distance("slide", 2 + data["count"] / 3, 0.6)
        elif name == "impact":
            self._impact(data, locked=False)
        elif name == "settling":
            snd.play("creak_short", 0.45)
        elif name == "turn":
            self.turn_t = self.time
        elif name == "over":
            self._game_over(data)

    def _pan(self, cell) -> float:
        x, _ = layer_center(*cell)
        return (x - LAYER_PX / 2) / (LAYER_PX / 2) * 0.7

    def _impact(self, data: dict, locked: bool) -> None:
        strength = clamp01(data["distance"] / 6)
        self.lands[data["cid"]] = (self.time, 0.07 + 0.16 * strength, data["gravity"], locked)
        x, y = layer_center(*data["cell"])
        self.fx.dust(x, y, GRAVITY_VEC[data["gravity"]], strength)
        if locked:
            self.fx.ring(x, y, T.PLAYER_GLOW[data["player"]])
            self.fx.kick(0.8 + 2.4 * strength)
            self.app.sound.play_distance("lock", data["distance"], 0.9, self._pan(data["cell"]))
        else:
            self.fx.kick(0.4 + 1.2 * strength)
            self.app.sound.play_distance("thud", data["distance"], 0.5, self._pan(data["cell"]))

    def _game_over(self, data: dict) -> None:
        self.over_t = 0.0
        self.app.sound.play("victory" if data["winner"] else "draw", 0.8)
        self.beams = []
        for player, cells in self.match.view.highlight.items():
            for r, c in sorted(cells):
                for dr, dc in ((0, 1), (1, 0), (1, 1), (-1, 1)):
                    if (r + dr, c + dc) in cells:
                        self.beams.append((player, layer_center(r, c), layer_center(r + dr, c + dc)))
            for cell in cells:
                self.fx.sparks(*layer_center(*cell), T.PLAYER_GLOW[player], 10)
        if self.beams:
            self.fx.kick(4)

    # --- drawing ---------------------------------------------------------------------

    def hold_motion(self) -> tuple[float, float]:
        """How the hold rolls in a storm: (angle in degrees, vertical heave in px).
        Left/right shifts roll the hold; up/down shifts heave it."""
        storm = self.match.view.storm
        if not storm:
            return 0.0, 0.0
        roll = {Gravity.LEFT: 1, Gravity.RIGHT: -1}.get(storm.new, 0)
        heave = {Gravity.DOWN: 1, Gravity.UP: -1}.get(storm.new, 0)
        p, tremble = storm.progress, 0.0
        if storm.stage == "warn":
            k = ease_in_out_cubic(clamp01((p - 0.55) / 0.45))
            tremble = math.sin(self.time * 47) * 1.1 * clamp01(p / 0.15) * (1 - k)
        elif storm.stage == "slide":
            k = 1.0
        else:
            k = (1 - p) ** 2 * math.cos(p * math.pi * 2.2)  # rocks back past level, then settles
        return roll * 7.5 * k + tremble, heave * 16 * k

    def draw(self, surf) -> None:
        angle, heave = self.hold_motion()
        self.app.scenery.draw(surf, sway=angle)

        layer = self.layer
        layer.fill((0, 0, 0, 0))
        layer.blit(self.app.art.board, (0, 0))
        self._draw_hover(layer)
        self._draw_crates(layer)
        self._draw_beams(layer)
        self.fx.draw(layer)
        self._draw_lane_labels(layer)

        sx, sy = self.fx.shake_offset()
        center = (LAYER_X + LAYER_PX / 2 + sx, LAYER_Y + LAYER_PX / 2 + heave + sy)
        img = pygame.transform.rotozoom(layer, angle, 1.0) if abs(angle) > 0.05 else layer
        surf.blit(img, img.get_rect(center=center))

        self._draw_storm_banner(surf)
        self._draw_hud(surf)
        if self.result_ready:
            self._draw_result(surf)
        T.text(surf, self.app.fonts.small, FOOTER + ("    ·    muted" if self.app.sound.muted else ""),
               T.MUTED, center=(WIDTH // 2, HEIGHT - 15))
        self.app.scenery.draw_flash(surf)

    def _draw_hover(self, layer) -> None:
        lane = self.hover_lane()
        if lane is None:
            return
        g = self.match.view.gravity
        if g.lanes_are_columns:
            rect = pygame.Rect(MARGIN + lane * CELL, MARGIN, CELL, BOARD_PX)
        else:
            rect = pygame.Rect(MARGIN, MARGIN + lane * CELL, BOARD_PX, CELL)
        tint = pygame.Surface(rect.size, pygame.SRCALPHA)
        tint.fill((255, 236, 200, 22))
        # Chevrons flowing toward the gravity wall show which way the crate will go.
        dr, dc = GRAVITY_VEC[g]
        a = math.radians(T.GRAVITY_ANGLE[g])
        for k in range(3):
            s = (self.time * 0.9 + k / 3) % 1.0
            along = s * BOARD_PX if (dr + dc) > 0 else BOARD_PX - s * BOARD_PX
            x, y = (CELL / 2, along) if g.lanes_are_columns else (along, CELL / 2)
            fade = int(70 * math.sin(math.pi * s))
            tip = (x + math.cos(a) * 8, y + math.sin(a) * 8)
            for side in (-1, 1):
                wing = (x - math.cos(a) * 6 + math.cos(a + side * math.pi / 2) * 12,
                        y - math.sin(a) * 6 + math.sin(a + side * math.pi / 2) * 12)
                pygame.draw.line(tint, (255, 236, 200, fade), wing, tip, 3)
        layer.blit(tint, rect)

    def _draw_crates(self, layer) -> None:
        art, view = self.app.art, self.match.view
        lit: dict[int, int] = {}
        if self.over_t is not None:
            for player, cells in view.highlight.items():
                for cell in cells:
                    if cell in view.at:
                        lit[view.at[cell]] = player
        pulse = 0.5 + 0.5 * math.sin(self.time * 5)
        size = T.CRATE_PX
        for cid, crate in view.crates.items():
            x, y = layer_center(crate.row, crate.col)
            sprite = art.crate[crate.player]

            # Motion trail while moving fast.
            prev = self.prev.get(cid)
            self.prev[cid] = (x, y)
            if prev:
                vx, vy = x - prev[0], y - prev[1]
                if math.hypot(vx, vy) > 4:
                    for k, alpha in ((1, 70), (2, 38), (3, 16)):
                        blit_alpha(layer, sprite, sprite.get_rect(center=(x - vx * k * 0.8, y - vy * k * 0.8)), alpha)

            # Squash and stretch after a landing, anchored to the wall it hit.
            sw = sh = 1.0
            flash = 0.0
            land = self.lands.get(cid)
            if land:
                t0, amp, g, locked = land
                age = self.time - t0
                if age < 0.6:
                    a = amp * math.exp(-age * 9) * math.cos(age * 24)
                    if g.lanes_are_columns:
                        sh, sw = 1 - a, 1 + a * 0.6
                    else:
                        sw, sh = 1 - a, 1 + a * 0.6
                    dr, dc = GRAVITY_VEC[g]
                    x += dc * (1 - sw) * size / 2
                    y += dr * (1 - sh) * size / 2
                    if locked:
                        flash = clamp01(1 - age / 0.28) ** 2
                else:
                    del self.lands[cid]

            if cid in lit:
                glow = pygame.transform.smoothscale_by(art.glow[lit[cid]], 0.85 + 0.25 * pulse)
                layer.blit(glow, glow.get_rect(center=(x, y)), special_flags=pygame.BLEND_RGB_ADD)

            scaled = (sw, sh) != (1.0, 1.0)
            dims = (max(1, round(size * sw)), max(1, round(size * sh)))
            img = pygame.transform.smoothscale(sprite, dims) if scaled else sprite
            rect = img.get_rect(center=(x, y))
            layer.blit(img, rect)
            if flash > 0:
                f = art.flash[crate.player]
                f = pygame.transform.smoothscale(f, dims) if scaled else f
                blit_alpha(layer, f, rect, int(110 * flash))
            if cid in lit:
                pygame.draw.rect(layer, T.mix(T.PLAYER_GLOW[lit[cid]], (255, 255, 255), pulse * 0.5), rect.inflate(6, 6), 3, border_radius=8)

    def _draw_beams(self, layer) -> None:
        if not self.beams or self.over_t is None:
            return
        reveal = clamp01(self.over_t / 0.5) * len(self.beams)
        overlay = pygame.Surface(layer.get_size(), pygame.SRCALPHA)
        for i, (player, a, b) in enumerate(self.beams):
            part = clamp01(reveal - i)
            if part <= 0:
                break
            end = (lerp(a[0], b[0], part), lerp(a[1], b[1], part))
            glow = T.PLAYER_GLOW[player]
            pygame.draw.line(overlay, (*glow, 70), a, end, 16)
            pygame.draw.line(overlay, (*glow, 200), a, end, 6)
            pygame.draw.line(overlay, (255, 250, 235, 255), a, end, 2)
        layer.blit(overlay, (0, 0))

    def _draw_lane_labels(self, layer) -> None:
        m = self.match
        if m.view.storm or self.result_ready:
            return
        f, art = self.app.fonts, self.app.art
        g = m.view.gravity
        hover = self.hover_lane()
        open_lanes = set(legal_lanes(m.state)) if not m.busy else set(range(SIZE))
        for lane in range(SIZE):
            x, y = layer_center(*outside_entry(lane_cells(lane, g)[0], g))
            if lane == hover and lane in open_lanes:
                dr, dc = GRAVITY_VEC[g]
                bob = math.sin(self.time * 4) * 3
                sprite = art.crate[m.state.to_move]
                blit_alpha(layer, sprite, sprite.get_rect(center=(x - dc * bob, y - dr * bob)), 165)
                continue
            color = T.MUTED if lane in open_lanes else T.shade(T.MUTED, 0.45)
            T.text(layer, f.number, str(lane + 1), color, center=(x, y))

    def _draw_storm_banner(self, surf) -> None:
        storm = self.match.view.storm
        if not storm or storm.stage == "settle":
            return
        f = self.app.fonts
        p = storm.progress
        alpha = clamp01(p / 0.12) if storm.stage == "warn" else clamp01(1 - p * 2.5)
        if alpha <= 0:
            return
        cy = ORIGIN_Y + BOARD_PX // 2
        band = pygame.Surface((WIDTH, 130), pygame.SRCALPHA)
        for i in range(130):
            edge = min(i, 129 - i) / 30
            pygame.draw.line(band, (6, 8, 12, int(175 * min(1, edge) * alpha)), (0, i), (WIDTH, i))
        surf.blit(band, (0, cy - 65))
        jitter = math.sin(self.time * 53) * 2.5 if storm.stage == "warn" else 0
        T.text(surf, f.title, "STORM!", T.TEXT, center=(WIDTH // 2 + jitter, cy - 18), alpha=int(255 * alpha))
        reveal = clamp01((p - 0.62) / 0.15) if storm.stage == "warn" else 1.0
        if reveal > 0:
            T.text(surf, f.label, f"Gravity shifts {T.GRAVITY_NAME[storm.new]}", T.LANTERN,
                   center=(WIDTH // 2, cy + 34 + 8 * (1 - reveal)), alpha=int(255 * alpha * reveal))

    def _compass_state(self) -> tuple[float, float, float]:
        """The two-stage spin: the needle whirls and settles on an axis (horizontal or
        vertical) as a plain bar, then the red arrowhead picks the direction."""
        view = self.match.view
        storm = view.storm
        if not storm or storm.stage != "warn":
            return T.GRAVITY_ANGLE[view.gravity], 1.0, 0.0
        p = storm.progress
        old, new = T.GRAVITY_ANGLE[storm.old], T.GRAVITY_ANGLE[storm.new]
        if p < 0.08:
            return old + math.sin(self.time * 60) * 7 * (p / 0.08), 1 - p / 0.08, p / 0.08
        axis_target = old + 720 + ((new - old) % 180)
        spin = clamp01((p - 0.08) / 0.52)
        angle = lerp(old, axis_target, ease_out_back(ease_out_cubic(spin), 0.8))
        pick = clamp01((p - 0.62) / 0.22)
        if pick > 0 and axis_target % 360 != new % 360:
            angle += 180  # a bar looks the same either way round; the head decides
        return angle, ease_out_back(pick) if pick > 0 else 0.0, 1.0

    def _draw_hud(self, surf) -> None:
        m, f, art = self.match, self.app.fonts, self.app.art
        s = m.shown
        if not self.result_ready:
            who = s.to_move
            e = ease_out_cubic(clamp01((self.time - self.turn_t) / 0.35))
            alpha = int(255 * e)
            blit_alpha(surf, art.crate_small[who], (30, 22 + 6 * (1 - e)), alpha)
            T.text(surf, f.body, f"{T.PLAYER_NAME[who]}'s turn", T.TEXT, topleft=(78, 16 + 6 * (1 - e)), alpha=alpha)
            if m.view.storm:
                sub = "Hold on..."
            elif not m.is_human(who):
                sub = "AI is thinking" + "." * (1 + int(self.time * 3) % 3)
            else:
                sub = "Your move" if self.mode is Mode.VS_AI else "Pick a lane"
            T.text(surf, f.small, sub, T.MUTED, topleft=(80, 50), alpha=alpha)
        angle, head, glow = self._compass_state()
        T.draw_compass(surf, (WIDTH - 62, 46), 31, angle, head, glow)
        text_right(surf, f.small, "GRAVITY", T.MUTED, WIDTH - 112, 30)
        spinning = m.view.storm and m.view.storm.stage == "warn" and head < 0.5
        text_right(surf, f.label, "..." if spinning else T.GRAVITY_NAME[m.view.gravity], T.TEXT, WIDTH - 112, 58)

    def _draw_result(self, surf) -> None:
        s, f = self.match.shown, self.app.fonts
        e = self.over_t or 0.0
        q = clamp01(e / 0.45)
        if s.winner:
            title, color = f"{T.PLAYER_NAME[s.winner]} Wins!", T.PLAYER_GLOW[s.winner]
        else:
            title, color = "Draw!", T.TEXT
        img = f.heading.render(title, True, color)
        shadow = f.heading.render(title, True, (0, 0, 0))
        scale = lerp(1.7, 1.0, ease_out_back(q))
        center = (WIDTH // 2, ORIGIN_Y - 64)
        for src, offset, a in ((shadow, 3, 150), (img, 0, 255)):
            scaled = pygame.transform.smoothscale_by(src, scale)
            blit_alpha(surf, scaled, scaled.get_rect(center=(center[0] + offset, center[1] + offset)), int(a * q))
        sub = {
            EndReason.WIN_DROP: "Four in a row",
            EndReason.WIN_SHIFT: "The storm lined them up",
            EndReason.DRAW_FULL: "The hold is full",
            EndReason.DRAW_DOUBLE_FOUR: "The storm lined up both crews",
        }[s.end]
        T.text(surf, f.body, sub, T.TEXT, center=(WIDTH // 2, ORIGIN_Y - 22), alpha=int(255 * clamp01((e - 0.25) / 0.3)))
        appear = ease_out_cubic(clamp01((e - 0.6) / 0.35))
        if appear > 0:
            rect = self.restart_rect.move(0, 24 * (1 - appear))
            T.draw_plank_button(surf, f, rect, "Play Again  (R)", None, self.restart_hover)


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------


class App:
    FADE_OUT = 0.25
    FADE_IN = 0.35

    def __init__(self, screen: pygame.Surface, seed: int | None = None, sound: bool = True):
        self.screen = screen
        self.seed = seed
        self.fonts = T.Fonts()
        self.art = T.Art()
        self.sound = SoundBank(enabled=sound)
        self.scenery = Scenery(self.art)
        self.scene = MenuScene(self)
        self.fade = 1.0  # fade in from black on launch
        self._pending = None

    def _go(self, make_scene) -> None:
        if self._pending is None:
            self._pending = make_scene

    def menu(self) -> None:
        self._go(lambda: MenuScene(self))

    def start(self, mode: Mode) -> None:
        self._go(lambda: GameScene(self, mode, self.seed))

    def frame(self, dt: float, events) -> None:
        if self._pending:
            self.fade = min(1.0, self.fade + dt / self.FADE_OUT)
            if self.fade >= 1.0:
                self.scene, self._pending = self._pending(), None
        else:
            self.fade = max(0.0, self.fade - dt / self.FADE_IN)
            for event in events:
                self.scene.handle(event)
        self.scene.update(dt)
        self.scenery.update(dt, self.scene.storm_level)
        self.sound.set_rain(0.12 + 0.4 * self.scenery.rain)
        self.scene.draw(self.screen)
        if self.fade > 0:
            veil = pygame.Surface((WIDTH, HEIGHT))
            veil.fill((0, 0, 0))
            veil.set_alpha(int(255 * ease_in_out_cubic(self.fade)))
            self.screen.blit(veil, (0, 0))


def main(seed: int | None = None) -> None:
    pygame.mixer.pre_init(44_100, -16, 2, 512)
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Cargo Storm")
    app = App(screen, seed)
    clock = pygame.time.Clock()
    while True:
        dt = min(clock.tick(60) / 1000, 0.05)  # clamp so a stall doesn't skip animations
        events = pygame.event.get()
        if any(e.type == pygame.QUIT for e in events):
            break
        app.frame(dt, events)
        pygame.display.flip()
    pygame.quit()
