"""Screen geometry: where cells are, and which lane a mouse position points at.
Pure arithmetic, no pygame, so it is unit-tested."""

from __future__ import annotations

from cargostorm.engine import SIZE, Gravity

CELL = 72
BOARD_PX = CELL * SIZE
MARGIN = CELL  # room outside the board for lane numbers and the incoming crate
HUD_H = 100
FOOTER_H = 44

WIDTH = 900
ORIGIN_X = (WIDTH - BOARD_PX) // 2
ORIGIN_Y = HUD_H + MARGIN
HEIGHT = ORIGIN_Y + BOARD_PX + MARGIN + FOOTER_H

# The board and its margins are drawn onto one square "layer" so the whole hold can
# roll during a storm. Layer coordinates have (0, 0) at the layer's top-left.
LAYER_PX = BOARD_PX + 2 * MARGIN
LAYER_X = ORIGIN_X - MARGIN
LAYER_Y = ORIGIN_Y - MARGIN

# Unit step (d_row, d_col) in the direction crates fall.
GRAVITY_VEC = {
    Gravity.DOWN: (1, 0),
    Gravity.UP: (-1, 0),
    Gravity.LEFT: (0, -1),
    Gravity.RIGHT: (0, 1),
}


def layer_center(row: float, col: float) -> tuple[float, float]:
    """Pixel center of a (possibly fractional, possibly off-board) cell position, in layer
    coordinates."""
    return MARGIN + (col + 0.5) * CELL, MARGIN + (row + 0.5) * CELL


def outside_entry(entry: tuple[int, int], gravity: Gravity) -> tuple[int, int]:
    """The position one cell beyond the entry wall, where a dropped crate appears."""
    dr, dc = GRAVITY_VEC[gravity]
    return entry[0] - dr, entry[1] - dc


def lane_at(x: float, y: float, gravity: Gravity) -> int | None:
    """Lane under the mouse. Anywhere along the lane counts, including the margins."""
    col = (x - ORIGIN_X) // CELL
    row = (y - ORIGIN_Y) // CELL
    if gravity.lanes_are_columns:
        lane, across = col, row
    else:
        lane, across = row, col
    if 0 <= lane < SIZE and -1 <= across <= SIZE:
        return int(lane)
    return None
