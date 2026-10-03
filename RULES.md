# Cargo Storm — Rulebook

Connect 4, re-imagined: a ship's hold in a storm. Gravity shifts at random, and when it
does, every crate in the hold slides.

This file is the single source of truth. The game engine (`cargostorm/engine.py`)
implements exactly these rules, and so do the fast engines used for training and search.

## 1. Board and players

- **1.1** The hold is a **7 × 7** grid. Cells are addressed `(row, col)`, row 0 at the top.
- **1.2** Two players: **Player 1** (moves first) and **Player 2**.
- **1.3** Gravity always points toward one wall: **Down, Up, Left or Right**. A new game
  starts with gravity **Down**.

## 2. Lanes and dropping a crate

- **2.1** A *lane* is a line of 7 cells running parallel to gravity. Under Down/Up gravity
  the lanes are the columns; under Left/Right they are the rows.
- **2.2** Lanes are numbered **1–7** (index 0–6): left to right for columns, top to bottom
  for rows. The player picks a lane by clicking it or pressing its number key.
- **2.3** A crate enters from the wall *opposite* gravity and lands in the cell nearest the
  gravity wall that is still empty. It is found by scanning the lane from the gravity wall
  toward the entry wall and taking the first empty index. No physics: just a matrix scan.
- **2.4** A lane is full when its entry cell is occupied. Picking a full lane is ignored,
  and the turn does not pass.

## 3. Turn sequence ("play, shift")

Each turn runs in this order:

1. **Drop:** the player to move drops a crate (rule 2).
2. **Drop check:** if the mover now has four in a row, **the mover wins immediately**.
   No shift happens.
3. **Full check:** if the board is now full, the game is a **draw**. No shift happens.
4. **Storm roll:** with probability **50%** a shift happens. The new gravity is chosen
   **uniformly from the 3 other directions**, so a shift always changes gravity.
5. **Cascade:** on a shift, every crate slides toward the new gravity wall. Crates keep
   their order within a lane and cannot pass each other, so each lane simply packs
   against the new wall.
6. **Shift check:** after the cascade settles:
   - If exactly one player has four in a row, **that player wins**, even if it was not
     their move.
   - If both players have four in a row, the game is a **draw**.
7. Otherwise the turn passes to the other player.

## 4. Four in a row

- **4.1** Four of one player's crates in consecutive cells: horizontally, vertically,
  diagonally ascending (bottom-left → top-right) or diagonally descending
  (top-left → bottom-right). Longer lines count too.

## 5. Guarantees (properties the tests verify)

- **5.1** After every turn, every lane is packed against the current gravity wall.
- **5.2** So a lane has room exactly when its entry cell is empty, and a board that isn't
  full always has at least one legal lane. The game can never deadlock.
- **5.3** The shift odds look the same from every gravity direction, and are symmetric
  left/right. That is what lets the AI see the board rotated so gravity always points
  down, without needing gravity as an input.
