"""Pausing and resuming training.

At every evaluation each trainer saves its full state to runs/<name>/state/:
- state.pt: networks, optimizers, target networks, SAC's temperature, the self-play
  snapshot pool, all counters and schedules, and random-number generator states;
- replay/*.npy: the replay memory (DQN and SAC), which is what makes a resume seamless
  for off-policy learners.

`--resume` continues from there. The folder is written under a temporary name and then
renamed, so an interrupted save never leaves a half-written state behind.

For runs that were stopped before this existed, `--init-from <checkpoint.pt>` does a
*warm start* instead: the saved network weights and move count are restored, but
optimizer moments and replay memory start fresh (and anything else that wasn't saved).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from cargostorm.rl.replay import ReplayBuffer

BUFFER_FIELDS = ("obs", "next_obs", "next_legal", "action", "reward", "discount")


def save_state(run_dir: Path, payload: dict, buffer: ReplayBuffer | None = None) -> None:
    final = Path(run_dir) / "state"
    tmp = Path(run_dir) / "state.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    torch.save(payload, tmp / "state.pt")
    if buffer is not None:
        (tmp / "replay").mkdir()
        n = buffer.size
        for name in BUFFER_FIELDS:
            np.save(tmp / "replay" / f"{name}.npy", getattr(buffer, name)[:n])
        (tmp / "replay" / "meta.json").write_text(json.dumps({"size": buffer.size, "pos": buffer.pos}))
    old = Path(run_dir) / "state.old"
    shutil.rmtree(old, ignore_errors=True)
    if final.exists():
        os.replace(final, old)
    os.replace(tmp, final)
    shutil.rmtree(old, ignore_errors=True)


def load_state(run_dir: Path, buffer: ReplayBuffer | None = None) -> dict | None:
    """The saved payload (and the replay memory restored into `buffer`), or None."""
    path = Path(run_dir) / "state" / "state.pt"
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    replay = Path(run_dir) / "state" / "replay"
    if buffer is not None and replay.exists():
        meta = json.loads((replay / "meta.json").read_text())
        n = meta["size"]
        for name in BUFFER_FIELDS:
            getattr(buffer, name)[:n] = np.load(replay / f"{name}.npy")
        buffer.size, buffer.pos = n, meta["pos"] % buffer.capacity
    return payload


def snapshot_state(snapshots) -> dict:
    return {
        "models": [None if m is None else {k: v.cpu() for k, v in m.state_dict().items()} for m in snapshots.models],
        "next": snapshots.next,
    }


def restore_snapshots(snapshots, saved: dict, template: torch.nn.Module) -> None:
    from cargostorm.rl.common import frozen_copy

    for i, sd in enumerate(saved["models"]):
        if sd is not None:
            model = frozen_copy(template)
            model.load_state_dict(sd)
            snapshots.models[i] = model
    snapshots.next = saved["next"]


def rng_state(*generators: np.random.Generator) -> list[dict]:
    return [g.bit_generator.state for g in generators]


def restore_rngs(states: list[dict], *generators: np.random.Generator) -> None:
    for g, s in zip(generators, states):
        g.bit_generator.state = s


def truncate_log(run_dir: Path, step: int) -> None:
    """Drop metrics recorded after `step`: the stretch of training being redone after a
    resume, so the log never holds the same moves twice."""
    path = Path(run_dir) / "metrics.jsonl"
    if path.exists():
        kept = [line for line in path.read_text().splitlines() if line.strip() and json.loads(line).get("step", 0) <= step]
        path.write_text("".join(f"{line}\n" for line in kept))


def next_multiple(step: int, every: int) -> int:
    """The first multiple of `every` after `step` (when the next log/eval/snapshot is due)."""
    return (step // every + 1) * every


def best_score_so_far(run_dir: Path, step: int) -> float:
    """Best (heuristic + emm2) / 2 evaluation score recorded up to `step`."""
    path = Path(run_dir) / "metrics.jsonl"
    best = -1.0
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("kind") == "eval" and row["step"] <= step:
                best = max(best, (row["score_heuristic"] + row["score_emm2"]) / 2)
    return best
