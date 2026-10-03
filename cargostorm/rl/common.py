"""Plumbing shared by every RL algorithm: devices, checkpoints, acting with a network,
the self-play opponent pool, evaluation against the baselines, and run logging."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from cargostorm.agents import ExpectiminimaxAgent, HeuristicAgent
from cargostorm.envs import Policy, VecVsOpponent, agent_policy, random_policy
from cargostorm.rl.nets import ActorNet, PolicyNet, QNet, masked
from cargostorm.vecgame import encode

MODELS = {"QNet": QNet, "PolicyNet": PolicyNet, "ActorNet": ActorNet}


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --- checkpoints ---------------------------------------------------------------------


def save_checkpoint(path: Path, model: torch.nn.Module, algo: str, **meta) -> None:
    """Written to a temp file and then renamed, so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "algo": algo,
        "model": type(model).__name__,
        "kwargs": model.init_kwargs,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "meta": meta,
    }, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = MODELS[ckpt["model"]](**ckpt["kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


# --- acting --------------------------------------------------------------------------


def net_policy(model: torch.nn.Module, device, epsilon: float = 0.0, seed: int | None = None) -> Policy:
    """A batched player: the network's best legal move, or a random one with probability
    `epsilon`. Works for Q-networks (Q-values) and policy networks (logits) alike."""
    rng = np.random.default_rng(seed)

    @torch.no_grad()
    def act(boards, to_move, legal):
        out = model(torch.as_tensor(encode(boards, to_move), device=device))
        scores = out[0] if isinstance(out, tuple) else out  # PolicyNet returns (logits, value)
        best = masked(scores, torch.as_tensor(legal, device=device)).argmax(1).cpu().numpy()
        if epsilon > 0:
            explore = rng.random(len(best)) < epsilon
            best = np.where(explore, (rng.random(legal.shape) * legal).argmax(1), best)
        return best

    return act


def frozen_copy(model: torch.nn.Module) -> torch.nn.Module:
    snapshot = copy.deepcopy(model).eval()
    for p in snapshot.parameters():
        p.requires_grad_(False)
    return snapshot


# --- training opponents --------------------------------------------------------------

OPPONENT_NAMES = ["random", "heuristic"]  # followed by the self-play snapshot slots


def opponent_mix(step: int, total: int, filled: int, slots: int, p_random_start: float, p_random_end: float,
                 p_heuristic: float) -> list[float]:
    """Opponent weights [random, heuristic, slot_0..slot_k]: Random fades from
    p_random_start to p_random_end over the first 30% of training, Heuristic stays fixed,
    and self-play snapshots share the rest equally once any exist."""
    frac = min(1.0, step / (0.3 * total))
    p_random = p_random_start + frac * (p_random_end - p_random_start)
    p_self = max(0.0, 1.0 - p_random - p_heuristic) if filled else 0.0
    weights = [p_self / filled if i < filled else 0.0 for i in range(slots)]
    if not filled:  # no snapshots yet: share self-play's weight among the fixed opponents
        p_random = 1.0 - p_heuristic
    return [p_random, p_heuristic] + weights


def training_opponents(seed: int, snapshots: "Snapshots") -> list[Policy]:
    """The opponent pool every trainer uses, in OPPONENT_NAMES order, then self-play."""
    return [random_policy(seed + 1), agent_policy(HeuristicAgent(seed + 2))] + snapshots.policies


class Snapshots:
    """Fixed slots of frozen past learners, for self-play. Slots are reused round-robin,
    so opponent ids stay valid while the pool refreshes."""

    def __init__(self, slots: int, device, epsilon: float, seed: int):
        self.models: list[torch.nn.Module | None] = [None] * slots
        self.device, self.epsilon, self.seed = device, epsilon, seed
        self.next = 0
        self.policies = [self._slot_policy(i) for i in range(slots)]

    def _slot_policy(self, i):
        rng = np.random.default_rng(self.seed + i)

        def act(boards, to_move, legal):
            model = self.models[i]
            if model is None:  # never drawn (its weight is 0), but be safe
                return (rng.random(legal.shape) * legal).argmax(1)
            return net_policy(model, self.device, self.epsilon, int(rng.integers(1 << 30)))(boards, to_move, legal)

        return act

    def add(self, model) -> None:
        self.models[self.next] = frozen_copy(model)
        self.next = (self.next + 1) % len(self.models)

    @property
    def filled(self) -> int:
        return sum(m is not None for m in self.models)


# --- evaluation ----------------------------------------------------------------------

BASELINES = {
    "random": lambda: random_policy(123),
    "heuristic": lambda: agent_policy(HeuristicAgent(123)),
    "emm1": lambda: agent_policy(ExpectiminimaxAgent(1)),
    "emm2": lambda: agent_policy(ExpectiminimaxAgent(2)),
}


def evaluate(policy: Policy, opponent: str, games: int, seed: int = 0) -> dict:
    """Play exactly `games` games against a baseline (seats alternate) and return
    win/draw/loss counts and the score (win = 1, draw = 0.5)."""
    env = VecVsOpponent(games, BASELINES[opponent](), seed=seed, seats="alternate")
    obs, legal = env.reset()
    result = np.full(games, np.nan)
    while np.isnan(result).any():
        g = env.game
        obs, legal, reward, done, _ = env.step(policy(g.boards, g.to_move, legal))
        first = done & np.isnan(result)
        result[first] = reward[first]
    wins, draws = int((result == 1).sum()), int((result == 0).sum())
    return {"wins": wins, "draws": draws, "losses": games - wins - draws, "score": (wins + 0.5 * draws) / games}


def evaluate_and_save(run: "RunLog", model, algo: str, device, step: int, games: int, best_score: float) -> float:
    """The periodic evaluation every trainer runs: score against the baselines, log it,
    save latest.pt, and best.pt when the (heuristic + emm2) average improves.
    Returns the (possibly updated) best score."""
    greedy = net_policy(model, device)
    results = {opp: evaluate(greedy, opp, games, seed=step) for opp in BASELINES}
    score = (results["heuristic"]["score"] + results["emm2"]["score"]) / 2
    run.write(kind="eval", step=step, **{f"score_{k}": v["score"] for k, v in results.items()})
    run.echo("EVAL  " + "  ".join(f"{k} {v['score']:.3f}" for k, v in results.items()))
    save_checkpoint(run.dir / "latest.pt", model, algo, step=step, eval=results)
    if score > best_score:
        save_checkpoint(run.dir / "best.pt", model, algo, step=step, eval=results)
        run.echo(f"      new best (heuristic + emm2 average {score:.3f})")
        return score
    return best_score


# --- logging -------------------------------------------------------------------------


class RunLog:
    """Writes runs/<name>/config.json, appends one JSON object per line to
    runs/<name>/metrics.jsonl, and echoes progress to the console."""

    def __init__(self, run_dir: Path, config: dict):
        self.dir = run_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
        self.file = open(self.dir / "metrics.jsonl", "a")
        self.start = time.time()

    def write(self, **row) -> None:
        row = {"time": round(time.time() - self.start, 1), **row}
        self.file.write(json.dumps(row, default=float) + "\n")
        self.file.flush()

    def echo(self, text: str) -> None:
        print(f"[{(time.time() - self.start) / 60:6.1f} min] {text}", flush=True)


def win_rates(recent: dict[int, deque]) -> dict[str, float | None]:
    """Win rate over the recent training games against each opponent kind, with the
    self-play slots pooled together as "self"."""
    def rate(results):
        return float(np.mean(np.array(results) == 1)) if results else None

    rates = {name: rate(recent[i]) for i, name in enumerate(OPPONENT_NAMES)}
    rates["self"] = rate([r for i in range(len(OPPONENT_NAMES), len(recent)) for r in recent[i]])
    return rates


def fmt(value: float | None) -> str:
    return "  -  " if value is None else f"{value:.2f}"


def run_cli(config_cls, train, doc: str) -> None:
    """Command line for a trainer: every config field becomes a --flag."""
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = config_cls()
    for field, value in asdict(defaults).items():
        kind = (lambda s: s.lower() in ("1", "true", "yes")) if isinstance(value, bool) else type(value)
        parser.add_argument(f"--{field.replace('_', '-')}", type=kind, default=value)
    parser.add_argument("--steps", type=int, dest="total_steps", default=defaults.total_steps)
    train(config_cls(**vars(parser.parse_args())))
