"""Deep Q-learning (Double DQN, dueling head, n-step returns) for Cargo Storm.

Q-learning learns Q(s, a): the expected final result if I play `a` now and play well
afterwards. Its update pulls Q(s, a) toward a *target*: the reward plus the discounted
value of the best move in the next position. DQN does this with a neural network, and
adds two stabilisers: a replay buffer (learn from shuffled past experience, not just the
latest moves) and a slowly-moving target network (so the target doesn't chase itself).

Additions used here, each fixing a known weakness:
- Double DQN: the online net *picks* the next move, the target net *scores* it. Plain
  DQN uses one net for both, which systematically overestimates values; the 50% storm
  makes outcomes noisy, which makes that bias worse.
- Dueling head, n-step returns, mirror augmentation: see nets.py and replay.py.
- Action masking: illegal moves are never chosen and never used in targets.

Exploration is epsilon-greedy: with probability epsilon play a random legal move.
Epsilon decays linearly from 1.0 to 0.05 over the first 40% of training.

Opponents: each game draws from Random, Heuristic, or a frozen snapshot of the learner
itself (self-play). Early games are mostly Random; self-play takes over as it improves.

    python -m cargostorm.rl.dqn --steps 3000000 --name dqn-v1
    python -m cargostorm.rl.dqn --steps 3000000 --name dqn-v1 --resume true
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cargostorm.envs import VecVsOpponent
from cargostorm.rl.common import (RunLog, Snapshots, evaluate_and_save, fmt, frozen_copy, load_checkpoint, opponent_mix,
                                  pick_device, run_cli, training_opponents, win_rates)
from cargostorm.rl.nets import QNet, count_params, masked
from cargostorm.rl.replay import NStepCollector, ReplayBuffer
from cargostorm.rl.resume import (best_score_so_far, load_state, next_multiple, restore_rngs, restore_snapshots,
                                  rng_state, save_state, snapshot_state, truncate_log)


@dataclass
class DQNConfig:
    name: str = "dqn"
    total_steps: int = 3_000_000  # learner moves: the compute budget shared by every algorithm
    n_envs: int = 64
    seed: int = 0
    # network
    channels: int = 64
    blocks: int = 4
    # learning
    lr: float = 2e-4
    batch: int = 256
    gamma: float = 0.99
    n_step: int = 3
    buffer: int = 400_000
    learning_starts: int = 20_000
    updates_per_step: int = 1  # replay ratio = batch * updates / n_envs = 4
    target_tau: float = 0.005  # soft target update per gradient step
    grad_clip: float = 10.0
    # exploration
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.4
    # opponents
    p_random_start: float = 0.6
    p_random_end: float = 0.1
    p_heuristic: float = 0.2
    snapshot_every: int = 100_000
    snapshot_slots: int = 8
    snapshot_epsilon: float = 0.05
    # bookkeeping
    log_every: int = 50_000
    eval_every: int = 250_000
    eval_games: int = 200
    # pausing and resuming (see resume.py)
    resume: bool = False  # continue from runs/<name>/state
    init_from: str = ""  # or warm-start from a checkpoint (.pt) of an interrupted run


def epsilon_at(cfg: DQNConfig, step: int) -> float:
    frac = min(1.0, step / (cfg.eps_decay_frac * cfg.total_steps))
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def train(cfg: DQNConfig) -> Path:
    torch.manual_seed(cfg.seed)
    device = pick_device()
    run = RunLog(Path("runs") / cfg.name, asdict(cfg))
    rng = np.random.default_rng(cfg.seed)

    online = QNet(cfg.channels, cfg.blocks).to(device)
    target = frozen_copy(online)
    opt = torch.optim.Adam(online.parameters(), lr=cfg.lr)
    run.echo(f"DQN on {device}: {count_params(online):,} parameters, budget {cfg.total_steps:,} moves")

    snapshots = Snapshots(cfg.snapshot_slots, device, cfg.snapshot_epsilon, cfg.seed + 100)
    opponents = training_opponents(cfg.seed, snapshots)

    def weights(step):
        return opponent_mix(step, cfg.total_steps, snapshots.filled, cfg.snapshot_slots,
                            cfg.p_random_start, cfg.p_random_end, cfg.p_heuristic)

    env = VecVsOpponent(cfg.n_envs, opponents, seed=cfg.seed)
    buffer = ReplayBuffer(cfg.buffer, cfg.seed)
    collector = NStepCollector(cfg.n_envs, cfg.n_step, cfg.gamma, buffer)

    steps, updates = 0, 0
    best_score, next_log, next_eval, next_snap = -1.0, cfg.log_every, cfg.eval_every, cfg.snapshot_every
    if cfg.resume and (state := load_state(run.dir, buffer)) is not None:
        online.load_state_dict(state["online"])
        target.load_state_dict(state["target"])
        opt.load_state_dict(state["opt"])
        restore_snapshots(snapshots, state["snapshots"], online)
        restore_rngs(state["rngs"], rng, env.rng, env.game.rng)
        steps, updates, best_score = state["steps"], state["updates"], state["best_score"]
        next_log, next_eval, next_snap = state["next_log"], state["next_eval"], state["next_snap"]
        truncate_log(run.dir, steps)
        run.echo(f"resumed at {steps:,} moves with {buffer.size:,} moves of replay memory")
    elif cfg.init_from:
        model, ckpt = load_checkpoint(cfg.init_from, device)
        online.load_state_dict(model.state_dict())
        target = frozen_copy(online)
        steps = int(ckpt["meta"]["step"])
        truncate_log(run.dir, steps)
        best_score = best_score_so_far(run.dir, steps)
        next_log, next_eval = next_multiple(steps, cfg.log_every), next_multiple(steps, cfg.eval_every)
        next_snap = next_multiple(steps, cfg.snapshot_every)
        snapshots.add(online)  # seed the self-play pool with the restored network
        run.echo(f"warm start from {cfg.init_from} at {steps:,} moves (fresh optimizer and replay memory)")
    env.weights = weights(steps)

    obs, legal = env.reset()
    losses, qs = deque(maxlen=500), deque(maxlen=500)
    recent = {i: deque(maxlen=400) for i in range(len(opponents))}
    tick, last_log_steps = time.time(), steps

    while steps < cfg.total_steps:
        # --- act: epsilon-greedy over legal moves ------------------------------------
        eps = epsilon_at(cfg, steps)
        with torch.no_grad():
            q = masked(online(torch.as_tensor(obs, device=device)), torch.as_tensor(legal, device=device))
        actions = q.argmax(1).cpu().numpy()
        explore = rng.random(cfg.n_envs) < eps
        actions = np.where(explore, (rng.random(legal.shape) * legal).argmax(1), actions)

        next_obs, next_legal, reward, done, info = env.step(actions)
        collector.add(obs, actions, reward, next_obs, next_legal, done)
        for e in np.flatnonzero(done):
            recent[int(info["opponent"][e])].append(reward[e])
        obs, legal = next_obs, next_legal
        steps += cfg.n_envs

        # --- learn -------------------------------------------------------------------
        if buffer.size >= cfg.learning_starts:
            for _ in range(cfg.updates_per_step):
                b = {k: torch.as_tensor(v, device=device) for k, v in buffer.sample(cfg.batch).items()}
                q_taken = online(b["obs"]).gather(1, b["action"][:, None]).squeeze(1)
                with torch.no_grad():
                    best_next = masked(online(b["next_obs"]), b["next_legal"]).argmax(1, keepdim=True)  # online picks
                    next_value = target(b["next_obs"]).gather(1, best_next).squeeze(1)  # target scores
                    y = b["reward"] + b["discount"] * next_value
                loss = F.smooth_l1_loss(q_taken, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), cfg.grad_clip)
                opt.step()
                with torch.no_grad():
                    for pt, po in zip(target.parameters(), online.parameters()):
                        pt.lerp_(po, cfg.target_tau)
                updates += 1
                if updates % 10 == 0:
                    losses.append(loss.item())
                    qs.append(q_taken.mean().item())

        # --- self-play snapshots ------------------------------------------------------
        if steps >= next_snap:
            snapshots.add(online)
            next_snap += cfg.snapshot_every
        env.weights = weights(steps)

        # --- logging, evaluation, saving ----------------------------------------------
        if steps >= next_log:
            rate = (steps - last_log_steps) / (time.time() - tick)
            tick, last_log_steps = time.time(), steps
            win = win_rates(recent)
            loss_avg, q_avg = (float(np.mean(losses)), float(np.mean(qs))) if losses else (None, None)
            run.write(kind="train", step=steps, updates=updates, epsilon=eps, loss=loss_avg, mean_q=q_avg,
                      buffer=buffer.size, moves_per_s=rate, **{f"win_{k}": v for k, v in win.items()})
            run.echo(f"step {steps:>9,}  eps {eps:.2f}  loss {fmt(loss_avg)}  Q {fmt(q_avg)}  win vs rand "
                     f"{fmt(win['random'])} heur {fmt(win['heuristic'])} self {fmt(win['self'])}  {rate:,.0f} mv/s")
            next_log += cfg.log_every

        if steps >= next_eval or steps >= cfg.total_steps:
            best_score = evaluate_and_save(run, online, "dqn", device, steps, cfg.eval_games, best_score)
            next_eval += cfg.eval_every
            save_state(run.dir, {
                "online": online.state_dict(), "target": target.state_dict(), "opt": opt.state_dict(),
                "snapshots": snapshot_state(snapshots), "rngs": rng_state(rng, env.rng, env.game.rng),
                "steps": steps, "updates": updates, "best_score": best_score,
                "next_log": next_log, "next_eval": next_eval, "next_snap": next_snap,
            }, buffer)

    run.echo("done")
    return run.dir


if __name__ == "__main__":
    run_cli(DQNConfig, train, __doc__)
