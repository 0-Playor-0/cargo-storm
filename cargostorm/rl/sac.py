"""Soft Actor-Critic, discrete-action version (Christodoulou 2019), for Cargo Storm.

SAC maximises reward *plus* the policy's entropy: it wants to win while staying as
unpredictable as it can afford. That bakes exploration into the objective itself, rather
than bolting it on (DQN's epsilon) or adding a bonus (PPO's entropy term).

The pieces:
- An actor (policy over 7 columns) and two critics Q1, Q2 with slow target copies.
  Using the smaller of the two critics' estimates fights the same overestimation that
  Double DQN fights.
- The soft value of a position: V(s) = sum_a pi(a|s) * (min Q(s, a) - alpha * log pi(a|s)).
  In the discrete case this sum is exact (7 moves), so no sampling is needed.
- Critic target: r + gamma^n * V(s') from the target critics, learned off-policy from the
  same n-step replay buffer as DQN (with mirror augmentation).
- Actor update: move pi toward softmax(min Q / alpha): favour good moves, but keep
  entropy, priced by the temperature alpha.
- Alpha tunes itself to hold the policy's entropy at a target. The target is a fraction
  of the maximum possible entropy (log of the number of legal moves) and decays from
  60% to 20%: explore early, commit late.

Discrete SAC is known to be touchy (the entropy target matters a lot); expect it to be
the hardest of the three to get right. That is part of what we're measuring.

    python -m cargostorm.rl.sac --steps 3000000 --name sac-v1
    python -m cargostorm.rl.sac --steps 3000000 --name sac-v1 --resume true
"""

from __future__ import annotations

import math
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
from cargostorm.rl.nets import ActorNet, QNet, count_params, masked
from cargostorm.rl.replay import NStepCollector, ReplayBuffer
from cargostorm.rl.resume import (best_score_so_far, load_state, next_multiple, restore_rngs, restore_snapshots,
                                  rng_state, save_state, snapshot_state, truncate_log)


@dataclass
class SACConfig:
    name: str = "sac"
    total_steps: int = 3_000_000  # learner moves, same budget as DQN and PPO
    n_envs: int = 64
    seed: int = 0
    # networks
    channels: int = 64
    blocks: int = 4
    # learning (replay settings match DQN)
    actor_lr: float = 1e-4
    critic_lr: float = 2e-4
    alpha_lr: float = 3e-4
    batch: int = 256
    gamma: float = 0.99
    n_step: int = 3
    buffer: int = 400_000
    learning_starts: int = 20_000
    updates_per_step: int = 1
    target_tau: float = 0.005
    grad_clip: float = 10.0
    # entropy target, as a fraction of log(#legal moves)
    entropy_frac_start: float = 0.6
    entropy_frac_end: float = 0.2
    entropy_decay_frac: float = 0.5
    init_alpha: float = 0.1
    # opponents (same mix as DQN and PPO)
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
    init_from: str = ""  # or warm-start the actor from a checkpoint of an interrupted run
    critic_warmup: int = 100_000  # after a warm start: moves of critic-only learning (fresh critics)


def entropy_frac_at(cfg: SACConfig, step: int) -> float:
    frac = min(1.0, step / (cfg.entropy_decay_frac * cfg.total_steps))
    return cfg.entropy_frac_start + frac * (cfg.entropy_frac_end - cfg.entropy_frac_start)


def policy_terms(actor, obs, legal):
    """Probabilities and log-probabilities over legal moves (illegal: p = 0, log p = 0)."""
    logits = masked(actor(obs), legal)
    log_p = F.log_softmax(logits, dim=1)
    p = log_p.exp()
    return p, torch.where(legal, log_p, torch.zeros_like(log_p))


def train(cfg: SACConfig) -> Path:
    torch.manual_seed(cfg.seed)
    device = pick_device()
    run = RunLog(Path("runs") / cfg.name, asdict(cfg))

    actor = ActorNet(cfg.channels, cfg.blocks).to(device)
    q1, q2 = QNet(cfg.channels, cfg.blocks).to(device), QNet(cfg.channels, cfg.blocks).to(device)
    q1_target, q2_target = frozen_copy(q1), frozen_copy(q2)
    log_alpha = torch.tensor(math.log(cfg.init_alpha), device=device, requires_grad=True)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=cfg.critic_lr)
    alpha_opt = torch.optim.Adam([log_alpha], lr=cfg.alpha_lr)
    run.echo(f"SAC on {device}: actor {count_params(actor):,} + 2 critics {2 * count_params(q1):,} parameters, "
             f"budget {cfg.total_steps:,} moves")

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
    actor_frozen_until = 0  # during a critic warm-up the actor and temperature don't learn
    nets = {"actor": actor, "q1": q1, "q2": q2, "q1_target": q1_target, "q2_target": q2_target}
    optims = {"actor": actor_opt, "critic": critic_opt, "alpha": alpha_opt}
    if cfg.resume and (state := load_state(run.dir, buffer)) is not None:
        for k, net in nets.items():
            net.load_state_dict(state[k])
        for k, o in optims.items():
            o.load_state_dict(state[f"opt_{k}"])
        with torch.no_grad():
            log_alpha.copy_(state["log_alpha"].to(device))
        restore_snapshots(snapshots, state["snapshots"], actor)
        restore_rngs(state["rngs"], env.rng, env.game.rng)
        steps, updates, best_score = state["steps"], state["updates"], state["best_score"]
        next_log, next_eval, next_snap = state["next_log"], state["next_eval"], state["next_snap"]
        actor_frozen_until = state.get("actor_frozen_until", 0)
        truncate_log(run.dir, steps)
        run.echo(f"resumed at {steps:,} moves with {buffer.size:,} moves of replay memory")
    elif cfg.init_from:
        model, ckpt = load_checkpoint(cfg.init_from, device)
        actor.load_state_dict(model.state_dict())
        steps = int(ckpt["meta"]["step"])
        truncate_log(run.dir, steps)
        best_score = best_score_so_far(run.dir, steps)
        next_log, next_eval = next_multiple(steps, cfg.log_every), next_multiple(steps, cfg.eval_every)
        next_snap = next_multiple(steps, cfg.snapshot_every)
        snapshots.add(actor)
        actor_frozen_until = steps + cfg.critic_warmup
        run.echo(f"warm start from {cfg.init_from} at {steps:,} moves: actor restored, critics, optimizers and "
                 f"replay memory fresh; actor frozen for {cfg.critic_warmup:,} moves while the critics catch up")
    env.weights = weights(steps)

    obs, legal = env.reset()
    recent = {i: deque(maxlen=400) for i in range(len(opponents))}
    tracked = deque(maxlen=200)
    tick, last_log_steps = time.time(), steps

    while steps < cfg.total_steps:
        # --- act: sample from the stochastic policy -----------------------------------
        with torch.no_grad():
            p, _ = policy_terms(actor, torch.as_tensor(obs, device=device), torch.as_tensor(legal, device=device))
            actions = torch.multinomial(p, 1).squeeze(1).cpu().numpy()
        next_obs, next_legal, reward, done, info = env.step(actions)
        collector.add(obs, actions, reward, next_obs, next_legal, done)
        for e in np.flatnonzero(done):
            recent[int(info["opponent"][e])].append(reward[e])
        obs, legal = next_obs, next_legal
        steps += cfg.n_envs

        # --- learn ---------------------------------------------------------------------
        if buffer.size >= cfg.learning_starts:
            frac = entropy_frac_at(cfg, steps)
            for _ in range(cfg.updates_per_step):
                b = {k: torch.as_tensor(v, device=device) for k, v in buffer.sample(cfg.batch).items()}
                alpha = log_alpha.exp().detach()

                # Critics: regress Q(s, a) onto r + gamma^n * soft V(s').
                with torch.no_grad():
                    p_next, logp_next = policy_terms(actor, b["next_obs"], b["next_legal"])
                    q_next = torch.min(q1_target(b["next_obs"]), q2_target(b["next_obs"]))
                    v_next = (p_next * (q_next - alpha * logp_next)).sum(1)
                    y = b["reward"] + b["discount"] * v_next
                a = b["action"][:, None]
                q1_taken, q2_taken = q1(b["obs"]).gather(1, a).squeeze(1), q2(b["obs"]).gather(1, a).squeeze(1)
                critic_loss = F.smooth_l1_loss(q1_taken, y) + F.smooth_l1_loss(q2_taken, y)
                critic_opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), cfg.grad_clip)
                critic_opt.step()

                # Actor: minimise E_pi[alpha * log pi - min Q] (exact sum over 7 moves).
                legal_now = b["obs"][:, 0, 0, :] > 0.5  # plane 0 = empty; top row empty = legal column
                p_now, logp_now = policy_terms(actor, b["obs"], legal_now)
                with torch.no_grad():
                    q_now = torch.min(q1(b["obs"]), q2(b["obs"]))
                actor_loss = (p_now * (alpha * logp_now - q_now)).sum(1).mean()
                entropy = -(p_now * logp_now).sum(1).detach()
                target = frac * torch.log(legal_now.sum(1).float())
                if steps >= actor_frozen_until:
                    actor_opt.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.grad_clip)
                    actor_opt.step()

                    # Temperature: raise alpha if entropy is below target, lower it if above.
                    alpha_loss = (log_alpha * (entropy - target)).mean()
                    alpha_opt.zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    alpha_opt.step()

                with torch.no_grad():
                    for target_net, net in ((q1_target, q1), (q2_target, q2)):
                        for pt, po in zip(target_net.parameters(), net.parameters()):
                            pt.lerp_(po, cfg.target_tau)
                updates += 1
                if updates % 10 == 0:
                    tracked.append(dict(critic_loss=critic_loss.item(), actor_loss=actor_loss.item(),
                                        alpha=alpha.item(), entropy=entropy.mean().item(),
                                        target_entropy=target.mean().item(), mean_q=q1_taken.mean().item()))

        # --- self-play snapshots ------------------------------------------------------
        if steps >= next_snap:
            snapshots.add(actor)
            next_snap += cfg.snapshot_every
        env.weights = weights(steps)

        # --- logging, evaluation, saving ----------------------------------------------
        if steps >= next_log:
            rate = (steps - last_log_steps) / (time.time() - tick)
            tick, last_log_steps = time.time(), steps
            win = win_rates(recent)
            avg = {k: float(np.mean([t[k] for t in tracked])) for k in tracked[0]} if tracked else {}
            run.write(kind="train", step=steps, updates=updates, moves_per_s=rate,
                      **avg, **{f"win_{k}": v for k, v in win.items()})
            run.echo(f"step {steps:>9,}  alpha {fmt(avg.get('alpha'))}  entropy {fmt(avg.get('entropy'))}"
                     f"/{fmt(avg.get('target_entropy'))}  Q {fmt(avg.get('mean_q'))}  win vs rand {fmt(win['random'])} "
                     f"heur {fmt(win['heuristic'])} self {fmt(win['self'])}  {rate:,.0f} mv/s")
            next_log += cfg.log_every

        if steps >= next_eval or steps >= cfg.total_steps:
            best_score = evaluate_and_save(run, actor, "sac", device, steps, cfg.eval_games, best_score)
            next_eval += cfg.eval_every
            save_state(run.dir, {
                **{k: net.state_dict() for k, net in nets.items()},
                **{f"opt_{k}": o.state_dict() for k, o in optims.items()},
                "log_alpha": log_alpha.detach().cpu(), "snapshots": snapshot_state(snapshots),
                "rngs": rng_state(env.rng, env.game.rng), "steps": steps, "updates": updates, "best_score": best_score,
                "next_log": next_log, "next_eval": next_eval, "next_snap": next_snap, "actor_frozen_until": actor_frozen_until,
            }, buffer)

    run.echo("done")
    return run.dir


if __name__ == "__main__":
    run_cli(SACConfig, train, __doc__)
