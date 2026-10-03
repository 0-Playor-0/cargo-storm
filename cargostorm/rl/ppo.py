"""Proximal Policy Optimization (PPO) for Cargo Storm.

Where DQN learns *values* and acts greedily on them, PPO learns the *policy* directly:
a probability for each move. After playing a batch of games it nudges up the
probability of moves that turned out better than expected and nudges down the rest.

The pieces:
- Advantage: how much better a move did than the critic expected. Estimated with GAE
  (generalized advantage estimation), which blends short- and long-horizon estimates
  (lambda = 0.95) to trade noise against bias.
- The clip: the new policy may not move any move's probability more than ~20% away from
  the policy that collected the data. Big policy jumps are what make plain policy
  gradients unstable; the clip keeps each update small and safe.
- The critic (value head) learns to predict the final result, which is what makes the
  advantages low-noise.
- Entropy bonus: rewards keeping options open, so the policy doesn't collapse onto one
  move too early. Its weight decays from 0.02 to 0.002: explore early, commit late.
  This is PPO's counterpart to DQN's decaying epsilon.

PPO is *on-policy*: it learns only from games played by the current policy, then throws
them away. That is less sample-efficient than DQN's replay, but more stable.

    python -m cargostorm.rl.ppo --steps 3000000 --name ppo-v2
    python -m cargostorm.rl.ppo --steps 3000000 --name ppo-v2 --resume true
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from cargostorm.engine import SIZE
from cargostorm.envs import VecVsOpponent
from cargostorm.rl.common import (RunLog, Snapshots, evaluate_and_save, fmt, load_checkpoint, opponent_mix, pick_device,
                                  run_cli, training_opponents, win_rates)
from cargostorm.rl.nets import PolicyNet, count_params, masked
from cargostorm.rl.resume import (best_score_so_far, load_state, next_multiple, restore_rngs, restore_snapshots,
                                  rng_state, save_state, snapshot_state, truncate_log)


@dataclass
class PPOConfig:
    name: str = "ppo"
    total_steps: int = 3_000_000  # learner moves, same budget as DQN
    n_envs: int = 128
    rollout: int = 64  # moves per game slot per update: 128 x 64 = 8192 moves per batch
    seed: int = 0
    # network
    channels: int = 64
    blocks: int = 4
    # learning
    lr: float = 1e-4  # v1 used 3e-4: updates grew too large as the policy sharpened, and it collapsed
    lr_end: float = 1e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epochs: int = 4
    minibatch: int = 1024
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_start: float = 0.02
    ent_end: float = 0.005
    max_grad_norm: float = 0.5
    target_kl: float = 0.03  # stop an update the moment the policy has moved this far from the data's policy
    mirror: bool = True
    # opponents (same mix as DQN)
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


def gae(rewards, values, dones, last_value, gamma, lam):
    """Advantages and returns for a (T, N) rollout. `dones[t]` means the game ended on
    move t, so nothing after it is bootstrapped into it."""
    T = len(rewards)
    adv = np.zeros_like(rewards)
    running = np.zeros_like(last_value)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        alive = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * alive - values[t]
        running = delta + gamma * lam * alive * running
        adv[t] = running
    return adv, adv + values


def train(cfg: PPOConfig) -> Path:
    torch.manual_seed(cfg.seed)
    device = pick_device()
    run = RunLog(Path("runs") / cfg.name, asdict(cfg))

    model = PolicyNet(cfg.channels, cfg.blocks).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    run.echo(f"PPO on {device}: {count_params(model):,} parameters, budget {cfg.total_steps:,} moves")

    snapshots = Snapshots(cfg.snapshot_slots, device, cfg.snapshot_epsilon, cfg.seed + 100)
    opponents = training_opponents(cfg.seed, snapshots)

    def weights(step):
        return opponent_mix(step, cfg.total_steps, snapshots.filled, cfg.snapshot_slots,
                            cfg.p_random_start, cfg.p_random_end, cfg.p_heuristic)

    env = VecVsOpponent(cfg.n_envs, opponents, seed=cfg.seed)
    T, N = cfg.rollout, cfg.n_envs
    buf_obs = np.zeros((T, N, 3, SIZE, SIZE), np.float32)
    buf_legal = np.zeros((T, N, SIZE), bool)
    buf_act = np.zeros((T, N), np.int64)
    buf_logp = np.zeros((T, N), np.float32)
    buf_val = np.zeros((T, N), np.float32)
    buf_rew = np.zeros((T, N), np.float32)
    buf_done = np.zeros((T, N), np.float32)

    steps = 0
    best_score, next_log, next_eval, next_snap = -1.0, cfg.log_every, cfg.eval_every, cfg.snapshot_every
    if cfg.resume and (state := load_state(run.dir)) is not None:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        restore_snapshots(snapshots, state["snapshots"], model)
        restore_rngs(state["rngs"], env.rng, env.game.rng)
        steps, best_score = state["steps"], state["best_score"]
        next_log, next_eval, next_snap = state["next_log"], state["next_eval"], state["next_snap"]
        truncate_log(run.dir, steps)
        run.echo(f"resumed at {steps:,} moves")
    elif cfg.init_from:
        loaded, ckpt = load_checkpoint(cfg.init_from, device)
        model.load_state_dict(loaded.state_dict())
        steps = int(ckpt["meta"]["step"])
        truncate_log(run.dir, steps)
        best_score = best_score_so_far(run.dir, steps)
        next_log, next_eval = next_multiple(steps, cfg.log_every), next_multiple(steps, cfg.eval_every)
        next_snap = next_multiple(steps, cfg.snapshot_every)
        snapshots.add(model)
        run.echo(f"warm start from {cfg.init_from} at {steps:,} moves (fresh optimizer)")
    env.weights = weights(steps)

    obs, legal = env.reset()
    recent = {i: deque(maxlen=400) for i in range(len(opponents))}
    stats = deque(maxlen=20)
    tick, last_log_steps = time.time(), steps

    def to_t(x):
        return torch.as_tensor(x, device=device)

    while steps < cfg.total_steps:
        progress = steps / cfg.total_steps
        for group in opt.param_groups:
            group["lr"] = cfg.lr + progress * (cfg.lr_end - cfg.lr)
        ent_coef = cfg.ent_start + progress * (cfg.ent_end - cfg.ent_start)

        # --- collect a rollout with the current policy ---------------------------------
        for t in range(T):
            with torch.no_grad():
                logits, value = model(to_t(obs))
                dist = Categorical(logits=masked(logits, to_t(legal)))
                action = dist.sample()
            buf_obs[t], buf_legal[t] = obs, legal
            buf_act[t] = action.cpu().numpy()
            buf_logp[t] = dist.log_prob(action).cpu().numpy()
            buf_val[t] = value.cpu().numpy()
            obs, legal, reward, done, info = env.step(buf_act[t])
            buf_rew[t], buf_done[t] = reward, done
            for e in np.flatnonzero(done):
                recent[int(info["opponent"][e])].append(reward[e])
        steps += T * N
        with torch.no_grad():
            last_value = model(to_t(obs))[1].cpu().numpy()
        adv, ret = gae(buf_rew, buf_val, buf_done, last_value, cfg.gamma, cfg.gae_lambda)

        b_obs = buf_obs.reshape(T * N, 3, SIZE, SIZE)
        b_legal = buf_legal.reshape(T * N, SIZE)
        b_act, b_logp = buf_act.reshape(-1), buf_logp.reshape(-1)
        b_adv, b_ret = adv.reshape(-1), ret.reshape(-1)
        if cfg.mirror:
            # The mirrored position is equally good, with the mirrored move. Its old
            # log-probability must come from the same (pre-update) policy.
            m_obs, m_legal, m_act = b_obs[..., ::-1].copy(), b_legal[:, ::-1].copy(), SIZE - 1 - b_act
            with torch.no_grad():
                m_logp = []
                for i in range(0, len(m_obs), 4096):
                    lg = masked(model(to_t(m_obs[i:i + 4096]))[0], to_t(m_legal[i:i + 4096]))
                    m_logp.append(Categorical(logits=lg).log_prob(to_t(m_act[i:i + 4096])).cpu().numpy())
            b_obs, b_legal = np.concatenate([b_obs, m_obs]), np.concatenate([b_legal, m_legal])
            b_act, b_logp = np.concatenate([b_act, m_act]), np.concatenate([b_logp, np.concatenate(m_logp)])
            b_adv, b_ret = np.concatenate([b_adv, b_adv]), np.concatenate([b_ret, b_ret])

        # --- learn: several epochs of clipped updates over the batch --------------------
        g_obs, g_legal, g_act, g_logp = to_t(b_obs), to_t(b_legal), to_t(b_act), to_t(b_logp)
        g_adv, g_ret = to_t(b_adv), to_t(b_ret)
        g_adv = (g_adv - g_adv.mean()) / (g_adv.std() + 1e-8)
        n = len(b_act)
        kls, clipfracs = [], []
        stopped_early = False
        for _ in range(cfg.epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, cfg.minibatch):
                mb = perm[i:i + cfg.minibatch]
                logits, value = model(g_obs[mb])
                dist = Categorical(logits=masked(logits, g_legal[mb]))
                logp = dist.log_prob(g_act[mb])
                log_ratio = logp - g_logp[mb]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    kl = ((ratio - 1) - log_ratio).mean().item()
                kls.append(kl)
                # Checked before every step (v1 only checked after a full pass, by which time
                # the damage was done): once the policy has drifted too far, stop updating.
                if kl > 1.5 * cfg.target_kl:
                    stopped_early = True
                    break
                pg_loss = -torch.min(ratio * g_adv[mb], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * g_adv[mb]).mean()
                v_loss = 0.5 * (value - g_ret[mb]).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = pg_loss + cfg.vf_coef * v_loss - ent_coef * entropy
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    clipfracs.append(((ratio - 1).abs() > cfg.clip).float().mean().item())
            if stopped_early:
                break
        stats.append(dict(pg_loss=pg_loss.item(), v_loss=v_loss.item(), entropy=entropy.item(), kl=float(np.mean(kls)),
                          clipfrac=float(np.mean(clipfracs)) if clipfracs else 0.0, minibatches=len(clipfracs)))

        # --- self-play snapshots ------------------------------------------------------
        if steps >= next_snap:
            snapshots.add(model)
            next_snap += cfg.snapshot_every
        env.weights = weights(steps)

        # --- logging, evaluation, saving ----------------------------------------------
        if steps >= next_log:
            rate = (steps - last_log_steps) / (time.time() - tick)
            tick, last_log_steps = time.time(), steps
            win = win_rates(recent)
            avg = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
            run.write(kind="train", step=steps, ent_coef=ent_coef, moves_per_s=rate,
                      **avg, **{f"win_{k}": v for k, v in win.items()})
            run.echo(f"step {steps:>9,}  entropy {avg['entropy']:.2f}  kl {avg['kl']:.3f}  clip {avg['clipfrac']:.2f}  "
                     f"vloss {avg['v_loss']:.3f}  win vs rand {fmt(win['random'])} heur {fmt(win['heuristic'])} "
                     f"self {fmt(win['self'])}  {rate:,.0f} mv/s")
            next_log = steps + cfg.log_every

        if steps >= next_eval or steps >= cfg.total_steps:
            best_score = evaluate_and_save(run, model, "ppo", device, steps, cfg.eval_games, best_score)
            next_eval += cfg.eval_every
            save_state(run.dir, {
                "model": model.state_dict(), "opt": opt.state_dict(), "snapshots": snapshot_state(snapshots),
                "rngs": rng_state(env.rng, env.game.rng), "steps": steps, "best_score": best_score,
                "next_log": next_log, "next_eval": next_eval, "next_snap": next_snap,
            })

    run.echo("done")
    return run.dir


if __name__ == "__main__":
    run_cli(PPOConfig, train, __doc__)
