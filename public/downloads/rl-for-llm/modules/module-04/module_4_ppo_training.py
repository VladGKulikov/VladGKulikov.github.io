#!/usr/bin/env python3
"""Self-contained PPO training utilities for RL for LLM, Module 4.

The implementation intentionally exposes the pieces that are often hidden by
an environment/vectorization framework:

* CartPole dynamics and separate termination/time-limit truncation;
* fixed-size rollout collection with a nonterminal buffer boundary;
* old log-probabilities and old values stored before optimization;
* generalized advantage estimation with separate bootstrap and trace masks;
* repeated, shuffled minibatch PPO updates;
* clipped policy and value objectives, entropy regularization, gradient-norm
  clipping, and sample diagnostics;
* deterministic multi-seed evaluation on CPU.

It is compact enough to audit but it is not a one-step toy implementation.  The
same tensor contracts appear in large-scale PPO systems; only the policy and
environment are deliberately small.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import math
import random

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass(frozen=True)
class PPOConfig:
    seed: int = 0
    num_envs: int = 8
    rollout_steps: int = 64
    updates: int = 24
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    update_epochs: int = 4
    minibatch_size: int = 256
    clip_epsilon: float = 0.2
    value_clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    hidden_size: int = 64
    max_episode_steps: int = 500
    evaluation_episodes: int = 8
    evaluation_every: int = 1

    def validate(self) -> None:
        if self.num_envs < 1 or self.rollout_steps < 2 or self.updates < 1:
            raise ValueError("positive num_envs/updates and rollout_steps >= 2 are required")
        if not (0.0 <= self.gamma <= 1.0 and 0.0 <= self.gae_lambda <= 1.0):
            raise ValueError("gamma and gae_lambda must lie in [0, 1]")
        batch_size = self.num_envs * self.rollout_steps
        if self.minibatch_size < 1 or self.minibatch_size > batch_size:
            raise ValueError("minibatch_size must lie in [1, rollout batch size]")
        if self.update_epochs < 1 or self.learning_rate <= 0.0:
            raise ValueError("positive update_epochs and learning_rate are required")
        if self.max_episode_steps < 1 or self.evaluation_episodes < 1:
            raise ValueError("positive episode limits are required")


@dataclass
class RolloutBatch:
    observations: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    rewards: np.ndarray
    old_values: np.ndarray
    next_values: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    trace_cuts: np.ndarray
    advantages: np.ndarray
    value_targets: np.ndarray
    completed_returns: list[float]
    completed_lengths: list[int]

    @property
    def batch_size(self) -> int:
        return int(self.actions.size)


@dataclass(frozen=True)
class UpdateDiagnostics:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    ratio_outside_fraction: float
    active_clip_fraction: float
    gradient_norm: float
    epochs_completed: int


class CartPoleBatch:
    """A small vectorized implementation of the standard CartPole equations.

    Failure of the pole/cart is returned as ``terminated``. Reaching
    ``max_episode_steps`` is returned as ``truncated``. Environments that end
    are reset immediately for the next action, while ``bootstrap_observation``
    retains the final observation before reset.
    """

    gravity = 9.8
    mass_cart = 1.0
    mass_pole = 0.1
    total_mass = mass_cart + mass_pole
    half_length = 0.5
    polemass_length = mass_pole * half_length
    force_mag = 10.0
    tau = 0.02
    theta_threshold = 12.0 * 2.0 * math.pi / 360.0
    x_threshold = 2.4

    def __init__(self, num_envs: int, *, seed: int, max_episode_steps: int = 500) -> None:
        if num_envs < 1 or max_episode_steps < 1:
            raise ValueError("num_envs and max_episode_steps must be positive")
        self.num_envs = int(num_envs)
        self.max_episode_steps = int(max_episode_steps)
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros((self.num_envs, 4), dtype=np.float32)
        self.elapsed = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_return = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_length = np.zeros(self.num_envs, dtype=np.int32)
        self.reset()

    def _sample_state(self, count: int) -> np.ndarray:
        return self.rng.uniform(-0.05, 0.05, size=(count, 4)).astype(np.float32)

    def reset(self, indices: np.ndarray | list[int] | None = None) -> np.ndarray:
        if indices is None:
            idx = np.arange(self.num_envs)
        else:
            idx = np.asarray(indices, dtype=int)
        if idx.size:
            self.state[idx] = self._sample_state(int(idx.size))
            self.elapsed[idx] = 0
            self.episode_return[idx] = 0.0
            self.episode_length[idx] = 0
        return self.state.copy()

    def step(self, actions: np.ndarray) -> dict[str, Any]:
        a = np.asarray(actions, dtype=np.int64)
        if a.shape != (self.num_envs,) or np.any((a < 0) | (a > 1)):
            raise ValueError("actions must be a binary vector of shape [num_envs]")

        x, x_dot, theta, theta_dot = self.state.T.astype(np.float64)
        force = np.where(a == 1, self.force_mag, -self.force_mag)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        temp = (force + self.polemass_length * theta_dot**2 * sin_theta) / self.total_mass
        theta_acc = (
            self.gravity * sin_theta - cos_theta * temp
        ) / (
            self.half_length * (4.0 / 3.0 - self.mass_pole * cos_theta**2 / self.total_mass)
        )
        x_acc = temp - self.polemass_length * theta_acc * cos_theta / self.total_mass

        # Explicit Euler, matching the common CartPole implementation.
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc
        self.state = np.stack([x, x_dot, theta, theta_dot], axis=1).astype(np.float32)
        self.elapsed += 1
        self.episode_return += 1.0
        self.episode_length += 1

        terminated = (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta < -self.theta_threshold)
            | (theta > self.theta_threshold)
        )
        truncated = (self.elapsed >= self.max_episode_steps) & (~terminated)
        ended = terminated | truncated
        bootstrap_observation = self.state.copy()
        rewards = np.ones(self.num_envs, dtype=np.float32)

        completed_returns = self.episode_return[ended].astype(float).tolist()
        completed_lengths = self.episode_length[ended].astype(int).tolist()
        if np.any(ended):
            self.reset(np.flatnonzero(ended))

        return {
            "observation": self.state.copy(),
            "bootstrap_observation": bootstrap_observation,
            "reward": rewards,
            "terminated": terminated.astype(bool),
            "truncated": truncated.astype(bool),
            "completed_returns": completed_returns,
            "completed_lengths": completed_lengths,
        }


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int = 4, action_dim: int = 2, hidden_size: int = 64) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(observations)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)

    def distribution_and_value(self, observations: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        logits, value = self(observations)
        return Categorical(logits=logits), value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Determinism is inexpensive for this small CPU network.
    torch.use_deterministic_algorithms(True)


def _vector_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    trace_cuts: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    if not (rewards.shape == values.shape == next_values.shape == terminated.shape == trace_cuts.shape):
        raise ValueError("all rollout arrays must have matching [T, N] shapes")
    deltas = rewards + gamma * (~terminated).astype(np.float32) * next_values - values
    advantages = np.empty_like(deltas, dtype=np.float32)
    running = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in range(rewards.shape[0] - 1, -1, -1):
        running = deltas[t] + gamma * gae_lambda * (~trace_cuts[t]).astype(np.float32) * running
        advantages[t] = running
    return advantages


@torch.no_grad()
def collect_rollout(
    model: ActorCritic,
    env: CartPoleBatch,
    observation: np.ndarray,
    config: PPOConfig,
    *,
    device: torch.device,
) -> tuple[RolloutBatch, np.ndarray]:
    t_steps, n_envs = config.rollout_steps, config.num_envs
    obs_buf = np.empty((t_steps, n_envs, 4), dtype=np.float32)
    action_buf = np.empty((t_steps, n_envs), dtype=np.int64)
    logp_buf = np.empty((t_steps, n_envs), dtype=np.float32)
    reward_buf = np.empty((t_steps, n_envs), dtype=np.float32)
    value_buf = np.empty((t_steps, n_envs), dtype=np.float32)
    next_value_buf = np.empty((t_steps, n_envs), dtype=np.float32)
    terminated_buf = np.empty((t_steps, n_envs), dtype=bool)
    truncated_buf = np.empty((t_steps, n_envs), dtype=bool)
    trace_cut_buf = np.empty((t_steps, n_envs), dtype=bool)
    completed_returns: list[float] = []
    completed_lengths: list[int] = []

    model.eval()
    current = np.asarray(observation, dtype=np.float32)
    for t in range(t_steps):
        obs_tensor = torch.as_tensor(current, dtype=torch.float32, device=device)
        distribution, values = model.distribution_and_value(obs_tensor)
        actions = distribution.sample()
        log_probs = distribution.log_prob(actions)
        transition = env.step(actions.cpu().numpy())
        bootstrap_tensor = torch.as_tensor(
            transition["bootstrap_observation"], dtype=torch.float32, device=device
        )
        _, next_values = model(bootstrap_tensor)

        obs_buf[t] = current
        action_buf[t] = actions.cpu().numpy()
        logp_buf[t] = log_probs.cpu().numpy()
        reward_buf[t] = transition["reward"]
        value_buf[t] = values.cpu().numpy()
        next_value_buf[t] = next_values.cpu().numpy()
        terminated_buf[t] = transition["terminated"]
        truncated_buf[t] = transition["truncated"]
        trace_cut_buf[t] = transition["terminated"] | transition["truncated"]
        completed_returns.extend(transition["completed_returns"])
        completed_lengths.extend(transition["completed_lengths"])
        current = transition["observation"]

    # The last record of this fixed-size buffer has no A_{t+1} inside the
    # current array. It still bootstraps from next_value_buf.
    trace_cut_buf[-1, :] = True
    advantages = _vector_gae(
        reward_buf,
        value_buf,
        next_value_buf,
        terminated_buf,
        trace_cut_buf,
        config.gamma,
        config.gae_lambda,
    )
    targets = advantages + value_buf
    batch = RolloutBatch(
        observations=obs_buf.reshape(-1, 4),
        actions=action_buf.reshape(-1),
        old_log_probs=logp_buf.reshape(-1),
        rewards=reward_buf.reshape(-1),
        old_values=value_buf.reshape(-1),
        next_values=next_value_buf.reshape(-1),
        terminated=terminated_buf.reshape(-1),
        truncated=truncated_buf.reshape(-1),
        trace_cuts=trace_cut_buf.reshape(-1),
        advantages=advantages.reshape(-1),
        value_targets=targets.reshape(-1),
        completed_returns=completed_returns,
        completed_lengths=completed_lengths,
    )
    return batch, current


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    config: PPOConfig,
    *,
    device: torch.device,
    rng: np.random.Generator,
) -> UpdateDiagnostics:
    model.train()
    obs = torch.as_tensor(rollout.observations, dtype=torch.float32, device=device)
    actions = torch.as_tensor(rollout.actions, dtype=torch.long, device=device)
    old_logp = torch.as_tensor(rollout.old_log_probs, dtype=torch.float32, device=device)
    old_values = torch.as_tensor(rollout.old_values, dtype=torch.float32, device=device)
    targets = torch.as_tensor(rollout.value_targets, dtype=torch.float32, device=device)
    advantages = torch.as_tensor(rollout.advantages, dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    diagnostics: list[tuple[float, ...]] = []
    epochs_completed = 0
    stop_early = False
    for epoch in range(config.update_epochs):
        indices = rng.permutation(rollout.batch_size)
        for start in range(0, rollout.batch_size, config.minibatch_size):
            mb_np = indices[start : start + config.minibatch_size]
            mb = torch.as_tensor(mb_np, dtype=torch.long, device=device)
            distribution, new_values = model.distribution_and_value(obs[mb])
            new_logp = distribution.log_prob(actions[mb])
            entropy = distribution.entropy().mean()
            log_ratio = new_logp - old_logp[mb]
            ratio = log_ratio.exp()
            a = advantages[mb]
            unclipped = ratio * a
            clipped = torch.clamp(
                ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon
            ) * a
            policy_loss = -torch.minimum(unclipped, clipped).mean()

            value_delta = new_values - old_values[mb]
            clipped_values = old_values[mb] + torch.clamp(
                value_delta,
                -config.value_clip_epsilon,
                config.value_clip_epsilon,
            )
            raw_value_loss = (new_values - targets[mb]).pow(2)
            clipped_value_loss = (clipped_values - targets[mb]).pow(2)
            value_loss = 0.5 * torch.maximum(raw_value_loss, clipped_value_loss).mean()
            total_loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                outside = (
                    (ratio < 1.0 - config.clip_epsilon)
                    | (ratio > 1.0 + config.clip_epsilon)
                ).float().mean()
                active = (torch.minimum(unclipped, clipped) < unclipped - 1e-12).float().mean()
                diagnostics.append(
                    (
                        float(policy_loss),
                        float(value_loss),
                        float(entropy),
                        float(approx_kl),
                        float(outside),
                        float(active),
                        float(grad_norm),
                    )
                )
                if config.target_kl is not None and float(approx_kl) > 1.5 * config.target_kl:
                    stop_early = True
                    break
        epochs_completed = epoch + 1
        if stop_early:
            break

    means = np.mean(np.asarray(diagnostics, dtype=float), axis=0)
    return UpdateDiagnostics(
        policy_loss=float(means[0]),
        value_loss=float(means[1]),
        entropy=float(means[2]),
        approx_kl=float(means[3]),
        ratio_outside_fraction=float(means[4]),
        active_clip_fraction=float(means[5]),
        gradient_norm=float(means[6]),
        epochs_completed=int(epochs_completed),
    )


@torch.no_grad()
def evaluate_policy(
    model: ActorCritic,
    *,
    seed: int,
    episodes: int = 8,
    max_episode_steps: int = 500,
    device: torch.device | None = None,
) -> tuple[float, float, list[float]]:
    device = device or torch.device("cpu")
    env = CartPoleBatch(1, seed=seed, max_episode_steps=max_episode_steps)
    observation = env.state.copy()
    returns: list[float] = []
    model.eval()
    while len(returns) < episodes:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
        logits, _ = model(obs_tensor)
        action = torch.argmax(logits, dim=-1).cpu().numpy()
        transition = env.step(action)
        returns.extend(transition["completed_returns"])
        observation = transition["observation"]
    selected = returns[:episodes]
    return float(np.mean(selected)), float(np.std(selected, ddof=0)), selected


def train_ppo(config: PPOConfig) -> dict[str, Any]:
    """Train one deterministic PPO run and return serializable history."""
    config.validate()
    seed_everything(config.seed)
    # Small matrix operations are faster and more reproducible with one thread.
    torch.set_num_threads(1)
    device = torch.device("cpu")
    env = CartPoleBatch(
        config.num_envs,
        seed=config.seed + 101,
        max_episode_steps=config.max_episode_steps,
    )
    model = ActorCritic(hidden_size=config.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    rng = np.random.default_rng(config.seed + 202)
    observation = env.state.copy()
    recent_completed: list[float] = []

    history: dict[str, list[float | int]] = {
        "update": [],
        "environment_steps": [],
        "rollout_return_mean": [],
        "evaluation_return_mean": [],
        "evaluation_return_std": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "ratio_outside_fraction": [],
        "active_clip_fraction": [],
        "gradient_norm": [],
        "epochs_completed": [],
    }

    for update in range(1, config.updates + 1):
        rollout, observation = collect_rollout(model, env, observation, config, device=device)
        recent_completed.extend(rollout.completed_returns)
        recent_completed = recent_completed[-100:]
        diagnostic = ppo_update(model, optimizer, rollout, config, device=device, rng=rng)
        if update % config.evaluation_every == 0 or update == config.updates:
            eval_mean, eval_std, _ = evaluate_policy(
                model,
                seed=config.seed + 10_000 + update,
                episodes=config.evaluation_episodes,
                max_episode_steps=config.max_episode_steps,
                device=device,
            )
        else:
            eval_mean = float("nan")
            eval_std = float("nan")
        history["update"].append(update)
        history["environment_steps"].append(update * config.num_envs * config.rollout_steps)
        history["rollout_return_mean"].append(float(np.mean(recent_completed)) if recent_completed else float("nan"))
        history["evaluation_return_mean"].append(eval_mean)
        history["evaluation_return_std"].append(eval_std)
        for name in (
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "ratio_outside_fraction",
            "active_clip_fraction",
            "gradient_norm",
            "epochs_completed",
        ):
            history[name].append(getattr(diagnostic, name))

    final_mean, final_std, final_returns = evaluate_policy(
        model,
        seed=config.seed + 50_000,
        episodes=max(12, config.evaluation_episodes),
        max_episode_steps=config.max_episode_steps,
        device=device,
    )
    return {
        "config": asdict(config),
        "history": history,
        "final_evaluation_mean": final_mean,
        "final_evaluation_std": final_std,
        "final_evaluation_returns": final_returns,
        "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
    }


def aggregate_training_runs(runs: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Aggregate aligned run histories into median and quartile curves."""
    if not runs:
        raise ValueError("at least one run is required")
    steps = np.asarray(runs[0]["history"]["environment_steps"], dtype=int)
    curves = []
    for run in runs:
        run_steps = np.asarray(run["history"]["environment_steps"], dtype=int)
        if not np.array_equal(run_steps, steps):
            raise ValueError("run histories must use aligned environment steps")
        curves.append(np.asarray(run["history"]["evaluation_return_mean"], dtype=float))
    matrix = np.stack(curves)
    valid = np.any(np.isfinite(matrix), axis=0)
    median = np.full(matrix.shape[1], np.nan, dtype=float)
    q25 = np.full(matrix.shape[1], np.nan, dtype=float)
    q75 = np.full(matrix.shape[1], np.nan, dtype=float)
    median[valid] = np.nanmedian(matrix[:, valid], axis=0)
    q25[valid] = np.nanquantile(matrix[:, valid], 0.25, axis=0)
    q75[valid] = np.nanquantile(matrix[:, valid], 0.75, axis=0)
    return {
        "environment_steps": steps,
        "median": median,
        "q25": q25,
        "q75": q75,
        "all_curves": matrix,
        "final_means": np.asarray([run["final_evaluation_mean"] for run in runs], dtype=float),
    }


def profile_config(profile: str, *, seed: int = 0) -> PPOConfig:
    """Return the documented notebook profiles."""
    if profile == "smoke":
        return PPOConfig(
            seed=seed,
            num_envs=8,
            rollout_steps=64,
            updates=16,
            update_epochs=4,
            minibatch_size=256,
            evaluation_episodes=5,
        )
    if profile == "full":
        return PPOConfig(
            seed=seed,
            num_envs=16,
            rollout_steps=128,
            updates=64,
            update_epochs=4,
            minibatch_size=512,
            evaluation_episodes=12,
        )
    raise ValueError("profile must be 'smoke' or 'full'")


__all__ = [
    "ActorCritic",
    "CartPoleBatch",
    "PPOConfig",
    "RolloutBatch",
    "UpdateDiagnostics",
    "aggregate_training_runs",
    "collect_rollout",
    "evaluate_policy",
    "ppo_update",
    "profile_config",
    "seed_everything",
    "train_ppo",
]
