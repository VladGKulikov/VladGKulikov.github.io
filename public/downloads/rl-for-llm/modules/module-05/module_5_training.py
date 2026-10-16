#!/usr/bin/env python3
"""Realistic CPU reference experiments for RL for LLM, Module 5.

The module deliberately avoids Gymnasium so that the notebooks remain runnable
without network access.  It contains:

* a small but non-trivial KeyDoorGrid environment with termination and
  truncation kept separate;
* a shortest-path expert, behavioral cloning, and iterative DAgger;
* online DQN and Double DQN with replay and target networks;
* construction of a fixed mixed-quality transition log;
* discrete offline baselines: BC, naive fitted Q iteration, a BCQ-like support
  constraint, CQL, IQL, and fitted Q evaluation (FQE);
* a compact causal Decision Transformer trained on fixed trajectories.

The code is educational rather than benchmark-optimized.  Each public function
has deterministic seeds and bounded workloads suitable for a CPU smoke run.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import copy
import math
import random
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

Array = np.ndarray

ACTION_NAMES = ("up", "right", "down", "left")
ACTION_DELTAS = np.asarray(((-1, 0), (0, 1), (1, 0), (0, -1)), dtype=np.int64)


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python, NumPy, and PyTorch and return a NumPy generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    return np.random.default_rng(seed)


@dataclass(frozen=True)
class GridState:
    row: int
    col: int
    has_key: bool


@dataclass
class Transition:
    state: Array
    action: int
    reward: float
    next_state: Array
    terminated: bool
    truncated: bool


@dataclass
class Episode:
    states: list[Array]
    actions: list[int]
    rewards: list[float]
    terminated: bool
    truncated: bool

    @property
    def return_(self) -> float:
        return float(np.sum(self.rewards))

    @property
    def length(self) -> int:
        return len(self.actions)


class KeyDoorGrid:
    """A deterministic grid with a key, a locked door, traps, and a time limit.

    The agent must collect the key before crossing the locked door and reaching
    the goal.  Walking into a wall leaves the state unchanged.  A trap is a true
    terminal; the external step limit produces truncation instead.
    """

    height = 7
    width = 9
    start_positions = (
        (6, 0), (6, 1), (5, 0), (5, 1), (4, 0), (6, 2), (5, 2),
        (4, 1), (3, 0), (6, 3), (5, 3), (4, 2),
    )
    key_position = (1, 2)
    door_position = (2, 6)
    goal_position = (0, 8)
    trap_positions = frozenset({(4, 4), (5, 7), (1, 7)})
    wall_positions = frozenset({
        (0, 4), (1, 4), (2, 4), (3, 2), (3, 3), (3, 4),
        (3, 6), (4, 6), (5, 6), (6, 6), (5, 4),
    })

    def __init__(self, *, max_steps: int = 64, seed: int = 0) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)
        self.state = GridState(*self.start_positions[0], False)
        self.steps = 0
        self._key_collected = False

    @property
    def observation_dim(self) -> int:
        # position one-hot for each key status plus compact geometric features
        return 2 * self.height * self.width + 15

    @property
    def action_dim(self) -> int:
        return 4

    @classmethod
    def is_inside(cls, row: int, col: int) -> bool:
        return 0 <= row < cls.height and 0 <= col < cls.width

    @classmethod
    def is_blocked(cls, row: int, col: int, has_key: bool) -> bool:
        if not cls.is_inside(row, col) or (row, col) in cls.wall_positions:
            return True
        if (row, col) == cls.door_position and not has_key:
            return True
        return False

    def reset(
        self,
        *,
        seed: int | None = None,
        start: tuple[int, int] | None = None,
    ) -> tuple[Array, dict[str, object]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if start is None:
            start = self.start_positions[int(self.rng.integers(len(self.start_positions)))]
        if self.is_blocked(*start, False) or start in self.trap_positions or start == self.goal_position:
            raise ValueError(f"invalid start position: {start}")
        self.state = GridState(start[0], start[1], False)
        self.steps = 0
        self._key_collected = False
        return self.encode(self.state), {"state": self.state}

    def encode(self, state: GridState | None = None) -> Array:
        s = self.state if state is None else state
        one_hot = np.zeros(2 * self.height * self.width, dtype=np.float32)
        offset = self.height * self.width if s.has_key else 0
        one_hot[offset + s.row * self.width + s.col] = 1.0

        def delta(target: tuple[int, int]) -> tuple[float, float]:
            return (
                (target[0] - s.row) / max(1, self.height - 1),
                (target[1] - s.col) / max(1, self.width - 1),
            )

        key_dr, key_dc = delta(self.key_position)
        door_dr, door_dc = delta(self.door_position)
        goal_dr, goal_dc = delta(self.goal_position)
        local_blocked = []
        for dr, dc in ACTION_DELTAS:
            local_blocked.append(float(self.is_blocked(s.row + int(dr), s.col + int(dc), s.has_key)))
        extras = np.asarray(
            [
                s.row / (self.height - 1),
                s.col / (self.width - 1),
                float(s.has_key),
                key_dr,
                key_dc,
                door_dr,
                door_dc,
                goal_dr,
                goal_dc,
                float((s.row, s.col) == self.key_position),
                float((s.row, s.col) == self.door_position),
                *local_blocked,
            ],
            dtype=np.float32,
        )
        # Keep the declared dimension stable if geometric features change.
        return np.concatenate([one_hot, extras], dtype=np.float32)

    def decode(self, observation: Array) -> GridState:
        x = np.asarray(observation, dtype=np.float32)
        base = 2 * self.height * self.width
        if x.shape != (self.observation_dim,):
            raise ValueError(f"observation must have shape {(self.observation_dim,)}")
        index = int(np.argmax(x[:base]))
        has_key = index >= self.height * self.width
        if has_key:
            index -= self.height * self.width
        return GridState(index // self.width, index % self.width, has_key)

    def step(self, action: int) -> tuple[Array, float, bool, bool, dict[str, object]]:
        if not 0 <= int(action) < self.action_dim:
            raise ValueError("invalid action")
        self.steps += 1
        s = self.state
        dr, dc = ACTION_DELTAS[int(action)]
        nr, nc = s.row + int(dr), s.col + int(dc)
        bumped = self.is_blocked(nr, nc, s.has_key)
        if bumped:
            nr, nc = s.row, s.col

        reward = -0.02
        if bumped:
            reward -= 0.03

        has_key = s.has_key
        collected = False
        if (nr, nc) == self.key_position and not has_key:
            has_key = True
            collected = True
            reward += 0.20

        terminated = False
        outcome = "running"
        if (nr, nc) in self.trap_positions:
            reward -= 1.0
            terminated = True
            outcome = "trap"
        elif (nr, nc) == self.goal_position:
            reward += 1.0 if has_key else -0.25
            terminated = True
            outcome = "goal" if has_key else "locked-goal"

        self.state = GridState(nr, nc, has_key)
        truncated = (self.steps >= self.max_steps) and not terminated
        if truncated:
            outcome = "time-limit"
        return self.encode(), float(reward), terminated, truncated, {
            "state": self.state,
            "bumped": bumped,
            "collected_key": collected,
            "outcome": outcome,
        }

    @classmethod
    def all_reachable_states(cls) -> list[GridState]:
        states: list[GridState] = []
        for has_key in (False, True):
            for r in range(cls.height):
                for c in range(cls.width):
                    if cls.is_blocked(r, c, has_key) or (r, c) in cls.trap_positions:
                        continue
                    if not has_key and (r, c) == cls.goal_position:
                        continue
                    states.append(GridState(r, c, has_key))
        return states


class ShortestPathExpert:
    """Deterministic shortest-path oracle for the KeyDoorGrid."""

    def __init__(self, env: KeyDoorGrid) -> None:
        self.env = env
        self._cache: dict[GridState, int] = {}

    def _successors(self, state: GridState) -> Iterator[tuple[int, GridState]]:
        for action, (dr, dc) in enumerate(ACTION_DELTAS):
            nr, nc = state.row + int(dr), state.col + int(dc)
            if self.env.is_blocked(nr, nc, state.has_key):
                nr, nc = state.row, state.col
            if (nr, nc) in self.env.trap_positions:
                continue
            has_key = state.has_key or (nr, nc) == self.env.key_position
            yield action, GridState(nr, nc, has_key)

    def action(self, observation: Array | GridState) -> int:
        state = observation if isinstance(observation, GridState) else self.env.decode(observation)
        if state in self._cache:
            return self._cache[state]
        queue: deque[GridState] = deque([state])
        parent: dict[GridState, tuple[GridState, int] | None] = {state: None}
        goal: GridState | None = None
        while queue:
            current = queue.popleft()
            if (current.row, current.col) == self.env.goal_position and current.has_key:
                goal = current
                break
            for action, nxt in self._successors(current):
                if nxt not in parent:
                    parent[nxt] = (current, action)
                    queue.append(nxt)
        if goal is None:
            raise RuntimeError(f"no expert path from {state}")
        cursor = goal
        first_action = 0
        while parent[cursor] is not None:
            prev, action = parent[cursor]  # type: ignore[misc]
            first_action = action
            if prev == state:
                break
            cursor = prev
        self._cache[state] = int(first_action)
        return int(first_action)


class MLPPolicy(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations).squeeze(-1)


@torch.no_grad()
def policy_action(model: nn.Module, observation: Array, *, stochastic: bool = False, rng: np.random.Generator | None = None) -> int:
    logits = model(torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)).squeeze(0)
    if stochastic:
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        local_rng = np.random.default_rng() if rng is None else rng
        return int(local_rng.choice(len(probs), p=probs))
    return int(torch.argmax(logits).item())


def rollout(
    env: KeyDoorGrid,
    action_fn: Callable[[Array], int],
    *,
    episodes: int,
    seed: int,
    starts: Sequence[tuple[int, int]] | None = None,
    collect: bool = True,
) -> tuple[list[Episode], list[Transition]]:
    episodes_out: list[Episode] = []
    transitions: list[Transition] = []
    start_list = tuple(starts) if starts is not None else env.start_positions
    for episode_index in range(episodes):
        start = start_list[episode_index % len(start_list)]
        obs, _ = env.reset(seed=seed + episode_index, start=start)
        states: list[Array] = []
        actions: list[int] = []
        rewards: list[float] = []
        terminated = truncated = False
        while not (terminated or truncated):
            action = int(action_fn(obs))
            next_obs, reward, terminated, truncated, _ = env.step(action)
            if collect:
                transitions.append(
                    Transition(obs.copy(), action, reward, next_obs.copy(), terminated, truncated)
                )
            states.append(obs.copy())
            actions.append(action)
            rewards.append(reward)
            obs = next_obs
        episodes_out.append(Episode(states, actions, rewards, terminated, truncated))
    return episodes_out, transitions


def transitions_to_arrays(transitions: Sequence[Transition]) -> dict[str, Array]:
    if not transitions:
        raise ValueError("transitions must be non-empty")
    return {
        "states": np.stack([t.state for t in transitions]).astype(np.float32),
        "actions": np.asarray([t.action for t in transitions], dtype=np.int64),
        "rewards": np.asarray([t.reward for t in transitions], dtype=np.float32),
        "next_states": np.stack([t.next_state for t in transitions]).astype(np.float32),
        "terminated": np.asarray([t.terminated for t in transitions], dtype=np.float32),
        "truncated": np.asarray([t.truncated for t in transitions], dtype=np.float32),
    }


def train_behavior_cloning(
    states: Array,
    actions: Array,
    *,
    seed: int,
    epochs: int = 60,
    batch_size: int = 128,
    learning_rate: float = 3e-3,
    hidden: int = 96,
) -> tuple[MLPPolicy, list[float]]:
    seed_everything(seed)
    x = torch.as_tensor(states, dtype=torch.float32)
    y = torch.as_tensor(actions, dtype=torch.long)
    model = MLPPolicy(x.shape[1], int(y.max().item()) + 1, hidden=hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=min(batch_size, len(x)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    history: list[float] = []
    for _ in range(epochs):
        losses: list[float] = []
        for xb, yb in loader:
            loss = F.cross_entropy(model(xb), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return model, history


@torch.no_grad()
def classification_accuracy(model: nn.Module, states: Array, actions: Array) -> float:
    logits = model(torch.as_tensor(states, dtype=torch.float32))
    pred = logits.argmax(dim=-1).cpu().numpy()
    return float(np.mean(pred == np.asarray(actions)))


def collect_expert_demonstrations(
    env: KeyDoorGrid,
    expert: ShortestPathExpert,
    *,
    episodes: int,
    seed: int,
    starts: Sequence[tuple[int, int]] | None = None,
) -> tuple[list[Episode], list[Transition]]:
    return rollout(env, expert.action, episodes=episodes, seed=seed, starts=starts, collect=True)


def dagger_experiment(
    *,
    seed: int = 17,
    expert_episodes: int = 18,
    rounds: int = 3,
    learner_episodes_per_round: int = 14,
    epochs: int = 50,
    evaluation_episodes: int = 36,
) -> dict[str, object]:
    """Run BC followed by iterative DAgger on learner-induced states."""
    rng = seed_everything(seed)
    env = KeyDoorGrid(seed=seed)
    expert = ShortestPathExpert(env)
    expert_starts = env.start_positions[: max(3, len(env.start_positions) // 2)]
    _, demo_transitions = collect_expert_demonstrations(
        env, expert, episodes=expert_episodes, seed=seed, starts=expert_starts
    )
    states = np.stack([t.state for t in demo_transitions])
    actions = np.asarray([t.action for t in demo_transitions], dtype=np.int64)
    model, _ = train_behavior_cloning(states, actions, seed=seed, epochs=epochs)

    eval_starts = env.start_positions
    records: list[dict[str, float]] = []

    def evaluate(label: str, round_index: int) -> None:
        eps, _ = rollout(
            env,
            lambda obs: policy_action(model, obs),
            episodes=evaluation_episodes,
            seed=seed + 10_000 + round_index * 100,
            starts=eval_starts,
            collect=False,
        )
        returns = np.asarray([e.return_ for e in eps], dtype=float)
        success = np.asarray([e.terminated and e.return_ > 0.0 for e in eps], dtype=float)
        records.append(
            {
                "round": float(round_index),
                "dataset_size": float(len(actions)),
                "mean_return": float(returns.mean()),
                "median_return": float(np.median(returns)),
                "success_rate": float(success.mean()),
                "expert_state_accuracy": classification_accuracy(model, states[: len(demo_transitions)], actions[: len(demo_transitions)]),
                "label": label,
            }
        )

    evaluate("BC", 0)
    for round_index in range(1, rounds + 1):
        learner_eps, _ = rollout(
            env,
            lambda obs: policy_action(model, obs, stochastic=True, rng=rng),
            episodes=learner_episodes_per_round,
            seed=seed + 1000 * round_index,
            starts=eval_starts,
            collect=False,
        )
        new_states: list[Array] = []
        new_actions: list[int] = []
        for episode in learner_eps:
            for obs in episode.states:
                new_states.append(obs)
                new_actions.append(expert.action(obs))
        states = np.concatenate([states, np.stack(new_states)], axis=0)
        actions = np.concatenate([actions, np.asarray(new_actions, dtype=np.int64)], axis=0)
        model, _ = train_behavior_cloning(
            states, actions, seed=seed + round_index, epochs=epochs
        )
        evaluate("DAgger", round_index)
    return {"records": records, "model": model, "dataset_size": len(actions)}


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int) -> None:
        self.capacity = int(capacity)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.position = 0

    def add(self, transition: Transition) -> None:
        i = self.position
        self.states[i] = transition.state
        self.actions[i] = transition.action
        self.rewards[i] = transition.reward
        self.next_states[i] = transition.next_state
        self.terminated[i] = float(transition.terminated)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[torch.Tensor, ...]:
        if self.size < batch_size:
            raise ValueError("not enough samples")
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[idx]),
            torch.as_tensor(self.actions[idx]),
            torch.as_tensor(self.rewards[idx]),
            torch.as_tensor(self.next_states[idx]),
            torch.as_tensor(self.terminated[idx]),
        )


def evaluate_q_policy(
    env: KeyDoorGrid,
    q_network: nn.Module,
    *,
    episodes: int = 36,
    seed: int = 0,
    action_mask_fn: Callable[[Array], Array] | None = None,
) -> dict[str, float]:
    @torch.no_grad()
    def act(obs: Array) -> int:
        q = q_network(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).cpu().numpy()
        if action_mask_fn is not None:
            mask = np.asarray(action_mask_fn(obs), dtype=bool)
            if mask.shape != q.shape or not np.any(mask):
                raise ValueError("action mask must allow at least one action")
            q = np.where(mask, q, -np.inf)
        return int(np.argmax(q))

    eps, _ = rollout(
        env, act, episodes=episodes, seed=seed, starts=env.start_positions, collect=False
    )
    returns = np.asarray([e.return_ for e in eps], dtype=float)
    success = np.asarray([e.terminated and e.return_ > 0.0 for e in eps], dtype=float)
    return {
        "mean_return": float(returns.mean()),
        "median_return": float(np.median(returns)),
        "success_rate": float(success.mean()),
        "mean_length": float(np.mean([e.length for e in eps])),
    }


def train_online_dqn(
    *,
    seed: int = 23,
    double: bool = False,
    episodes: int = 220,
    batch_size: int = 64,
    gamma: float = 0.98,
    learning_rate: float = 1e-3,
    replay_capacity: int = 20_000,
    warmup: int = 256,
    target_sync: int = 160,
    update_every: int = 4,
    expert_prefill_episodes: int = 10,
    evaluation_episodes: int = 48,
) -> dict[str, object]:
    """Train DQN or Double DQN with replay and a target network.

    The small KeyDoor task has sparse terminal reward.  The replay buffer is
    therefore warm-started with a few noisy expert-guided trajectories.  This
    is an explicit data-engineering choice, not part of vanilla DQN; setting
    ``expert_prefill_episodes=0`` recovers a purely exploratory start.
    """
    if update_every < 1:
        raise ValueError("update_every must be positive")
    rng = seed_everything(seed)
    env = KeyDoorGrid(max_steps=56, seed=seed)
    q = QNetwork(env.observation_dim, env.action_dim, hidden=72)
    target = copy.deepcopy(q)
    target.eval()
    optimizer = torch.optim.Adam(q.parameters(), lr=learning_rate)
    buffer = ReplayBuffer(replay_capacity, env.observation_dim)
    episode_returns: list[float] = []
    episode_success: list[float] = []
    losses: list[float] = []
    global_step = 0

    if expert_prefill_episodes:
        expert = ShortestPathExpert(env)
        for prefill_index in range(expert_prefill_episodes):
            obs, _ = env.reset(seed=seed + 80_000 + prefill_index)
            terminated = truncated = False
            while not (terminated or truncated):
                # Keep some off-expert actions so the replay contains recoveries.
                action = behavior_action(expert, obs, rng=rng, epsilon=0.22)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                buffer.add(Transition(obs.copy(), action, reward, next_obs.copy(), terminated, truncated))
                obs = next_obs

    for episode_index in range(episodes):
        epsilon = 0.04 + 0.96 * math.exp(-episode_index / max(1.0, episodes * 0.32))
        obs, _ = env.reset(seed=seed + episode_index)
        terminated = truncated = False
        total = 0.0
        while not (terminated or truncated):
            if rng.random() < epsilon:
                action = int(rng.integers(env.action_dim))
            else:
                action = policy_action(q, obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            buffer.add(Transition(obs.copy(), action, reward, next_obs.copy(), terminated, truncated))
            obs = next_obs
            total += reward
            global_step += 1

            if (
                buffer.size >= max(warmup, batch_size)
                and global_step % update_every == 0
            ):
                states, actions, rewards, next_states, terminals = buffer.sample(batch_size, rng)
                with torch.no_grad():
                    if double:
                        next_actions = q(next_states).argmax(dim=1)
                        next_values = target(next_states).gather(1, next_actions[:, None]).squeeze(1)
                    else:
                        next_values = target(next_states).max(dim=1).values
                    y = rewards + gamma * (1.0 - terminals) * next_values
                pred = q(states).gather(1, actions[:, None]).squeeze(1)
                loss = F.smooth_l1_loss(pred, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                optimizer.step()
                losses.append(float(loss.detach()))
                if global_step % target_sync < update_every:
                    target.load_state_dict(q.state_dict())
        episode_returns.append(float(total))
        episode_success.append(float(terminated and total > 0.0))

    target.load_state_dict(q.state_dict())
    evaluation = evaluate_q_policy(
        env, q, episodes=evaluation_episodes, seed=seed + 50_000
    )
    return {
        "model": q,
        "returns": np.asarray(episode_returns, dtype=np.float32),
        "success": np.asarray(episode_success, dtype=np.float32),
        "losses": np.asarray(losses, dtype=np.float32),
        "evaluation": evaluation,
        "double": double,
        "prefill_episodes": int(expert_prefill_episodes),
        "update_every": int(update_every),
    }


def behavior_action(
    expert: ShortestPathExpert,
    observation: Array,
    *,
    rng: np.random.Generator,
    epsilon: float,
    forbidden_random_actions: set[int] | None = None,
) -> int:
    if rng.random() >= epsilon:
        return expert.action(observation)
    candidates = [a for a in range(4) if forbidden_random_actions is None or a not in forbidden_random_actions]
    return int(rng.choice(candidates))


def collect_offline_dataset(
    *,
    seed: int = 31,
    episodes: int = 360,
    profile: str = "mixed",
) -> tuple[list[Episode], list[Transition]]:
    """Create a fixed log with explicit behavior-quality and support profiles."""
    if profile not in {"narrow", "mixed", "broad"}:
        raise ValueError("profile must be 'narrow', 'mixed', or 'broad'")
    rng = seed_everything(seed)
    env = KeyDoorGrid(seed=seed)
    expert = ShortestPathExpert(env)
    episodes_out: list[Episode] = []
    transitions: list[Transition] = []

    for episode_index in range(episodes):
        if profile == "narrow":
            epsilon = 0.04
            starts = env.start_positions[:6]
        elif profile == "mixed":
            epsilon = (0.05, 0.18, 0.42)[episode_index % 3]
            starts = env.start_positions
        else:
            epsilon = (0.08, 0.30, 0.70)[episode_index % 3]
            starts = env.start_positions
        start = starts[episode_index % len(starts)]
        obs, _ = env.reset(seed=seed + episode_index, start=start)
        states: list[Array] = []
        actions: list[int] = []
        rewards: list[float] = []
        terminated = truncated = False
        while not (terminated or truncated):
            action = behavior_action(expert, obs, rng=rng, epsilon=epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            transitions.append(Transition(obs.copy(), action, reward, next_obs.copy(), terminated, truncated))
            states.append(obs.copy())
            actions.append(action)
            rewards.append(reward)
            obs = next_obs
        episodes_out.append(Episode(states, actions, rewards, terminated, truncated))
    return episodes_out, transitions


def _make_loader(arrays: dict[str, Array], batch_size: int, seed: int) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(arrays["states"], dtype=torch.float32),
        torch.as_tensor(arrays["actions"], dtype=torch.long),
        torch.as_tensor(arrays["rewards"], dtype=torch.float32),
        torch.as_tensor(arrays["next_states"], dtype=torch.float32),
        torch.as_tensor(arrays["terminated"], dtype=torch.float32),
    )
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )


def train_offline_fqi(
    arrays: dict[str, Array],
    *,
    seed: int,
    action_dim: int = 4,
    epochs: int = 80,
    batch_size: int = 256,
    gamma: float = 0.98,
    learning_rate: float = 1e-3,
    cql_alpha: float = 0.0,
) -> tuple[QNetwork, list[float]]:
    """Fitted Q iteration on a fixed log; CQL is enabled by cql_alpha > 0."""
    seed_everything(seed)
    state_dim = arrays["states"].shape[1]
    q = QNetwork(state_dim, action_dim)
    target = copy.deepcopy(q)
    optimizer = torch.optim.Adam(q.parameters(), lr=learning_rate)
    loader = _make_loader(arrays, batch_size, seed)
    history: list[float] = []
    for epoch in range(epochs):
        epoch_losses: list[float] = []
        for states, actions, rewards, next_states, terminals in loader:
            with torch.no_grad():
                y = rewards + gamma * (1.0 - terminals) * target(next_states).max(dim=1).values
            all_q = q(states)
            chosen = all_q.gather(1, actions[:, None]).squeeze(1)
            bellman = F.smooth_l1_loss(chosen, y)
            conservative = torch.logsumexp(all_q, dim=1).mean() - chosen.mean()
            loss = bellman + cql_alpha * conservative
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        if (epoch + 1) % 4 == 0:
            target.load_state_dict(q.state_dict())
        history.append(float(np.mean(epoch_losses)))
    return q, history


def train_behavior_model(
    arrays: dict[str, Array],
    *,
    seed: int,
    epochs: int = 60,
) -> tuple[MLPPolicy, list[float]]:
    return train_behavior_cloning(
        arrays["states"], arrays["actions"], seed=seed, epochs=epochs
    )


def make_bcq_mask_fn(
    behavior_model: nn.Module,
    *,
    threshold_ratio: float = 0.30,
) -> Callable[[Array], Array]:
    if not 0.0 < threshold_ratio <= 1.0:
        raise ValueError("threshold_ratio must lie in (0, 1]")

    @torch.no_grad()
    def mask(observation: Array) -> Array:
        probs = torch.softmax(
            behavior_model(torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)), dim=-1
        ).squeeze(0).cpu().numpy()
        allowed = probs >= threshold_ratio * float(probs.max())
        if not np.any(allowed):
            allowed[int(np.argmax(probs))] = True
        return allowed

    return mask


def train_iql(
    arrays: dict[str, Array],
    *,
    seed: int,
    action_dim: int = 4,
    epochs: int = 90,
    batch_size: int = 256,
    gamma: float = 0.98,
    expectile: float = 0.75,
    inverse_temperature: float = 3.0,
    max_weight: float = 50.0,
    learning_rate: float = 1e-3,
) -> dict[str, object]:
    """Discrete IQL with Q, V, and advantage-weighted policy extraction."""
    seed_everything(seed)
    state_dim = arrays["states"].shape[1]
    q = QNetwork(state_dim, action_dim)
    q_target = copy.deepcopy(q)
    v = ValueNetwork(state_dim)
    policy = MLPPolicy(state_dim, action_dim)
    q_opt = torch.optim.Adam(q.parameters(), lr=learning_rate)
    v_opt = torch.optim.Adam(v.parameters(), lr=learning_rate)
    p_opt = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    loader = _make_loader(arrays, batch_size, seed)
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        q_losses: list[float] = []
        v_losses: list[float] = []
        p_losses: list[float] = []
        for states, actions, rewards, next_states, terminals in loader:
            with torch.no_grad():
                q_data_target = q_target(states).gather(1, actions[:, None]).squeeze(1)
            value = v(states)
            residual = q_data_target - value
            weight = torch.where(residual >= 0, expectile, 1.0 - expectile)
            value_loss = (weight * residual.square()).mean()
            v_opt.zero_grad(set_to_none=True)
            value_loss.backward()
            v_opt.step()

            with torch.no_grad():
                y = rewards + gamma * (1.0 - terminals) * v(next_states)
            q_data = q(states).gather(1, actions[:, None]).squeeze(1)
            q_loss = F.smooth_l1_loss(q_data, y)
            q_opt.zero_grad(set_to_none=True)
            q_loss.backward()
            q_opt.step()

            with torch.no_grad():
                advantage = q_target(states).gather(1, actions[:, None]).squeeze(1) - v(states)
                aw = torch.exp(inverse_temperature * advantage).clamp(max=max_weight)
            log_probs = F.log_softmax(policy(states), dim=-1)
            policy_loss = -(aw * log_probs.gather(1, actions[:, None]).squeeze(1)).mean()
            p_opt.zero_grad(set_to_none=True)
            policy_loss.backward()
            p_opt.step()

            q_losses.append(float(q_loss.detach()))
            v_losses.append(float(value_loss.detach()))
            p_losses.append(float(policy_loss.detach()))
        if (epoch + 1) % 2 == 0:
            with torch.no_grad():
                tau = 0.02
                for target_param, source_param in zip(q_target.parameters(), q.parameters()):
                    target_param.mul_(1.0 - tau).add_(tau * source_param)
        history.append(
            {
                "q": float(np.mean(q_losses)),
                "v": float(np.mean(v_losses)),
                "policy": float(np.mean(p_losses)),
            }
        )
    return {"q": q, "v": v, "policy": policy, "history": history}


def evaluate_policy_model(
    env: KeyDoorGrid,
    model: nn.Module,
    *,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    eps, _ = rollout(
        env,
        lambda obs: policy_action(model, obs),
        episodes=episodes,
        seed=seed,
        starts=env.start_positions,
        collect=False,
    )
    returns = np.asarray([e.return_ for e in eps], dtype=float)
    return {
        "mean_return": float(returns.mean()),
        "median_return": float(np.median(returns)),
        "success_rate": float(np.mean([e.terminated and e.return_ > 0.0 for e in eps])),
        "mean_length": float(np.mean([e.length for e in eps])),
    }


def fitted_q_evaluation(
    arrays: dict[str, Array],
    policy: nn.Module,
    *,
    seed: int,
    action_dim: int = 4,
    epochs: int = 80,
    gamma: float = 0.98,
) -> QNetwork:
    """FQE for a deterministic greedy policy learned from the same observation space."""
    seed_everything(seed)
    state_dim = arrays["states"].shape[1]
    q = QNetwork(state_dim, action_dim)
    target = copy.deepcopy(q)
    optimizer = torch.optim.Adam(q.parameters(), lr=1e-3)
    loader = _make_loader(arrays, 256, seed)
    policy.eval()
    for epoch in range(epochs):
        for states, actions, rewards, next_states, terminals in loader:
            with torch.no_grad():
                next_actions = policy(next_states).argmax(dim=1)
                continuation = target(next_states).gather(1, next_actions[:, None]).squeeze(1)
                y = rewards + gamma * (1.0 - terminals) * continuation
            pred = q(states).gather(1, actions[:, None]).squeeze(1)
            loss = F.smooth_l1_loss(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 4 == 0:
            target.load_state_dict(q.state_dict())
    return q


@torch.no_grad()
def estimate_initial_value(
    fqe: QNetwork,
    policy: nn.Module,
    env: KeyDoorGrid,
    *,
    starts: Sequence[tuple[int, int]] | None = None,
) -> float:
    start_list = env.start_positions if starts is None else starts
    observations = []
    for start in start_list:
        obs, _ = env.reset(start=start)
        observations.append(obs)
    x = torch.as_tensor(np.stack(observations), dtype=torch.float32)
    actions = policy(x).argmax(dim=1)
    values = fqe(x).gather(1, actions[:, None]).squeeze(1)
    return float(values.mean())


class TrajectoryDataset(torch.utils.data.Dataset):
    """Fixed-length windows for Decision Transformer training."""

    def __init__(
        self,
        episodes: Sequence[Episode],
        *,
        context_steps: int,
        gamma: float = 1.0,
        sample_stride: int = 2,
    ) -> None:
        if sample_stride < 1:
            raise ValueError("sample_stride must be positive")
        self.samples: list[dict[str, Array]] = []
        self.context_steps = int(context_steps)
        for episode in episodes:
            rewards = np.asarray(episode.rewards, dtype=np.float32)
            rtg = np.empty_like(rewards)
            running = 0.0
            for t in range(len(rewards) - 1, -1, -1):
                running = float(rewards[t]) + gamma * running
                rtg[t] = running
            states = np.stack(episode.states).astype(np.float32)
            actions = np.asarray(episode.actions, dtype=np.int64)
            ends = list(range(1, len(actions) + 1, sample_stride))
            if ends[-1] != len(actions):
                ends.append(len(actions))
            for end in ends:
                start = max(0, end - self.context_steps)
                self.samples.append(
                    {
                        "states": states[start:end],
                        "actions": actions[start:end],
                        "rtg": rtg[start:end],
                        "timesteps": np.arange(start, end, dtype=np.int64),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Array]:
        return self.samples[index]


def collate_trajectory_batch(batch: Sequence[dict[str, Array]], context_steps: int) -> dict[str, torch.Tensor]:
    batch_size = len(batch)
    state_dim = batch[0]["states"].shape[1]
    states = np.zeros((batch_size, context_steps, state_dim), dtype=np.float32)
    actions = np.zeros((batch_size, context_steps), dtype=np.int64)
    rtg = np.zeros((batch_size, context_steps, 1), dtype=np.float32)
    timesteps = np.zeros((batch_size, context_steps), dtype=np.int64)
    mask = np.zeros((batch_size, context_steps), dtype=np.float32)
    for i, sample in enumerate(batch):
        length = min(context_steps, len(sample["actions"]))
        states[i, -length:] = sample["states"][-length:]
        actions[i, -length:] = sample["actions"][-length:]
        rtg[i, -length:, 0] = sample["rtg"][-length:]
        timesteps[i, -length:] = sample["timesteps"][-length:]
        mask[i, -length:] = 1.0
    return {
        "states": torch.as_tensor(states),
        "actions": torch.as_tensor(actions),
        "rtg": torch.as_tensor(rtg),
        "timesteps": torch.as_tensor(timesteps),
        "mask": torch.as_tensor(mask),
    }


class DecisionTransformer(nn.Module):
    """A compact causal Transformer over (return, state, action) triplets."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        hidden: int = 48,
        layers: int = 1,
        heads: int = 4,
        max_timestep: int = 128,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.hidden = hidden
        self.state_embed = nn.Linear(state_dim, hidden)
        self.return_embed = nn.Linear(1, hidden)
        self.action_embed = nn.Embedding(action_dim, hidden)
        self.timestep_embed = nn.Embedding(max_timestep, hidden)
        self.type_embed = nn.Embedding(3, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=2 * hidden,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden)
        self.action_head = nn.Linear(hidden, action_dim)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps, _ = states.shape
        time = self.timestep_embed(timesteps.clamp(max=self.timestep_embed.num_embeddings - 1))
        r = self.return_embed(returns_to_go) + time + self.type_embed(torch.zeros_like(timesteps))
        s = self.state_embed(states) + time + self.type_embed(torch.ones_like(timesteps))
        a = self.action_embed(actions) + time + self.type_embed(torch.full_like(timesteps, 2))
        tokens = torch.stack((r, s, a), dim=2).reshape(batch, 3 * steps, self.hidden)
        token_mask = attention_mask.unsqueeze(-1).expand(-1, -1, 3).reshape(batch, 3 * steps)
        length = tokens.shape[1]
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=tokens.device), diagonal=1)
        encoded = self.transformer(
            tokens,
            mask=causal,
            src_key_padding_mask=~token_mask.bool(),
        )
        encoded = self.norm(encoded)
        state_positions = encoded[:, 1::3]
        return self.action_head(state_positions)


def train_decision_transformer(
    episodes: Sequence[Episode],
    *,
    seed: int,
    state_dim: int,
    action_dim: int,
    context_steps: int = 20,
    epochs: int = 16,
    batch_size: int = 96,
    learning_rate: float = 3e-4,
) -> tuple[DecisionTransformer, list[float]]:
    seed_everything(seed)
    dataset = TrajectoryDataset(episodes, context_steps=context_steps, sample_stride=2)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=lambda batch: collate_trajectory_batch(batch, context_steps),
    )
    model = DecisionTransformer(state_dim, action_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history: list[float] = []
    for _ in range(epochs):
        losses: list[float] = []
        for batch in loader:
            logits = model(
                batch["states"], batch["actions"], batch["rtg"], batch["timesteps"], batch["mask"]
            )
            flat_mask = batch["mask"].bool()
            loss = F.cross_entropy(logits[flat_mask], batch["actions"][flat_mask])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return model, history


@torch.no_grad()
def decision_transformer_action(
    model: DecisionTransformer,
    states: list[Array],
    actions: list[int],
    returns_to_go: list[float],
    *,
    context_steps: int,
) -> int:
    if len(states) != len(returns_to_go) or len(actions) not in {len(states) - 1, len(states)}:
        raise ValueError("history lengths are inconsistent")
    used_states = states[-context_steps:]
    used_rtg = returns_to_go[-context_steps:]
    used_actions = actions[-context_steps:]
    # The current action token is a placeholder and comes after the current state,
    # so the causal mask prevents leakage into the prediction at that state.
    padded_actions = list(used_actions)
    if len(padded_actions) < len(used_states):
        padded_actions.append(0)
    x_states = torch.as_tensor(np.stack(used_states), dtype=torch.float32).unsqueeze(0)
    x_actions = torch.as_tensor(padded_actions, dtype=torch.long).unsqueeze(0)
    x_rtg = torch.as_tensor(np.asarray(used_rtg, dtype=np.float32)[:, None]).unsqueeze(0)
    x_time = torch.arange(len(used_states), dtype=torch.long).unsqueeze(0)
    x_mask = torch.ones((1, len(used_states)), dtype=torch.float32)
    logits = model(x_states, x_actions, x_rtg, x_time, x_mask)
    return int(logits[0, -1].argmax().item())


def evaluate_decision_transformer(
    env: KeyDoorGrid,
    model: DecisionTransformer,
    *,
    target_return: float,
    episodes: int,
    seed: int,
    context_steps: int = 20,
) -> dict[str, float]:
    returns: list[float] = []
    successes: list[float] = []
    for episode_index in range(episodes):
        start = env.start_positions[episode_index % len(env.start_positions)]
        obs, _ = env.reset(seed=seed + episode_index, start=start)
        states: list[Array] = []
        actions: list[int] = []
        desired: list[float] = []
        running_target = float(target_return)
        total = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            states.append(obs.copy())
            desired.append(running_target)
            action = decision_transformer_action(
                model, states, actions, desired, context_steps=context_steps
            )
            obs, reward, terminated, truncated, _ = env.step(action)
            actions.append(action)
            running_target -= reward
            total += reward
        returns.append(total)
        successes.append(float(terminated and total > 0.0))
    return {
        "target_return": float(target_return),
        "mean_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "success_rate": float(np.mean(successes)),
    }


def offline_experiment(
    *,
    seed: int = 41,
    dataset_episodes: int = 180,
    train_epochs: int = 40,
    evaluation_episodes: int = 36,
) -> dict[str, object]:
    """Train and compare fixed-data methods on one controlled log."""
    episodes, transitions = collect_offline_dataset(
        seed=seed, episodes=dataset_episodes, profile="mixed"
    )
    arrays = transitions_to_arrays(transitions)
    env = KeyDoorGrid(seed=seed)

    bc, _ = train_behavior_model(arrays, seed=seed + 1, epochs=train_epochs)
    naive_q, _ = train_offline_fqi(arrays, seed=seed + 2, epochs=train_epochs, cql_alpha=0.0)
    cql_q, _ = train_offline_fqi(arrays, seed=seed + 3, epochs=train_epochs, cql_alpha=0.15)
    behavior, _ = train_behavior_model(arrays, seed=seed + 4, epochs=train_epochs)
    bcq_mask = make_bcq_mask_fn(behavior, threshold_ratio=0.35)
    iql = train_iql(arrays, seed=seed + 5, epochs=train_epochs)

    results: dict[str, dict[str, float]] = {
        "Behavior cloning": evaluate_policy_model(env, bc, episodes=evaluation_episodes, seed=seed + 10_000),
        "Offline DQN": evaluate_q_policy(env, naive_q, episodes=evaluation_episodes, seed=seed + 20_000),
        "BCQ-like": evaluate_q_policy(env, naive_q, episodes=evaluation_episodes, seed=seed + 30_000, action_mask_fn=bcq_mask),
        "CQL": evaluate_q_policy(env, cql_q, episodes=evaluation_episodes, seed=seed + 40_000),
        "IQL": evaluate_policy_model(env, iql["policy"], episodes=evaluation_episodes, seed=seed + 50_000),
    }

    fqe_estimates: dict[str, float] = {}
    for index, (name, policy) in enumerate(
        (("Behavior cloning", bc), ("IQL", iql["policy"])), start=1
    ):
        fqe = fitted_q_evaluation(arrays, policy, seed=seed + 100 + index, epochs=train_epochs)
        fqe_estimates[name] = estimate_initial_value(fqe, policy, env)

    action_counts = np.bincount(arrays["actions"], minlength=env.action_dim)
    returns = np.asarray([episode.return_ for episode in episodes], dtype=np.float32)
    return {
        "episodes": episodes,
        "arrays": arrays,
        "results": results,
        "fqe": fqe_estimates,
        "action_counts": action_counts,
        "dataset_return_mean": float(returns.mean()),
        "dataset_return_median": float(np.median(returns)),
        "models": {"bc": bc, "naive_q": naive_q, "cql_q": cql_q, "behavior": behavior, "iql": iql},
    }


__all__ = [
    "ACTION_NAMES",
    "GridState",
    "Transition",
    "Episode",
    "KeyDoorGrid",
    "ShortestPathExpert",
    "MLPPolicy",
    "QNetwork",
    "ValueNetwork",
    "DecisionTransformer",
    "seed_everything",
    "rollout",
    "transitions_to_arrays",
    "collect_expert_demonstrations",
    "train_behavior_cloning",
    "dagger_experiment",
    "train_online_dqn",
    "collect_offline_dataset",
    "train_offline_fqi",
    "train_behavior_model",
    "make_bcq_mask_fn",
    "train_iql",
    "evaluate_q_policy",
    "evaluate_policy_model",
    "fitted_q_evaluation",
    "estimate_initial_value",
    "train_decision_transformer",
    "evaluate_decision_transformer",
    "offline_experiment",
]
