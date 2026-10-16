#!/usr/bin/env python3
"""A compact but complete recurrent agent-training experiment for Module 11.

The environment is deliberately small enough for CPU execution while keeping
contracts that matter in real agentic RL: partial observations, structured
macro-actions, valid-action masks, true task termination, collector truncation,
behavior log-probabilities, policy versions, and outcome/potential rewards.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
import random

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

ACTIONS = ("inspect", "pickup_red", "pickup_blue", "open_red", "open_blue", "finish")
OBSERVATIONS = (
    "start", "clue_red", "clue_blue", "carrying_red", "carrying_blue",
    "opened_red", "opened_blue", "invalid", "success", "failure",
    "task_budget_exhausted", "collector_truncation",
)
OBS_TO_ID = {name: index for index, name in enumerate(OBSERVATIONS)}


@dataclass(frozen=True)
class Transition:
    observation: str
    action: str
    next_observation: str
    base_reward: float
    shaped_reward: float
    behavior_logprob: float
    value: float
    valid_action: bool
    terminated: bool
    truncated: bool
    stop_reason: str | None


class KeyDoorAgentEnv:
    """Partially observed key-door task with a task horizon and optional collector cut."""

    def __init__(self, *, task_horizon: int = 6, collector_limit: int | None = None, seed: int = 0):
        if task_horizon < 1:
            raise ValueError("task_horizon must be positive")
        if collector_limit is not None and collector_limit < 1:
            raise ValueError("collector_limit must be positive")
        self.task_horizon = int(task_horizon)
        self.collector_limit = collector_limit
        self.rng = random.Random(seed)
        self.reset()

    def reset(self, *, target: str | None = None) -> tuple[str, dict[str, Any]]:
        if target is not None and target not in {"red", "blue"}:
            raise ValueError("target must be red or blue")
        self.target = target or self.rng.choice(("red", "blue"))
        self.inspected = False
        self.carrying: str | None = None
        self.opened: str | None = None
        self.steps = 0
        self.done = False
        return "start", {"target": self.target, "task_horizon": self.task_horizon}

    def potential(self) -> float:
        if self.done:
            return 0.0
        if self.opened == self.target:
            return 0.85
        if self.carrying == self.target:
            return 0.55
        if self.inspected:
            return 0.25
        return 0.0

    def valid_action_mask(self) -> np.ndarray:
        """Mask syntactically feasible phases without revealing the hidden color."""
        mask = np.zeros(len(ACTIONS), dtype=bool)
        if self.opened is not None:
            mask[ACTIONS.index("finish")] = True
        elif self.carrying is not None:
            mask[ACTIONS.index("open_red")] = True
            mask[ACTIONS.index("open_blue")] = True
        elif self.inspected:
            mask[ACTIONS.index("pickup_red")] = True
            mask[ACTIONS.index("pickup_blue")] = True
        else:
            mask[ACTIONS.index("inspect")] = True
        return mask

    def step(self, action: str, *, gamma: float = 0.99) -> tuple[str, float, bool, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("reset is required after a stopped episode")
        if action not in ACTIONS:
            raise ValueError(f"unknown action: {action}")
        phi_before = self.potential()
        self.steps += 1
        valid = True
        terminated = False
        truncated = False
        success = False
        stop_reason: str | None = None

        if action == "inspect":
            self.inspected = True
            observation = f"clue_{self.target}"
        elif action.startswith("pickup_"):
            color = action.removeprefix("pickup_")
            if not self.inspected or color != self.target:
                valid = False
                observation = "invalid"
            else:
                self.carrying = color
                observation = f"carrying_{color}"
        elif action.startswith("open_"):
            color = action.removeprefix("open_")
            if self.carrying != color:
                valid = False
                observation = "invalid"
            else:
                self.opened = color
                observation = f"opened_{color}"
        elif action == "finish":
            terminated = True
            success = self.opened == self.target
            observation = "success" if success else "failure"
            stop_reason = "success" if success else "premature_finish"
        else:  # pragma: no cover - ACTIONS guards this branch
            raise AssertionError(action)

        if not terminated and self.steps >= self.task_horizon:
            terminated = True
            observation = "task_budget_exhausted"
            stop_reason = "task_budget_exhausted"
        elif not terminated and self.collector_limit is not None and self.steps >= self.collector_limit:
            truncated = True
            observation = "collector_truncation"
            stop_reason = "collector_limit"

        self.done = terminated or truncated
        base_reward = 1.0 if success else 0.0
        phi_after = 0.0 if self.done else self.potential()
        shaped_reward = base_reward + float(gamma) * phi_after - phi_before
        info = {
            "success": success,
            "valid_action": valid,
            "target": self.target,
            "steps": self.steps,
            "potential_before": phi_before,
            "potential_after": phi_after,
            "shaped_reward": shaped_reward,
            "stop_reason": stop_reason,
        }
        return observation, base_reward, terminated, truncated, info


class RecurrentActorCritic(nn.Module):
    """Tiny recurrent policy whose hidden state must retain the one-turn clue."""

    def __init__(self, *, embedding_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.observation_embedding = nn.Embedding(len(OBSERVATIONS), embedding_dim)
        self.cell = nn.GRUCell(embedding_dim + len(ACTIONS), hidden_dim)
        self.actor = nn.Linear(hidden_dim, len(ACTIONS))
        self.critic = nn.Linear(hidden_dim, 1)

    def initial_state(self, *, device: torch.device | None = None) -> torch.Tensor:
        return torch.zeros(1, self.cell.hidden_size, device=device)

    def forward_step(
        self,
        observation_id: int,
        previous_action: int | None,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = hidden.device
        observation = self.observation_embedding(torch.tensor([observation_id], device=device))
        action_one_hot = torch.zeros(1, len(ACTIONS), device=device)
        if previous_action is not None:
            action_one_hot[0, previous_action] = 1.0
        next_hidden = self.cell(torch.cat([observation, action_one_hot], dim=-1), hidden)
        return self.actor(next_hidden).squeeze(0), self.critic(next_hidden).squeeze(), next_hidden


def masked_logits(logits: torch.Tensor, valid_mask: np.ndarray) -> torch.Tensor:
    mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=logits.device)
    if mask.shape != logits.shape or not bool(mask.any()):
        raise ValueError("valid action mask must match logits and select at least one action")
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


def collect_episode(
    model: RecurrentActorCritic,
    env: KeyDoorAgentEnv,
    *,
    reward_mode: Literal["sparse", "potential"] = "sparse",
    greedy: bool = False,
    target: str | None = None,
    policy_version: str = "policy-0",
    gamma: float = 0.99,
) -> tuple[list[Transition], list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], dict[str, Any]]:
    observation, reset_info = env.reset(target=target)
    hidden = model.initial_state(device=next(model.parameters()).device)
    previous_action: int | None = None
    transitions: list[Transition] = []
    tensors: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    while True:
        logits, value, hidden = model.forward_step(OBS_TO_ID[observation], previous_action, hidden)
        behavior_logits = masked_logits(logits, env.valid_action_mask())
        distribution = Categorical(logits=behavior_logits)
        action_id = int(torch.argmax(behavior_logits).item()) if greedy else int(distribution.sample().item())
        logprob = distribution.log_prob(torch.tensor(action_id, device=behavior_logits.device))
        entropy = distribution.entropy()
        next_observation, base_reward, terminated, truncated, info = env.step(ACTIONS[action_id], gamma=gamma)
        selected_reward = base_reward if reward_mode == "sparse" else float(info["shaped_reward"])
        transitions.append(
            Transition(
                observation=observation,
                action=ACTIONS[action_id],
                next_observation=next_observation,
                base_reward=float(base_reward),
                shaped_reward=float(info["shaped_reward"]),
                behavior_logprob=float(logprob.detach()),
                value=float(value.detach()),
                valid_action=bool(info["valid_action"]),
                terminated=bool(terminated),
                truncated=bool(truncated),
                stop_reason=info["stop_reason"],
            )
        )
        tensors.append((logprob, value, entropy))
        observation = next_observation
        previous_action = action_id
        if terminated or truncated:
            break

    report = {
        "target": reset_info["target"],
        "success": bool(transitions[-1].base_reward > 0.0),
        "terminated": transitions[-1].terminated,
        "truncated": transitions[-1].truncated,
        "stop_reason": transitions[-1].stop_reason,
        "policy_version": policy_version,
        "selected_return": float(sum((gamma**i) * (t.base_reward if reward_mode == "sparse" else t.shaped_reward) for i, t in enumerate(transitions))),
    }
    return transitions, tensors, report


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    values: list[float] = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + float(gamma) * running
        values.append(running)
    return torch.tensor(list(reversed(values)), dtype=torch.float32)


def evaluate_policy(model: RecurrentActorCritic, *, episodes: int = 100, seed: int = 0) -> dict[str, float]:
    if episodes < 2:
        raise ValueError("episodes must be at least two")
    successes = 0
    lengths: list[int] = []
    invalid = 0
    with torch.no_grad():
        for index in range(episodes):
            env = KeyDoorAgentEnv(seed=seed + index)
            target = "red" if index % 2 == 0 else "blue"
            transitions, _, report = collect_episode(model, env, greedy=True, target=target)
            successes += int(report["success"])
            lengths.append(len(transitions))
            invalid += sum(not transition.valid_action for transition in transitions)
    return {
        "success_rate": successes / episodes,
        "mean_turns": float(np.mean(lengths)),
        "invalid_action_rate": invalid / max(1, sum(lengths)),
    }


def train_actor_critic(
    *,
    seed: int = 1,
    reward_mode: Literal["sparse", "potential"] = "sparse",
    updates: int = 30,
    episodes_per_update: int = 32,
    learning_rate: float = 3e-3,
    gamma: float = 0.99,
    entropy_coefficient: float = 0.02,
    value_coefficient: float = 0.4,
    evaluation_interval: int = 5,
    evaluation_episodes: int = 100,
) -> tuple[RecurrentActorCritic, list[dict[str, float]]]:
    """Train an on-policy recurrent actor-critic and return its learning curve."""
    if updates < 1 or episodes_per_update < 1:
        raise ValueError("updates and episodes_per_update must be positive")
    if evaluation_interval < 1 or evaluation_episodes < 2:
        raise ValueError("evaluation_interval must be positive and evaluation_episodes must be at least two")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)

    model = RecurrentActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    env = KeyDoorAgentEnv(seed=seed)
    curve: list[dict[str, float]] = []

    initial = evaluate_policy(model, episodes=evaluation_episodes, seed=seed + 10_000)
    curve.append({"update": -1.0, "batch_success": float("nan"), "loss": float("nan"), **initial})

    for update in range(updates):
        policy_terms: list[torch.Tensor] = []
        value_terms: list[torch.Tensor] = []
        entropy_terms: list[torch.Tensor] = []
        batch_success: list[float] = []

        for episode in range(episodes_per_update):
            target = "red" if (update * episodes_per_update + episode) % 2 == 0 else "blue"
            transitions, tensors, report = collect_episode(
                model,
                env,
                reward_mode=reward_mode,
                target=target,
                policy_version=f"update-{update}",
                gamma=gamma,
            )
            rewards = [t.base_reward if reward_mode == "sparse" else t.shaped_reward for t in transitions]
            returns = discounted_returns(rewards, gamma)
            for (logprob, value, entropy), target_return in zip(tensors, returns, strict=True):
                advantage = target_return.to(value.device) - value.detach()
                policy_terms.append(-logprob * advantage)
                value_terms.append((value - target_return.to(value.device)).pow(2))
                entropy_terms.append(entropy)
            batch_success.append(float(report["success"]))

        loss = (
            torch.stack(policy_terms).mean()
            + value_coefficient * torch.stack(value_terms).mean()
            - entropy_coefficient * torch.stack(entropy_terms).mean()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if update % evaluation_interval == 0 or update == updates - 1:
            metrics = evaluate_policy(model, episodes=evaluation_episodes, seed=seed + 20_000)
            curve.append(
                {
                    "update": float(update),
                    "batch_success": float(np.mean(batch_success)),
                    "loss": float(loss.detach()),
                    **metrics,
                }
            )
    return model, curve


def rollout_contract_from_transitions(
    transitions: list[Transition],
    *,
    task_id: str,
    environment_version: str,
    policy_version: str,
) -> dict[str, Any]:
    """Create a compact turn-level contract record for validation examples."""
    if not transitions:
        raise ValueError("transitions must be non-empty")
    action_ids = [ACTIONS.index(t.action) for t in transitions]
    observations = [transitions[0].observation] + [t.next_observation for t in transitions]
    record = {
        "task_id": task_id,
        "environment_id": "keydoor-agent-v1",
        "environment_version": environment_version,
        "token_ids": action_ids,  # one symbolic policy token per macro-action in this small environment
        "policy_mask": [1] * len(action_ids),
        "behavior_logprobs": [t.behavior_logprob for t in transitions],
        "policy_versions": {str(i): policy_version for i in range(len(action_ids))},
        "turn_ids": list(range(len(action_ids))),
        "action_ids": action_ids,
        "observations": observations,
        "rewards": [t.base_reward for t in transitions],
        "stop_reason": transitions[-1].stop_reason,
        "terminated": transitions[-1].terminated,
        "truncated": transitions[-1].truncated,
        "infra_error": False,
    }
    return record


__all__ = [name for name in globals() if not name.startswith("_")]
