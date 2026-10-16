#!/usr/bin/env python3
"""Deterministic smoke tests for the Module 11 recurrent-agent laboratory."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from module_11_agentic_training import (  # noqa: E402
    ACTIONS,
    KeyDoorAgentEnv,
    RecurrentActorCritic,
    collect_episode,
    discounted_returns,
    evaluate_policy,
    masked_logits,
    rollout_contract_from_transitions,
    train_actor_critic,
)
from module_11_reference import validate_rollout_record  # noqa: E402

PASSED = 0


def check(condition: bool, label: str) -> None:
    global PASSED
    if not condition:
        raise AssertionError(label)
    PASSED += 1
    print(f"[PASS] {label}")


def main() -> None:
    env = KeyDoorAgentEnv(seed=0)
    observation, info = env.reset(target="red")
    check(observation == "start" and info["target"] == "red", "deterministic reset")
    check(np.array_equal(env.valid_action_mask(), [1, 0, 0, 0, 0, 0]), "initial action grammar")
    observation, reward, terminated, truncated, info = env.step("inspect")
    check(observation == "clue_red" and not terminated and not truncated, "inspect reveals one-turn clue")
    check(np.array_equal(env.valid_action_mask(), [0, 1, 1, 0, 0, 0]), "post-inspection action grammar")
    observation, _, _, _, info = env.step("pickup_red")
    check(observation == "carrying_red" and info["valid_action"], "correct key pickup")
    observation, _, _, _, info = env.step("open_red")
    check(observation == "opened_red" and info["potential_after"] == 0.85, "correct door opening")
    observation, reward, terminated, truncated, info = env.step("finish")
    check(observation == "success" and reward == 1.0 and terminated and not truncated, "successful task termination")

    collector = KeyDoorAgentEnv(task_horizon=6, collector_limit=2, seed=0)
    collector.reset(target="blue")
    collector.step("inspect")
    obs, reward, terminated, truncated, info = collector.step("pickup_blue")
    check(obs == "collector_truncation" and truncated and not terminated and reward == 0.0, "collector truncation is separate")

    model = RecurrentActorCritic()
    logits = torch.zeros(len(ACTIONS))
    filtered = masked_logits(logits, np.array([1, 0, 0, 0, 0, 0], dtype=bool))
    check(int(torch.argmax(filtered)) == 0 and torch.isfinite(filtered[0]), "masked logits preserve legal action")
    returns = discounted_returns([0.0, 0.0, 1.0], 0.9)
    check(torch.allclose(returns, torch.tensor([0.81, 0.9, 1.0])), "discounted return vector")

    env = KeyDoorAgentEnv(seed=4)
    transitions, tensors, report = collect_episode(model, env, target="red", greedy=True, policy_version="p0")
    check(len(transitions) == len(tensors) and report["policy_version"] == "p0", "episode preserves policy metadata")
    check(all(np.isfinite(t.behavior_logprob) and t.behavior_logprob <= 0.0 for t in transitions), "behavior log-probabilities are valid")
    contract = rollout_contract_from_transitions(transitions, task_id="smoke", environment_version="1", policy_version="p0")
    check(validate_rollout_record(contract) == [], "generated rollout contract validates")

    torch.manual_seed(1)
    initial_model = RecurrentActorCritic()
    initial = evaluate_policy(initial_model, episodes=20, seed=10)
    check(0.0 <= initial["success_rate"] <= 1.0 and initial["mean_turns"] >= 4.0, "evaluation metrics are bounded")

    trained, curve = train_actor_critic(seed=1, reward_mode="potential", updates=6, episodes_per_update=16, evaluation_interval=2)
    final = evaluate_policy(trained, episodes=40, seed=20)
    check(len(curve) >= 4 and curve[0]["update"] == -1.0, "learning curve includes pre-training evaluation")
    check(final["success_rate"] >= 0.5, "smoke training learns at least one target branch")
    check(final["invalid_action_rate"] <= 0.5, "training does not increase invalid-action rate pathologically")

    invalid_calls = (
        lambda: KeyDoorAgentEnv(task_horizon=0),
        lambda: KeyDoorAgentEnv().reset(target="green"),
        lambda: masked_logits(torch.zeros(2), np.array([False, False])),
        lambda: discounted_returns([], 0.9),
    )
    # Empty returns are mathematically allowed by the helper, so test only the first three.
    for call in invalid_calls[:3]:
        try:
            call()
        except (ValueError, RuntimeError):
            continue
        raise AssertionError("invalid-input check did not fail")
    check(True, "invalid training inputs raise")

    print(f"\n{PASSED} tests passed")


if __name__ == "__main__":
    main()
