#!/usr/bin/env python3
"""Deterministic integration tests for Module 5 environment and learners."""
from __future__ import annotations

import unittest
import numpy as np
import torch

from module_5_training import (
    DecisionTransformer,
    KeyDoorGrid,
    ShortestPathExpert,
    collect_offline_dataset,
    dagger_experiment,
    offline_experiment,
    rollout,
    train_decision_transformer,
    train_online_dqn,
    transitions_to_arrays,
)


class TestKeyDoorGrid(unittest.TestCase):
    def test_observation_contract_round_trips(self) -> None:
        env = KeyDoorGrid(seed=1)
        observation, _ = env.reset(seed=2, start=(6, 0))
        self.assertEqual(observation.shape, (env.observation_dim,))
        state = env.decode(observation)
        self.assertEqual((state.row, state.col, state.has_key), (6, 0, False))

    def test_expert_reaches_goal(self) -> None:
        env = KeyDoorGrid(seed=3)
        expert = ShortestPathExpert(env)
        episodes, _ = rollout(
            env,
            expert.action,
            episodes=len(env.start_positions),
            seed=10,
            starts=env.start_positions,
            collect=False,
        )
        self.assertTrue(all(ep.terminated and not ep.truncated for ep in episodes))
        self.assertTrue(all(ep.return_ > 0.0 for ep in episodes))

    def test_time_limit_is_truncation_not_termination(self) -> None:
        env = KeyDoorGrid(max_steps=1, seed=4)
        env.reset(start=(6, 0))
        _, _, terminated, truncated, info = env.step(3)  # bump into the boundary
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["outcome"], "time-limit")

    def test_fixed_log_keeps_both_flags(self) -> None:
        _, transitions = collect_offline_dataset(seed=5, episodes=8, profile="mixed")
        arrays = transitions_to_arrays(transitions)
        self.assertEqual(arrays["states"].shape[1], KeyDoorGrid().observation_dim)
        self.assertEqual(arrays["terminated"].shape, arrays["truncated"].shape)
        self.assertFalse(np.any((arrays["terminated"] == 1) & (arrays["truncated"] == 1)))


class TestLearningPipelines(unittest.TestCase):
    def test_dagger_labels_learner_states(self) -> None:
        result = dagger_experiment(
            seed=17,
            expert_episodes=4,
            rounds=1,
            learner_episodes_per_round=4,
            epochs=5,
            evaluation_episodes=12,
        )
        records = result["records"]
        self.assertEqual(len(records), 2)
        self.assertGreater(records[1]["dataset_size"], records[0]["dataset_size"])
        self.assertGreaterEqual(records[1]["success_rate"], records[0]["success_rate"])

    def test_online_dqn_smoke_contract(self) -> None:
        result = train_online_dqn(
            seed=9,
            episodes=8,
            double=True,
            warmup=32,
            expert_prefill_episodes=2,
            evaluation_episodes=4,
        )
        self.assertEqual(result["returns"].shape, (8,))
        self.assertEqual(result["success"].shape, (8,))
        self.assertTrue(result["double"])
        self.assertIn("success_rate", result["evaluation"])

    def test_offline_pipeline_smoke(self) -> None:
        result = offline_experiment(
            seed=13,
            dataset_episodes=12,
            train_epochs=2,
            evaluation_episodes=4,
        )
        self.assertEqual(
            set(result["results"]),
            {"Behavior cloning", "Offline DQN", "BCQ-like", "CQL", "IQL"},
        )
        self.assertEqual(result["action_counts"].shape, (4,))
        for metrics in result["results"].values():
            self.assertTrue(0.0 <= metrics["success_rate"] <= 1.0)

    def test_decision_transformer_forward_and_tiny_fit(self) -> None:
        episodes, _ = collect_offline_dataset(seed=15, episodes=6, profile="mixed")
        model, history = train_decision_transformer(
            episodes,
            seed=16,
            state_dim=KeyDoorGrid().observation_dim,
            action_dim=4,
            context_steps=6,
            epochs=1,
            batch_size=16,
        )
        self.assertIsInstance(model, DecisionTransformer)
        self.assertEqual(len(history), 1)
        self.assertTrue(np.isfinite(history[0]))
        with torch.no_grad():
            states = torch.zeros(2, 3, KeyDoorGrid().observation_dim)
            actions = torch.zeros(2, 3, dtype=torch.long)
            rtg = torch.zeros(2, 3, 1)
            timesteps = torch.arange(3).repeat(2, 1)
            mask = torch.ones(2, 3)
            logits = model(states, actions, rtg, timesteps, mask)
        self.assertEqual(tuple(logits.shape), (2, 3, 4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
