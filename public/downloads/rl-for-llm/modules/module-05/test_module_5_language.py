#!/usr/bin/env python3
"""Tests for the neural prefix/action offline-RL example in Module 5."""
from __future__ import annotations

import unittest
import numpy as np

from module_5_language import (
    EOS,
    PROMPTS,
    TARGETS,
    collect_fixed_language_log,
    decode_policy,
    decode_value_guided,
    episodes_to_transitions,
    evaluate_decoder,
    sequence_reward,
    train_language_bc,
    train_language_iql,
)


class TestLanguageOfflineRL(unittest.TestCase):
    def test_reward_and_transition_contract(self) -> None:
        episodes = collect_fixed_language_log(seed=1, episodes_per_prompt=5)
        transitions = episodes_to_transitions(episodes)
        self.assertEqual(len(episodes), len(PROMPTS) * 5)
        self.assertTrue(all(episode.tokens[-1] == EOS for episode in episodes))
        self.assertTrue(all(t.reward == 0.0 for t in transitions if not t.terminated))
        self.assertTrue(all(t.next_prefix[-1] == t.action for t in transitions))
        for prompt_id, target in enumerate(TARGETS):
            self.assertEqual(sequence_reward(prompt_id, target), 1.0)

    def test_small_neural_training_and_decoding(self) -> None:
        episodes = collect_fixed_language_log(seed=3, episodes_per_prompt=8)
        transitions = episodes_to_transitions(episodes)
        behavior, _ = train_language_bc(transitions, seed=4, epochs=2, batch_size=64)
        iql = train_language_iql(transitions, seed=5, epochs=2, batch_size=64)
        behavior_eval = evaluate_decoder(lambda prompt: decode_policy(behavior, prompt))
        guided_eval = evaluate_decoder(
            lambda prompt: decode_value_guided(behavior, iql["target"], prompt)
        )
        for metrics in (behavior_eval, guided_eval):
            self.assertEqual(len(metrics["sequences"]), len(PROMPTS))
            self.assertTrue(np.isfinite(metrics["mean_reward"]))
            self.assertTrue(0.0 <= metrics["exact_match"] <= 1.0)
            self.assertTrue(all(sequence[-1] == EOS for sequence in metrics["sequences"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
