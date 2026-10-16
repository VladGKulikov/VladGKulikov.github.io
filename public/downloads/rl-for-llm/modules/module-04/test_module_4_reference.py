#!/usr/bin/env python3
"""Deterministic tests for Module 4 reference implementations."""
from __future__ import annotations

import unittest

import numpy as np

from module_4_ppo_training import (
    ActorCritic,
    CartPoleBatch,
    PPOConfig,
    collect_rollout,
    train_ppo,
)


from module_4_llm_sequence import (
    EOS,
    TinyAutoregressivePolicy,
    collect_sequence_batch,
    seed_everything as seed_sequence,
    train_tiny_sequence_policy,
)

from module_4_reference import (
    action_independent_baseline_expectation,
    broadcast_sequence_advantages,
    discounted_score_weights,
    expected_in_sample_mean_scale,
    exact_expectation_gradient,
    gae_advantages,
    grad_log_softmax,
    group_mean_advantages,
    group_zscore_advantages,
    leave_one_out_baselines,
    masked_token_ppo_objective,
    masked_token_ppo_terms,
    policy_ratio,
    ppo_clipped_terms,
    ppo_diagnostics,
    ppo_policy_objective,
    reinforce_gradient,
    response_token_mask,
    reward_to_go,
    rloo_advantages,
    score_norm_weighted_baseline,
    sequence_log_probability,
    sequence_log_probability_ratio,
    sequence_probability_ratio,
    simulate_group_baseline_bias,
    stable_softmax,
    token_probability_ratio,
    trajectory_importance_ratio,
    value_targets,
)


class Module4ReferenceTests(unittest.TestCase):
    def test_01_stable_softmax_large_logits(self) -> None:
        np.testing.assert_allclose(stable_softmax([1000.0, 999.0]), [0.7310585786, 0.2689414214], rtol=2e-10)

    def test_02_grad_log_softmax(self) -> None:
        np.testing.assert_allclose(grad_log_softmax([1.0, 2.0, 3.0], 2), [-0.09003057, -0.24472847, 0.33475904], rtol=1e-7)

    def test_03_reward_to_go(self) -> None:
        np.testing.assert_allclose(reward_to_go([1.0, 2.0, 3.0], 0.5), [2.75, 3.5, 3.0])

    def test_04_discounted_score_weights(self) -> None:
        np.testing.assert_allclose(discounted_score_weights([1.0, 2.0, 3.0], 0.5), [2.75, 1.75, 0.75])

    def test_05_reinforce_gradient_reductions(self) -> None:
        logits = np.array([0.0, 0.0])
        actions = np.array([0, 1])
        weights = np.array([2.0, -1.0])
        per = reinforce_gradient(logits, actions, weights, reduction="none")
        expected = np.array([[1.0, -1.0], [0.5, -0.5]])
        np.testing.assert_allclose(per, expected)
        np.testing.assert_allclose(reinforce_gradient(logits, actions, weights, reduction="sum"), expected.sum(axis=0))
        np.testing.assert_allclose(reinforce_gradient(logits, actions, weights), expected.mean(axis=0))

    def test_06_action_independent_baseline_expectation_zero(self) -> None:
        np.testing.assert_allclose(action_independent_baseline_expectation([0.4, -0.2, 1.1], 17.0), np.zeros(3), atol=1e-14)

    def test_07_exact_weighted_baseline_differs_from_value(self) -> None:
        logits = np.array([2.0, 0.0])
        q = np.array([1.0, 4.0])
        p = stable_softmax(logits)
        value = float(np.dot(p, q))
        exact = score_norm_weighted_baseline(logits, q)
        self.assertAlmostEqual(value, 1.357608766, places=8)
        self.assertAlmostEqual(exact, 3.642391234, places=8)
        self.assertNotAlmostEqual(value, exact, places=5)

    def test_08_leave_one_out_baselines(self) -> None:
        np.testing.assert_allclose(leave_one_out_baselines([1.0, 3.0, 8.0, 2.0]), [13/3, 11/3, 2.0, 4.0])

    def test_09_centered_advantages_sum_zero(self) -> None:
        a = group_mean_advantages([1.0, 3.0, 8.0, 2.0])
        np.testing.assert_allclose(a, [-2.5, -0.5, 4.5, -1.5])
        self.assertAlmostEqual(float(np.sum(a)), 0.0)

    def test_10_rloo_proportional_to_centered(self) -> None:
        rewards = np.array([1.0, 3.0, 8.0, 2.0])
        np.testing.assert_allclose(rloo_advantages(rewards), 4/3 * group_mean_advantages(rewards))

    def test_11_group_zscore(self) -> None:
        z = group_zscore_advantages([1.0, 3.0, 8.0, 2.0])
        self.assertAlmostEqual(float(np.mean(z)), 0.0, places=14)
        self.assertAlmostEqual(float(np.mean(z ** 2)), 1.0, places=14)
        np.testing.assert_array_equal(group_zscore_advantages([5.0, 5.0, 5.0]), np.zeros(3))

    def test_12_expected_scale(self) -> None:
        self.assertEqual(expected_in_sample_mean_scale(4), 0.75)
        self.assertEqual(expected_in_sample_mean_scale(1), 0.0)

    def test_13_simulated_group_bias(self) -> None:
        result = simulate_group_baseline_bias(group_size=8, groups=250_000, seed=9)
        self.assertAlmostEqual(result["exact"], 0.273, places=12)
        self.assertAlmostEqual(result["predicted_same_group"], 0.238875, places=12)
        self.assertLess(abs(result["plain_mean"] - result["exact"]), 0.0015)
        self.assertLess(abs(result["same_group_mean"] - result["predicted_same_group"]), 0.0015)
        self.assertLess(abs(result["rloo_mean"] - result["exact"]), 0.0015)

    def test_14_gae_regular_trajectory(self) -> None:
        rewards = np.array([1.0, 2.0])
        values = np.array([0.5, 0.7])
        next_values = np.array([0.7, 0.0])
        term = np.array([False, True])
        ends = np.array([False, True])
        result = gae_advantages(rewards, values, next_values, term, ends, gamma=0.9, gae_lambda=0.8)
        delta1 = 2.0 - 0.7
        delta0 = 1.0 + 0.9 * 0.7 - 0.5
        np.testing.assert_allclose(result, [delta0 + 0.9 * 0.8 * delta1, delta1])

    def test_15_gae_truncation_bootstraps_without_crossing_reset(self) -> None:
        result = gae_advantages([1.0, 100.0], [2.0, 0.0], [10.0, 0.0], [False, True], [True, True], gamma=0.9, gae_lambda=1.0)
        np.testing.assert_allclose(result, [8.0, 100.0])

    def test_16_value_targets(self) -> None:
        np.testing.assert_allclose(value_targets([2.0, 3.0], [0.5, -1.0]), [2.5, 2.0])

    def test_17_policy_and_trajectory_ratios(self) -> None:
        new = np.log([0.6, 0.2, 0.8])
        old = np.log([0.3, 0.4, 0.4])
        np.testing.assert_allclose(policy_ratio(new, old), [2.0, 0.5, 2.0])
        self.assertAlmostEqual(trajectory_importance_ratio(new, old), 2.0)
        self.assertAlmostEqual(trajectory_importance_ratio(new, old, [1, 0, 1]), 4.0)

    def test_18_ppo_sign_cases(self) -> None:
        ratio = np.array([1.5, 0.5, 1.5, 0.5])
        advantage = np.array([2.0, 2.0, -2.0, -2.0])
        _, _, pessimistic = ppo_clipped_terms(ratio, advantage, 0.2)
        np.testing.assert_allclose(pessimistic, [2.4, 1.0, -3.0, -1.6])
        self.assertAlmostEqual(ppo_policy_objective(ratio, advantage, 0.2), -0.3)

    def test_19_diagnostics_outside_vs_active(self) -> None:
        ratio = np.array([1.5, 0.5, 1.5, 0.5])
        new = np.log(ratio)
        old = np.zeros(4)
        advantage = np.array([2.0, 2.0, -2.0, -2.0])
        d = ppo_diagnostics(new, old, advantage, 0.2)
        self.assertEqual(d.ratio_outside_fraction, 1.0)
        self.assertEqual(d.active_clip_fraction, 0.5)
        self.assertGreaterEqual(d.approx_kl_nonnegative, 0.0)

    def test_20_sequence_log_probability_mask(self) -> None:
        lp = np.array([[-1.0, -2.0, -3.0], [-0.1, -0.2, -0.3]])
        mask = np.array([[1, 1, 0], [1, 0, 0]])
        np.testing.assert_allclose(sequence_log_probability(lp, mask), [-3.0, -0.1])

    def test_21_broadcast_advantages(self) -> None:
        mask = np.array([[1, 1, 0], [1, 0, 0]])
        np.testing.assert_allclose(broadcast_sequence_advantages([2.0, -1.0], mask), [[2, 2, 0], [-1, 0, 0]])

    def test_22_invalid_loo(self) -> None:
        with self.assertRaises(ValueError):
            leave_one_out_baselines([1.0])

    def test_23_invalid_ppo_ratio(self) -> None:
        with self.assertRaises(ValueError):
            ppo_clipped_terms([0.0], [1.0])


    def test_24_score_function_with_direct_term(self) -> None:
        p = np.array([0.25, 0.75])
        scores = np.array([[1.5, -0.5], [-0.5, 1/6]])
        values = np.array([2.0, -1.0])
        direct = np.array([[0.2, 0.0], [0.2, 0.0]])
        got = exact_expectation_gradient(p, scores, values, direct)
        manual = np.sum(p[:, None] * (values[:, None] * scores + direct), axis=0)
        np.testing.assert_allclose(got, manual)
        without_direct = exact_expectation_gradient(p, scores, values)
        np.testing.assert_allclose(got - without_direct, [0.2, 0.0])

    def test_25_gae_lambda_one_telescopes_at_terminal(self) -> None:
        rewards = np.array([1.0, 2.0, 4.0])
        values = np.array([0.4, 0.7, 1.2])
        next_values = np.array([0.7, 1.2, 0.0])
        terminated = np.array([False, False, True])
        cuts = np.array([False, False, True])
        gamma = 0.9
        a = gae_advantages(rewards, values, next_values, terminated, cuts, gamma=gamma, gae_lambda=1.0)
        returns = reward_to_go(rewards, gamma)
        np.testing.assert_allclose(a, returns - values)

    def test_26_rollout_boundary_bootstraps_but_cuts_recursion(self) -> None:
        # Record 0 is the last transition of one rollout buffer.  The next
        # value belongs to the same environment episode, but record 1 is an
        # unrelated array element and must not leak into record 0's GAE.
        a = gae_advantages(
            rewards=[1.0, 100.0],
            values=[2.0, 0.0],
            next_values=[10.0, 0.0],
            terminated=[False, True],
            episode_ends=[True, True],
            gamma=0.9,
            gae_lambda=1.0,
        )
        np.testing.assert_allclose(a, [1.0 + 0.9 * 10.0 - 2.0, 100.0])

    def test_27_response_mask_shift_eos_and_padding(self) -> None:
        # Original positions: BOS P1 P2 R1 R2 EOS PAD.  The returned columns
        # align with labels input_ids[:, 1:], so response begins at column 2.
        ids = np.array([
            [1, 10, 11, 20, 21, 2, 0],
            [1, 12, 30, 2, 99, 0, 0],
        ])
        mask = response_token_mask(ids, [3, 2], pad_token_id=0, eos_token_id=2)
        np.testing.assert_array_equal(mask, [
            [0, 0, 1, 1, 1, 0],
            [0, 1, 1, 0, 0, 0],
        ])
        no_eos = response_token_mask(ids, [3, 2], pad_token_id=0, eos_token_id=2, include_eos=False)
        np.testing.assert_array_equal(no_eos, [
            [0, 0, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ])


    def test_28_cartpole_distinguishes_termination_and_truncation(self) -> None:
        terminated_env = CartPoleBatch(1, seed=1, max_episode_steps=500)
        terminated_env.state[0] = np.array([2.5, 0.0, 0.0, 0.0], dtype=np.float32)
        transition = terminated_env.step(np.array([1]))
        self.assertTrue(bool(transition["terminated"][0]))
        self.assertFalse(bool(transition["truncated"][0]))

        truncated_env = CartPoleBatch(1, seed=2, max_episode_steps=1)
        truncated_env.state[0] = np.zeros(4, dtype=np.float32)
        transition = truncated_env.step(np.array([1]))
        self.assertFalse(bool(transition["terminated"][0]))
        self.assertTrue(bool(transition["truncated"][0]))

    def test_29_rollout_contract_and_buffer_cut(self) -> None:
        import torch
        config = PPOConfig(
            seed=5,
            num_envs=3,
            rollout_steps=8,
            updates=1,
            minibatch_size=12,
            evaluation_episodes=2,
        )
        model = ActorCritic(hidden_size=16)
        env = CartPoleBatch(3, seed=6, max_episode_steps=20)
        batch, next_observation = collect_rollout(
            model, env, env.state.copy(), config, device=torch.device("cpu")
        )
        self.assertEqual(batch.observations.shape, (24, 4))
        self.assertEqual(batch.actions.shape, (24,))
        self.assertEqual(next_observation.shape, (3, 4))
        cuts = batch.trace_cuts.reshape(8, 3)
        self.assertTrue(bool(np.all(cuts[-1])))
        self.assertTrue(bool(np.all(np.isfinite(batch.advantages))))
        self.assertTrue(bool(np.all(np.isfinite(batch.value_targets))))

    def test_30_short_ppo_run_is_finite_and_reproducible(self) -> None:
        config = PPOConfig(
            seed=7,
            num_envs=4,
            rollout_steps=32,
            updates=2,
            update_epochs=2,
            minibatch_size=64,
            evaluation_episodes=2,
            max_episode_steps=100,
        )
        first = train_ppo(config)
        second = train_ppo(config)
        np.testing.assert_allclose(
            first["history"]["evaluation_return_mean"],
            second["history"]["evaluation_return_mean"],
        )
        self.assertTrue(np.isfinite(first["final_evaluation_mean"]))
        self.assertEqual(len(first["history"]["update"]), 2)


    def test_31_token_and_sequence_probability_ratios(self) -> None:
        old = np.log(np.array([[0.50, 0.25, 0.10], [0.40, 0.20, 0.30]]))
        new = np.log(np.array([[0.60, 0.20, 0.90], [0.20, 0.30, 0.70]]))
        mask = np.array([[1, 1, 0], [1, 1, 1]])
        token = token_probability_ratio(new, old, mask)
        np.testing.assert_allclose(token, [[1.2, 0.8, 1.0], [0.5, 1.5, 7/3]])
        expected_sequence = np.array([1.2 * 0.8, 0.5 * 1.5 * (7/3)])
        np.testing.assert_allclose(sequence_probability_ratio(new, old, mask), expected_sequence)
        np.testing.assert_allclose(sequence_log_probability_ratio(new, old, mask), np.log(expected_sequence))

    def test_32_masked_token_ppo_sign_cases(self) -> None:
        old = np.zeros((2, 3))
        new = np.log(np.array([[1.5, 0.5, 9.0], [1.5, 0.5, 2.0]]))
        mask = np.array([[1, 1, 0], [1, 1, 1]])
        advantage = np.array([2.0, -2.0])
        ratio, unclipped, clipped, pessimistic = masked_token_ppo_terms(
            new, old, advantage, mask, clip_epsilon=0.2
        )
        np.testing.assert_allclose(ratio, [[1.5, 0.5, 1.0], [1.5, 0.5, 2.0]])
        np.testing.assert_allclose(unclipped, [[3.0, 1.0, 0.0], [-3.0, -1.0, -4.0]])
        np.testing.assert_allclose(clipped, [[2.4, 1.6, 0.0], [-2.4, -1.6, -2.4]])
        np.testing.assert_allclose(pessimistic, [[2.4, 1.0, 0.0], [-3.0, -1.6, -4.0]])

    def test_33_token_reductions_are_explicit_and_different(self) -> None:
        old = np.zeros((2, 3))
        new = np.zeros((2, 3))
        mask = np.array([[1, 0, 0], [1, 1, 1]])
        advantage = np.array([2.0, 1.0])
        self.assertAlmostEqual(
            masked_token_ppo_objective(new, old, advantage, mask, reduction="token_mean"),
            5.0 / 4.0,
        )
        self.assertAlmostEqual(
            masked_token_ppo_objective(new, old, advantage, mask, reduction="sequence_mean"),
            1.5,
        )
        self.assertAlmostEqual(
            masked_token_ppo_objective(new, old, advantage, mask, reduction="sequence_sum_mean"),
            2.5,
        )

    def test_34_masked_padding_cannot_change_token_objective(self) -> None:
        old = np.array([[-0.2, -0.4, -100.0]])
        new_a = np.array([[-0.1, -0.7, 100.0]])
        new_b = np.array([[-0.1, -0.7, -999.0]])
        mask = np.array([[1, 1, 0]])
        a = masked_token_ppo_objective(new_a, old, [0.7], mask)
        b = masked_token_ppo_objective(new_b, old, [0.7], mask)
        self.assertAlmostEqual(a, b)

    def test_35_token_contract_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            masked_token_ppo_objective([[0.0]], [[0.0]], [1.0], [[0.0]])
        with self.assertRaises(ValueError):
            masked_token_ppo_objective([[0.0, 0.0]], [[0.0]], [1.0], [[1.0, 1.0]])
        with self.assertRaises(ValueError):
            masked_token_ppo_terms([[0.0]], [[0.0]], [1.0], [[1.0]], clip_epsilon=-0.1)
        with self.assertRaises(ValueError):
            masked_token_ppo_objective([[0.0]], [[0.0]], [1.0], [[1.0]], reduction="mystery")

    def test_36_sequence_batch_aligns_mask_and_old_log_probs(self) -> None:
        seed_sequence(13)
        model = TinyAutoregressivePolicy(embedding_dim=12, hidden_size=24)
        batch = collect_sequence_batch(model, group_size=4)
        self.assertEqual(tuple(batch.response_mask.shape), tuple(batch.old_token_log_probs.shape))
        self.assertEqual(batch.input_ids.shape[0], 12)
        self.assertTrue(bool(np.all(np.isfinite(batch.old_token_log_probs.cpu().numpy()))))
        ids = batch.input_ids.cpu().numpy()
        mask = batch.response_mask.cpu().numpy()
        for row, row_mask in zip(ids, mask):
            response = row[2:]
            labels = row[1:]
            active = np.flatnonzero(row_mask > 0.5)
            self.assertGreater(active.size, 0)
            if EOS in response:
                eos_label_position = int(np.flatnonzero(labels == EOS)[0])
                self.assertEqual(int(row_mask[eos_label_position]), 1)
                self.assertTrue(bool(np.all(row_mask[eos_label_position + 1:] == 0)))

    def test_37_sequence_rloo_advantages_sum_to_zero_per_prompt(self) -> None:
        seed_sequence(17)
        batch = collect_sequence_batch(TinyAutoregressivePolicy(), group_size=5)
        for prompt in range(3):
            idx = np.flatnonzero(batch.prompt_classes == prompt)
            self.assertEqual(idx.size, 5)
            self.assertAlmostEqual(float(np.sum(batch.advantages[idx])), 0.0, places=12)

    def test_38_tiny_sequence_training_is_reproducible_and_learns(self) -> None:
        kwargs = dict(seed=5, iterations=30, group_size=8, learning_rate=5e-3, ppo_epochs=3)
        first = train_tiny_sequence_policy(**kwargs)
        second = train_tiny_sequence_policy(**kwargs)
        np.testing.assert_allclose(
            first["history"]["evaluation_mean_reward"],
            second["history"]["evaluation_mean_reward"],
        )
        initial = first["history"]["evaluation_mean_reward"][0]
        final = first["final_evaluation"]["mean_reward"]
        self.assertGreater(final, initial + 0.5)
        self.assertGreater(first["final_evaluation"]["success_rate"], 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
