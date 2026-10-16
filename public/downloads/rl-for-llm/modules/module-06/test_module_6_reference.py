#!/usr/bin/env python3
"""Deterministic tests for Module 6 reference utilities."""
from __future__ import annotations

import math
import unittest

import numpy as np

from module_6_reference import (
    behavior_distribution,
    behavior_logprob,
    binary_tree_sequence_kl,
    classify_stop_event,
    decoding_distribution_with_order,
    categorical_entropy,
    causal_response_stats,
    exact_kl_from_logits,
    greedy_action,
    importance_corrected_mean,
    kl_estimator_moments,
    kl_estimators_from_selected_logprobs,
    log_softmax,
    masked_sequence_stats,
    rollout_summary,
    sample_kl_estimators,
    sequence_logprob_ratio,
    selected_token_logprobs,
    softmax,
    temperature_distribution,
    top_k_distribution,
    top_p_distribution,
    truncation_mismatch,
    validate_rollout_batch,
)


class Module6ReferenceTests(unittest.TestCase):
    def test_01_log_softmax_large_logits(self) -> None:
        actual = log_softmax(np.array([1000.0, 999.0, 0.0]))
        self.assertTrue(np.all(np.isfinite(actual)))
        self.assertAlmostEqual(float(np.exp(actual).sum()), 1.0, places=12)
        self.assertAlmostEqual(float(actual[0]), -0.3132616875, places=9)

    def test_02_softmax_normalizes(self) -> None:
        p = softmax(np.array([[0.0, 1.0], [2.0, -2.0]]), axis=-1)
        np.testing.assert_allclose(p.sum(axis=-1), np.ones(2), atol=1e-12)

    def test_03_selected_token_logprobs(self) -> None:
        logits = np.array([[0.0, 0.0], [2.0, 0.0]])
        ids = np.array([1, 0])
        actual = selected_token_logprobs(logits, ids)
        expected = np.array([-math.log(2), -math.log1p(math.exp(-2.0))])
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_04_masked_sequence_stats(self) -> None:
        logits = np.log(np.array([[0.5, 0.5], [0.25, 0.75], [0.1, 0.9]]))
        ids = np.array([0, 0, 1])
        stats = masked_sequence_stats(logits, ids, np.array([1, 1, 0]))
        self.assertEqual(stats.token_count, 2)
        self.assertAlmostEqual(stats.sum_logprob, math.log(0.125), places=12)
        self.assertAlmostEqual(stats.perplexity, math.sqrt(8), places=12)

    def test_05_causal_shift_and_mask(self) -> None:
        logits = np.zeros((4, 3))
        ids = np.array([0, 1, 2, 1])
        token_mask = np.array([0, 0, 1, 1])
        stats = causal_response_stats(logits, ids, token_mask)
        self.assertEqual(stats.token_count, 2)
        self.assertAlmostEqual(stats.sum_logprob, -2 * math.log(3), places=12)

    def test_06_causal_requires_two_tokens(self) -> None:
        with self.assertRaises(ValueError):
            causal_response_stats(np.zeros((1, 2)), np.array([0]), np.array([1]))

    def test_07_exact_kl_identity(self) -> None:
        z = np.array([[1.0, -1.0], [0.5, 0.5]])
        np.testing.assert_allclose(exact_kl_from_logits(z, z), np.zeros(2), atol=1e-14)

    def test_08_exact_kl_asymmetry(self) -> None:
        p = np.log(np.array([0.8, 0.2]))
        q = np.log(np.array([0.5, 0.5]))
        pq = float(exact_kl_from_logits(p, q))
        qp = float(exact_kl_from_logits(q, p))
        self.assertGreater(pq, 0)
        self.assertGreater(qp, 0)
        self.assertNotAlmostEqual(pq, qp, places=6)

    def test_09_k3_nonnegative_and_k1_signed(self) -> None:
        d = np.linspace(-4.0, 4.0, 101)
        # log_q - log_p = -d, hence k1=d.
        est = kl_estimators_from_selected_logprobs(np.zeros_like(d), -d)
        self.assertTrue(np.all(est.k3 >= -1e-14))
        self.assertTrue(np.all(est.k2 >= 0))
        self.assertTrue(np.any(est.k1 < 0))

    def test_10_k1_and_k3_unbiased_by_enumeration(self) -> None:
        p = np.array([0.7, 0.3])  # sampling distribution
        q = np.array([0.4, 0.6])
        est = kl_estimators_from_selected_logprobs(np.log(p), np.log(q))
        exact = float(np.sum(p * (np.log(p) - np.log(q))))
        self.assertAlmostEqual(float(np.sum(p * est.k1)), exact, places=12)
        self.assertAlmostEqual(float(np.sum(p * est.k3)), exact, places=12)
        self.assertNotAlmostEqual(float(np.sum(p * est.k2)), exact, places=6)

    def test_11_sample_kl_reproducible(self) -> None:
        a = sample_kl_estimators(np.array([0.0, 1.0]), np.array([1.0, 0.0]), 5000, seed=7)
        b = sample_kl_estimators(np.array([0.0, 1.0]), np.array([1.0, 0.0]), 5000, seed=7)
        self.assertEqual(a, b)
        self.assertGreater(a["k1_negative_fraction"], 0.0)

    def test_12_temperature_one_matches_softmax(self) -> None:
        z = np.array([2.0, 0.0, -1.0])
        np.testing.assert_allclose(temperature_distribution(z, 1.0), softmax(z), atol=1e-12)

    def test_13_higher_temperature_higher_entropy_for_fixed_logits(self) -> None:
        z = np.array([4.0, 1.0, -2.0])
        self.assertGreater(
            categorical_entropy(temperature_distribution(z, 2.0)),
            categorical_entropy(temperature_distribution(z, 0.5)),
        )

    def test_14_greedy_action(self) -> None:
        self.assertEqual(greedy_action(np.array([1.0, 3.0, 3.0])), 1)

    def test_15_top_k_support(self) -> None:
        p = top_k_distribution(np.log(np.array([0.4, 0.3, 0.2, 0.1])), 2)
        np.testing.assert_allclose(p, np.array([4 / 7, 3 / 7, 0, 0]), atol=1e-12)

    def test_16_top_p_smallest_prefix(self) -> None:
        p = top_p_distribution(np.log(np.array([0.5, 0.3, 0.15, 0.05])), 0.75)
        np.testing.assert_allclose(p, np.array([0.625, 0.375, 0, 0]), atol=1e-12)

    def test_17_behavior_distribution_order(self) -> None:
        p = behavior_distribution(np.array([3.0, 2.0, 1.0, 0.0]), temperature=1.0, top_k=3, top_p=0.8)
        self.assertAlmostEqual(float(p.sum()), 1.0, places=12)
        self.assertLessEqual(int(np.count_nonzero(p)), 3)

    def test_18_behavior_mask_and_bias(self) -> None:
        p = behavior_distribution(
            np.zeros(3), allowed_mask=np.array([1, 0, 1]), logit_bias=np.array([0.0, 99.0, 1.0])
        )
        self.assertEqual(float(p[1]), 0.0)
        self.assertGreater(float(p[2]), float(p[0]))

    def test_19_behavior_logprob_outside_support(self) -> None:
        self.assertEqual(behavior_logprob(np.array([0.7, 0.3, 0.0]), 2), float("-inf"))

    def test_20_truncation_shift(self) -> None:
        result = truncation_mismatch(np.array([0.5, 0.3, 0.15, 0.05]), 0.75, 1)
        self.assertEqual(result["in_support"], 1.0)
        self.assertAlmostEqual(result["retained_mass"], 0.8, places=12)
        self.assertAlmostEqual(result["logprob_shift"], -math.log(0.8), places=12)

    def test_21_binary_tree_chain_rule(self) -> None:
        sequence, chain = binary_tree_sequence_kl(
            np.array([0.6, 0.4]),
            np.array([0.5, 0.5]),
            {0: np.array([0.8, 0.2]), 1: np.array([0.3, 0.7])},
            {0: np.array([0.7, 0.3]), 1: np.array([0.6, 0.4])},
        )
        self.assertAlmostEqual(sequence, chain, places=12)

    def test_22_rollout_summary(self) -> None:
        result = rollout_summary(
            np.array([1, 1, 0]),
            np.array([0, 1, 0]),
            np.array([-0.5, -0.7, 0.0]),
            np.array([-0.4, -0.9, 0.0]),
        )
        self.assertEqual(result["response_tokens"], 2.0)
        self.assertEqual(result["terminal_events"], 1.0)
        self.assertAlmostEqual(result["sequence_importance_ratio"], math.exp(-0.1), places=12)

    def test_23_importance_correction_full_support(self) -> None:
        # Exact enumeration under uniform behavior: each row contributes with probability 1/2.
        values = np.array([1.0, 3.0])
        target = np.array([0.8, 0.2])
        behavior = np.array([0.5, 0.5])
        estimate = importance_corrected_mean(values, np.log(target), np.log(behavior))
        self.assertAlmostEqual(estimate, float(np.sum(target * values)), places=12)

    def valid_batch(self) -> dict[str, np.ndarray]:
        return {
            "response_ids": np.array([[4, 5, 0], [7, 0, 0]]),
            "response_mask": np.array([[1, 1, 0], [1, 0, 0]]),
            "old_logprobs": np.array([[-0.4, -0.6, np.nan], [-0.2, np.nan, np.nan]]),
            "ref_logprobs": np.array([[-0.5, -0.7, np.nan], [-0.3, np.nan, np.nan]]),
            "sequence_rewards": np.array([1.0, 0.0]),
            "terminated": np.array([True, False]),
            "truncated": np.array([False, True]),
            "stop_reason": np.array(["eos", "length"], dtype=object),
        }

    def test_24_rollout_batch_valid(self) -> None:
        result = validate_rollout_batch(self.valid_batch())
        self.assertEqual(result.active_tokens, 3)
        self.assertEqual(result.terminated_sequences, 1)
        self.assertEqual(result.truncated_sequences, 1)

    def test_25_rollout_batch_rejects_mask_hole(self) -> None:
        batch = self.valid_batch()
        batch["response_mask"] = np.array([[1, 0, 1], [1, 0, 0]])
        with self.assertRaises(ValueError):
            validate_rollout_batch(batch)

    def test_26_rollout_batch_requires_stop_reason(self) -> None:
        batch = self.valid_batch()
        del batch["stop_reason"]
        with self.assertRaises(ValueError):
            validate_rollout_batch(batch)

    def test_27_rollout_batch_allows_coincident_boundaries(self) -> None:
        batch = self.valid_batch()
        batch["truncated"] = np.array([True, True])
        result = validate_rollout_batch(batch)
        self.assertEqual(result.terminated_sequences, 1)
        self.assertEqual(result.truncated_sequences, 2)

    def test_28_stop_reason_semantics_are_task_specific(self) -> None:
        terminal = classify_stop_event(
            "stop_string", terminal_reasons=("eos", "stop_string"), truncation_reasons=("length",)
        )
        external = classify_stop_event(
            "stop_string", terminal_reasons=("eos",), truncation_reasons=("stop_string", "length")
        )
        self.assertTrue(terminal.terminated and not terminal.truncated)
        self.assertTrue(external.truncated and not external.terminated)

    def test_29_unknown_stop_reason_is_not_silently_terminal(self) -> None:
        event = classify_stop_event("safety", terminal_reasons=("eos",), truncation_reasons=("length",))
        self.assertFalse(event.terminated)
        self.assertFalse(event.truncated)

    def test_30_kl_moments_near_policy(self) -> None:
        result = kl_estimator_moments(
            np.array([0.35, 0.30, 0.22, 0.13]),
            np.array([0.40, 0.30, 0.20, 0.10]),
        )
        self.assertAlmostEqual(result["exact_kl"], 0.008114094642195101, places=14)
        self.assertAlmostEqual(result["k1_mean"], result["exact_kl"], places=14)
        self.assertAlmostEqual(result["k3_mean"], result["exact_kl"], places=14)
        self.assertLess(result["k3_std"], result["k1_std"])

    def test_31_kl_moments_heavy_ratio_tail(self) -> None:
        result = kl_estimator_moments(np.array([0.9, 0.1]), np.array([0.999, 0.001]))
        self.assertAlmostEqual(result["exact_kl"], 0.09965048512293041, places=14)
        self.assertGreater(result["k3_std"], 15.0 * result["k1_std"])

    def test_32_sequence_logprob_ratio(self) -> None:
        result = sequence_logprob_ratio(
            np.log(np.array([0.6, 0.3, 0.8])),
            np.log(np.array([0.5, 0.4, 0.2])),
            np.array([1, 1, 0]),
        )
        self.assertAlmostEqual(result["log_ratio"], math.log(0.9), places=12)
        self.assertAlmostEqual(result["ratio"], 0.9, places=12)

    def test_33_processor_order_can_change_distribution(self) -> None:
        logits = np.array([1.720, 3.523, 1.987])
        a = decoding_distribution_with_order(
            logits, temperature=1.7, top_k=2, top_p=0.7,
            order=("temperature", "top_k", "top_p"),
        )
        b = decoding_distribution_with_order(
            logits, temperature=1.7, top_k=2, top_p=0.7,
            order=("temperature", "top_p", "top_k"),
        )
        self.assertFalse(np.allclose(a, b))
        self.assertEqual(np.count_nonzero(a), 1)
        self.assertEqual(np.count_nonzero(b), 2)

    def test_34_kl_moments_require_common_positive_support(self) -> None:
        with self.assertRaises(ValueError):
            kl_estimator_moments(np.array([1.0, 0.0]), np.array([0.8, 0.2]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
