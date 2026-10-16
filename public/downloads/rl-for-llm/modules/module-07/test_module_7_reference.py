#!/usr/bin/env python3
"""Deterministic tests for ``module_7_reference.py``."""
from __future__ import annotations

import math
import unittest

import numpy as np

from module_7_reference import (
    PairObservation,
    aggregate_swapped_pairwise,
    binary_log_loss,
    brier_score,
    bt_metrics,
    constrained_scalarize,
    ensemble_mean_std,
    expected_calibration_error,
    fit_pairwise_scores,
    logsumexp,
    lower_confidence_score,
    noisy_label_oracle_accuracy,
    overoptimization_experiment,
    pairwise_gradient,
    pairwise_nll,
    plackett_luce_nll,
    reliability_bins,
    scalarize_rubric,
    sigmoid,
    simulate_noisy_label_accuracy,
    softplus,
    validate_pairwise_split,
)


class Module7ReferenceTests(unittest.TestCase):
    def test_01_sigmoid_stability(self) -> None:
        got = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
        np.testing.assert_allclose(got, [0.0, 0.5, 1.0], atol=1e-15)

    def test_02_softplus_stability(self) -> None:
        got = softplus(np.array([-1000.0, 0.0, 1000.0]))
        np.testing.assert_allclose(got, [0.0, math.log(2.0), 1000.0], atol=1e-12)

    def test_03_pairwise_hard_target(self) -> None:
        self.assertAlmostEqual(float(pairwise_nll(2.0, 1.0)), math.log1p(math.exp(-2.0)), places=14)
        self.assertAlmostEqual(float(pairwise_gradient(2.0, 1.0)), 1/(1+math.exp(-2.0)) - 1, places=14)

    def test_04_pairwise_soft_tie(self) -> None:
        self.assertAlmostEqual(float(pairwise_gradient(0.0, 0.5)), 0.0, places=14)
        self.assertAlmostEqual(float(pairwise_nll(0.0, 0.5)), math.log(2.0), places=14)
        self.assertGreater(float(pairwise_nll(4.0, 0.5)), math.log(2.0))

    def test_05_bt_metrics(self) -> None:
        result = bt_metrics([2.0, 0.0, 5.0], [1.0, 1.0, -5.0])
        np.testing.assert_allclose(result["probabilities"], [0.7310585786, 0.2689414214, 0.9999546021])
        self.assertAlmostEqual(float(result["accuracy"]), 2/3)
        self.assertAlmostEqual(float(result["mean_margin"]), 10/3)

    def test_06_binary_metrics(self) -> None:
        p = np.array([0.9, 0.2, 0.6, 0.4])
        y = np.array([1.0, 0.0, 1.0, 0.0])
        self.assertAlmostEqual(brier_score(p, y), 0.0925)
        self.assertAlmostEqual(binary_log_loss(p, y), 0.3375388286260043)

    def test_07_logsumexp(self) -> None:
        self.assertAlmostEqual(logsumexp([1000.0, 1000.0]), 1000.0 + math.log(2.0))

    def test_08_plackett_luce(self) -> None:
        scores = np.log([4.0, 2.0, 1.0])
        got = plackett_luce_nll(scores, [0, 1, 2])
        expected = -math.log(4/7) - math.log(2/3)
        self.assertAlmostEqual(got, expected, places=14)

    def test_09_plackett_luce_shift_invariance(self) -> None:
        s = np.array([0.2, -0.1, 1.7])
        self.assertAlmostEqual(plackett_luce_nll(s, [2, 0, 1]), plackett_luce_nll(s + 100, [2, 0, 1]))

    def test_10_fit_pairwise_scores_order(self) -> None:
        obs = [
            PairObservation(0, 1), PairObservation(0, 1),
            PairObservation(1, 2), PairObservation(1, 2),
            PairObservation(0, 2),
        ]
        scores = fit_pairwise_scores(3, obs, learning_rate=0.2, steps=3000, l2=0.01)
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])
        self.assertAlmostEqual(float(scores.mean()), 0.0, places=12)

    def test_11_fit_cycle_stays_finite(self) -> None:
        obs = [PairObservation(0, 1), PairObservation(1, 2), PairObservation(2, 0)]
        scores = fit_pairwise_scores(3, obs, learning_rate=0.1, steps=2000)
        np.testing.assert_allclose(scores, np.zeros(3), atol=1e-10)

    def test_12_reliability_bins(self) -> None:
        p = [0.1, 0.2, 0.8, 0.9]
        y = [0, 1, 1, 1]
        bins = reliability_bins(p, y, n_bins=2)
        self.assertEqual([b["count"] for b in bins], [2, 2])
        self.assertAlmostEqual(float(bins[0]["mean_confidence"]), 0.15)
        self.assertAlmostEqual(float(bins[0]["empirical_frequency"]), 0.5)

    def test_13_ece(self) -> None:
        p = [0.1, 0.2, 0.8, 0.9]
        y = [0, 1, 1, 1]
        self.assertAlmostEqual(expected_calibration_error(p, y, n_bins=2), 0.25)

    def test_14_ensemble(self) -> None:
        members = np.array([[1.0, 2.0], [3.0, 2.0], [2.0, 2.0]])
        mean, std = ensemble_mean_std(members)
        np.testing.assert_allclose(mean, [2.0, 2.0])
        np.testing.assert_allclose(std, [math.sqrt(2/3), 0.0])
        conservative = lower_confidence_score(members, coefficient=1.0)
        self.assertGreater(conservative[1], conservative[0])

    def test_15_swapped_pairwise(self) -> None:
        self.assertAlmostEqual(aggregate_swapped_pairwise(0.8, 0.7), 0.55)
        self.assertAlmostEqual(aggregate_swapped_pairwise(0.9, 0.1), 0.9)

    def test_16_scalarization(self) -> None:
        self.assertAlmostEqual(scalarize_rubric([0.8, 0.5, 1.0], [0.5, 0.3, 0.2]), 0.75)
        self.assertEqual(
            constrained_scalarize([0.8, 0.5], [0.5, 0.5], constraint_values=[0.7], lower_bounds=[0.8]),
            float("-inf"),
        )

    def test_17_noisy_label_oracle(self) -> None:
        self.assertAlmostEqual(noisy_label_oracle_accuracy(0.2), 0.8)
        estimate = simulate_noisy_label_accuracy(n_pairs=200_000, flip_probability=0.2, seed=7)
        self.assertLess(abs(estimate - 0.8), 0.003)

    def test_18_overoptimization_reproducible(self) -> None:
        a = overoptimization_experiment(trials=1000, seed=17)
        b = overoptimization_experiment(trials=1000, seed=17)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])

    def test_19_overoptimization_shape(self) -> None:
        result = overoptimization_experiment(trials=4000, seed=7)
        self.assertTrue(np.all(np.diff(result["proxy"]) > 0))
        best = int(np.argmax(result["gold"]))
        self.assertGreater(best, 0)
        self.assertLess(best, len(result["gold"]) - 1)
        self.assertLess(result["gold"][-1], result["gold"][best])

    def test_20_prompt_split(self) -> None:
        result = validate_pairwise_split(["a", "b", "b"], ["c", "b"])
        self.assertEqual(result["overlap_count"], 1)
        self.assertFalse(bool(result["is_disjoint"]))

    def test_21_invalid_probabilities(self) -> None:
        with self.assertRaises(ValueError):
            brier_score([1.2], [1.0])
        with self.assertRaises(ValueError):
            aggregate_swapped_pairwise(-0.1, 0.5)

    def test_22_invalid_ranking(self) -> None:
        with self.assertRaises(ValueError):
            plackett_luce_nll([0, 1, 2], [0, 0, 2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
