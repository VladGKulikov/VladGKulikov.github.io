#!/usr/bin/env python3
"""Unit tests for Module 1 reference implementations."""
from __future__ import annotations

import math
import unittest

import numpy as np

from module_1_reference import (
    categorical_entropy,
    corridor_standard_error,
    discounted_return,
    estimate_corridor_value,
    exact_corridor_value,
    policy_entropy,
    run_corridor_episode,
    softmax_with_temperature,
    temperature_entropy_curve,
)


class DiscountedReturnTests(unittest.TestCase):
    def test_known_example(self) -> None:
        self.assertAlmostEqual(discounted_return([1, 0, 0, 10], 0.9), 8.29, places=12)

    def test_boundary_discounts(self) -> None:
        self.assertEqual(discounted_return([3, 4, 5], 0.0), 3.0)
        self.assertEqual(discounted_return([3, 4, 5], 1.0), 12.0)

    def test_input_validation(self) -> None:
        for bad_gamma in (-0.1, 1.1, math.inf, math.nan):
            with self.subTest(gamma=bad_gamma), self.assertRaises(ValueError):
                discounted_return([1], bad_gamma)
        for rewards in ([], [[1, 2]], [1, math.nan]):
            with self.subTest(rewards=rewards), self.assertRaises(ValueError):
                discounted_return(rewards, 0.9)


class SoftmaxAndEntropyTests(unittest.TestCase):
    def test_probabilities_are_finite_and_normalized(self) -> None:
        probabilities = softmax_with_temperature([1000, 999, 998], 0.5)
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=14)
        self.assertTrue((probabilities > 0).all())

    def test_shift_invariance(self) -> None:
        first = softmax_with_temperature([2, 1, 0], 1.7)
        second = softmax_with_temperature([1002, 1001, 1000], 1.7)
        np.testing.assert_allclose(first, second, rtol=0, atol=2e-14)

    def test_entropy_known_values(self) -> None:
        self.assertAlmostEqual(categorical_entropy([0.5, 0.5]), math.log(2), places=12)
        self.assertAlmostEqual(policy_entropy([2, 1, 0, -1], 1.0), 0.9475369639754255, places=12)

    def test_entropy_monotone_for_fixed_logits(self) -> None:
        grid = np.geomspace(0.1, 10.0, 100)
        values = temperature_entropy_curve([2, 1, 0, -1], grid)
        self.assertTrue(np.all(np.diff(values) >= -1e-12))
        self.assertLessEqual(float(values[-1]), math.log(4) + 1e-12)

    def test_input_validation(self) -> None:
        for temperature in (0, -1, math.nan, math.inf):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                softmax_with_temperature([1, 2], temperature)
        for logits in ([], [[1, 2]], [1, math.nan]):
            with self.subTest(logits=logits), self.assertRaises(ValueError):
                softmax_with_temperature(logits, 1.0)
        for probabilities in ([0.2, 0.2], [-0.1, 1.1], [math.nan, 1.0]):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                categorical_entropy(probabilities)


class CorridorTests(unittest.TestCase):
    def test_exact_values(self) -> None:
        self.assertEqual(exact_corridor_value(max_steps=None), 0.5)
        self.assertAlmostEqual(exact_corridor_value(max_steps=20), 0.49951171875, places=14)
        self.assertAlmostEqual(exact_corridor_value(max_steps=2), 0.25, places=14)

    def test_seeded_monte_carlo_is_reproducible(self) -> None:
        first = estimate_corridor_value(1000, seed=7, max_steps=20)
        second = estimate_corridor_value(1000, seed=7, max_steps=20)
        self.assertEqual(first, second)
        self.assertEqual(first, 0.494)

    def test_monte_carlo_is_statistically_reasonable(self) -> None:
        n_episodes = 20_000
        estimate = estimate_corridor_value(n_episodes, seed=23, max_steps=20)
        exact = exact_corridor_value(max_steps=20)
        stderr = corridor_standard_error(n_episodes, max_steps=20)
        self.assertLess(abs(estimate - exact), 4.0 * stderr)

    def test_episode_output_and_validation(self) -> None:
        rng = np.random.default_rng(0)
        self.assertIn(run_corridor_episode(rng, max_steps=20), (0.0, 1.0))
        for max_steps in (0, -1, 1.5, True):
            with self.subTest(max_steps=max_steps), self.assertRaises(ValueError):
                run_corridor_episode(np.random.default_rng(0), max_steps=max_steps)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            run_corridor_episode("not a generator")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            estimate_corridor_value(0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
