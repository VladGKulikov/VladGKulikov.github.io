#!/usr/bin/env python3
"""Unit tests for Module 2 reference implementations."""
from __future__ import annotations

import unittest

import numpy as np

from module_2_reference import (
    CliffEnv,
    advantage,
    bellman_expectation_residual,
    build_frozen_lake_4x4,
    build_gridworld_4x4,
    double_q_target,
    exact_corridor_values,
    exact_policy_value,
    greedy_cliff_rollout,
    greedy_policy,
    n_step_td_prediction_corridor,
    iterative_policy_evaluation,
    monte_carlo_prediction_corridor,
    n_step_target,
    policy_induced_model,
    q_from_v,
    q_learning_target,
    sarsa_target,
    td0_prediction_corridor,
    td_target,
    train_cliff_double_q,
    train_cliff_control,
    v_from_q,
    value_iteration,
    value_iteration_trace,
)


class ValueObjectTests(unittest.TestCase):
    def test_v_and_advantage(self) -> None:
        q = np.asarray([[1.0, 3.0], [0.0, 2.0]])
        pi = np.asarray([[0.5, 0.5], [0.2, 0.8]])
        np.testing.assert_allclose(v_from_q(q, pi), [2.0, 1.6])
        expected = np.asarray([[-1.0, 1.0], [-1.6, 0.4]])
        np.testing.assert_allclose(advantage(q, pi), expected)
        np.testing.assert_allclose(np.sum(pi * expected, axis=1), 0.0, atol=1e-14)

    def test_policy_validation(self) -> None:
        with self.assertRaises(ValueError):
            v_from_q([[1.0, 2.0]], [[0.2, 0.2]])
        with self.assertRaises(ValueError):
            v_from_q([[1.0, 2.0]], [[-0.1, 1.1]])


class BellmanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.p, self.r, self.terminals = build_gridworld_4x4()
        self.pi = np.full((16, 4), 0.25)

    def test_exact_gridworld(self) -> None:
        values = exact_policy_value(self.p, self.r, self.pi, 1.0, self.terminals)
        expected = np.asarray(
            [0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0],
            dtype=np.float64,
        )
        np.testing.assert_allclose(values, expected, rtol=0, atol=2e-12)
        np.testing.assert_allclose(
            bellman_expectation_residual(self.p, self.r, self.pi, values, 1.0, self.terminals),
            0.0,
            atol=2e-12,
        )

    def test_iterative_matches_exact(self) -> None:
        exact = exact_policy_value(self.p, self.r, self.pi, 0.9, self.terminals)
        iterative, sweeps = iterative_policy_evaluation(
            self.p,
            self.r,
            self.pi,
            0.9,
            self.terminals,
            tolerance=1e-12,
        )
        self.assertGreater(sweeps, 1)
        np.testing.assert_allclose(iterative, exact, atol=1e-10, rtol=0)

    def test_policy_induced_model_shapes(self) -> None:
        p_pi, r_pi, terminal_mask = policy_induced_model(self.p, self.r, self.pi, self.terminals)
        self.assertEqual(p_pi.shape, (16, 16))
        self.assertEqual(r_pi.shape, (16,))
        self.assertEqual(terminal_mask.shape, (16,))
        np.testing.assert_allclose(p_pi.sum(axis=1), 1.0)

    def test_q_from_v_terminal_values_are_zero(self) -> None:
        q = q_from_v(self.p, self.r, np.zeros(16), 1.0, self.terminals)
        np.testing.assert_allclose(q[self.terminals], 0.0)


class FrozenLakeTests(unittest.TestCase):
    def test_value_iteration_reference(self) -> None:
        p, r, terminals = build_frozen_lake_4x4()
        values, sweeps = value_iteration(p, r, 0.9, terminals, tolerance=1e-10)
        self.assertEqual(sweeps, 145)
        expected = np.asarray(
            [0.0689, 0.0614, 0.0744, 0.0558, 0.0919, 0.0, 0.1122, 0.0,
             0.1454, 0.2475, 0.2996, 0.0, 0.0, 0.3799, 0.6390, 0.0]
        )
        np.testing.assert_allclose(values, expected, atol=5e-5, rtol=0)
        policy = greedy_policy(p, r, values, 0.9, terminals)
        np.testing.assert_array_equal(policy, [0, 3, 0, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0])

    def test_transition_rows_sum_to_one(self) -> None:
        p, _, _ = build_frozen_lake_4x4()
        np.testing.assert_allclose(p.sum(axis=2), 1.0, atol=1e-14)

    def test_value_iteration_trace_matches_final_solution(self) -> None:
        p, r, terminals = build_frozen_lake_4x4()
        trace, sweeps = value_iteration_trace(p, r, 0.9, terminals, tolerance=1e-10)
        values, reference_sweeps = value_iteration(p, r, 0.9, terminals, tolerance=1e-10)
        self.assertEqual(sweeps, reference_sweeps)
        self.assertEqual(trace.shape, (sweeps + 1, 16))
        np.testing.assert_allclose(trace[0], 0.0)
        np.testing.assert_allclose(trace[-1], values, atol=0.0, rtol=0.0)
        self.assertAlmostEqual(float(trace[1, 14]), 1.0 / 3.0, places=14)


class TargetTests(unittest.TestCase):
    def test_n_step_target(self) -> None:
        self.assertAlmostEqual(n_step_target([1.0, 2.0, 3.0], 4.0, 0.5), 3.25, places=14)
        self.assertAlmostEqual(
            n_step_target([1.0, 2.0, 3.0], 4.0, 0.5, terminated_at_end=True),
            2.75,
            places=14,
        )

    def test_termination_and_truncation_masks(self) -> None:
        self.assertEqual(td_target(1.0, 100.0, 0.9, terminated=True), 1.0)
        self.assertEqual(td_target(1.0, float("nan"), 0.9, terminated=True), 1.0)
        self.assertEqual(td_target(1.0, 2.0, 0.9, terminated=False), 2.8)
        self.assertEqual(td_target(1.0, 2.0, 0.9, terminated=False, truncated=True), 2.8)
        self.assertEqual(td_target(1.0, 2.0, 0.9, terminated=True, truncated=True), 1.0)
        self.assertEqual(n_step_target([1.0, 2.0], float("nan"), 0.5, terminated_at_end=True), 2.0)
        self.assertEqual(q_learning_target(1.0, [], 0.9, terminated=True), 1.0)
        self.assertEqual(double_q_target(1.0, [], [], 0.9, terminated=True), 1.0)

    def test_q_learning_and_sarsa_targets(self) -> None:
        self.assertAlmostEqual(q_learning_target(1.0, [0.2, 0.5], 0.9, terminated=False), 1.45)
        self.assertAlmostEqual(sarsa_target(1.0, 0.2, 0.9, terminated=False), 1.18)

    def test_double_q_decouples_selection_and_evaluation(self) -> None:
        ordinary = q_learning_target(0.0, [10.0, 9.0], 1.0, terminated=False)
        doubled = double_q_target(0.0, [10.0, 9.0], [1.0, 8.0], 1.0, terminated=False)
        self.assertEqual(ordinary, 10.0)
        self.assertEqual(doubled, 1.0)


class PredictionTests(unittest.TestCase):
    def test_exact_corridor_values(self) -> None:
        np.testing.assert_allclose(exact_corridor_values(), [0.0, 0.25, 0.5, 0.75, 0.0])

    def test_monte_carlo_reproducible(self) -> None:
        first = monte_carlo_prediction_corridor(500, seed=7)
        second = monte_carlo_prediction_corridor(500, seed=7)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first, [0.0, 0.24615385, 0.51, 0.75, 0.0], atol=1e-8)

    def test_td_reproducible(self) -> None:
        values = td0_prediction_corridor(500, seed=42, alpha=0.1)
        np.testing.assert_allclose(values, [0.0, 0.15674152, 0.42727120, 0.70767648, 0.0], atol=1e-8)

    def test_prediction_is_reasonable(self) -> None:
        truth = exact_corridor_values()
        mc = monte_carlo_prediction_corridor(10_000, seed=11)
        td = td0_prediction_corridor(10_000, seed=11, alpha=0.03)
        self.assertLess(float(np.sqrt(np.mean((mc[1:4] - truth[1:4]) ** 2))), 0.03)
        self.assertLess(float(np.sqrt(np.mean((td[1:4] - truth[1:4]) ** 2))), 0.08)

    def test_three_step_td_is_reproducible(self) -> None:
        values = n_step_td_prediction_corridor(
            500,
            n_steps=3,
            seed=42,
            alpha=0.1,
        )
        np.testing.assert_allclose(
            values,
            [0.0, 0.18132571, 0.51475210, 0.71873893, 0.0],
            atol=1e-8,
        )


class ControlTests(unittest.TestCase):
    def test_cliff_environment(self) -> None:
        env = CliffEnv()
        self.assertEqual(env.reset(), 36)
        state, reward, terminated = env.step(1)
        self.assertEqual((state, reward, terminated), (36, -100.0, False))

    def test_q_learning_reference(self) -> None:
        q, returns = train_cliff_control(
            "q_learning", seed=0, n_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1
        )
        greedy_return, path = greedy_cliff_rollout(q)
        self.assertEqual(greedy_return, -13.0)
        self.assertAlmostEqual(float(returns.mean()), -54.72, places=2)
        self.assertEqual(path[-1], 47)

    def test_sarsa_reference(self) -> None:
        q, returns = train_cliff_control(
            "sarsa", seed=0, n_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1
        )
        greedy_return, path = greedy_cliff_rollout(q)
        self.assertEqual(greedy_return, -17.0)
        self.assertAlmostEqual(float(returns.mean()), -33.66, places=2)
        self.assertEqual(path[-1], 47)

    def test_double_q_reference(self) -> None:
        q_a, q_b, returns = train_cliff_double_q(
            seed=0,
            n_episodes=500,
            alpha=0.5,
            gamma=1.0,
            epsilon=0.1,
        )
        greedy_return, path = greedy_cliff_rollout(q_a + q_b)
        self.assertEqual(greedy_return, -17.0)
        self.assertAlmostEqual(float(returns.mean()), -38.204, places=3)
        self.assertEqual(path[-1], 47)


if __name__ == "__main__":
    unittest.main(verbosity=2)
