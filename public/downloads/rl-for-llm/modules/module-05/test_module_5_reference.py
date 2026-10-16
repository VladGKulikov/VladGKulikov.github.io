#!/usr/bin/env python3
"""Deterministic tests for Module 5 reference code."""
from __future__ import annotations

import math
import unittest

import numpy as np

from module_5_reference import (
    advantage_weighted_nll,
    behavior_support_mask,
    cql_discrete_penalty,
    double_dqn_targets,
    dqn_targets,
    expectile_loss,
    fqe_targets,
    iql_advantage_weights,
    logsumexp,
    masked_behavior_cloning_nll,
    normalized_tightrope_regret,
    pessimistic_values,
    return_to_go,
    tightrope_regret,
    tightrope_return,
    update_target_return,
)


class TestModule5Reference(unittest.TestCase):
    def test_masked_bc_mean(self) -> None:
        lp = np.array([[-0.1, -0.2, -1.0], [-0.3, -0.7, -0.4]])
        mask = np.array([[0, 1, 1], [0, 0, 1]])
        self.assertAlmostEqual(masked_behavior_cloning_nll(lp, mask), 1.6 / 3.0)

    def test_masked_bc_sum_and_none(self) -> None:
        lp = np.array([-1.0, -2.0])
        m = np.array([1, 0])
        self.assertAlmostEqual(masked_behavior_cloning_nll(lp, m, reduction="sum"), 1.0)
        np.testing.assert_allclose(masked_behavior_cloning_nll(lp, m, reduction="none"), [1.0, 0.0])

    def test_masked_bc_empty_mean_rejected(self) -> None:
        with self.assertRaises(ValueError):
            masked_behavior_cloning_nll(np.zeros(2), np.zeros(2))

    def test_tightrope_zero_error(self) -> None:
        self.assertEqual(tightrope_return(0.0, 20), 20.0)
        self.assertEqual(normalized_tightrope_regret(0.0, 20), 0.0)

    def test_tightrope_known_case(self) -> None:
        self.assertAlmostEqual(tightrope_return(0.02, 20), 16.287209384000366)
        self.assertAlmostEqual(tightrope_regret(0.02, 20), 3.7127906159996336)
        self.assertAlmostEqual(normalized_tightrope_regret(0.02, 20), 0.4640988269999542)

    def test_tightrope_invalid(self) -> None:
        with self.assertRaises(ValueError):
            tightrope_return(-0.1, 3)
        with self.assertRaises(ValueError):
            tightrope_return(0.1, -1)

    def test_dqn_targets(self) -> None:
        result = dqn_targets(
            np.array([1.0, -2.0, 0.5]),
            np.array([0, 1, 0]),
            np.array([[2.0, 4.0], [100.0, 100.0], [-1.0, 3.0]]),
            gamma=0.9,
        )
        np.testing.assert_allclose(result, [4.6, -2.0, 3.2])

    def test_double_dqn_targets(self) -> None:
        rewards = np.array([0.0, 1.0])
        term = np.array([0, 0])
        online = np.array([[3.0, 5.0, 4.0], [2.0, 1.0, 0.0]])
        target = np.array([[10.0, 1.0, 2.0], [-1.0, 6.0, 7.0]])
        np.testing.assert_allclose(dqn_targets(rewards, term, target, gamma=0.5), [5.0, 4.5])
        np.testing.assert_allclose(double_dqn_targets(rewards, term, online, target, gamma=0.5), [0.5, 0.5])

    def test_terminal_masks_bootstrap(self) -> None:
        result = double_dqn_targets([3.0], [1], [[999.0]], [[999.0]], gamma=0.9)
        np.testing.assert_allclose(result, [3.0])

    def test_pessimistic_values(self) -> None:
        scores = pessimistic_values([0.8, 1.2, 0.5], [100, 4, 0], radius_scale=1.0)
        np.testing.assert_allclose(scores[:2], [0.7, 0.7])
        self.assertTrue(np.isneginf(scores[2]))

    def test_logsumexp(self) -> None:
        self.assertAlmostEqual(float(logsumexp(np.array([1000.0, 1000.0]))), 1000.0 + math.log(2.0))
        np.testing.assert_allclose(logsumexp(np.array([[0.0, 0.0], [1.0, 2.0]]), axis=1), [math.log(2.0), 2.313261687518223])

    def test_cql_penalty(self) -> None:
        expected = math.log(math.exp(2.0) + 1.0 + math.exp(-1.0)) - 2.0
        self.assertAlmostEqual(cql_discrete_penalty([2.0, 0.0, -1.0], 0), expected)

    def test_cql_batch(self) -> None:
        q = np.array([[1.0, 0.0], [0.0, 2.0]])
        expected = 0.5 * ((math.log(math.e + 1.0) - 1.0) + (math.log(1.0 + math.exp(2.0)) - 2.0))
        self.assertAlmostEqual(cql_discrete_penalty(q, [0, 1]), expected)

    def test_expectile(self) -> None:
        values = expectile_loss(np.array([-2.0, 1.0, 3.0]), 0.8, reduction="none")
        np.testing.assert_allclose(values, [0.8, 0.8, 7.2])
        self.assertAlmostEqual(expectile_loss([-2.0, 1.0, 3.0], 0.8), 2.9333333333333336)

    def test_expectile_tau_rejected(self) -> None:
        with self.assertRaises(ValueError):
            expectile_loss([1.0], 1.0)

    def test_iql_weights(self) -> None:
        np.testing.assert_allclose(iql_advantage_weights([-1.0, 0.0, 1.0], 2.0), np.exp([-2.0, 0.0, 2.0]))

    def test_iql_weight_clip(self) -> None:
        result = iql_advantage_weights([100.0], 1.0, max_weight=10.0)
        self.assertAlmostEqual(float(result[0]), 10.0)

    def test_behavior_support_mask(self) -> None:
        mask = behavior_support_mask([0.60, 0.25, 0.15], threshold_ratio=0.4)
        np.testing.assert_array_equal(mask, [True, True, False])
        batch = behavior_support_mask([[0.1, 0.9], [0.5, 0.5]], threshold_ratio=0.8)
        np.testing.assert_array_equal(batch, [[False, True], [True, True]])

    def test_fqe_targets(self) -> None:
        result = fqe_targets(
            [1.0, -1.0], [0, 1],
            [[2.0, 4.0], [100.0, 100.0]],
            [[0.25, 0.75], [0.5, 0.5]], gamma=0.5,
        )
        np.testing.assert_allclose(result, [2.75, -1.0])

    def test_advantage_weighted_nll(self) -> None:
        value = advantage_weighted_nll([-0.2, -1.0], [0.0, 1.0], 1.0)
        self.assertAlmostEqual(value, (0.2 + math.e) / 2.0)
        values = advantage_weighted_nll([-0.2], [100.0], 1.0, max_weight=10.0, reduction="none")
        np.testing.assert_allclose(values, [2.0])

    def test_return_to_go_undiscounted(self) -> None:
        np.testing.assert_allclose(return_to_go([1.0, 0.0, -2.0, 4.0]), [3.0, 2.0, 2.0, 4.0])

    def test_return_to_go_discounted(self) -> None:
        np.testing.assert_allclose(return_to_go([1.0, 2.0, 3.0], gamma=0.5), [2.75, 3.5, 3.0])

    def test_target_return_updates(self) -> None:
        target = 5.0
        sequence = []
        for reward in [1.0, 0.0, -2.0, 4.0]:
            target = update_target_return(target, reward)
            sequence.append(target)
        np.testing.assert_allclose(sequence, [4.0, 4.0, 6.0, 2.0])
        self.assertEqual(update_target_return(5.0, 1.0, gamma=0.5), 8.0)

    def test_zero_gamma_target_update_rejected(self) -> None:
        with self.assertRaises(ValueError):
            update_target_return(1.0, 0.0, gamma=0.0)

    def test_shape_validation(self) -> None:
        with self.assertRaises(ValueError):
            dqn_targets([1.0, 2.0], [0, 0], [[1.0, 2.0]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
