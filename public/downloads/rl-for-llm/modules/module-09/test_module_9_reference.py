"""Deterministic tests for Module 9 reference implementations."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

import module_9_reference as m9  # noqa: E402


class VerifierTests(unittest.TestCase):
    def test_extracts_last_boxed_answer(self) -> None:
        self.assertEqual(m9.extract_numeric_answer(r"first \\boxed{6}, final \\boxed{42}"), "42")

    def test_fallback_uses_last_fraction(self) -> None:
        self.assertEqual(m9.extract_numeric_answer("try 1, final 3 / 7"), "3 / 7")

    def test_fraction_matches_decimal_with_tolerance(self) -> None:
        self.assertEqual(m9.verify_numeric_answer(r"\\boxed{3/7}", "0.428571"), 1)

    def test_wrong_or_invalid_answer_is_rejected(self) -> None:
        self.assertEqual(m9.verify_numeric_answer("42 items minus 1 = 41", "42"), 0)
        self.assertEqual(m9.verify_numeric_answer(r"\\boxed{1/0}", "0"), 0)
        self.assertEqual(m9.verify_numeric_answer("no numeric answer", "1"), 0)


class GroupAdvantageTests(unittest.TestCase):
    def test_unique_success_has_sqrt_three_advantage(self) -> None:
        a = m9.group_relative_advantages([1, 0, 0, 0])
        self.assertTrue(np.allclose(a, [math.sqrt(3), -1 / math.sqrt(3), -1 / math.sqrt(3), -1 / math.sqrt(3)], atol=1e-7))

    def test_unscaled_advantages_are_centered(self) -> None:
        a = m9.group_relative_advantages([0.9, 1.0, 0.8, 0.9], scale=False)
        self.assertTrue(np.allclose(a, [0.0, 0.1, -0.1, 0.0]))
        self.assertAlmostEqual(float(a.sum()), 0.0)

    def test_degenerate_group_returns_zero(self) -> None:
        self.assertTrue(np.array_equal(m9.group_relative_advantages([0.5, 0.5, 0.5]), np.zeros(3)))

    def test_invalid_group_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            m9.group_relative_advantages([])
        with self.assertRaises(ValueError):
            m9.group_relative_advantages([0, np.nan])

    def test_informative_probability_at_half(self) -> None:
        self.assertAlmostEqual(m9.informative_group_probability(0.5, 8), 1 - 2 / 256)

    def test_informative_probability_is_symmetric(self) -> None:
        for p in (0.01, 0.1, 0.2, 0.7):
            self.assertAlmostEqual(m9.informative_group_probability(p, 8), m9.informative_group_probability(1 - p, 8))

    def test_expected_draws_handles_zero_signal(self) -> None:
        self.assertTrue(math.isinf(m9.expected_draws_for_informative_groups(8, 0.0, 4)))
        self.assertAlmostEqual(m9.expected_draws_for_informative_groups(8, 0.5, 4), 8 / 0.875)


class ClippingTests(unittest.TestCase):
    def test_symmetric_clipped_terms(self) -> None:
        out = m9.clipped_surrogate_terms([1.5, 0.7, 1.1], [1.0, -1.0, -1.0], eps_low=0.2)
        self.assertTrue(np.allclose(out, [1.2, -0.8, -1.1]))

    def test_clip_higher_releases_positive_ratio(self) -> None:
        sym = m9.clipped_surrogate_terms([1.25], [1.0], eps_low=0.2)[0]
        asym = m9.clipped_surrogate_terms([1.25], [1.0], eps_low=0.2, eps_high=0.28)[0]
        self.assertAlmostEqual(sym, 1.2)
        self.assertAlmostEqual(asym, 1.25)

    def test_clip_activity_uses_advantage_sign(self) -> None:
        mask = m9.clip_activity([1.3, 0.7, 1.3, 0.7], [1.0, 1.0, -1.0, -1.0], eps_low=0.2)
        self.assertTrue(np.array_equal(mask, [True, False, False, True]))


class ReductionTests(unittest.TestCase):
    def test_original_grpo_equalizes_response_contributions(self) -> None:
        rewards = [1.0, 0.0]
        weights = m9.original_grpo_token_weights(rewards, [10, 1000])
        contrib = m9.response_contributions(weights)
        self.assertTrue(np.allclose(contrib, [0.5, -0.5], atol=1e-7))
        self.assertAlmostEqual(weights[0][0] / abs(weights[1][0]), 100.0, places=5)

    def test_dr_grpo_response_weight_is_proportional_to_length(self) -> None:
        weights = m9.dr_grpo_token_weights([1.0, 0.0], [10, 100], fixed_length=100)
        contrib = m9.response_contributions(weights)
        self.assertTrue(np.allclose(contrib, [0.025, -0.25]))

    def test_dapo_uses_one_token_normalizer(self) -> None:
        weights = m9.dapo_token_weights([1.0, 0.0], [10, 100], scale_rewards=False)
        self.assertAlmostEqual(weights[0][0], 0.5 / 110)
        self.assertAlmostEqual(weights[1][0], -0.5 / 110)
        contrib = m9.response_contributions(weights)
        self.assertTrue(np.allclose(contrib, [5 / 110, -50 / 110]))

    def test_invalid_lengths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            m9.original_grpo_token_weights([1, 0], [10, 0])
        with self.assertRaises(ValueError):
            m9.dr_grpo_token_weights([1, 0], [10, 20], fixed_length=0)


class SequenceRatioTests(unittest.TestCase):
    def test_constant_token_ratio_gives_same_sequence_ratio(self) -> None:
        old = np.log([0.2, 0.4, 0.5])
        new = old + math.log(1.1)
        self.assertAlmostEqual(m9.sequence_importance_ratio(new, old), 1.1)

    def test_mask_excludes_padding(self) -> None:
        old = np.log([0.2, 0.4, 0.5])
        new = old + [math.log(1.2), math.log(1.2), math.log(10.0)]
        self.assertAlmostEqual(m9.sequence_importance_ratio(new, old, [1, 1, 0]), 1.2)

    def test_gspo_clips_one_sequence_ratio(self) -> None:
        old = np.log([0.2, 0.4])
        new = old + math.log(1.001)
        term = m9.gspo_surrogate_term(new, old, 2.0, eps_low=0.0003, eps_high=0.0004)
        self.assertAlmostEqual(term, 2.0008, places=7)

    def test_empty_sequence_mask_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            m9.sequence_importance_ratio([-1, -2], [-1, -2], [0, 0])


class LengthEntropyAndRolloutTests(unittest.TestCase):
    def test_soft_overlong_penalty(self) -> None:
        self.assertEqual(m9.soft_overlong_penalty(80, 100, 20), 0.0)
        self.assertAlmostEqual(m9.soft_overlong_penalty(90, 100, 20), -0.5)
        self.assertEqual(m9.soft_overlong_penalty(100, 100, 20), -1.0)
        self.assertEqual(m9.soft_overlong_penalty(120, 100, 20), -1.0)

    def test_entropy_of_two_equal_modes(self) -> None:
        self.assertAlmostEqual(m9.categorical_entropy([0.5, 0.5, 0.0]), math.log(2))

    def test_iid_pass_at_k(self) -> None:
        self.assertAlmostEqual(m9.pass_at_k_independent(0.2, 4), 1 - 0.8**4)

    def test_valid_rollout_record(self) -> None:
        record = m9.RolloutRecord(
            prompt_id="p1",
            group_id="g1",
            token_ids=(10, 11, 12),
            action_mask=(0, 1, 1),
            old_logprobs=(-1.0, -0.2, -0.3),
            reward_components=(1.0, 0.2),
            stop_reason="eos",
            generator_version="checkpoint-17",
        )
        m9.validate_rollout_record(record)

    def test_invalid_rollout_record(self) -> None:
        record = m9.RolloutRecord("p", "g", (1,), (0,), (-1.0,), (1.0,), "eos", "v")
        with self.assertRaises(ValueError):
            m9.validate_rollout_record(record)


class SyntheticTrainingTests(unittest.TestCase):
    def test_no_entropy_bonus_concentrates_on_best_mode(self) -> None:
        result = m9.synthetic_mode_training(beta_entropy=0.0, learning_rate=0.5, steps=2000)
        p = result["probabilities"]
        assert isinstance(p, np.ndarray)
        self.assertGreater(p[0], 0.98)
        self.assertLess(p[1], 0.02)
        self.assertGreater(result["successful_mass"], 0.999)

    def test_entropy_bonus_preserves_second_successful_mode(self) -> None:
        result = m9.synthetic_mode_training(beta_entropy=0.05, learning_rate=0.5, steps=2000)
        p = result["probabilities"]
        assert isinstance(p, np.ndarray)
        self.assertGreater(p[1], 0.35)
        entropies = result["entropies"]
        assert isinstance(entropies, np.ndarray)
        self.assertGreater(entropies[-1], 0.65)

    def test_softmax_is_numerically_stable(self) -> None:
        p = m9.softmax([1000, 1001, 999])
        self.assertAlmostEqual(float(p.sum()), 1.0)
        self.assertTrue(np.all(np.isfinite(p)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
