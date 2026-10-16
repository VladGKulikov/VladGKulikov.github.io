"""Tests for the autoregressive RLVR/GRPO teaching implementation in Module 9."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import math
import sys
import unittest

import numpy as np
import torch

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

import module_9_training as m9t  # noqa: E402


class GrammarAndVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prompts = m9t.arithmetic_prompts()
        cls.vocab = m9t.build_vocabulary(cls.prompts)
        cls.prompt = cls.prompts[0]

    def test_vocabulary_and_prompt_contract(self) -> None:
        prefix = m9t.prompt_prefix(self.vocab, self.prompt)
        self.assertEqual(self.vocab.decode(prefix), [m9t.BOS, "add", "0", "1", m9t.SEP])
        self.assertEqual(m9t.allowed_next_tokens(self.vocab, self.prompt, []), (m9t.DIRECT, m9t.WORK))

    def test_verifier_decomposes_correct_and_wrong_responses(self) -> None:
        correct = m9t.verify_response(
            self.prompt,
            [m9t.WORK, "0", m9t.PLUS, "1", m9t.EQUALS, "1", m9t.ANS_OPEN, "1", m9t.ANS_CLOSE, m9t.EOS],
        )
        wrong = m9t.verify_response(
            self.prompt,
            [m9t.DIRECT, m9t.ANS_OPEN, "2", m9t.ANS_CLOSE, m9t.EOS],
        )
        self.assertEqual((correct.accuracy, correct.consistency, correct.format_ok), (1.0, 1.0, 1.0))
        self.assertAlmostEqual(correct.reward, 1.1)
        self.assertEqual(wrong.accuracy, 0.0)
        self.assertEqual(wrong.reward, 0.0)

    def test_masked_policy_has_exact_support(self) -> None:
        logits = torch.arange(len(self.vocab.id_to_token), dtype=torch.float32)
        allowed = self.vocab.encode([m9t.DIRECT, m9t.WORK])
        logp = m9t.masked_log_probs(logits, allowed)
        self.assertTrue(torch.isfinite(logp[allowed]).all())
        forbidden = [i for i in range(len(logp)) if i not in allowed]
        self.assertTrue(torch.isneginf(logp[forbidden]).all())
        self.assertAlmostEqual(float(logp[allowed].exp().sum()), 1.0, places=6)


class PolicyMechanicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(17)
        cls.prompts = m9t.arithmetic_prompts()
        cls.vocab = m9t.build_vocabulary(cls.prompts)
        cls.prompt = cls.prompts[1]
        cls.model = m9t.TinyCausalLM(len(cls.vocab.id_to_token), hidden_size=16).eval()

    def test_sampled_old_logprobs_reproduce_under_frozen_policy(self) -> None:
        generator = torch.Generator().manual_seed(123)
        row = m9t.generate_response(
            self.model,
            self.vocab,
            self.prompt,
            generator=generator,
            group_id="g",
            generator_version="v1",
        )
        logp, entropy = m9t.response_logprobs_and_entropy(
            self.model, self.vocab, self.prompt, row.response_tokens
        )
        self.assertTrue(np.allclose(logp.detach().numpy(), row.old_logprobs, atol=1e-7))
        self.assertTrue(torch.isfinite(entropy).all())
        self.assertTrue(row.terminated)
        self.assertFalse(row.truncated)
        self.assertEqual(len(row.response_tokens), len(row.old_logprobs))

    def test_exact_kl_is_zero_for_identical_models(self) -> None:
        response = [m9t.DIRECT, m9t.ANS_OPEN, "3", m9t.ANS_CLOSE, m9t.EOS]
        kl = m9t.exact_teacher_forced_kl(
            self.model, self.model, self.vocab, self.prompt, response
        )
        self.assertAlmostEqual(float(kl.detach()), 0.0, places=7)

    def test_group_advantages_center_and_handle_degenerate_rows(self) -> None:
        rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]])
        advantage = m9t._group_advantages(rewards, True)
        self.assertTrue(torch.allclose(advantage.sum(dim=1), torch.zeros(2), atol=1e-6))
        self.assertTrue(torch.equal(advantage[1], torch.zeros(4)))

    def test_current_old_and_reference_roles_are_separate(self) -> None:
        reference = deepcopy(self.model).eval()
        policy = deepcopy(self.model).train()
        with torch.no_grad():
            next(policy.parameters()).add_(0.1)
        response = [m9t.DIRECT, m9t.ANS_OPEN, "3", m9t.ANS_CLOSE, m9t.EOS]
        p_logp, _ = m9t.response_logprobs_and_entropy(policy, self.vocab, self.prompt, response)
        q_logp, _ = m9t.response_logprobs_and_entropy(reference, self.vocab, self.prompt, response)
        self.assertFalse(torch.allclose(p_logp, q_logp))
        self.assertGreaterEqual(
            float(m9t.exact_teacher_forced_kl(policy, reference, self.vocab, self.prompt, response).detach()),
            -1e-7,
        )


class EndToEndExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = m9t.run_toy_grpo_experiment(profile="smoke", seed=9, loss_type="dapo")

    def test_experiment_is_structurally_complete(self) -> None:
        result = self.result
        self.assertEqual(result["profile"], "smoke")
        self.assertEqual(len(result["history"]), 7)
        for key in ("before_train", "after_train", "before_heldout", "after_heldout"):
            metrics = result[key]
            for value in metrics.values():
                self.assertTrue(math.isfinite(float(value)))

    def test_training_improves_logged_training_accuracy_but_not_heldout(self) -> None:
        result = self.result
        self.assertGreater(result["after_train"]["accuracy"], result["before_train"]["accuracy"])
        self.assertLessEqual(result["after_heldout"]["accuracy"], result["before_heldout"]["accuracy"])
        self.assertGreater(result["after_train"]["teacher_forced_kl"], 0.0)

    def test_smoke_metrics_match_the_registered_course_snapshot(self) -> None:
        expected = {
            "before_train": {"accuracy": 0.312500, "mean_reward": 0.340625, "mean_length": 7.031250, "entropy": 0.2915775, "teacher_forced_kl": 0.0, "direct_fraction": 0.593750, "pass_at_4": 0.5833333333333334},
            "after_train": {"accuracy": 0.4083333333333333, "mean_reward": 0.4483333333333333, "mean_length": 8.375000, "entropy": 0.1483265, "teacher_forced_kl": 0.2188581, "direct_fraction": 0.325000, "pass_at_4": 0.4500000000000000},
            "before_heldout": {"accuracy": 0.0333333333333333, "mean_reward": 0.0366666666666667, "mean_length": 6.916667, "entropy": 0.3816914, "teacher_forced_kl": 0.0, "direct_fraction": 0.616667, "pass_at_4": 0.1111111111111111},
            "after_heldout": {"accuracy": 0.0, "mean_reward": 0.0, "mean_length": 9.236111, "entropy": 0.2847782, "teacher_forced_kl": 0.1953977, "direct_fraction": 0.152778, "pass_at_4": 0.0},
        }
        for split, metrics in expected.items():
            for name, target in metrics.items():
                self.assertAlmostEqual(float(self.result[split][name]), target, places=5, msg=f"{split}.{name}")

    def test_entropy_and_prompt_coverage_diagnostics_expose_concentration(self) -> None:
        result = self.result
        self.assertLess(result["after_train"]["entropy"], result["before_train"]["entropy"])
        before = result["before_train_by_prompt"]
        after = result["after_train_by_prompt"]
        self.assertEqual(set(before), set(after))
        # The deterministic teaching run becomes perfect on some prompts and
        # loses sampled success on others: average reward alone hides this.
        self.assertGreaterEqual(sum(v == 1.0 for v in after.values()), 3)
        self.assertGreaterEqual(sum(v == 0.0 for v in after.values()), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
