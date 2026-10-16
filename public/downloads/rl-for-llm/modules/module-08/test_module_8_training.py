#!/usr/bin/env python3
"""Deterministic tests for the small autoregressive DPO workflow in Module 8."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from module_8_training import (
    TinyCausalLM,
    build_vocabulary,
    completion_sequence_logprobs,
    dpo_loss_from_logprobs,
    encode_prompt_completion,
    pair_batch,
    run_toy_dpo_experiment,
    selected_token_logprobs,
    toy_preference_examples,
)


class Module8TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.examples = toy_preference_examples()
        cls.vocab = build_vocabulary(cls.examples)
        cls.batch = pair_batch(cls.examples, cls.vocab)

    def test_01_pair_batch_shapes_and_masks(self) -> None:
        b = self.batch
        self.assertEqual(b["chosen_ids"].ndim, 2)
        self.assertEqual(b["chosen_mask"].shape[1], b["chosen_ids"].shape[1] - 1)
        self.assertEqual(b["rejected_mask"].shape[1], b["rejected_ids"].shape[1] - 1)
        for i, row in enumerate(self.examples):
            self.assertEqual(int(b["chosen_mask"][i].sum()), len(row.chosen) + 1)
            self.assertEqual(int(b["rejected_mask"][i].sum()), len(row.rejected) + 1)

    def test_02_prompt_length_contract(self) -> None:
        row = self.examples[0]
        seq, prompt_len = encode_prompt_completion(self.vocab, row.prompt, row.chosen)
        self.assertEqual(prompt_len, len(row.prompt) + 2)  # BOS + prompt + SEP
        self.assertEqual(len(seq), prompt_len + len(row.chosen) + 1)  # + EOS

    def test_03_selected_logprob_shape(self) -> None:
        torch.manual_seed(3)
        model = TinyCausalLM(len(self.vocab.id_to_token), hidden_size=12)
        ids = self.batch["chosen_ids"]
        selected = selected_token_logprobs(model, ids)
        self.assertEqual(selected.shape, (ids.shape[0], ids.shape[1] - 1))
        self.assertTrue(torch.isfinite(selected).all())

    def test_04_completion_logprob_uses_mask(self) -> None:
        torch.manual_seed(4)
        model = TinyCausalLM(len(self.vocab.id_to_token), hidden_size=12)
        value = completion_sequence_logprobs(
            model, self.batch["chosen_ids"], self.batch["chosen_mask"]
        )
        self.assertEqual(value.shape, (len(self.examples),))
        self.assertTrue(torch.isfinite(value).all())

    def test_05_dpo_loss_known_case(self) -> None:
        pc = torch.tensor([-1.0, -2.0])
        pr = torch.tensor([-2.0, -2.5])
        rc = torch.tensor([-1.2, -2.2])
        rr = torch.tensor([-1.8, -2.4])
        loss, margin, logits = dpo_loss_from_logprobs(pc, pr, rc, rr, beta=0.5)
        expected_margin = (pc - rc) - (pr - rr)
        torch.testing.assert_close(margin, expected_margin)
        torch.testing.assert_close(logits, 0.5 * expected_margin)
        torch.testing.assert_close(loss, torch.nn.functional.softplus(-logits).mean())

    def test_06_smoke_experiment_diagnostics(self) -> None:
        result = run_toy_dpo_experiment(profile="smoke", seed=7)
        self.assertEqual(result["after"]["pair_accuracy"], 1.0)
        self.assertEqual(result["stress"]["pair_accuracy"], 1.0)
        self.assertEqual(result["policy_generation_match"], 1.0)
        self.assertLess(result["stress_generation_match"], result["policy_generation_match"])
        self.assertLess(
            result["stress"]["mean_chosen_logp"],
            result["after"]["mean_chosen_logp"],
        )
        self.assertGreater(result["teacher_forced_kl_chosen"], 0.0)
        self.assertGreater(result["teacher_forced_kl_rejected"], 0.0)

    def test_07_smoke_experiment_is_reproducible(self) -> None:
        a = run_toy_dpo_experiment(profile="smoke", seed=7)
        b = run_toy_dpo_experiment(profile="smoke", seed=7)
        for section in ("before", "after", "stress"):
            self.assertEqual(a[section], b[section])
        self.assertEqual(a["policy_generation_match"], b["policy_generation_match"])
        self.assertEqual(a["stress_generations"], b["stress_generations"])

    def test_08_invalid_profile(self) -> None:
        with self.assertRaises(ValueError):
            run_toy_dpo_experiment(profile="invalid", seed=7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
