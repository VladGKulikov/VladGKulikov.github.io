from __future__ import annotations

import math
import unittest
import numpy as np
import module_8_reference as m8


class Module8ReferenceTests(unittest.TestCase):
    def test_softmax_and_log_softmax(self):
        p = m8.softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(float(p.sum()), 1.0)
        np.testing.assert_allclose(np.exp(m8.log_softmax([1.0, 2.0, 3.0])), p)

    def test_categorical_kl(self):
        self.assertAlmostEqual(m8.categorical_kl([0.5, 0.5], [0.5, 0.5]), 0.0)
        self.assertAlmostEqual(m8.categorical_kl([1.0, 0.0], [0.5, 0.5]), math.log(2.0))

    def test_kl_infinite_when_support_is_missing(self):
        self.assertTrue(math.isinf(m8.categorical_kl([0.5, 0.5], [1.0, 0.0])))

    def test_optimal_policy_known_case(self):
        p = m8.optimal_kl_policy([1/3, 1/3, 1/3], [1.0, 0.0, -1.0], 1.0)
        np.testing.assert_allclose(p, [0.66524096, 0.24472847, 0.09003057], rtol=1e-7)

    def test_optimal_policy_preserves_reference_support(self):
        p = m8.optimal_kl_policy([0.7, 0.3, 0.0], [0.0, 1.0, 100.0], 0.5)
        self.assertEqual(float(p[2]), 0.0)
        self.assertAlmostEqual(float(p.sum()), 1.0)

    def test_reward_shift_invariance(self):
        self.assertTrue(m8.reward_shift_invariant([0.6, 0.3, 0.1], [1.2, -0.4, 0.8], 0.7, 17.0))

    def test_closed_form_beats_reference_objective(self):
        ref = np.array([0.5, 0.3, 0.2]); r = np.array([0.0, 1.0, -0.5])
        p = m8.optimal_kl_policy(ref, r, 0.8)
        self.assertGreaterEqual(m8.regularized_objective(p, r, ref, 0.8), m8.regularized_objective(ref, r, ref, 0.8))

    def test_sequence_logprob_and_ratio_mask(self):
        mask = [0, 1, 1, 0]
        self.assertAlmostEqual(m8.sequence_logprob([-0.2, -1.0, -2.0, -4.0], mask), -3.0)
        self.assertAlmostEqual(m8.sampled_sequence_log_ratio([-0.2, -1.0, -2.0, -4.0], [-0.2, -1.4, -1.8, -4.0], mask), 0.2)

    def test_token_reward_shaping_identity(self):
        p=np.array([-1.0,-0.5,-2.0]); q=np.array([-1.2,-0.5,-1.5])
        shaped=m8.shape_token_rewards(p,q,terminal_reward=1.0,beta=0.1)
        np.testing.assert_allclose(shaped,[-0.02,0.0,1.05])
        self.assertAlmostEqual(float(shaped.sum()),1.0-0.1*float((p-q).sum()))

    def test_dpo_margin(self):
        self.assertAlmostEqual(m8.dpo_margin(-2.0,-3.0,-2.2,-2.4),0.8)

    def test_dpo_loss_and_weight(self):
        self.assertAlmostEqual(m8.dpo_loss_from_margin(1.0,1.0),math.log1p(math.exp(-1.0)))
        self.assertAlmostEqual(m8.dpo_gradient_weight(1.0,1.0),1.0/(1.0+math.e))

    def test_cDPO_reduces_to_dpo_at_zero_smoothing(self):
        a=m8.conservative_dpo_loss(np.array([-1.0,0.0,2.0]),0.3,0.0)
        b=m8.dpo_loss_from_margin(np.array([-1.0,0.0,2.0]),0.3)
        np.testing.assert_allclose(a,b)

    def test_cDPO_symmetric_at_half_limit(self):
        z=4.0
        a=0.5*m8.softplus(-z)+0.5*m8.softplus(z)
        b=0.5*m8.softplus(z)+0.5*m8.softplus(-z)
        self.assertAlmostEqual(float(a),float(b))

    def test_ipo_target(self):
        tau=0.25; target=1.0/(2.0*tau)
        self.assertAlmostEqual(m8.ipo_loss(target,tau),0.0)
        self.assertGreater(m8.ipo_loss(target+1.0,tau),0.0)

    def test_simpo_reward_uses_mean_logprob(self):
        self.assertAlmostEqual(m8.simpo_reward(-10.0,5,2.0),-4.0)
        self.assertAlmostEqual(m8.simpo_reward(-4.0,2,2.0),-4.0)

    def test_simpo_loss(self):
        self.assertAlmostEqual(m8.simpo_loss(-4.0,2,-6.0,3,beta=1.0,target_margin=0.5),float(m8.softplus(0.5)))

    def test_log_odds(self):
        p=0.2
        self.assertAlmostEqual(m8.log_odds_from_logprob(math.log(p)),math.log(p/(1-p)))

    def test_orpo_loss(self):
        loss=m8.orpo_preference_loss(math.log(0.4),math.log(0.2))
        expected=math.log((0.4/0.6)/(0.2/0.8))
        self.assertAlmostEqual(loss,math.log1p(math.exp(-expected)))

    def test_constrained_lagrangian(self):
        value = m8.constrained_lagrangian(0.7, [0.30, 0.10], [0.25, 0.15], [0.4, 0.2])
        self.assertAlmostEqual(value, 0.7 - 0.4 * 0.05 + 0.2 * 0.05)

    def test_dual_ascent_step(self):
        updated = m8.dual_ascent_step([0.2, 0.1], [0.6, 0.0], [0.25, 0.2], 0.5)
        np.testing.assert_allclose(updated, [0.375, 0.0])

    def test_two_action_constrained_optimum(self):
        q = m8.two_action_constrained_optimum(0.6, 1.0, 0.0, 1.0, 0.25)
        self.assertAlmostEqual(q, 0.25)
        self.assertAlmostEqual(m8.two_action_constrained_optimum(1.0, 0.6, 0.0, 1.0, 0.25), 0.0)
        with self.assertRaises(ValueError):
            m8.two_action_constrained_optimum(0.6, 1.0, 0.2, 1.0, 0.1)

    def test_pair_diagnostics(self):
        d=m8.pair_diagnostics(-2.0,-3.0,-2.2,-2.4,0.5)
        self.assertAlmostEqual(d.margin,0.8); self.assertAlmostEqual(d.dpo_logit,0.4); self.assertGreater(d.gradient_weight,0.0)

    def test_likelihood_displacement_exact_margin(self):
        for t in [0.0,0.2,1.0,3.0]:
            p=m8.likelihood_displacement_probabilities(t)
            self.assertAlmostEqual(math.log(p[0])-math.log(p[1]),2.0*t,places=10)

    def test_likelihood_displacement_chosen_falls(self):
        p0=m8.likelihood_displacement_probabilities(0.0); p1=m8.likelihood_displacement_probabilities(1.0)
        self.assertLess(p1[0],p0[0]); self.assertGreater(p1[2],p0[2])

    def test_reward_kl_frontier(self):
        result=m8.reward_kl_frontier([0.5,0.3,0.2],[0.0,1.0,-0.5],[4.0,1.0,0.25])
        self.assertEqual(result["policy"].shape,(3,3)); self.assertTrue(np.all(np.diff(result["reward"])>0)); self.assertTrue(np.all(np.diff(result["kl"])>0))

    def test_summarize_pairs(self):
        out=m8.summarize_pairs([-2.0,-1.0],[-3.0,-1.5],[-2.2,-1.1],[-2.8,-1.4],beta=0.5)
        self.assertEqual(out["pair_accuracy"],1.0); self.assertIn("mean_chosen_logp",out); self.assertGreater(out["mean_gradient_weight"],0.0)

    def test_validation_errors(self):
        with self.assertRaises(ValueError): m8.optimal_kl_policy([0.5,0.5],[1.0,0.0],0.0)
        with self.assertRaises(ValueError): m8.ipo_loss(1.0,0.0)
        with self.assertRaises(ValueError): m8.log_odds_from_logprob(0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
