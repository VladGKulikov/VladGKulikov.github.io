#!/usr/bin/env python3
"""Deterministic tests for Module 3 reference code."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from module_3_reference import (  # noqa: E402
    best_of_n_success,
    beta_bernoulli_posterior,
    categorical_entropy,
    centered_group_advantages,
    doubly_robust_value,
    entropy_regularized_policy,
    incremental_mean,
    ips_value,
    linucb_score,
    linucb_statistics,
    minimum_samples_for_target,
    pass_at_k,
    pseudo_regret_curve,
    run_epsilon_greedy,
    run_thompson_bernoulli,
    run_ucb1,
    snips_value,
    stable_softmax,
    summarize_algorithm,
    simulate_many_bandits,
    summarize_many,
)


def assert_close(actual, expected, *, atol=1e-10):
    if not np.allclose(actual, expected, atol=atol, rtol=0.0):
        raise AssertionError(f"actual={actual!r}, expected={expected!r}")


def run_tests() -> None:
    tests = []

    def test(name):
        def decorator(fn):
            tests.append((name, fn))
            return fn
        return decorator

    @test("incremental mean")
    def _():
        assert_close(incremental_mean(2.0, 4, 3.0), 2.2)

    @test("pseudo-regret decomposition")
    def _():
        assert_close(pseudo_regret_curve([0.2, 0.5, 0.7], [0, 2, 1]), [0.5, 0.5, 0.7])

    @test("epsilon-greedy deterministic seed")
    def _():
        run = run_epsilon_greedy([0.1, 0.4, 0.6], 40, 0.1, 17)
        assert run.actions.shape == (40,)
        assert int(run.counts.sum()) == 40
        assert_close(run.pseudo_regret[-1], 8.6)
        assert_close(run.rewards.sum(), 15.0)

    @test("UCB1 tries every arm first")
    def _():
        run = run_ucb1([0.1, 0.4, 0.6], 3, 5)
        assert np.array_equal(run.actions, [0, 1, 2])
        assert np.array_equal(run.counts, [1, 1, 1])

    @test("UCB1 deterministic checkpoint")
    def _():
        run = run_ucb1([0.1, 0.4, 0.6], 100, 23)
        assert_close(run.pseudo_regret[-1], 9.4)
        assert np.array_equal(run.counts, [10, 22, 68])

    @test("Thompson deterministic checkpoint")
    def _():
        run = run_thompson_bernoulli([0.1, 0.4, 0.6], 100, 23)
        assert_close(run.pseudo_regret[-1], 6.6)
        assert np.array_equal(run.counts, [10, 8, 82])

    @test("Monte Carlo summary shapes")
    def _():
        summary = summarize_algorithm(run_ucb1, [0.1, 0.4, 0.6], 20, np.arange(4))
        assert summary.mean_regret.shape == (20,)
        assert np.all(summary.upper_95 >= summary.lower_95)
        assert np.all((summary.mean_optimal_rate >= 0) & (summary.mean_optimal_rate <= 1))

    @test("vectorized Monte Carlo experiment")
    def _():
        result = simulate_many_bandits([0.1, 0.4, 0.6], 25, 8, 2026, "ucb1")
        summary = summarize_many(result)
        assert result.pseudo_regret.shape == (8, 25)
        assert summary.mean_regret.shape == (25,)
        assert np.all(np.diff(result.pseudo_regret, axis=1) >= -1e-12)

    @test("Beta-Bernoulli posterior")
    def _():
        assert beta_bernoulli_posterior(7, 3) == (8.0, 4.0)

    @test("LinUCB statistics")
    def _():
        x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y = np.array([1.0, 0.0, 1.0])
        theta, a, a_inv = linucb_statistics(x, y, ridge=1.0)
        assert_close(a, [[3.0, 1.0], [1.0, 3.0]])
        assert_close(theta, [0.625, 0.125])
        assert_close(a @ a_inv, np.eye(2))

    @test("LinUCB score")
    def _():
        score = linucb_score([1.0, 2.0], [0.4, -0.1], np.eye(2), 0.5)
        assert_close(score, 0.2 + 0.5 * math.sqrt(5.0))

    @test("IPS value")
    def _():
        actual = ips_value([1.0, 0.0, 1.0], [0.5, 0.25, 0.5], [1.0, 0.0, 0.5])
        assert_close(actual, 1.0)

    @test("SNIPS value")
    def _():
        actual = snips_value([1.0, 0.0, 1.0], [0.5, 0.25, 0.5], [1.0, 0.0, 0.5])
        assert_close(actual, 1.0)

    @test("doubly robust correction")
    def _():
        actual = doubly_robust_value(
            [1.0, 0.0], [0.5, 0.5], [1.0, 0.0], [0.6, 0.4], [0.7, 0.2]
        )
        assert_close(actual, 0.85)

    @test("stable softmax")
    def _():
        p = stable_softmax([1000.0, 1001.0])
        assert_close(p.sum(), 1.0)
        assert_close(p[1] / p[0], math.e)

    @test("entropy uniform distribution")
    def _():
        assert_close(categorical_entropy([0.25] * 4), math.log(4.0))

    @test("entropy-regularized optimum")
    def _():
        p = entropy_regularized_policy([0.0, 1.0], 0.5)
        assert_close(p, stable_softmax([0.0, 1.0], 0.5))
        q = entropy_regularized_policy([0.0, 1.0], 0.5, [0.8, 0.2])
        assert_close(q.sum(), 1.0)
        assert q[1] > q[0]

    @test("best-of-N success")
    def _():
        assert_close(best_of_n_success(0.2, 5), 1.0 - 0.8**5)
        assert minimum_samples_for_target(0.2, 0.9) == 11

    @test("pass@k estimator")
    def _():
        assert_close(pass_at_k(10, 2, 3), 1.0 - math.comb(8, 3) / math.comb(10, 3))
        assert pass_at_k(10, 9, 2) == 1.0

    @test("equal group rewards give zero advantages")
    def _():
        assert_close(centered_group_advantages([1.0, 1.0, 1.0]), [0.0, 0.0, 0.0])
        advantages = centered_group_advantages([0.0, 1.0, 2.0])
        assert_close(advantages.mean(), 0.0)
        assert_close(np.mean(advantages**2), 1.0)

    @test("input validation")
    def _():
        cases = [
            lambda: run_ucb1([0.2], 10, 0),
            lambda: ips_value([1], [0], [1]),
            lambda: stable_softmax([0, 1], 0),
            lambda: pass_at_k(2, 3, 1),
        ]
        for fn in cases:
            try:
                fn()
            except ValueError:
                pass
            else:
                raise AssertionError("expected ValueError")

    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as exc:  # noqa: BLE001 - test harness should report all failures
            failures.append((name, exc))
            print(f"[FAIL] {name}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} tests passed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    run_tests()
