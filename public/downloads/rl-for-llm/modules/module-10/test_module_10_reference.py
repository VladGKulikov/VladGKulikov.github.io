#!/usr/bin/env python3
"""Deterministic tests for Module 10 reference calculations."""
from __future__ import annotations

from math import inf, isclose
from pathlib import Path
import sys

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from module_10_reference import (  # noqa: E402
    aggregate_pass_curve,
    allocate_budget_greedy,
    binary_verifier_limit,
    binary_verifier_success,
    break_even_queries,
    canonicalize_short_answer,
    cost_regularized_utility,
    marginal_pass_gain,
    mean_coverage,
    pareto_frontier,
    pass_at_n,
    plurality_accuracy_exact,
    plurality_accuracy_mc,
    prefix_success_estimate,
    process_score,
    proxy_overoptimization_curve,
    unbiased_pass_at_k,
    uct_score,
    wilson_interval,
)

PASSED = 0


def check(condition: bool, label: str) -> None:
    global PASSED
    if not condition:
        raise AssertionError(label)
    PASSED += 1
    print(f"[PASS] {label}")


def main() -> None:
    check(isclose(pass_at_n(0.5, 4), 0.9375), "pass@4 exact")
    check(pass_at_n(0.0, 100) == 0.0 and pass_at_n(1.0, 3) == 1.0, "pass curve endpoints")
    curve = aggregate_pass_curve([0.5, 0.2, 0.05, 0.9], [1, 2, 4, 8])
    check(np.allclose(curve, [0.4125, 0.549375, 0.6783234375, 0.7912247961]), "aggregate pass curve")
    check(isclose(unbiased_pass_at_k(200, 10, 5), 0.22828446073733433, rel_tol=1e-9), "finite-sample pass@k estimator")
    check(unbiased_pass_at_k(10, 0, 5) == 0.0 and unbiased_pass_at_k(10, 8, 5) == 1.0, "pass@k estimator boundaries")

    scattered = plurality_accuracy_exact([0.4, 0.15, 0.15, 0.15, 0.15], 16)
    systematic = plurality_accuracy_exact([0.3, 0.45, 0.25], 64)
    check(isclose(scattered, 0.794221578810956, rel_tol=1e-12), "exact plurality with scattered errors")
    check(isclose(systematic, 0.07707365945087288, rel_tol=1e-12), "exact plurality with a systematic wrong mode")
    mc = plurality_accuracy_mc([0.4, 0.15, 0.15, 0.15, 0.15], 16, trials=200_000, seed=10)
    check(abs(mc - scattered) < 0.004, "Monte Carlo plurality agrees with exact tie-aware result")
    check(canonicalize_short_answer("  Forty   Two. ") == "forty two", "narrow answer canonicalizer")

    rejected = binary_verifier_success(8, 0.2, 0.9, 0.1, fallback_mode="rejected_pool")
    fresh = binary_verifier_success(8, 0.2, 0.9, 0.1, fallback_mode="fresh_unverified")
    check(isclose(rejected.accept_probability, 0.26), "binary verifier accept probability")
    check(isclose(rejected.accepted_precision, 9 / 13), "accepted-candidate precision")
    check(isclose(rejected.rejected_precision, 0.02 / 0.74), "rejected-candidate conditional correctness")
    check(isclose(rejected.no_accept_probability, 0.74**8), "finite-N no-accept probability")
    check(isclose(rejected.success_probability, 0.632486004809728, rel_tol=1e-12), "fallback to rejected pool")
    check(isclose(fresh.success_probability, 0.6480396435591987, rel_tol=1e-12), "fresh unverified fallback")
    check(isclose(binary_verifier_limit(0.2, 0.9, 0.1), 9 / 13), "binary verifier limiting precision")

    check(isclose(process_score([0.95, 0.95, 0.30, 0.95], "mean"), 0.7875), "process-score mean")
    check(isclose(process_score([0.95, 0.95, 0.30, 0.95], "product"), 0.2572125), "process-score product")
    check(process_score([0.95, 0.95, 0.30, 0.95], "min") == 0.30, "process-score minimum")
    check(process_score([0.95, 0.95, 0.30, 0.95], "last") == 0.95, "process-score last")
    check(isclose(prefix_success_estimate(3, 5), 0.6), "unsmoothed prefix success estimate")
    check(isclose(prefix_success_estimate(0, 2, smoothing=1.0), 0.25), "smoothed prefix success estimate")
    lo, hi = wilson_interval(6, 8)
    check(isclose(lo, 0.40927543031016883) and isclose(hi, 0.9285207872478909), "Wilson interval")
    check(isclose(uct_score(0.6, 100, 10, 2**0.5), 1.5597051824376142, rel_tol=1e-10), "UCT score")

    budgets = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    proxy, goal = proxy_overoptimization_curve(budgets)
    check(np.all(np.diff(proxy) > 0), "proxy score rises monotonically in teaching experiment")
    check(int(np.argmax(goal)) == 5, "independent goal peaks at N=32 in teaching experiment")
    check(goal[-1] < goal[5], "independent goal falls under aggressive proxy selection")
    check(isclose(marginal_pass_gain(0.2, 3), 0.1024), "marginal pass gain")

    allocation = allocate_budget_greedy([0.02, 0.08, 0.2, 0.5], 16)
    check(np.array_equal(allocation, [1, 5, 6, 4]), "greedy adaptive allocation")
    adaptive = mean_coverage([0.02, 0.08, 0.2, 0.5], allocation)
    uniform = mean_coverage([0.02, 0.08, 0.2, 0.5], [4, 4, 4, 4])
    check(isclose(adaptive, 0.5090686192, rel_tol=1e-9) and adaptive > uniform, "adaptive allocation improves stated model")
    frontier = pareto_frontier([0.70, 0.80, 0.75, 0.85], [1.0, 2.0, 3.0, 4.0])
    check(np.array_equal(frontier, [True, True, False, True]), "Pareto frontier removes dominated mode")
    check(isclose(break_even_queries(12_000, 0.18, 0.06), 100_000), "finite break-even query count")
    check(break_even_queries(100, 0.1, 0.1) == inf, "no break-even without per-query saving")
    check(isclose(cost_regularized_utility(0.8, 4.0, 0.05), 0.6), "cost-regularized utility")

    invalid_calls = (
        lambda: pass_at_n(-0.1, 3),
        lambda: unbiased_pass_at_k(10, 11, 3),
        lambda: plurality_accuracy_exact([0.5, 0.6], 3),
        lambda: binary_verifier_success(3, 0.2, 0.9, 0.1, fallback_mode="unknown"),  # type: ignore[arg-type]
        lambda: process_score([], "mean"),
        lambda: allocate_budget_greedy([0.2, 1.2], 4),
        lambda: wilson_interval(11, 10),
        lambda: pareto_frontier([0.5], [-1.0]),
    )
    for call in invalid_calls:
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("invalid-input check did not fail")
    check(True, "invalid inputs raise ValueError")
    print(f"\n{PASSED} tests passed")


if __name__ == "__main__":
    main()
