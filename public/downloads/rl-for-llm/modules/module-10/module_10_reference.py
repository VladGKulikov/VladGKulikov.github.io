#!/usr/bin/env python3
"""Deterministic reference calculations for RL for LLM, Module 10.

The functions in this file are deliberately small enough to audit line by
line.  They support the lecture, the closed assessment banks, the bilingual
notebooks, and the release figures.  Results produced by synthetic experiments
are course-owned demonstrations, not external model benchmarks.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, inf, isfinite, lgamma, log, sqrt
from typing import Iterable, Iterator, Literal, Sequence

import numpy as np


FallbackMode = Literal["rejected_pool", "fresh_unverified"]


def _probability(value: float, name: str = "p") -> float:
    value = float(value)
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return value


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def pass_at_n(p: float, n: int) -> float:
    """Coverage under ``n`` conditionally independent Bernoulli trials."""
    p = _probability(p)
    n = _positive_int(n, "n")
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0
    return float(-np.expm1(n * np.log1p(-p)))


def pass_curve(p: float, budgets: Iterable[int]) -> np.ndarray:
    budgets = list(budgets)
    if not budgets:
        raise ValueError("budgets must be non-empty")
    return np.asarray([pass_at_n(p, n) for n in budgets], dtype=np.float64)


def aggregate_pass_curve(ps: Sequence[float], budgets: Iterable[int]) -> np.ndarray:
    probabilities = np.asarray(ps, dtype=np.float64)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("ps must be a non-empty one-dimensional sequence")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("every probability must lie in [0, 1]")
    budgets = list(budgets)
    if not budgets:
        raise ValueError("budgets must be non-empty")
    return np.asarray(
        [np.mean([pass_at_n(float(p), int(n)) for p in probabilities]) for n in budgets],
        dtype=np.float64,
    )


def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """Finite-sample pass@k estimator used in code-generation evaluation."""
    n = _positive_int(n, "n")
    k = _positive_int(k, "k")
    c = _nonnegative_int(c, "c")
    if c > n:
        raise ValueError("c must not exceed n")
    if k > n:
        raise ValueError("k cannot exceed n")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    # Product form avoids large binomial coefficients.
    failure = 1.0
    for j in range(k):
        failure *= (n - c - j) / (n - j)
    return float(1.0 - failure)


def _compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    """Yield all weak compositions of ``total`` into ``parts`` entries."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for tail in _compositions(total - first, parts - 1):
            yield (first, *tail)


def plurality_accuracy_exact(
    probabilities: Sequence[float],
    k: int,
    *,
    correct_index: int = 0,
) -> float:
    """Exact plurality accuracy with uniform random tie-breaking.

    The implementation sums the multinomial mass over all count vectors.  It
    is intended for teaching-sized category counts and budgets (for example,
    five categories with ``k=16`` or three categories with ``k=64``).
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1 or probs.size < 2:
        raise ValueError("probabilities must describe at least two answers")
    if not np.isfinite(probs).all() or np.any(probs < 0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.isclose(probs.sum(), 1.0, atol=1e-12):
        raise ValueError("probabilities must sum to one")
    k = _positive_int(k, "k")
    correct_index = int(correct_index)
    if not 0 <= correct_index < probs.size:
        raise ValueError("correct_index is out of range")

    log_k_factorial = lgamma(k + 1)
    total = 0.0
    for counts in _compositions(k, int(probs.size)):
        max_count = max(counts)
        winners = [i for i, value in enumerate(counts) if value == max_count]
        if correct_index not in winners:
            continue
        # Zero-probability categories can only have zero count.
        impossible = any(count > 0 and probs[i] == 0.0 for i, count in enumerate(counts))
        if impossible:
            continue
        log_mass = log_k_factorial - sum(lgamma(count + 1) for count in counts)
        for i, count in enumerate(counts):
            if count:
                log_mass += count * log(float(probs[i]))
        total += exp(log_mass) / len(winners)
    return float(total)


def plurality_accuracy_mc(
    probabilities: Sequence[float],
    k: int,
    *,
    correct_index: int = 0,
    trials: int = 50_000,
    seed: int = 10,
) -> float:
    """Monte Carlo plurality accuracy with uniform random tie-breaking."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 1 or probabilities.size < 2:
        raise ValueError("probabilities must describe at least two answers")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-12):
        raise ValueError("probabilities must sum to one")
    k = _positive_int(k, "k")
    trials = _positive_int(trials, "trials")
    correct_index = int(correct_index)
    if not 0 <= correct_index < probabilities.size:
        raise ValueError("correct_index is out of range")
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(k, probabilities, size=trials)
    maxima = counts.max(axis=1, keepdims=True)
    winner_mask = counts == maxima
    tie_sizes = winner_mask.sum(axis=1)
    # Conditional on a count vector, uniform tie-breaking credits the correct
    # category by 1 / number_of_tied_winners.  Vectorizing this expectation
    # keeps the diagnostic inexpensive even inside the repository-wide suite.
    credit = winner_mask[:, correct_index] / tie_sizes
    return float(np.mean(credit))


def canonicalize_short_answer(text: str) -> str:
    """Narrow canonicalizer for short-answer teaching examples."""
    normalized = " ".join(str(text).strip().lower().split())
    return normalized[:-1] if normalized.endswith(".") else normalized


@dataclass(frozen=True)
class BinaryVerifierResult:
    n: int
    fallback_mode: FallbackMode
    accept_probability: float
    accepted_precision: float
    rejected_precision: float
    no_accept_probability: float
    fallback_success: float
    success_probability: float


def binary_verifier_success(
    n: int,
    p_correct: float,
    true_positive_rate: float,
    false_positive_rate: float,
    *,
    fallback_mode: FallbackMode = "rejected_pool",
) -> BinaryVerifierResult:
    """Success probability of a fully specified binary-verifier protocol.

    Candidates are iid.  If at least one candidate is accepted, the system
    samples uniformly from the accepted candidates.  When none is accepted:

    ``rejected_pool``
        sample uniformly from the same candidates, all of which are known to
        have been rejected.  Their conditional correctness is generally *not*
        the original base rate;

    ``fresh_unverified``
        draw a fresh independent candidate without running the verifier.  Its
        success probability is the original base rate ``p_correct``.
    """
    n = _positive_int(n, "n")
    p = _probability(p_correct, "p_correct")
    tpr = _probability(true_positive_rate, "true_positive_rate")
    fpr = _probability(false_positive_rate, "false_positive_rate")
    if fallback_mode not in ("rejected_pool", "fresh_unverified"):
        raise ValueError("fallback_mode must be 'rejected_pool' or 'fresh_unverified'")

    q = p * tpr + (1.0 - p) * fpr
    no_accept = (1.0 - q) ** n
    accepted_precision = p if q == 0.0 else p * tpr / q
    rejected_mass = 1.0 - q
    rejected_precision = p if rejected_mass == 0.0 else p * (1.0 - tpr) / rejected_mass
    fallback_success = rejected_precision if fallback_mode == "rejected_pool" else p
    success = (1.0 - no_accept) * accepted_precision + no_accept * fallback_success
    return BinaryVerifierResult(
        n=n,
        fallback_mode=fallback_mode,
        accept_probability=float(q),
        accepted_precision=float(accepted_precision),
        rejected_precision=float(rejected_precision),
        no_accept_probability=float(no_accept),
        fallback_success=float(fallback_success),
        success_probability=float(success),
    )


def binary_verifier_limit(p_correct: float, true_positive_rate: float, false_positive_rate: float) -> float:
    """Large-N limit when the verifier accepts with positive probability."""
    p = _probability(p_correct, "p_correct")
    tpr = _probability(true_positive_rate, "true_positive_rate")
    fpr = _probability(false_positive_rate, "false_positive_rate")
    q = p * tpr + (1.0 - p) * fpr
    return float(p if q == 0.0 else p * tpr / q)


def process_score(step_scores: Sequence[float], method: str) -> float:
    """Aggregate step scores with a named teaching rule."""
    values = np.asarray(step_scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("step_scores must be a non-empty one-dimensional sequence")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("step scores must lie in [0, 1]")
    method = str(method).lower()
    if method == "min":
        return float(np.min(values))
    if method == "mean":
        return float(np.mean(values))
    if method == "product":
        return float(np.prod(values))
    if method == "last":
        return float(values[-1])
    if method == "log_product":
        if np.any(values == 0):
            return float(-inf)
        return float(np.log(values).sum())
    raise ValueError("method must be one of: min, mean, product, last, log_product")


def prefix_success_estimate(successes: int, rollouts: int, *, smoothing: float = 0.0) -> float:
    """Monte Carlo prefix-value estimate with optional symmetric smoothing."""
    successes = _nonnegative_int(successes, "successes")
    rollouts = _positive_int(rollouts, "rollouts")
    smoothing = float(smoothing)
    if successes > rollouts:
        raise ValueError("successes must not exceed rollouts")
    if not isfinite(smoothing) or smoothing < 0:
        raise ValueError("smoothing must be finite and non-negative")
    return float((successes + smoothing) / (rollouts + 2.0 * smoothing))


def wilson_interval(successes: int, trials: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a Bernoulli proportion."""
    successes = _nonnegative_int(successes, "successes")
    trials = _positive_int(trials, "trials")
    z = float(z)
    if successes > trials:
        raise ValueError("successes must not exceed trials")
    if not isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    phat = successes / trials
    denom = 1.0 + z * z / trials
    centre = (phat + z * z / (2.0 * trials)) / denom
    radius = z * sqrt(phat * (1.0 - phat) / trials + z * z / (4.0 * trials * trials)) / denom
    return float(max(0.0, centre - radius)), float(min(1.0, centre + radius))


def uct_score(mean_value: float, parent_visits: int, child_visits: int, exploration: float) -> float:
    """Classical UCT-style optimistic score for a visited child."""
    mean_value = float(mean_value)
    parent_visits = _positive_int(parent_visits, "parent_visits")
    child_visits = _positive_int(child_visits, "child_visits")
    exploration = float(exploration)
    if not isfinite(mean_value) or not isfinite(exploration) or exploration < 0:
        raise ValueError("mean_value must be finite and exploration non-negative")
    return float(mean_value + exploration * sqrt(log(parent_visits) / child_visits))


def proxy_overoptimization_curve(
    budgets: Sequence[int],
    *,
    trials: int = 10_000,
    seed: int = 202_610,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproducible best-of-N proxy-selection experiment."""
    budgets_array = np.asarray(budgets, dtype=np.int64)
    if budgets_array.ndim != 1 or budgets_array.size == 0 or np.any(budgets_array <= 0):
        raise ValueError("budgets must be positive integers")
    trials = _positive_int(trials, "trials")
    max_budget = int(budgets_array.max())
    rng = np.random.default_rng(seed)
    useful = rng.normal(size=(trials, max_budget))
    spurious = rng.normal(size=(trials, max_budget))
    proxy = useful + spurious
    goal = useful - np.maximum(spurious - 1.0, 0.0) ** 2
    rows = np.arange(trials)
    proxy_means: list[float] = []
    goal_means: list[float] = []
    for n in budgets_array:
        selected = np.argmax(proxy[:, : int(n)], axis=1)
        proxy_means.append(float(proxy[rows, selected].mean()))
        goal_means.append(float(goal[rows, selected].mean()))
    return np.asarray(proxy_means), np.asarray(goal_means)


def marginal_pass_gain(p: float, current_samples: int) -> float:
    """Gain from adding sample ``current_samples + 1`` to an iid pass curve."""
    p = _probability(p)
    current_samples = _nonnegative_int(current_samples, "current_samples")
    return float(p * (1.0 - p) ** current_samples)


def allocate_budget_greedy(
    ps: Sequence[float],
    total_samples: int,
    *,
    minimum_per_prompt: int = 1,
) -> np.ndarray:
    """Allocate samples by the largest current marginal pass@N gain."""
    probabilities = np.asarray(ps, dtype=np.float64)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("ps must be a non-empty one-dimensional sequence")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    minimum = _nonnegative_int(minimum_per_prompt, "minimum_per_prompt")
    total_samples = _positive_int(total_samples, "total_samples")
    required = minimum * probabilities.size
    if total_samples < required:
        raise ValueError("total_samples is smaller than the required minimum")
    allocation = np.full(probabilities.size, minimum, dtype=np.int64)
    for _ in range(total_samples - required):
        gains = probabilities * np.power(1.0 - probabilities, allocation)
        allocation[int(np.argmax(gains))] += 1
    return allocation


def mean_coverage(ps: Sequence[float], allocation: Sequence[int]) -> float:
    probabilities = np.asarray(ps, dtype=np.float64)
    counts = np.asarray(allocation, dtype=np.int64)
    if probabilities.shape != counts.shape or probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("ps and allocation must be non-empty vectors of the same shape")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    if np.any(counts < 0):
        raise ValueError("allocation must be non-negative")
    return float(np.mean(1.0 - np.power(1.0 - probabilities, counts)))


def pareto_frontier(quality: Sequence[float], cost: Sequence[float]) -> np.ndarray:
    """Return a Boolean mask of non-dominated points (maximize quality, minimize cost)."""
    q = np.asarray(quality, dtype=np.float64)
    c = np.asarray(cost, dtype=np.float64)
    if q.shape != c.shape or q.ndim != 1 or q.size == 0:
        raise ValueError("quality and cost must be non-empty vectors of equal shape")
    if not np.isfinite(q).all() or not np.isfinite(c).all() or np.any(c < 0):
        raise ValueError("quality must be finite; cost finite and non-negative")
    keep = np.ones(q.size, dtype=bool)
    for i in range(q.size):
        dominates_i = (q >= q[i]) & (c <= c[i]) & ((q > q[i]) | (c < c[i]))
        if np.any(dominates_i):
            keep[i] = False
    return keep


def break_even_queries(fixed_training_cost: float, old_cost_per_query: float, new_cost_per_query: float) -> float:
    """Query count after which a fixed investment pays back through savings."""
    fixed = float(fixed_training_cost)
    old = float(old_cost_per_query)
    new = float(new_cost_per_query)
    if not all(isfinite(x) and x >= 0 for x in (fixed, old, new)):
        raise ValueError("costs must be finite and non-negative")
    saving = old - new
    if saving <= 0:
        return inf
    return float(fixed / saving)


def cost_regularized_utility(quality: float, cost: float, penalty: float) -> float:
    quality = float(quality)
    cost = float(cost)
    penalty = float(penalty)
    if not all(isfinite(x) for x in (quality, cost, penalty)) or cost < 0 or penalty < 0:
        raise ValueError("quality must be finite; cost and penalty must be finite and non-negative")
    return float(quality - penalty * cost)
