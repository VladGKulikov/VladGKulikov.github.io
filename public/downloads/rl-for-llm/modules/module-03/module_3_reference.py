#!/usr/bin/env python3
"""Reference NumPy implementations for RL for LLM, Module 3.

The module intentionally uses small, inspectable algorithms. It is not a
production bandit library. All stochastic routines accept an explicit seed so
published examples and figures remain reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class BanditRun:
    """One trajectory from a stationary Bernoulli bandit."""

    actions: IntArray
    rewards: FloatArray
    estimates: FloatArray
    counts: IntArray
    pseudo_regret: FloatArray


@dataclass(frozen=True)
class BanditSummary:
    """Monte Carlo summary over several independent bandit trajectories."""

    mean_regret: FloatArray
    lower_95: FloatArray
    upper_95: FloatArray
    mean_optimal_rate: FloatArray


def _as_probability_vector(values: ArrayLike, *, name: str = "probabilities") -> FloatArray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"{name} must be a one-dimensional array with at least two entries")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return arr.astype(float, copy=True)


def incremental_mean(current: float, count_before: int, observation: float) -> float:
    """Update a sample mean after one additional observation."""
    if count_before < 0:
        raise ValueError("count_before must be non-negative")
    if not np.isfinite(current) or not np.isfinite(observation):
        raise ValueError("current and observation must be finite")
    return float(current + (observation - current) / (count_before + 1))


def pseudo_regret_curve(means: ArrayLike, actions: ArrayLike) -> FloatArray:
    """Cumulative pseudo-regret for a known stationary bandit instance."""
    mu = np.asarray(means, dtype=float)
    act = np.asarray(actions, dtype=int)
    if mu.ndim != 1 or mu.size < 2 or not np.all(np.isfinite(mu)):
        raise ValueError("means must be a finite one-dimensional array")
    if act.ndim != 1:
        raise ValueError("actions must be one-dimensional")
    if np.any((act < 0) | (act >= mu.size)):
        raise ValueError("actions contain an invalid arm index")
    gaps = float(mu.max()) - mu[act]
    return np.cumsum(gaps, dtype=float)


def _finish_run(
    probabilities: FloatArray,
    actions: IntArray,
    rewards: FloatArray,
    estimates: FloatArray,
    counts: IntArray,
) -> BanditRun:
    return BanditRun(
        actions=actions,
        rewards=rewards,
        estimates=estimates,
        counts=counts,
        pseudo_regret=pseudo_regret_curve(probabilities, actions),
    )


def run_epsilon_greedy(
    probabilities: ArrayLike,
    horizon: int,
    epsilon: float,
    seed: int,
    *,
    initial_value: float = 0.0,
) -> BanditRun:
    """Run constant-epsilon greedy on a stationary Bernoulli bandit.

    Ties are broken uniformly, avoiding the accidental preference for arm zero
    that a plain ``argmax`` would create under equal estimates.
    """
    probs = _as_probability_vector(probabilities)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must lie in [0, 1]")
    if not np.isfinite(initial_value):
        raise ValueError("initial_value must be finite")

    rng = np.random.default_rng(seed)
    k = probs.size
    estimates = np.full(k, float(initial_value))
    counts = np.zeros(k, dtype=np.int64)
    actions = np.empty(horizon, dtype=np.int64)
    rewards = np.empty(horizon, dtype=float)

    for t in range(horizon):
        if rng.random() < epsilon:
            arm = int(rng.integers(k))
        else:
            best = np.flatnonzero(np.isclose(estimates, estimates.max()))
            arm = int(rng.choice(best))
        reward = float(rng.random() < probs[arm])
        estimates[arm] = incremental_mean(estimates[arm], int(counts[arm]), reward)
        counts[arm] += 1
        actions[t] = arm
        rewards[t] = reward

    return _finish_run(probs, actions, rewards, estimates, counts)


def run_ucb1(probabilities: ArrayLike, horizon: int, seed: int) -> BanditRun:
    """Run the canonical UCB1 index for rewards in [0, 1]."""
    probs = _as_probability_vector(probabilities)
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    rng = np.random.default_rng(seed)
    k = probs.size
    estimates = np.zeros(k, dtype=float)
    counts = np.zeros(k, dtype=np.int64)
    actions = np.empty(horizon, dtype=np.int64)
    rewards = np.empty(horizon, dtype=float)

    for t in range(horizon):
        if t < k:
            arm = t
        else:
            bonus = np.sqrt(2.0 * np.log(t + 1.0) / counts)
            indices = estimates + bonus
            best = np.flatnonzero(np.isclose(indices, indices.max()))
            arm = int(rng.choice(best))
        reward = float(rng.random() < probs[arm])
        estimates[arm] = incremental_mean(estimates[arm], int(counts[arm]), reward)
        counts[arm] += 1
        actions[t] = arm
        rewards[t] = reward

    return _finish_run(probs, actions, rewards, estimates, counts)


def run_thompson_bernoulli(
    probabilities: ArrayLike,
    horizon: int,
    seed: int,
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> BanditRun:
    """Run Beta-Bernoulli Thompson sampling."""
    probs = _as_probability_vector(probabilities)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if prior_alpha <= 0.0 or prior_beta <= 0.0:
        raise ValueError("Beta prior parameters must be positive")

    rng = np.random.default_rng(seed)
    k = probs.size
    alpha = np.full(k, float(prior_alpha))
    beta = np.full(k, float(prior_beta))
    counts = np.zeros(k, dtype=np.int64)
    actions = np.empty(horizon, dtype=np.int64)
    rewards = np.empty(horizon, dtype=float)

    for t in range(horizon):
        samples = rng.beta(alpha, beta)
        best = np.flatnonzero(np.isclose(samples, samples.max()))
        arm = int(rng.choice(best))
        reward = float(rng.random() < probs[arm])
        alpha[arm] += reward
        beta[arm] += 1.0 - reward
        counts[arm] += 1
        actions[t] = arm
        rewards[t] = reward

    posterior_means = alpha / (alpha + beta)
    return _finish_run(probs, actions, rewards, posterior_means, counts)


BanditAlgorithm = Callable[[ArrayLike, int, int], BanditRun]


def summarize_algorithm(
    runner: BanditAlgorithm,
    probabilities: ArrayLike,
    horizon: int,
    seeds: ArrayLike,
) -> BanditSummary:
    """Aggregate pseudo-regret and optimal-arm rate over independent seeds."""
    probs = _as_probability_vector(probabilities)
    seed_array = np.asarray(seeds, dtype=np.int64)
    if seed_array.ndim != 1 or seed_array.size < 2:
        raise ValueError("seeds must be a one-dimensional array with at least two values")
    regrets = np.empty((seed_array.size, horizon), dtype=float)
    optimal = np.empty((seed_array.size, horizon), dtype=float)
    optimal_arm = int(np.argmax(probs))
    for row, seed in enumerate(seed_array):
        run = runner(probs, horizon, int(seed))
        regrets[row] = run.pseudo_regret
        optimal[row] = np.cumsum(run.actions == optimal_arm) / np.arange(1, horizon + 1)
    mean = regrets.mean(axis=0)
    stderr = regrets.std(axis=0, ddof=1) / np.sqrt(seed_array.size)
    half_width = 1.96 * stderr
    return BanditSummary(
        mean_regret=mean,
        lower_95=mean - half_width,
        upper_95=mean + half_width,
        mean_optimal_rate=optimal.mean(axis=0),
    )


def beta_bernoulli_posterior(
    successes: int,
    failures: int,
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> tuple[float, float]:
    """Posterior parameters after Bernoulli observations."""
    if successes < 0 or failures < 0:
        raise ValueError("successes and failures must be non-negative")
    if prior_alpha <= 0.0 or prior_beta <= 0.0:
        raise ValueError("prior parameters must be positive")
    return prior_alpha + successes, prior_beta + failures


def linucb_statistics(
    features: ArrayLike,
    rewards: ArrayLike,
    *,
    ridge: float = 1.0,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return ``(theta_hat, A, A_inv)`` for a linear reward model."""
    x = np.asarray(features, dtype=float)
    y = np.asarray(rewards, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("features must have shape (n, d) with n > 0")
    if y.shape != (x.shape[0],):
        raise ValueError("rewards must have shape (n,)")
    if ridge <= 0.0 or not np.isfinite(ridge):
        raise ValueError("ridge must be positive and finite")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("features and rewards must be finite")
    a = ridge * np.eye(x.shape[1]) + x.T @ x
    b = x.T @ y
    theta = np.linalg.solve(a, b)
    a_inv = np.linalg.inv(a)
    return theta, a, a_inv


def linucb_score(
    feature: ArrayLike,
    theta_hat: ArrayLike,
    a_inv: ArrayLike,
    alpha: float,
) -> float:
    """LinUCB mean estimate plus an elliptical uncertainty bonus."""
    phi = np.asarray(feature, dtype=float)
    theta = np.asarray(theta_hat, dtype=float)
    inverse = np.asarray(a_inv, dtype=float)
    if phi.ndim != 1 or theta.shape != phi.shape or inverse.shape != (phi.size, phi.size):
        raise ValueError("incompatible LinUCB shapes")
    if alpha < 0.0 or not np.isfinite(alpha):
        raise ValueError("alpha must be non-negative and finite")
    variance_proxy = float(phi @ inverse @ phi)
    if variance_proxy < -1e-12:
        raise ValueError("a_inv must define a non-negative quadratic form")
    return float(phi @ theta + alpha * np.sqrt(max(0.0, variance_proxy)))


def ips_value(
    rewards: ArrayLike,
    behavior_probabilities: ArrayLike,
    target_probabilities: ArrayLike,
) -> float:
    """Inverse-propensity estimate for logged contextual-bandit feedback.

    ``target_probabilities[i]`` is the target policy's probability of the
    action that was actually logged on row ``i``.
    """
    r = np.asarray(rewards, dtype=float)
    p = np.asarray(behavior_probabilities, dtype=float)
    q = np.asarray(target_probabilities, dtype=float)
    if r.ndim != 1 or p.shape != r.shape or q.shape != r.shape or r.size == 0:
        raise ValueError("all inputs must be non-empty one-dimensional arrays of equal length")
    if np.any(p <= 0.0) or np.any(p > 1.0):
        raise ValueError("behavior probabilities must lie in (0, 1]")
    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("target probabilities must lie in [0, 1]")
    if not np.all(np.isfinite(r)):
        raise ValueError("rewards must be finite")
    return float(np.mean(q * r / p))


def snips_value(
    rewards: ArrayLike,
    behavior_probabilities: ArrayLike,
    target_probabilities: ArrayLike,
) -> float:
    """Self-normalized IPS estimate; lower variance but generally biased."""
    r = np.asarray(rewards, dtype=float)
    p = np.asarray(behavior_probabilities, dtype=float)
    q = np.asarray(target_probabilities, dtype=float)
    if r.ndim != 1 or p.shape != r.shape or q.shape != r.shape or r.size == 0:
        raise ValueError("all inputs must be non-empty one-dimensional arrays of equal length")
    if np.any(p <= 0.0) or np.any(p > 1.0) or np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("invalid probabilities")
    weights = q / p
    denominator = float(weights.sum())
    if denominator <= 0.0:
        raise ValueError("target policy assigns zero total mass to logged actions")
    return float(np.dot(weights, r) / denominator)


def doubly_robust_value(
    rewards: ArrayLike,
    behavior_probabilities: ArrayLike,
    target_probabilities: ArrayLike,
    predictions_logged: ArrayLike,
    predictions_target: ArrayLike,
) -> float:
    """Doubly robust estimate from row-wise reward-model predictions.

    ``predictions_logged`` is ``r_hat(x_i, a_i)`` and
    ``predictions_target`` is ``sum_a pi(a|x_i) r_hat(x_i, a)``.
    """
    r = np.asarray(rewards, dtype=float)
    p = np.asarray(behavior_probabilities, dtype=float)
    q = np.asarray(target_probabilities, dtype=float)
    logged = np.asarray(predictions_logged, dtype=float)
    target = np.asarray(predictions_target, dtype=float)
    if r.ndim != 1 or any(arr.shape != r.shape for arr in (p, q, logged, target)) or r.size == 0:
        raise ValueError("all inputs must be non-empty one-dimensional arrays of equal length")
    if np.any(p <= 0.0) or np.any(p > 1.0) or np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("invalid probabilities")
    if not all(np.all(np.isfinite(arr)) for arr in (r, logged, target)):
        raise ValueError("rewards and predictions must be finite")
    correction = q / p * (r - logged)
    return float(np.mean(target + correction))


def stable_softmax(logits: ArrayLike, temperature: float = 1.0) -> FloatArray:
    """Temperature-scaled softmax with finite-input validation."""
    z = np.asarray(logits, dtype=float)
    if z.ndim != 1 or z.size == 0 or not np.all(np.isfinite(z)):
        raise ValueError("logits must be a non-empty finite vector")
    if temperature <= 0.0 or not np.isfinite(temperature):
        raise ValueError("temperature must be positive and finite")
    scaled = z / temperature
    shifted = scaled - scaled.max()
    weights = np.exp(shifted)
    return weights / weights.sum()


def categorical_entropy(probabilities: ArrayLike) -> float:
    """Categorical Shannon entropy in nats, with ``0 log 0 = 0``."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or p.size == 0 or not np.all(np.isfinite(p)):
        raise ValueError("probabilities must be a non-empty finite vector")
    if np.any(p < 0.0) or not np.isclose(p.sum(), 1.0, atol=1e-10):
        raise ValueError("probabilities must be non-negative and sum to one")
    positive = p[p > 0.0]
    return float(-np.dot(positive, np.log(positive)))


def entropy_regularized_policy(
    values: ArrayLike,
    temperature: float,
    reference: ArrayLike | None = None,
) -> FloatArray:
    """Exact optimizer of a one-step entropy/KL-regularized objective.

    With ``reference=None`` this returns ``softmax(values / temperature)``.
    With a reference distribution ``rho`` it returns a normalized
    ``rho * exp(values / temperature)`` and requires positive reference mass.
    """
    q = np.asarray(values, dtype=float)
    if q.ndim != 1 or q.size == 0 or not np.all(np.isfinite(q)):
        raise ValueError("values must be a non-empty finite vector")
    if temperature <= 0.0 or not np.isfinite(temperature):
        raise ValueError("temperature must be positive and finite")
    if reference is None:
        return stable_softmax(q, temperature)
    rho = np.asarray(reference, dtype=float)
    if rho.shape != q.shape or np.any(rho <= 0.0) or not np.isclose(rho.sum(), 1.0, atol=1e-10):
        raise ValueError("reference must be positive and sum to one")
    log_weights = np.log(rho) + q / temperature
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    return weights / weights.sum()


def best_of_n_success(success_probability: float, n: int) -> float:
    """Probability of at least one success in ``n`` independent samples."""
    if not 0.0 <= success_probability <= 1.0:
        raise ValueError("success_probability must lie in [0, 1]")
    if n < 1:
        raise ValueError("n must be positive")
    return float(-np.expm1(n * np.log1p(-success_probability))) if success_probability < 1.0 else 1.0


def minimum_samples_for_target(success_probability: float, target: float) -> int:
    """Smallest N with ``1 - (1-p)^N >= target`` under independence."""
    if not 0.0 < success_probability <= 1.0:
        raise ValueError("success_probability must lie in (0, 1]")
    if not 0.0 < target < 1.0:
        raise ValueError("target must lie in (0, 1)")
    if success_probability == 1.0:
        return 1
    return int(np.ceil(np.log1p(-target) / np.log1p(-success_probability)))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased finite-sample pass@k estimator from the Codex evaluation.

    ``n`` is the number of sampled programs and ``c`` the number that pass.
    The estimator is ``1 - C(n-c, k) / C(n, k)``.
    """
    if n <= 0 or c < 0 or c > n or k <= 0 or k > n:
        raise ValueError("require n > 0, 0 <= c <= n, and 1 <= k <= n")
    if n - c < k:
        return 1.0
    # Product form avoids very large intermediate binomial coefficients.
    failure = 1.0
    for j in range(n - c + 1, n + 1):
        failure *= 1.0 - k / j
    direct = 1.0 - comb(n - c, k) / comb(n, k)
    value = 1.0 - failure
    if not np.isclose(value, direct, atol=1e-12):
        raise ArithmeticError("stable and direct pass@k formulas disagree")
    return float(value)


def centered_group_advantages(rewards: ArrayLike, *, epsilon: float = 1e-8) -> FloatArray:
    """Mean-center and standardize group rewards; equal rewards map to zero."""
    r = np.asarray(rewards, dtype=float)
    if r.ndim != 1 or r.size == 0 or not np.all(np.isfinite(r)):
        raise ValueError("rewards must be a non-empty finite vector")
    if epsilon <= 0.0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be positive and finite")
    centered = r - r.mean()
    scale = float(np.sqrt(np.mean(centered**2)))
    if scale < epsilon:
        return np.zeros_like(centered)
    return centered / scale

@dataclass(frozen=True)
class ManyBanditRuns:
    """Run-level curves from a vectorized Monte Carlo experiment."""

    pseudo_regret: FloatArray
    optimal_rate: FloatArray


def _choose_random_tie(rng: np.random.Generator, scores: FloatArray) -> IntArray:
    """Uniformly choose one maximal column in every row of a score matrix."""
    maxima = scores.max(axis=1, keepdims=True)
    ties = np.isclose(scores, maxima, rtol=0.0, atol=1e-12)
    tie_breakers = rng.random(scores.shape)
    tie_breakers[~ties] = -1.0
    return np.argmax(tie_breakers, axis=1).astype(np.int64)


def simulate_many_bandits(
    probabilities: ArrayLike,
    horizon: int,
    n_runs: int,
    seed: int,
    algorithm: str,
    *,
    epsilon: float = 0.1,
) -> ManyBanditRuns:
    """Vectorized reproducible Bernoulli-bandit experiment.

    ``algorithm`` is one of ``"greedy"``, ``"epsilon_greedy"``, ``"ucb1"``,
    or ``"thompson"``. The function is designed for figures and confidence
    intervals; its random stream is intentionally independent of the simple
    single-trajectory implementations above.
    """
    probs = _as_probability_vector(probabilities)
    if horizon <= 0 or n_runs < 2:
        raise ValueError("horizon must be positive and n_runs must be at least two")
    if algorithm not in {"greedy", "epsilon_greedy", "ucb1", "thompson"}:
        raise ValueError("unknown algorithm")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must lie in [0, 1]")

    rng = np.random.default_rng(seed)
    k = probs.size
    estimates = np.zeros((n_runs, k), dtype=float)
    counts = np.zeros((n_runs, k), dtype=np.int64)
    alpha = np.ones((n_runs, k), dtype=float)
    beta = np.ones((n_runs, k), dtype=float)
    regret = np.empty((n_runs, horizon), dtype=float)
    optimal_rate = np.empty((n_runs, horizon), dtype=float)
    cumulative_regret = np.zeros(n_runs, dtype=float)
    cumulative_optimal = np.zeros(n_runs, dtype=float)
    optimal_arm = int(np.argmax(probs))
    rows = np.arange(n_runs)

    for t in range(horizon):
        if algorithm in {"greedy", "epsilon_greedy"}:
            arms = _choose_random_tie(rng, estimates)
            exploration = epsilon if algorithm == "epsilon_greedy" else 0.0
            explore_mask = rng.random(n_runs) < exploration
            if np.any(explore_mask):
                arms[explore_mask] = rng.integers(0, k, size=int(explore_mask.sum()))
        elif algorithm == "ucb1":
            if t < k:
                arms = np.full(n_runs, t, dtype=np.int64)
            else:
                indices = estimates + np.sqrt(2.0 * np.log(t + 1.0) / counts)
                arms = _choose_random_tie(rng, indices)
        else:
            samples = rng.beta(alpha, beta)
            arms = np.argmax(samples, axis=1).astype(np.int64)

        rewards = (rng.random(n_runs) < probs[arms]).astype(float)
        old_counts = counts[rows, arms]
        if algorithm == "thompson":
            alpha[rows, arms] += rewards
            beta[rows, arms] += 1.0 - rewards
            counts[rows, arms] += 1
            estimates[rows, arms] = alpha[rows, arms] / (alpha[rows, arms] + beta[rows, arms])
        else:
            estimates[rows, arms] += (rewards - estimates[rows, arms]) / (old_counts + 1)
            counts[rows, arms] += 1

        cumulative_regret += probs[optimal_arm] - probs[arms]
        cumulative_optimal += arms == optimal_arm
        regret[:, t] = cumulative_regret
        optimal_rate[:, t] = cumulative_optimal / (t + 1)

    return ManyBanditRuns(pseudo_regret=regret, optimal_rate=optimal_rate)


def summarize_many(result: ManyBanditRuns) -> BanditSummary:
    """Mean and normal-approximation 95% intervals from run-level curves."""
    if result.pseudo_regret.ndim != 2 or result.optimal_rate.shape != result.pseudo_regret.shape:
        raise ValueError("run matrices must have equal two-dimensional shapes")
    n_runs = result.pseudo_regret.shape[0]
    if n_runs < 2:
        raise ValueError("at least two runs are required")
    mean = result.pseudo_regret.mean(axis=0)
    stderr = result.pseudo_regret.std(axis=0, ddof=1) / np.sqrt(n_runs)
    half_width = 1.96 * stderr
    return BanditSummary(
        mean_regret=mean,
        lower_95=mean - half_width,
        upper_95=mean + half_width,
        mean_optimal_rate=result.optimal_rate.mean(axis=0),
    )
