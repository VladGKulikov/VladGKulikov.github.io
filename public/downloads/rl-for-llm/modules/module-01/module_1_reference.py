#!/usr/bin/env python3
"""Reference implementations for RL for LLM, Module 1.

The module deliberately uses only NumPy and the Python standard library so the
examples run on CPU, in a browser notebook, or in the Stepik Python runner.
"""
from __future__ import annotations

from collections.abc import Iterable
import math

import numpy as np


ArrayLike1D = Iterable[float] | np.ndarray


def _as_finite_1d(values: ArrayLike1D, *, name: str) -> np.ndarray:
    """Convert input to a non-empty finite 1D float64 NumPy array."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def discounted_return(rewards: ArrayLike1D, gamma: float) -> float:
    """Compute G_0 = sum_k gamma**k * r_{k+1} using a backward recursion.

    Parameters
    ----------
    rewards:
        A non-empty finite sequence ``[r_1, ..., r_T]``.
    gamma:
        Discount factor in ``[0, 1]``.
    """
    reward_array = _as_finite_1d(rewards, name="rewards")
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and lie in [0, 1]")

    value = 0.0
    for reward in reward_array[::-1]:
        value = float(reward) + gamma * value
    return float(value)


def softmax_with_temperature(logits: ArrayLike1D, temperature: float) -> np.ndarray:
    """Return a numerically stable categorical softmax at ``temperature``.

    The function models the *pure* temperature-softmax transform. It does not
    include top-k/top-p truncation, penalties, safety filters, or any
    provider-specific decoding transformation.
    """
    logit_array = _as_finite_1d(logits, name="logits")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")

    scaled = logit_array / temperature
    shifted = scaled - np.max(scaled)
    weights = np.exp(shifted)
    normalizer = float(np.sum(weights))
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise FloatingPointError("softmax normalization failed")
    probabilities = weights / normalizer
    return probabilities.astype(np.float64, copy=False)


def categorical_entropy(probabilities: ArrayLike1D) -> float:
    """Compute Shannon entropy in nats for a categorical distribution."""
    p = _as_finite_1d(probabilities, name="probabilities")
    if np.any(p < 0.0):
        raise ValueError("probabilities must be non-negative")
    total = float(np.sum(p))
    if not math.isclose(total, 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("probabilities must sum to 1")

    positive = p > 0.0
    return float(-np.sum(p[positive] * np.log(p[positive])))


def policy_entropy(logits: ArrayLike1D, temperature: float) -> float:
    """Entropy in nats of a pure temperature-softmax policy."""
    return categorical_entropy(softmax_with_temperature(logits, temperature))


def temperature_entropy_curve(
    logits: ArrayLike1D,
    temperatures: ArrayLike1D,
) -> np.ndarray:
    """Evaluate policy entropy for a positive temperature grid."""
    grid = _as_finite_1d(temperatures, name="temperatures")
    if np.any(grid <= 0.0):
        raise ValueError("temperatures must be positive")
    return np.asarray([policy_entropy(logits, float(t)) for t in grid])


def run_corridor_episode(
    rng: np.random.Generator,
    *,
    max_steps: int = 20,
) -> float:
    """Run one episode in the five-cell corridor under a fair random policy.

    States 0 and 4 are terminal. The agent starts in state 2. Action 0 moves
    left, action 1 moves right. Reaching state 4 yields return 1; all other
    terminations (left terminal or time limit) yield return 0.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an instance of numpy.random.Generator")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")

    state = 2
    for _ in range(max_steps):
        action = int(rng.integers(0, 2))
        state += -1 if action == 0 else 1
        if state == 0:
            return 0.0
        if state == 4:
            return 1.0
    return 0.0


def estimate_corridor_value(
    n_episodes: int,
    *,
    seed: int = 7,
    max_steps: int = 20,
) -> float:
    """Estimate the start-state value by Monte Carlo simulation."""
    if not isinstance(n_episodes, int) or isinstance(n_episodes, bool) or n_episodes <= 0:
        raise ValueError("n_episodes must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    rng = np.random.default_rng(seed)
    returns = [run_corridor_episode(rng, max_steps=max_steps) for _ in range(n_episodes)]
    return float(np.mean(returns))


def exact_corridor_value(*, max_steps: int | None = 20) -> float:
    """Return the exact start-state value for the fair random corridor policy.

    With ``max_steps=None`` the absorbing random walk has value exactly 1/2.
    With a finite horizon, dynamic programming computes the probability of
    reaching the right terminal within the allowed number of actions.
    """
    if max_steps is None:
        return 0.5
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer or None")

    # v_h[s] = probability of reaching state 4 within h remaining actions.
    values = np.zeros(5, dtype=np.float64)
    values[4] = 1.0
    for _ in range(max_steps):
        updated = values.copy()
        for state in (1, 2, 3):
            updated[state] = 0.5 * values[state - 1] + 0.5 * values[state + 1]
        updated[0] = 0.0
        updated[4] = 1.0
        values = updated
    return float(values[2])


def corridor_standard_error(
    n_episodes: int,
    *,
    max_steps: int | None = 20,
) -> float:
    """Bernoulli standard error around the exact corridor value."""
    if not isinstance(n_episodes, int) or isinstance(n_episodes, bool) or n_episodes <= 0:
        raise ValueError("n_episodes must be a positive integer")
    probability = exact_corridor_value(max_steps=max_steps)
    return math.sqrt(probability * (1.0 - probability) / n_episodes)


if __name__ == "__main__":
    logits = np.asarray([2.0, 1.0, 0.0, -1.0])
    temperatures = np.asarray([0.25, 0.5, 1.0, 2.0, 4.0, 10.0])
    entropies = temperature_entropy_curve(logits, temperatures)

    print("Temperature -> entropy (nats)")
    for temperature, entropy in zip(temperatures, entropies, strict=True):
        print(f"{temperature:>5.2f} -> {entropy:.6f}")

    exact = exact_corridor_value(max_steps=20)
    estimate = estimate_corridor_value(10_000, seed=7, max_steps=20)
    print(f"Exact corridor value (20-step limit): {exact:.9f}")
    print(f"Monte Carlo estimate (N=10000, seed=7): {estimate:.4f}")
