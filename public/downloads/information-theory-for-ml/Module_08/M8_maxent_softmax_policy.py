"""Reproducible experiments for Module 8.

The script demonstrates:
1. maximum entropy on a die with a prescribed mean;
2. dependence of MaxEnt on the chosen state space;
3. the softmax variational identity;
4. a finite-action reward-overoptimization path.

Run:
    python M8_maxent_softmax_policy.py

Dependencies:
    numpy, scipy, matplotlib
"""

from __future__ import annotations

from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp


OUTPUT_DIR = Path(__file__).resolve().parent / "generated_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    """Return normalized probabilities from unnormalized log weights."""
    log_weights = np.asarray(log_weights, dtype=float)
    return np.exp(log_weights - logsumexp(log_weights))


def entropy_bits(probabilities: np.ndarray) -> float:
    """Shannon entropy in bits."""
    p = np.asarray(probabilities, dtype=float)
    positive = p > 0
    return float(-np.sum(p[positive] * np.log2(p[positive])))


def kl_nats(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in nats for strictly positive q on the support of p."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    positive = p > 0
    if np.any(q[positive] <= 0):
        return float("inf")
    return float(np.sum(p[positive] * (np.log(p[positive]) - np.log(q[positive]))))


def solve_exponential_tilt(states: np.ndarray, target_mean: float) -> tuple[float, np.ndarray]:
    """Solve p_lambda(x) proportional to exp(lambda*x) with a target mean."""
    states = np.asarray(states, dtype=float)
    if states.ndim != 1 or states.size == 0 or not np.all(np.isfinite(states)):
        raise ValueError("states must be a nonempty finite one-dimensional array")
    low, high = states.min(), states.max()
    if not (low <= target_mean <= high):
        raise ValueError("target_mean must lie in the convex hull of states")

    if low == high:
        return 0.0, np.full(states.size, 1.0 / states.size)
    if target_mean == low:
        p = (states == low).astype(float)
        p /= p.sum()
        return float("-inf"), p
    if target_mean == high:
        p = (states == high).astype(float)
        p /= p.sum()
        return float("inf"), p

    # Rescale the statistic; this leaves the distribution unchanged and makes
    # bracketing independent of the units in which states are measured.
    scaled = (states - low) / (high - low)
    target = (target_mean - low) / (high - low)

    def probabilities(eta: float) -> np.ndarray:
        return normalize_log_weights(eta * scaled)

    def residual(eta: float) -> float:
        return float(probabilities(eta) @ scaled - target)

    left, right = -1.0, 1.0
    while residual(left) > 0:
        left *= 2.0
    while residual(right) < 0:
        right *= 2.0
    root = brentq(residual, left, right, xtol=1e-12)
    return float(root / (high - low)), probabilities(root)


def experiment_die(target_mean: float = 4.5) -> dict[str, object]:
    """MaxEnt distribution on a six-sided die with a prescribed mean."""
    states = np.arange(1, 7, dtype=float)
    lam, p = solve_exponential_tilt(states, target_mean)
    entropy = entropy_bits(p)
    uniform = np.full_like(p, 1.0 / len(p))
    deficit = np.log2(len(p)) - entropy
    kl_to_uniform_bits = kl_nats(p, uniform) / np.log(2.0)

    assert np.isclose(p.sum(), 1.0, atol=1e-12)
    assert np.isclose(p @ states, target_mean, atol=1e-11)
    assert np.isclose(deficit, kl_to_uniform_bits, atol=1e-12)

    print("\n=== Experiment 1: MaxEnt die ===")
    print(f"lambda = {lam:.12f}")
    print("probabilities =", np.array2string(p, precision=9))
    print(f"mean = {p @ states:.12f}")
    print(f"entropy = {entropy:.12f} bits")
    print(f"log2(6) - H = {deficit:.12f} bits")
    print(f"KL(p || uniform) = {kl_to_uniform_bits:.12f} bits")

    plt.figure(figsize=(7.2, 4.6))
    plt.bar(states.astype(int), p)
    plt.xlabel("Die face")
    plt.ylabel("Probability")
    plt.title(f"Maximum-entropy die with E[X] = {target_mean:g}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "M8_die_maxent.png", dpi=180)
    plt.close()

    return {
        "lambda": lam,
        "probabilities": p,
        "entropy_bits": entropy,
        "deficit_bits": deficit,
    }


def experiment_state_space() -> dict[str, object]:
    """Compare two MaxEnt problems with n=10 and E[K]=7."""
    n = 10
    target_mean = 7.0
    counts = np.arange(n + 1, dtype=float)

    # Model A: the count k itself is one state, with a uniform reference on counts.
    lam_count, p_count = solve_exponential_tilt(counts, target_mean)

    # Model B: uniform reference over 2^n binary microstates.
    bernoulli_p = target_mean / n
    p_binomial = np.array(
        [
            float(math.comb(n, int(k)))
            * bernoulli_p**k
            * (1.0 - bernoulli_p) ** (n - k)
            for k in counts
        ],
        dtype=float,
    )

    p_binomial /= p_binomial.sum()

    assert np.isclose(p_count @ counts, target_mean, atol=1e-10)
    assert np.isclose(p_binomial @ counts, target_mean, atol=1e-10)
    assert not np.allclose(p_count, p_binomial)

    print("\n=== Experiment 2: one mean, two state spaces ===")
    print(f"count-space lambda = {lam_count:.12f}")
    print("count-space probabilities =", np.array2string(p_count, precision=8))
    print("binomial probabilities   =", np.array2string(p_binomial, precision=8))
    print(f"count-space entropy = {entropy_bits(p_count):.9f} bits")
    print(f"binomial entropy    = {entropy_bits(p_binomial):.9f} bits")

    width = 0.38
    plt.figure(figsize=(8.2, 4.8))
    plt.bar(counts - width / 2, p_count, width=width, label="Uniform reference on counts")
    plt.bar(counts + width / 2, p_binomial, width=width, label="Uniform reference on microstates")
    plt.xlabel("Number of ones K")
    plt.ylabel("Probability")
    plt.title("Same mean E[K] = 7, different state spaces")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "M8_state_space.png", dpi=180)
    plt.close()

    return {
        "lambda_count": lam_count,
        "count_probabilities": p_count,
        "binomial_probabilities": p_binomial,
    }


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Stable softmax."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = np.asarray(logits, dtype=float) / temperature
    return normalize_log_weights(z)


def entropy_nats(probabilities: np.ndarray) -> float:
    """Shannon entropy in nats."""
    p = np.asarray(probabilities, dtype=float)
    positive = p > 0
    return float(-np.sum(p[positive] * np.log(p[positive])))


def experiment_softmax_variational() -> dict[str, float]:
    """Numerically verify the softmax variational formula."""
    logits = np.array([2.0, 0.5, -1.0, 1.2], dtype=float)
    temperature = 0.7
    optimum = softmax(logits, temperature)
    optimum_value = temperature * logsumexp(logits / temperature)
    direct_value = float(optimum @ logits + temperature * entropy_nats(optimum))

    rng = np.random.default_rng(8)
    candidate = rng.dirichlet(np.ones(len(logits)))
    candidate_value = float(candidate @ logits + temperature * entropy_nats(candidate))
    gap = optimum_value - candidate_value
    predicted_gap = temperature * kl_nats(candidate, optimum)

    assert np.isclose(optimum_value, direct_value, atol=1e-12)
    assert gap >= -1e-12
    assert np.isclose(gap, predicted_gap, atol=1e-12)

    print("\n=== Experiment 3: softmax variational identity ===")
    print("softmax optimum =", np.array2string(optimum, precision=9))
    print(f"tau * logsumexp(z/tau) = {optimum_value:.12f}")
    print(f"objective at optimum    = {direct_value:.12f}")
    print(f"gap for random q        = {gap:.12f}")
    print(f"tau * KL(q || softmax)  = {predicted_gap:.12f}")

    return {
        "optimum_value": optimum_value,
        "candidate_gap": gap,
    }


def experiment_reward_overoptimization() -> dict[str, object]:
    """A finite-action example in which proxy reward rises while true reward eventually falls."""
    regular_action = np.linspace(-2.0, 2.0, 100)
    true_regular = -(regular_action - 0.75) ** 2
    proxy_regular = true_regular + 0.03 * np.sin(5.0 * regular_action)

    # Add a rare action that looks attractive to the proxy but is poor under the true objective.
    true_reward = np.concatenate([true_regular, np.array([-3.0])])
    proxy_reward = np.concatenate([proxy_regular, np.array([0.8])])
    reference = np.concatenate([np.full(100, 0.999 / 100.0), np.array([0.001])])

    betas = np.geomspace(3.0, 0.02, 120)
    inverse_betas = 1.0 / betas
    expected_proxy = np.empty_like(betas)
    expected_true = np.empty_like(betas)
    kl_to_reference = np.empty_like(betas)
    hack_probability = np.empty_like(betas)

    for i, beta in enumerate(betas):
        policy = normalize_log_weights(np.log(reference) + proxy_reward / beta)
        expected_proxy[i] = policy @ proxy_reward
        expected_true[i] = policy @ true_reward
        kl_to_reference[i] = kl_nats(policy, reference)
        hack_probability[i] = policy[-1]

    # As beta decreases, alpha=1/beta increases. The exact identities predict monotone
    # proxy reward and monotone KL along this finite Gibbs path.
    assert np.all(np.diff(expected_proxy) >= -1e-10)
    assert np.all(np.diff(kl_to_reference) >= -1e-10)
    peak_index = int(np.argmax(expected_true))
    assert 0 < peak_index < len(expected_true) - 1
    assert expected_true[-1] < expected_true[peak_index] - 1.0

    print("\n=== Experiment 4: reward overoptimization ===")
    print(f"true reward peaks at beta = {betas[peak_index]:.6f}")
    print(f"peak expected true reward = {expected_true[peak_index]:.9f}")
    print(f"final expected true reward = {expected_true[-1]:.9f}")
    print(f"final expected proxy reward = {expected_proxy[-1]:.9f}")
    print(f"final KL(policy || reference) = {kl_to_reference[-1]:.9f} nats")
    print(f"final reward-hacking action probability = {hack_probability[-1]:.9f}")

    plt.figure(figsize=(7.6, 4.8))
    plt.plot(inverse_betas, expected_proxy, label="Expected proxy reward")
    plt.plot(inverse_betas, expected_true, label="Expected true reward")
    plt.xscale("log")
    plt.xlabel("Optimization pressure 1/beta")
    plt.ylabel("Expected reward")
    plt.title("A possible reward-overoptimization path")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "M8_reward_overoptimization.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.6, 4.8))
    plt.plot(inverse_betas, kl_to_reference, label="KL to reference")
    plt.plot(inverse_betas, hack_probability, label="Probability of hacking action")
    plt.xscale("log")
    plt.xlabel("Optimization pressure 1/beta")
    plt.ylabel("Value")
    plt.title("Policy drift along the Gibbs path")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "M8_policy_drift.png", dpi=180)
    plt.close()

    return {
        "betas": betas,
        "expected_proxy": expected_proxy,
        "expected_true": expected_true,
        "kl_to_reference": kl_to_reference,
        "hack_probability": hack_probability,
        "peak_index": peak_index,
    }


def main() -> None:
    experiment_die()
    state_space = experiment_state_space()
    softmax_result = experiment_softmax_variational()
    reward_path = experiment_reward_overoptimization()

    print("\nGenerated figures:")
    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(" -", path.name)

    # Compact final checks that also serve as machine-readable invariants.
    assert np.isclose(state_space["count_probabilities"].sum(), 1.0)
    assert softmax_result["candidate_gap"] >= 0
    assert reward_path["expected_true"][-1] < reward_path["expected_true"][reward_path["peak_index"]]


if __name__ == "__main__":
    main()
