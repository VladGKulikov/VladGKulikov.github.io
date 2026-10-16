from __future__ import annotations

"""Reproducible experiments for Module 13.

The script demonstrates three separate mechanisms:
1. a sequential scratchpad can make parity linearly accessible without adding
   Shannon information beyond the input;
2. correlated reasoning errors sharply reduce the gain from self-consistency;
3. oracle pass@N may rise while a weak verifier's selected accuracy eventually
   declines because extreme false positives become more likely.
"""

from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

SEED = 13
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "generated_experiment"
OUT.mkdir(exist_ok=True)


def parity_experiment(rng: np.random.Generator) -> dict[str, float]:
    n_bits = 12
    n_train = 12_000
    n_test = 6_000

    x_train = rng.integers(0, 2, size=(n_train, n_bits), dtype=np.int8)
    x_test = rng.integers(0, 2, size=(n_test, n_bits), dtype=np.int8)
    y_train = np.bitwise_xor.reduce(x_train, axis=1)
    y_test = np.bitwise_xor.reduce(x_test, axis=1)

    direct = LogisticRegression(C=1.0, max_iter=2_000, random_state=SEED)
    direct.fit(x_train, y_train)
    direct_acc = accuracy_score(y_test, direct.predict(x_test))

    # Sequential scratchpad state S_t = parity of the first t bits.
    s_train = np.bitwise_xor.accumulate(x_train, axis=1)
    s_test = np.bitwise_xor.accumulate(x_test, axis=1)

    # A linear readout from the final one-bit state now solves the task.
    scratch = LogisticRegression(C=1.0, max_iter=2_000, random_state=SEED)
    scratch.fit(s_train[:, [-1]], y_train)
    scratch_acc = accuracy_score(y_test, scratch.predict(s_test[:, [-1]]))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(["Linear readout\non raw bits", "Linear readout\non final scratch state"],
           [direct_acc, scratch_acc])
    ax.axhline(0.5, linestyle="--", linewidth=1.2, label="chance")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Parity: sequential state changes accessibility, not input information")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "M13_parity_scratchpad.png", dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    return {
        "direct_linear_accuracy": float(direct_acc),
        "scratchpad_linear_accuracy": float(scratch_acc),
    }


def independent_majority_probability(n: int, p: float) -> float:
    threshold = n // 2 + 1
    return float(sum(comb(n, r) * p**r * (1.0 - p)**(n - r)
                     for r in range(threshold, n + 1)))


def self_consistency_experiment(
    rng: np.random.Generator, p: float = 0.62,
) -> dict[str, list[float]]:
    rho = 0.55
    n_values = np.array([1, 3, 5, 9, 15, 25, 51], dtype=int)
    n_tasks = 250_000

    independent = np.array([independent_majority_probability(int(n), p)
                            for n in n_values])

    # Correlated model: on a fraction rho of tasks, all trajectories share one
    # latent Bernoulli outcome; otherwise trajectories are independent.
    correlated_values: list[float] = []
    for n in n_values:
        shared_mask = rng.random(n_tasks) < rho
        shared_outcome = rng.random(n_tasks) < p
        independent_draws = rng.random((n_tasks, int(n))) < p
        votes = independent_draws.sum(axis=1)
        independent_majority = votes >= (int(n) // 2 + 1)
        majority = np.where(shared_mask, shared_outcome, independent_majority)
        correlated_values.append(float(majority.mean()))

    correlated = np.array(correlated_values)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(n_values, independent, marker="o", label="independent errors")
    ax.plot(n_values, correlated, marker="s", label=f"correlated mixture (rho={rho:.2f})")
    ax.axhline(p, linestyle="--", linewidth=1.1, label="single trajectory")
    ax.set_xlabel("Number of trajectories N")
    ax.set_ylabel("Majority-vote accuracy")
    ax.set_ylim(0.0, 1.01)
    ax.set_title("Self-consistency: correlation reduces effective sample count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "M13_self_consistency.png", dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    return {
        "N": n_values.tolist(),
        "independent": independent.tolist(),
        "correlated": correlated.tolist(),
    }


def best_of_n_experiment(rng: np.random.Generator) -> dict[str, list[float]]:
    p_correct = 0.20
    n_values = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=int)
    n_tasks = 70_000

    oracle: list[float] = []
    selected_good: list[float] = []
    selected_weak: list[float] = []

    for n in n_values:
        correct = rng.random((n_tasks, int(n))) < p_correct
        oracle.append(float(correct.any(axis=1).mean()))

        # Strong verifier: well-separated, light-tailed score distributions.
        score_good = np.where(
            correct,
            rng.normal(2.0, 0.45, size=correct.shape),
            rng.normal(0.0, 0.55, size=correct.shape),
        )
        chosen_good = np.argmax(score_good, axis=1)
        selected_good.append(float(correct[np.arange(n_tasks), chosen_good].mean()))

        # Weak verifier: correct candidates are usually preferred, but a small
        # fraction of incorrect candidates receive exploitatively high scores.
        base_wrong = rng.normal(0.0, 0.75, size=correct.shape)
        exploit = rng.random(correct.shape) < 0.018
        exploit_score = rng.normal(3.8, 0.45, size=correct.shape)
        wrong_score = np.where(exploit, exploit_score, base_wrong)
        score_weak = np.where(
            correct,
            rng.normal(1.35, 0.55, size=correct.shape),
            wrong_score,
        )
        chosen_weak = np.argmax(score_weak, axis=1)
        selected_weak.append(float(correct[np.arange(n_tasks), chosen_weak].mean()))

    oracle_arr = np.array(oracle)
    good_arr = np.array(selected_good)
    weak_arr = np.array(selected_weak)

    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    ax.plot(n_values, oracle_arr, marker="o", label="oracle pass@N")
    ax.plot(n_values, good_arr, marker="s", label="selected by strong verifier")
    ax.plot(n_values, weak_arr, marker="^", label="selected by weak verifier")
    ax.set_xscale("log", base=2)
    ax.set_xticks(n_values, [str(x) for x in n_values])
    ax.set_xlabel("Number of candidates N")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Best-of-N: more candidates expose both solutions and verifier exploits")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "M13_best_of_n.png", dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    return {
        "N": n_values.tolist(),
        "oracle_pass_at_n": oracle_arr.tolist(),
        "selected_strong": good_arr.tolist(),
        "selected_weak": weak_arr.tolist(),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)

    parity = parity_experiment(rng)
    self_consistency = self_consistency_experiment(rng)
    best_of_n = best_of_n_experiment(rng)

    print("=== Parity and scratchpad ===")
    for key, value in parity.items():
        print(f"{key}: {value:.6f}")

    print("\n=== Self-consistency ===")
    for n, ind, corr in zip(self_consistency["N"],
                            self_consistency["independent"],
                            self_consistency["correlated"]):
        print(f"N={n:3d}: independent={ind:.6f}, correlated={corr:.6f}")

    print("\n=== Best-of-N ===")
    for n, oracle, strong, weak in zip(best_of_n["N"],
                                       best_of_n["oracle_pass_at_n"],
                                       best_of_n["selected_strong"],
                                       best_of_n["selected_weak"]):
        print(f"N={n:3d}: oracle={oracle:.6f}, strong={strong:.6f}, weak={weak:.6f}")


if __name__ == "__main__":
    main()
