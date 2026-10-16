from __future__ import annotations

from pathlib import Path
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit, ndtr

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "generated_experiment"
OUT.mkdir(exist_ok=True)
SEED = 202612


def binary_kl(q: float, p: float) -> float:
    """Bernoulli KL kl(q || p), in nats."""
    eps = 1e-15
    q = float(np.clip(q, eps, 1 - eps))
    p = float(np.clip(p, eps, 1 - eps))
    return q * math.log(q / p) + (1 - q) * math.log((1 - q) / (1 - p))


def kl_upper_inverse(q: float, c: float) -> float:
    """Largest p in [q,1] such that kl(q || p) <= c."""
    if c <= 0:
        return float(q)
    lo = float(q)
    hi = 1.0 - 1e-12
    if binary_kl(q, hi) <= c:
        return 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if binary_kl(q, mid) <= c:
            lo = mid
        else:
            hi = mid
    return lo


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / np.sum(weights, axis=1, keepdims=True)


def experiment_adaptive_selection() -> dict:
    """Adaptive selection among equally good hypotheses.

    Each data point is a vector of M Bernoulli losses. All hypotheses have the
    same population risk p. The algorithm chooses a hypothesis using a Gibbs
    distribution over empirical losses. This makes the selection bias and the
    MI bound directly visible.
    """
    rng = np.random.default_rng(SEED)
    m = 32
    n = 80
    true_risk = 0.20
    trials = 120_000
    empirical = rng.binomial(n, true_risk, size=(trials, m)) / n
    temperatures = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
    rows: list[dict] = []

    # A deliberately mismatched fixed prior for the KL decomposition demo.
    skew_prior = np.full(m, 0.5 / (m - 1))
    skew_prior[0] = 0.5

    for tau in temperatures:
        if tau == 0.0:
            minima = np.min(empirical, axis=1, keepdims=True)
            ties = empirical == minima
            q_cond = ties / np.sum(ties, axis=1, keepdims=True)
        else:
            q_cond = softmax_rows(-empirical / tau)

        marginal = np.mean(q_cond, axis=0)
        eps = 1e-300
        h_marginal = -float(np.sum(marginal * np.log(np.clip(marginal, eps, None))))
        h_conditional = -float(
            np.mean(np.sum(q_cond * np.log(np.clip(q_cond, eps, None)), axis=1))
        )
        mutual_information = h_marginal - h_conditional
        selected_empirical = float(np.mean(np.sum(q_cond * empirical, axis=1)))
        expected_gap = true_risk - selected_empirical

        # A [0,1]-valued loss is 1/2-sub-Gaussian by Hoeffding's lemma,
        # hence sigma^2 = 1/4 and the Xu-Raginsky bound is sqrt(I/(2n)).
        mi_bound = math.sqrt(mutual_information / (2 * n))

        avg_kl_uniform = float(
            np.mean(np.sum(q_cond * (np.log(np.clip(q_cond, eps, None)) + math.log(m)), axis=1))
        )
        avg_kl_skew = float(
            np.mean(
                np.sum(
                    q_cond
                    * (
                        np.log(np.clip(q_cond, eps, None))
                        - np.log(skew_prior)[None, :]
                    ),
                    axis=1,
                )
            )
        )
        marginal_kl_skew = float(
            np.sum(marginal * (np.log(np.clip(marginal, eps, None)) - np.log(skew_prior)))
        )

        rows.append(
            {
                "temperature": tau,
                "mutual_information_nats": mutual_information,
                "expected_empirical_risk": selected_empirical,
                "population_risk": true_risk,
                "expected_generalization_gap": expected_gap,
                "mi_bound": mi_bound,
                "marginal_entropy_nats": h_marginal,
                "conditional_entropy_nats": h_conditional,
                "avg_kl_to_uniform_prior": avg_kl_uniform,
                "avg_kl_to_skew_prior": avg_kl_skew,
                "marginal_kl_to_skew_prior": marginal_kl_skew,
            }
        )

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    x = np.array([r["mutual_information_nats"] for r in rows])
    gap = np.array([r["expected_generalization_gap"] for r in rows])
    bound = np.array([r["mi_bound"] for r in rows])
    ax.plot(x, gap, marker="o", label="estimated expected gap")
    ax.plot(x, bound, marker="s", linestyle="--", label="MI upper bound")
    for r in rows:
        label = "argmin" if r["temperature"] == 0 else f"τ={r['temperature']:g}"
        ax.annotate(label, (r["mutual_information_nats"], r["expected_generalization_gap"]),
                    xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("I(W; S), nats")
    ax.set_ylabel("expected generalization gap")
    ax.set_title("Adaptive selection: less information about the sample, less selection bias")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "M12_experiment_adaptive_selection.png", dpi=180)
    plt.close(fig)

    # Decomposition for tau=0.05.
    chosen = next(r for r in rows if abs(r["temperature"] - 0.05) < 1e-12)
    terms = [
        chosen["mutual_information_nats"],
        chosen["marginal_kl_to_skew_prior"],
        chosen["avg_kl_to_skew_prior"],
    ]
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.bar(["I(W;S)", "KL(P_W || P)", "E KL(Q_S || P)"], terms)
    ax.set_ylabel("nats")
    ax.set_title("Exact decomposition of average posterior-to-prior KL")
    ax.text(2, terms[2] + 0.02, "sum of the first two terms", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "M12_experiment_kl_decomposition.png", dpi=180)
    plt.close(fig)

    return {
        "M": m,
        "n": n,
        "true_risk": true_risk,
        "trials": trials,
        "rows": rows,
    }


def fit_logistic_regression(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Simple deterministic full-batch logistic regression."""
    n, d = x.shape
    w = np.zeros(d)
    learning_rate = 0.10
    l2 = 1e-3
    for _ in range(1800):
        margins = y * (x @ w)
        grad = -(x.T @ (y * expit(-margins))) / n + l2 * w
        w -= learning_rate * grad
    return w


def gaussian_kl_isotropic(mean: np.ndarray, sigma_q: float, sigma_p: float) -> float:
    k = mean.size
    return 0.5 * float(
        np.sum((sigma_q**2 + mean**2) / sigma_p**2 - 1 + math.log(sigma_p**2 / sigma_q**2))
    )


def gibbs_linear_risk(x: np.ndarray, y: np.ndarray, mean: np.ndarray, sigma_q: float) -> float:
    """Exact Gibbs 0-1 risk for W~N(mean, sigma_q^2 I) and sign(W^T x)."""
    signed_mean = y * (x @ mean)
    std = sigma_q * np.linalg.norm(x, axis=1)
    z = -signed_mean / np.maximum(std, 1e-15)
    return float(np.mean(ndtr(z)))


def experiment_pac_bayes_classifier() -> dict:
    rng = np.random.default_rng(SEED + 1)
    n_train = 4000
    n_test = 80_000
    d = 5
    delta = 0.05
    sigma_prior = 1.0

    true_w = rng.normal(size=d)
    true_w /= np.linalg.norm(true_w)

    def sample_data(n: int) -> tuple[np.ndarray, np.ndarray]:
        x = rng.normal(size=(n, d))
        prob = expit(2.0 * (x @ true_w))
        y = np.where(rng.random(n) < prob, 1.0, -1.0)
        x_aug = np.column_stack([x, np.ones(n)])
        return x_aug, y

    x_train, y_train = sample_data(n_train)
    x_test, y_test = sample_data(n_test)
    mean = fit_logistic_regression(x_train, y_train)
    deterministic_train = float(np.mean(y_train * (x_train @ mean) <= 0))
    deterministic_test = float(np.mean(y_test * (x_test @ mean) <= 0))

    sigmas = [0.03, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00]
    rows: list[dict] = []
    confidence_term = math.log(2 * math.sqrt(n_train) / delta)

    for sigma_q in sigmas:
        empirical_gibbs = gibbs_linear_risk(x_train, y_train, mean, sigma_q)
        test_gibbs = gibbs_linear_risk(x_test, y_test, mean, sigma_q)
        kl = gaussian_kl_isotropic(mean, sigma_q, sigma_prior)
        complexity = (kl + confidence_term) / n_train
        kl_bound = kl_upper_inverse(empirical_gibbs, complexity)
        pinsker_bound = min(1.0, empirical_gibbs + math.sqrt(complexity / 2))
        rows.append(
            {
                "sigma_posterior": sigma_q,
                "empirical_gibbs_risk": empirical_gibbs,
                "test_gibbs_risk": test_gibbs,
                "kl_q_p_nats": kl,
                "pac_bayes_kl_bound": kl_bound,
                "pinsker_square_root_bound": pinsker_bound,
            }
        )

    best = min(rows, key=lambda r: r["pac_bayes_kl_bound"])

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    xs = np.array(sigmas)
    ax.plot(xs, [r["empirical_gibbs_risk"] for r in rows], marker="o", label="empirical Gibbs risk")
    ax.plot(xs, [r["test_gibbs_risk"] for r in rows], marker="o", label="test Gibbs risk")
    ax.plot(xs, [r["pac_bayes_kl_bound"] for r in rows], marker="s", label="PAC-Bayes-kl certificate")
    ax.plot(xs, [r["pinsker_square_root_bound"] for r in rows], linestyle="--", label="square-root corollary")
    ax.axhline(deterministic_test, linestyle=":", label="deterministic mean test error")
    ax.set_xscale("log")
    ax.set_xlabel("posterior standard deviation σ_Q")
    ax.set_ylabel("0-1 risk")
    ax.set_title("Stochastic linear classifier: empirical fit, KL complexity, and certificate")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "M12_experiment_pac_bayes_classifier.png", dpi=180)
    plt.close(fig)

    return {
        "n_train": n_train,
        "n_test": n_test,
        "dimension_with_bias": d + 1,
        "delta": delta,
        "sigma_prior": sigma_prior,
        "mean_norm": float(np.linalg.norm(mean)),
        "deterministic_train_error": deterministic_train,
        "deterministic_test_error": deterministic_test,
        "rows": rows,
        "best_by_pac_bayes_kl": best,
    }


def main() -> None:
    results = {
        "adaptive_selection": experiment_adaptive_selection(),
        "pac_bayes_classifier": experiment_pac_bayes_classifier(),
    }
    (OUT / "M12_experiment_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    best = results["pac_bayes_classifier"]["best_by_pac_bayes_kl"]
    print("Adaptive selection (argmin):")
    row0 = results["adaptive_selection"]["rows"][0]
    print(f"  I(W;S): {row0['mutual_information_nats']:.6f} nats")
    print(f"  expected gap: {row0['expected_generalization_gap']:.6f}")
    print(f"  MI bound: {row0['mi_bound']:.6f}")
    print("Best stochastic classifier certificate:")
    print(f"  sigma_Q: {best['sigma_posterior']:.3f}")
    print(f"  empirical Gibbs risk: {best['empirical_gibbs_risk']:.6f}")
    print(f"  test Gibbs risk: {best['test_gibbs_risk']:.6f}")
    print(f"  KL(Q||P): {best['kl_q_p_nats']:.6f} nats")
    print(f"  PAC-Bayes-kl bound: {best['pac_bayes_kl_bound']:.6f}")


if __name__ == "__main__":
    main()
