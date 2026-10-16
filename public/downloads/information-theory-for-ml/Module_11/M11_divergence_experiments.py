from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "generated_experiment"
OUT.mkdir(exist_ok=True)


def discrete_variational_experiment() -> dict[str, object]:
    """Check NWJ, DV, classifier-ratio, and original-GAN identities."""
    p = np.array([0.45, 0.05, 0.45, 0.05], dtype=np.float64)
    q = np.array([0.05, 0.45, 0.05, 0.45], dtype=np.float64)
    x = np.arange(len(p), dtype=np.float64)

    log_ratio = np.log(p / q)
    exact_kl = float(np.sum(p * log_ratio))

    # Exact population optima.
    t_nwj = 1.0 + log_ratio
    nwj_full = float(np.sum(p * t_nwj) - np.sum(q * np.exp(t_nwj - 1.0)))

    g_dv = log_ratio
    dv_full = float(np.sum(p * g_dv) - math.log(float(np.sum(q * np.exp(g_dv)))))

    # Restricted linear critics. NWJ uses a*x+b; DV fixes the irrelevant intercept.
    def nwj_negative(params: np.ndarray) -> float:
        a, b = params
        t = a * x + b
        value = np.sum(p * t) - np.sum(q * np.exp(t - 1.0))
        return -float(value)

    nwj_result = minimize(nwj_negative, x0=np.zeros(2), method="BFGS")
    if not nwj_result.success:
        raise RuntimeError(f"NWJ critic optimization failed: {nwj_result.message}")
    a_nwj, b_nwj = nwj_result.x
    nwj_linear = -float(nwj_result.fun)

    def dv_negative(a_arr: np.ndarray) -> float:
        a = float(a_arr[0])
        g = a * x
        value = np.sum(p * g) - math.log(float(np.sum(q * np.exp(g))))
        return -float(value)

    dv_result = minimize(dv_negative, x0=np.zeros(1), method="BFGS")
    if not dv_result.success:
        raise RuntimeError(f"DV critic optimization failed: {dv_result.message}")
    a_dv = float(dv_result.x[0])
    dv_linear = -float(dv_result.fun)

    # Bayes classifier recovers the log density ratio for equal class priors.
    d_star = p / (p + q)
    classifier_logit = np.log(d_star / (1.0 - d_star))
    classifier_max_abs_error = float(np.max(np.abs(classifier_logit - log_ratio)))

    # Original GAN identity.
    m = 0.5 * (p + q)
    js = 0.5 * float(np.sum(p * np.log(p / m))) + 0.5 * float(np.sum(q * np.log(q / m)))
    gan_value = float(np.sum(p * np.log(d_star)) + np.sum(q * np.log(1.0 - d_star)))
    gan_identity_rhs = -math.log(4.0) + 2.0 * js

    labels = ["exact KL", "NWJ full", "DV full", "NWJ linear", "DV linear"]
    values = [exact_kl, nwj_full, dv_full, nwj_linear, dv_linear]
    fig, ax = plt.subplots(figsize=(9.0, 5.1))
    ax.bar(np.arange(len(labels)), values)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=15, ha="right")
    ax.set_ylabel("nats")
    ax.set_title("Restricted critics leave a variational gap")
    ax.axhline(exact_kl, linestyle="--", linewidth=1.0, label="exact KL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "M11_experiment_variational_bounds.png", dpi=190)
    plt.close(fig)

    return {
        "p": p.tolist(),
        "q": q.tolist(),
        "log_density_ratio": log_ratio.tolist(),
        "exact_kl_nats": exact_kl,
        "nwj_full_nats": nwj_full,
        "dv_full_nats": dv_full,
        "nwj_linear_nats": nwj_linear,
        "dv_linear_nats": dv_linear,
        "nwj_linear_a": float(a_nwj),
        "nwj_linear_b": float(b_nwj),
        "dv_linear_a": a_dv,
        "classifier_logit_max_abs_error": classifier_max_abs_error,
        "js_nats": js,
        "gan_value": gan_value,
        "gan_identity_rhs": gan_identity_rhs,
        "gan_identity_abs_error": abs(gan_value - gan_identity_rhs),
    }


def gaussian_log_ratio(x: np.ndarray, y: np.ndarray, rho: float) -> np.ndarray:
    """Return log p(y|x)/p(y) for a standard jointly Gaussian pair."""
    var = 1.0 - rho * rho
    return -0.5 * math.log(var) - 0.5 * (y - rho * x) ** 2 / var + 0.5 * y**2


def estimate_infonce_bound(
    rho: float,
    k: int,
    n_trials: int,
    rng: np.random.Generator,
    chunk_size: int = 500,
) -> tuple[float, float, float]:
    """Monte Carlo InfoNCE bound using the Bayes-optimal score."""
    losses: list[np.ndarray] = []
    remaining = n_trials
    sigma = math.sqrt(1.0 - rho * rho)
    while remaining > 0:
        n = min(chunk_size, remaining)
        x = rng.standard_normal((n, 1))
        y_pos = rho * x + sigma * rng.standard_normal((n, 1))
        y_neg = rng.standard_normal((n, k - 1)) if k > 1 else np.empty((n, 0))
        y_all = np.concatenate([y_pos, y_neg], axis=1)
        scores = gaussian_log_ratio(x, y_all, rho)
        loss = logsumexp(scores, axis=1) - scores[:, 0]
        losses.append(loss)
        remaining -= n
    all_losses = np.concatenate(losses)
    mean_loss = float(np.mean(all_losses))
    stderr = float(np.std(all_losses, ddof=1) / math.sqrt(len(all_losses)))
    return math.log(k) - mean_loss, mean_loss, stderr


def infonce_experiment() -> dict[str, object]:
    rng = np.random.default_rng(20260713)
    rhos = [0.5, 0.9, 0.99, 0.9999]
    ks = [2, 4, 16, 64, 256]
    n_trials = 20_000

    rows: list[dict[str, float | int]] = []
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for rho in rhos:
        exact = -0.5 * math.log(1.0 - rho * rho)
        bounds = []
        for k in ks:
            bound, loss, stderr = estimate_infonce_bound(rho, k, n_trials, rng)
            bounds.append(bound)
            rows.append({
                "rho": rho,
                "K": k,
                "exact_mi_nats": exact,
                "infonce_loss_nats": loss,
                "infonce_bound_nats": bound,
                "loss_stderr": stderr,
                "log_K": math.log(k),
            })
        ax.plot(ks, bounds, marker="o", label=fr"$\rho={rho}$ bound")
        ax.axhline(exact, linestyle="--", linewidth=0.9, alpha=0.55)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of candidates K")
    ax.set_ylabel("nats")
    ax.set_title("InfoNCE approaches MI but is capped by log K")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "M11_experiment_infonce_saturation.png", dpi=190)
    plt.close(fig)

    return {"n_trials": n_trials, "rows": rows}


def support_geometry_experiment() -> dict[str, object]:
    theta = np.linspace(-3.0, 3.0, 601)
    nonzero = np.abs(theta) > 1e-12
    js = np.where(nonzero, math.log(2.0), 0.0)
    tv = np.where(nonzero, 1.0, 0.0)
    w1 = np.abs(theta)

    fig, ax = plt.subplots(figsize=(8.8, 5.1))
    ax.plot(theta, w1, label=r"$W_1(\delta_0,\delta_\theta)=|\theta|$")
    ax.plot(theta, js, label=r"$D_{JS}$")
    ax.plot(theta, tv, label="TV")
    ax.axvline(0.0, linewidth=0.8)
    ax.set_xlabel(r"shift $\theta$")
    ax.set_ylabel("discrepancy")
    ax.set_title("Ground geometry distinguishes disjoint supports")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "M11_experiment_support_geometry.png", dpi=190)
    plt.close(fig)

    return {
        "theta_examples": [-2.0, -0.1, 0.0, 0.1, 2.0],
        "js_for_nonzero_theta": math.log(2.0),
        "tv_for_nonzero_theta": 1.0,
        "kl_for_nonzero_theta": "infinity",
        "w1_formula": "abs(theta)",
    }


def main() -> None:
    result = {
        "discrete_variational": discrete_variational_experiment(),
        "infonce": infonce_experiment(),
        "support_geometry": support_geometry_experiment(),
    }
    path = OUT / "M11_experiment_results.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    d = result["discrete_variational"]
    print("Exact KL:", f"{d['exact_kl_nats']:.9f}", "nats")
    print("NWJ full:", f"{d['nwj_full_nats']:.9f}", "nats")
    print("DV full:", f"{d['dv_full_nats']:.9f}", "nats")
    print("NWJ linear:", f"{d['nwj_linear_nats']:.9f}", "nats")
    print("DV linear:", f"{d['dv_linear_nats']:.9f}", "nats")
    print("Classifier logit max error:", f"{d['classifier_logit_max_abs_error']:.3e}")
    print("GAN identity error:", f"{d['gan_identity_abs_error']:.3e}")
    print("Saved results to", path)


if __name__ == "__main__":
    main()
