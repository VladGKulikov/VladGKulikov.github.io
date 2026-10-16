from __future__ import annotations

from itertools import product
from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "generated_experiment"
OUT.mkdir(exist_ok=True)


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def hyperprior_experiment() -> dict[str, float]:
    n = 8
    p0 = 0.05
    p1 = 0.50
    p_marg = 0.5 * (p0 + p1)

    factorized = n * binary_entropy(p_marg)
    conditional_one = 0.5 * binary_entropy(p0) + 0.5 * binary_entropy(p1)
    hierarchical = 1.0 + n * conditional_one

    probs: list[float] = []
    for bits in product((0, 1), repeat=n):
        k = sum(bits)
        prob_z0 = p0**k * (1.0 - p0) ** (n - k)
        prob_z1 = p1**k * (1.0 - p1) ** (n - k)
        probs.append(0.5 * prob_z0 + 0.5 * prob_z1)

    ideal_joint = -sum(p * math.log2(p) for p in probs if p > 0.0)

    labels = ["Factorized", "Hierarchical", "Ideal joint"]
    values = [factorized, hierarchical, ideal_joint]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.bar(labels, values)
    ax.set_ylabel("Bits per 8-symbol block")
    ax.set_title("Toy hyperprior: model factorization versus ideal joint coding")
    for i, value in enumerate(values):
        ax.text(i, value + 0.04, f"{value:.3f}", ha="center", va="bottom")
    ax.set_ylim(0.0, max(values) + 0.7)
    fig.tight_layout()
    fig.savefig(OUT / "M14_hyperprior_rates.png", dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    return {
        "factorized_rate_bits": factorized,
        "hierarchical_rate_bits": hierarchical,
        "ideal_joint_entropy_bits": ideal_joint,
        "hierarchical_overhead_vs_joint": hierarchical - ideal_joint,
        "saving_vs_factorized": factorized - hierarchical,
    }


def gaussian_immse_experiment() -> dict[str, float]:
    variance = 2.0
    gamma = np.linspace(0.0, 12.0, 2001)
    mutual_information = 0.5 * np.log1p(variance * gamma)
    mmse = variance / (1.0 + variance * gamma)

    derivative_numeric = np.diff(mutual_information) / np.diff(gamma)
    gamma_mid = 0.5 * (gamma[:-1] + gamma[1:])
    derivative_theory = 0.5 * variance / (1.0 + variance * gamma_mid)
    max_error = float(np.max(np.abs(derivative_numeric - derivative_theory)))

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot(gamma, mutual_information, label=r"$I(X;Y_\gamma)$, nats")
    ax.plot(gamma, 0.5 * mmse, label=r"$\frac{1}{2}\,\mathrm{mmse}(\gamma)$")
    ax.set_xlabel("SNR $\\gamma$")
    ax.set_ylabel("Value")
    ax.set_title("Gaussian channel: mutual information and the I–MMSE derivative")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "M14_gaussian_immse.png", dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    alpha = 0.8
    sigma = 0.6
    var_xt = alpha**2 * variance + sigma**2
    posterior_coefficient = alpha * variance / var_xt
    x_grid = np.linspace(-4.0, 4.0, 1001)
    posterior_mean = posterior_coefficient * x_grid
    score_from_denoiser = (alpha * posterior_mean - x_grid) / sigma**2
    score_direct = -x_grid / var_xt
    score_max_error = float(np.max(np.abs(score_from_denoiser - score_direct)))

    return {
        "immse_finite_difference_max_error": max_error,
        "score_identity_max_error": score_max_error,
        "posterior_mean_coefficient": posterior_coefficient,
        "xt_variance": var_xt,
    }


def groupwise_symmetric_quantize(
    weights: np.ndarray,
    bits: int,
    group_size: int = 128,
    clip_percentile: float = 99.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize each group with its own stored FP16 scale.

    The payload uses signed symmetric integer levels ``[-qmax, qmax]``.  Each
    group is clipped independently and the resulting scale is rounded to FP16
    before both quantization and reconstruction, matching the 16-bit overhead
    charged by the rate calculation.
    """
    if bits < 2:
        raise ValueError("bits must be at least 2 for signed symmetric quantization")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if not 0.0 < clip_percentile <= 100.0:
        raise ValueError("clip_percentile must lie in (0, 100]")

    array = np.asarray(weights, dtype=float)
    flat = array.reshape(-1)
    if flat.size == 0:
        return array.copy(), np.empty(0, dtype=float)

    qmax = 2 ** (bits - 1) - 1
    reconstructed = np.empty_like(flat)
    n_groups = math.ceil(flat.size / group_size)
    scales = np.empty(n_groups, dtype=float)
    smallest_fp16 = float(np.finfo(np.float16).smallest_subnormal)

    for group_index, start in enumerate(range(0, flat.size, group_size)):
        group = flat[start:start + group_size]
        clip = float(np.percentile(np.abs(group), clip_percentile))
        if clip == 0.0:
            scale = 1.0
        else:
            scale = float(np.float16(clip / qmax))
            if scale == 0.0:
                scale = smallest_fp16

        q = np.clip(np.rint(group / scale), -qmax, qmax)
        reconstructed[start:start + group.size] = q * scale
        scales[group_index] = scale

    return reconstructed.reshape(array.shape), scales


def quantization_experiment() -> list[dict[str, float]]:
    rng = np.random.default_rng(14)
    group_size = 128
    n_groups = 1_563
    n = n_groups * group_size
    weights = rng.normal(0.0, 1.0, size=n)
    outlier_mask = rng.random(n) < 0.005
    weights[outlier_mask] += rng.normal(0.0, 8.0, size=int(outlier_mask.sum()))

    variance = float(np.var(weights))
    scale_bits = 16
    clip_percentile = 99.9
    rows: list[dict[str, float]] = []

    for bits in (2, 3, 4, 6, 8):
        reconstructed, scales = groupwise_symmetric_quantize(
            weights,
            bits,
            group_size=group_size,
            clip_percentile=clip_percentile,
        )
        if scales.size != n_groups:
            raise RuntimeError("unexpected number of quantization groups")
        mse = float(np.mean((weights - reconstructed) ** 2))
        effective_bits = bits + scale_bits / group_size
        rows.append(
            {
                "bits": float(bits),
                "effective_bits_per_weight": effective_bits,
                "mse": mse,
                "nmse": mse / variance,
                "mean_fp16_scale": float(np.mean(scales)),
            }
        )

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(
        [row["effective_bits_per_weight"] for row in rows],
        [row["nmse"] for row in rows],
        marker="o",
    )
    for row in rows:
        ax.annotate(
            f"INT{int(row['bits'])}",
            (row["effective_bits_per_weight"], row["nmse"]),
            xytext=(4, 5),
            textcoords="offset points",
        )
    ax.set_xlabel("Effective bits per weight (including scale overhead)")
    ax.set_ylabel("Normalized weight MSE")
    ax.set_yscale("log")
    ax.set_title("Group-wise weight quantization: a local rate–distortion curve")
    fig.tight_layout()
    fig.savefig(OUT / "M14_weight_quantization_rd.png", dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    return rows


def main() -> None:
    print("=== Hyperprior rate accounting ===")
    hyper = hyperprior_experiment()
    for key, value in hyper.items():
        print(f"{key}: {value:.6f}")

    print("\n=== Gaussian I-MMSE ===")
    immse = gaussian_immse_experiment()
    for key, value in immse.items():
        print(f"{key}: {value:.10f}")

    print("\n=== Group-wise symmetric weight quantization ===")
    rows = quantization_experiment()
    print("scheme: group_size=128, per-group p99.9 clipping, one FP16 scale per group")
    print("bits  effective_bpw   nmse         mse")
    for row in rows:
        print(
            f"{int(row['bits']):>4d}  "
            f"{row['effective_bits_per_weight']:>13.6f}  "
            f"{row['nmse']:>11.8f}  "
            f"{row['mse']:>11.8f}"
        )


if __name__ == "__main__":
    main()
