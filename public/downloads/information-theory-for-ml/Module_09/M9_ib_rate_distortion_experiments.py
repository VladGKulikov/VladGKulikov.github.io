from __future__ import annotations

"""Reproducible numerical experiments for Module 9.

The script:
1. plots Bernoulli and Gaussian rate-distortion curves;
2. solves a finite Information Bottleneck problem by self-consistent updates;
3. verifies the exact VAE-rate decomposition on a discrete toy encoder.

All logarithms used for information quantities are base 2 unless stated otherwise.
"""

from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "generated_experiment"
OUT.mkdir(exist_ok=True)
EPS = 1e-15


def normalize(a: np.ndarray, axis: int = -1) -> np.ndarray:
    total = a.sum(axis=axis, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("Cannot normalize non-positive mass")
    return a / total


def xlog2x(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, x * np.log2(x), 0.0)


def binary_entropy(p: np.ndarray | float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def mutual_information(joint: np.ndarray) -> float:
    joint = np.asarray(joint, dtype=float)
    joint = joint / joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    ratio = np.divide(joint, px @ py, out=np.ones_like(joint), where=joint > 0)
    return float(np.sum(np.where(joint > 0, joint * np.log2(ratio), 0.0)))


def kl_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return all pairwise KL divergences D(p_i || q_j) in bits."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p_safe = np.clip(p, EPS, 1)
    q_safe = np.clip(q, EPS, 1)
    return np.sum(p[:, None, :] * (np.log2(p_safe[:, None, :]) - np.log2(q_safe[None, :, :])), axis=-1)


def plot_rate_distortion() -> None:
    p = 0.25
    d_b = np.linspace(0, p, 250)
    r_b = binary_entropy(p) - binary_entropy(d_b)
    r_b[-1] = 0.0

    sigma2 = 1.0
    d_g = np.geomspace(0.01, sigma2, 250)
    r_g = 0.5 * np.log2(sigma2 / d_g)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(d_b, r_b, label=r"Bernoulli($p=0.25$), Hamming")
    ax.plot(d_g, r_g, label=r"Gaussian($\sigma^2=1$), MSE")
    ax.set_xlabel("Allowed distortion D")
    ax.set_ylabel("Rate R(D), bits/source symbol")
    ax.set_title("Two rate-distortion benchmarks")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "M9_experiment_rate_distortion.png", dpi=180)
    plt.close(fig)


def ib_metrics(px: np.ndarray, py_given_x: np.ndarray, qt_given_x: np.ndarray) -> tuple[float, float]:
    pxt = px[:, None] * qt_given_x
    ixt = mutual_information(pxt)
    pty = np.einsum("x,xt,xy->ty", px, qt_given_x, py_given_x)
    ity = mutual_information(pty)
    return ixt, ity


def solve_ib(
    px: np.ndarray,
    py_given_x: np.ndarray,
    n_t: int,
    beta: float,
    rng: np.random.Generator,
    restarts: int = 24,
    max_iter: int = 5000,
    tol: float = 1e-12,
) -> tuple[np.ndarray, float, float, float]:
    """Solve a finite IB problem by self-consistent updates.

    Returns q(t|x), I(X;T), I(T;Y), and the IB objective I(X;T)-beta I(T;Y).
    Multiple random restarts are used because the fixed-point equations may have
    several stationary solutions.
    """
    best: tuple[np.ndarray, float, float, float] | None = None
    n_x = px.size

    for restart in range(restarts):
        if restart == 0:
            qtx = np.full((n_x, n_t), 1.0 / n_t)
            qtx += 1e-3 * rng.random((n_x, n_t))
        else:
            qtx = rng.dirichlet(np.ones(n_t), size=n_x)
        qtx = normalize(qtx, axis=1)

        for _ in range(max_iter):
            pt = px @ qtx
            pt = np.clip(pt, EPS, None)
            pyt = np.einsum("x,xt,xy->ty", px, qtx, py_given_x)
            pyt = normalize(pyt, axis=1)
            distortion = kl_rows(py_given_x, pyt)

            # Exponents use natural exp. Convert KL bits to nats by ln(2).
            log_new = np.log(pt)[None, :] - beta * math.log(2.0) * distortion
            log_new -= log_new.max(axis=1, keepdims=True)
            new = normalize(np.exp(log_new), axis=1)
            if np.max(np.abs(new - qtx)) < tol:
                qtx = new
                break
            qtx = new

        ixt, ity = ib_metrics(px, py_given_x, qtx)
        objective = ixt - beta * ity
        candidate = (qtx.copy(), ixt, ity, objective)
        if best is None or objective < best[3] - 1e-11:
            best = candidate

    assert best is not None
    return best


def run_ib_experiment() -> list[tuple[float, float, float]]:
    px = np.full(6, 1 / 6)
    # Three progressively more positive regions. Neighboring states have similar
    # predictive distributions, making the bottleneck geometry visible.
    p1 = np.array([0.03, 0.08, 0.30, 0.70, 0.92, 0.97])
    py_given_x = np.column_stack([1 - p1, p1])
    rng = np.random.default_rng(20260713)
    betas = np.array([0.05, 0.10, 0.20, 0.40, 0.75, 1.25, 2.0, 3.5, 6.0, 10.0, 18.0])
    rows: list[tuple[float, float, float]] = []

    for beta in betas:
        _, ixt, ity, _ = solve_ib(px, py_given_x, n_t=6, beta=float(beta), rng=rng)
        rows.append((float(beta), ixt, ity))

    ixy = mutual_information(px[:, None] * py_given_x)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ixt = np.array([r[1] for r in rows])
    ity = np.array([r[2] for r in rows])
    ax.plot(ixt, ity, marker="o", label="IB fixed points")
    for beta, x, y in rows[::2]:
        ax.annotate(f"β={beta:g}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.axhline(ixy, linestyle="--", linewidth=1.2, label=f"I(X;Y)={ixy:.3f} bits")
    ax.set_xlabel("I(X;T), bits")
    ax.set_ylabel("I(T;Y), bits")
    ax.set_title("Toy Information Bottleneck frontier")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "M9_experiment_ib_frontier.png", dpi=180)
    plt.close(fig)
    return rows


def expected_conditional_kl(px: np.ndarray, qz_given_x: np.ndarray, prior: np.ndarray) -> float:
    """Average KL in bits for normalized probability arrays, without smoothing."""
    joint = px[:, None] * qz_given_x
    positive = joint > 0
    prior_grid = np.broadcast_to(prior, qz_given_x.shape)
    # A prior must cover every latent with positive joint mass. Rows with
    # px[x] = 0 and terms with q(z|x) = 0 contribute nothing, including 0 log 0.
    if np.any(prior_grid[positive] == 0):
        return math.inf
    return float(np.sum(joint[positive] * (
        np.log2(qz_given_x[positive]) - np.log2(prior_grid[positive])
    )))


def run_rate_decomposition(bad_prior: np.ndarray | None = None) -> dict[str, float]:
    px = np.array([0.40, 0.35, 0.25])
    qz_given_x = np.array([
        [0.80, 0.15, 0.05],
        [0.20, 0.65, 0.15],
        [0.05, 0.20, 0.75],
    ])
    qz = px @ qz_given_x
    good_prior = qz.copy()
    if bad_prior is None:
        bad_prior = np.array([0.78, 0.12, 0.10])

    joint = px[:, None] * qz_given_x
    ixz = mutual_information(joint)
    # Ordinary KL is the same calculation with a single source state.
    mismatch_good = expected_conditional_kl(np.ones(1), qz[None, :], good_prior)
    mismatch_bad = expected_conditional_kl(np.ones(1), qz[None, :], bad_prior)
    rate_good = expected_conditional_kl(px, qz_given_x, good_prior)
    rate_bad = expected_conditional_kl(px, qz_given_x, bad_prior)

    if not np.isclose(rate_good, ixz + mismatch_good, atol=1e-12):
        raise AssertionError("Good-prior decomposition failed")
    if not np.isclose(rate_bad, ixz + mismatch_bad, atol=1e-12):
        raise AssertionError("Bad-prior decomposition failed")

    labels = ["I(X;Z)", "prior mismatch", "total KL-rate"]
    good = [ixz, mismatch_good, rate_good]
    bad = [ixz, mismatch_bad, rate_bad]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.7, 5.0))
    ax.bar(x - width / 2, good, width, label="prior = aggregate posterior")
    finite = np.isfinite(bad)
    ax.bar(x[finite] + width / 2, np.asarray(bad)[finite], width,
           label="mismatched prior")
    for position in x[~finite] + width / 2:
        ax.text(position, 0.78, "∞", transform=ax.get_xaxis_transform(),
                ha="center", color="C1", fontsize=22)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Bits")
    ax.set_title("VAE-rate decomposition: information plus prior mismatch")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    if not np.all(finite):
        # Text annotations do not participate in Matplotlib's autoscaling.
        ax.set_xlim(-0.6, len(labels) - 0.4)
        fig.text(0.5, 0.02, "∞: prior assigns zero mass to a possible latent",
                 ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.06 if not np.all(finite) else 0, 1, 1))
    fig.savefig(OUT / "M9_experiment_rate_decomposition.png", dpi=180)
    plt.close(fig)

    return {
        "I(X;Z)": ixz,
        "good_mismatch": mismatch_good,
        "good_rate": rate_good,
        "bad_mismatch": mismatch_bad,
        "bad_rate": rate_bad,
    }


def main() -> None:
    plot_rate_distortion()
    rows = run_ib_experiment()
    decomposition = run_rate_decomposition()

    print("Toy IB results (bits):")
    print("beta\tI(X;T)\tI(T;Y)")
    for beta, ixt, ity in rows:
        print(f"{beta:>5g}\t{ixt:.6f}\t{ity:.6f}")

    print("\nVAE-rate decomposition (bits):")
    for key, value in decomposition.items():
        print(f"{key}: {value:.9f}")
    print("\nGenerated figures:")
    for path in sorted(OUT.glob("M9_experiment_*.png")):
        print(path.name)


if __name__ == "__main__":
    main()
