from __future__ import annotations

import math
import numpy as np


def binary_entropy(p: float) -> float:
    """Binary entropy in bits."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def mutual_information(p_x: np.ndarray, channel: np.ndarray) -> float:
    """I(X;Y) in bits for an input distribution and a DMC matrix."""
    p_y = p_x @ channel
    joint = p_x[:, None] * channel
    x, y = np.nonzero(joint > 0)
    # Only positive joint mass contributes; unreachable outputs have p_y=0.
    return float(np.sum(joint[x, y] * np.log2(channel[x, y] / p_y[y])))


def blahut_arimoto(
    channel: np.ndarray,
    tol: float = 1e-12,
    max_iter: int = 100_000,
) -> tuple[float, np.ndarray, int]:
    """Return DMC capacity, a maximizing input distribution, and iterations."""
    channel = np.asarray(channel, dtype=float)
    if channel.ndim != 2 or np.any(channel < 0):
        raise ValueError("channel must be a nonnegative 2D array")
    if not np.allclose(channel.sum(axis=1), 1.0):
        raise ValueError("each row of channel must sum to one")

    p_x = np.full(channel.shape[0], 1.0 / channel.shape[0])
    for iteration in range(1, max_iter + 1):
        p_y = p_x @ channel
        ratio = np.divide(
            channel,
            p_y[None, :],
            out=np.ones_like(channel),
            where=channel > 0,
        )
        d_x = np.sum(
            np.where(channel > 0, channel * np.log(ratio), 0.0),
            axis=1,
        )
        p_new = p_x * np.exp(d_x)
        p_new /= p_new.sum()
        if np.max(np.abs(p_new - p_x)) < tol:
            p_x = p_new
            break
        p_x = p_new
    else:
        raise RuntimeError("Blahut-Arimoto did not converge")

    return mutual_information(p_x, channel), p_x, iteration


def random_noise_mask(
    rng: np.random.Generator,
    n: int,
    epsilon: float,
) -> np.uint32:
    """Generate an n-bit BSC noise mask packed into uint32."""
    bits = rng.random(n) < epsilon
    powers = np.left_shift(np.uint32(1), np.arange(n, dtype=np.uint32))
    return np.sum(bits.astype(np.uint32) * powers, dtype=np.uint32)


def simulate_random_codebook(
    rng: np.random.Generator,
    n: int,
    rate: float,
    epsilon: float,
    trials: int = 300,
) -> float:
    """Estimate block error for one distinct random binary codebook."""
    universe = 1 << n
    m = max(2, int(round(2 ** (n * rate))))
    if m > universe:
        raise ValueError("rate requires more distinct codewords than exist")

    codebook = rng.choice(universe, size=m, replace=False).astype(np.uint32)
    errors = 0

    for _ in range(trials):
        message = int(rng.integers(m))
        received = codebook[message] ^ random_noise_mask(rng, n, epsilon)
        distances = np.bitwise_count(codebook ^ received)
        min_distance = distances.min()
        candidates = np.flatnonzero(distances == min_distance)
        decoded = int(rng.choice(candidates))
        errors += decoded != message

    return errors / trials


def main() -> None:
    bsc = np.array([[0.9, 0.1], [0.1, 0.9]])
    bec = np.array([[0.9, 0.0, 0.1], [0.0, 0.9, 0.1]])
    z_channel = np.array([[1.0, 0.0], [0.25, 0.75]])

    print("Blahut-Arimoto")
    for name, channel in [
        ("BSC(0.1)", bsc),
        ("BEC(0.1)", bec),
        ("Z(0.25)", z_channel),
    ]:
        capacity, p_x, iterations = blahut_arimoto(channel)
        print(
            f"{name:9s}: C={capacity:.9f}, "
            f"p_X={p_x}, iterations={iterations}"
        )

    epsilon = 0.1
    capacity = 1.0 - binary_entropy(epsilon)
    print(f"\nBSC(0.1) capacity: {capacity:.9f} bits/use")

    rng = np.random.default_rng(7)
    blocklengths = [8, 12, 16, 20]
    rates = [0.25, 0.80]
    n_codebooks = 10
    trials = 300

    for rate in rates:
        side = "below" if rate < capacity else "above"
        print(f"\nR={rate:.2f} ({side} capacity)")
        for n in blocklengths:
            estimates = np.array(
                [
                    simulate_random_codebook(rng, n, rate, epsilon, trials)
                    for _ in range(n_codebooks)
                ]
            )
            m = max(2, int(round(2 ** (n * rate))))
            print(
                f"n={n:2d}, M={m:6d}: "
                f"mean P_e={estimates.mean():.6f}, "
                f"std={estimates.std(ddof=1):.6f}"
            )


if __name__ == "__main__":
    main()
