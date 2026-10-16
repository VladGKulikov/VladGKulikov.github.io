#!/usr/bin/env python3
"""Reference NumPy utilities for Module 7: reward models and preferences.

The file intentionally stays independent of PyTorch and network access.  It
covers the mathematical objects used in the main route and the CPU exercises:
Bradley--Terry and Plackett--Luce likelihoods, calibration diagnostics,
ensemble aggregation, position-swap aggregation for an LLM judge, a narrow
label-noise model, and a reproducible toy overoptimization experiment.

These functions are pedagogical references, not a production reward-model
trainer.  In particular, ``fit_pairwise_scores`` fits one free scalar per item;
it does not learn a Transformer representation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

ArrayLike = np.ndarray | Sequence[float] | float


def _as_float_array(x: ArrayLike) -> np.ndarray:
    """Convert input to a finite ``float64`` NumPy array."""
    arr = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("all values must be finite")
    return arr


def sigmoid(x: ArrayLike) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    arr = _as_float_array(x)
    out = np.empty_like(arr)
    nonnegative = arr >= 0
    out[nonnegative] = 1.0 / (1.0 + np.exp(-arr[nonnegative]))
    exp_x = np.exp(arr[~nonnegative])
    out[~nonnegative] = exp_x / (1.0 + exp_x)
    return out


def softplus(x: ArrayLike) -> np.ndarray:
    """Numerically stable ``log(1 + exp(x))``."""
    arr = _as_float_array(x)
    return np.log1p(np.exp(-np.abs(arr))) + np.maximum(arr, 0.0)


def binary_log_loss(probabilities: ArrayLike, targets: ArrayLike, *, eps: float = 1e-12) -> float:
    """Mean binary log loss for hard or soft targets in ``[0, 1]``."""
    p = _as_float_array(probabilities)
    y = _as_float_array(targets)
    if p.shape != y.shape:
        raise ValueError("probabilities and targets must have the same shape")
    if np.any((p < 0) | (p > 1)) or np.any((y < 0) | (y > 1)):
        raise ValueError("probabilities and targets must lie in [0, 1]")
    p = np.clip(p, eps, 1.0 - eps)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log1p(-p))))


def pairwise_nll(delta: ArrayLike, targets: ArrayLike = 1.0) -> np.ndarray:
    r"""Bradley--Terry negative log-likelihood from score differences.

    With ``p = sigmoid(delta)`` and a target ``y`` in ``[0, 1]``:

    ``loss = softplus(delta) - y * delta``.

    ``y=1`` denotes that the first item is preferred, ``y=0`` denotes the
    reverse preference, and ``y=0.5`` is a soft tie target.  A soft tie is not
    the Davidson tie model; it only asks for a zero score difference under the
    binary logistic likelihood.
    """
    d = _as_float_array(delta)
    y = _as_float_array(targets)
    try:
        y = np.broadcast_to(y, d.shape)
    except ValueError as exc:
        raise ValueError("targets must be broadcastable to delta") from exc
    if np.any((y < 0) | (y > 1)):
        raise ValueError("targets must lie in [0, 1]")
    return softplus(d) - y * d


def pairwise_gradient(delta: ArrayLike, targets: ArrayLike = 1.0) -> np.ndarray:
    """Derivative of ``pairwise_nll`` with respect to ``delta``."""
    d = _as_float_array(delta)
    y = _as_float_array(targets)
    try:
        y = np.broadcast_to(y, d.shape)
    except ValueError as exc:
        raise ValueError("targets must be broadcastable to delta") from exc
    if np.any((y < 0) | (y > 1)):
        raise ValueError("targets must lie in [0, 1]")
    return sigmoid(d) - y


def bt_metrics(chosen_scores: ArrayLike, rejected_scores: ArrayLike) -> dict[str, float | np.ndarray]:
    """Return probabilities, per-pair losses, mean loss, accuracy, and margin."""
    chosen = _as_float_array(chosen_scores)
    rejected = _as_float_array(rejected_scores)
    if chosen.shape != rejected.shape:
        raise ValueError("chosen_scores and rejected_scores must have the same shape")
    if chosen.size == 0:
        raise ValueError("at least one pair is required")
    delta = chosen - rejected
    losses = pairwise_nll(delta, 1.0)
    probabilities = sigmoid(delta)
    return {
        "delta": delta,
        "probabilities": probabilities,
        "losses": losses,
        "mean_loss": float(np.mean(losses)),
        "accuracy": float(np.mean(delta > 0)),
        "mean_margin": float(np.mean(delta)),
    }


def logsumexp(values: ArrayLike) -> float:
    """Stable scalar log-sum-exp."""
    x = _as_float_array(values).reshape(-1)
    if x.size == 0:
        raise ValueError("values must be non-empty")
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


def plackett_luce_nll(scores: ArrayLike, ranking: Sequence[int]) -> float:
    r"""Negative log-likelihood of a complete ranking under Plackett--Luce.

    ``ranking[0]`` is the most preferred item.  The final choice contributes
    zero because only one item remains in its denominator.
    """
    s = _as_float_array(scores).reshape(-1)
    order = np.asarray(ranking, dtype=np.int64)
    if order.ndim != 1 or len(order) != len(s):
        raise ValueError("ranking must be a complete one-dimensional permutation")
    if set(order.tolist()) != set(range(len(s))):
        raise ValueError("ranking must be a permutation of item indices")
    loss = 0.0
    remaining = list(order.tolist())
    while len(remaining) > 1:
        selected = remaining[0]
        loss += logsumexp(s[remaining]) - float(s[selected])
        remaining.pop(0)
    return float(loss)


@dataclass(frozen=True)
class PairObservation:
    """One pairwise observation over item indices.

    ``target`` is the probability target for ``first`` beating ``second``.
    Use 1.0 for a first-item win, 0.0 for a second-item win, and 0.5 for a
    soft tie.  ``weight`` must be positive.
    """

    first: int
    second: int
    target: float = 1.0
    weight: float = 1.0


def fit_pairwise_scores(
    n_items: int,
    observations: Iterable[PairObservation],
    *,
    learning_rate: float = 0.1,
    steps: int = 2_000,
    l2: float = 0.0,
) -> np.ndarray:
    """Fit one centered Bradley--Terry score per item by full-batch GD.

    Centering fixes the additive gauge.  ``l2`` is optional and changes the
    statistical model; it is included only to make poorly connected examples
    numerically well behaved.
    """
    if n_items <= 0:
        raise ValueError("n_items must be positive")
    if learning_rate <= 0 or steps <= 0 or l2 < 0:
        raise ValueError("invalid optimization hyperparameters")
    obs = list(observations)
    if not obs:
        raise ValueError("at least one observation is required")
    for o in obs:
        if not (0 <= o.first < n_items and 0 <= o.second < n_items):
            raise ValueError("observation index out of range")
        if o.first == o.second:
            raise ValueError("an item cannot be compared with itself")
        if not (0.0 <= o.target <= 1.0) or o.weight <= 0:
            raise ValueError("invalid target or weight")

    scores = np.zeros(n_items, dtype=np.float64)
    total_weight = float(sum(o.weight for o in obs))
    for _ in range(steps):
        grad = np.zeros_like(scores)
        for o in obs:
            d = scores[o.first] - scores[o.second]
            g = float(pairwise_gradient(d, o.target)) * o.weight
            grad[o.first] += g
            grad[o.second] -= g
        grad /= total_weight
        if l2:
            grad += l2 * scores
        scores -= learning_rate * grad
        scores -= scores.mean()  # additive gauge
    return scores


def brier_score(probabilities: ArrayLike, targets: ArrayLike) -> float:
    """Mean squared error of binary probabilities."""
    p = _as_float_array(probabilities)
    y = _as_float_array(targets)
    if p.shape != y.shape:
        raise ValueError("probabilities and targets must have the same shape")
    if np.any((p < 0) | (p > 1)) or np.any((y < 0) | (y > 1)):
        raise ValueError("probabilities and targets must lie in [0, 1]")
    return float(np.mean((p - y) ** 2))


def reliability_bins(
    probabilities: ArrayLike,
    targets: ArrayLike,
    *,
    n_bins: int = 10,
) -> list[dict[str, float | int]]:
    """Equal-width reliability bins on ``[0, 1]``.

    Empty bins are omitted.  The right endpoint 1.0 belongs to the last bin.
    """
    p = _as_float_array(probabilities).reshape(-1)
    y = _as_float_array(targets).reshape(-1)
    if p.shape != y.shape or p.size == 0:
        raise ValueError("probabilities and targets must be non-empty and aligned")
    if np.any((p < 0) | (p > 1)) or np.any((y < 0) | (y > 1)):
        raise ValueError("probabilities and targets must lie in [0, 1]")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    indices = np.minimum((p * n_bins).astype(int), n_bins - 1)
    result: list[dict[str, float | int]] = []
    for b in range(n_bins):
        mask = indices == b
        if not np.any(mask):
            continue
        result.append(
            {
                "bin": b,
                "count": int(np.sum(mask)),
                "mean_confidence": float(np.mean(p[mask])),
                "empirical_frequency": float(np.mean(y[mask])),
            }
        )
    return result


def expected_calibration_error(
    probabilities: ArrayLike,
    targets: ArrayLike,
    *,
    n_bins: int = 10,
) -> float:
    """Conventional equal-width expected calibration error (ECE)."""
    p = _as_float_array(probabilities).reshape(-1)
    bins = reliability_bins(p, targets, n_bins=n_bins)
    total = len(p)
    return float(
        sum(
            (int(b["count"]) / total)
            * abs(float(b["mean_confidence"]) - float(b["empirical_frequency"]))
            for b in bins
        )
    )


def ensemble_mean_std(member_scores: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Mean and population standard deviation across ensemble members.

    ``member_scores`` has shape ``[n_members, n_candidates]``.
    """
    scores = _as_float_array(member_scores)
    if scores.ndim != 2 or scores.shape[0] < 1:
        raise ValueError("member_scores must have shape [members, candidates]")
    return scores.mean(axis=0), scores.std(axis=0, ddof=0)


def lower_confidence_score(member_scores: ArrayLike, *, coefficient: float = 1.0) -> np.ndarray:
    r"""Heuristic conservative score ``mean - coefficient * std``.

    The name is descriptive, not a finite-sample confidence guarantee.  Such a
    guarantee would require assumptions about independence, calibration, and
    the data-generating process that this utility does not make.
    """
    if coefficient < 0:
        raise ValueError("coefficient must be non-negative")
    mean, std = ensemble_mean_std(member_scores)
    return mean - coefficient * std


def aggregate_swapped_pairwise(prob_a_when_first: float, prob_b_when_first: float) -> float:
    """Aggregate a pairwise judge after presenting both answer orders.

    Parameters are:
    * ``prob_a_when_first``: judge probability that A wins in prompt ``A, B``;
    * ``prob_b_when_first``: judge probability that B wins in prompt ``B, A``.

    The second quantity is converted to an A-win probability before averaging.
    """
    p1 = float(prob_a_when_first)
    p2 = float(prob_b_when_first)
    if not (0 <= p1 <= 1 and 0 <= p2 <= 1):
        raise ValueError("probabilities must lie in [0, 1]")
    return 0.5 * (p1 + (1.0 - p2))


def scalarize_rubric(values: ArrayLike, weights: ArrayLike) -> float:
    """Weighted sum of a rubric vector after validating dimensions."""
    v = _as_float_array(values).reshape(-1)
    w = _as_float_array(weights).reshape(-1)
    if v.shape != w.shape or v.size == 0:
        raise ValueError("values and weights must be non-empty and aligned")
    return float(np.dot(v, w))


def constrained_scalarize(
    values: ArrayLike,
    utility_weights: ArrayLike,
    *,
    constraint_values: ArrayLike | None = None,
    lower_bounds: ArrayLike | None = None,
) -> float:
    """Return weighted utility or ``-inf`` when a hard lower bound is violated."""
    utility = scalarize_rubric(values, utility_weights)
    if constraint_values is None and lower_bounds is None:
        return utility
    if constraint_values is None or lower_bounds is None:
        raise ValueError("constraint_values and lower_bounds must be provided together")
    c = _as_float_array(constraint_values).reshape(-1)
    b = _as_float_array(lower_bounds).reshape(-1)
    if c.shape != b.shape:
        raise ValueError("constraint_values and lower_bounds must have the same shape")
    return utility if np.all(c >= b) else float("-inf")


def noisy_label_oracle_accuracy(flip_probability: float) -> float:
    """Expected observed-label accuracy of a latent-order oracle under iid flips."""
    eta = float(flip_probability)
    if not 0 <= eta <= 1:
        raise ValueError("flip_probability must lie in [0, 1]")
    return 1.0 - eta


def simulate_noisy_label_accuracy(
    *,
    n_pairs: int = 100_000,
    flip_probability: float = 0.2,
    seed: int = 7,
) -> float:
    """Monte Carlo check of the narrow independent label-flip model."""
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")
    eta = float(flip_probability)
    if not 0 <= eta <= 1:
        raise ValueError("flip_probability must lie in [0, 1]")
    rng = np.random.default_rng(seed)
    flips = rng.random(n_pairs) < eta
    return float(np.mean(~flips))


def overoptimization_experiment(
    pool_sizes: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
    *,
    trials: int = 4_000,
    seed: int = 7,
    proxy_spurious_weight: float = 1.0,
    gold_penalty: float = 1.0,
    spurious_threshold: float = 1.0,
) -> dict[str, np.ndarray]:
    """Reproducible toy selection experiment with proxy overoptimization.

    Each candidate has a useful feature ``g`` and a spurious feature ``h``.
    The proxy score is ``g + w*h``.  The intended score rewards ``g`` but
    penalizes large positive ``h`` after a threshold.  For each pool size, the
    candidate with maximum proxy score is selected.

    The construction is illustrative.  It does not reproduce any paper's
    language model, reward model, or fitted scaling law.
    """
    sizes = np.asarray(pool_sizes, dtype=np.int64)
    if sizes.ndim != 1 or sizes.size == 0 or np.any(sizes <= 0):
        raise ValueError("pool_sizes must be a non-empty sequence of positive integers")
    if trials <= 0 or proxy_spurious_weight < 0 or gold_penalty < 0:
        raise ValueError("invalid experiment parameters")
    rng = np.random.default_rng(seed)
    proxy_means = np.empty(len(sizes), dtype=np.float64)
    gold_means = np.empty(len(sizes), dtype=np.float64)
    useful_means = np.empty(len(sizes), dtype=np.float64)
    spurious_means = np.empty(len(sizes), dtype=np.float64)

    for j, n in enumerate(sizes):
        g = rng.normal(size=(trials, int(n)))
        h = rng.normal(size=(trials, int(n)))
        proxy = g + proxy_spurious_weight * h
        gold = g - gold_penalty * np.maximum(h - spurious_threshold, 0.0) ** 2
        index = np.argmax(proxy, axis=1)
        row = np.arange(trials)
        proxy_means[j] = float(np.mean(proxy[row, index]))
        gold_means[j] = float(np.mean(gold[row, index]))
        useful_means[j] = float(np.mean(g[row, index]))
        spurious_means[j] = float(np.mean(h[row, index]))

    return {
        "pool_sizes": sizes,
        "proxy": proxy_means,
        "gold": gold_means,
        "useful_feature": useful_means,
        "spurious_feature": spurious_means,
    }


def validate_pairwise_split(
    train_prompt_ids: Sequence[str],
    eval_prompt_ids: Sequence[str],
) -> dict[str, int | bool]:
    """Check prompt-level overlap between a training and evaluation split."""
    train = set(train_prompt_ids)
    eval_ = set(eval_prompt_ids)
    overlap = train & eval_
    return {
        "train_prompts": len(train),
        "eval_prompts": len(eval_),
        "overlap_count": len(overlap),
        "is_disjoint": not overlap,
    }


__all__ = [
    "PairObservation",
    "aggregate_swapped_pairwise",
    "binary_log_loss",
    "brier_score",
    "bt_metrics",
    "constrained_scalarize",
    "ensemble_mean_std",
    "expected_calibration_error",
    "fit_pairwise_scores",
    "logsumexp",
    "lower_confidence_score",
    "noisy_label_oracle_accuracy",
    "overoptimization_experiment",
    "pairwise_gradient",
    "pairwise_nll",
    "plackett_luce_nll",
    "reliability_bins",
    "scalarize_rubric",
    "sigmoid",
    "simulate_noisy_label_accuracy",
    "softplus",
    "validate_pairwise_split",
]
