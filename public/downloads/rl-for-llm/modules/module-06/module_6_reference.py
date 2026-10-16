#!/usr/bin/env python3
"""Reference NumPy utilities for RL for LLM, Module 6.

The module isolates the probability-accounting operations needed when an
autoregressive LLM is treated as a policy: causal alignment, response masks,
sequence metrics, exact and sampled KL quantities, decoding transforms, and a
minimal rollout-batch contract. It is educational reference code rather than a
replacement for fused production kernels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class SequenceMetrics:
    """Response-only sequence statistics."""

    token_count: Array | int
    sum_logprob: Array | float
    mean_nll: Array | float
    perplexity: Array | float


@dataclass(frozen=True)
class KLEstimatorValues:
    """Selected-token k1, k2, and k3 values for samples from p."""

    k1: Array
    k2: Array
    k3: Array
    log_ratio_q_over_p: Array
    ratio_q_over_p: Array


@dataclass(frozen=True)
class StopEvent:
    """Task-level interpretation of a machine-readable generation stop reason."""

    stop_reason: str
    terminated: bool
    truncated: bool
    inferred: bool = False




def stable_log_softmax(logits: Array, axis: int = -1) -> Array:
    """Numerically stable log-softmax computed in float64."""
    z = np.asarray(logits, dtype=np.float64)
    if z.size == 0:
        raise ValueError("logits must be non-empty")
    z_max = np.max(z, axis=axis, keepdims=True)
    shifted = z - z_max
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def log_softmax(logits: Array, axis: int = -1) -> Array:
    """Alias used in course snippets."""
    return stable_log_softmax(logits, axis=axis)


def softmax(logits: Array, axis: int = -1) -> Array:
    """Numerically stable softmax."""
    return np.exp(stable_log_softmax(logits, axis=axis))


def selected_token_logprobs(logits: Array, token_ids: Array) -> Array:
    """Gather selected-token log-probabilities from ``(..., vocabulary)`` logits."""
    z = np.asarray(logits)
    ids = np.asarray(token_ids, dtype=np.int64)
    if z.ndim < 2:
        raise ValueError("logits must have at least two dimensions")
    if ids.shape != z.shape[:-1]:
        raise ValueError(f"token_ids shape {ids.shape} must equal {z.shape[:-1]}")
    if np.any(ids < 0) or np.any(ids >= z.shape[-1]):
        raise ValueError("token id is outside the vocabulary")
    all_logprobs = stable_log_softmax(z, axis=-1)
    return np.take_along_axis(all_logprobs, ids[..., None], axis=-1)[..., 0]


def causal_response_logprobs(
    model_logits: Array,
    input_ids: Array,
    response_token_mask: Array,
) -> tuple[Array, Array]:
    """Apply the causal one-position shift and return selected log-probs and mask.

    ``model_logits[..., t, :]`` predicts ``input_ids[..., t + 1]``. The supplied
    mask is defined over original token positions, so its first position is
    dropped together with the unscorable first token.
    """
    z = np.asarray(model_logits)
    ids = np.asarray(input_ids, dtype=np.int64)
    mask = np.asarray(response_token_mask, dtype=bool)
    if z.ndim < 2:
        raise ValueError("model_logits must have shape (..., length, vocabulary)")
    if ids.shape != z.shape[:-1] or mask.shape != ids.shape:
        raise ValueError("input_ids and response_token_mask must match logits without vocabulary axis")
    if z.shape[-2] < 2:
        raise ValueError("at least two token positions are required")
    return selected_token_logprobs(z[..., :-1, :], ids[..., 1:]), mask[..., 1:]


def masked_sequence_metrics(selected_logprobs: Array, mask: Array) -> SequenceMetrics:
    """Compute token count, sequence sum, mean NLL, and perplexity.

    A one-dimensional input returns scalar fields. A two-dimensional input
    returns one value per sequence.
    """
    lp = np.asarray(selected_logprobs, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if lp.shape != m.shape:
        raise ValueError("selected_logprobs and mask must have identical shapes")
    squeeze = lp.ndim == 1
    if squeeze:
        lp = lp[None, :]
        m = m[None, :]
    if lp.ndim != 2:
        raise ValueError("expected one sequence or a batch of sequences")
    counts = np.sum(m, axis=1)
    if np.any(counts == 0):
        raise ValueError("every sequence must contain at least one active token")
    if np.any(~np.isfinite(lp[m])):
        raise ValueError("active log-probabilities must be finite")
    sums = np.sum(np.where(m, lp, 0.0), axis=1, dtype=np.float64)
    mean_nll = -sums / counts
    perplexity = np.exp(mean_nll)
    if squeeze:
        return SequenceMetrics(
            token_count=int(counts[0]),
            sum_logprob=float(sums[0]),
            mean_nll=float(mean_nll[0]),
            perplexity=float(perplexity[0]),
        )
    return SequenceMetrics(
        token_count=counts.astype(np.int64),
        sum_logprob=sums,
        mean_nll=mean_nll,
        perplexity=perplexity,
    )

def masked_sequence_stats(logits: Array, token_ids: Array, mask: Array) -> SequenceMetrics:
    """Compatibility helper: gather selected log-probs and compute metrics."""
    return masked_sequence_metrics(selected_token_logprobs(logits, token_ids), mask)


def causal_response_stats(model_logits: Array, input_ids: Array, response_token_mask: Array) -> SequenceMetrics:
    """Compatibility helper for shifted response-only metrics."""
    selected, shifted_mask = causal_response_logprobs(model_logits, input_ids, response_token_mask)
    return masked_sequence_metrics(selected, shifted_mask)


def exact_kl_from_logits(p_logits: Array, q_logits: Array, axis: int = -1) -> Array:
    """Return ``D_KL(p || q)`` for softmax policies represented by logits."""
    p_z = np.asarray(p_logits, dtype=np.float64)
    q_z = np.asarray(q_logits, dtype=np.float64)
    if p_z.shape != q_z.shape:
        raise ValueError("p_logits and q_logits must have identical shapes")
    log_p = stable_log_softmax(p_z, axis=axis)
    log_q = stable_log_softmax(q_z, axis=axis)
    p = np.exp(log_p)
    return np.sum(p * (log_p - log_q), axis=axis)


def masked_conditional_kl_sum(p_logits: Array, q_logits: Array, mask: Array) -> Array:
    """Sum exact token-conditional KL values over active sequence positions."""
    token_kl = exact_kl_from_logits(p_logits, q_logits, axis=-1)
    m = np.asarray(mask, dtype=bool)
    if token_kl.shape != m.shape:
        raise ValueError("mask must match logits without the vocabulary axis")
    if token_kl.ndim == 1:
        return np.asarray(float(np.sum(token_kl[m])))
    return np.sum(np.where(m, token_kl, 0.0), axis=-1)


def kl_sample_values(log_p: Array, log_q: Array) -> KLEstimatorValues:
    """Return k1, k2, and k3 for samples drawn from p.

    With ``r = q(x) / p(x)``:

    * ``k1 = -log r`` is value-unbiased for ``D_KL(p || q)`` but can
      be negative per sample;
    * ``k2 = 0.5 * (log r)^2`` is non-negative and locally second-order;
    * ``k3 = (r - 1) - log r`` is non-negative. In this orientation, exact
      value-unbiasedness also needs the two normalized distributions to share
      compatible support: finite ``log r`` on p-samples and no q-mass outside
      the support of p.

    These value statements do not by themselves justify differentiating the
    sample expressions as loss functions.
    """
    lp = np.asarray(log_p, dtype=np.float64)
    lq = np.asarray(log_q, dtype=np.float64)
    if lp.shape != lq.shape:
        raise ValueError("log_p and log_q must have identical shapes")
    log_r = lq - lp
    ratio = np.exp(log_r)
    return KLEstimatorValues(
        k1=-log_r,
        k2=0.5 * np.square(log_r),
        k3=(ratio - 1.0) - log_r,
        log_ratio_q_over_p=log_r,
        ratio_q_over_p=ratio,
    )

def kl_estimators_from_selected_logprobs(log_p: Array, log_q: Array) -> KLEstimatorValues:
    """Compatibility alias for :func:`kl_sample_values`."""
    return kl_sample_values(log_p, log_q)


def sample_kl_estimators(p_logits: Array, q_logits: Array, n: int, seed: int = 0) -> dict[str, float]:
    """Monte Carlo means and standard deviations for ``D_KL(p || q)``."""
    if n <= 0:
        raise ValueError("n must be positive")
    p_z = np.asarray(p_logits, dtype=np.float64)
    q_z = np.asarray(q_logits, dtype=np.float64)
    if p_z.ndim != 1 or q_z.ndim != 1 or p_z.shape != q_z.shape:
        raise ValueError("both logit arrays must be aligned and one-dimensional")
    log_p = stable_log_softmax(p_z)
    log_q = stable_log_softmax(q_z)
    p_dist = np.exp(log_p)
    rng = np.random.default_rng(seed)
    ids = rng.choice(p_dist.size, size=n, p=p_dist)
    estimators = kl_sample_values(log_p[ids], log_q[ids])
    result: dict[str, float] = {"exact": float(np.sum(p_dist * (log_p - log_q)))}
    for name in ("k1", "k2", "k3"):
        values = getattr(estimators, name)
        result[f"{name}_mean"] = float(np.mean(values))
        result[f"{name}_std"] = float(np.std(values))
    result["k1_negative_fraction"] = float(np.mean(estimators.k1 < 0))
    return result

def kl_estimator_moments(p: Array, q: Array) -> dict[str, float]:
    """Exact k1/k2/k3 moments for samples ``x ~ q`` and ``r = p / q``.

    Both distributions must be strictly positive on the same finite support.
    The returned ``exact_kl`` is ``D_KL(q || p)``.
    """
    p_dist = _normalized_distribution(p)
    q_dist = _normalized_distribution(q)
    if p_dist.shape != q_dist.shape:
        raise ValueError("p and q must have identical finite support")
    if np.any(p_dist <= 0) or np.any(q_dist <= 0):
        raise ValueError("exact finite moments require strictly positive common support")
    log_r = np.log(p_dist) - np.log(q_dist)
    values = {
        "k1": -log_r,
        "k2": 0.5 * np.square(log_r),
        "k3": (p_dist / q_dist - 1.0) - log_r,
    }
    out: dict[str, float] = {
        "exact_kl": float(np.sum(q_dist * (np.log(q_dist) - np.log(p_dist))))
    }
    for name, sample_values in values.items():
        mean = float(np.sum(q_dist * sample_values))
        variance = float(np.sum(q_dist * np.square(sample_values - mean)))
        out[f"{name}_mean"] = mean
        out[f"{name}_std"] = float(np.sqrt(max(variance, 0.0)))
    return out


def sequence_logprob_ratio(
    target_logprobs: Array,
    behavior_logprobs: Array,
    mask: Array | None = None,
) -> dict[str, float]:
    """Return the exact log-ratio and ratio for one aligned token sequence."""
    target = np.asarray(target_logprobs, dtype=np.float64)
    behavior = np.asarray(behavior_logprobs, dtype=np.float64)
    if target.shape != behavior.shape:
        raise ValueError("target and behavior log-probabilities must have identical shapes")
    active = np.ones_like(target, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if active.shape != target.shape or not np.any(active):
        raise ValueError("mask must match the log-probabilities and select at least one token")
    if np.any(~np.isfinite(target[active])) or np.any(~np.isfinite(behavior[active])):
        raise ValueError("active log-probabilities must be finite")
    log_ratio = float(np.sum(target[active] - behavior[active], dtype=np.float64))
    return {"log_ratio": log_ratio, "ratio": float(np.exp(log_ratio))}


def classify_stop_event(
    stop_reason: str,
    *,
    terminal_reasons: tuple[str, ...] = ("eos",),
    truncation_reasons: tuple[str, ...] = ("length", "timeout"),
    inferred: bool = False,
) -> StopEvent:
    """Map an engine stop reason to task semantics supplied by the caller.

    The same raw reason can be mapped differently by another task contract. A
    reason may deliberately belong to both sets (for example, a true terminal
    event coinciding with an external horizon), and an unlisted reason remains
    unclassified rather than being silently treated as terminal.
    """
    if not isinstance(stop_reason, str) or not stop_reason.strip():
        raise ValueError("stop_reason must be a non-empty string")
    reason = stop_reason.strip().lower()
    terminal = {str(x).strip().lower() for x in terminal_reasons}
    truncated = {str(x).strip().lower() for x in truncation_reasons}
    if "" in terminal or "" in truncated:
        raise ValueError("reason sets must contain non-empty strings")
    return StopEvent(
        stop_reason=reason,
        terminated=reason in terminal,
        truncated=reason in truncated,
        inferred=bool(inferred),
    )


def importance_corrected_mean(
    values: Array,
    target_logprobs: Array,
    behavior_logprobs: Array,
    mask: Array | None = None,
    *,
    self_normalized: bool = False,
    max_weight: float | None = None,
) -> float:
    """Estimate a target expectation from behavior samples with support overlap.

    Clipping and self-normalization can reduce variance but generally introduce
    bias. Hard support truncation cannot be repaired when the target assigns
    positive probability outside the behavior support.
    """
    x = np.asarray(values, dtype=np.float64)
    target = np.asarray(target_logprobs, dtype=np.float64)
    behavior = np.asarray(behavior_logprobs, dtype=np.float64)
    if not (x.shape == target.shape == behavior.shape):
        raise ValueError("values and both log-probability arrays must have identical shapes")
    m = np.ones_like(x, dtype=np.float64) if mask is None else np.asarray(mask, dtype=np.float64)
    if m.shape != x.shape or np.any((m != 0) & (m != 1)):
        raise ValueError("mask must be binary and match values")
    weights = np.exp(target - behavior)
    active = m.astype(bool)
    if np.any(~np.isfinite(weights[active])):
        raise ValueError("importance weights must be finite on active items")
    if max_weight is not None:
        if max_weight <= 0:
            raise ValueError("max_weight must be positive")
        weights = np.minimum(weights, max_weight)
    effective = weights * m
    denominator = float(np.sum(effective) if self_normalized else np.sum(m))
    if denominator <= 0:
        raise ValueError("mask selects no items")
    return float(np.sum(x * effective) / denominator)


def temperature_distribution(logits: Array, temperature: float) -> Array:
    """Softmax after positive-temperature scaling."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    return softmax(np.asarray(logits, dtype=np.float64) / float(temperature), axis=-1)


def greedy_action(logits: Array) -> int:
    """Return the first maximum, matching NumPy's deterministic tie rule."""
    z = np.asarray(logits)
    if z.ndim != 1:
        raise ValueError("greedy_action expects one-dimensional logits")
    return int(np.argmax(z))


def categorical_entropy_from_logits(logits: Array, axis: int = -1) -> Array:
    """Categorical entropy in nats."""
    log_p = stable_log_softmax(logits, axis=axis)
    p = np.exp(log_p)
    return -np.sum(p * log_p, axis=axis)


def _normalized_distribution(probabilities: Array) -> Array:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(p)) or np.any(p < 0):
        raise ValueError("probabilities must be finite and non-negative")
    total = float(np.sum(p))
    if total <= 0:
        raise ValueError("probabilities must contain positive mass")
    return p / total


def categorical_entropy(probabilities: Array) -> float:
    """Entropy in nats, with the convention ``0 log 0 = 0``."""
    p = _normalized_distribution(probabilities)
    positive = p > 0
    return float(-np.sum(p[positive] * np.log(p[positive])))


def top_k_distribution(logits: Array, k: int) -> Array:
    """Apply ideal top-k to one-dimensional logits and renormalize."""
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if not isinstance(k, (int, np.integer)) or k <= 0 or k > z.size:
        raise ValueError("k must be an integer in [1, vocabulary size]")
    p = temperature_distribution(z, 1.0)
    order = np.argsort(-p, kind="stable")
    keep = order[: int(k)]
    out = np.zeros_like(p)
    out[keep] = p[keep]
    out /= np.sum(out)
    return out


def top_p_distribution(logits: Array, top_p: float, *, min_tokens_to_keep: int = 1) -> Array:
    """Apply ideal nucleus sampling to one-dimensional logits and renormalize."""
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if not np.isfinite(top_p) or not (0 < top_p <= 1):
        raise ValueError("top_p must lie in (0, 1]")
    if min_tokens_to_keep < 1 or min_tokens_to_keep > z.size:
        raise ValueError("min_tokens_to_keep is outside the vocabulary")
    p = temperature_distribution(z, 1.0)
    if top_p == 1.0:
        return p
    order = np.argsort(-p, kind="stable")
    cumulative = np.cumsum(p[order])
    count = int(np.searchsorted(cumulative, top_p, side="left") + 1)
    count = max(count, min_tokens_to_keep)
    keep = order[:count]
    out = np.zeros_like(p)
    out[keep] = p[keep]
    out /= np.sum(out)
    return out


def behavior_distribution(
    logits: Array,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    allowed_mask: Array | None = None,
    logit_bias: Array | None = None,
) -> Array:
    """Transparent educational decoding transform.

    Operation order: bias, allowed-token mask, temperature, top-k, top-p.
    Production engines may use another order; that order is part of the
    behavior-policy specification.
    """
    z = np.asarray(logits, dtype=np.float64).copy()
    if z.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if logit_bias is not None:
        bias = np.asarray(logit_bias, dtype=np.float64)
        if bias.shape != z.shape:
            raise ValueError("logit_bias must match logits")
        z += bias
    if allowed_mask is not None:
        allowed = np.asarray(allowed_mask, dtype=bool)
        if allowed.shape != z.shape or not np.any(allowed):
            raise ValueError("allowed_mask must match logits and retain at least one token")
        z[~allowed] = -np.inf
    p = temperature_distribution(z, temperature)
    if top_k is not None:
        order = np.argsort(-p, kind="stable")
        keep = order[: int(top_k)]
        masked = np.zeros_like(p)
        masked[keep] = p[keep]
        p = masked / masked.sum()
    if top_p < 1.0:
        # top_p_distribution accepts logits; log(p) preserves the distribution.
        safe_logits = np.full_like(p, -np.inf)
        positive = p > 0
        safe_logits[positive] = np.log(p[positive])
        p = top_p_distribution(safe_logits, top_p)
    return p


def decoding_distribution_with_order(
    logits: Array,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    order: tuple[str, ...] = ("temperature", "top_k", "top_p"),
) -> Array:
    """Apply temperature/top-k/top-p in an explicit, testable order.

    The implementation keeps a logits-like score vector with ``-inf`` outside
    the current support. Reordering processors can therefore change the final
    distribution; production code must record the engine's actual order.
    """
    scores = np.asarray(logits, dtype=np.float64).copy()
    if scores.ndim != 1 or scores.size == 0 or not np.any(np.isfinite(scores)):
        raise ValueError("logits must be a non-empty one-dimensional vector with finite support")
    valid_ops = {"temperature", "top_k", "top_p"}
    if set(order) != valid_ops or len(order) != 3:
        raise ValueError("order must contain temperature, top_k, and top_p exactly once")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if top_k is not None and (not isinstance(top_k, (int, np.integer)) or not (1 <= int(top_k) <= scores.size)):
        raise ValueError("top_k must be None or an integer in [1, vocabulary size]")
    if not np.isfinite(top_p) or not (0 < top_p <= 1):
        raise ValueError("top_p must lie in (0, 1]")

    for op in order:
        if op == "temperature":
            scores = scores / float(temperature)
        elif op == "top_k" and top_k is not None:
            probs = softmax(scores)
            keep = np.argsort(-probs, kind="stable")[: int(top_k)]
            mask = np.zeros(scores.size, dtype=bool)
            mask[keep] = True
            scores[~mask] = -np.inf
        elif op == "top_p" and top_p < 1.0:
            probs = softmax(scores)
            sorted_ids = np.argsort(-probs, kind="stable")
            count = int(np.searchsorted(np.cumsum(probs[sorted_ids]), top_p, side="left") + 1)
            mask = np.zeros(scores.size, dtype=bool)
            mask[sorted_ids[:count]] = True
            scores[~mask] = -np.inf
    return softmax(scores)


def selected_logprob_from_distribution(distribution: Array, token_id: int) -> float:
    """Log-probability of a token; return ``-inf`` outside the support."""
    p = _normalized_distribution(distribution)
    if token_id < 0 or token_id >= p.size:
        raise ValueError("token_id is outside the vocabulary")
    return float("-inf") if p[token_id] == 0 else float(np.log(p[token_id]))


def behavior_logprob(distribution: Array, token_id: int) -> float:
    """Alias for selected log-probability under the actual behavior distribution."""
    return selected_logprob_from_distribution(distribution, token_id)


def truncation_mismatch(base_probabilities: Array, top_p: float, token_id: int) -> dict[str, float]:
    """Compare base and ideal top-p behavior probabilities for one token."""
    p = _normalized_distribution(base_probabilities)
    b = top_p_distribution(np.log(p), top_p)
    base_lp = selected_logprob_from_distribution(p, token_id)
    behavior_lp = selected_logprob_from_distribution(b, token_id)
    retained_mass = float(np.sum(p[b > 0]))
    return {
        "base_logprob": base_lp,
        "behavior_logprob": behavior_lp,
        "logprob_shift": behavior_lp - base_lp,
        "retained_mass": retained_mass,
        "in_support": float(b[token_id] > 0),
    }


def binary_tree_sequence_kl(
    p_root: Array,
    q_root: Array,
    p_children: Mapping[int, Array],
    q_children: Mapping[int, Array],
) -> tuple[float, float]:
    """Exact joint KL and its chain-rule decomposition for a depth-two tree."""
    p0 = _normalized_distribution(p_root)
    q0 = _normalized_distribution(q_root)
    if p0.shape != q0.shape:
        raise ValueError("root supports must match")
    expected = set(range(p0.size))
    if set(p_children) != expected or set(q_children) != expected:
        raise ValueError("children must be supplied for every root action")
    joint_p: list[float] = []
    joint_q: list[float] = []
    root_kl = float(np.sum(p0 * (np.log(p0) - np.log(q0))))
    conditional = 0.0
    for action in range(p0.size):
        pc = _normalized_distribution(p_children[action])
        qc = _normalized_distribution(q_children[action])
        if pc.shape != qc.shape:
            raise ValueError("child supports must match")
        conditional += p0[action] * float(np.sum(pc * (np.log(pc) - np.log(qc))))
        joint_p.extend((p0[action] * pc).tolist())
        joint_q.extend((q0[action] * qc).tolist())
    jp = np.asarray(joint_p)
    jq = np.asarray(joint_q)
    joint = float(np.sum(jp * (np.log(jp) - np.log(jq))))
    return joint, root_kl + conditional


def rollout_summary(
    response_mask: Array,
    terminal_mask: Array,
    behavior_logprobs: Array,
    trainer_logprobs: Array,
) -> dict[str, float]:
    """Minimal token-level diagnostics for one padded response."""
    response = np.asarray(response_mask, dtype=bool)
    terminal = np.asarray(terminal_mask, dtype=bool)
    behavior = np.asarray(behavior_logprobs, dtype=np.float64)
    trainer = np.asarray(trainer_logprobs, dtype=np.float64)
    if not (response.shape == terminal.shape == behavior.shape == trainer.shape):
        raise ValueError("all rollout arrays must have the same shape")
    if not np.any(response):
        raise ValueError("response_mask selects no tokens")
    delta = (trainer - behavior)[response]
    return {
        "response_tokens": float(np.sum(response)),
        "terminal_events": float(np.sum(terminal & response)),
        "mean_abs_logprob_delta": float(np.mean(np.abs(delta))),
        "max_abs_logprob_delta": float(np.max(np.abs(delta))),
        "sequence_importance_ratio": float(np.exp(np.sum(delta))),
    }


@dataclass(frozen=True)
class RolloutBatchSummary:
    batch_size: int
    max_response_tokens: int
    active_tokens: int
    terminated_sequences: int
    truncated_sequences: int
    mean_response_length: float


def validate_rollout_batch(batch: Mapping[str, Any]) -> RolloutBatchSummary:
    """Validate a minimal padded response-only rollout batch."""
    required = {
        "response_ids",
        "response_mask",
        "old_logprobs",
        "ref_logprobs",
        "sequence_rewards",
        "terminated",
        "truncated",
        "stop_reason",
    }
    missing = sorted(required - set(batch))
    if missing:
        raise ValueError("missing fields: " + ", ".join(missing))
    ids = np.asarray(batch["response_ids"])
    mask = np.asarray(batch["response_mask"], dtype=bool)
    old = np.asarray(batch["old_logprobs"], dtype=np.float64)
    ref = np.asarray(batch["ref_logprobs"], dtype=np.float64)
    rewards = np.asarray(batch["sequence_rewards"], dtype=np.float64)
    terminated = np.asarray(batch["terminated"], dtype=bool)
    truncated = np.asarray(batch["truncated"], dtype=bool)
    stop_reason = np.asarray(batch["stop_reason"], dtype=object)
    if ids.ndim != 2:
        raise ValueError("response_ids must have shape (batch, time)")
    if not (mask.shape == old.shape == ref.shape == ids.shape):
        raise ValueError("token-level arrays must have the same shape")
    batch_size, max_tokens = ids.shape
    if not (rewards.shape == terminated.shape == truncated.shape == stop_reason.shape == (batch_size,)):
        raise ValueError("sequence-level arrays, including stop_reason, must have shape (batch,)")
    lengths = np.sum(mask, axis=1).astype(int)
    if np.any(lengths <= 0):
        raise ValueError("each response must contain at least one active token")
    expected = np.arange(max_tokens)[None, :] < lengths[:, None]
    if not np.array_equal(mask, expected):
        raise ValueError("active response positions must form a contiguous prefix")
    if np.any(~np.isfinite(old[mask])) or np.any(~np.isfinite(ref[mask])):
        raise ValueError("active log-probabilities must be finite")
    if np.any(np.isfinite(old[~mask])) or np.any(np.isfinite(ref[~mask])):
        raise ValueError("padded log-probabilities must be NaN")
    if np.any(~np.isfinite(rewards)):
        raise ValueError("sequence rewards must be finite")
    if any(not isinstance(reason, str) or not reason.strip() for reason in stop_reason.tolist()):
        raise ValueError("every stop_reason must be a non-empty string")
    return RolloutBatchSummary(
        batch_size=int(batch_size),
        max_response_tokens=int(max_tokens),
        active_tokens=int(np.sum(mask)),
        terminated_sequences=int(np.sum(terminated)),
        truncated_sequences=int(np.sum(truncated)),
        mean_response_length=float(np.mean(lengths)),
    )


__all__ = [
    "SequenceMetrics",
    "KLEstimatorValues",
    "StopEvent",
    "RolloutBatchSummary",
    "stable_log_softmax",
    "log_softmax",
    "softmax",
    "selected_token_logprobs",
    "causal_response_logprobs",
    "masked_sequence_metrics",
    "masked_sequence_stats",
    "causal_response_stats",
    "exact_kl_from_logits",
    "masked_conditional_kl_sum",
    "kl_sample_values",
    "kl_estimators_from_selected_logprobs",
    "sample_kl_estimators",
    "kl_estimator_moments",
    "sequence_logprob_ratio",
    "classify_stop_event",
    "importance_corrected_mean",
    "temperature_distribution",
    "greedy_action",
    "categorical_entropy_from_logits",
    "categorical_entropy",
    "top_k_distribution",
    "top_p_distribution",
    "behavior_distribution",
    "decoding_distribution_with_order",
    "selected_logprob_from_distribution",
    "behavior_logprob",
    "truncation_mismatch",
    "binary_tree_sequence_kl",
    "rollout_summary",
    "validate_rollout_batch",
]
