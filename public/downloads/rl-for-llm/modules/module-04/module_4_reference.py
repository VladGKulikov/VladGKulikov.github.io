#!/usr/bin/env python3
"""Deterministic NumPy reference implementations for RL for LLM, Module 4.

The module deliberately keeps the functions small and explicit.  They are
used by the exercises, diagrams, notebook smoke checks, and release tests.
No network access, accelerator, or stochastic framework is required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Array = np.ndarray


def stable_softmax(logits: Array, axis: int = -1) -> Array:
    """Numerically stable softmax."""
    x = np.asarray(logits, dtype=float)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def grad_log_softmax(logits: Array, action: int) -> Array:
    """Gradient of ``log softmax(logits)[action]`` with respect to logits."""
    p = stable_softmax(np.asarray(logits, dtype=float))
    if p.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if not 0 <= action < p.size:
        raise ValueError("action is out of range")
    result = -p
    result[action] += 1.0
    return result


def reward_to_go(rewards: Array, gamma: float = 1.0) -> Array:
    """Return ``G_t = r_{t+1} + gamma r_{t+2} + ...`` for each position."""
    r = np.asarray(rewards, dtype=float)
    if r.ndim != 1:
        raise ValueError("rewards must be one-dimensional")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")
    result = np.empty_like(r)
    running = 0.0
    for t in range(len(r) - 1, -1, -1):
        running = r[t] + gamma * running
        result[t] = running
    return result


def discounted_score_weights(rewards: Array, gamma: float = 1.0) -> Array:
    """Weights ``gamma**t * G_t`` for the literal discounted-start objective."""
    g = reward_to_go(rewards, gamma)
    return (gamma ** np.arange(len(g), dtype=float)) * g


def reinforce_gradient(
    logits: Array,
    actions: Array,
    weights: Array,
    *,
    reduction: Literal["none", "sum", "mean"] = "mean",
) -> Array:
    """REINFORCE score-function estimate in logit coordinates.

    ``logits`` may be one vector shared by all observations or one vector per
    observation.  ``actions`` and ``weights`` must have matching leading size.
    """
    z = np.asarray(logits, dtype=float)
    a = np.asarray(actions, dtype=int)
    w = np.asarray(weights, dtype=float)
    if a.ndim != 1 or w.ndim != 1 or a.shape != w.shape:
        raise ValueError("actions and weights must be one-dimensional and aligned")
    if z.ndim == 1:
        z = np.broadcast_to(z, (len(a), z.size))
    if z.ndim != 2 or z.shape[0] != len(a):
        raise ValueError("logits must be shape [K] or [N, K]")
    rows = np.stack([grad_log_softmax(z_i, int(a_i)) for z_i, a_i in zip(z, a)])
    result = rows * w[:, None]
    if reduction == "none":
        return result
    if reduction == "sum":
        return np.sum(result, axis=0)
    if reduction == "mean":
        return np.mean(result, axis=0)
    raise ValueError("reduction must be 'none', 'sum', or 'mean'")



def exact_expectation_gradient(
    probabilities: Array,
    score_vectors: Array,
    values: Array,
    direct_gradients: Array | None = None,
) -> Array:
    """Exact finite-space gradient for ``E_{p_theta}[f_theta(X)]``.

    The returned vector is

    ``sum_x p(x) * (f(x) * grad_log_p(x) + grad_f(x))``.

    ``direct_gradients=None`` implements the usual score-function case in
    which the sampled value has no direct parameter dependence.
    """
    p = np.asarray(probabilities, dtype=float)
    scores = np.asarray(score_vectors, dtype=float)
    f = np.asarray(values, dtype=float)
    if p.ndim != 1 or f.shape != p.shape or scores.ndim != 2 or scores.shape[0] != p.size:
        raise ValueError("probabilities and values must be [K], score_vectors [K, P]")
    if np.any(p < 0.0) or not np.isclose(np.sum(p), 1.0):
        raise ValueError("probabilities must be non-negative and sum to one")
    if direct_gradients is None:
        direct = np.zeros_like(scores)
    else:
        direct = np.asarray(direct_gradients, dtype=float)
        if direct.shape != scores.shape:
            raise ValueError("direct_gradients must match score_vectors")
    return np.sum(p[:, None] * (f[:, None] * scores + direct), axis=0)

def action_independent_baseline_expectation(logits: Array, baseline: float) -> Array:
    """Exact expectation of ``baseline * grad log pi(A)`` over ``A ~ pi``."""
    z = np.asarray(logits, dtype=float)
    p = stable_softmax(z)
    return baseline * sum(p[a] * grad_log_softmax(z, a) for a in range(len(p)))


def score_norm_weighted_baseline(logits: Array, action_values: Array) -> float:
    """Variance-minimizing scalar baseline in score-vector squared norm."""
    z = np.asarray(logits, dtype=float)
    q = np.asarray(action_values, dtype=float)
    p = stable_softmax(z)
    if z.ndim != 1 or q.shape != z.shape:
        raise ValueError("logits and action_values must have the same 1D shape")
    norms = np.asarray([np.dot(grad_log_softmax(z, a), grad_log_softmax(z, a)) for a in range(len(z))])
    denominator = float(np.dot(p, norms))
    if denominator <= 0.0:
        raise ValueError("score norm denominator must be positive")
    return float(np.dot(p, q * norms) / denominator)


def leave_one_out_baselines(rewards: Array) -> Array:
    """Mean of all other rewards for each member of a group."""
    r = np.asarray(rewards, dtype=float)
    if r.ndim != 1 or len(r) < 2:
        raise ValueError("at least two one-dimensional rewards are required")
    return (np.sum(r) - r) / (len(r) - 1)


def group_mean_advantages(rewards: Array) -> Array:
    """Advantages centered by the in-sample group mean."""
    r = np.asarray(rewards, dtype=float)
    if r.ndim != 1 or len(r) == 0:
        raise ValueError("rewards must be a non-empty vector")
    return r - np.mean(r)


def group_zscore_advantages(rewards: Array, eps: float = 1e-8) -> Array:
    """Group-centered advantages divided by population standard deviation."""
    centered = group_mean_advantages(rewards)
    scale = float(np.sqrt(np.mean(centered ** 2)))
    if scale < eps:
        return np.zeros_like(centered)
    return centered / scale


def rloo_advantages(rewards: Array) -> Array:
    """Leave-one-out sequence-level advantages ``R_i - mean(R_-i)``."""
    r = np.asarray(rewards, dtype=float)
    return r - leave_one_out_baselines(r)


def expected_in_sample_mean_scale(group_size: int) -> float:
    """Expected gradient scale induced by a same-group mean baseline."""
    if group_size < 1:
        raise ValueError("group_size must be positive")
    return (group_size - 1.0) / group_size


def gae_advantages(
    rewards: Array,
    values: Array,
    next_values: Array,
    terminated: Array,
    episode_ends: Array,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Array:
    """Generalized advantage estimation with separate terminal/end masks.

    ``terminated`` masks value bootstrap because no continuation exists.
    ``episode_ends`` is the recursion-cut mask: it is true before a reset and
    at a local rollout-buffer boundary when the following advantage is absent
    from the current array. A time-limit truncation therefore uses
    ``terminated=False`` and ``episode_ends=True``; a nonterminal final record
    of a fixed-size buffer uses the same pair while bootstrapping from its real
    next observation.
    """
    r = np.asarray(rewards, dtype=float)
    v = np.asarray(values, dtype=float)
    nv = np.asarray(next_values, dtype=float)
    term = np.asarray(terminated, dtype=bool)
    ends = np.asarray(episode_ends, dtype=bool)
    if not (r.shape == v.shape == nv.shape == term.shape == ends.shape) or r.ndim != 1:
        raise ValueError("all GAE inputs must be aligned one-dimensional arrays")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must lie in [0, 1]")
    deltas = r + gamma * (~term).astype(float) * nv - v
    advantages = np.empty_like(deltas)
    running = 0.0
    for t in range(len(r) - 1, -1, -1):
        running = deltas[t] + gamma * gae_lambda * (not ends[t]) * running
        advantages[t] = running
    return advantages


def value_targets(values: Array, advantages: Array) -> Array:
    """Critic regression targets ``V_old + A_hat``."""
    v = np.asarray(values, dtype=float)
    a = np.asarray(advantages, dtype=float)
    if v.shape != a.shape:
        raise ValueError("values and advantages must have matching shapes")
    return v + a


def policy_ratio(new_log_prob: Array, old_log_prob: Array) -> Array:
    """Stable local importance ratio ``exp(log pi_new - log pi_old)``."""
    new = np.asarray(new_log_prob, dtype=float)
    old = np.asarray(old_log_prob, dtype=float)
    if new.shape != old.shape:
        raise ValueError("log-probability arrays must match")
    return np.exp(new - old)


def trajectory_importance_ratio(new_log_probs: Array, old_log_probs: Array, mask: Array | None = None) -> float:
    """Exact product ratio of an observed trajectory, computed in log space."""
    new = np.asarray(new_log_probs, dtype=float)
    old = np.asarray(old_log_probs, dtype=float)
    if new.shape != old.shape:
        raise ValueError("log-probability arrays must match")
    if mask is None:
        m = np.ones_like(new, dtype=float)
    else:
        m = np.asarray(mask, dtype=float)
        if m.shape != new.shape:
            raise ValueError("mask must match log-probability arrays")
    return float(np.exp(np.sum((new - old) * m)))


def ppo_clipped_terms(ratio: Array, advantage: Array, clip_epsilon: float = 0.2) -> tuple[Array, Array, Array]:
    """Return unclipped, clipped, and pessimistic PPO policy terms."""
    r = np.asarray(ratio, dtype=float)
    a = np.asarray(advantage, dtype=float)
    if r.shape != a.shape:
        raise ValueError("ratio and advantage must have matching shapes")
    if np.any(r <= 0.0):
        raise ValueError("importance ratios must be positive")
    if clip_epsilon < 0.0:
        raise ValueError("clip_epsilon must be non-negative")
    unclipped = r * a
    clipped = np.clip(r, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * a
    pessimistic = np.minimum(unclipped, clipped)
    return unclipped, clipped, pessimistic


def ppo_policy_objective(ratio: Array, advantage: Array, clip_epsilon: float = 0.2) -> float:
    """Mean clipped PPO surrogate objective (to maximize)."""
    return float(np.mean(ppo_clipped_terms(ratio, advantage, clip_epsilon)[2]))


@dataclass(frozen=True)
class PPODiagnostics:
    ratio_outside_fraction: float
    active_clip_fraction: float
    approx_kl_nonnegative: float
    mean_log_ratio: float


def ppo_diagnostics(new_log_prob: Array, old_log_prob: Array, advantage: Array, clip_epsilon: float = 0.2) -> PPODiagnostics:
    """Useful descriptive PPO diagnostics; none is a universal safety bound."""
    log_ratio = np.asarray(new_log_prob, dtype=float) - np.asarray(old_log_prob, dtype=float)
    ratio = np.exp(log_ratio)
    a = np.asarray(advantage, dtype=float)
    if ratio.shape != a.shape:
        raise ValueError("log probabilities and advantages must align")
    unclipped, _, pessimistic = ppo_clipped_terms(ratio, a, clip_epsilon)
    outside = np.mean((ratio < 1.0 - clip_epsilon) | (ratio > 1.0 + clip_epsilon))
    active = np.mean(pessimistic < unclipped - 1e-15)
    # The expression is pointwise non-negative because x - 1 >= log x.
    approx_kl = np.mean((ratio - 1.0) - log_ratio)
    return PPODiagnostics(float(outside), float(active), float(approx_kl), float(np.mean(log_ratio)))



def response_token_mask(
    input_ids: Array,
    prompt_lengths: Array,
    *,
    pad_token_id: int,
    eos_token_id: int | None = None,
    include_eos: bool = True,
) -> Array:
    """Build a causal-LM response mask aligned with ``input_ids[:, 1:]``.

    ``prompt_lengths[b]`` is the number of original tokens belonging to the
    prompt (including an optional BOS token).  Column ``j-1`` of the returned
    mask corresponds to the label token at original position ``j``.  Prompt
    and padding labels are excluded.  The first generated EOS is included by
    default and every later position is excluded.
    """
    ids = np.asarray(input_ids)
    lengths = np.asarray(prompt_lengths, dtype=int)
    if ids.ndim != 2 or lengths.ndim != 1 or lengths.shape[0] != ids.shape[0]:
        raise ValueError("input_ids must be [B, T] and prompt_lengths [B]")
    batch, width = ids.shape
    if width < 2:
        raise ValueError("at least two token positions are required")
    if np.any(lengths < 1) or np.any(lengths >= width):
        raise ValueError("each prompt length must lie in [1, T-1]")
    mask = np.zeros((batch, width - 1), dtype=float)
    for b in range(batch):
        active = True
        for original_pos in range(1, width):
            token = int(ids[b, original_pos])
            if original_pos < int(lengths[b]) or token == pad_token_id or not active:
                continue
            if eos_token_id is not None and token == eos_token_id:
                if include_eos:
                    mask[b, original_pos - 1] = 1.0
                active = False
            else:
                mask[b, original_pos - 1] = 1.0
    return mask

def sequence_log_probability(token_log_probs: Array, mask: Array | None = None) -> Array:
    """Sum token log-probabilities along the last axis."""
    lp = np.asarray(token_log_probs, dtype=float)
    if mask is None:
        return np.sum(lp, axis=-1)
    m = np.asarray(mask, dtype=float)
    if m.shape != lp.shape:
        raise ValueError("mask must match token_log_probs")
    return np.sum(lp * m, axis=-1)



def _validated_token_arrays(
    new_log_probs: Array,
    old_log_probs: Array,
    token_mask: Array,
) -> tuple[Array, Array, Array]:
    """Validate and return aligned ``[batch, time]`` token arrays.

    The mask is numerical rather than Boolean so the same helper can support
    binary masks and non-negative per-token weights. At least one valid token
    is required in every sequence; silently accepting an empty response would
    make sequence-normalized reductions ill-defined.
    """
    new = np.asarray(new_log_probs, dtype=float)
    old = np.asarray(old_log_probs, dtype=float)
    mask = np.asarray(token_mask, dtype=float)
    if new.ndim != 2 or old.ndim != 2 or mask.ndim != 2:
        raise ValueError("new_log_probs, old_log_probs, and token_mask must be two-dimensional")
    if new.shape != old.shape or new.shape != mask.shape:
        raise ValueError("new_log_probs, old_log_probs, and token_mask must have identical shapes")
    if not np.all(np.isfinite(new)) or not np.all(np.isfinite(old)):
        raise ValueError("log-probabilities must be finite")
    if not np.all(np.isfinite(mask)) or np.any(mask < 0.0):
        raise ValueError("token_mask must contain finite non-negative values")
    if np.any(mask.sum(axis=1) <= 0.0):
        raise ValueError("every sequence must contain at least one valid token")
    return new, old, mask


def token_probability_ratio(
    new_log_probs: Array,
    old_log_probs: Array,
    token_mask: Array,
) -> Array:
    """Return local PPO ratios aligned with response-token positions.

    Masked positions are returned as one, the multiplicative identity. This is
    convenient for diagnostics and makes it explicit that padding or prompt
    columns must not alter a whole-sequence ratio.
    """
    new, old, mask = _validated_token_arrays(new_log_probs, old_log_probs, token_mask)
    ratio = np.exp(new - old)
    return np.where(mask > 0.0, ratio, 1.0)


def sequence_log_probability_ratio(
    new_log_probs: Array,
    old_log_probs: Array,
    token_mask: Array,
) -> Array:
    """Return the exact log probability ratio for every complete response."""
    new, old, mask = _validated_token_arrays(new_log_probs, old_log_probs, token_mask)
    return np.sum(mask * (new - old), axis=1)


def sequence_probability_ratio(
    new_log_probs: Array,
    old_log_probs: Array,
    token_mask: Array,
) -> Array:
    """Exponentiate the exact sequence log-ratio, raising on overflow."""
    log_ratio = sequence_log_probability_ratio(new_log_probs, old_log_probs, token_mask)
    try:
        with np.errstate(over="raise", invalid="raise"):
            return np.exp(log_ratio)
    except FloatingPointError as exc:
        raise OverflowError(
            "the exact sequence ratio overflowed; inspect or clip the log-ratio for diagnostics"
        ) from exc


def masked_token_ppo_terms(
    new_log_probs: Array,
    old_log_probs: Array,
    sequence_advantages: Array,
    token_mask: Array,
    clip_epsilon: float = 0.2,
) -> tuple[Array, Array, Array, Array]:
    """Return ratio, unclipped, clipped, and pessimistic tokenwise PPO terms.

    ``sequence_advantages`` has shape ``[batch]`` and is broadcast over valid
    response tokens. All returned objective terms are zero on masked columns.
    """
    if clip_epsilon < 0.0:
        raise ValueError("clip_epsilon must be non-negative")
    new, old, mask = _validated_token_arrays(new_log_probs, old_log_probs, token_mask)
    advantage = np.asarray(sequence_advantages, dtype=float)
    if advantage.ndim != 1 or advantage.shape[0] != new.shape[0]:
        raise ValueError("sequence_advantages must have shape [batch]")
    if not np.all(np.isfinite(advantage)):
        raise ValueError("sequence_advantages must be finite")
    ratio = np.exp(new - old)
    broadcast = advantage[:, None]
    unclipped = ratio * broadcast
    clipped = np.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * broadcast
    pessimistic = np.minimum(unclipped, clipped)
    return (
        np.where(mask > 0.0, ratio, 1.0),
        mask * unclipped,
        mask * clipped,
        mask * pessimistic,
    )


def masked_token_ppo_objective(
    new_log_probs: Array,
    old_log_probs: Array,
    sequence_advantages: Array,
    token_mask: Array,
    clip_epsilon: float = 0.2,
    *,
    reduction: str = "token_mean",
) -> float:
    """Reduce the masked tokenwise PPO surrogate under an explicit convention.

    ``token_mean`` gives equal weight to every valid token in the batch.
    ``sequence_mean`` first averages valid tokens within each response and then
    averages responses, so every response receives equal weight.
    ``sequence_sum_mean`` first sums within each response and then averages
    responses; this is the score-function sum convention and weights longer
    responses more heavily when their per-token terms are comparable.
    """
    _, _, _, pessimistic = masked_token_ppo_terms(
        new_log_probs,
        old_log_probs,
        sequence_advantages,
        token_mask,
        clip_epsilon,
    )
    mask = np.asarray(token_mask, dtype=float)
    token_counts = mask.sum(axis=1)
    sequence_sums = pessimistic.sum(axis=1)
    if reduction == "token_mean":
        return float(sequence_sums.sum() / token_counts.sum())
    if reduction == "sequence_mean":
        return float(np.mean(sequence_sums / token_counts))
    if reduction == "sequence_sum_mean":
        return float(np.mean(sequence_sums))
    raise ValueError("reduction must be 'token_mean', 'sequence_mean', or 'sequence_sum_mean'")


def broadcast_sequence_advantages(sequence_advantages: Array, token_mask: Array) -> Array:
    """Broadcast one sequence-level coefficient to valid response tokens."""
    a = np.asarray(sequence_advantages, dtype=float)
    m = np.asarray(token_mask, dtype=float)
    if m.ndim != 2 or a.ndim != 1 or m.shape[0] != a.shape[0]:
        raise ValueError("advantages must be [B] and token_mask [B, T]")
    return a[:, None] * m


def simulate_group_baseline_bias(
    *,
    group_size: int = 8,
    groups: int = 200_000,
    seed: int = 20260715,
    action_probability: float = 0.35,
    reward_action_1: float = 1.0,
    reward_action_0: float = -0.2,
) -> dict[str, float]:
    """Reproduce the finite-group expectation of mean and LOO baselines.

    A one-parameter Bernoulli policy uses score ``A - p``.  The exact gradient
    is ``p(1-p)(r1-r0)``.  The simulation estimates the plain, in-sample mean,
    and leave-one-out estimators over independent groups.
    """
    if group_size < 2 or groups < 1:
        raise ValueError("group_size >= 2 and groups >= 1 are required")
    p = float(action_probability)
    rng = np.random.default_rng(seed)
    actions = rng.binomial(1, p, size=(groups, group_size)).astype(float)
    rewards = np.where(actions == 1.0, reward_action_1, reward_action_0)
    scores = actions - p
    plain = np.mean(rewards * scores, axis=1)
    centered = rewards - np.mean(rewards, axis=1, keepdims=True)
    mean_est = np.mean(centered * scores, axis=1)
    loo_base = (np.sum(rewards, axis=1, keepdims=True) - rewards) / (group_size - 1)
    loo_est = np.mean((rewards - loo_base) * scores, axis=1)
    exact = p * (1.0 - p) * (reward_action_1 - reward_action_0)
    return {
        "exact": float(exact),
        "plain_mean": float(np.mean(plain)),
        "same_group_mean": float(np.mean(mean_est)),
        "rloo_mean": float(np.mean(loo_est)),
        "predicted_same_group": float(expected_in_sample_mean_scale(group_size) * exact),
        "plain_std": float(np.std(plain, ddof=1)),
        "same_group_std": float(np.std(mean_est, ddof=1)),
        "rloo_std": float(np.std(loo_est, ddof=1)),
    }


__all__ = [
    "PPODiagnostics",
    "action_independent_baseline_expectation",
    "broadcast_sequence_advantages",
    "discounted_score_weights",
    "expected_in_sample_mean_scale",
    "exact_expectation_gradient",
    "gae_advantages",
    "grad_log_softmax",
    "group_mean_advantages",
    "group_zscore_advantages",
    "leave_one_out_baselines",
    "policy_ratio",
    "ppo_clipped_terms",
    "ppo_diagnostics",
    "ppo_policy_objective",
    "reinforce_gradient",
    "response_token_mask",
    "reward_to_go",
    "rloo_advantages",
    "score_norm_weighted_baseline",
    "sequence_log_probability",
    "simulate_group_baseline_bias",
    "stable_softmax",
    "masked_token_ppo_objective",
    "masked_token_ppo_terms",
    "sequence_log_probability_ratio",
    "sequence_probability_ratio",
    "token_probability_ratio",
    "trajectory_importance_ratio",
    "value_targets",
]
