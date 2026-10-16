#!/usr/bin/env python3
"""Reference NumPy implementations for RL for LLM, Module 5.

The functions are intentionally small and explicit.  They support the analytic
and programming exercises without depending on a deep-learning framework.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

Array = np.ndarray


def masked_behavior_cloning_nll(selected_log_probs: Array, mask: Array, *, reduction: str = "mean") -> float | Array:
    """Negative log-likelihood over selected actions/tokens marked by ``mask``.

    ``selected_log_probs`` contains log pi(a|s) for the demonstrated action,
    not a full logit tensor.  ``reduction='mean'`` divides by the number of
    active entries rather than by sequence or batch length.
    """
    lp = np.asarray(selected_log_probs, dtype=float)
    m = np.asarray(mask, dtype=float)
    if lp.shape != m.shape:
        raise ValueError("selected_log_probs and mask must have matching shapes")
    if np.any((m != 0.0) & (m != 1.0)):
        raise ValueError("mask must contain only 0 and 1")
    values = -lp * m
    if reduction == "none":
        return values
    if reduction == "sum":
        return float(np.sum(values))
    if reduction == "mean":
        count = float(np.sum(m))
        if count <= 0.0:
            raise ValueError("mean reduction requires at least one active entry")
        return float(np.sum(values) / count)
    raise ValueError("reduction must be 'none', 'sum', or 'mean'")


def tightrope_return(epsilon: float, horizon: int) -> float:
    """Expected reward for an irreversible tightrope with per-step error rate."""
    eps = float(epsilon)
    if not 0.0 <= eps <= 1.0:
        raise ValueError("epsilon must lie in [0, 1]")
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if horizon == 0:
        return 0.0
    if eps == 0.0:
        return float(horizon)
    q = 1.0 - eps
    return float(q * (1.0 - q**horizon) / eps)


def tightrope_regret(epsilon: float, horizon: int) -> float:
    """Expert return minus tightrope imitator return."""
    return float(horizon) - tightrope_return(epsilon, horizon)


def normalized_tightrope_regret(epsilon: float, horizon: int) -> float:
    """Regret divided by epsilon*T^2, with zero for a zero denominator."""
    denom = float(epsilon) * float(horizon) ** 2
    return 0.0 if denom == 0.0 else tightrope_regret(epsilon, horizon) / denom


def _aligned_transition_inputs(rewards: Array, terminated: Array, next_q: Array) -> tuple[Array, Array, Array]:
    r = np.asarray(rewards, dtype=float)
    term = np.asarray(terminated, dtype=bool)
    q = np.asarray(next_q, dtype=float)
    if r.ndim != 1 or term.shape != r.shape or q.ndim != 2 or q.shape[0] != r.shape[0]:
        raise ValueError("rewards and terminated must be [B], next_q must be [B, A]")
    if q.shape[1] < 1:
        raise ValueError("at least one action is required")
    return r, term, q


def dqn_targets(rewards: Array, terminated: Array, next_q_target: Array, *, gamma: float = 0.99) -> Array:
    """One-step DQN targets using a target-network maximum."""
    r, term, q_target = _aligned_transition_inputs(rewards, terminated, next_q_target)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")
    return r + gamma * (~term).astype(float) * np.max(q_target, axis=1)


def double_dqn_targets(
    rewards: Array,
    terminated: Array,
    next_q_online: Array,
    next_q_target: Array,
    *,
    gamma: float = 0.99,
) -> Array:
    """Double-DQN targets: select with online values, evaluate with target values."""
    r, term, q_online = _aligned_transition_inputs(rewards, terminated, next_q_online)
    _, _, q_target = _aligned_transition_inputs(rewards, terminated, next_q_target)
    if q_online.shape != q_target.shape:
        raise ValueError("online and target Q arrays must have matching shapes")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")
    selected = np.argmax(q_online, axis=1)
    values = q_target[np.arange(len(r)), selected]
    return r + gamma * (~term).astype(float) * values


def pessimistic_values(empirical_means: Array, counts: Array, *, radius_scale: float = 1.0) -> Array:
    """Simple lower-confidence scores ``mean - scale/sqrt(count)``.

    Actions with zero observations receive ``-inf``.  This is an educational
    support guard, not a general offline-RL guarantee.
    """
    means = np.asarray(empirical_means, dtype=float)
    n = np.asarray(counts, dtype=float)
    if means.shape != n.shape or means.ndim != 1:
        raise ValueError("empirical_means and counts must be matching vectors")
    if np.any(n < 0.0) or radius_scale < 0.0:
        raise ValueError("counts and radius_scale must be non-negative")
    out = np.full_like(means, -np.inf)
    observed = n > 0.0
    out[observed] = means[observed] - radius_scale / np.sqrt(n[observed])
    return out


def logsumexp(values: Array, axis: int | None = None) -> Array:
    """Numerically stable log-sum-exp."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("values must be non-empty")
    m = np.max(x, axis=axis, keepdims=True)
    result = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis) if axis is not None else np.asarray(result).reshape(() )


def cql_discrete_penalty(q_values: Array, data_actions: Array | int) -> float:
    """Mean discrete CQL term: logsumexp over actions minus logged-action Q."""
    q = np.asarray(q_values, dtype=float)
    if q.ndim == 1:
        q = q[None, :]
    if q.ndim != 2 or q.shape[1] < 1:
        raise ValueError("q_values must be [A] or [B, A]")
    actions = np.asarray(data_actions, dtype=int)
    if actions.ndim == 0:
        actions = np.full(q.shape[0], int(actions), dtype=int)
    if actions.shape != (q.shape[0],) or np.any(actions < 0) or np.any(actions >= q.shape[1]):
        raise ValueError("data_actions must contain one valid action per row")
    terms = logsumexp(q, axis=1) - q[np.arange(q.shape[0]), actions]
    return float(np.mean(terms))


def expectile_loss(residuals: Array, tau: float, *, reduction: str = "mean") -> float | Array:
    """Asymmetric squared loss ``|tau - 1{u<0}| * u^2``."""
    u = np.asarray(residuals, dtype=float)
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must lie strictly between 0 and 1")
    values = np.abs(tau - (u < 0.0).astype(float)) * u**2
    if reduction == "none":
        return values
    if reduction == "sum":
        return float(np.sum(values))
    if reduction == "mean":
        if values.size == 0:
            raise ValueError("mean reduction requires non-empty residuals")
        return float(np.mean(values))
    raise ValueError("reduction must be 'none', 'sum', or 'mean'")


def iql_advantage_weights(advantages: Array, beta: float, *, max_weight: float | None = 100.0) -> Array:
    """Exponentiated IQL policy weights with optional clipping."""
    a = np.asarray(advantages, dtype=float)
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    log_w = beta * a
    if max_weight is not None:
        if max_weight <= 0.0:
            raise ValueError("max_weight must be positive")
        log_w = np.minimum(log_w, math.log(max_weight))
    # Avoid overflow even when clipping is disabled.
    log_w = np.minimum(log_w, np.log(np.finfo(float).max))
    return np.exp(log_w)



def behavior_support_mask(action_probabilities: Array, *, threshold_ratio: float = 0.3) -> Array:
    """Return actions whose probability is a fraction of the row maximum.

    This is the discrete support filter used by the module's BCQ-like example.
    It is a heuristic approximation to behavior support, not an identification
    theorem.  Every row is guaranteed to retain at least one action.
    """
    probabilities = np.asarray(action_probabilities, dtype=float)
    was_vector = probabilities.ndim == 1
    if was_vector:
        probabilities = probabilities[None, :]
    if probabilities.ndim != 2 or probabilities.shape[1] < 1:
        raise ValueError("action_probabilities must be [A] or [B, A]")
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("action probabilities must be finite and non-negative")
    if not 0.0 < threshold_ratio <= 1.0:
        raise ValueError("threshold_ratio must lie in (0, 1]")
    maxima = probabilities.max(axis=1, keepdims=True)
    if np.any(maxima <= 0.0):
        raise ValueError("each row must have positive probability mass")
    mask = probabilities >= threshold_ratio * maxima
    # Numerical ties and positive maxima already guarantee at least one True.
    return mask[0] if was_vector else mask


def fqe_targets(
    rewards: Array,
    terminated: Array,
    next_q_values: Array,
    target_action_probabilities: Array,
    *,
    gamma: float = 0.99,
) -> Array:
    """One-step fitted-Q-evaluation targets for a fixed target policy."""
    r, term, q = _aligned_transition_inputs(rewards, terminated, next_q_values)
    probabilities = np.asarray(target_action_probabilities, dtype=float)
    if probabilities.shape != q.shape:
        raise ValueError("target_action_probabilities must match next_q_values")
    if np.any(probabilities < 0.0) or np.any(~np.isfinite(probabilities)):
        raise ValueError("target probabilities must be finite and non-negative")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-8):
        raise ValueError("target probabilities must sum to one in each row")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")
    continuation = np.sum(probabilities * q, axis=1)
    return r + gamma * (~term).astype(float) * continuation


def advantage_weighted_nll(
    selected_log_probs: Array,
    advantages: Array,
    beta: float,
    *,
    max_weight: float = 100.0,
    reduction: str = "mean",
) -> float | Array:
    """Advantage-weighted negative log-likelihood used in IQL extraction."""
    log_probs = np.asarray(selected_log_probs, dtype=float)
    advantages_array = np.asarray(advantages, dtype=float)
    if log_probs.shape != advantages_array.shape:
        raise ValueError("selected_log_probs and advantages must match")
    weights = iql_advantage_weights(advantages_array, beta, max_weight=max_weight)
    values = -weights * log_probs
    if reduction == "none":
        return values
    if reduction == "sum":
        return float(values.sum())
    if reduction == "mean":
        if values.size == 0:
            raise ValueError("mean reduction requires non-empty arrays")
        return float(values.mean())
    raise ValueError("reduction must be 'none', 'sum', or 'mean'")

def return_to_go(rewards: Iterable[float] | Array, *, gamma: float = 1.0) -> Array:
    """Return-to-go under ``G_t = r_t + gamma*G_{t+1}``."""
    r = np.asarray(list(rewards) if not isinstance(rewards, np.ndarray) else rewards, dtype=float)
    if r.ndim != 1:
        raise ValueError("rewards must be one-dimensional")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")
    out = np.empty_like(r)
    running = 0.0
    for t in range(len(r) - 1, -1, -1):
        running = r[t] + gamma * running
        out[t] = running
    return out


def update_target_return(current_target: float, reward: float, *, gamma: float = 1.0) -> float:
    """Update a Decision Transformer target after observing a reward."""
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must lie in (0, 1]")
    return (float(current_target) - float(reward)) / gamma


__all__ = [
    "masked_behavior_cloning_nll",
    "tightrope_return",
    "tightrope_regret",
    "normalized_tightrope_regret",
    "dqn_targets",
    "double_dqn_targets",
    "pessimistic_values",
    "logsumexp",
    "cql_discrete_penalty",
    "expectile_loss",
    "iql_advantage_weights",
    "behavior_support_mask",
    "fqe_targets",
    "advantage_weighted_nll",
    "return_to_go",
    "update_target_return",
]
