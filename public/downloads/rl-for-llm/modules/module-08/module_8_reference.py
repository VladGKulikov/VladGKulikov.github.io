"""Deterministic reference implementations for Module 8.

The functions are intentionally NumPy-only. They expose the mathematics behind
KL-regularized RLHF, DPO and several direct preference objectives without
requiring a language model or an accelerator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

ArrayLike = Sequence[float] | np.ndarray


def _as_float_array(x: ArrayLike, *, name: str = "array") -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    return arr


def logsumexp(x: ArrayLike) -> float:
    arr = _as_float_array(x, name="x")
    m = float(np.max(arr))
    return m + float(np.log(np.exp(arr - m).sum()))


def softmax(logits: ArrayLike) -> np.ndarray:
    z = _as_float_array(logits, name="logits")
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def log_softmax(logits: ArrayLike) -> np.ndarray:
    z = _as_float_array(logits, name="logits")
    return z - logsumexp(z)


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x_arr)
    pos = x_arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x_arr[pos]))
    exp_x = np.exp(x_arr[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return float(out) if out.ndim == 0 else out


def softplus(x: float | np.ndarray) -> float | np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.maximum(x_arr, 0.0) + np.log1p(np.exp(-np.abs(x_arr)))
    return float(out) if out.ndim == 0 else out


def validate_distribution(p: ArrayLike, *, name: str = "p") -> np.ndarray:
    arr = _as_float_array(p, name=name)
    if np.any(arr < 0):
        raise ValueError(f"{name} must be non-negative")
    total = float(arr.sum())
    if not np.isclose(total, 1.0, atol=1e-10):
        raise ValueError(f"{name} must sum to one; got {total}")
    return arr


def categorical_kl(p: ArrayLike, q: ArrayLike) -> float:
    """Return D_KL(p || q) for finite categorical distributions."""
    p_arr = validate_distribution(p, name="p")
    q_arr = validate_distribution(q, name="q")
    if p_arr.shape != q_arr.shape:
        raise ValueError("p and q must have the same shape")
    if np.any((p_arr > 0) & (q_arr == 0)):
        return float("inf")
    mask = p_arr > 0
    return float(np.sum(p_arr[mask] * (np.log(p_arr[mask]) - np.log(q_arr[mask]))))


def regularized_objective(policy: ArrayLike, rewards: ArrayLike, reference: ArrayLike, beta: float) -> float:
    """E_pi[r] - beta * D_KL(pi || reference)."""
    if beta < 0 or not np.isfinite(beta):
        raise ValueError("beta must be finite and non-negative")
    pi = validate_distribution(policy, name="policy")
    ref = validate_distribution(reference, name="reference")
    r = _as_float_array(rewards, name="rewards")
    if not (pi.shape == ref.shape == r.shape):
        raise ValueError("policy, reference, and rewards must have the same shape")
    kl = categorical_kl(pi, ref)
    if np.isinf(kl) and beta == 0:
        return float(np.dot(pi, r))
    return float(np.dot(pi, r) - beta * kl)


def optimal_kl_policy(reference: ArrayLike, rewards: ArrayLike, beta: float) -> np.ndarray:
    """Closed-form optimizer of E_pi[r] - beta KL(pi||reference), beta > 0."""
    if beta <= 0 or not np.isfinite(beta):
        raise ValueError("beta must be finite and strictly positive")
    ref = validate_distribution(reference, name="reference")
    r = _as_float_array(rewards, name="rewards")
    if ref.shape != r.shape:
        raise ValueError("reference and rewards must have the same shape")
    support = ref > 0
    logits = np.full_like(ref, -np.inf)
    logits[support] = np.log(ref[support]) + r[support] / beta
    finite_logits = logits[support]
    m = np.max(finite_logits)
    weights = np.zeros_like(ref)
    weights[support] = np.exp(finite_logits - m)
    return weights / weights.sum()


def reward_shift_invariant(reference: ArrayLike, rewards: ArrayLike, beta: float, shift: float) -> bool:
    a = optimal_kl_policy(reference, rewards, beta)
    b = optimal_kl_policy(reference, np.asarray(rewards, dtype=np.float64) + shift, beta)
    return bool(np.allclose(a, b, atol=1e-12, rtol=1e-12))


def sequence_logprob(token_logprobs: ArrayLike, mask: ArrayLike | None = None) -> float:
    logp = _as_float_array(token_logprobs, name="token_logprobs")
    if mask is None:
        return float(logp.sum())
    m = _as_float_array(mask, name="mask")
    if m.shape != logp.shape:
        raise ValueError("mask and token_logprobs must have the same shape")
    if np.any((m != 0) & (m != 1)):
        raise ValueError("mask must contain only 0 and 1")
    return float(np.dot(logp, m))


def sampled_sequence_log_ratio(policy_logprobs: ArrayLike, reference_logprobs: ArrayLike, mask: ArrayLike | None = None) -> float:
    p = _as_float_array(policy_logprobs, name="policy_logprobs")
    q = _as_float_array(reference_logprobs, name="reference_logprobs")
    if p.shape != q.shape:
        raise ValueError("policy and reference log-probabilities must have the same shape")
    return sequence_logprob(p - q, mask)


def shape_token_rewards(policy_logprobs: ArrayLike, reference_logprobs: ArrayLike, terminal_reward: float, beta: float, mask: ArrayLike | None = None) -> np.ndarray:
    """Add -beta*(log pi - log pi_ref) at each active token and terminal reward last."""
    if beta < 0 or not np.isfinite(beta):
        raise ValueError("beta must be finite and non-negative")
    p = _as_float_array(policy_logprobs, name="policy_logprobs")
    q = _as_float_array(reference_logprobs, name="reference_logprobs")
    if p.shape != q.shape:
        raise ValueError("policy and reference log-probabilities must have the same shape")
    if not np.isfinite(terminal_reward):
        raise ValueError("terminal_reward must be finite")
    active = np.ones_like(p)
    if mask is not None:
        active = _as_float_array(mask, name="mask")
        if active.shape != p.shape or np.any((active != 0) & (active != 1)):
            raise ValueError("mask must match the log-probabilities and contain only 0/1")
        if active.sum() == 0:
            raise ValueError("mask must contain at least one active token")
    shaped = -beta * (p - q) * active
    last = int(np.flatnonzero(active)[-1])
    shaped[last] += terminal_reward
    return shaped


def dpo_margin(policy_chosen_logp: float, policy_rejected_logp: float, reference_chosen_logp: float, reference_rejected_logp: float) -> float:
    values = [policy_chosen_logp, policy_rejected_logp, reference_chosen_logp, reference_rejected_logp]
    if not np.all(np.isfinite(values)):
        raise ValueError("all log-probabilities must be finite")
    return float((policy_chosen_logp - reference_chosen_logp) - (policy_rejected_logp - reference_rejected_logp))


def dpo_logit(margin: float | np.ndarray, beta: float) -> float | np.ndarray:
    if beta <= 0 or not np.isfinite(beta):
        raise ValueError("beta must be finite and strictly positive")
    return beta * np.asarray(margin, dtype=np.float64)


def dpo_loss_from_margin(margin: float | np.ndarray, beta: float) -> float | np.ndarray:
    return softplus(-dpo_logit(margin, beta))


def dpo_gradient_weight(margin: float | np.ndarray, beta: float) -> float | np.ndarray:
    """Positive magnitude beta * sigma(-beta*margin)."""
    z = dpo_logit(margin, beta)
    return beta * sigmoid(-z)


def conservative_dpo_loss(margin: float | np.ndarray, beta: float, label_smoothing: float) -> float | np.ndarray:
    """Binary label-smoothed DPO / conservative DPO loss."""
    eps = float(label_smoothing)
    if not 0 <= eps < 0.5:
        raise ValueError("label_smoothing must lie in [0, 0.5)")
    z = dpo_logit(margin, beta)
    return (1.0 - eps) * softplus(-z) + eps * softplus(z)


def ipo_loss(margin: float | np.ndarray, tau: float) -> float | np.ndarray:
    """Sampled IPO loss (h - 1/(2*tau))^2."""
    if tau <= 0 or not np.isfinite(tau):
        raise ValueError("tau must be finite and strictly positive")
    m = np.asarray(margin, dtype=np.float64)
    out = (m - 1.0 / (2.0 * tau)) ** 2
    return float(out) if out.ndim == 0 else out


def simpo_reward(sequence_logp: float, length: int, beta: float) -> float:
    if length <= 0:
        raise ValueError("length must be positive")
    if beta <= 0 or not np.isfinite(beta):
        raise ValueError("beta must be finite and strictly positive")
    if not np.isfinite(sequence_logp):
        raise ValueError("sequence_logp must be finite")
    return float(beta * sequence_logp / length)


def simpo_loss(chosen_sequence_logp: float, chosen_length: int, rejected_sequence_logp: float, rejected_length: int, beta: float, target_margin: float) -> float:
    if target_margin < 0 or not np.isfinite(target_margin):
        raise ValueError("target_margin must be finite and non-negative")
    z = simpo_reward(chosen_sequence_logp, chosen_length, beta) - simpo_reward(rejected_sequence_logp, rejected_length, beta) - target_margin
    return float(softplus(-z))


def log_odds_from_logprob(logp: float) -> float:
    """Stable log[p/(1-p)] from log p, where 0 < p < 1."""
    if not np.isfinite(logp) or logp >= 0:
        raise ValueError("logp must be finite and strictly negative")
    p = float(np.exp(logp))
    return float(logp - np.log1p(-p))


def orpo_preference_loss(chosen_logp: float, rejected_logp: float) -> float:
    log_odds_ratio = log_odds_from_logprob(chosen_logp) - log_odds_from_logprob(rejected_logp)
    return float(softplus(-log_odds_ratio))


def orpo_total_loss(chosen_mean_nll: float, chosen_logp: float, rejected_logp: float, lam: float) -> float:
    if chosen_mean_nll < 0 or not np.isfinite(chosen_mean_nll):
        raise ValueError("chosen_mean_nll must be finite and non-negative")
    if lam < 0 or not np.isfinite(lam):
        raise ValueError("lam must be finite and non-negative")
    return float(chosen_mean_nll + lam * orpo_preference_loss(chosen_logp, rejected_logp))


def constrained_lagrangian(
    expected_reward: float,
    expected_costs: ArrayLike,
    limits: ArrayLike,
    multipliers: ArrayLike,
) -> float:
    """Return E[R] - sum_j lambda_j * (E[C_j] - d_j)."""
    if not np.isfinite(expected_reward):
        raise ValueError("expected_reward must be finite")
    costs = _as_float_array(expected_costs, name="expected_costs")
    bounds = _as_float_array(limits, name="limits")
    lam = _as_float_array(multipliers, name="multipliers")
    if not (costs.shape == bounds.shape == lam.shape):
        raise ValueError("expected_costs, limits, and multipliers must have the same shape")
    if np.any(lam < 0):
        raise ValueError("multipliers must be non-negative")
    return float(expected_reward - np.dot(lam, costs - bounds))


def dual_ascent_step(
    multipliers: ArrayLike,
    expected_costs: ArrayLike,
    limits: ArrayLike,
    step_size: float,
) -> np.ndarray:
    """Projected dual ascent: [lambda + eta * (cost - limit)]_+."""
    if step_size <= 0 or not np.isfinite(step_size):
        raise ValueError("step_size must be finite and strictly positive")
    lam = _as_float_array(multipliers, name="multipliers")
    costs = _as_float_array(expected_costs, name="expected_costs")
    bounds = _as_float_array(limits, name="limits")
    if not (lam.shape == costs.shape == bounds.shape):
        raise ValueError("multipliers, expected_costs, and limits must have the same shape")
    if np.any(lam < 0):
        raise ValueError("multipliers must be non-negative")
    return np.maximum(0.0, lam + step_size * (costs - bounds))


def two_action_constrained_optimum(
    safe_reward: float,
    risky_reward: float,
    safe_cost: float,
    risky_cost: float,
    cost_limit: float,
) -> float:
    """Optimal risky-action probability for the teaching two-action mixture.

    The contract assumes the risky action has strictly higher cost. If it also
    has higher reward, the optimum uses the largest feasible probability;
    otherwise the safe action is optimal.
    """
    values = np.asarray(
        [safe_reward, risky_reward, safe_cost, risky_cost, cost_limit],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("all inputs must be finite")
    if risky_cost <= safe_cost:
        raise ValueError("risky_cost must be strictly greater than safe_cost")
    if cost_limit < safe_cost:
        raise ValueError("cost_limit is infeasible even for the safe action")
    q_max = float(np.clip((cost_limit - safe_cost) / (risky_cost - safe_cost), 0.0, 1.0))
    return q_max if risky_reward > safe_reward else 0.0


@dataclass(frozen=True)
class PairDiagnostics:
    chosen_log_ratio: float
    rejected_log_ratio: float
    margin: float
    dpo_logit: float
    loss: float
    gradient_weight: float
    chosen_logp: float
    rejected_logp: float


def pair_diagnostics(policy_chosen_logp: float, policy_rejected_logp: float, reference_chosen_logp: float, reference_rejected_logp: float, beta: float) -> PairDiagnostics:
    chosen_ratio = float(policy_chosen_logp - reference_chosen_logp)
    rejected_ratio = float(policy_rejected_logp - reference_rejected_logp)
    margin = chosen_ratio - rejected_ratio
    z = float(dpo_logit(margin, beta))
    return PairDiagnostics(chosen_ratio, rejected_ratio, margin, z, float(softplus(-z)), float(beta * sigmoid(-z)), float(policy_chosen_logp), float(policy_rejected_logp))


def likelihood_displacement_probabilities(t: float) -> np.ndarray:
    """Teaching toy with logits [-t, -3t, 4t]."""
    if not np.isfinite(t):
        raise ValueError("t must be finite")
    return softmax(np.array([-t, -3.0 * t, 4.0 * t], dtype=np.float64))


def likelihood_displacement_curve(ts: Iterable[float]) -> dict[str, np.ndarray]:
    values = np.asarray(list(ts), dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("ts must be a finite non-empty one-dimensional iterable")
    probs = np.stack([likelihood_displacement_probabilities(float(t)) for t in values])
    return {"t": values, "chosen": probs[:, 0], "rejected": probs[:, 1], "other": probs[:, 2], "log_margin": 2.0 * values}


def reward_kl_frontier(reference: ArrayLike, rewards: ArrayLike, betas: ArrayLike) -> dict[str, np.ndarray]:
    ref = validate_distribution(reference, name="reference")
    r = _as_float_array(rewards, name="rewards")
    bs = _as_float_array(betas, name="betas")
    if r.shape != ref.shape:
        raise ValueError("reference and rewards must have the same shape")
    if np.any(bs <= 0):
        raise ValueError("all betas must be positive")
    policies = np.stack([optimal_kl_policy(ref, r, float(b)) for b in bs])
    expected_reward = policies @ r
    kl = np.array([categorical_kl(p, ref) for p in policies])
    return {"beta": bs, "policy": policies, "reward": expected_reward, "kl": kl}


def summarize_pairs(policy_chosen_logp: ArrayLike, policy_rejected_logp: ArrayLike, reference_chosen_logp: ArrayLike, reference_rejected_logp: ArrayLike, beta: float) -> dict[str, float]:
    pc = _as_float_array(policy_chosen_logp, name="policy_chosen_logp")
    pr = _as_float_array(policy_rejected_logp, name="policy_rejected_logp")
    rc = _as_float_array(reference_chosen_logp, name="reference_chosen_logp")
    rr = _as_float_array(reference_rejected_logp, name="reference_rejected_logp")
    if not (pc.shape == pr.shape == rc.shape == rr.shape):
        raise ValueError("all arrays must have the same shape")
    margins = (pc - rc) - (pr - rr)
    logits = beta * margins
    return {
        "mean_loss": float(np.mean(softplus(-logits))),
        "pair_accuracy": float(np.mean(margins > 0)),
        "mean_margin": float(np.mean(margins)),
        "mean_chosen_logp": float(np.mean(pc)),
        "mean_rejected_logp": float(np.mean(pr)),
        "mean_implicit_chosen_reward": float(np.mean(beta * (pc - rc))),
        "mean_implicit_rejected_reward": float(np.mean(beta * (pr - rr))),
        "mean_gradient_weight": float(np.mean(beta * sigmoid(-logits))),
    }
