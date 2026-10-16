"""Deterministic reference implementations for Module 9.

The module is NumPy-only and intentionally small. It exposes the mechanics of
verifiable rewards, group-relative advantages, several loss-reduction choices,
and sequence-level importance ratios without requiring a language model.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import re
from typing import Iterable, Sequence

import numpy as np

ArrayLike = Sequence[float] | np.ndarray

# A deliberately narrow teaching parser: fractions must be matched before
# decimals, otherwise ``1/2`` could be split into ``1`` and ``2``.
NUMBER_PATTERN = re.compile(r"-?\d+\s*/\s*-?\d+|-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*)\}")
THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")


def _array(x: ArrayLike, *, name: str, ndim: int | None = 1) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    return arr


def extract_numeric_answer(text: str) -> str | None:
    """Extract the last boxed expression, otherwise the last number/fraction.

    This function intentionally does not attempt symbolic equivalence. It is a
    transparent teaching contract suitable for arithmetic examples only.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    boxed = BOXED_PATTERN.findall(text)
    if boxed:
        candidate = boxed[-1].strip()
        return candidate or None
    normalized = THOUSANDS_SEPARATOR.sub("", text)
    matches = list(NUMBER_PATTERN.finditer(normalized))
    return matches[-1].group(0).strip() if matches else None


def parse_numeric(value: str) -> float:
    """Parse a finite integer, decimal, scientific notation, or fraction."""
    if not isinstance(value, str):
        raise TypeError("value must be a string")
    value = THOUSANDS_SEPARATOR.sub("", value).replace(" ", "").strip()
    if not value:
        raise ValueError("empty numeric value")
    if "/" in value:
        parts = value.split("/")
        if len(parts) != 2:
            raise ValueError("fraction must contain one slash")
        numerator, denominator = (int(part) for part in parts)
        if denominator == 0:
            raise ZeroDivisionError("fraction denominator must be non-zero")
        result = float(Fraction(numerator, denominator))
    else:
        result = float(value)
    if not math.isfinite(result):
        raise ValueError("numeric value must be finite")
    return result


def verify_numeric_answer(
    response: str,
    gold: str,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> int:
    """Return 1 when the extracted response matches the gold value numerically."""
    if rtol < 0 or atol < 0 or not math.isfinite(rtol) or not math.isfinite(atol):
        raise ValueError("rtol and atol must be finite and non-negative")
    answer = extract_numeric_answer(response)
    if answer is None:
        return 0
    try:
        observed = parse_numeric(answer)
        expected = parse_numeric(gold)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return 0
    return int(abs(observed - expected) <= atol + rtol * max(1.0, abs(expected)))


def group_relative_advantages(
    rewards: ArrayLike,
    *,
    scale: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """Center one reward group and optionally divide by population std."""
    r = _array(rewards, name="rewards")
    if eps < 0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and non-negative")
    centered = r - r.mean()
    if not scale:
        return centered
    std = float(r.std(ddof=0))
    if std == 0.0:
        return np.zeros_like(r)
    return centered / (std + eps)


def informative_group_probability(p: float, group_size: int) -> float:
    """P(group contains at least one success and one failure)."""
    if not 0.0 <= p <= 1.0 or not math.isfinite(p):
        raise ValueError("p must lie in [0, 1]")
    if group_size < 2:
        raise ValueError("group_size must be at least two")
    return float(1.0 - p**group_size - (1.0 - p) ** group_size)


def expected_draws_for_informative_groups(target: int, p: float, group_size: int) -> float:
    """Expected prompt draws for ``target`` informative groups under IID Bernoulli rewards."""
    if target <= 0:
        raise ValueError("target must be positive")
    prob = informative_group_probability(p, group_size)
    return float("inf") if prob == 0.0 else float(target / prob)


def clipped_surrogate_terms(
    ratios: ArrayLike,
    advantages: ArrayLike,
    *,
    eps_low: float = 0.2,
    eps_high: float | None = None,
) -> np.ndarray:
    """Return PPO/GRPO clipped surrogate terms, before the leading minus sign."""
    rho = _array(ratios, name="ratios")
    adv = _array(advantages, name="advantages")
    if rho.shape != adv.shape:
        raise ValueError("ratios and advantages must have the same shape")
    if np.any(rho <= 0):
        raise ValueError("ratios must be strictly positive")
    if eps_high is None:
        eps_high = eps_low
    if eps_low < 0 or eps_high < 0 or not math.isfinite(eps_low) or not math.isfinite(eps_high):
        raise ValueError("clip widths must be finite and non-negative")
    clipped = np.clip(rho, 1.0 - eps_low, 1.0 + eps_high)
    return np.minimum(rho * adv, clipped * adv)


def clip_activity(
    ratios: ArrayLike,
    advantages: ArrayLike,
    *,
    eps_low: float = 0.2,
    eps_high: float | None = None,
) -> np.ndarray:
    """Return a Boolean mask where the clipped branch is active and has zero ratio gradient."""
    rho = _array(ratios, name="ratios")
    adv = _array(advantages, name="advantages")
    if rho.shape != adv.shape:
        raise ValueError("ratios and advantages must have the same shape")
    if eps_high is None:
        eps_high = eps_low
    return ((adv > 0) & (rho > 1.0 + eps_high)) | ((adv < 0) & (rho < 1.0 - eps_low))


def _validate_lengths(lengths: Sequence[int] | np.ndarray, *, size: int) -> np.ndarray:
    arr = np.asarray(lengths, dtype=np.int64)
    if arr.ndim != 1 or arr.size != size:
        raise ValueError("lengths must be a one-dimensional array matching rewards")
    if np.any(arr <= 0):
        raise ValueError("all lengths must be positive")
    return arr


def original_grpo_token_weights(rewards: ArrayLike, lengths: Sequence[int], *, eps: float = 1e-8) -> list[np.ndarray]:
    """Per-token coefficients A_i / (G * |o_i|) for original GRPO reduction."""
    r = _array(rewards, name="rewards")
    lens = _validate_lengths(lengths, size=r.size)
    adv = group_relative_advantages(r, scale=True, eps=eps)
    g = r.size
    return [np.full(int(length), adv_i / (g * length), dtype=np.float64) for adv_i, length in zip(adv, lens)]


def dr_grpo_token_weights(
    rewards: ArrayLike,
    lengths: Sequence[int],
    *,
    fixed_length: int,
) -> list[np.ndarray]:
    """Per-token coefficients (r_i-mean)/(G*L) for Dr. GRPO."""
    r = _array(rewards, name="rewards")
    lens = _validate_lengths(lengths, size=r.size)
    if fixed_length <= 0:
        raise ValueError("fixed_length must be positive")
    adv = group_relative_advantages(r, scale=False)
    g = r.size
    return [np.full(int(length), adv_i / (g * fixed_length), dtype=np.float64) for adv_i, length in zip(adv, lens)]


def dapo_token_weights(
    rewards: ArrayLike,
    lengths: Sequence[int],
    *,
    scale_rewards: bool = True,
    eps: float = 1e-8,
) -> list[np.ndarray]:
    """Per-token coefficients A_i / total_active_tokens for DAPO-style reduction."""
    r = _array(rewards, name="rewards")
    lens = _validate_lengths(lengths, size=r.size)
    adv = group_relative_advantages(r, scale=scale_rewards, eps=eps)
    total = int(lens.sum())
    return [np.full(int(length), adv_i / total, dtype=np.float64) for adv_i, length in zip(adv, lens)]


def response_contributions(token_weights: Sequence[np.ndarray]) -> np.ndarray:
    """Sum per-token coefficients inside each response."""
    if not token_weights:
        raise ValueError("token_weights must be non-empty")
    return np.asarray([float(np.asarray(w, dtype=np.float64).sum()) for w in token_weights])


def sequence_importance_ratio(
    new_token_logprobs: ArrayLike,
    old_token_logprobs: ArrayLike,
    mask: ArrayLike | None = None,
) -> float:
    """Length-normalized GSPO ratio: exp(mean active token log-ratio)."""
    new = _array(new_token_logprobs, name="new_token_logprobs")
    old = _array(old_token_logprobs, name="old_token_logprobs")
    if new.shape != old.shape:
        raise ValueError("new and old token log-probabilities must have the same shape")
    active = np.ones_like(new, dtype=bool)
    if mask is not None:
        m = _array(mask, name="mask")
        if m.shape != new.shape or np.any((m != 0) & (m != 1)):
            raise ValueError("mask must match log-probabilities and contain only 0/1")
        active = m.astype(bool)
    if not active.any():
        raise ValueError("at least one token must be active")
    mean_log_ratio = float(np.mean((new - old)[active]))
    return float(np.exp(mean_log_ratio))


def gspo_surrogate_term(
    new_token_logprobs: ArrayLike,
    old_token_logprobs: ArrayLike,
    advantage: float,
    *,
    eps_low: float = 3e-4,
    eps_high: float = 4e-4,
    mask: ArrayLike | None = None,
) -> float:
    """Sequence-level clipped GSPO term for one response."""
    if not math.isfinite(advantage):
        raise ValueError("advantage must be finite")
    ratio = sequence_importance_ratio(new_token_logprobs, old_token_logprobs, mask)
    return float(clipped_surrogate_terms([ratio], [advantage], eps_low=eps_low, eps_high=eps_high)[0])


def soft_overlong_penalty(length: int, max_length: int, cache_length: int) -> float:
    """DAPO soft overlong punishment from Eq. 13 of the paper."""
    if max_length <= 0 or cache_length <= 0 or cache_length > max_length:
        raise ValueError("require 0 < cache_length <= max_length")
    if length < 0:
        raise ValueError("length must be non-negative")
    boundary = max_length - cache_length
    if length <= boundary:
        return 0.0
    if length <= max_length:
        return float((boundary - length) / cache_length)
    return -1.0


def categorical_entropy(probabilities: ArrayLike) -> float:
    p = _array(probabilities, name="probabilities")
    if np.any(p < 0) or not np.isclose(p.sum(), 1.0, atol=1e-10):
        raise ValueError("probabilities must be non-negative and sum to one")
    mask = p > 0
    return float(-np.sum(p[mask] * np.log(p[mask])))


def pass_at_k_independent(success_probability: float, k: int) -> float:
    """At least one success in k IID attempts; not the finite-sample pass@k estimator."""
    if not 0 <= success_probability <= 1 or not math.isfinite(success_probability):
        raise ValueError("success_probability must lie in [0, 1]")
    if k <= 0:
        raise ValueError("k must be positive")
    return float(1.0 - (1.0 - success_probability) ** k)


@dataclass(frozen=True)
class RolloutRecord:
    prompt_id: str
    group_id: str
    token_ids: tuple[int, ...]
    action_mask: tuple[int, ...]
    old_logprobs: tuple[float, ...]
    reward_components: tuple[float, ...]
    stop_reason: str
    generator_version: str


def validate_rollout_record(record: RolloutRecord) -> None:
    """Validate the minimum internal consistency of a teaching rollout record."""
    if not record.prompt_id or not record.group_id or not record.generator_version:
        raise ValueError("prompt_id, group_id, and generator_version must be non-empty")
    n = len(record.token_ids)
    if n == 0 or len(record.action_mask) != n or len(record.old_logprobs) != n:
        raise ValueError("token_ids, action_mask, and old_logprobs must have equal non-zero length")
    if any(m not in (0, 1) for m in record.action_mask) or not any(record.action_mask):
        raise ValueError("action_mask must contain 0/1 and at least one active action")
    if not all(math.isfinite(x) for x in record.old_logprobs):
        raise ValueError("old_logprobs must be finite")
    if not record.reward_components or not all(math.isfinite(x) for x in record.reward_components):
        raise ValueError("reward_components must be a non-empty finite sequence")
    if record.stop_reason not in {"eos", "stop_string", "truncated", "environment", "error"}:
        raise ValueError("unsupported stop_reason")


def softmax(logits: ArrayLike) -> np.ndarray:
    z = _array(logits, name="logits")
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def synthetic_mode_training(
    *,
    beta_entropy: float,
    learning_rate: float,
    steps: int,
    rewards: ArrayLike = (1.0, 0.98, 0.0, 0.0),
) -> dict[str, np.ndarray | float]:
    """Exact expected-gradient toy illustrating concentration among solution modes."""
    if beta_entropy < 0 or learning_rate <= 0 or steps < 0:
        raise ValueError("require beta_entropy >= 0, learning_rate > 0, steps >= 0")
    r = _array(rewards, name="rewards")
    theta = np.zeros_like(r)
    checkpoints = {0, steps // 2, steps}
    entropies: list[float] = []
    for step in range(steps + 1):
        p = softmax(theta)
        if step in checkpoints:
            entropies.append(categorical_entropy(p))
        if step == steps:
            break
        mean_reward = float(np.dot(p, r))
        entropy = categorical_entropy(p)
        entropy_grad = p * (-np.log(np.maximum(p, 1e-300)) - entropy)
        reward_grad = p * (r - mean_reward)
        theta += learning_rate * (reward_grad + beta_entropy * entropy_grad)
    p = softmax(theta)
    return {
        "probabilities": p,
        "entropies": np.asarray(entropies, dtype=np.float64),
        "expected_reward": float(np.dot(p, r)),
        "successful_mass": float(p[:2].sum()) if p.size >= 2 else float(p.sum()),
    }


def flatten_token_weights(weights: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(w, dtype=np.float64) for w in weights]
    if not arrays:
        raise ValueError("weights must be non-empty")
    return np.concatenate(arrays)
