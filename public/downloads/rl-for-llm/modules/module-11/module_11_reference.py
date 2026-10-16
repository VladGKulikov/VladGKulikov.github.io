#!/usr/bin/env python3
"""Reference mathematics and contracts for RL for LLM, Module 11.

The module deliberately separates three layers:

* exact accounting identities (masks, probability ratios, TD telescoping,
  potential shaping, semi-Markov discounting);
* contract validation for agent trajectories;
* a small deterministic key-and-door example used in exercises and tests.

The larger recurrent-policy experiment lives in ``module_11_agentic_training``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil, exp, expm1, isfinite, log, log1p
from typing import Any

import numpy as np

ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
ALLOWED_PROVENANCE = {"policy", "environment", "template", "controller"}
ALLOWED_COLORS = {"red", "blue"}
KEYDOOR_ACTIONS = (
    "inspect",
    "pickup_red",
    "pickup_blue",
    "open_red",
    "open_blue",
    "finish",
)


@dataclass(frozen=True)
class MaskReport:
    policy_tokens: int
    total_tokens: int
    masked_fraction: float
    dilution_factor: float


@dataclass(frozen=True)
class StragglerReport:
    total_work: float
    wall_clock: float
    utilization: float
    idle_fraction: float


def _as_finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite numbers")
    return array


def _check_gamma(gamma: float) -> float:
    gamma = float(gamma)
    if not isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")
    return gamma


def build_policy_mask(roles: list[str], lengths: list[int]) -> list[int]:
    """Return a token mask with one only for assistant segments.

    This helper represents the common chat-template case.  When a controller
    inserts or rewrites assistant-looking text, use :func:`provenance_mask`
    instead: role labels alone are then insufficient.
    """
    if len(roles) != len(lengths):
        raise ValueError("roles and lengths must have equal lengths")
    result: list[int] = []
    for role, length in zip(roles, lengths, strict=True):
        if role not in ALLOWED_ROLES:
            raise ValueError(f"unknown role: {role}")
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise ValueError("segment lengths must be non-negative integers")
        result.extend([1 if role == "assistant" else 0] * length)
    return result


def provenance_mask(provenance: Sequence[str]) -> np.ndarray:
    """Return one exactly for tokens sampled by the behavior policy."""
    values = list(provenance)
    unknown = sorted(set(values) - ALLOWED_PROVENANCE)
    if unknown:
        raise ValueError(f"unknown token provenance: {', '.join(unknown)}")
    return np.asarray([int(value == "policy") for value in values], dtype=int)


def trajectory_mask_report(segments: Sequence[tuple[str, int]]) -> MaskReport:
    """Summarize policy and non-policy token counts for role-length segments."""
    if not segments:
        raise ValueError("segments must be non-empty")
    roles = [role for role, _ in segments]
    lengths = [length for _, length in segments]
    mask = build_policy_mask(roles, lengths)
    total = len(mask)
    policy = int(sum(mask))
    if total == 0:
        raise ValueError("trajectory must contain at least one token")
    if policy == 0:
        raise ValueError("trajectory contains no policy tokens")
    return MaskReport(
        policy_tokens=policy,
        total_tokens=total,
        masked_fraction=1.0 - policy / total,
        dilution_factor=total / policy,
    )


def masked_weighted_mean(values: Sequence[float], mask: Sequence[bool | int]) -> float:
    """Return the arithmetic mean over positions selected by ``mask``."""
    values_array = _as_finite_vector(values, "values")
    mask_array = np.asarray(mask)
    if mask_array.ndim != 1 or mask_array.shape != values_array.shape:
        raise ValueError("values and mask must be one-dimensional and have equal shape")
    if not np.isin(mask_array, [0, 1, False, True]).all():
        raise ValueError("mask must contain only zeros and ones")
    selected = mask_array.astype(bool)
    if not selected.any():
        raise ValueError("mask selects no positions")
    return float(values_array[selected].mean())


def macro_action_logprob(token_logprobs: Sequence[float]) -> float:
    """Log-probability of one autoregressively generated action string."""
    values = _as_finite_vector(token_logprobs, "token_logprobs")
    if values.size == 0:
        raise ValueError("token_logprobs must be non-empty")
    if np.any(values > 1e-12):
        raise ValueError("log-probabilities must be non-positive")
    return float(values.sum())


def turn_probability_ratio(
    current_token_logprobs: Sequence[float],
    behavior_token_logprobs: Sequence[float],
    *,
    max_log_ratio: float | None = None,
) -> float:
    """Exact ratio for the sampled action string, computed in log space."""
    current = _as_finite_vector(current_token_logprobs, "current_token_logprobs")
    behavior = _as_finite_vector(behavior_token_logprobs, "behavior_token_logprobs")
    if current.shape != behavior.shape or current.size == 0:
        raise ValueError("current and behavior log-probabilities must have equal non-empty shape")
    if np.any(current > 1e-12) or np.any(behavior > 1e-12):
        raise ValueError("log-probabilities must be non-positive")
    log_ratio = float(np.sum(current - behavior))
    if max_log_ratio is not None:
        bound = float(max_log_ratio)
        if not isfinite(bound) or bound <= 0.0:
            raise ValueError("max_log_ratio must be positive and finite")
        log_ratio = float(np.clip(log_ratio, -bound, bound))
    return float(np.exp(log_ratio))


def semantic_action_probability(
    string_probabilities: Mapping[str, float],
    parser: Mapping[str, str],
    target_action: str,
) -> float:
    """Sum probabilities of all strings parsed as ``target_action``."""
    if not string_probabilities:
        raise ValueError("string_probabilities must be non-empty")
    total = 0.0
    target = 0.0
    for text, probability_value in string_probabilities.items():
        if text not in parser:
            raise ValueError(f"missing parser entry for {text!r}")
        probability = float(probability_value)
        if not isfinite(probability) or probability < 0.0:
            raise ValueError("probabilities must be finite and non-negative")
        total += probability
        if parser[text] == target_action:
            target += probability
    if not np.isclose(total, 1.0, atol=1e-10):
        raise ValueError("string probabilities must sum to one")
    return float(target)


def td_residuals(
    rewards: Sequence[float],
    values: Sequence[float],
    gamma: float,
    terminated: Sequence[bool] | None = None,
) -> list[float]:
    """Compute one-step TD residuals at turn or macro-action boundaries.

    ``values`` has length ``T+1``.  ``terminated[t]`` removes bootstrapping
    after a true environment terminal transition.  A collector truncation
    should pass ``False`` and provide the value of the final observation.
    """
    rewards_array = _as_finite_vector(rewards, "rewards")
    values_array = _as_finite_vector(values, "values")
    gamma = _check_gamma(gamma)
    if values_array.size != rewards_array.size + 1:
        raise ValueError("values must have length len(rewards)+1")
    if terminated is None:
        terminal_array = np.zeros(rewards_array.size, dtype=bool)
    else:
        terminal_array = np.asarray(terminated, dtype=bool)
        if terminal_array.ndim != 1 or terminal_array.shape != rewards_array.shape:
            raise ValueError("terminated must match rewards")
    bootstrap = gamma * (~terminal_array).astype(float) * values_array[1:]
    return (rewards_array + bootstrap - values_array[:-1]).tolist()


def collector_target(reward: float, next_value: float, *, gamma: float, terminated: bool) -> float:
    """One-step target that bootstraps through collector truncation, not terminality."""
    reward = float(reward)
    next_value = float(next_value)
    if not isfinite(reward) or not isfinite(next_value):
        raise ValueError("reward and next_value must be finite")
    gamma = _check_gamma(gamma)
    return reward + gamma * (0.0 if terminated else next_value)


def discounted_return(rewards: Sequence[float], gamma: float) -> float:
    """Return ``sum_t gamma**t * rewards[t]``."""
    rewards_array = _as_finite_vector(rewards, "rewards")
    gamma = _check_gamma(gamma)
    powers = np.power(gamma, np.arange(rewards_array.size, dtype=float))
    return float(np.dot(powers, rewards_array))


def discounted_td_sum(residuals: Sequence[float], gamma: float) -> float:
    """Return ``sum_t gamma**t * delta[t]``."""
    residual_array = _as_finite_vector(residuals, "residuals")
    gamma = _check_gamma(gamma)
    powers = np.power(gamma, np.arange(residual_array.size, dtype=float))
    return float(np.dot(powers, residual_array))


def potential_shaping(
    rewards: Sequence[float],
    potentials: Sequence[float],
    gamma: float,
) -> list[float]:
    """Return ``r[t] + gamma*Phi[t+1] - Phi[t]``."""
    rewards_array = _as_finite_vector(rewards, "rewards")
    potentials_array = _as_finite_vector(potentials, "potentials")
    gamma = _check_gamma(gamma)
    if potentials_array.size != rewards_array.size + 1:
        raise ValueError("potentials must have length len(rewards)+1")
    return (rewards_array + gamma * potentials_array[1:] - potentials_array[:-1]).tolist()


def potential_shaped_rewards(
    rewards: Sequence[float],
    potentials: Sequence[float],
    *,
    gamma: float,
) -> np.ndarray:
    """Array compatibility wrapper around :func:`potential_shaping`."""
    return np.asarray(potential_shaping(rewards, potentials, gamma), dtype=float)


def shaped_return_identity(
    rewards: Sequence[float],
    potentials: Sequence[float],
    gamma: float,
) -> tuple[float, float, float]:
    """Return original, shaped, and telescoping-predicted shaped returns."""
    rewards_array = _as_finite_vector(rewards, "rewards")
    potentials_array = _as_finite_vector(potentials, "potentials")
    shaped = potential_shaping(rewards_array, potentials_array, gamma)
    original_return = discounted_return(rewards_array, gamma)
    shaped_return = discounted_return(shaped, gamma)
    horizon = rewards_array.size
    predicted = original_return - potentials_array[0] + (gamma**horizon) * potentials_array[-1]
    return original_return, shaped_return, float(predicted)


def informative_group_probability(p: float, group_size: int) -> float:
    """Probability of at least one success and one failure in a binary group."""
    p = float(p)
    if not isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
        raise ValueError("group_size must be a positive integer")
    if p in (0.0, 1.0):
        return 0.0
    at_least_one_success = -expm1(group_size * log1p(-p))
    all_success = exp(group_size * log(p))
    value = at_least_one_success - all_success
    return float(min(1.0, max(0.0, value)))


def degenerate_group_probability(p: float, group_size: int) -> float:
    return 1.0 - informative_group_probability(p, group_size)


def expected_degenerate_fraction(ps: Sequence[float], group_size: int) -> float:
    probabilities = _as_finite_vector(ps, "ps")
    if probabilities.size == 0 or np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("ps must be non-empty and lie in [0, 1]")
    return float(np.mean([degenerate_group_probability(float(p), group_size) for p in probabilities]))


def expected_groups_per_informative(p: float, group_size: int) -> float:
    probability = informative_group_probability(p, group_size)
    return float("inf") if probability == 0.0 else 1.0 / probability


def minimum_group_size_for_informative_probability(
    p: float,
    target_informative_probability: float,
    *,
    max_group_size: int = 100_000,
) -> int:
    p = float(p)
    target = float(target_informative_probability)
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly between zero and one")
    if not 0.0 < target < 1.0:
        raise ValueError("target must lie strictly between zero and one")
    if isinstance(max_group_size, bool) or not isinstance(max_group_size, int) or max_group_size < 1:
        raise ValueError("max_group_size must be a positive integer")
    dominant = max(p, 1.0 - p)
    estimate = max(1, int(ceil(log(1.0 - target) / log(dominant))))
    for group_size in range(max(1, estimate - 4), max_group_size + 1):
        if informative_group_probability(p, group_size) >= target:
            return group_size
    raise ValueError("target is not reached within max_group_size")


def loss_reductions(trajectories: Sequence[Sequence[float]]) -> dict[str, float]:
    """Compute token- and trajectory-level arithmetic reductions."""
    if not trajectories:
        raise ValueError("trajectories must be non-empty")
    arrays = [_as_finite_vector(values, f"trajectory {i}") for i, values in enumerate(trajectories)]
    if any(array.size == 0 for array in arrays):
        raise ValueError("each trajectory must contain at least one term")
    return {
        "token_mean": float(np.concatenate(arrays).mean()),
        "trajectory_mean": float(np.mean([array.mean() for array in arrays])),
    }


def semi_markov_return(
    rewards: Sequence[float],
    durations: Sequence[int],
    gamma: float,
) -> float:
    """Discount macro-action rewards by elapsed primitive time.

    ``rewards[i]`` is paid when macro-action ``i`` completes.  If cumulative
    primitive time after that action is ``c_i``, its contribution is
    ``gamma**c_i * rewards[i]``.  The convention is explicit because other
    codebases may index a completion reward one primitive step earlier.
    """
    reward_values = _as_finite_vector(rewards, "rewards")
    if len(durations) != reward_values.size or reward_values.size == 0:
        raise ValueError("rewards and durations must have equal non-empty lengths")
    if any(isinstance(d, bool) or not isinstance(d, int) or d <= 0 for d in durations):
        raise ValueError("durations must be positive integers")
    gamma = _check_gamma(gamma)
    completion_times = np.cumsum(np.asarray(durations, dtype=int))
    return float(np.dot(np.power(gamma, completion_times), reward_values))


def semi_markov_terminal_return(durations: Sequence[int], terminal_reward: float, gamma: float) -> float:
    return semi_markov_return([float(terminal_reward)], [sum(durations)], gamma)


def synchronous_straggler_report(durations: Sequence[float]) -> StragglerReport:
    values = _as_finite_vector(durations, "durations")
    if values.size == 0 or np.any(values <= 0.0):
        raise ValueError("durations must be non-empty and positive")
    wall_clock = float(values.max())
    total_work = float(values.sum())
    capacity = values.size * wall_clock
    utilization = total_work / capacity
    return StragglerReport(total_work, wall_clock, utilization, 1.0 - utilization)


def first_visit_milestone_rewards(
    visited_states: Sequence[str],
    milestone_rewards: Mapping[str, float],
) -> np.ndarray:
    """Pay each named milestone only on its first visit."""
    paid: set[str] = set()
    rewards: list[float] = []
    for state in visited_states:
        if state in milestone_rewards and state not in paid:
            value = float(milestone_rewards[state])
            if not isfinite(value):
                raise ValueError("milestone rewards must be finite")
            rewards.append(value)
            paid.add(state)
        else:
            rewards.append(0.0)
    return np.asarray(rewards, dtype=float)


def validate_rollout_record(record: Mapping[str, Any]) -> list[str]:
    """Return contract violations; an empty list denotes a valid record."""
    required = {
        "task_id",
        "environment_id",
        "environment_version",
        "token_ids",
        "policy_mask",
        "behavior_logprobs",
        "policy_versions",
        "turn_ids",
        "action_ids",
        "observations",
        "rewards",
        "stop_reason",
        "terminated",
        "truncated",
        "infra_error",
    }
    errors: list[str] = []
    missing = sorted(required - set(record))
    if missing:
        return [f"missing fields: {', '.join(missing)}"]

    token_ids = record["token_ids"]
    policy_mask = record["policy_mask"]
    turn_ids = record["turn_ids"]
    behavior_logprobs = record["behavior_logprobs"]
    policy_versions = record["policy_versions"]

    if not all(isinstance(item, list) for item in (token_ids, policy_mask, turn_ids, behavior_logprobs)):
        return ["token_ids, policy_mask, turn_ids, and behavior_logprobs must be lists"]
    if not (len(token_ids) == len(policy_mask) == len(turn_ids)):
        errors.append("token_ids, policy_mask, and turn_ids must have equal lengths")
    if not all(value in (0, 1) and not isinstance(value, bool) for value in policy_mask):
        errors.append("policy_mask must contain integer zeros and ones")
    policy_count = sum(value == 1 for value in policy_mask)
    if policy_count != len(behavior_logprobs):
        errors.append("behavior_logprobs length must equal the number of policy tokens")
    try:
        finite_logprobs = all(isfinite(float(value)) and float(value) <= 1e-12 for value in behavior_logprobs)
    except (TypeError, ValueError):
        finite_logprobs = False
    if not finite_logprobs:
        errors.append("behavior_logprobs must be finite non-positive numbers")

    if not isinstance(policy_versions, Mapping):
        errors.append("policy_versions must map turn ids to versions")
    elif len(token_ids) == len(policy_mask) == len(turn_ids):
        policy_turns = {str(turn) for turn, mask in zip(turn_ids, policy_mask, strict=True) if mask == 1}
        version_turns = {str(turn) for turn in policy_versions}
        missing_versions = sorted(policy_turns - version_turns)
        if missing_versions:
            errors.append(f"missing policy versions for turns: {', '.join(missing_versions)}")

    actions = record["action_ids"]
    observations = record["observations"]
    rewards = record["rewards"]
    if not all(isinstance(item, list) for item in (actions, observations, rewards)):
        errors.append("action_ids, observations, and rewards must be lists")
    elif not (len(actions) == len(rewards) and len(observations) == len(actions) + 1):
        errors.append("observations must have one more item than actions and rewards")

    status_count = sum(bool(record[name]) for name in ("terminated", "truncated", "infra_error"))
    if status_count > 1:
        errors.append("at most one terminal status may be true")
    if status_count == 0 and record["stop_reason"] is not None:
        errors.append("stop_reason requires a terminal, truncation, or infrastructure status")
    if status_count == 1 and not isinstance(record["stop_reason"], str):
        errors.append("a stopped rollout must have a string stop_reason")
    return errors


def run_keydoor_episode(
    actions: list[str],
    *,
    key_color: str,
    door_color: str,
    max_steps: int = 6,
) -> dict[str, Any]:
    """Execute a deterministic key-and-door episode and return a report."""
    if key_color not in ALLOWED_COLORS or door_color not in ALLOWED_COLORS:
        raise ValueError("key_color and door_color must be 'red' or 'blue'")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if not isinstance(actions, list):
        raise ValueError("actions must be a list")

    inspected = False
    carrying: str | None = None
    opened: set[str] = set()
    invalid_actions = 0
    terminated = False
    success = False
    steps = 0

    for action in actions[:max_steps]:
        steps += 1
        valid = True
        if action == "inspect":
            inspected = True
        elif action in {"pickup_red", "pickup_blue"}:
            color = action.removeprefix("pickup_")
            if not inspected or color != key_color:
                valid = False
            else:
                carrying = color
        elif action in {"open_red", "open_blue"}:
            color = action.removeprefix("open_")
            if carrying != color:
                valid = False
            else:
                opened.add(color)
        elif action == "finish":
            terminated = True
            success = door_color in opened
        else:
            valid = False
        if not valid:
            invalid_actions += 1
        if terminated:
            break

    truncated = not terminated and steps >= max_steps
    return {
        "reward": 1.0 if success else 0.0,
        "success": success,
        "terminated": terminated,
        "truncated": truncated,
        "steps": steps,
        "invalid_actions": invalid_actions,
        "carrying": carrying,
        "opened": sorted(opened),
    }


def keydoor_state_key(
    *,
    inspected: bool,
    carrying: str | None,
    opened: str | None,
    steps_left: int,
) -> tuple[int, int, int, int]:
    carrying_code = {None: 0, "red": 1, "blue": 2}[carrying]
    opened_code = {None: 0, "red": 1, "blue": 2}[opened]
    return int(inspected), carrying_code, opened_code, int(steps_left)


__all__ = [name for name in globals() if not name.startswith("_")]
