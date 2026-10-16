#!/usr/bin/env python3
"""Deterministic reference calculations for Module 14.

The learner notebooks embed the resulting JSON payload so they remain independent of
adjacent repository files in a clean Colab runtime. Maintainers regenerate the payload
only with the explicit ``--write`` flag.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parent
REFERENCE_PATH = MODULE_ROOT / "data" / "reference_metrics.json"


def stable_sigmoid(x: np.ndarray | float) -> np.ndarray:
    """Return sigmoid(x) without overflow for large positive or negative inputs."""

    array = np.asarray(x, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -array))


def dpo_statistics(
    policy_chosen: np.ndarray | float,
    policy_rejected: np.ndarray | float,
    reference_chosen: np.ndarray | float,
    reference_rejected: np.ndarray | float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return DPO margin, stable loss, and positive gradient-weight magnitude."""

    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    margin = beta * (
        (np.asarray(policy_chosen, dtype=np.float64) - np.asarray(reference_chosen, dtype=np.float64))
        - (np.asarray(policy_rejected, dtype=np.float64) - np.asarray(reference_rejected, dtype=np.float64))
    )
    loss = np.logaddexp(0.0, -margin)  # softplus(-margin) == -log(sigmoid(margin))
    weight = stable_sigmoid(-margin)
    return margin, loss, weight


def masked_sum(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Sum token values over a boolean completion mask, validating every sequence."""

    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if values.shape != mask.shape or values.ndim != 2:
        raise ValueError("values and mask must have the same [batch, tokens] shape")
    if np.any(mask.sum(axis=1) == 0):
        raise ValueError("every sequence must contain at least one completion token")
    if not np.all(np.isfinite(values[mask])):
        raise ValueError("active token log-probabilities must be finite")
    return np.where(mask, values, 0.0).sum(axis=1, dtype=np.float64)


def grpo_advantages(rewards: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Compute population-standardized group rewards and reject zero-variance groups."""

    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 1 or rewards.size < 2 or not np.all(np.isfinite(rewards)):
        raise ValueError("rewards must be a finite one-dimensional group of size >= 2")
    mean = float(rewards.mean())
    std = float(rewards.std(ddof=0))
    if std == 0.0:
        raise ValueError("a degenerate group has zero reward variance and must be filtered")
    return (rewards - mean) / std, mean, std


def memory_gib(n_params: float, stack: list[str]) -> float:
    """Estimate parameter-state memory in GiB for the stated teaching assumptions."""

    if not math.isfinite(n_params) or n_params <= 0:
        raise ValueError("n_params must be finite and positive")
    bytes_per_parameter = {"train": 16, "frozen_bf16": 2}
    try:
        total_bytes = sum(bytes_per_parameter[kind] * n_params for kind in stack)
    except KeyError as exc:
        raise ValueError(f"unknown stack component: {exc.args[0]}") from exc
    return total_bytes / 2**30


def build_reference() -> dict[str, Any]:
    """Build the complete frozen numerical contract for the module."""

    deltas = np.asarray([0.5, 1.0, 2.0, 4.0], dtype=np.float64)
    probabilities = stable_sigmoid(deltas)
    inverse_075 = math.log(0.75 / 0.25)

    scalar_cases: dict[str, dict[str, float]] = {}
    raw_cases = {
        "pair_I_beta_01": (-12.0, -11.5, -13.0, -10.8, 0.1),
        "pair_II_beta_01": (-12.0, -11.5, -12.8, -11.65, 0.1),
        "pair_I_beta_05": (-12.0, -11.5, -13.0, -10.8, 0.5),
    }
    for name, (pc, pr, rc, rr, beta) in raw_cases.items():
        margin, loss, weight = dpo_statistics(pc, pr, rc, rr, beta)
        scalar_cases[name] = {
            "policy_chosen": pc,
            "policy_rejected": pr,
            "reference_chosen": rc,
            "reference_rejected": rr,
            "beta": beta,
            "margin": float(margin),
            "loss": float(loss),
            "weight": float(weight),
        }

    # The sentinel 99.0 sits outside each completion mask. It proves that padding and
    # prompt positions are excluded rather than accidentally included in the sequence sum.
    completion_mask = np.asarray([[1, 1, 1, 1, 0]] * 3, dtype=bool)
    token_logps = {
        "policy_chosen": np.asarray(
            [[-2.0, -3.0, -3.5, -3.5, 99.0], [-3.0, -3.0, -3.0, -3.0, 99.0], [-2.0, -2.5, -3.0, -3.0, 99.0]],
            dtype=np.float64,
        ),
        "reference_chosen": np.asarray(
            [[-2.5, -3.0, -3.5, -4.0, 99.0], [-3.2, -3.2, -3.2, -3.2, 99.0], [-2.0, -2.5, -2.5, -3.0, 99.0]],
            dtype=np.float64,
        ),
        "policy_rejected": np.asarray(
            [[-2.0, -2.5, -3.0, -4.0, 99.0], [-2.5, -2.5, -3.0, -3.5, 99.0], [-2.0, -2.5, -3.0, -3.5, 99.0]],
            dtype=np.float64,
        ),
        "reference_rejected": np.asarray(
            [[-1.8, -2.2, -3.0, -3.8, 99.0], [-2.5, -2.65, -3.0, -3.5, 99.0], [-2.5, -3.0, -3.0, -3.5, 99.0]],
            dtype=np.float64,
        ),
    }
    sequence_logps = {name: masked_sum(values, completion_mask) for name, values in token_logps.items()}
    batch_margin, batch_loss, batch_weight = dpo_statistics(
        sequence_logps["policy_chosen"],
        sequence_logps["policy_rejected"],
        sequence_logps["reference_chosen"],
        sequence_logps["reference_rejected"],
        beta=0.1,
    )

    rewards = np.asarray([1, 0, 0, 1, 0, 0, 1, 0], dtype=np.float64)
    advantages, reward_mean, reward_std = grpo_advantages(rewards)
    pass_probability = 0.3
    group_size = 8
    degenerate_probability = (1.0 - pass_probability) ** group_size + pass_probability**group_size

    bernoulli_p = 0.3
    true_gradient = bernoulli_p * (1.0 - bernoulli_p)
    full_group_expected = (group_size - 1) / group_size * true_gradient

    n_params = 8.03e9
    train_gib = memory_gib(n_params, ["train"])
    frozen_gib = memory_gib(n_params, ["frozen_bf16"])
    ppo_gib = memory_gib(n_params, ["train", "train", "frozen_bf16", "frozen_bf16"])
    dpo_gib = memory_gib(n_params, ["train", "frozen_bf16"])

    prompts, generations, completion_tokens = 256, 8, 4000
    accelerators = 8
    decode_tokens_per_second = 209.4
    accelerator_flops = 990e12
    mfu = 0.40
    generated_tokens = prompts * generations * completion_tokens
    generation_minutes = generated_tokens / (accelerators * decode_tokens_per_second) / 60.0
    training_minutes = 6 * n_params * generated_tokens / (accelerators * accelerator_flops * mfu) / 60.0

    return {
        "schema_version": 2,
        "bt": {
            "deltas": deltas.tolist(),
            "probabilities": probabilities.tolist(),
            "inverse_win_rate": 0.75,
            "inverse_margin": inverse_075,
        },
        "dpo": {
            "scalar_cases": scalar_cases,
            "batch": {
                "beta": 0.1,
                "completion_mask": completion_mask.astype(int).tolist(),
                "token_logps": {name: values.tolist() for name, values in token_logps.items()},
                "sequence_logps": {name: values.tolist() for name, values in sequence_logps.items()},
                "margin": batch_margin.tolist(),
                "loss": batch_loss.tolist(),
                "weight": batch_weight.tolist(),
                "accuracy": (batch_margin > 0).tolist(),
                "mean_loss": float(batch_loss.mean()),
            },
        },
        "grpo": {
            "rewards": rewards.tolist(),
            "advantages": advantages.tolist(),
            "mean": reward_mean,
            "std": reward_std,
            "pass_probability": pass_probability,
            "group_size": group_size,
            "degenerate_probability": degenerate_probability,
        },
        "baseline_estimator": {
            "bernoulli_p": bernoulli_p,
            "group_size": group_size,
            "true_gradient": true_gradient,
            "full_group_mean_expected": full_group_expected,
        },
        "memory": {
            "n_params": n_params,
            "bytes_per_parameter": {"train": 16, "frozen_bf16": 2},
            "train_gib": train_gib,
            "frozen_gib": frozen_gib,
            "ppo_gib": ppo_gib,
            "dpo_grpo_gib": dpo_gib,
            "ratio": ppo_gib / dpo_gib,
        },
        "rollout": {
            "prompts": prompts,
            "generations_per_prompt": generations,
            "completion_tokens": completion_tokens,
            "accelerators": accelerators,
            "decode_tokens_per_second_per_accelerator": decode_tokens_per_second,
            "accelerator_flops": accelerator_flops,
            "mfu": mfu,
            "generated_tokens": generated_tokens,
            "generation_minutes": generation_minutes,
            "training_minutes": training_minutes,
            "ratio": generation_minutes / training_minutes,
        },
    }


def write_reference(path: Path = REFERENCE_PATH) -> Path:
    """Write a deterministic UTF-8 JSON payload and return its path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(build_reference(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    """Print the contract; rewrite it only when explicitly requested."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="rewrite data/reference_metrics.json")
    args = parser.parse_args()
    payload = build_reference()
    if args.write:
        path = write_reference()
        print(f"wrote {path}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        print("read-only mode; pass --write to update the frozen payload")


if __name__ == "__main__":
    main()
