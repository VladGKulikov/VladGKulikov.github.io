#!/usr/bin/env python3
"""Deterministic reference calculations for Module 15.

The learner notebooks embed the resulting JSON payload so they remain self-contained in
Google Colab.  Maintainer tools read this module-owned payload; they never derive numbers
from mutable lecture prose.  Rewriting the frozen data requires the explicit ``--write``
flag.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parent
REFERENCE_PATH = MODULE_ROOT / "data" / "reference_metrics.json"


def _probability(name: str, value: float) -> float:
    """Validate and return one finite probability in ``[0, 1]``."""

    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return value


def compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    """Yield all weak compositions of ``total`` into ``parts`` non-negative counts."""

    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError("total must be a non-negative integer")
    if isinstance(parts, bool) or not isinstance(parts, int) or parts <= 0:
        raise ValueError("parts must be a positive integer")
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for tail in compositions(total - first, parts - 1):
            yield (first, *tail)


def multinomial_probability(counts: Sequence[int], probabilities: Sequence[float]) -> float:
    """Return the multinomial mass for one count vector using log-gamma arithmetic."""

    if len(counts) != len(probabilities) or len(counts) == 0:
        raise ValueError("counts and probabilities must have the same non-zero length")
    if any(isinstance(count, bool) or (not isinstance(count, int)) or count < 0 for count in counts):
        raise ValueError("counts must be non-negative integers")
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1 or not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    if not math.isclose(float(probs.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("probabilities must sum to one")

    n = sum(counts)
    log_mass = math.lgamma(n + 1)
    for count, probability in zip(counts, probs, strict=True):
        log_mass -= math.lgamma(count + 1)
        if count:
            if probability == 0.0:
                return 0.0
            log_mass += count * math.log(float(probability))
    return math.exp(log_mass)


def exact_vote_statistics(
    n_samples: int,
    p_correct: float,
    n_distractors: int,
) -> dict[str, float]:
    """Compute exact self-consistency statistics under symmetric distractors.

    Candidate zero is the correct final answer.  ``unique_mode_success`` requires its
    count to exceed every distractor count.  ``random_tie_success`` gives the correct
    answer probability ``1 / number_of_modes`` when it shares the top count.
    """

    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    if isinstance(n_distractors, bool) or not isinstance(n_distractors, int) or n_distractors <= 0:
        raise ValueError("n_distractors must be a positive integer")
    p_correct = _probability("p_correct", p_correct)
    probabilities = [p_correct] + [(1.0 - p_correct) / n_distractors] * n_distractors

    unique_mode_success = 0.0
    random_tie_success = 0.0
    any_mode_tie = 0.0
    correct_in_top_tie = 0.0
    total_mass = 0.0

    for counts in compositions(n_samples, n_distractors + 1):
        mass = multinomial_probability(counts, probabilities)
        total_mass += mass
        top = max(counts)
        modes = [index for index, count in enumerate(counts) if count == top]
        if len(modes) > 1:
            any_mode_tie += mass
        if counts[0] > max(counts[1:]):
            unique_mode_success += mass
        if 0 in modes:
            random_tie_success += mass / len(modes)
            if len(modes) > 1:
                correct_in_top_tie += mass

    if not math.isclose(total_mass, 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise AssertionError(f"multinomial mass does not sum to one: {total_mass}")
    return {
        "unique_mode_success": unique_mode_success,
        "random_tie_success": random_tie_success,
        "any_mode_tie": any_mode_tie,
        "correct_in_top_tie": correct_in_top_tie,
    }


def perfect_verifier_coverage(p_correct: float, n_samples: int) -> float:
    """Return the oracle probability that at least one of ``n_samples`` is correct."""

    p_correct = _probability("p_correct", p_correct)
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    return 1.0 if p_correct == 1 else -math.expm1(n_samples * math.log1p(-p_correct))


def noisy_verifier_success(
    p_correct: float,
    n_samples: int,
    true_positive_rate: float,
    false_positive_rate: float,
    *,
    fallback: str = "random_candidate",
) -> float:
    """Return exact selection accuracy for an accept/reject verifier.

    Correct and incorrect candidates are accepted independently with the supplied true-
    and false-positive rates.  Selection is uniform among accepted candidates.  If none
    are accepted, ``random_candidate`` falls back to a uniformly chosen generated
    candidate, while ``fail`` assigns zero success.
    """

    p_correct = _probability("p_correct", p_correct)
    true_positive_rate = _probability("true_positive_rate", true_positive_rate)
    false_positive_rate = _probability("false_positive_rate", false_positive_rate)
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    if fallback not in {"random_candidate", "fail"}:
        raise ValueError("fallback must be 'random_candidate' or 'fail'")

    success = 0.0
    for n_correct in range(n_samples + 1):
        p_n_correct = (
            math.comb(n_samples, n_correct)
            * p_correct**n_correct
            * (1.0 - p_correct) ** (n_samples - n_correct)
        )
        for accepted_correct in range(n_correct + 1):
            p_accepted_correct = (
                math.comb(n_correct, accepted_correct)
                * true_positive_rate**accepted_correct
                * (1.0 - true_positive_rate) ** (n_correct - accepted_correct)
            )
            n_wrong = n_samples - n_correct
            for accepted_wrong in range(n_wrong + 1):
                p_accepted_wrong = (
                    math.comb(n_wrong, accepted_wrong)
                    * false_positive_rate**accepted_wrong
                    * (1.0 - false_positive_rate) ** (n_wrong - accepted_wrong)
                )
                accepted_total = accepted_correct + accepted_wrong
                if accepted_total:
                    conditional_success = accepted_correct / accepted_total
                elif fallback == "random_candidate":
                    conditional_success = n_correct / n_samples
                else:
                    conditional_success = 0.0
                success += (
                    p_n_correct
                    * p_accepted_correct
                    * p_accepted_wrong
                    * conditional_success
                )
    return success


@dataclass(frozen=True)
class CostScenario:
    """Teaching assumptions for dense-decoder latency, FLOPs, and KV accounting."""

    n_parameters: float = 8.03e9
    decode_tokens_per_second: float = 209.4
    n_layers: int = 32
    kv_width: int = 1024
    bytes_per_element: int = 2
    device_budget_gib: float = 80.0
    weight_bytes_per_parameter: int = 2
    runtime_reserve_gib: float = 0.0

    def validate(self) -> None:
        """Reject non-finite, non-positive, or internally impossible assumptions."""

        numeric_positive = {
            "n_parameters": self.n_parameters,
            "decode_tokens_per_second": self.decode_tokens_per_second,
            "device_budget_gib": self.device_budget_gib,
        }
        for name, value in numeric_positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in {
            "n_layers": self.n_layers,
            "kv_width": self.kv_width,
            "bytes_per_element": self.bytes_per_element,
            "weight_bytes_per_parameter": self.weight_bytes_per_parameter,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.runtime_reserve_gib) or self.runtime_reserve_gib < 0:
            raise ValueError("runtime_reserve_gib must be finite and non-negative")
        if self.available_kv_gib <= 0:
            raise ValueError("weights plus runtime reserve leave no room for KV cache")

    @property
    def weight_gib(self) -> float:
        """Return model-weight memory in GiB under the stated byte assumption."""

        return self.n_parameters * self.weight_bytes_per_parameter / 2**30

    @property
    def available_kv_gib(self) -> float:
        """Return the teaching budget left for KV after weights and explicit reserve."""

        return self.device_budget_gib - self.weight_gib - self.runtime_reserve_gib

    @property
    def kv_bytes_per_token(self) -> int:
        """Return K+V bytes per generated token across all layers."""

        return 2 * self.bytes_per_element * self.kv_width * self.n_layers


def estimate_reasoning_cost(tokens: int, scenario: CostScenario) -> dict[str, float | int]:
    """Estimate fixed-rate decode, approximate weight-matmul work and generated KV.\n\n    The legacy dense_decode_flops_floor key is not a general FLOP lower bound."""

    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
        raise ValueError("tokens must be a positive integer")
    scenario.validate()
    latency_seconds = tokens / scenario.decode_tokens_per_second
    dense_decode_flops_floor = 2.0 * scenario.n_parameters * tokens
    kv_bytes = scenario.kv_bytes_per_token * tokens
    kv_gib = kv_bytes / 2**30
    concurrency_upper_bound = int(scenario.available_kv_gib // kv_gib)
    return {
        "tokens": tokens,
        "latency_seconds": latency_seconds,
        "dense_decode_flops_floor": dense_decode_flops_floor,
        "kv_bytes": kv_bytes,
        "kv_gib": kv_gib,
        "concurrency_upper_bound": concurrency_upper_bound,
    }


def simulate_adaptive_search(
    *,
    n_prompts: int,
    max_candidates: int,
    threshold: float,
    seed: int,
    difficulty_weights: Sequence[float],
    p_correct_by_difficulty: Sequence[float],
) -> dict[str, float]:
    """Simulate verifier-threshold early stopping on a labelled synthetic workload.

    Scores overlap deliberately: a correct candidate receives ``0.52 + 0.48 U`` and an
    incorrect candidate ``0.03 + 0.90 U²``.  The simulation is a controlled mechanism
    demo, not a benchmark of any real verifier.
    """

    if isinstance(n_prompts, bool) or not isinstance(n_prompts, int) or n_prompts <= 0:
        raise ValueError("n_prompts must be a positive integer")
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates <= 0:
        raise ValueError("max_candidates must be a positive integer")
    threshold = _probability("threshold", threshold)
    weights = np.asarray(difficulty_weights, dtype=np.float64)
    probabilities = np.asarray(p_correct_by_difficulty, dtype=np.float64)
    if weights.ndim != 1 or probabilities.shape != weights.shape or weights.size == 0:
        raise ValueError("difficulty weights and probabilities must be matching vectors")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("difficulty weights must be finite and non-negative")
    if not math.isclose(float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("difficulty weights must sum to one")
    if np.any(~np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("per-difficulty correctness probabilities must lie in [0, 1]")

    rng = np.random.default_rng(seed)
    difficulty = rng.choice(weights.size, size=n_prompts, p=weights)
    is_correct = rng.random((n_prompts, max_candidates)) < probabilities[difficulty, None]
    u = rng.random((n_prompts, max_candidates))
    scores = np.where(is_correct, 0.52 + 0.48 * u, 0.03 + 0.90 * u**2)

    crosses = scores >= threshold
    has_crossing = crosses.any(axis=1)
    first_crossing = crosses.argmax(axis=1)
    best_index = scores.argmax(axis=1)
    selected_index = np.where(has_crossing, first_crossing, best_index)
    attempts = np.where(has_crossing, first_crossing + 1, max_candidates)

    rows = np.arange(n_prompts)
    return {
        "one_shot_accuracy": float(is_correct[:, 0].mean()),
        "adaptive_accuracy": float(is_correct[rows, selected_index].mean()),
        "adaptive_mean_candidates": float(attempts.mean()),
        "fixed_best_of_n_accuracy": float(is_correct[rows, best_index].mean()),
        "oracle_coverage": float(is_correct.any(axis=1).mean()),
        "threshold_crossing_rate": float(has_crossing.mean()),
    }


def build_reference() -> dict[str, Any]:
    """Build the complete frozen numerical contract for Module 15."""

    p_correct = 0.4
    n_distractors = 6
    sample_counts = [1, 3, 5, 9, 17]
    vote = {
        str(n): exact_vote_statistics(n, p_correct, n_distractors)
        for n in sample_counts
    }
    perfect = {str(n): perfect_verifier_coverage(p_correct, n) for n in sample_counts}
    noisy_tpr, noisy_fpr = 0.90, 0.05
    noisy = {
        str(n): noisy_verifier_success(
            p_correct,
            n,
            noisy_tpr,
            noisy_fpr,
            fallback="random_candidate",
        )
        for n in sample_counts
    }

    candidate_batch = {
        "answer_ids": [
            [0, 1, 0, 2, 0],
            [3, 4, 5, 3, 6],
            [7, 8, 7, 8, 9],
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 20],
        ],
        "correct_answer_ids": [0, 3, 7, 14, 24],
        "verifier_scores": [
            [0.84, 0.22, 0.92, 0.33, 0.76],
            [0.55, 0.90, 0.18, 0.62, 0.31],
            [0.70, 0.80, 0.65, 0.58, 0.25],
            [0.12, 0.20, 0.18, 0.24, 0.88],
            [0.48, 0.32, 0.29, 0.17, 0.51],
        ],
        "token_counts": [
            [120, 96, 144, 110, 132],
            [210, 170, 190, 225, 160],
            [88, 105, 92, 99, 80],
            [310, 280, 295, 260, 340],
            [130, 118, 140, 125, 150],
        ],
    }

    ladder = {"low": 1_000, "medium": 4_000, "high": 16_000, "xhigh": 64_000}
    ideal_scenario = CostScenario(runtime_reserve_gib=0.0)
    reserved_scenario = CostScenario(runtime_reserve_gib=8.0)
    costs = {name: estimate_reasoning_cost(tokens, ideal_scenario) for name, tokens in ladder.items()}
    reserved_concurrency = {
        name: estimate_reasoning_cost(tokens, reserved_scenario)["concurrency_upper_bound"]
        for name, tokens in ladder.items()
    }
    short = estimate_reasoning_cost(512, ideal_scenario)
    forcing_extra_seconds = (8_000 - 2_000) / ideal_scenario.decode_tokens_per_second

    adaptive_mix = {
        "classes": ["simple", "medium", "hard"],
        "weights": [0.7, 0.2, 0.1],
        "budgets": [500, 4_000, 32_000],
        "uniform_budget": 32_000,
    }
    adaptive_average = float(
        np.dot(adaptive_mix["weights"], adaptive_mix["budgets"])
    )

    adaptive_search_parameters = {
        "n_prompts": 50_000,
        "max_candidates": 16,
        "threshold": 0.82,
        "seed": 13,
        "difficulty_weights": [0.7, 0.2, 0.1],
        "p_correct_by_difficulty": [0.75, 0.45, 0.20],
    }
    adaptive_search = simulate_adaptive_search(**adaptive_search_parameters)

    overthinking_toy = {
        "budgets": [256, 1_024, 4_096, 16_384],
        "simple_accuracy": [0.93, 0.95, 0.94, 0.90],
        "hard_accuracy": [0.18, 0.31, 0.54, 0.68],
        "note": "synthetic mechanism illustration, not a benchmark",
    }

    return {
        "schema_version": 2,
        "vote": {
            "p_correct": p_correct,
            "n_distractors": n_distractors,
            "sample_counts": sample_counts,
            "statistics": vote,
        },
        "verifier": {
            "perfect_coverage": perfect,
            "noisy": {
                "true_positive_rate": noisy_tpr,
                "false_positive_rate": noisy_fpr,
                "fallback": "random_candidate",
                "selection_accuracy": noisy,
            },
        },
        "candidate_batch": candidate_batch,
        "cost": {
            "ladder": ladder,
            "base_tokens": 512,
            "scenario": ideal_scenario.__dict__,
            "reserved_scenario": reserved_scenario.__dict__,
            "costs": costs,
            "reserved_concurrency_upper_bound": reserved_concurrency,
            "short_512": short,
            "forcing_extra_seconds": forcing_extra_seconds,
        },
        "adaptive_mix": {
            **adaptive_mix,
            "average_budget": adaptive_average,
            "saving_factor": adaptive_mix["uniform_budget"] / adaptive_average,
        },
        "adaptive_search": {
            "parameters": adaptive_search_parameters,
            "metrics": adaptive_search,
        },
        "overthinking_toy": overthinking_toy,
    }


def write_reference(path: Path = REFERENCE_PATH) -> Path:
    """Write deterministic UTF-8 JSON and return the output path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(build_reference(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    """Print the contract; rewrite it only with ``--write``."""

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
