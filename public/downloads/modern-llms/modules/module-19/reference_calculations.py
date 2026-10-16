#!/usr/bin/env python3
"""Reference calculations and validation helpers for Module 19.

The module owns a frozen, lecture-independent numerical contract for the learner
notebooks. It deliberately separates aggregate-rate approximations, paired model
comparisons, finite-sample pass@k, judge protocol effects, a synthetic long-context
profile, and transparent golden-set cost arithmetic.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parent
REFERENCE_PATH = MODULE_ROOT / "data" / "reference_metrics.json"


def _positive_int(name: str, value: int) -> int:
    """Validate and return a strictly positive, non-boolean integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(name: str, value: int) -> int:
    """Validate and return a non-negative, non-boolean integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _probability(name: str, value: float) -> float:
    """Validate and return a finite probability in the closed unit interval."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return number


def _positive_finite(name: str, value: float) -> float:
    """Validate and return a finite positive real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def binomial_standard_error(success_probability: float, sample_size: int) -> float:
    """Return the plug-in standard error of one Bernoulli proportion."""
    p = _probability("success_probability", success_probability)
    n = _positive_int("sample_size", sample_size)
    return math.sqrt(p * (1.0 - p) / n)


def aggregate_difference_stats(
    first_rate: float,
    second_rate: float,
    sample_size: int,
    z_value: float = 1.96,
) -> dict[str, float]:
    """Return a rough independent-rate comparison from aggregate percentages.

    When both models are evaluated on the same tasks, exact inference should use
    paired per-example outcomes. Aggregate scores alone cannot reconstruct covariance.
    """
    p1 = _probability("first_rate", first_rate)
    p2 = _probability("second_rate", second_rate)
    n = _positive_int("sample_size", sample_size)
    z_value = _positive_finite("z_value", z_value)
    se = math.sqrt(binomial_standard_error(p1, n) ** 2 + binomial_standard_error(p2, n) ** 2)
    delta = p1 - p2
    half_width = z_value * se
    return {
        "delta": delta,
        "se": se,
        "z_score": delta / se if se else None,
        "ci_low": delta - half_width,
        "ci_high": delta + half_width,
        "half_width": half_width,
    }


def required_independent_tasks(delta: float, ceiling_rate: float = 0.95, z_value: float = 1.96) -> int:
    """Plan tasks per model for CI half-width delta, not a specified test power."""
    delta = _positive_finite("delta", delta)
    p = _probability("ceiling_rate", ceiling_rate)
    z_value = _positive_finite("z_value", z_value)
    if p in {0.0, 1.0}:
        raise ValueError("ceiling_rate must be strictly between 0 and 1")
    return math.ceil(2.0 * p * (1.0 - p) * (z_value / delta) ** 2)


def wilson_interval(successes: int, total: int, z_value: float = 1.96) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial proportion."""
    n = _positive_int("total", total)
    s = _non_negative_int("successes", successes)
    if s > n:
        raise ValueError("successes cannot exceed total")
    z_value = _positive_finite("z_value", z_value)
    p = s / n
    z2 = z_value**2
    denominator = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denominator
    radius = z_value * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n**2)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def mcnemar_exact_pvalue(first_only: int, second_only: int) -> float:
    """Return the two-sided exact McNemar p-value for discordant paired outcomes."""
    first_only = _non_negative_int("first_only", first_only)
    second_only = _non_negative_int("second_only", second_only)
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = min(first_only, second_only)
    cdf = sum(math.comb(discordant, k) for k in range(tail + 1)) / (1 << discordant)
    return min(1.0, 2.0 * cdf)


def pass_at_k_unbiased(total_samples: int, correct_samples: int, k: int) -> float:
    """Return the finite-sample unbiased pass@k estimator of Chen et al."""
    n = _positive_int("total_samples", total_samples)
    c = _non_negative_int("correct_samples", correct_samples)
    k = _positive_int("k", k)
    if c > n:
        raise ValueError("correct_samples cannot exceed total_samples")
    if k > n:
        raise ValueError("k cannot exceed total_samples")
    if n - c < k:
        return 1.0
    # Use the shorter symmetric product and avoid cancellation near zero.
    if c == 0:
        return 0.0
    count, removed = (k, c) if k <= c else (c, k)
    if removed / n == 1.0:  # the success probability rounds to one at float precision
        return 1.0
    return -math.expm1(sum(math.log1p(-removed / (n - i)) for i in range(count)))


def pass_at_k_naive(total_samples: int, correct_samples: int, k: int) -> float:
    """Return the plug-in estimate ``1 - (1 - c/n)**k`` for comparison."""
    n = _positive_int("total_samples", total_samples)
    c = _non_negative_int("correct_samples", correct_samples)
    k = _positive_int("k", k)
    if c > n:
        raise ValueError("correct_samples cannot exceed total_samples")
    if k > n:
        raise ValueError("k cannot exceed total_samples")
    return 1.0 if c / n == 1.0 else -math.expm1(k * math.log1p(-c / n))


def working_context_share(edge: float, dip: float, threshold: float) -> float:
    """Return the fraction of positions above threshold for a synthetic U profile."""
    edge = _probability("edge", edge)
    threshold = _probability("threshold", threshold)
    dip = _probability("dip", dip)
    if edge - dip < 0.0:
        raise ValueError("edge - dip must remain a valid probability")
    if threshold > edge:
        return 0.0
    if threshold <= edge - dip:
        return 1.0
    return 2.0 * math.asin(math.sqrt((edge - threshold) / dip)) / math.pi


def golden_sample_size(margin: float, expected_rate: float = 0.8, z_value: float = 1.96) -> int:
    """Return a normal-approximation planning size for a target half-width."""
    margin = _positive_finite("margin", margin)
    p = _probability("expected_rate", expected_rate)
    z_value = _positive_finite("z_value", z_value)
    if p in {0.0, 1.0}:
        raise ValueError("expected_rate must be strictly between 0 and 1")
    return math.ceil(p * (1.0 - p) * (z_value / margin) ** 2)


def canary_guess_probability(hex_characters: int) -> float:
    """Return the chance of uniformly guessing a hexadecimal canary exactly."""
    length = _positive_int("hex_characters", hex_characters)
    return 16.0 ** (-length)


@dataclass(frozen=True)
class EvalCostScenario:
    """Conditional 2PT accounting; field names retain backward compatibility."""
    parameter_count: int
    examples: int
    prompt_tokens: int
    completion_tokens: int
    effective_flops_per_second: float

    def __post_init__(self) -> None:
        _positive_int("parameter_count", self.parameter_count)
        _positive_int("examples", self.examples)
        _positive_int("prompt_tokens", self.prompt_tokens)
        _positive_int("completion_tokens", self.completion_tokens)
        _positive_finite("effective_flops_per_second", self.effective_flops_per_second)

    def total_tokens(self) -> int:
        """Return prompt plus completion tokens across all examples."""
        return self.examples * (self.prompt_tokens + self.completion_tokens)

    def dense_flops_floor(self) -> float:
        """Return the approximate linear-in-parameters 2PT operation count."""
        return 2.0 * self.parameter_count * self.total_tokens()

    def seconds_floor(self) -> float:
        """Return a conditional arithmetic equivalent, not a latency bound."""
        return self.dense_flops_floor() / self.effective_flops_per_second


def build_reference_payload() -> dict[str, Any]:
    """Build the deterministic numerical contract embedded in the notebooks."""
    leaderboard_n = 500
    p1, p2 = 0.955, 0.950
    aggregate = aggregate_difference_stats(p1, p2, leaderboard_n)
    deltas = (0.05, 0.02, 0.01, 0.005)

    paired = {"both_correct": 150, "first_only": 20, "second_only": 10, "both_wrong": 20}
    paired_total = sum(paired.values())
    paired |= {
        "total": paired_total,
        "first_accuracy": (paired["both_correct"] + paired["first_only"]) / paired_total,
        "second_accuracy": (paired["both_correct"] + paired["second_only"]) / paired_total,
        "delta": (paired["first_only"] - paired["second_only"]) / paired_total,
        "mcnemar_p": mcnemar_exact_pvalue(paired["first_only"], paired["second_only"]),
    }

    pass_n, pass_c = 20, 4
    pass_ks = (1, 5, 10)

    judge_pairs = 1_000
    judge_bias = 0.15
    p_first = 0.5 + judge_bias
    fwd_a = round(judge_pairs * p_first)
    swapped_a = round(judge_pairs * (1.0 - p_first))

    edge, dip, threshold = 0.95, 0.35, 0.90
    share = working_context_share(edge, dip, threshold)

    cost = EvalCostScenario(
        parameter_count=8_030_000_000,
        examples=700,
        prompt_tokens=400,
        completion_tokens=150,
        effective_flops_per_second=990e12 * 0.5,
    )

    return {
        "schema_version": 1,
        "leaderboard": {
            "sample_size": leaderboard_n,
            "first_rate": p1,
            "second_rate": p2,
            "aggregate_approximation": aggregate,
            "required_tasks": {str(delta): required_independent_tasks(delta) for delta in deltas},
            "paired_demo": paired,
        },
        "pass_at_k": {
            "total_samples": pass_n,
            "correct_samples": pass_c,
            "unbiased": {str(k): pass_at_k_unbiased(pass_n, pass_c, k) for k in pass_ks},
            "naive": {str(k): pass_at_k_naive(pass_n, pass_c, k) for k in pass_ks},
        },
        "judge": {
            "pairs": judge_pairs,
            "position_bias": judge_bias,
            "forward_a_wins": fwd_a,
            "swapped_a_wins": swapped_a,
            "averaged_a_rate": (fwd_a + swapped_a) / (2.0 * judge_pairs),
            "judges": 3,
            "orders": 2,
            "calls_per_pair": 6,
        },
        "lost_in_middle": {
            "edge": edge,
            "dip": dip,
            "threshold": threshold,
            "mean": edge - dip / 2.0,
            "working_share": share,
            "working_tokens_of_1m": round(share * 1_000_000),
        },
        "golden": {
            "expected_rate": 0.8,
            "n_for_3pp": golden_sample_size(0.03),
            "n_for_5pp": golden_sample_size(0.05),
            "canary_hex_characters": 32,
            "canary_guess_probability": canary_guess_probability(32),
            "cost": asdict(cost) | {
                "total_tokens": cost.total_tokens(),
                "dense_flops_floor": cost.dense_flops_floor(),
                "seconds_floor": cost.seconds_floor(),
            },
        },
    }


def write_reference(path: Path = REFERENCE_PATH) -> dict[str, Any]:
    """Write and return the canonical reference payload."""
    payload = build_reference_payload()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    """Print the contract or update it only with an explicit ``--write`` flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="rewrite data/reference_metrics.json")
    args = parser.parse_args()
    payload = write_reference() if args.write else build_reference_payload()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.write:
        print(f"wrote {REFERENCE_PATH}")


if __name__ == "__main__":
    main()
