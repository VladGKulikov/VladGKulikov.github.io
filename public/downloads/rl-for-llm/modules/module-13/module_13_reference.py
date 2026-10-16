#!/usr/bin/env python3
"""Deterministic reference calculations for RL for LLM, Module 13.

The module treats evaluation as a measurement system and a release gate rather
than as a final leaderboard table.  The functions below deliberately keep the
statistical unit, the scorer contract, and the decision rule explicit.

Only NumPy and the Python standard library are required.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, comb, erf, exp, isfinite, log, sqrt
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


@dataclass(frozen=True)
class PairedTable:
    both_correct: int
    a_only: int
    b_only: int
    both_wrong: int


@dataclass(frozen=True)
class JudgeAudit:
    items: int
    consistent: int
    inconsistent: int
    ties_both_orders: int
    consistency_rate: float
    candidate_a_rate: float
    candidate_b_rate: float


@dataclass(frozen=True)
class PositionAudit:
    judgments: int
    first_position_wins: int
    second_position_wins: int
    ties: int
    first_position_rate_non_ties: float
    first_position_rate_tie_half: float


@dataclass(frozen=True)
class SafetyRates:
    harmful_attempts: int
    benign_prompts: int
    attack_success_rate: float
    benign_refusal_rate: float
    severity_weighted_attack_rate: float

    @property
    def over_refusal_rate(self) -> float:
        """Alias that avoids treating every benign refusal as a classifier FPR."""
        return self.benign_refusal_rate


@dataclass(frozen=True)
class ReleaseDecision:
    passed: bool
    failures: tuple[str, ...]


def _count(successes: int, total: int) -> tuple[int, int]:
    if isinstance(successes, bool) or isinstance(total, bool):
        raise ValueError("counts must be integers")
    if not isinstance(successes, int) or not isinstance(total, int):
        raise ValueError("counts must be integers")
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("require total > 0 and 0 <= successes <= total")
    return successes, total


def _finite_vector(values: Sequence[float], name: str, *, nonempty: bool = True) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or (nonempty and arr.size == 0):
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain finite values")
    return arr


def normal_interval(successes: int, total: int, z: float = 1.96) -> Interval:
    """Unclipped Wald interval, retained only as a cautionary baseline."""
    successes, total = _count(successes, total)
    z = float(z)
    if not isfinite(z) or z <= 0:
        raise ValueError("z must be positive and finite")
    p = successes / total
    half = z * sqrt(p * (1.0 - p) / total)
    return Interval(p, p - half, p + half)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Interval:
    """Wilson score interval for a binomial proportion."""
    successes, total = _count(successes, total)
    z = float(z)
    if not isfinite(z) or z <= 0:
        raise ValueError("z must be positive and finite")
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    half = z * sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total)) / denom
    return Interval(p, max(0.0, center - half), min(1.0, center + half))


def worst_case_binomial_sample_size(margin: float, z: float = 1.96) -> int:
    """Worst-case normal-approximation planning value at p=1/2."""
    margin = float(margin)
    z = float(z)
    if not isfinite(margin) or not 0.0 < margin < 1.0:
        raise ValueError("margin must lie in (0, 1)")
    if not isfinite(z) or z <= 0:
        raise ValueError("z must be positive and finite")
    return ceil((z * z) / (4.0 * margin * margin))


def paired_table(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> PairedTable:
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    if a.ndim != 1 or b.ndim != 1 or a.size == 0 or a.shape != b.shape:
        raise ValueError("paired outcomes must be non-empty one-dimensional arrays of equal length")
    return PairedTable(
        both_correct=int(np.sum(a & b)),
        a_only=int(np.sum(a & ~b)),
        b_only=int(np.sum(~a & b)),
        both_wrong=int(np.sum(~a & ~b)),
    )


def paired_accuracy_difference(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> float:
    """Mean of the item-level differences A-B."""
    a = _finite_vector(a_correct, "a_correct")
    b = _finite_vector(b_correct, "b_correct")
    if a.shape != b.shape:
        raise ValueError("paired outcomes must have equal shape")
    return float(np.mean(a - b))


def mcnemar_z(a_only: int, b_only: int, *, continuity: bool = False) -> float:
    """Large-sample McNemar z; prefer the exact test for little discordance."""
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in (a_only, b_only)):
        raise ValueError("discordant counts must be non-negative integers")
    n = a_only + b_only
    if n == 0:
        return 0.0
    numerator = abs(a_only - b_only)
    if continuity:
        numerator = max(0, numerator - 1)
    return numerator / sqrt(n)


def exact_mcnemar_pvalue(a_only: int, b_only: int) -> float:
    """Two-sided exact binomial p-value under equal discordant probabilities."""
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in (a_only, b_only)):
        raise ValueError("discordant counts must be non-negative integers")
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def normal_two_sided_pvalue(z: float) -> float:
    z = abs(float(z))
    if not isfinite(z):
        raise ValueError("z must be finite")
    return 1.0 - erf(z / sqrt(2.0))


def paired_bootstrap_interval(
    a_scores: Sequence[float],
    b_scores: Sequence[float],
    *,
    confidence: float = 0.95,
    replicates: int = 20_000,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap of a paired mean difference, resampling task pairs."""
    a = _finite_vector(a_scores, "a_scores")
    b = _finite_vector(b_scores, "b_scores")
    if a.shape != b.shape:
        raise ValueError("a_scores and b_scores must have equal shape")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 100:
        raise ValueError("replicates must be an integer >= 100")
    rng = np.random.default_rng(seed)
    diff = a - b
    indices = rng.integers(0, diff.size, size=(replicates, diff.size))
    estimates = diff[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return Interval(float(diff.mean()), float(low), float(high))


def hierarchical_bootstrap_interval(
    a_scores: Sequence[Sequence[float]],
    b_scores: Sequence[Sequence[float]],
    *,
    confidence: float = 0.95,
    replicates: int = 10_000,
    seed: int = 0,
    resample_within_prompt: bool = False,
) -> Interval:
    """Prompt-cluster bootstrap for repeated completions.

    The primary unit is the prompt.  If ``resample_within_prompt`` is true,
    completions are resampled inside every selected prompt as a second stage.
    """
    a = np.asarray(a_scores, dtype=float)
    b = np.asarray(b_scores, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape or a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError("scores must be equal non-empty two-dimensional arrays")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("scores must be finite")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 100:
        raise ValueError("replicates must be an integer >= 100")
    rng = np.random.default_rng(seed)
    n_prompts, n_completions = a.shape
    prompt_indices = rng.integers(0, n_prompts, size=(replicates, n_prompts))
    estimates = np.empty(replicates, dtype=float)
    diff = a - b
    for r in range(replicates):
        selected = diff[prompt_indices[r]]
        if resample_within_prompt:
            col_indices = rng.integers(0, n_completions, size=(n_prompts, n_completions))
            selected = np.take_along_axis(selected, col_indices, axis=1)
        estimates[r] = selected.mean(axis=1).mean()
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return Interval(float(diff.mean(axis=1).mean()), float(low), float(high))


def bonferroni_alpha(alpha: float, comparisons: int) -> float:
    alpha = float(alpha)
    if not isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if isinstance(comparisons, bool) or not isinstance(comparisons, int) or comparisons <= 0:
        raise ValueError("comparisons must be a positive integer")
    return alpha / comparisons


def pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Finite-pool HumanEval estimator 1-C(n-c,k)/C(n,k)."""
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (n, c, k)):
        raise ValueError("n, c, and k must be integers")
    if n <= 0 or not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("require n > 0, 0 <= c <= n, and 1 <= k <= n")
    if n - c < k:
        return 1.0
    failure = 1.0
    for j in range(k):
        failure *= (n - c - j) / (n - j)
    return 1.0 - failure


def pass_at_k_curve(n: int, c: int, ks: Sequence[int]) -> np.ndarray:
    return np.asarray([pass_at_k_unbiased(n, c, int(k)) for k in ks], dtype=float)


def independent_trial_success(probability: float, attempts: int) -> float:
    """Probability of at least one success in IID Bernoulli trials."""
    p = float(probability)
    if not 0.0 <= p <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts <= 0:
        raise ValueError("attempts must be a positive integer")
    return 1.0 - (1.0 - p) ** attempts


def selection_success(correct: Sequence[bool], selected_index: int) -> bool:
    values = np.asarray(correct, dtype=bool)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("correct must be a non-empty one-dimensional sequence")
    if isinstance(selected_index, bool) or not isinstance(selected_index, int) or not 0 <= selected_index < values.size:
        raise ValueError("selected_index is out of range")
    return bool(values[selected_index])


def ngram_set(text: str, n: int = 8) -> set[tuple[str, ...]]:
    """Lower-cased whitespace word n-grams used by the inexpensive detector."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    tokens = text.lower().split()
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def ngram_containment(benchmark_text: str, corpus_text: str, n: int = 8) -> float:
    """Fraction of benchmark n-grams contained in one corpus document.

    This asymmetric quantity matches the formula used in the lecture.  It is a
    literal-overlap screen, not evidence that a benchmark is uncontaminated.
    """
    benchmark = ngram_set(benchmark_text, n)
    corpus = ngram_set(corpus_text, n)
    if not benchmark:
        return 0.0
    return len(benchmark & corpus) / len(benchmark)


def ngram_jaccard(left: str, right: str, n: int = 8) -> float:
    """Symmetric Jaccard overlap, retained as a separate optional diagnostic."""
    a = ngram_set(left, n)
    b = ngram_set(right, n)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def ngram_overlap(left: str, right: str, n: int = 8) -> float:
    """Backward-compatible name for benchmark-to-corpus containment."""
    return ngram_containment(left, right, n)


def contamination_report(
    benchmark_items: Sequence[str],
    corpus_documents: Sequence[str],
    *,
    n: int = 8,
    threshold: float = 0.2,
) -> dict[str, Any]:
    """Return the strongest literal-containment match for every benchmark item."""
    if not benchmark_items or not corpus_documents:
        raise ValueError("benchmark_items and corpus_documents must be non-empty")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    maxima: list[float] = []
    matches: list[tuple[int, int, float]] = []
    for i, item in enumerate(benchmark_items):
        scores = [ngram_containment(item, doc, n) for doc in corpus_documents]
        best_j = int(np.argmax(scores))
        best = float(scores[best_j])
        maxima.append(best)
        if best >= threshold:
            matches.append((i, best_j, best))
    return {
        "metric": "benchmark_ngram_containment",
        "n": int(n),
        "threshold": float(threshold),
        "max_overlap": tuple(maxima),
        "matches": tuple(matches),
        "flagged_fraction": len(matches) / len(benchmark_items),
    }


def judge_swap_audit(forward: Sequence[str], reverse_canonical: Sequence[str]) -> JudgeAudit:
    """Audit consistency after reverse-order labels are mapped back to A/B/TIE."""
    f = [str(x).upper() for x in forward]
    r = [str(x).upper() for x in reverse_canonical]
    if not f or len(f) != len(r):
        raise ValueError("forward and reverse_canonical must be non-empty and equally long")
    allowed = {"A", "B", "TIE"}
    if any(x not in allowed for x in f + r):
        raise ValueError("judge labels must be A, B, or TIE")
    consistent = sum(x == y for x, y in zip(f, r))
    a_votes = sum(x == "A" for x in f + r)
    b_votes = sum(x == "B" for x in f + r)
    total_non_tie = a_votes + b_votes
    return JudgeAudit(
        items=len(f),
        consistent=consistent,
        inconsistent=len(f) - consistent,
        ties_both_orders=sum(x == y == "TIE" for x, y in zip(f, r)),
        consistency_rate=consistent / len(f),
        candidate_a_rate=(a_votes / total_non_tie) if total_non_tie else 0.0,
        candidate_b_rate=(b_votes / total_non_tie) if total_non_tie else 0.0,
    )


def position_audit(raw_forward: Sequence[str], raw_reverse: Sequence[str]) -> PositionAudit:
    """Measure preference for the first screen position before canonical remapping.

    ``raw_forward`` and ``raw_reverse`` contain FIRST/SECOND/TIE labels.  Keeping
    this diagnostic separate from candidate identity prevents position bias
    from being confused with a stable preference for candidate A or B.
    """
    f = [str(x).upper() for x in raw_forward]
    r = [str(x).upper() for x in raw_reverse]
    if not f or len(f) != len(r):
        raise ValueError("raw order labels must be non-empty and equally long")
    allowed = {"FIRST", "SECOND", "TIE"}
    if any(x not in allowed for x in f + r):
        raise ValueError("position labels must be FIRST, SECOND, or TIE")
    labels = f + r
    first = sum(x == "FIRST" for x in labels)
    second = sum(x == "SECOND" for x in labels)
    ties = sum(x == "TIE" for x in labels)
    non_ties = first + second
    return PositionAudit(
        judgments=len(labels),
        first_position_wins=first,
        second_position_wins=second,
        ties=ties,
        first_position_rate_non_ties=(first / non_ties) if non_ties else 0.0,
        first_position_rate_tie_half=(first + 0.5 * ties) / len(labels),
    )


def first_position_win_rate(
    first_won: Sequence[bool],
    *,
    include_ties: bool = False,
    ties: Sequence[bool] | None = None,
) -> float:
    """Compact helper retained for small exercises."""
    first = np.asarray(first_won, dtype=bool)
    if first.ndim != 1 or first.size == 0:
        raise ValueError("first_won must be a non-empty one-dimensional sequence")
    if ties is None:
        return float(first.mean())
    tie_arr = np.asarray(ties, dtype=bool)
    if tie_arr.shape != first.shape:
        raise ValueError("ties must match first_won")
    if include_ties:
        return float(np.mean(first.astype(float) + 0.5 * tie_arr.astype(float)))
    mask = ~tie_arr
    if not np.any(mask):
        return 0.0
    return float(first[mask].mean())


def sycophancy_metrics(
    correct_without_user_claim: Sequence[bool],
    correct_with_false_user_claim: Sequence[bool],
    agrees_with_false_claim: Sequence[bool],
) -> dict[str, float]:
    base = np.asarray(correct_without_user_claim, dtype=bool)
    conditioned = np.asarray(correct_with_false_user_claim, dtype=bool)
    agree = np.asarray(agrees_with_false_claim, dtype=bool)
    if base.ndim != 1 or base.size == 0 or base.shape != conditioned.shape or base.shape != agree.shape:
        raise ValueError("all sycophancy arrays must be non-empty, one-dimensional, and equally shaped")
    eligible = base
    flip = eligible & ~conditioned
    return {
        "baseline_accuracy": float(base.mean()),
        "conditioned_accuracy": float(conditioned.mean()),
        "accuracy_drop": float(base.mean() - conditioned.mean()),
        "flip_rate_among_baseline_correct": float(flip.sum() / eligible.sum()) if eligible.any() else 0.0,
        "unsupported_agreement_rate": float(agree.mean()),
    }


def selection_optimism_experiment(
    *,
    true_probability: float = 0.6,
    items: int = 100,
    configurations: int = 40,
    trials: int = 20_000,
    seed: int = 13,
) -> dict[str, float]:
    """Choose the best noisy configuration when every configuration is equal."""
    p = float(true_probability)
    if not 0.0 <= p <= 1.0:
        raise ValueError("true_probability must lie in [0, 1]")
    for name, value in (("items", items), ("configurations", configurations), ("trials", trials)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    rng = np.random.default_rng(seed)
    observed = rng.binomial(items, p, size=(trials, configurations)) / items
    maxima = observed.max(axis=1)
    return {
        "mean_best_observed": float(maxima.mean()),
        "mean_optimism": float(maxima.mean() - p),
        "median_best_observed": float(np.median(maxima)),
        "q95_best_observed": float(np.quantile(maxima, 0.95)),
    }


def reward_pressure_experiment(
    candidate_counts: Sequence[int] = (1, 4, 16, 64, 256),
    *,
    trials: int = 8_000,
    seed: int = 17,
) -> dict[int, tuple[float, float]]:
    """Proxy rises while an independent goal can peak and then decline.

    This is a mechanism demonstration, not a claim about a particular LLM.
    """
    counts = [int(x) for x in candidate_counts]
    if not counts or any(x <= 0 for x in counts):
        raise ValueError("candidate_counts must contain positive integers")
    if isinstance(trials, bool) or not isinstance(trials, int) or trials <= 0:
        raise ValueError("trials must be a positive integer")
    rng = np.random.default_rng(seed)
    result: dict[int, tuple[float, float]] = {}
    for n in counts:
        useful = rng.normal(size=(trials, n))
        hack = rng.normal(size=(trials, n))
        proxy = useful + hack
        goal = useful - np.maximum(hack - 1.0, 0.0) ** 2
        chosen = np.argmax(proxy, axis=1)
        rows = np.arange(trials)
        result[n] = (float(proxy[rows, chosen].mean()), float(goal[rows, chosen].mean()))
    return result


def safety_rates(
    attack_succeeded: Sequence[bool],
    benign_refused: Sequence[bool],
    *,
    severities: Sequence[float] | None = None,
) -> SafetyRates:
    """Keep harmful-compliance and benign-over-refusal denominators separate."""
    attack = np.asarray(attack_succeeded, dtype=bool)
    benign = np.asarray(benign_refused, dtype=bool)
    if attack.ndim != 1 or benign.ndim != 1 or attack.size == 0 or benign.size == 0:
        raise ValueError("attack_succeeded and benign_refused must be non-empty one-dimensional sequences")
    if severities is None:
        sev = np.ones(attack.size, dtype=float)
    else:
        sev = _finite_vector(severities, "severities")
        if sev.shape != attack.shape or np.any(sev < 0.0):
            raise ValueError("severities must match attack_succeeded and be non-negative")
    denom = float(sev.sum())
    weighted = float(np.sum(sev * attack) / denom) if denom > 0 else 0.0
    return SafetyRates(
        harmful_attempts=int(attack.size),
        benign_prompts=int(benign.size),
        attack_success_rate=float(attack.mean()),
        benign_refusal_rate=float(benign.mean()),
        severity_weighted_attack_rate=weighted,
    )


def stratified_binary_rates(outcomes: Sequence[bool], groups: Sequence[str]) -> dict[str, Interval]:
    """Wilson intervals by named stratum, useful for capability and safety slices."""
    y = np.asarray(outcomes, dtype=bool)
    g = np.asarray(groups, dtype=object)
    if y.ndim != 1 or g.ndim != 1 or y.size == 0 or y.shape != g.shape:
        raise ValueError("outcomes and groups must be equal non-empty vectors")
    result: dict[str, Interval] = {}
    for name in sorted({str(x) for x in g}):
        mask = g.astype(str) == name
        result[name] = wilson_interval(int(y[mask].sum()), int(mask.sum()))
    return result


def release_decision(
    metrics: Mapping[str, float],
    thresholds: Mapping[str, tuple[str, float]],
    *,
    critical_failures: Sequence[str] = (),
) -> ReleaseDecision:
    """Apply predeclared min/max gates plus non-negotiable critical failures."""
    failures = [f"critical:{item}" for item in critical_failures]
    for name, (direction, threshold) in thresholds.items():
        if name not in metrics:
            failures.append(f"missing:{name}")
            continue
        value = float(metrics[name])
        threshold = float(threshold)
        if not isfinite(value) or not isfinite(threshold):
            failures.append(f"nonfinite:{name}")
        elif direction == "max" and value > threshold:
            failures.append(f"{name}>{threshold:g}")
        elif direction == "min" and value < threshold:
            failures.append(f"{name}<{threshold:g}")
        elif direction not in {"min", "max"}:
            failures.append(f"invalid_direction:{name}")
    return ReleaseDecision(not failures, tuple(failures))


REQUIRED_PASSPORT_FIELDS = (
    "run_id",
    "model_id",
    "model_revision",
    "tokenizer_id",
    "chat_template_hash",
    "dataset_id",
    "dataset_revision",
    "split",
    "solver_id",
    "scorer_id",
    "scorer_revision",
    "decoding",
    "seeds",
    "sample_count",
    "metrics",
    "cost",
    "artifact_hashes",
)


def validate_measurement_passport(passport: Mapping[str, Any]) -> tuple[str, ...]:
    """Return all schema errors for an evaluation passport."""
    errors: list[str] = []
    for field in REQUIRED_PASSPORT_FIELDS:
        if field not in passport:
            errors.append(f"missing:{field}")
    if "run_id" in passport and (not isinstance(passport["run_id"], str) or not passport["run_id"]):
        errors.append("invalid:run_id")
    if "sample_count" in passport and (
        isinstance(passport["sample_count"], bool)
        or not isinstance(passport["sample_count"], int)
        or passport["sample_count"] <= 0
    ):
        errors.append("invalid:sample_count")
    if "seeds" in passport and (
        not isinstance(passport["seeds"], Sequence)
        or isinstance(passport["seeds"], (str, bytes))
        or len(passport["seeds"]) == 0
    ):
        errors.append("invalid:seeds")
    for field in ("decoding", "metrics", "cost", "artifact_hashes"):
        if field in passport and not isinstance(passport[field], Mapping):
            errors.append(f"invalid:{field}")
    return tuple(errors)
