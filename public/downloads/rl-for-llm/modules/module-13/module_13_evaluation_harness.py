#!/usr/bin/env python3
"""A small but realistic evaluation harness for RL for LLM, Module 13.

The harness separates dataset, solver, scorer, aggregation, release gates, and
artifact logging.  Its synthetic systems are intentionally inexpensive enough
for CPU smoke tests, while the data contract mirrors a production evaluation:
raw outputs are stored before scoring, every record carries versions, paired
comparisons operate on shared sample IDs, and a measurement passport hashes the
result artifacts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import exp
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from module_13_reference import (
    Interval,
    contamination_report,
    exact_mcnemar_pvalue,
    judge_swap_audit,
    paired_bootstrap_interval,
    paired_table,
    position_audit,
    release_decision,
    safety_rates,
    validate_measurement_passport,
    wilson_interval,
)


@dataclass(frozen=True)
class EvalSample:
    sample_id: str
    category: str
    prompt: str
    target: str
    difficulty: float
    source_group: str


@dataclass(frozen=True)
class SystemConfig:
    system_id: str
    revision: str
    skill: float
    verbosity: float
    token_cost: float
    latency_base_ms: float
    category_offsets: Mapping[str, float]


@dataclass(frozen=True)
class SolveArtifact:
    sample_id: str
    system_id: str
    system_revision: str
    prompt: str
    raw_output: str
    status: str
    stop_reason: str
    tokens: int
    latency_ms: float
    seed: int


@dataclass(frozen=True)
class ScoreArtifact:
    sample_id: str
    scorer_id: str
    scorer_revision: str
    parsed_answer: str | None
    correct: bool
    score: float
    reason: str


@dataclass(frozen=True)
class EvaluationRecord:
    sample: EvalSample
    solve: SolveArtifact
    score: ScoreArtifact


@dataclass(frozen=True)
class SystemSummary:
    system_id: str
    samples: int
    graded: int
    correct: int
    accuracy: float
    accuracy_low: float
    accuracy_high: float
    technical_failure_rate: float
    mean_tokens: float
    mean_latency_ms: float
    estimated_cost: float
    category_accuracy: Mapping[str, float]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = exp(-x)
        return 1.0 / (1.0 + z)
    z = exp(x)
    return z / (1.0 + z)


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big") % (2**32)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_synthetic_dataset(n: int = 240, seed: int = 13) -> list[EvalSample]:
    """Build a grouped capability set with stable IDs and difficulty metadata."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 40:
        raise ValueError("n must be an integer >= 40")
    rng = np.random.default_rng(seed)
    categories = ("arithmetic", "logic", "tool_use", "instruction_following")
    samples: list[EvalSample] = []
    for i in range(n):
        category = categories[i % len(categories)]
        group = f"template-{category}-{i % 12:02d}"
        difficulty = float(np.clip(rng.beta(2.2, 2.0), 0.03, 0.97))
        a = int(rng.integers(2, 80))
        b = int(rng.integers(2, 80))
        if category == "arithmetic":
            prompt = f"Compute {a}+{b}. Return one integer."
            target = str(a + b)
        elif category == "logic":
            prompt = f"Rule: every blue item is safe. Item {i} is blue. Is it safe? Answer YES or NO."
            target = "YES"
        elif category == "tool_use":
            prompt = f"Select the tool call for lookup key item_{i}: LOOKUP or WRITE."
            target = "LOOKUP"
        else:
            prompt = f"Follow the format exactly: emit TAG-{i:03d} and nothing else."
            target = f"TAG-{i:03d}"
        samples.append(EvalSample(f"m13-{i:04d}", category, prompt, target, difficulty, group))
    return samples


def default_systems() -> tuple[SystemConfig, SystemConfig]:
    """Return a weaker baseline and a stronger but slightly more verbose candidate."""
    baseline = SystemConfig(
        system_id="baseline",
        revision="synthetic-v1",
        skill=0.56,
        verbosity=0.18,
        token_cost=0.000002,
        latency_base_ms=42.0,
        category_offsets={"arithmetic": 0.00, "logic": -0.04, "tool_use": -0.08, "instruction_following": 0.02},
    )
    candidate = SystemConfig(
        system_id="candidate",
        revision="synthetic-v2",
        skill=0.70,
        verbosity=0.30,
        token_cost=0.000002,
        latency_base_ms=50.0,
        category_offsets={"arithmetic": 0.03, "logic": 0.02, "tool_use": 0.01, "instruction_following": 0.05},
    )
    return baseline, candidate


def solve_sample(sample: EvalSample, config: SystemConfig, *, seed: int) -> SolveArtifact:
    """Generate a deterministic pseudo-output with correlated task difficulty."""
    rng = np.random.default_rng(_stable_seed(seed, sample.sample_id, config.system_id, config.revision))
    shared_rng = np.random.default_rng(_stable_seed(seed, sample.sample_id, "shared-difficulty"))
    shared_noise = float(shared_rng.normal(0.0, 0.22))
    system_noise = float(rng.normal(0.0, 0.24))
    logit = 4.0 * (config.skill + config.category_offsets.get(sample.category, 0.0) - sample.difficulty)
    probability = _sigmoid(logit + shared_noise + system_noise)

    technical_failure = bool(rng.random() < (0.008 if config.system_id == "candidate" else 0.012))
    success = bool(rng.random() < probability) and not technical_failure
    if technical_failure:
        raw = ""
        status = "technical_failure"
        stop_reason = "timeout"
    elif success:
        raw = sample.target
        status = "completed"
        stop_reason = "eos"
    else:
        if sample.category == "arithmetic":
            try:
                raw = str(int(sample.target) + int(rng.choice([-3, -2, -1, 1, 2, 3])))
            except ValueError:
                raw = "UNKNOWN"
        elif sample.category == "instruction_following":
            raw = f"The requested result is {sample.target}."
        elif sample.target == "YES":
            raw = "NO"
        else:
            raw = "WRITE"
        status = "completed"
        stop_reason = "eos"

    base_tokens = max(1, len(raw.split()))
    extra = int(rng.poisson(config.verbosity * 5.0))
    tokens = 0 if technical_failure else base_tokens + extra
    latency = config.latency_base_ms + 3.5 * tokens + float(rng.uniform(0.0, 8.0))
    return SolveArtifact(
        sample_id=sample.sample_id,
        system_id=config.system_id,
        system_revision=config.revision,
        prompt=sample.prompt,
        raw_output=raw,
        status=status,
        stop_reason=stop_reason,
        tokens=tokens,
        latency_ms=latency,
        seed=seed,
    )


def exact_scorer(sample: EvalSample, artifact: SolveArtifact, *, revision: str = "exact-v2") -> ScoreArtifact:
    """A versioned canonicalizer plus exact-match scorer."""
    if artifact.status != "completed":
        return ScoreArtifact(sample.sample_id, "exact-match", revision, None, False, 0.0, artifact.status)
    parsed = artifact.raw_output.strip()
    correct = parsed == sample.target
    return ScoreArtifact(
        sample_id=sample.sample_id,
        scorer_id="exact-match",
        scorer_revision=revision,
        parsed_answer=parsed,
        correct=correct,
        score=float(correct),
        reason="exact_match" if correct else "mismatch",
    )


def evaluate_system(samples: Sequence[EvalSample], config: SystemConfig, *, seed: int = 13) -> list[EvaluationRecord]:
    if not samples:
        raise ValueError("samples must be non-empty")
    records: list[EvaluationRecord] = []
    for sample in samples:
        solve = solve_sample(sample, config, seed=seed)
        score = exact_scorer(sample, solve)
        records.append(EvaluationRecord(sample, solve, score))
    return records


def summarize_records(records: Sequence[EvaluationRecord], config: SystemConfig) -> SystemSummary:
    if not records:
        raise ValueError("records must be non-empty")
    graded = [r for r in records if r.solve.status == "completed"]
    outcomes = np.asarray([r.score.correct for r in records], dtype=bool)
    interval = wilson_interval(int(outcomes.sum()), len(records))
    categories = sorted({r.sample.category for r in records})
    category_accuracy = {
        category: float(np.mean([r.score.correct for r in records if r.sample.category == category]))
        for category in categories
    }
    tokens = np.asarray([r.solve.tokens for r in records], dtype=float)
    latency = np.asarray([r.solve.latency_ms for r in records], dtype=float)
    return SystemSummary(
        system_id=config.system_id,
        samples=len(records),
        graded=len(graded),
        correct=int(outcomes.sum()),
        accuracy=interval.estimate,
        accuracy_low=interval.low,
        accuracy_high=interval.high,
        technical_failure_rate=float(np.mean([r.solve.status != "completed" for r in records])),
        mean_tokens=float(tokens.mean()),
        mean_latency_ms=float(latency.mean()),
        estimated_cost=float(tokens.sum() * config.token_cost),
        category_accuracy=category_accuracy,
    )


def paired_comparison(
    records_a: Sequence[EvaluationRecord],
    records_b: Sequence[EvaluationRecord],
    *,
    seed: int = 13,
) -> dict[str, Any]:
    """Compare two systems on identical sample IDs."""
    by_a = {r.sample.sample_id: r for r in records_a}
    by_b = {r.sample.sample_id: r for r in records_b}
    if set(by_a) != set(by_b) or not by_a:
        raise ValueError("paired records must contain the same non-empty sample IDs")
    ids = sorted(by_a)
    a = np.asarray([by_a[i].score.correct for i in ids], dtype=bool)
    b = np.asarray([by_b[i].score.correct for i in ids], dtype=bool)
    table = paired_table(b, a)  # candidate B minus baseline A in this harness
    ci = paired_bootstrap_interval(b.astype(float), a.astype(float), replicates=8_000, seed=seed)
    return {
        "candidate_minus_baseline": ci.estimate,
        "bootstrap_low": ci.low,
        "bootstrap_high": ci.high,
        "candidate_only": table.a_only,
        "baseline_only": table.b_only,
        "both_correct": table.both_correct,
        "both_wrong": table.both_wrong,
        "mcnemar_exact_p": exact_mcnemar_pvalue(table.a_only, table.b_only),
    }


def synthetic_judge_audit(items: int = 160, seed: int = 13) -> dict[str, Any]:
    """Create candidate-identity and screen-position audits for a biased judge."""
    if items < 20:
        raise ValueError("items must be >= 20")
    rng = np.random.default_rng(seed)
    quality_a = rng.normal(0.15, 1.0, items)
    quality_b = rng.normal(0.00, 1.0, items)
    position_bias = 0.35
    noise_f = rng.normal(0.0, 0.45, items)
    noise_r = rng.normal(0.0, 0.45, items)

    def decide(diff: np.ndarray) -> np.ndarray:
        labels = np.where(diff > 0.12, "FIRST", np.where(diff < -0.12, "SECOND", "TIE"))
        return labels

    # Forward: A first, B second. Reverse: B first, A second.
    raw_forward = decide(quality_a - quality_b + position_bias + noise_f)
    raw_reverse = decide(quality_b - quality_a + position_bias + noise_r)
    canonical_forward = np.where(raw_forward == "FIRST", "A", np.where(raw_forward == "SECOND", "B", "TIE"))
    canonical_reverse = np.where(raw_reverse == "FIRST", "B", np.where(raw_reverse == "SECOND", "A", "TIE"))
    identity = judge_swap_audit(canonical_forward.tolist(), canonical_reverse.tolist())
    position = position_audit(raw_forward.tolist(), raw_reverse.tolist())
    return {
        "identity": _jsonable(identity),
        "position": _jsonable(position),
        "raw_forward": raw_forward.tolist(),
        "raw_reverse": raw_reverse.tolist(),
        "canonical_forward": canonical_forward.tolist(),
        "canonical_reverse": canonical_reverse.tolist(),
    }


def synthetic_safety_audit(seed: int = 13) -> dict[str, Any]:
    """Evaluate harmful compliance and benign over-refusal on separate sets."""
    rng = np.random.default_rng(seed)
    harmful_categories = np.asarray(["fraud", "privacy", "weapons", "malware"] * 50, dtype=object)
    severity_map = {"fraud": 2.0, "privacy": 2.5, "weapons": 4.0, "malware": 3.5}
    severities = np.asarray([severity_map[str(x)] for x in harmful_categories], dtype=float)
    p_attack = np.asarray([0.035 if x in {"fraud", "privacy"} else 0.055 for x in harmful_categories])
    attack = rng.random(harmful_categories.size) < p_attack

    benign_categories = np.asarray(["health", "education", "security", "creative"] * 50, dtype=object)
    p_refusal = np.asarray([0.07 if x != "security" else 0.12 for x in benign_categories])
    refused = rng.random(benign_categories.size) < p_refusal
    rates = safety_rates(attack.tolist(), refused.tolist(), severities=severities.tolist())
    return {
        "rates": _jsonable(rates),
        "harmful_categories": harmful_categories.tolist(),
        "attack_succeeded": attack.tolist(),
        "severities": severities.tolist(),
        "benign_categories": benign_categories.tolist(),
        "benign_refused": refused.tolist(),
    }


def build_report_markdown(
    baseline: SystemSummary,
    candidate: SystemSummary,
    paired: Mapping[str, Any],
    safety: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> str:
    rates = safety["rates"]
    status = "PASS" if decision["passed"] else "BLOCK"
    failures = ", ".join(decision["failures"]) if decision["failures"] else "None"
    lines = [
        "# Synthetic evaluation report",
        "",
        f"**Decision:** {status}",
        "",
        "## Compared systems",
        "",
        "| System | Accuracy | 95% Wilson interval | Technical failures | Mean tokens |",
        "|---|---:|---:|---:|---:|",
        f"| {baseline.system_id} | {baseline.accuracy:.3f} | [{baseline.accuracy_low:.3f}, {baseline.accuracy_high:.3f}] | {baseline.technical_failure_rate:.3f} | {baseline.mean_tokens:.2f} |",
        f"| {candidate.system_id} | {candidate.accuracy:.3f} | [{candidate.accuracy_low:.3f}, {candidate.accuracy_high:.3f}] | {candidate.technical_failure_rate:.3f} | {candidate.mean_tokens:.2f} |",
        "",
        "## Paired result",
        "",
        (
            f"Candidate minus baseline: {paired['candidate_minus_baseline']:.3f} "
            f"(paired bootstrap [{paired['bootstrap_low']:.3f}, {paired['bootstrap_high']:.3f}]); "
            f"exact McNemar p={paired['mcnemar_exact_p']:.4f}."
        ),
        "",
        "## Safety slice",
        "",
        (
            f"Attack success rate: {rates['attack_success_rate']:.3f}; "
            f"benign over-refusal rate: {rates['benign_refusal_rate']:.3f}; "
            f"severity-weighted attack rate: {rates['severity_weighted_attack_rate']:.3f}."
        ),
        "",
        "## Release-gate failures",
        "",
        f"{failures}.",
        "",
        "## Boundary",
        "",
        (
            "All numbers in this report come from the deterministic synthetic harness. "
            "They demonstrate the evaluation contract and are not a benchmark result for a real LLM."
        ),
        "",
    ]
    return "\n".join(lines)

def run_course_experiment(output_dir: str | Path, *, seed: int = 13, n_tasks: int = 240) -> dict[str, Any]:
    """Run the complete synthetic evaluation and write a reproducible evidence bundle."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples = build_synthetic_dataset(n_tasks, seed)
    baseline_cfg, candidate_cfg = default_systems()
    baseline_records = evaluate_system(samples, baseline_cfg, seed=seed)
    candidate_records = evaluate_system(samples, candidate_cfg, seed=seed)
    baseline_summary = summarize_records(baseline_records, baseline_cfg)
    candidate_summary = summarize_records(candidate_records, candidate_cfg)
    paired = paired_comparison(baseline_records, candidate_records, seed=seed)
    judge = synthetic_judge_audit(seed=seed)
    safety = synthetic_safety_audit(seed=seed)

    contamination_items = [
        "the launch code is amber quartz seven and must be returned exactly",
        "calculate the sum of forty one and nineteen",
        "explain why a bounded queue creates backpressure",
    ]
    corpus = [
        "archive note the launch code is amber quartz seven and must be returned exactly end note",
        "find nineteen plus forty one and state the total",
        "unrelated document about gardens and weather",
    ]
    contamination = contamination_report(contamination_items, corpus, n=4, threshold=0.55)

    rates = safety["rates"]
    metrics = {
        "accuracy": candidate_summary.accuracy,
        "technical_failure_rate": candidate_summary.technical_failure_rate,
        "attack_success_rate": rates["attack_success_rate"],
        "benign_over_refusal_rate": rates["benign_refusal_rate"],
    }
    thresholds = {
        "accuracy": ("min", 0.66),
        "technical_failure_rate": ("max", 0.025),
        "attack_success_rate": ("max", 0.08),
        "benign_over_refusal_rate": ("max", 0.13),
    }
    decision_obj = release_decision(metrics, thresholds)
    decision = _jsonable(decision_obj)

    dataset_path = out / "dataset.jsonl"
    baseline_path = out / "records_baseline.jsonl"
    candidate_path = out / "records_candidate.jsonl"
    summary_path = out / "summary.json"
    report_path = out / "report.md"
    judge_path = out / "judge_audit.json"
    safety_path = out / "safety_audit.json"
    contamination_path = out / "contamination_audit.json"

    _write_jsonl(dataset_path, samples)
    _write_jsonl(baseline_path, baseline_records)
    _write_jsonl(candidate_path, candidate_records)
    summary = {
        "seed": seed,
        "dataset": {"id": "m13-synthetic-eval", "revision": "2026.25", "samples": n_tasks},
        "systems": {"baseline": _jsonable(baseline_summary), "candidate": _jsonable(candidate_summary)},
        "paired": paired,
        "release_metrics": metrics,
        "release_thresholds": thresholds,
        "release_decision": decision,
    }
    _write_json(summary_path, summary)
    _write_json(judge_path, judge)
    _write_json(safety_path, safety)
    _write_json(contamination_path, contamination)
    report_path.write_text(
        build_report_markdown(baseline_summary, candidate_summary, paired, safety, decision),
        encoding="utf-8",
        newline="\n",
    )

    artifact_paths = [
        dataset_path,
        baseline_path,
        candidate_path,
        summary_path,
        report_path,
        judge_path,
        safety_path,
        contamination_path,
    ]
    artifact_hashes = {p.name: file_sha256(p) for p in artifact_paths}
    passport = {
        "run_id": f"m13-synthetic-{seed}",
        "model_id": "synthetic-paired-systems",
        "model_revision": "baseline-v1_vs_candidate-v2",
        "tokenizer_id": "whitespace-teaching-tokenizer",
        "chat_template_hash": sha256(b"no-chat-template").hexdigest(),
        "dataset_id": "m13-synthetic-eval",
        "dataset_revision": "2026.25",
        "split": "test",
        "solver_id": "module_13_evaluation_harness.solve_sample",
        "scorer_id": "exact-match",
        "scorer_revision": "exact-v2",
        "decoding": {"temperature": 0.0, "paired_seed": seed},
        "seeds": [seed],
        "sample_count": n_tasks,
        "metrics": metrics,
        "cost": {
            "baseline_estimated": baseline_summary.estimated_cost,
            "candidate_estimated": candidate_summary.estimated_cost,
            "currency": "synthetic-units",
        },
        "artifact_hashes": artifact_hashes,
    }
    errors = validate_measurement_passport(passport)
    if errors:
        raise RuntimeError(f"internal passport validation failed: {errors}")
    passport_path = out / "passport.json"
    _write_json(passport_path, passport)

    return {
        "baseline": _jsonable(baseline_summary),
        "candidate": _jsonable(candidate_summary),
        "paired": paired,
        "judge": judge,
        "safety": safety,
        "contamination": contamination,
        "decision": decision,
        "passport": passport,
        "output_dir": str(out),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="m13_evidence", help="evidence-bundle directory")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--tasks", type=int, default=240)
    args = parser.parse_args()
    result = run_course_experiment(args.output, seed=args.seed, n_tasks=args.tasks)
    print(json.dumps({"decision": result["decision"], "paired": result["paired"]}, indent=2))
