#!/usr/bin/env python3
"""Deterministic integration tests for the Module 13 evaluation harness."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import module_13_evaluation_harness as h  # noqa: E402


passed = 0


def check(condition: bool, label: str) -> None:
    global passed
    if not condition:
        raise AssertionError(label)
    passed += 1
    print(f"[PASS] {label}")


samples = h.build_synthetic_dataset(80, seed=13)
check(len(samples) == 80 and len({s.sample_id for s in samples}) == 80, "stable unique sample IDs")
check({s.category for s in samples} == {"arithmetic", "logic", "tool_use", "instruction_following"}, "four capability strata")
base, cand = h.default_systems()
ra = h.evaluate_system(samples, base, seed=13)
rb = h.evaluate_system(samples, cand, seed=13)
check(len(ra) == len(rb) == 80, "both systems evaluate the same dataset")
check([r.sample.sample_id for r in ra] == [r.sample.sample_id for r in rb], "paired record order is preserved")
sa = h.summarize_records(ra, base)
sb = h.summarize_records(rb, cand)
check(0.0 <= sa.accuracy_low <= sa.accuracy <= sa.accuracy_high <= 1.0, "baseline Wilson interval")
check(0.0 <= sb.technical_failure_rate <= 1.0 and sb.mean_tokens >= 0.0, "candidate operational metrics")
paired = h.paired_comparison(ra, rb, seed=7)
check(abs(paired["candidate_minus_baseline"] - (sb.accuracy - sa.accuracy)) < 1e-12, "paired difference matches marginal difference")
check(paired["bootstrap_low"] <= paired["candidate_minus_baseline"] <= paired["bootstrap_high"], "paired interval contains estimate")
check(paired["candidate_only"] + paired["baseline_only"] + paired["both_correct"] + paired["both_wrong"] == 80, "paired table partitions samples")

judge = h.synthetic_judge_audit(items=100, seed=13)
check(judge["identity"]["items"] == 100, "judge identity audit size")
check(judge["position"]["judgments"] == 200, "position audit uses both orders")
check(judge["position"]["first_position_rate_non_ties"] > 0.5, "synthetic judge exposes first-position bias")
safety = h.synthetic_safety_audit(seed=13)
check(safety["rates"]["harmful_attempts"] == 200 and safety["rates"]["benign_prompts"] == 200, "separate safety denominators")
check(0.0 <= safety["rates"]["severity_weighted_attack_rate"] <= 1.0, "severity-weighted rate range")

work = Path(tempfile.mkdtemp(prefix="m13-harness-test-"))
try:
    first = h.run_course_experiment(work / "run1", seed=13, n_tasks=120)
    second = h.run_course_experiment(work / "run2", seed=13, n_tasks=120)
    required = {
        "dataset.jsonl",
        "records_baseline.jsonl",
        "records_candidate.jsonl",
        "summary.json",
        "report.md",
        "judge_audit.json",
        "safety_audit.json",
        "contamination_audit.json",
        "passport.json",
    }
    check(required == {p.name for p in (work / "run1").iterdir()}, "evidence bundle has all required artifacts")
    check(first["decision"]["passed"] is True, "predeclared synthetic release gates pass")
    check(first["paired"]["candidate_minus_baseline"] > 0.10, "candidate improves paired capability")
    check(first["contamination"]["flagged_fraction"] == 1 / 3, "literal containment flags only the copied item")
    check(first["passport"]["artifact_hashes"] == second["passport"]["artifact_hashes"], "evidence artifacts are deterministic")
    passport = json.loads((work / "run1" / "passport.json").read_text(encoding="utf-8"))
    check(passport["scorer_revision"] == "exact-v2" and passport["sample_count"] == 120, "passport records scorer and sample count")
    report = (work / "run1" / "report.md").read_text(encoding="utf-8")
    check("not a benchmark result for a real LLM" in report, "report states the synthetic boundary")
    for name, digest in passport["artifact_hashes"].items():
        check(h.file_sha256(work / "run1" / name) == digest, f"artifact hash verifies: {name}")
finally:
    shutil.rmtree(work)

print(f"\nSUMMARY: {passed}/{passed} tests passed")
