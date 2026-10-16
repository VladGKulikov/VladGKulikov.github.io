#!/usr/bin/env python3
"""Deterministic tests for Module 13 reference calculations."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import module_13_reference as m13  # noqa: E402


passed = 0


def check(condition: bool, label: str) -> None:
    global passed
    if not condition:
        raise AssertionError(label)
    passed += 1
    print(f"[PASS] {label}")


# Intervals and planning.
w = m13.wilson_interval(0, 10)
check(w.low == 0.0 and 0.27 < w.high < 0.29, "Wilson remains non-degenerate at zero successes")
n = m13.normal_interval(0, 10)
check(n.low == n.high == 0.0, "Wald baseline exposes the boundary failure")
w62 = m13.wilson_interval(62, 100)
check(abs(w62.estimate - 0.62) < 1e-12 and 0.52 < w62.low < 0.53 and 0.70 < w62.high < 0.72, "Wilson 62/100 example")
w020 = m13.wilson_interval(0, 20)
check(abs(w020.high - 0.16113012549493322) < 1e-12, "Wilson 0/20 example")
check(m13.worst_case_binomial_sample_size(0.03) == 1068, "worst-case 3pp planning")
check(m13.worst_case_binomial_sample_size(0.01) == 9604, "worst-case 1pp planning")

# Paired comparison.
A = [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1]
B = [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0]
t = m13.paired_table(A, B)
check((t.both_correct, t.a_only, t.b_only, t.both_wrong) == (4, 5, 2, 4), "paired contingency table")
check(abs(m13.paired_accuracy_difference(A, B) - 0.2) < 1e-12, "paired mean difference")
check(abs(m13.mcnemar_z(9, 2) - 7 / math.sqrt(11)) < 1e-12, "asymptotic McNemar z")
check(abs(m13.exact_mcnemar_pvalue(9, 2) - 0.0654296875) < 1e-15, "exact McNemar p-value")
check(m13.exact_mcnemar_pvalue(0, 0) == 1.0, "exact paired test handles no discordance")
check(0.03 < m13.normal_two_sided_pvalue(m13.mcnemar_z(9, 2)) < 0.04, "asymptotic p differs from exact p")
ci = m13.paired_bootstrap_interval(A, B, replicates=4000, seed=2)
check(abs(ci.estimate - 0.2) < 1e-12 and ci.low <= ci.estimate <= ci.high, "paired bootstrap resamples task pairs")

a2 = [[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]
b2 = [[1, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
h = m13.hierarchical_bootstrap_interval(a2, b2, replicates=3000, seed=3)
check(abs(h.estimate - 0.1875) < 1e-12 and h.low <= h.estimate <= h.high, "prompt-cluster bootstrap")
h2 = m13.hierarchical_bootstrap_interval(a2, b2, replicates=1000, seed=3, resample_within_prompt=True)
check(h2.low <= h2.estimate <= h2.high, "two-stage hierarchical bootstrap")
check(abs(m13.bonferroni_alpha(0.05, 20) - 0.0025) < 1e-15, "Bonferroni threshold")

# pass@k and selection.
check(abs(m13.pass_at_k_unbiased(200, 10, 5) - 0.22828446073733422) < 1e-15, "finite-pool pass@5")
check(m13.pass_at_k_unbiased(10, 0, 5) == 0.0, "pass@k zero successes")
check(m13.pass_at_k_unbiased(10, 8, 5) == 1.0, "pass@k guaranteed coverage")
curve = m13.pass_at_k_curve(20, 2, [1, 2, 5, 10])
check(np.all(np.diff(curve) >= 0), "pass@k curve is monotone")
check(abs(m13.independent_trial_success(0.05, 20) - 0.6415140775914581) < 1e-15, "IID at-least-one-success calculation")
check(m13.selection_success([False, True, False], 1) is True, "selection success is separate from coverage")

# Literal contamination diagnostics: the course formula is asymmetric containment.
text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
longer = "prefix words alpha beta gamma delta epsilon zeta eta theta iota kappa lambda suffix words"
check(m13.ngram_containment(text, longer, 4) == 1.0, "benchmark n-gram containment matches lecture formula")
check(m13.ngram_containment(longer, text, 4) < 1.0, "containment is intentionally asymmetric")
check(m13.ngram_jaccard(text, longer, 4) < 1.0, "Jaccard remains a separate symmetric diagnostic")
check(m13.ngram_overlap(text, longer, 4) == 1.0, "legacy overlap name maps to containment")
check(m13.ngram_containment("one two three", "four five six", 2) == 0.0, "disjoint n-gram containment")
report = m13.contamination_report([text, "fresh unrelated prompt"], [longer, "another clean document"], n=4, threshold=0.5)
check(report["metric"] == "benchmark_ngram_containment", "contamination report declares its metric")
check(report["flagged_fraction"] == 0.5 and len(report["matches"]) == 1, "contamination report flags literal match")
check(m13.ngram_set("too short", 3) == set(), "short text yields an empty n-gram set")

# Judge and sycophancy audits.
j = m13.judge_swap_audit(["A", "A", "B", "TIE"], ["A", "B", "B", "TIE"])
check(j.consistent == 3 and abs(j.consistency_rate - 0.75) < 1e-12, "candidate-identity swap consistency")
check(j.ties_both_orders == 1, "judge audit counts stable ties")
pos = m13.position_audit(["FIRST", "SECOND", "FIRST"], ["FIRST", "TIE", "SECOND"])
check(pos.judgments == 6 and pos.first_position_wins == 3 and pos.ties == 1, "position audit keeps screen position separate")
check(abs(pos.first_position_rate_non_ties - 0.6) < 1e-12, "non-tie first-position rate")
check(abs(m13.first_position_win_rate([True, False, True, False]) - 0.5) < 1e-12, "compact first-position rate")
check(abs(m13.first_position_win_rate([True, False], include_ties=True, ties=[False, True]) - 0.75) < 1e-12, "tie-half position rate")
syc = m13.sycophancy_metrics([1, 1, 1, 0], [1, 0, 0, 0], [0, 1, 1, 1])
check(abs(syc["flip_rate_among_baseline_correct"] - 2 / 3) < 1e-12, "counterfactual sycophancy flip rate")
check(abs(syc["unsupported_agreement_rate"] - 0.75) < 1e-12, "unsupported agreement rate")

# Adaptive selection and reward pressure.
opt = m13.selection_optimism_experiment(items=100, configurations=40, trials=4000, seed=13)
check(opt["mean_best_observed"] > 0.68 and opt["mean_optimism"] > 0.08, "selection on the test creates optimism")
pressure = m13.reward_pressure_experiment((1, 4, 16, 64, 256), trials=4000, seed=17)
check(all(pressure[a][0] < pressure[b][0] for a, b in zip((1, 4, 16, 64), (4, 16, 64, 256))), "proxy rises with selection pressure")
check(max(pressure, key=lambda count: pressure[count][1]) in {16, 64}, "independent goal peaks before maximum pressure")

# Safety and release gates.
rates = m13.safety_rates([0, 1, 0, 1], [0, 0, 1, 0], severities=[1, 4, 1, 2])
check(abs(rates.attack_success_rate - 0.5) < 1e-12, "attack success denominator")
check(abs(rates.benign_refusal_rate - 0.25) < 1e-12, "benign over-refusal denominator")
check(rates.over_refusal_rate == rates.benign_refusal_rate, "explicit over-refusal alias")
check(abs(rates.severity_weighted_attack_rate - 0.75) < 1e-12, "severity-weighted attack rate")
strata = m13.stratified_binary_rates([1, 0, 1, 1, 0, 0], ["a", "a", "a", "b", "b", "b"])
check(set(strata) == {"a", "b"} and abs(strata["a"].estimate - 2 / 3) < 1e-12, "stratified Wilson summaries")
decision = m13.release_decision({"asr": 0.03, "utility": 0.82}, {"asr": ("max", 0.05), "utility": ("min", 0.80)})
check(decision.passed and not decision.failures, "release gate passes all thresholds")
blocked = m13.release_decision(
    {"asr": 0.08, "utility": 0.84},
    {"asr": ("max", 0.05), "utility": ("min", 0.80)},
    critical_failures=["sandbox_escape"],
)
check(not blocked.passed and len(blocked.failures) == 2, "critical event and threshold independently block release")
missing = m13.release_decision({}, {"asr": ("max", 0.05)})
check(not missing.passed and missing.failures == ("missing:asr",), "missing release metric fails closed")

# Passport and defensive checks.
passport = {
    "run_id": "m13-smoke",
    "model_id": "synthetic",
    "model_revision": "v1",
    "tokenizer_id": "none",
    "chat_template_hash": "none",
    "dataset_id": "synthetic-eval",
    "dataset_revision": "v1",
    "split": "test",
    "solver_id": "direct",
    "scorer_id": "exact-match",
    "scorer_revision": "v1",
    "decoding": {"temperature": 0.0},
    "seeds": [13],
    "sample_count": 100,
    "metrics": {"accuracy": 0.73},
    "cost": {"tokens": 0},
    "artifact_hashes": {"results.json": "abc"},
}
check(m13.validate_measurement_passport(passport) == (), "complete measurement passport")
bad = dict(passport)
bad.pop("scorer_revision")
bad["sample_count"] = 0
errors = m13.validate_measurement_passport(bad)
check("missing:scorer_revision" in errors and "invalid:sample_count" in errors, "passport reports multiple schema errors")

for func, args, label in [
    (m13.wilson_interval, (11, 10), "interval rejects impossible counts"),
    (m13.pass_at_k_unbiased, (10, 3, 11), "pass@k rejects k>n"),
    (m13.ngram_set, ("text", 0), "n-gram detector rejects n=0"),
]:
    try:
        func(*args)
    except ValueError:
        check(True, label)
    else:
        check(False, label)

try:
    m13.hierarchical_bootstrap_interval([[1, 0]], [[1]], replicates=100)
except ValueError:
    check(True, "hierarchical bootstrap rejects mismatched arrays")
else:
    check(False, "hierarchical bootstrap rejects mismatched arrays")

print(f"\nSUMMARY: {passed}/{passed} tests passed")
