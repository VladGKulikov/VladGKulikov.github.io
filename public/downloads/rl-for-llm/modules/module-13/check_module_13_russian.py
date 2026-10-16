#!/usr/bin/env python3
"""Module-specific Russian prose and terminology gate for Module 13."""
from __future__ import annotations

from pathlib import Path
import re
import sys

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import check_russian_prose as shared  # noqa: E402

shared.ALLOWED.update({
    "exp",
    "Wilson", "McNemar", "Wald", "Efron", "Bonferroni", "Bradley", "Terry",
    "ASR", "FPR", "TPR", "ECE", "ESS", "PPL", "NLL", "KL", "RLHF", "RLVR",
    "GPQA", "HLE", "HumanEval", "SWE-bench", "BFCL", "LiveCodeBench", "LiveBench",
    "RewardBench", "AILuminate", "HELM", "Inspect", "NIST", "CRFM", "EleutherAI",
    "lm-evaluation-harness", "lm-eval", "Qwen", "Qwen3", "Colab", "T4", "NumPy",
    "Python", "JSON", "YAML", "HTML", "MathML", "TeX", "API", "CPU", "GPU",
    "Best-of-N", "Best-of-64", "pass@k", "pass@20", "pass@5", "A/B", "B/A", "canary",
    "n", "Inspect", "HELM", "GPQA", "Diamond", "HLE", "AIME",
    "LiveCodeBench", "LiveBench", "SWE-bench", "Verified", "BFCL",
    "AILuminate", "v1.0", "ASR", "FPR", "ASR/FPR", "NIST", "AI", "RMF",
    "YAML", "Stanford", "CRFM", "top-p/top-k", "SHA-256", "V4",
})
shared.KNOWN_PHRASES += (
    "Evaluation современных LLM",
    "Hugging Face",
    "false-positive rate",
    "benchmark-side containment",
    "Wilson score interval",
    "central processing unit",
    "attack success rate",
    "false positive rate",
    "true positive rate",
    "large language model",
    "exact McNemar test",
    "paired bootstrap",
    "hierarchical bootstrap",
    "expected calibration error",
    "protocol passport",
    "reward hacking",
    "prompt injection",
    "winner's curse",
)

DEFAULT_FILES = [
    MODULE_DIR / "Module_13_RU.md",
    MODULE_DIR / "Module_13_Assessment_RU.md",
]

_ORIGINAL_PREPARE = shared.prepare

def _prepare_without_primary_source_titles(text: str) -> str:
    text = text.split("\n## Маршрут по первичным источникам", 1)[0]
    return _ORIGINAL_PREPARE(text)

shared.prepare = _prepare_without_primary_source_titles

BANNED_PROSE = {
    "dataset": "набор данных",
    "prompt": "запрос или системное сообщение — по смыслу",
    "score": "оценка, логит или значение метрики — по смыслу",
    "target": "целевая величина",
    "baseline": "базовая линия или исходная система",
    "pipeline": "конвейер",
    "framework": "библиотека или система",
    "leaderboard": "лидерборд",
    "judge prompt": "запрос к судье",
    "over-refusal": "избыточный отказ",
    "gold set": "эталонный набор",
    "deploy": "выпуск или развёртывание",
}

ORDINARY_ENGLISH = re.compile(
    r"\b(?:dataset|prompt|score|target|baseline|pipeline|framework|server|worker|"
    r"training|inference|throughput|deployment|evaluation|accuracy|utility|"
    r"selection|coverage|judge|scorer|verifier|evidence|safety|release)\b",
    flags=re.IGNORECASE,
)


def visible_prose(path: Path) -> str:
    return shared.prepare(path.read_text(encoding="utf-8"))


def main(argv: list[str]) -> int:
    paths = [Path(arg).resolve() for arg in argv] if argv else DEFAULT_FILES
    code = shared.main([str(path) for path in paths])
    if code:
        return code

    issues: list[str] = []
    for path in paths:
        prose = visible_prose(path)
        lowered = prose.lower()
        for bad, preferred in BANNED_PROSE.items():
            if bad.lower() in lowered:
                issues.append(f"{path.name}: '{bad}' -> prefer '{preferred}'")
        match = ORDINARY_ENGLISH.search(prose)
        if match:
            issues.append(
                f"{path.name}: ordinary English noun remains in visible Russian prose: "
                f"'{match.group(0)}'"
            )
    if issues:
        print("Module 13 Russian language issues:")
        for issue in issues:
            print("  " + issue)
        print(f"FAILED: {len(issues)} issue(s)")
        return 1
    print("PASS: Module 13 Russian terminology, abbreviation, and calque checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
