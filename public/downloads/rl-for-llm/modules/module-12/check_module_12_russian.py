#!/usr/bin/env python3
"""Module-specific Russian prose and terminology gate for Module 12.

The shared checker removes code, TeX, URLs, and English term expansions in
backticks. This wrapper adds proper names used in the infrastructure module and
rejects a small set of recurrent calques and project-name misspellings.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import check_russian_prose as shared  # noqa: E402

shared.ALLOWED.update({
    # Distributed training, precision, and memory.
    "Adam", "Adafactor", "SGD", "DDP", "FSDP", "FSDP2", "ZeRO", "LoRA",
    "DTensor", "NCCL", "CUDA", "OOM", "GB", "GiB", "KiB", "FP32",
    "float32", "bfloat16", "bf16", "KV", "Qwen2.5-7B", "Async",
    # Serving and orchestration systems.
    "vLLM", "SGLang", "PagedAttention", "RadixAttention", "DeepSpeed",
    "Megatron", "Ray", "Kubernetes", "Apache", "Arrow", "Parquet",
    "Transformers", "Accelerate", "OpenRLHF", "NeMo", "PRIME-RL",
    "SkyRL", "AReaL", "HybridFlow", "AsyncGRPOTrainer", "NVIDIA", "verl",
    # Statistical and systems abbreviations.
    "TIM", "ESS", "p50", "p95", "p99", "p50/p95/p99", "MTBF",
    "FSDP/ZeRO", "FSDP/Megatron", "FSDP/DTensor", "RLHF/RLVR",
    # Conventional labels and versioned names.
    "D-HybridEngine", "Checkpoint", "Engine", "teacher", "forcing",
})
shared.KNOWN_PHRASES += (
    "Group Relative Policy Optimization",
    "Fully Sharded Data Parallel",
    "Zero Redundancy Optimizer",
    "Distributed Data Parallel",
    "Low-Rank Adaptation",
    "Kullback–Leibler divergence",
    "central processing unit",
    "graphics processing unit",
    "end of sequence",
    "effective sample size",
    "training–inference mismatch",
    "policy lag",
    "continuous batching",
    "truncated importance sampling",
    "activation checkpointing",
    "chunked prefill",
    "fused kernels",
    "Checkpoint Engine",
    "Megatron Core",
    "Apache Arrow",
)

DEFAULT_FILES = [
    MODULE_DIR / "Module_12_RU.md",
    MODULE_DIR / "Module_12_Assessment_RU.md",
]

# The key is the rejected phrase in visible prose; the value documents the
# preferred course wording for the diagnostic message.
BANNED_PROSE = {
    "чекпойнт": "контрольная точка",
    "инвалидирует": "делает недействительным",
    "аллокатор": "распределитель памяти",
    "бэкенд": "система, движок или серверная часть — по смыслу",
    "backend": "система, движок или серверная часть — по смыслу",
    "pipeline": "конвейер",
    "dataset": "набор данных",
    "prompt": "запрос",
    "reward": "награда",
    "policy loss": "функция потерь политики",
    "wall-clock": "фактическое время выполнения",
    "стейл": "устаревший или несвежий",
    "деплой": "публикация или развёртывание",
    "рестарт": "перезапуск",
}

ORDINARY_ENGLISH = re.compile(
    r"\b(?:dataset|pipeline|baseline|prompt|reward|policy|loss|target|score|"
    r"worker|backend|server|training|inference|throughput|staleness|framework)\b",
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
        for bad, preferred in (("veRL", "verl"), ("prime-rl", "PRIME-RL")):
            if bad in prose:
                issues.append(f"{path.name}: '{bad}' -> prefer '{preferred}'")
        # Inline code and English explanations have already been removed by
        # shared.prepare(), so these are accidental English prose nouns.
        match = ORDINARY_ENGLISH.search(prose)
        if match:
            issues.append(
                f"{path.name}: ordinary English noun remains in visible Russian prose: "
                f"'{match.group(0)}'"
            )

    if issues:
        print("Module 12 Russian language issues:")
        for issue in issues:
            print("  " + issue)
        print(f"FAILED: {len(issues)} issue(s)")
        return 1

    print("PASS: Module 12 Russian terminology, project-name, and calque checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
