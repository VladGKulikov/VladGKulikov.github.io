#!/usr/bin/env python3
"""Read-only validator for Module 18 numerical and safety contracts.

Use ``--write`` only when intentionally updating the frozen module-owned payload.
Without flags, the script recomputes every value and requires exact equality with
``data/reference_metrics.json``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from reference_calculations import (  # noqa: E402
    REFERENCE_PATH,
    build_reference_payload,
    canonical_json,
    safe_calculate,
    validate_reference,
)


def main() -> None:
    """Validate by default and rewrite only with explicit user intent."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="overwrite frozen reference JSON")
    args = parser.parse_args()
    if args.write:
        REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        REFERENCE_PATH.write_text(canonical_json(build_reference_payload()), encoding="utf-8")
        print(f"wrote: {REFERENCE_PATH}")
    payload = validate_reference()
    print(f"read-only validation passed: {REFERENCE_PATH}")
    context = payload["context"]
    print(
        "context:",
        f"{context['naive_prefill_tokens']:,} -> {context['cached_prefill_tokens']:,} tokens",
        f"KV {context['final_kv_gib']:.3f} GiB",
    )
    latency = payload["latency"]
    print(
        "conditional step:",
        f"{latency['prefill_floor_ms']:.1f} + {latency['decode_ms']:.1f} + "
        f"{latency['tool_latency_ms']:.1f} = {latency['total_ms']:.1f} ms",
    )
    workflow = payload["workflow"]
    print(
        "workflow:",
        f"work-baseline x{workflow['work_baseline_speedup']:.2f} "
        f"(limit x{workflow['work_baseline_limit']:.0f});",
        f"end-to-end x{workflow['end_to_end_speedup']:.2f} "
        f"(limit x{workflow['end_to_end_limit']:.0f});",
        f"KV {workflow['kv_gib']:.3f} GiB",
    )
    print("safe calculator:", safe_calculate("13.96 * 2"))


if __name__ == "__main__":
    main()
