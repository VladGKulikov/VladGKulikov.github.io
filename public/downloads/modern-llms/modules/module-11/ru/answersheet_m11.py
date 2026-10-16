#!/usr/bin/env python3
"""Read-only validator for Module 11 reference calculations.

Use ``--write`` only when intentionally updating the frozen module-owned payload.
Running without flags recomputes probability, speculative-decoding, quantization, and
weight-stream arithmetic and requires exact equality with
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
    load_reference_payload,
    sha256_json,
    validate_reference_payload,
)


def write_reference() -> None:
    """Intentionally write the canonical payload after explicit CLI confirmation."""

    import json

    payload = build_reference_payload()
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Validate by default and write only with explicit user intent."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="overwrite the frozen reference JSON")
    args = parser.parse_args()
    if args.write:
        write_reference()
        print(f"wrote: {REFERENCE_PATH}")

    payload = load_reference_payload()
    validate_reference_payload(payload)
    print(f"read-only validation passed: {REFERENCE_PATH}")
    print(f"sha256: {sha256_json(payload)}")

    trace = payload["trace"]
    print(
        "rejection:",
        f"alpha={trace['alpha']:.2f}",
        f"TV={trace['total_variation']:.2f}",
        f"residual={trace['residual']}",
    )
    external = payload["speedup"]["external"]
    mtp = payload["speedup"]["mtp_scenario"]
    print(
        "speedup scenarios:",
        f"external gamma*={external['gamma_star']} S={external['speedup_star']:.4f}",
        f"MTP S={mtp['speedup']:.4f}",
    )
    roof = payload["roofline"]
    print(
        "weight-only scenarios:",
        f"bf16={roof['formats']['bf16']['weight_stream_ceiling_tokens_per_second']:.3f} tok/s",
        f"NVFP4+MTP={roof['combined_nvfp4_mtp_scenario_tokens_per_second']:.3f} tok/s",
    )


if __name__ == "__main__":
    main()
