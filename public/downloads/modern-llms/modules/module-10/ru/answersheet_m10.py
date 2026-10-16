#!/usr/bin/env python3
"""Read-only validator for Module 10 serving and KV-cache calculations.

Use ``--write`` only when intentionally updating the frozen module-owned payload.
Without flags the script recomputes every value and requires exact canonical equality
with ``data/reference_metrics.json``.
"""
from __future__ import annotations

import argparse
import json
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
    """Write the canonical reference payload after explicit CLI confirmation."""

    payload = build_reference_payload()
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Validate by default and write only with explicit user intent."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="overwrite the frozen Module 10 reference JSON",
    )
    args = parser.parse_args()
    if args.write:
        write_reference()
        print(f"wrote: {REFERENCE_PATH}")

    payload = load_reference_payload()
    validate_reference_payload(payload)
    print(f"read-only validation passed: {REFERENCE_PATH}")
    print(f"sha256: {sha256_json(payload)}")

    ladder = payload["kv_ladder"]
    print(
        "KV ladder (KiB/token):",
        ", ".join(
            f"{name}={entry['kib_per_token']:g}"
            for name, entry in ladder.items()
        ),
    )
    roof = payload["roofline"]
    print(
        "decode ceilings (tok/s):",
        ", ".join(
            f"N={context}: {entry['aggregate_tokens_per_second']:.3f}"
            for context, entry in roof["ceilings"].items()
        ),
    )
    paged = payload["paged"]
    print(
        "paged trace:",
        f"shared savings={paged['trace']['after_share']['sharing_savings_tokens']} tokens,",
        f"after COW={paged['trace']['after_b_append']['sharing_savings_tokens']} tokens",
    )


if __name__ == "__main__":
    main()
