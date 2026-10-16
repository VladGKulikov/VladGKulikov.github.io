#!/usr/bin/env python3
"""Validate or explicitly regenerate the frozen Module-6 numerical contract.

The default command is read-only. Pass ``--write`` only when intentionally updating
module-owned reference data after reviewing the resulting diff.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from reference_calculations import build_reference_payload

REFERENCE_PATH = MODULE_ROOT / "data" / "reference_metrics.json"


def canonical_json(payload: dict[str, object]) -> str:
    """Serialize reference data deterministically for stable diffs and hashes."""

    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def main() -> None:
    """Validate the committed payload or rewrite it only with ``--write``."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite data/reference_metrics.json after intentional review",
    )
    args = parser.parse_args()

    expected_text = canonical_json(build_reference_payload())
    if args.write:
        REFERENCE_PATH.write_text(expected_text, encoding="utf-8")
        print(f"wrote {REFERENCE_PATH}")
    else:
        committed_text = REFERENCE_PATH.read_text(encoding="utf-8")
        sys.path.insert(0, str(MODULE_ROOT.parents[1] / "tools"))
        from numeric_contract import assert_contract
        assert_contract(json.loads(committed_text), json.loads(expected_text))
        print(f"validated {REFERENCE_PATH}")

    payload = json.loads(expected_text)
    params = payload["llama_3_1_8b"]["parameters"]
    curves = payload["depth_experiment"]["curves"]
    print("SHA-256:", hashlib.sha256(expected_text.encode("utf-8")).hexdigest())
    print("parameters:", f"{params['total']:,}")
    print(
        "depth layer 36:",
        {mode: round(values[-1], 6) for mode, values in curves.items()},
    )
    print(
        "gpt-oss table:",
        [round(v, 5) for v in payload["gpt_oss_swiglu"]["diagonal_outputs"]],
    )


if __name__ == "__main__":
    main()
