#!/usr/bin/env python3
"""Maintainer answer sheet for Module 15.

The default mode is read-only: it prints the deterministic module contract.  Rewriting
``data/reference_metrics.json`` requires the explicit ``--write`` flag so routine runs
cannot silently change learner assertions or figures.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from reference_calculations import build_reference, write_reference


def main() -> None:
    """Print reference values and optionally rewrite the frozen payload."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="rewrite data/reference_metrics.json")
    args = parser.parse_args()
    payload = build_reference()
    if args.write:
        path = write_reference()
        print(f"wrote {path}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        print("read-only mode; pass --write to update the frozen payload")


if __name__ == "__main__":
    main()
