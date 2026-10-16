#!/usr/bin/env python3
"""Read-only validator for Module 17 reference calculations.

Use ``--write`` only when intentionally updating the frozen module-owned payload.
Running the script without flags recomputes every storage, retrieval, metric, and
context-packing result and requires exact equality with `data/reference_metrics.json`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from reference_calculations import REFERENCE_PATH, validate_reference, write_reference  # noqa: E402


def main() -> None:
    """Validate by default and write only with explicit user intent."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="overwrite the frozen reference JSON")
    args = parser.parse_args()
    if args.write:
        print(f"wrote: {write_reference()}")
    payload = validate_reference()
    print(f"read-only validation passed: {REFERENCE_PATH}")
    storage = payload["storage"]
    print(
        "storage:",
        f"fp32 {storage['fp32_gib']:.3f} GiB",
        "->",
        f"binary {storage['binary_gib']:.3f} GiB",
    )
    corpus = payload["corpus"]
    print(
        "corpus:",
        f"{corpus['chunks']:,} chunks",
        f"binary {corpus['binary_mib']:.3f} MiB",
    )
    ann = payload["ann"]
    print(
        "ANN scenario:",
        f"{ann['exact_seconds']:.3f} s -> {ann['ann_milliseconds']:.3f} ms",
        f"x{ann['speedup']:.0f}",
    )
    lc = payload["long_context_vs_rag"]
    print(
        "LC/RAG:",
        f"{lc['long_kv_gib']:.3f}/{lc['rag_kv_gib']:.3f} GiB",
        f"x{lc['ratio']:.2f}",
    )
    for language in ("ru", "en"):
        print(language, payload["retrieval_results"][language]["metrics"])


if __name__ == "__main__":
    main()
