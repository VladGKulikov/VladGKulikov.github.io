#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only answer sheet for Module 13; rewrite the frozen payload only with --write."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))
from reference_calculations import build_reference, write_reference  # noqa: E402


def print_report(payload: dict) -> None:
    """Print the numerical contract with explicit assumptions and units."""

    tm = payload["template"]
    print("== A. Chat-template accounting ==")
    print(
        f"total {tm['total_tokens']} · trainable {tm['trainable_tokens']} "
        f"({tm['trainable_fraction']*100:.1f}%) · assistant content + EOT only"
    )

    pk = payload["packing"]
    print("\n== B. Deterministic FFD assignments ==")
    print(
        f"{pk['n_naive']} examples -> {pk['n_ffd']} bins · utilization "
        f"{pk['naive_utilization']*100:.1f}% -> {pk['ffd_utilization']*100:.1f}% "
        f"· фиксированных строк /{pk['forward_reduction']:.1f}"
    )
    for i, bucket in enumerate(pk["bins"]):
        print(f"bin {i}: used={bucket['used']:4d}/{pk['max_length']} · indices={bucket['indices']}")

    cp = payload["compute"]
    print("\n== C. Compute scenario ==")
    print(
        f"SFT {cp['sft_flops']:.3e} FLOPs · pretraining/SFT x{cp['ratio']:,.0f} "
        f"· {cp['sft_hours']:.1f} h at the stated H100/MFU assumptions"
    )
    s1 = payload["s1"]
    print(f"s1 {s1['flops']:.3e} FLOPs · {s1['minutes']:.1f} min on {s1['accelerators']} accelerators")

    best = payload["best_of_n"]
    print("\n== D. Best-of-N ==")
    for n, value in best["coverage"].items():
        n_int = int(n)
        print(f"N={n_int:2d}: coverage={value:.4f} · yield/N={value/n_int:.4f}")

    lo = payload["lora"]
    print("\n== E. LoRA / QLoRA parameter-state floor ==")
    print(
        f"LoRA {lo['params']/1e6:.2f}M ({lo['fraction']*100:.3f}%) · "
        f"states full {lo['full_training_states_gib']:.1f} GiB -> "
        f"LoRA {lo['lora_training_states_mib']:.0f} MiB "
        f"(x{lo['state_saving']:.0f})"
    )
    print(
        f"4-bit payload {lo['four_bit_payload_gib']:.2f} GiB + "
        f"adapter weights already included in "
        f"states {lo['lora_training_states_mib']:.0f} MiB = "
        f"{lo['qlora_parameter_floor_gib']:.2f} GiB floor"
    )
    print("This floor excludes activations, temporary buffers, allocator overhead, and sharding.")


def main() -> None:
    """Print the contract; use --write only for an intentional reference update."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="rewrite data/reference_metrics.json")
    args = parser.parse_args()
    payload = build_reference()
    if args.write:
        print(f"wrote {write_reference()}")
    print_report(payload)
    if not args.write:
        print("\nread-only mode; pass --write to update the frozen payload")


if __name__ == "__main__":
    main()
