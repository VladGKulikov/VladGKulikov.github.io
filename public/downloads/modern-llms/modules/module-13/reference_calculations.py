#!/usr/bin/env python3
"""Frozen numerical contract for Module 13 (SFT data mechanics).

The contract is owned by Module 13 and deliberately independent of mutable lecture prose.
It covers chat-template accounting, deterministic FFD assignments, compute estimates,
best-of-N coverage, and LoRA/QLoRA parameter-state arithmetic.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parent
REFERENCE_PATH = MODULE_ROOT / "data" / "reference_metrics.json"


def first_fit_decreasing(lengths: list[int], max_length: int) -> list[dict[str, Any]]:
    """Return deterministic FFD bins with original indices and used token counts."""

    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    normalized: list[int] = []
    for value in lengths:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("every sequence length must be a positive integer")
        if value > max_length:
            raise ValueError(f"sequence length {value} exceeds max_length={max_length}")
        normalized.append(value)

    bins: list[dict[str, Any]] = []
    for index in sorted(range(len(normalized)), key=lambda i: (-normalized[i], i)):
        length = normalized[index]
        for bucket in bins:
            if bucket["used"] + length <= max_length:
                bucket["indices"].append(index)
                bucket["lengths"].append(length)
                bucket["used"] += length
                break
        else:
            bins.append({"indices": [index], "lengths": [length], "used": length})
    return bins


def build_reference() -> dict[str, Any]:
    """Build the complete deterministic numerical payload for the learner notebooks."""

    content_lengths = {"system": 12, "user": 8, "assistant": 20}
    per_message_overhead = 5  # four header tokens plus EOT
    total_tokens = 1 + sum(per_message_overhead + n for n in content_lengths.values())
    trainable_tokens = content_lengths["assistant"] + 1  # assistant content + assistant EOT

    lengths = [
        403, 527, 315, 180, 267, 165, 425, 1347, 259, 230, 626, 556,
        443, 174, 392, 754, 120, 267, 72, 126, 76, 326, 128, 514,
    ]
    max_length = 2048
    bins = first_fit_decreasing(lengths, max_length)
    naive_utilization = sum(lengths) / (len(lengths) * max_length)
    ffd_utilization = sum(lengths) / (len(bins) * max_length)

    n_params = 8.03e9
    pretrain_tokens = 15e12
    sft_examples = 200_000
    average_sft_tokens = 1_200
    sft_epochs = 3
    sft_tokens = sft_examples * average_sft_tokens
    sft_flops = 6 * n_params * sft_tokens * sft_epochs
    pretrain_flops = 6 * n_params * pretrain_tokens
    accelerator_flops = 990e12
    mfu = 0.40
    sft_hours = sft_flops / (accelerator_flops * mfu) / 3600

    s1_params = 32e9
    s1_traces = 1_000
    s1_tokens_per_trace = 9_000
    s1_tokens = s1_traces * s1_tokens_per_trace
    s1_flops = 6 * s1_params * s1_tokens
    s1_accelerators = 8
    s1_minutes = s1_flops / (s1_accelerators * accelerator_flops * mfu) / 60

    pass_probability = 0.3
    best_of_n = {str(n): 1 - (1 - pass_probability) ** n for n in (1, 2, 4, 8, 16)}

    d_model, d_kv, layers, rank = 4096, 1024, 32, 16
    per_layer = (
        rank * (d_model + d_model)  # q
        + rank * (d_model + d_kv)  # k
        + rank * (d_model + d_kv)  # v
        + rank * (d_model + d_model)  # o
    )
    lora_params = per_layer * layers
    full_training_states_gib = 16 * n_params / 2**30
    lora_training_states_mib = 16 * lora_params / 2**20
    lora_adapter_bf16_mib = 2 * lora_params / 2**20
    four_bit_payload_gib = 0.5 * n_params / 2**30
    qlora_parameter_floor_gib = (
        four_bit_payload_gib
        + lora_training_states_mib / 1024
    )
    state_saving = full_training_states_gib * 1024 / lora_training_states_mib

    return {
        "schema_version": 3,
        "template": {
            "special_ids": {
                "pad": 0,
                "bos": 1,
                "start_header": 2,
                "end_header": 3,
                "eot": 4,
                "newline": 5,
                "role_system": 6,
                "role_user": 7,
                "role_assistant": 8,
                "role_tool": 9,
            },
            "content_lengths": content_lengths,
            "per_message_overhead": per_message_overhead,
            "total_tokens": total_tokens,
            "trainable_tokens": trainable_tokens,
            "trainable_fraction": trainable_tokens / total_tokens,
        },
        "packing": {
            "lengths": lengths,
            "max_length": max_length,
            "bins": bins,
            "n_naive": len(lengths),
            "n_ffd": len(bins),
            "naive_utilization": naive_utilization,
            "ffd_utilization": ffd_utilization,
            "forward_reduction": len(lengths) / len(bins),
        },
        "compute": {
            "n_params": n_params,
            "pretrain_tokens": pretrain_tokens,
            "sft_examples": sft_examples,
            "average_sft_tokens": average_sft_tokens,
            "sft_epochs": sft_epochs,
            "sft_tokens": sft_tokens,
            "sft_flops": sft_flops,
            "pretrain_flops": pretrain_flops,
            "ratio": pretrain_flops / sft_flops,
            "accelerator_flops": accelerator_flops,
            "mfu": mfu,
            "sft_hours": sft_hours,
        },
        "s1": {
            "n_params": s1_params,
            "traces": s1_traces,
            "tokens_per_trace": s1_tokens_per_trace,
            "tokens": s1_tokens,
            "flops": s1_flops,
            "accelerators": s1_accelerators,
            "minutes": s1_minutes,
        },
        "best_of_n": {
            "pass_probability": pass_probability,
            "coverage": best_of_n,
        },
        "lora": {
            "d_model": d_model,
            "d_kv": d_kv,
            "layers": layers,
            "rank": rank,
            "per_layer": per_layer,
            "params": lora_params,
            "fraction": lora_params / n_params,
            "full_training_states_gib": full_training_states_gib,
            "lora_training_states_mib": lora_training_states_mib,
            "lora_adapter_bf16_mib": lora_adapter_bf16_mib,
            "four_bit_payload_gib": four_bit_payload_gib,
            "qlora_parameter_floor_gib": qlora_parameter_floor_gib,
            "state_saving": state_saving,
        },
    }


def write_reference(path: Path = REFERENCE_PATH) -> Path:
    """Write the frozen payload in deterministic UTF-8 JSON form."""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(build_reference(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    """Print the contract; rewrite it only when explicitly requested."""

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
