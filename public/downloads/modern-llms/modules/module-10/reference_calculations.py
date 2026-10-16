#!/usr/bin/env python3
"""Canonical calculations and teaching primitives for Module 10.

The module owns its numerical inputs.  The functions below are intentionally small,
validated, and dependency-free so the same contracts can be embedded in the learner
notebooks, checked by the answer sheet, and reused by visualization generators.

Decimal units are used for bandwidth and vendor-style payload labels (GB/TB).  Binary
units are used whenever bytes are divided by powers of two (KiB/MiB/GiB/TiB).
"""
from __future__ import annotations

from dataclasses import dataclass
from teaching_core import decode_flops, prefill_flops, cached_input_cost
import hashlib
import json
import math
from pathlib import Path
from typing import Hashable, Iterable, Mapping, Sequence

MODULE_ROOT = Path(__file__).resolve().parent
REFERENCE_PATH = MODULE_ROOT / "data" / "reference_metrics.json"

KIB = 2**10
MIB = 2**20
GIB = 2**30
TIB = 2**40
DECIMAL_GB = 10**9
DECIMAL_TB = 10**12


def _positive_int(name: str, value: int) -> int:
    """Return a strictly positive integer, rejecting booleans and lossy casts."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(name: str, value: int) -> int:
    """Return a non-negative integer, rejecting booleans and lossy casts."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_finite(name: str, value: float) -> float:
    """Return a positive finite float."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def _fraction(name: str, value: float, *, include_zero: bool = False) -> float:
    """Validate a finite fraction in ``(0, 1]`` or ``[0, 1]``."""

    number = _positive_finite(name, value) if not include_zero else float(value)
    if include_zero:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if number > 1.0:
        raise ValueError(f"{name} must not exceed 1")
    return number


@dataclass(frozen=True)
class AttentionKVGeometry:
    """KV geometry for ordinary MHA/GQA/MQA attention.

    ``attention_layers`` counts only layers that materialize a KV cache.  This allows
    hybrid models to be represented without pretending that state-space layers store KV.
    The returned payload is for one token across all such layers and includes K and V.
    """

    name: str
    attention_layers: int
    kv_heads: int
    head_dim: int
    bytes_per_element: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        _positive_int("attention_layers", self.attention_layers)
        _positive_int("kv_heads", self.kv_heads)
        _positive_int("head_dim", self.head_dim)
        _positive_int("bytes_per_element", self.bytes_per_element)

    @property
    def bytes_per_token(self) -> int:
        """Return bytes of K and V for one token across all attention layers."""

        return (
            2
            * self.attention_layers
            * self.kv_heads
            * self.head_dim
            * self.bytes_per_element
        )


@dataclass(frozen=True)
class MLAKVGeometry:
    """KV geometry for an MLA-style latent plus a separate positional component."""

    name: str
    attention_layers: int
    latent_dim: int
    rope_dim: int
    bytes_per_element: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        _positive_int("attention_layers", self.attention_layers)
        _positive_int("latent_dim", self.latent_dim)
        _non_negative_int("rope_dim", self.rope_dim)
        _positive_int("bytes_per_element", self.bytes_per_element)

    @property
    def bytes_per_token(self) -> int:
        """Return latent-plus-position bytes for one token across all layers."""

        return (
            self.attention_layers
            * (self.latent_dim + self.rope_dim)
            * self.bytes_per_element
        )


KVGeometry = AttentionKVGeometry | MLAKVGeometry


@dataclass(frozen=True)
class ServingHardware:
    """Transparent hardware inputs for arithmetic serving ceilings.

    Bandwidth and peak FLOPs are decimal rates.  Device capacity and reserve are byte
    counts and are displayed in binary units by the notebook.
    """

    weight_bytes: int
    memory_bandwidth_bytes_per_second: float
    peak_flops_per_second: float
    parameter_count: int
    device_capacity_bytes: int
    runtime_reserve_bytes: int = 0

    def __post_init__(self) -> None:
        _positive_int("weight_bytes", self.weight_bytes)
        _positive_finite(
            "memory_bandwidth_bytes_per_second",
            self.memory_bandwidth_bytes_per_second,
        )
        _positive_finite("peak_flops_per_second", self.peak_flops_per_second)
        _positive_int("parameter_count", self.parameter_count)
        _positive_int("device_capacity_bytes", self.device_capacity_bytes)
        _non_negative_int("runtime_reserve_bytes", self.runtime_reserve_bytes)
        if self.weight_bytes + self.runtime_reserve_bytes > self.device_capacity_bytes:
            raise ValueError("weights plus runtime reserve exceed device capacity")


@dataclass(frozen=True)
class DecodeCeiling:
    """One arithmetic lower-bound traffic scenario for autoregressive decode."""

    batch_size: int
    context_tokens: int
    weight_bytes: int
    kv_bytes: int
    total_bytes_per_step: int
    aggregate_tokens_per_second: float
    per_request_tokens_per_second: float


def kv_payload_bytes(
    geometry: KVGeometry,
    context_tokens: int,
    batch_size: int = 1,
    kv_scale: float = 1.0,
) -> int:
    """Return total KV payload bytes for a batch of equal-length requests.

    ``kv_scale`` models storage compression.  The result must be an exact integer number
    of bytes; this prevents an apparently precise memory label from hiding fractional
    bytes created by an incompatible bit-rate assumption.
    """

    tokens = _non_negative_int("context_tokens", context_tokens)
    batch = _positive_int("batch_size", batch_size)
    scale = _positive_finite("kv_scale", kv_scale)
    raw = geometry.bytes_per_token * tokens * batch * scale
    rounded = round(raw)
    if not math.isclose(raw, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("scaled KV payload is not an integer number of bytes")
    return int(rounded)


def decode_ceiling(
    hardware: ServingHardware,
    geometry: KVGeometry,
    context_tokens: int,
    batch_size: int = 1,
    kv_scale: float = 1.0,
) -> DecodeCeiling:
    """Return a weight-plus-KV bandwidth ceiling for one decode step.

    Assumption: one generated token requires one full read of the model weights and of
    every request's resident KV payload.  Kernel overhead, activations, communication,
    allocator effects, and measured utilization are deliberately outside this floor.
    """

    batch = _positive_int("batch_size", batch_size)
    tokens = _non_negative_int("context_tokens", context_tokens)
    kv_bytes = kv_payload_bytes(geometry, tokens, batch, kv_scale)
    total = hardware.weight_bytes + kv_bytes
    aggregate = batch * hardware.memory_bandwidth_bytes_per_second / total
    return DecodeCeiling(
        batch_size=batch,
        context_tokens=tokens,
        weight_bytes=hardware.weight_bytes,
        kv_bytes=kv_bytes,
        total_bytes_per_step=total,
        aggregate_tokens_per_second=aggregate,
        per_request_tokens_per_second=aggregate / batch,
    )


def decode_arithmetic_intensity(
    hardware: ServingHardware,
    geometry: KVGeometry,
    context_tokens: int,
    batch_size: int = 1,
    kv_scale: float = 1.0,
) -> float:
    """Projection-only intensity; use decode_flops for context attention too."""

    ceiling = decode_ceiling(hardware, geometry, context_tokens, batch_size, kv_scale)
    flops_per_step = 2 * hardware.parameter_count * ceiling.batch_size
    return flops_per_step / ceiling.total_bytes_per_step


def batch_knee(
    hardware: ServingHardware,
    geometry: KVGeometry,
    context_tokens: int,
    kv_scale: float = 1.0,
) -> float:
    """Return ``B* = weight_bytes / KV_bytes_per_request`` for positive context."""

    tokens = _positive_int("context_tokens", context_tokens)
    per_request = kv_payload_bytes(geometry, tokens, 1, kv_scale)
    return hardware.weight_bytes / per_request


def max_resident_requests(
    hardware: ServingHardware,
    geometry: KVGeometry,
    context_tokens: int,
    kv_scale: float = 1.0,
) -> int:
    """Return a KV-only request-capacity upper bound after weights and reserve.

    The result excludes activations, temporary workspaces, page tables, fragmentation,
    communication buffers, and allocator overhead; it is therefore an upper bound.
    """

    tokens = _positive_int("context_tokens", context_tokens)
    per_request = kv_payload_bytes(geometry, tokens, 1, kv_scale)
    free = (
        hardware.device_capacity_bytes
        - hardware.weight_bytes
        - hardware.runtime_reserve_bytes
    )
    return free // per_request


def naive_preallocation_waste(max_sequence_tokens: int, used_tokens: int) -> float:
    """Return the unused fraction of a fixed maximum-length allocation."""

    maximum = _positive_int("max_sequence_tokens", max_sequence_tokens)
    used = _non_negative_int("used_tokens", used_tokens)
    if used > maximum:
        raise ValueError("used_tokens cannot exceed max_sequence_tokens")
    return (maximum - used) / maximum


def expected_paged_waste(avg_sequence_tokens: int, block_size: int) -> float:
    """Return expected internal waste under a uniform final-block remainder.

    The expected unused slots in the last block are ``(block_size - 1) / 2``.  This is a
    model assumption, not a measured workload statistic.
    """

    average = _positive_int("avg_sequence_tokens", avg_sequence_tokens)
    block = _positive_int("block_size", block_size)
    tail = (block - 1) / 2
    return tail / (average + tail)


@dataclass
class _PhysicalBlock:
    """Internal physical block state for the teaching page-pool simulator."""

    filled_slots: int = 0
    ref_count: int = 0


@dataclass
class _SequenceState:
    """Internal logical sequence state."""

    length: int
    blocks: list[int]


class PagedKVPool:
    """A validated page-table simulator with prefix sharing and copy-on-write.

    It models block ownership and logical lengths; it does not store real tensors.  A
    shared partial final block is copied before either sequence appends into it.  All
    allocations are preflighted so an out-of-memory error leaves the pool unchanged.
    """

    def __init__(self, total_blocks: int, block_size: int = 16) -> None:
        self.total_blocks = _positive_int("total_blocks", total_blocks)
        self.block_size = _positive_int("block_size", block_size)
        self._blocks = [_PhysicalBlock() for _ in range(self.total_blocks)]
        self._free = list(range(self.total_blocks - 1, -1, -1))
        self._sequences: dict[Hashable, _SequenceState] = {}

    def _require_sequence(self, sequence_id: Hashable) -> _SequenceState:
        try:
            return self._sequences[sequence_id]
        except KeyError as exc:
            raise KeyError(f"unknown sequence: {sequence_id!r}") from exc

    def create(self, sequence_id: Hashable) -> None:
        """Create an empty sequence with a unique hashable identifier."""

        try:
            hash(sequence_id)
        except TypeError as exc:
            raise TypeError("sequence_id must be hashable") from exc
        if sequence_id in self._sequences:
            raise ValueError(f"sequence already exists: {sequence_id!r}")
        self._sequences[sequence_id] = _SequenceState(length=0, blocks=[])

    def _blocks_needed_for_append(self, state: _SequenceState, token_count: int) -> int:
        if token_count == 0:
            return 0
        needed = 0
        remainder = state.length % self.block_size
        available_in_last = 0
        if state.blocks and remainder:
            last = self._blocks[state.blocks[-1]]
            if last.ref_count > 1:
                needed += 1  # copy-on-write replacement for the shared partial block
            available_in_last = self.block_size - remainder
        remaining = max(0, token_count - available_in_last)
        needed += math.ceil(remaining / self.block_size)
        return needed

    def _allocate_block(self, filled_slots: int = 0) -> int:
        block_id = self._free.pop()
        block = self._blocks[block_id]
        block.filled_slots = filled_slots
        block.ref_count = 1
        return block_id

    def append(self, sequence_id: Hashable, token_count: int) -> None:
        """Append logical tokens, allocating arbitrary physical blocks as needed."""

        count = _non_negative_int("token_count", token_count)
        state = self._require_sequence(sequence_id)
        required = self._blocks_needed_for_append(state, count)
        if required > len(self._free):
            raise MemoryError(
                f"need {required} free blocks, but only {len(self._free)} remain"
            )
        if count == 0:
            return

        remainder = state.length % self.block_size
        if state.blocks and remainder:
            last_id = state.blocks[-1]
            last = self._blocks[last_id]
            if last.ref_count > 1:
                last.ref_count -= 1
                replacement = self._allocate_block(filled_slots=remainder)
                state.blocks[-1] = replacement
                last_id = replacement
                last = self._blocks[last_id]
            take = min(count, self.block_size - remainder)
            last.filled_slots = remainder + take
            state.length += take
            count -= take

        while count:
            take = min(count, self.block_size)
            block_id = self._allocate_block(filled_slots=take)
            state.blocks.append(block_id)
            state.length += take
            count -= take
        self._refresh_fills()
        self.validate()

    def share_prefix(
        self,
        source_id: Hashable,
        target_id: Hashable,
        prefix_tokens: int,
    ) -> None:
        """Create ``target_id`` by sharing source blocks that cover a logical prefix."""

        source = self._require_sequence(source_id)
        prefix = _non_negative_int("prefix_tokens", prefix_tokens)
        if prefix > source.length:
            raise ValueError("prefix_tokens exceeds source length")
        if target_id in self._sequences:
            raise ValueError(f"target sequence already exists: {target_id!r}")
        try:
            hash(target_id)
        except TypeError as exc:
            raise TypeError("target_id must be hashable") from exc
        block_count = math.ceil(prefix / self.block_size) if prefix else 0
        table = source.blocks[:block_count]
        for block_id in table:
            self._blocks[block_id].ref_count += 1
        self._sequences[target_id] = _SequenceState(prefix, list(table))
        self.validate()

    def release(self, sequence_id: Hashable) -> None:
        """Release a sequence and return unreferenced physical blocks to the free list."""

        state = self._require_sequence(sequence_id)
        for block_id in state.blocks:
            block = self._blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                block.filled_slots = 0
                self._free.append(block_id)
        del self._sequences[sequence_id]
        self._refresh_fills()
        self.validate()

    def _refresh_fills(self) -> None:
        """Count live slots, including unequal prefixes sharing a physical block."""
        fills = [0] * self.total_blocks
        for live in self._sequences.values():
            for j, block_id in enumerate(live.blocks):
                fills[block_id] = max(fills[block_id], min(self.block_size, live.length - j * self.block_size))
        for block_id, block in enumerate(self._blocks):
            block.filled_slots = fills[block_id]

    def sequence_length(self, sequence_id: Hashable) -> int:
        """Return logical length in tokens."""

        return self._require_sequence(sequence_id).length

    def page_table(self, sequence_id: Hashable) -> tuple[int, ...]:
        """Return an immutable page table for inspection."""

        return tuple(self._require_sequence(sequence_id).blocks)

    @property
    def allocated_blocks(self) -> int:
        """Return the number of physical blocks with at least one reference."""

        return self.total_blocks - len(self._free)

    @property
    def free_blocks(self) -> int:
        """Return the number of currently free physical blocks."""

        return len(self._free)

    @property
    def physical_capacity_tokens(self) -> int:
        """Return token slots reserved by allocated physical blocks."""

        return self.allocated_blocks * self.block_size

    @property
    def physical_payload_tokens(self) -> int:
        """Return filled token slots stored once per physical block."""

        return sum(block.filled_slots for block in self._blocks if block.ref_count)

    @property
    def logical_tokens(self) -> int:
        """Return total logical tokens across all sequences, counting shared prefixes."""

        return sum(state.length for state in self._sequences.values())

    @property
    def internal_waste_tokens(self) -> int:
        """Return unused slots inside allocated physical blocks."""

        return self.physical_capacity_tokens - self.physical_payload_tokens

    @property
    def sharing_savings_tokens(self) -> int:
        """Return logical tokens not duplicated physically because of prefix sharing."""

        return self.logical_tokens - self.physical_payload_tokens

    def snapshot(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly state summary."""

        sequence_items = sorted(self._sequences.items(), key=lambda item: repr(item[0]))
        return {
            "sequences": {
                str(sequence_id): {
                    "length": state.length,
                    "blocks": list(state.blocks),
                }
                for sequence_id, state in sequence_items
            },
            "allocated_blocks": self.allocated_blocks,
            "free_blocks": self.free_blocks,
            "physical_capacity_tokens": self.physical_capacity_tokens,
            "physical_payload_tokens": self.physical_payload_tokens,
            "logical_tokens": self.logical_tokens,
            "internal_waste_tokens": self.internal_waste_tokens,
            "sharing_savings_tokens": self.sharing_savings_tokens,
        }

    def validate(self) -> None:
        """Check page-table, reference-count, fill, and capacity invariants."""

        if len(set(self._free)) != len(self._free):
            raise AssertionError("free list contains duplicates")
        if any(block_id < 0 or block_id >= self.total_blocks for block_id in self._free):
            raise AssertionError("free list contains an invalid block id")
        expected_refs = [0] * self.total_blocks
        for state in self._sequences.values():
            if state.length < 0:
                raise AssertionError("sequence length is negative")
            expected_count = math.ceil(state.length / self.block_size) if state.length else 0
            if len(state.blocks) != expected_count:
                raise AssertionError("page-table length does not match sequence length")
            if len(set(state.blocks)) != len(state.blocks):
                raise AssertionError("one sequence references a physical block twice")
            for block_id in state.blocks:
                if block_id < 0 or block_id >= self.total_blocks:
                    raise AssertionError("page table contains an invalid block id")
                expected_refs[block_id] += 1
        free_set = set(self._free)
        for block_id, block in enumerate(self._blocks):
            if block.ref_count != expected_refs[block_id]:
                raise AssertionError("physical ref_count does not match page tables")
            if block.ref_count == 0:
                if block_id not in free_set or block.filled_slots != 0:
                    raise AssertionError("unreferenced block is not clean and free")
            else:
                if block_id in free_set:
                    raise AssertionError("referenced block appears in free list")
                if not 1 <= block.filled_slots <= self.block_size:
                    raise AssertionError("referenced block has invalid fill")
        if self.logical_tokens < self.physical_payload_tokens:
            raise AssertionError("physical payload exceeds logical payload")


def prefill_time_floor_seconds(
    parameter_count: int,
    prompt_tokens: int,
    peak_flops_per_second: float,
) -> float:
    """Return the dense ``2 * parameters * tokens / peak_FLOPs`` arithmetic floor."""

    parameters = _positive_int("parameter_count", parameter_count)
    tokens = _non_negative_int("prompt_tokens", prompt_tokens)
    peak = _positive_finite("peak_flops_per_second", peak_flops_per_second)
    return 2 * parameters * tokens / peak


def streaming_keep_indices(length: int, budget: int, sink_tokens: int) -> tuple[int, ...]:
    """Return a deterministic sink-plus-recent-window retention set."""

    n = _non_negative_int("length", length)
    b = _positive_int("budget", budget)
    sinks = _non_negative_int("sink_tokens", sink_tokens)
    if sinks > b:
        raise ValueError("sink_tokens cannot exceed budget")
    if n <= b:
        return tuple(range(n))
    sink_count = min(sinks, n)
    recent_count = b - sink_count
    recent_start = max(sink_count, n - recent_count)
    result = tuple(range(sink_count)) + tuple(range(recent_start, n))
    if len(result) != b or len(set(result)) != b:
        raise AssertionError("retention policy produced duplicates or wrong budget")
    return result


def heavy_hitter_indices(attention_mass: Sequence[float], budget: int) -> tuple[int, ...]:
    """Return top-mass token indices, breaking ties by lower original index."""

    values = tuple(attention_mass)
    if not values:
        raise ValueError("attention_mass must be non-empty")
    b = _positive_int("budget", budget)
    if b > len(values):
        raise ValueError("budget cannot exceed the number of tokens")
    normalized: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"attention_mass[{index}] must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError("attention masses must be finite and non-negative")
        normalized.append(number)
    ranked = sorted(range(len(normalized)), key=lambda i: (-normalized[i], i))
    return tuple(sorted(ranked[:b]))


def _round_floats(value: object) -> object:
    """Round floats recursively to make the frozen JSON stable across runtimes."""

    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return round(value, 12)
    if isinstance(value, dict):
        return {str(key): _round_floats(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_floats(item) for item in value]
    return value


def build_reference_payload() -> dict[str, object]:
    """Build the canonical module-owned numerical payload."""

    geometries: list[KVGeometry] = [
        AttentionKVGeometry("70B MHA-equivalent (H=64)", 80, 64, 128, 2),
        AttentionKVGeometry("LLaMA-3.3-70B GQA", 80, 8, 128, 2),
        AttentionKVGeometry("Llama-3-8B GQA", 32, 8, 128, 2),
        MLAKVGeometry("DeepSeek-V3 MLA (latent+RoPE)", 61, 512, 64, 2),
    ]
    by_name = {geometry.name: geometry for geometry in geometries}
    g70 = by_name["LLaMA-3.3-70B GQA"]
    g8 = by_name["Llama-3-8B GQA"]
    hardware = ServingHardware(
        weight_bytes=16_000_000_000,
        memory_bandwidth_bytes_per_second=3.35e12,
        peak_flops_per_second=990e12,
        parameter_count=8_000_000_000,
        device_capacity_bytes=80 * GIB,
        runtime_reserve_bytes=8 * GIB,
    )

    ladder = {
        geometry.name: {
            "bytes_per_token": geometry.bytes_per_token,
            "kib_per_token": geometry.bytes_per_token / KIB,
        }
        for geometry in geometries
    }
    contexts = [8_192, 32_768, 131_072, 1_000_000, 10_000_000]
    context_memory = {
        str(tokens): {
            "bytes": kv_payload_bytes(g70, tokens),
            "decimal_gb": kv_payload_bytes(g70, tokens) / DECIMAL_GB,
            "binary_gib": kv_payload_bytes(g70, tokens) / GIB,
        }
        for tokens in contexts
    }

    ceiling_contexts = [0, 8_192, 131_072, 1_000_000]
    ceilings = {
        str(tokens): {
            "kv_bytes": decode_ceiling(hardware, g8, tokens).kv_bytes,
            "aggregate_tokens_per_second": decode_ceiling(
                hardware, g8, tokens
            ).aggregate_tokens_per_second,
        }
        for tokens in ceiling_contexts
    }
    intensity = decode_arithmetic_intensity(hardware, g8, 131_072)
    ridge = hardware.peak_flops_per_second / hardware.memory_bandwidth_bytes_per_second
    knees = {
        str(tokens): batch_knee(hardware, g8, tokens)
        for tokens in (8_192, 32_768, 131_072)
    }
    b32 = decode_ceiling(hardware, g8, 8_192, batch_size=32)
    ideal_b32 = 32 * hardware.memory_bandwidth_bytes_per_second / hardware.weight_bytes

    capacities = {
        str(tokens): max_resident_requests(hardware, g8, tokens)
        for tokens in (8_192, 32_768, 131_072, 1_000_000)
    }

    pool = PagedKVPool(total_blocks=8, block_size=4)
    pool.create("A")
    pool.append("A", 6)
    snapshot_a = pool.snapshot()
    pool.share_prefix("A", "B", 6)
    snapshot_shared = pool.snapshot()
    pool.append("B", 3)
    snapshot_cow = pool.snapshot()
    pool.release("A")
    snapshot_released = pool.snapshot()

    naive_waste = naive_preallocation_waste(32_768, 2_048)
    paged_waste = expected_paged_waste(2_048, 16)

    full_prefill = prefill_time_floor_seconds(
        hardware.parameter_count, 8_192, hardware.peak_flops_per_second
    )
    hit_prefill = prefill_time_floor_seconds(
        hardware.parameter_count, 512, hardware.peak_flops_per_second
    )
    decode_weight_floor = hardware.weight_bytes / hardware.memory_bandwidth_bytes_per_second
    chunked_itl = decode_weight_floor + hit_prefill
    full_causal = prefill_flops(8_000_000_000, 32, 32, 128, 8192) / 990e12
    suffix_causal = prefill_flops(8_000_000_000, 32, 32, 128, 512, 7680) / 990e12
    cost = cached_input_cost(7680, 512, 10)
    full_intensity = decode_flops(8_000_000_000, 32, 32, 128, 131072) / (16e9 + 131072 * g8.bytes_per_token)

    compressed_scale = 3 / 16
    turbo = decode_ceiling(hardware, g8, 131_072, kv_scale=compressed_scale)
    compressed_kib = g8.bytes_per_token * compressed_scale / KIB

    mass = (
        0.625,
        0.897,
        0.776,
        0.225,
        0.300,
        0.874,
        0.005,
        0.821,
        0.797,
        0.468,
        0.303,
        0.278,
        0.255,
        0.445,
        0.505,
        0.553,
    )
    h2o = heavy_hitter_indices(mass, 8)
    streaming = streaming_keep_indices(16, 8, 4)
    lost = tuple(index for index in h2o if index not in streaming)

    payload: dict[str, object] = {
        "schema_version": 3,
        "unit_policy": {
            "binary": "KiB/MiB/GiB/TiB use powers of two",
            "decimal": "GB/TB and bandwidth use powers of ten",
        },
        "kv_ladder": ladder,
        "context_memory_70b_gqa": context_memory,
        "hardware": {
            "weight_bytes": hardware.weight_bytes,
            "weight_decimal_gb": hardware.weight_bytes / DECIMAL_GB,
            "weight_binary_gib": hardware.weight_bytes / GIB,
            "bandwidth_decimal_tb_per_second": (
                hardware.memory_bandwidth_bytes_per_second / DECIMAL_TB
            ),
            "peak_decimal_tflops": hardware.peak_flops_per_second / DECIMAL_TB,
            "device_binary_gib": hardware.device_capacity_bytes / GIB,
            "runtime_reserve_binary_gib": hardware.runtime_reserve_bytes / GIB,
        },
        "roofline": {
            "ceilings": ceilings,
            "intensity_flops_per_byte_128k": intensity,
            "intensity_including_attention_128k": full_intensity,
            "ridge_to_full_intensity": ridge / full_intensity,
            "ridge_flops_per_byte": ridge,
            "ridge_to_intensity": ridge / intensity,
            "batch_knees": knees,
            "batch_32_8k": {
                "aggregate_tokens_per_second": b32.aggregate_tokens_per_second,
                "ideal_without_kv_tokens_per_second": ideal_b32,
                "kv_tax_factor": ideal_b32 / b32.aggregate_tokens_per_second,
            },
        },
        "capacity_upper_bound": {
            "device_binary_gib": 80,
            "runtime_reserve_binary_gib": 8,
            "max_requests": capacities,
        },
        "paged": {
            "naive_waste_fraction": naive_waste,
            "expected_paged_waste_fraction": paged_waste,
            "exact_2048_waste_fraction": 0.0,
            "preallocation_capacity_factor": 32_768 / 2_048,
            "trace": {
                "after_a": snapshot_a,
                "after_share": snapshot_shared,
                "after_b_append": snapshot_cow,
                "after_a_release": snapshot_released,
            },
        },
        "prefix_and_interference": {
            "full_prefill_ms": full_prefill * 1e3,
            "cached_suffix_prefill_ms": hit_prefill * 1e3,
            "arithmetic_speedup": full_prefill / hit_prefill,
            "full_causal_prefill_ms": full_causal * 1e3,
            "cached_causal_suffix_ms": suffix_causal * 1e3,
            "causal_arithmetic_ratio": full_causal / suffix_causal,
            "illustrative_input_cost": cost,
            "decode_weight_floor_ms": decode_weight_floor * 1e3,
            "monolithic_spike_ms": (full_prefill + decode_weight_floor) * 1e3,
            "chunked_itl_ms": chunked_itl * 1e3,
        },
        "compression": {
            "base_kib_per_token": g8.bytes_per_token / KIB,
            "compressed_bits_per_coordinate": 3,
            "base_bits_per_coordinate": 16,
            "compressed_kib_per_token": compressed_kib,
            "ceiling_128k_tokens_per_second": turbo.aggregate_tokens_per_second,
        },
        "eviction": {
            "attention_mass": mass,
            "heavy_hitters": h2o,
            "streaming": streaming,
            "lost_heavy_hitters": lost,
        },
    }
    return _round_floats(payload)  # type: ignore[return-value]


def canonical_json(payload: Mapping[str, object]) -> str:
    """Return stable compact UTF-8 JSON text."""

    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(payload: Mapping[str, object]) -> str:
    """Return SHA-256 of canonical JSON bytes."""

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def load_reference_payload() -> dict[str, object]:
    """Load the module-owned frozen reference payload."""

    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


def validate_reference_payload(payload: Mapping[str, object]) -> None:
    """Require exact canonical equality with a fresh recomputation."""

    expected = build_reference_payload()
    if canonical_json(payload) != canonical_json(expected):
        raise AssertionError(
            "reference_metrics.json differs from canonical calculations; "
            "inspect the diff before using --write"
        )


if __name__ == "__main__":
    computed = build_reference_payload()
    print(canonical_json(computed))
    print("sha256:", sha256_json(computed))
