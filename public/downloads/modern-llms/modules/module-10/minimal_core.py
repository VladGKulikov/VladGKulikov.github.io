"""Lecture-level geometry, bandwidth bound, and private page allocation."""
from dataclasses import dataclass, field
import math


def count(value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"expected an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class KVGeometry:
    attention_layers: int
    kv_heads: int
    head_dim: int
    bytes_per_element: int = 2

    def __post_init__(self):
        for value in (self.attention_layers, self.kv_heads, self.head_dim, self.bytes_per_element):
            count(value, 1)

    @property
    def bytes_per_token(self):
        return 2 * self.attention_layers * self.kv_heads * self.head_dim * self.bytes_per_element


def decode_bandwidth_ceiling(context_tokens, weight_bytes, geometry, bandwidth_Bps,
                             batch_size=1, kv_scale=1.0):
    """Aggregate tok/s, assuming one weight read and one KV read per step."""
    count(context_tokens); count(weight_bytes, 1); count(batch_size, 1)
    for value in (bandwidth_Bps, kv_scale):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError("bandwidth and scale must be positive and finite")
    kv = geometry.bytes_per_token * context_tokens * batch_size * kv_scale
    if not math.isclose(kv, round(kv), rel_tol=0, abs_tol=1e-9):
        raise ValueError("the scaled payload must contain an integer number of bytes")
    return batch_size * bandwidth_Bps / (weight_bytes + round(kv))


@dataclass
class PagedKV:
    """Private logical pages only; the notebook adds sharing and copy-on-write."""
    n_blocks: int
    block_size: int = 16
    free: list = field(init=False)
    tables: dict = field(default_factory=dict, init=False)
    lengths: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        count(self.n_blocks, 1); count(self.block_size, 1)
        self.free = list(range(self.n_blocks - 1, -1, -1))

    def append_to(self, seq_id, total_tokens):
        count(total_tokens)
        hash(seq_id)
        if total_tokens < self.lengths.get(seq_id, 0):
            raise ValueError("append_to cannot shorten a sequence")
        table = self.tables.get(seq_id, [])
        needed = (total_tokens + self.block_size - 1) // self.block_size - len(table)
        if needed > len(self.free):
            raise MemoryError("not enough free blocks")
        self.tables.setdefault(seq_id, table)
        table.extend(self.free.pop() for _ in range(needed))
        self.lengths[seq_id] = total_tokens

    def release(self, seq_id):
        self.free.extend(self.tables.pop(seq_id))
        del self.lengths[seq_id]
