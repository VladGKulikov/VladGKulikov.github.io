"""KV examples for M1 §§7–8; weight payload and runtime overhead are excluded."""
from decimal import Decimal


def kv_payload_bytes(retained_lengths, kv_heads, head_dim, bytes_per_value):
    """Count K and V for one request, with an explicit retained length per layer."""
    lengths = list(retained_lengths)
    for name, value in (("kv_heads", kv_heads), ("head_dim", head_dim), ("bytes_per_value", bytes_per_value)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if any(type(length) is not int or length < 0 for length in lengths):
        raise ValueError("retained lengths must be non-negative integers")
    return 2 * sum(lengths) * kv_heads * head_dim * bytes_per_value


def run_context_examples():
    lengths = [131072 if layer % 2 == 0 else 128 for layer in range(36)]
    hybrid = kv_payload_bytes(lengths, 8, 64, 2)
    full = kv_payload_bytes([131072]*36, 8, 64, 2)
    million = kv_payload_bytes([1_000_000]*80, 8, 128, 2)
    # Independent arithmetic: enumerate the K and V payload of each layer.
    enumerated = sum(length*8*64*2 for length in lengths for _ in ("K", "V"))
    assert hybrid == enumerated == 4_836_556_800
    assert Decimal(hybrid)/Decimal(2**30) == Decimal("4.50439453125")
    assert full == 9*2**30
    assert million == 327_680_000_000
    assert round(Decimal(million)/Decimal(2**30), 2) == Decimal("305.18")
    assert kv_payload_bytes([0, 0], 8, 64, 2) == 0
    for bad in ([True], [-1], [1.5]):
        try:
            kv_payload_bytes(bad, 8, 64, 2)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid cache length accepted")
    return {"hybrid_GiB": hybrid/2**30, "all_full_GiB": full/2**30,
            "million_token_GiB": million/2**30}


if __name__ == "__main__":
    print(run_context_examples())
