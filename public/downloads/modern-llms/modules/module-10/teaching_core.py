"""Small, dependency-free FLOP and billing models; no hardware timing claims."""
import math


def attention_flops(layers, query_heads, head_dim, queries, cached_prefix=0):
    """QK and AV for causal suffix queries, two FLOPs per multiply-add.

    Query j sees cached_prefix + j + 1 keys. Softmax, norms, projections,
    cache writes, and kernel padding are outside this count.
    """
    for name, value in (("layers", layers), ("query_heads", query_heads),
                        ("head_dim", head_dim), ("queries", queries),
                        ("cached_prefix", cached_prefix)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if min(layers, query_heads, head_dim) == 0:
        raise ValueError("attention geometry must be positive")
    pairs = queries * cached_prefix + queries * (queries + 1) // 2
    return 4 * layers * query_heads * head_dim * pairs


def decode_flops(parameters, layers, query_heads, head_dim, keys, batch=1):
    """Projection estimate plus QK/AV for one query attending to `keys`."""
    for value in (parameters, batch, keys):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("counts must be non-negative integers")
    if parameters == 0 or batch == 0:
        raise ValueError("parameters and batch must be positive")
    # Zero keys is a synthetic weights-only baseline, not a real decode query.
    attn = attention_flops(layers, query_heads, head_dim, 1, max(keys - 1, 0))
    return batch * (2 * parameters + (attn if keys else 0))


def prefill_flops(parameters, layers, query_heads, head_dim, suffix, prefix=0):
    """Only suffix projections are new; suffix attention still reads the prefix."""
    if isinstance(parameters, bool) or not isinstance(parameters, int) or parameters <= 0:
        raise ValueError("parameters must be a positive integer")
    return (attention_flops(layers, query_heads, head_dim, suffix, prefix)
            + 2 * parameters * suffix)


def cached_input_cost(prefix, suffix, requests, read_rate=0.1, write_rate=1.25):
    """Input-token equivalents: one prefix write, then complete prefix hits.

    Each suffix is uncached. No output cost, expiry, or extra cache writes.
    Multipliers are illustrative inputs, not an API tariff lookup.
    """
    for value in (prefix, suffix, requests):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("counts must be non-negative integers")
    if requests == 0 or prefix + suffix == 0:
        raise ValueError("need a request with non-empty input")
    for rate in (read_rate, write_rate):
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("rates must be positive and finite")
    baseline = requests * (prefix + suffix)
    cached = prefix * (write_rate + (requests - 1) * read_rate) + requests * suffix
    warm = prefix * read_rate + suffix
    return {"baseline": baseline, "cached": cached,
            "amortized_cost_factor": baseline / cached,
            "warm_request_cost_ratio": warm / (prefix + suffix)}
