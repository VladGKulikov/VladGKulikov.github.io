"""Canonical transparent M5 references; embedded verbatim in both notebooks."""
import math
from typing import Literal
import numpy as np

EmptyRowPolicy = Literal["error", "zero"]

def make_mask(n: int, window: int | None = None, n_sink: int = 0) -> np.ndarray:
    """Return an additive causal mask of shape [n, n].

    Allowed entries are 0 and forbidden entries are -inf. `window` keeps only the
    latest `window` causal keys; `n_sink` preserves initial real positions as StreamingLLM-style anchors.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be positive")
    if window is not None and (isinstance(window, bool) or not isinstance(window, int) or window <= 0):
        raise ValueError("window must be positive")
    if isinstance(n_sink, bool) or not isinstance(n_sink, int) or not 0 <= n_sink <= n:
        raise ValueError("n_sink must satisfy 0 <= n_sink <= n")
    if n_sink and window is None:
        raise ValueError("n_sink is meaningful only together with a finite window")

    query_pos = np.arange(n)[:, None]
    key_pos = np.arange(n)[None, :]
    allowed = key_pos <= query_pos
    if window is not None:
        allowed &= (query_pos - key_pos) < window
        allowed |= (key_pos < n_sink) & (key_pos <= query_pos)
    return np.where(allowed, 0.0, -np.inf)


def sdpa(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    sink: float | np.ndarray | None = None,
    empty_row: EmptyRowPolicy = "error",
    return_weights: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute transparent scaled dot-product attention.

    Shapes:
        Q [..., T_q, d_k], K [..., T_k, d_k], V [..., T_k, d_v].
        `mask` must broadcast to [..., T_q, T_k]. A finite entry is an additive
        bias; -inf forbids the key. A per-head sink should be reshaped so it
        broadcasts to [..., T_q, 1].

    `empty_row="error"` rejects a row with no key and no sink. `"zero"` returns
    zero weights/output for that row, which is useful for explicit padding rows.
    """
    if empty_row not in {"error", "zero"}:
        raise ValueError("empty_row must be 'error' or 'zero'")

    Q, K, V = (np.asarray(x) for x in (Q, K, V))
    if min(Q.ndim, K.ndim, V.ndim) < 2 or any(0 in x.shape for x in (Q, K, V)):
        raise ValueError("Q, K, and V must each have at least two dimensions")
    if Q.shape[-1] != K.shape[-1]:
        raise ValueError("Q and K must have the same d_k")
    if K.shape[-2] != V.shape[-2]:
        raise ValueError("K and V must have the same T_k")
    try:
        np.broadcast_shapes(Q.shape[:-2], K.shape[:-2], V.shape[:-2])
    except ValueError as exc:
        raise ValueError("Q/K/V batch dimensions are not broadcast-compatible") from exc
    if not all(np.isfinite(x).all() for x in (Q, K, V)):
        raise ValueError("Q, K, and V must contain only finite values")

    # Keep float32/float64 inputs in their precision; integer examples become float64.
    dtype = np.result_type(Q.dtype, K.dtype, V.dtype, np.float32)
    Q, K, V = (x.astype(dtype, copy=False) for x in (Q, K, V))
    d_k = Q.shape[-1]
    scale = np.asarray(1.0 / math.sqrt(d_k), dtype=dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        scores = (Q @ np.swapaxes(K, -1, -2)) * scale
    if not np.isfinite(scores).all():
        raise FloatingPointError("Q @ K.T overflowed; rescale the inputs")

    if mask is not None:
        mask = np.asarray(mask, dtype=dtype)
        if np.isnan(mask).any() or np.isposinf(mask).any():
            raise ValueError("mask may contain finite biases and -inf, but not NaN/+inf")
        try:
            mask = np.broadcast_to(mask, scores.shape)
        except ValueError as exc:
            raise ValueError(f"mask shape {mask.shape} does not broadcast to {scores.shape}") from exc
        finite_bias = np.isfinite(mask)
        with np.errstate(over="ignore", invalid="ignore"):
            scores = scores + mask
        if np.any(finite_bias & ~np.isfinite(scores)):
            raise FloatingPointError(
                "adding the finite mask bias overflowed; reduce the bias magnitude"
            )

    has_token = np.isfinite(scores).any(axis=-1, keepdims=True)
    token_max = np.max(scores, axis=-1, keepdims=True)

    sink_logits = None
    if sink is not None:
        sink_logits = np.asarray(sink, dtype=dtype)
        if not np.isfinite(sink_logits).all():
            raise ValueError("sink logits must be finite")
        try:
            sink_logits = np.broadcast_to(sink_logits, token_max.shape)
        except ValueError as exc:
            raise ValueError(
                f"sink shape {sink_logits.shape} does not broadcast to {token_max.shape}"
            ) from exc
        row_max = np.maximum(token_max, sink_logits)
    else:
        if empty_row == "error" and not has_token.all():
            bad = np.argwhere(~has_token.squeeze(-1))
            raise ValueError('Attention contains fully masked rows' + f": {bad[:8].tolist()}")
        # A zero-policy row gets a harmless max of 0; all its exponentials stay 0.
        row_max = np.where(has_token, token_max, np.zeros_like(token_max))

    token_exp = np.where(
        np.isfinite(scores), np.exp(scores - row_max), np.zeros_like(scores)
    )
    denominator = token_exp.sum(axis=-1, keepdims=True)
    if sink_logits is not None:
        denominator += np.exp(sink_logits - row_max)

    weights = np.divide(
        token_exp,
        denominator,
        out=np.zeros_like(token_exp),
        where=denominator > 0,
    )
    output = weights @ V
    return (output, weights) if return_weights else output


def gqa(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    n_heads: int,
    n_kv_heads: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Educational grouped-query attention for 2D projected sequences.

    Q [T, H*d_k], K [T, H_kv*d_k], V [T, H_kv*d_v].
    The output has shape [T, H*d_v].
    """
    Q, K, V = (np.asarray(x) for x in (Q, K, V))
    if any(x.ndim != 2 or 0 in x.shape for x in (Q, K, V)):
        raise ValueError("GQA expects 2D projected sequences")
    if not (Q.shape[0] == K.shape[0] == V.shape[0]):
        raise ValueError("Q, K, and V must have the same sequence length")
    if any(isinstance(h, bool) or not isinstance(h, int) or h <= 0 for h in (n_heads, n_kv_heads)) or n_heads % n_kv_heads:
        raise ValueError("n_heads must be a positive multiple of n_kv_heads")
    if Q.shape[1] % n_heads:
        raise ValueError("Q width must be divisible by n_heads")
    if K.shape[1] % n_kv_heads or V.shape[1] % n_kv_heads:
        raise ValueError("K/V widths must be divisible by n_kv_heads")

    T = Q.shape[0]
    d_k = Q.shape[1] // n_heads
    if K.shape[1] // n_kv_heads != d_k:
        raise ValueError("Q and K head dimensions must match")
    d_v = V.shape[1] // n_kv_heads

    q = Q.reshape(T, n_heads, d_k).transpose(1, 0, 2)
    k = K.reshape(T, n_kv_heads, d_k).transpose(1, 0, 2)
    v = V.reshape(T, n_kv_heads, d_v).transpose(1, 0, 2)
    group_size = n_heads // n_kv_heads
    # repeat preserves groups [kv0, kv0, ..., kv1, kv1, ...].
    k = np.repeat(k, group_size, axis=0)
    v = np.repeat(v, group_size, axis=0)
    z = sdpa(q, k, v, mask)
    return z.transpose(1, 0, 2).reshape(T, n_heads * d_v)


def online_softmax_dot(
    logit_tiles: list[list[float]],
    value_tiles: list[list[float]],
) -> tuple[float, list[tuple[float, float, float]]]:
    """Stream scalar values through online softmax and return the update trace."""
    if len(logit_tiles) != len(value_tiles) or not logit_tiles:
        raise ValueError("logit_tiles and value_tiles must have the same non-zero length")
    m, normalizer, numerator = -np.inf, 0.0, 0.0
    trace: list[tuple[float, float, float]] = []
    for logits, values in zip(logit_tiles, value_tiles):
        if len(logits) != len(values) or not logits:
            raise ValueError("each logit tile must match a non-empty value tile")
        if not np.isfinite(logits).all() or not np.isfinite(values).all():
            raise ValueError("online-softmax inputs must be finite")
        m_new = max(m, max(logits))
        alpha = math.exp(m - m_new) if np.isfinite(m) else 0.0
        weights = [math.exp(x - m_new) for x in logits]
        normalizer = alpha * normalizer + sum(weights)
        numerator = alpha * numerator + sum(w * v for w, v in zip(weights, values))
        m = m_new
        trace.append((m, normalizer, numerator))
    return numerator / normalizer, trace


def tiled_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    tile_size: int = 128,
    sink: float | None = None,
    empty_row: EmptyRowPolicy = "error",
    causal: bool = False,
    window: int | None = None,
    query_start: int = 0,
) -> np.ndarray:
    """Streaming 2D attention that never materializes a T_q x T_k score matrix."""
    Q, K, V = (np.asarray(x) for x in (Q, K, V))
    if any(x.ndim != 2 or 0 in x.shape for x in (Q, K, V)):
        raise ValueError("tiled_attention expects Q/K/V to be 2D")
    if Q.shape[1] != K.shape[1] or K.shape[0] != V.shape[0]:
        raise ValueError("incompatible Q/K/V shapes")
    if isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if empty_row not in {"error", "zero"}:
        raise ValueError("empty_row must be 'error' or 'zero'")
    if not all(np.isfinite(x).all() for x in (Q, K, V)):
        raise ValueError("Q/K/V must be finite")
    if sink is not None and not np.isfinite(sink):
        raise ValueError("sink must be finite")

    # Accumulate at least in float32. This mirrors the direct reference and avoids
    # performing the dot product itself in float16 when learners try mixed precision.
    dtype = np.result_type(Q.dtype, K.dtype, V.dtype, np.float32)
    Q, K, V = (x.astype(dtype, copy=False) for x in (Q, K, V))
    T_q, d_k = Q.shape
    T_k, d_v = V.shape
    if window is not None and (not causal or isinstance(window, bool) or not isinstance(window, int) or window <= 0):
        raise ValueError("window requires causal=True and a positive integer")
    if isinstance(query_start, bool) or not isinstance(query_start, int) or query_start < 0:
        raise ValueError("query_start must be a nonnegative integer")
    if mask is not None:
        mask = np.asarray(mask, dtype=dtype)
        if mask.shape != (T_q, T_k):
            raise ValueError(f"mask must have shape {(T_q, T_k)}, got {mask.shape}")
        if np.isnan(mask).any() or np.isposinf(mask).any():
            raise ValueError("mask may not contain NaN/+inf")

    scale = np.asarray(1.0 / math.sqrt(d_k), dtype=dtype)
    out = np.zeros((T_q, d_v), dtype=dtype)
    for qi in range(T_q):
        # A sink starts with denominator mass 1 and zero value contribution.
        m = float(sink) if sink is not None else -np.inf
        normalizer = 1.0 if sink is not None else 0.0
        acc = np.zeros(d_v, dtype=out.dtype)

        for start in range(0, T_k, tile_size):
            stop = min(start + tile_size, T_k)
            with np.errstate(over="ignore", invalid="ignore"):
                logits = (Q[qi] @ K[start:stop].T) * scale
            if not np.isfinite(logits).all():
                raise FloatingPointError("Q @ K.T overflowed; rescale the inputs")
            mask_tile = 0.0 if mask is None else mask[qi, start:stop]
            if causal:
                distance = query_start + qi - np.arange(start, stop)
                allowed = distance >= 0
                if window is not None:
                    allowed &= distance < window
                mask_tile = mask_tile + np.where(allowed, 0.0, -np.inf)
            finite_bias = np.isfinite(mask_tile)
            with np.errstate(over="ignore", invalid="ignore"):
                logits = logits + mask_tile
            if np.any(finite_bias & ~np.isfinite(logits)):
                raise FloatingPointError(
                    "adding the finite mask bias overflowed; reduce the bias magnitude"
                )
            finite = np.isfinite(logits)
            if not finite.any():
                continue

            tile_max = float(logits[finite].max())
            m_new = max(m, tile_max)
            alpha = math.exp(m - m_new) if np.isfinite(m) else 0.0
            weights = np.where(finite, np.exp(logits - m_new), np.zeros_like(logits))
            normalizer = alpha * normalizer + float(weights.sum())
            acc = alpha * acc + weights @ V[start:stop]
            m = m_new

        if normalizer == 0.0:
            if empty_row == "error":
                raise ValueError('fully masked row in tiled_attention' + f": {qi}")
            continue
        out[qi] = acc / normalizer
    return out


def kv_bytes_per_token(
    layers: int,
    kv_heads: int,
    head_dim: int,
    bytes_per_value: int = 2,
) -> int:
    """Return K+V cache bytes per generated token across all layers."""
    values = (layers, kv_heads, head_dim, bytes_per_value)
    if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in values):
        raise ValueError("all KV-cache dimensions must be positive integers")
    return 2 * layers * kv_heads * head_dim * bytes_per_value
