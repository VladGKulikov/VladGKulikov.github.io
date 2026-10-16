"""Canonical minimal split-half RoPE and YaRN listings for both lectures."""
import math
import numpy as np


def rope_inv_freq(head_dim: int, base: float = 150_000.0) -> np.ndarray:
    """Return theta_i = base**(-2i/D) for D/2 rotary pairs."""
    if isinstance(head_dim, bool) or not isinstance(head_dim, int):
        raise TypeError("head_dim must be an integer")
    if head_dim < 2 or head_dim % 2:
        raise ValueError("head_dim must be positive and even")
    if not math.isfinite(base) or base <= 1.0:
        raise ValueError("base must be finite and greater than 1")
    i = np.arange(head_dim // 2, dtype=np.float64)
    return base ** (-2.0 * i / head_dim)


def rotate_half(x: np.ndarray) -> np.ndarray:
    """Quarter-turn the split-half coordinate pairs."""
    if x.ndim < 2 or x.size == 0 or x.shape[-1] % 2:
        raise ValueError("x must have shape [..., T, D] with even D")
    half = x.shape[-1] // 2
    return np.concatenate((-x[..., half:], x[..., :half]), axis=-1)


def apply_rope(q, k, positions, inv_freq):
    """Apply split-half RoPE to q/k of shape [..., T, D]."""
    if any(np.iscomplexobj(x) for x in (q, k, inv_freq)):
        raise TypeError("q, k and inv_freq must be real")
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    positions = np.asarray(positions)
    inv = np.asarray(inv_freq, dtype=np.float64)
    if q.shape != k.shape or q.ndim < 2 or q.size == 0 or q.shape[-1] % 2:
        raise ValueError("q and k must share shape [..., T, D], with even D")
    if not np.isfinite(q).all() or not np.isfinite(k).all():
        raise ValueError("q and k must be finite")
    if positions.ndim != 1 or positions.dtype.kind not in "iu":
        raise ValueError("positions must be an integer vector [T]")
    if positions.size != q.shape[-2] or np.any(positions < 0):
        raise ValueError("positions do not match the sequence axis")
    if inv.shape != (q.shape[-1] // 2,) or not np.isfinite(inv).all() or np.any(inv <= 0):
        raise ValueError("inv_freq must be positive and have shape [D/2]")

    angles = positions.astype(np.float64)[:, None] * inv[None, :]
    angles = angles.reshape((1,) * (q.ndim - 2) + angles.shape)
    if not np.isfinite(angles).all():
        raise FloatingPointError("phase overflow")
    full = np.concatenate((angles, angles), axis=-1)
    cos, sin = np.cos(full), np.sin(full)
    qr, kr = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
    if not np.isfinite(qr).all() or not np.isfinite(kr).all():
        raise FloatingPointError("rotation overflow")
    return qr, kr


def yarn_inv_freq(head_dim, base, factor, original_context,
                  beta_fast=32.0, beta_slow=1.0, *, truncate=False):
    """Blend original and interpolated frequencies with a YaRN-style ramp."""
    inv = rope_inv_freq(head_dim, base)
    if not math.isfinite(factor) or factor < 1.0:
        raise ValueError("factor must be at least 1")
    if isinstance(original_context, bool) or not isinstance(original_context, int):
        raise TypeError("original_context must be an integer")
    if not math.isfinite(beta_fast) or not math.isfinite(beta_slow):
        raise ValueError("beta parameters must be finite")
    if original_context <= 0 or beta_fast <= beta_slow or beta_slow <= 0:
        raise ValueError("invalid context or beta parameters")
    if not isinstance(truncate, bool):
        raise TypeError("truncate must be bool")

    def correction_dim(turns: float) -> float:
        return head_dim * math.log(
            original_context / (turns * 2 * math.pi)
        ) / (2 * math.log(base))

    pair_count = head_dim // 2
    low_raw = correction_dim(beta_fast)
    high_raw = correction_dim(beta_slow)
    low = math.floor(low_raw) if truncate else low_raw
    high = math.ceil(high_raw) if truncate else high_raw
    low = min(max(float(low), 0.0), float(pair_count - 1))
    high = min(max(float(high), 0.0), float(pair_count - 1))
    if high <= low:
        raise ValueError("the YaRN ramp collapsed")

    ramp = np.clip(
        (np.arange(pair_count, dtype=np.float64) - low) / (high - low),
        0.0,
        1.0,
    )
    scaled = inv * (1.0 - ramp) + (inv / factor) * ramp
    attention_factor = 1.0 if factor <= 1 else 1.0 + 0.1 * math.log(factor)
    return scaled, attention_factor, attention_factor**2, low, high
