"""Minimal M8 operators; used by the validated reference and embedded in notebooks."""
import numpy as np

def zoh_diagonal(a, b, delta):
    """Diagonal A: [n], B: [n,p], positive scalar step; caller validates shapes."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    transition = np.exp(delta * a)
    factor = np.full_like(a, float(delta))
    np.divide(np.expm1(delta * a), a, out=factor, where=a != 0)
    return transition, factor[:, None] * b

def delta_step(state, key, value, beta, decay=1.0):
    """State [dv,dk], unit key [dk], value [dv]; decay before correction."""
    base = decay * state
    prediction = base @ key
    return base + beta * np.outer(value - prediction, key)

def compose_affine_pair(later, earlier):
    """Elementwise transforms; returns later(earlier(h)), not the reverse."""
    a2, b2 = later
    a1, b1 = earlier
    return a2 * a1, a2 * b1 + b2

