"""Minimal mechanisms embedded verbatim in the lecture and notebook."""
from dataclasses import dataclass
import math
from numbers import Real
import numpy as np


def positive(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


@dataclass(frozen=True)
class ScalingLaw:
    E: float = 1.69
    A: float = 406.4
    B: float = 410.7
    alpha: float = 0.34
    beta: float = 0.28
    kappa: float = 6.0

    def __post_init__(self):
        for name in ("A", "B", "alpha", "beta", "kappa"):
            object.__setattr__(self, name, positive(name, getattr(self, name)))
        if isinstance(self.E, (bool, np.bool_)) or not isinstance(self.E, Real):
            raise ValueError("E must be a real number")
        if not math.isfinite(self.E) or self.E < 0:
            raise ValueError("E must be finite and non-negative")
        object.__setattr__(self, "E", float(self.E))

    def loss(self, N, D):
        N, D = positive("N", N), positive("D", D)
        return self.E + self.A * N**(-self.alpha) + self.B * D**(-self.beta)

    def optimal_split(self, C):
        C = positive("C", C)
        N = (
            (self.alpha * self.A) / (self.beta * self.B)
            * (C / self.kappa) ** self.beta
        ) ** (1 / (self.alpha + self.beta))
        return N, C / (self.kappa * N)


def newton_schulz(G, quintic_steps=5):
    """Trace a rectangular polar-factor approximation."""
    X = np.asarray(G, dtype=np.float64)
    if X.ndim != 2 or 0 in X.shape or not np.isfinite(X).all():
        raise ValueError("G must be a finite non-empty matrix")
    if isinstance(quintic_steps, bool) or not isinstance(quintic_steps, int) or quintic_steps < 0:
        raise ValueError("quintic_steps must be a non-negative integer")
    scale = np.max(np.abs(X))
    if scale == 0:
        raise ValueError("G must not be all zeros")
    transposed = X.shape[0] > X.shape[1]
    X = (X.T if transposed else X) / scale
    X /= np.linalg.norm(X, ord="fro")
    for _ in range(quintic_steps):
        XXt = X @ X.T
        X = 3.4445 * X - 4.7750 * XXt @ X + 2.0315 * XXt @ XXt @ X
    return X.T if transposed else X


def qk_clip_scales(max_logits, tau=100.0, split=0.5):
    """Return gamma and Q/K scales for one monitoring step."""
    maxima = np.asarray(max_logits, dtype=np.float64)
    if maxima.ndim != 1 or maxima.size == 0 or not np.isfinite(maxima).all():
        raise ValueError("max_logits must be a finite vector")
    if np.any(maxima < 0) or not 0 <= split <= 1:
        raise ValueError("maxima must be non-negative and split must lie in [0, 1]")
    tau = positive("tau", tau)
    gamma = np.ones_like(maxima)
    over = maxima > tau
    gamma[over] = tau / maxima[over]
    q_scale = gamma**split
    k_scale = gamma**(1 - split)
    return gamma, q_scale, k_scale, maxima * q_scale * k_scale
