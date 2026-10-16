import math
import numpy as np


def stable_sigmoid(x):
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -x))


def dpo_loss(policy_lp_w, policy_lp_l, ref_lp_w, ref_lp_l, beta=0.1):
    """DPO from completion-token log-probability sums."""
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")

    pw, pl, rw, rl = np.broadcast_arrays(
        np.asarray(policy_lp_w, dtype=float),
        np.asarray(policy_lp_l, dtype=float),
        np.asarray(ref_lp_w, dtype=float),
        np.asarray(ref_lp_l, dtype=float),
    )
    if not all(np.all(np.isfinite(v)) for v in (pw, pl, rw, rl)):
        raise ValueError("log probabilities must be finite")

    z = beta * ((pw - rw) - (pl - rl))
    loss = np.logaddexp(0.0, -z)
    local_slope = stable_sigmoid(-z)
    return loss, local_slope, z > 0


def grpo_advantages(rewards, *, on_degenerate="raise"):
    """Population normalization with explicit zero-variance handling."""
    r = np.asarray(rewards, dtype=np.float64)
    if r.ndim != 1 or r.size < 2 or not np.all(np.isfinite(r)):
        raise ValueError("rewards must be a finite one-dimensional group of size >= 2")

    if on_degenerate not in ("raise", "zeros"):
        raise ValueError("on_degenerate must be raise or zeros")
    std = float(r.std(ddof=0))
    if std == 0.0:
        if on_degenerate == "zeros":
            return np.zeros_like(r)
        raise ValueError("the group contains no relative signal")
    return (r - r.mean()) / std


def stage_memory_gib(n_params, stack):
    """Teaching estimate of persistent model states, excluding activations."""
    if not math.isfinite(n_params) or n_params <= 0:
        raise ValueError("n_params must be finite and positive")
    bytes_per_parameter = {"train": 16, "frozen_bf16": 2}
    return sum(bytes_per_parameter[kind] * n_params for kind in stack) / 2**30


ppo = stage_memory_gib(
    8.03e9,
    ["train", "train", "frozen_bf16", "frozen_bf16"],
)
dpo = stage_memory_gib(8.03e9, ["train", "frozen_bf16"])
