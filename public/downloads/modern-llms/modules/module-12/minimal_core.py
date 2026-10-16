"""M12: summed masked loss and deterministic reveal schedule."""
import numpy as np


def linear_mask_schedule(length: int, steps: int) -> tuple[int, ...]:
    """Target counts from fully masked to fully revealed."""
    if any(isinstance(v, (bool, np.bool_)) or
           not isinstance(v, (int, np.integer)) or v <= 0
           for v in (length, steps)):
        raise ValueError("length and steps must be positive")
    return tuple(
        (int(length) * remaining + int(steps) - 1) // int(steps)
        for remaining in range(steps, -1, -1)
    )


def dllm_loss(logits, target_ids, loss_mask, time: float) -> float:
    """(1/t) times the summed NLL on explicitly masked positions."""
    logits = np.asarray(logits, dtype=np.float64)
    target_ids = np.asarray(target_ids)
    loss_mask = np.asarray(loss_mask)
    if logits.ndim != 2 or target_ids.shape != (logits.shape[0],):
        raise ValueError("expected logits [T,V] and target_ids [T]")
    if logits.shape[1] == 0 or target_ids.dtype.kind not in "iu":
        raise ValueError("target_ids must be integers and vocabulary non-empty")
    if np.any(target_ids < 0) or np.any(target_ids >= logits.shape[1]):
        raise ValueError("target IDs must be in [0, V)")
    if loss_mask.shape != target_ids.shape or loss_mask.dtype.kind != "b":
        raise ValueError("loss_mask must be boolean [T]")
    if isinstance(time, (bool, np.bool_)) or not (0.0 < time <= 1.0):
        raise ValueError("time must be in (0,1]")
    if not np.isfinite(logits).all():
        raise ValueError("logits must be finite")
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    nll = -log_probs[np.arange(target_ids.size), target_ids]
    return float(nll[loss_mask].sum() / time)


def sample_dllm(predict, length: int, steps: int, mask_id: int):
    """Greedy confidence reveal with one model invocation per iteration."""
    schedule = linear_mask_schedule(length, steps)
    if isinstance(mask_id, (bool, np.bool_)) or not isinstance(mask_id, (int, np.integer)):
        raise ValueError("mask_id must be an integer")
    vocabulary = None
    tokens = np.full(length, mask_id, dtype=np.int64)
    trace = []
    for step, target_masked in enumerate(schedule[1:], start=1):
        logits = np.asarray(predict(tokens.copy(), step), dtype=np.float64)
        if logits.ndim != 2 or logits.shape[0] != length or logits.shape[1] == 0 or not np.isfinite(logits).all():
            raise ValueError("predict must return finite logits [T,V]")
        if vocabulary is not None and logits.shape[1] != vocabulary:
            raise ValueError("predict changed vocabulary size")
        vocabulary = logits.shape[1]
        if 0 <= mask_id < vocabulary:
            raise ValueError("mask_id must lie outside the vocabulary")
        shifted = logits - logits.max(axis=-1, keepdims=True)
        probs = np.exp(shifted)
        probs /= probs.sum(axis=-1, keepdims=True)
        candidates = probs.argmax(axis=-1)
        confidence = probs[np.arange(length), candidates]
        masked = np.flatnonzero(tokens == mask_id)
        open_count = masked.size - target_masked
        order = np.lexsort((masked, -confidence[masked]))
        opened = masked[order[:open_count]]
        tokens[opened] = candidates[opened]
        trace.append((tuple(opened), tokens.copy()))
    return tokens, trace
