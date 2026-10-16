"""B17: bounded inference-only KV containers with absolute positions."""
from __future__ import annotations
import numpy as np
import torch

class NumpyKVCacheExplicit:
    """Owned [B,H,T,D] floating state; truncation cannot recover evicted tokens."""

    def __init__(self, *, window: int | None = None) -> None:
        if window is not None and (isinstance(window, bool) or not isinstance(window, int) or window <= 0):
            raise ValueError("window must be a positive integer or None")
        self.window = window
        self.k = self.v = None
        self.start_pos = self.next_pos = 0

    def append(self, k_new, v_new):
        k_new, v_new = np.asarray(k_new), np.asarray(v_new)
        if k_new.ndim != 4 or k_new.shape != v_new.shape or min(k_new.shape) <= 0:
            raise ValueError("K and V need the same non-empty [B,H,T,D] shape")
        if k_new.dtype != v_new.dtype or not np.issubdtype(k_new.dtype, np.floating):
            raise ValueError("K and V need the same floating dtype")
        if self.k is not None:
            if self.k.shape[:2] + self.k.shape[3:] != k_new.shape[:2] + k_new.shape[3:]:
                raise ValueError("cache geometry changed")
            if self.k.dtype != k_new.dtype:
                raise ValueError("cache dtype changed")
        # Validate first; keep old state intact if an input is rejected.
        k = k_new.copy() if self.k is None else np.concatenate([self.k, k_new], axis=-2)
        v = v_new.copy() if self.v is None else np.concatenate([self.v, v_new], axis=-2)
        drop = max(0, k.shape[-2] - self.window) if self.window is not None else 0
        # A slice alone retains its backing allocation. Make owned bounded copies.
        self.k = k[..., drop:, :].copy() if drop else k
        self.v = v[..., drop:, :].copy() if drop else v
        self.start_pos += drop
        self.next_pos += k_new.shape[-2]
        return self.k, self.v

    def truncate(self, absolute_end: int) -> None:
        if isinstance(absolute_end, bool) or not isinstance(absolute_end, int):
            raise ValueError("absolute_end must be an integer")
        if not self.start_pos <= absolute_end <= self.next_pos:
            raise ValueError("rollback lies outside the retained interval")
        keep = absolute_end - self.start_pos
        if self.k is not None:
            self.k = self.k[..., :keep, :].copy()
            self.v = self.v[..., :keep, :].copy()
        self.next_pos = absolute_end

    @property
    def positions(self):
        return np.arange(self.start_pos, self.next_pos, dtype=np.int64)


class TorchKVCacheExplicit:
    """Owned [B,H,T,D] floating state; truncation cannot recover evicted tokens."""

    def __init__(self, *, window: int | None = None) -> None:
        if window is not None and (isinstance(window, bool) or not isinstance(window, int) or window <= 0):
            raise ValueError("window must be a positive integer or None")
        self.window = window
        self.k = self.v = None
        self.start_pos = self.next_pos = 0

    def append(self, k_new, v_new):
        if not isinstance(k_new, torch.Tensor) or not isinstance(v_new, torch.Tensor):
            raise TypeError("K and V must be tensors")
        if k_new.ndim != 4 or k_new.shape != v_new.shape or min(k_new.shape) <= 0:
            raise ValueError("K and V need the same non-empty [B,H,T,D] shape")
        if k_new.dtype != v_new.dtype or k_new.device != v_new.device or not k_new.is_floating_point():
            raise ValueError("K and V need the same floating dtype and device")
        if k_new.requires_grad or v_new.requires_grad:
            raise ValueError("this cache is for inference; detach inputs first")
        if self.k is not None:
            if self.k.shape[:2] + self.k.shape[3:] != k_new.shape[:2] + k_new.shape[3:]:
                raise ValueError("cache geometry changed")
            if self.k.dtype != k_new.dtype or self.k.device != k_new.device:
                raise ValueError("cache dtype or device changed")
        # Validate first; keep old state intact if an input is rejected.
        k = k_new.clone() if self.k is None else torch.cat([self.k, k_new], dim=-2)
        v = v_new.clone() if self.v is None else torch.cat([self.v, v_new], dim=-2)
        drop = max(0, k.shape[-2] - self.window) if self.window is not None else 0
        # A slice alone retains its backing allocation. Make owned bounded copies.
        self.k = k[..., drop:, :].clone() if drop else k
        self.v = v[..., drop:, :].clone() if drop else v
        self.start_pos += drop
        self.next_pos += k_new.shape[-2]
        return self.k, self.v

    def truncate(self, absolute_end: int) -> None:
        if isinstance(absolute_end, bool) or not isinstance(absolute_end, int):
            raise ValueError("absolute_end must be an integer")
        if not self.start_pos <= absolute_end <= self.next_pos:
            raise ValueError("rollback lies outside the retained interval")
        keep = absolute_end - self.start_pos
        if self.k is not None:
            self.k = self.k[..., :keep, :].clone()
            self.v = self.v[..., :keep, :].clone()
        self.next_pos = absolute_end

    @property
    def positions(self):
        device = self.k.device if self.k is not None else None
        return torch.arange(self.start_pos, self.next_pos, dtype=torch.long, device=device)


class NumpyKVCache(NumpyKVCacheExplicit):
    """The same validated contract for reference callers."""


class TorchKVCache(TorchKVCacheExplicit):
    """The same validated contract for reference callers."""
