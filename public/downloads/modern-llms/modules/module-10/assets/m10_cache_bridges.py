"""Modern LLMs module 10: executable NumPy → PyTorch bridges (B17).

This file is generated from the canonical ``bridges/`` package.  It contains
all learner-facing implementations shown in the lecture and their parity
checks.  Run it directly with Python.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F


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


if __name__ == "__main__":
    np_cache = NumpyKVCacheExplicit(window=3); t_cache = TorchKVCacheExplicit(window=3)
    for value in range(5):
        k_np = np.full((1,1,1,2), value, dtype=np.float64); v_np = k_np + 100
        np_cache.append(k_np, v_np); t_cache.append(torch.tensor(k_np), torch.tensor(v_np))
    np.testing.assert_allclose(np_cache.k, t_cache.k.numpy())
    np.testing.assert_array_equal(np_cache.positions, t_cache.positions.numpy())
    assert (np_cache.start_pos, np_cache.next_pos) == (2,5)
    np_cache.truncate(4); t_cache.truncate(4)
    np.testing.assert_allclose(np_cache.k, t_cache.k.numpy())
    rng = np.random.default_rng(17)
    keys = rng.standard_normal((1,1,6,4)); values = rng.standard_normal((1,1,6,4)); queries = rng.standard_normal((1,1,6,4))
    cache = TorchKVCacheExplicit(window=3)
    for step in range(6):
        cache.append(torch.tensor(keys[..., step:step+1, :]), torch.tensor(values[..., step:step+1, :]))
        cached = F.scaled_dot_product_attention(torch.tensor(queries[..., step:step+1, :]), cache.k, cache.v)
        lo = max(0, step - 2)
        full = F.scaled_dot_product_attention(torch.tensor(queries[..., step:step+1, :]), torch.tensor(keys[..., lo:step+1, :]), torch.tensor(values[..., lo:step+1, :]))
        torch.testing.assert_close(cached, full)

    # Rejected input must not mutate state; rollback cannot resurrect evicted tokens.
    for cache_type, convert in ((NumpyKVCacheExplicit, np.asarray), (TorchKVCacheExplicit, torch.tensor)):
        c = cache_type(window=3)
        c.truncate(0)
        x = convert(np.ones((1,1,5,2), dtype=np.float64))
        c.append(x,x)
        assert (c.start_pos,c.next_pos) == (2,5)
        for bad in (-1,1,6,2.5,True):
            try: c.truncate(bad)
            except ValueError: pass
            else: raise AssertionError("invalid rollback accepted")
        assert (c.start_pos,c.next_pos) == (2,5)
        try: c.append(convert(np.zeros((1,2,1,2))),convert(np.zeros((1,2,1,2))))
        except ValueError: pass
        else: raise AssertionError("geometry change accepted")
        c.truncate(2)
        assert c.k.shape[-2] == 0
        c.append(convert(np.ones((1,1,1,2))),convert(np.ones((1,1,1,2))))
        assert (c.start_pos,c.next_pos) == (2,3)
        for window in (0,-1,2.5,True):
            try: cache_type(window=window)
            except ValueError: pass
            else: raise AssertionError("invalid window accepted")
    a = NumpyKVCacheExplicit(window=3)
    a.append(np.ones((1,1,10,2)),np.ones((1,1,10,2)))
    assert a.k.flags.owndata and a.k.nbytes == 3*2*8
    t = TorchKVCacheExplicit(window=3)
    t.append(torch.ones(1,1,10,2),torch.ones(1,1,10,2))
    assert t.k.untyped_storage().nbytes() == t.k.numel()*t.k.element_size()

    print("B17 window, ownership, rollback, and invalid inputs: PASS")
    print("Module 10 bridge asset: PASS")
