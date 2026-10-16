"""Explicit precision policy for MiniLLM training and inference.

The educational runtime keeps model parameters in FP32 unless a caller deliberately
converts them. Mixed precision is a separate compute policy, and the KV-cache dtype
is measured from the tensors that are actually stored. Keeping these three notions
separate prevents a config label such as ``bfloat16`` from being mistaken for the
real storage and execution contract.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import inspect
from typing import Literal

import torch

ComputeDType = Literal["auto", "fp32", "bf16", "fp16"]


@dataclass(frozen=True)
class PrecisionPolicy:
    parameter_dtype: str
    compute_dtype: str
    cache_dtype: str = "runtime"
    autocast_enabled: bool = False
    grad_scaler_enabled: bool = False


def _device_type(device: str | torch.device) -> str:
    return torch.device(device).type


def native_cuda_bf16_supported() -> bool:
    """Return whether the active CUDA device has a native BF16 path.

    PyTorch exposes ``torch.cuda.is_bf16_supported``. Newer releases accept
    ``including_emulation=False``; older releases do not, so we fall back to the
    hardware capability check (Ampere, compute capability 8.x, or newer).
    """
    if not torch.cuda.is_available():
        return False
    fn = torch.cuda.is_bf16_supported
    try:
        if "including_emulation" in inspect.signature(fn).parameters:
            return bool(fn(including_emulation=False))
    except (TypeError, ValueError):
        pass
    major, _ = torch.cuda.get_device_capability()
    return bool(fn()) and major >= 8


def resolve_compute_dtype(device: str | torch.device, requested: ComputeDType = "auto") -> str:
    """Resolve ``auto`` to a hardware-supported compute dtype.

    * CPU: FP32 for portability and deterministic reference tests.
    * CUDA with native BF16: BF16.
    * Older CUDA devices such as T4: FP16 (with GradScaler during training).
    """
    kind = _device_type(device)
    if requested not in {"auto", "fp32", "bf16", "fp16"}:
        raise ValueError(f"unknown compute dtype: {requested!r}")
    if kind != "cuda":
        if requested not in {"auto", "fp32"}:
            raise ValueError(f"{requested} compute is supported by this capstone only on CUDA; device={kind}")
        return "fp32"
    if requested == "auto":
        return "bf16" if native_cuda_bf16_supported() else "fp16"
    if requested == "bf16" and not native_cuda_bf16_supported():
        raise RuntimeError("BF16 was requested, but the current CUDA device has no native BF16 path; use auto/fp16")
    return requested


def torch_dtype(name: str) -> torch.dtype:
    mapping = {"fp32": torch.float32, "float32": torch.float32,
               "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
               "fp16": torch.float16, "float16": torch.float16}
    if name not in mapping:
        raise ValueError(f"unknown dtype name: {name!r}")
    return mapping[name]


def dtype_name(dtype: torch.dtype | None) -> str:
    if dtype is None:
        return "empty"
    mapping = {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}
    return mapping.get(dtype, str(dtype).replace("torch.", ""))


def autocast_context(device: str | torch.device, compute_dtype: str):
    kind = _device_type(device)
    if kind == "cuda" and compute_dtype in {"bf16", "fp16"}:
        return torch.autocast("cuda", dtype=torch_dtype(compute_dtype))
    return nullcontext()


def make_grad_scaler(device: str | torch.device, compute_dtype: str):
    enabled = _device_type(device) == "cuda" and compute_dtype == "fp16"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # compatibility with older PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def model_parameter_dtype(model: torch.nn.Module) -> str:
    first = next(model.parameters(), None)
    return dtype_name(None if first is None else first.dtype)
