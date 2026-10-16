"""Canonical NumPy → PyTorch bridge implementations for Modern LLMs.

NumPy functions expose readable mathematical references; PyTorch functions
expose tensor, dtype, device, autograd, and backend contracts. Lecture blocks,
module assets, and learner-notebook cells are generated from this source tree.
"""
from .attention import (
    b04_parity_demo,
    numpy_gqa,
    numpy_gqa_explicit,
    numpy_sdpa,
    numpy_sdpa_explicit,
    torch_gqa,
    torch_gqa_explicit,
    torch_gqa_optimized,
    torch_sdpa,
    torch_sdpa_explicit,
    torch_sdpa_optimized,
)
from .cache import NumpyKVCache, NumpyKVCacheExplicit, TorchKVCache, TorchKVCacheExplicit
from .moe import (
    numpy_expert_dispatch,
    numpy_expert_dispatch_explicit,
    numpy_topk_router,
    numpy_topk_router_explicit,
    torch_expert_dispatch,
    torch_expert_dispatch_explicit,
    torch_topk_router,
    torch_topk_router_explicit,
)
from .normalization import (
    numpy_rms_norm,
    numpy_swiglu,
    torch_rms_norm,
    torch_rms_norm_explicit,
    torch_swiglu,
)
from .quantization import (
    numpy_grouped_fake_quant,
    numpy_grouped_fake_quant_explicit,
    torch_grouped_fake_quant,
    torch_grouped_fake_quant_explicit,
)
from .rope import numpy_rope, numpy_rope_explicit, torch_rope, torch_rope_explicit
from .sft import (
    numpy_assistant_labels,
    numpy_block_causal_mask,
    numpy_segment_position_ids,
    torch_assistant_labels,
    torch_block_causal_mask,
    torch_segment_position_ids,
)
from .speculative import (
    numpy_rejection_kernel,
    numpy_rejection_kernel_explicit,
    torch_rejection_kernel,
    torch_rejection_kernel_explicit,
    torch_sample_speculative_token,
    torch_sample_speculative_token_explicit,
)

__all__ = [name for name in globals() if name.startswith(("numpy_", "torch_", "Numpy", "Torch", "b04_"))]
