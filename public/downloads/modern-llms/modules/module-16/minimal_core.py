"""Exact visual geometry and decoder-KV accounting; no implicit resize."""
from dataclasses import dataclass
from numbers import Integral


def positive_count(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("expected a positive integer")
    return int(value)


@dataclass(frozen=True)
class PatchLayout:
    grid_h: int
    grid_w: int
    merge: int
    visual_tokens: int


def patch_layout(height, width, *, patch=14, merge=1):
    height, width, patch, merge = map(positive_count, (height, width, patch, merge))
    if height % patch or width % patch:
        raise ValueError("image dimensions must be divisible by patch")
    gh, gw = height // patch, width // patch
    if gh % merge or gw % merge:
        raise ValueError("patch grid must be divisible by merge on both axes")
    return PatchLayout(gh, gw, merge, (gh // merge) * (gw // merge))


def kv_mib(tokens, *, layers=32, kv_heads=8, head_dim=128, bytes_per_value=2):
    tokens, layers, kv_heads, head_dim, bytes_per_value = map(
        positive_count, (tokens, layers, kv_heads, head_dim, bytes_per_value))
    return tokens * 2 * layers * kv_heads * head_dim * bytes_per_value / 2**20


def video_frames_fit(context=256_000, text_reserve=4_000, tokens_per_frame=576):
    context, tokens_per_frame = map(positive_count, (context, tokens_per_frame))
    if (isinstance(text_reserve, bool) or not isinstance(text_reserve, Integral)
            or not 0 <= text_reserve < context):
        raise ValueError("text_reserve must be an integer in [0, context)")
    return (context - int(text_reserve)) // tokens_per_frame
