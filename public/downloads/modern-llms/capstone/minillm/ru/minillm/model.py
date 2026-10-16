"""MiniLLM — decoder-only reference stack курса, собранный из блоков курса.

Что здесь реализовано и где это объяснено:

  RMSNorm с fp32-статистикой ........... M6 §5
  pre / post / Peri-LN / OLMo2 / none .......... M6 §6
  QK-norm .............................. M6 §7
  SwiGLU / GeGLU, правило 2/3, кламп ... M6 §8–9
  RoPE: две конвенции, linear, YaRN .... M4 §4–7
  GQA / MQA / MHA ...................... M5 §3
  attention sinks ...................... M5 §12
  sliding ↔ full по layer_types ........ M5 §11
  MoE: sigmoid + aux-loss-free bias .... M7 §6–8
  KV-кэш ............................... M10 §4–5

Уровень кода: reference implementation. Он корректен, документирован и обучает
модель по-настоящему, но это не движок инференса: внимание и MoE написаны так,
чтобы механизм было видно, а не чтобы выжать пропускную способность. Места, где
за читаемость заплачено скоростью, помечены в комментариях явно.

Контракт форм по всему файлу:
    B — batch, T — длина запроса, S — длина ключей (T + длина кэша),
    H — num_attention_heads, Hkv — num_key_value_heads, dh — head_dim,
    d — hidden_size, N — B·T (токены, «расплющенные» для роутера MoE).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

from .config import MiniLLMConfig


# =============================================================================
# Нормализация (M6 §5)
# =============================================================================
class RMSNorm(nn.Module):
    """RMSNorm со статистикой минимум в float32.

    Апкаст — не педантизм: при bf16-обучении квадраты малых чисел в half-точности
    теряются, и норма превращается в усилитель шума. Стоит копейки, убирает целый
    класс нестабильностей; так устроены многие референсные реализации, рассмотренные в модуле 6.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., d] -> [..., d]
        dtype = x.dtype
        x32 = x.float()
        rms = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * rms).to(dtype) * self.weight


# =============================================================================
# Позиционные кодировки (M4)
# =============================================================================
def _yarn_correction_dim(num_rotations: float, dim: int, base: float, max_pos: int) -> float:
    """Индекс измерения, совершающего ровно `num_rotations` оборотов на max_pos."""
    return dim * math.log(max_pos / (num_rotations * 2 * math.pi)) / (2 * math.log(base))


def _yarn_correction_range(beta_fast: float, beta_slow: float, dim: int,
                           base: float, max_pos: int) -> tuple[int, int]:
    low = math.floor(_yarn_correction_dim(beta_fast, dim, base, max_pos))
    high = math.ceil(_yarn_correction_dim(beta_slow, dim, base, max_pos))
    return max(low, 0), min(high, dim - 1)


def _yarn_ramp(low: float, high: float, dim: int) -> torch.Tensor:
    if low == high:
        high = high + 1e-3  # защита от деления на ноль при вырожденном диапазоне
    ramp = (torch.arange(dim, dtype=torch.float32) - low) / (high - low)
    return ramp.clamp(0.0, 1.0)


class RotaryEmbedding(nn.Module):
    """RoPE с двумя конвенциями спаривания и тремя режимами растяжки.

    Конвенции спаривания несовместимы между собой: веса, обученные с `adjacent`,
    нельзя инферить с `rotate_half` и наоборот (M4 §5). Флаг существует именно
    затем, чтобы студент это увидел, а не прочитал.

    Растяжка: None — чистая RoPE; linear (Position Interpolation) — деление всех
    частот на factor; yarn — NTK-by-parts, где высокочастотные измерения остаются
    экстраполяцией, а низкочастотные интерполируются, плюс поправка mscale.
    """

    def __init__(self, cfg: MiniLLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        dim = cfg.head_dim
        if dim % 2:
            raise ValueError("head_dim должен быть чётным для RoPE")

        pos_freqs = cfg.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        inv_freq = 1.0 / pos_freqs
        attention_scale = 1.0

        scaling = cfg.rope_scaling or {}
        rope_type = scaling.get("rope_type")
        if rope_type == "linear":
            inv_freq = inv_freq / float(scaling["factor"])
        elif rope_type == "yarn":
            factor = float(scaling["factor"])
            orig_max = int(scaling.get("original_max_position_embeddings",
                                       cfg.max_position_embeddings))
            beta_fast = float(scaling.get("beta_fast", 32.0))
            beta_slow = float(scaling.get("beta_slow", 1.0))
            inv_extra = inv_freq                      # экстраполяция: частоты как есть
            inv_inter = inv_freq / factor             # интерполяция: сжатые частоты
            low, high = _yarn_correction_range(beta_fast, beta_slow, dim,
                                               cfg.rope_theta, orig_max)
            extra_factor = 1.0 - _yarn_ramp(low, high, dim // 2)
            inv_freq = inv_inter * (1 - extra_factor) + inv_extra * extra_factor
            # mscale компенсирует падение «температуры» логитов при растяжке.
            attention_scale = 0.1 * math.log(factor) + 1.0
        elif rope_type is not None:
            raise ValueError(f"неизвестный rope_type: {rope_type!r}")

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scale = attention_scale

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build rotary factors for positions shaped ``[T]`` or ``[B, T]``."""
        freqs = positions.float()[..., None] * self.inv_freq
        if self.cfg.rope_pairing == "rotate_half":
            freqs = torch.cat([freqs, freqs], dim=-1)
        cos = freqs.cos() * self.attention_scale
        sin = freqs.sin() * self.attention_scale
        return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               pairing: str) -> torch.Tensor:
    """x [B, H, T, dh] -> повёрнутый тензор той же формы."""
    if cos.ndim == 2:                         # [T, D]
        cos, sin = cos[None, None], sin[None, None]
    elif cos.ndim == 3:                       # [B, T, D]
        cos, sin = cos[:, None], sin[:, None]
    else:
        raise ValueError(f"ожидались RoPE factors [T,D] или [B,T,D], получено {tuple(cos.shape)}")
    if pairing == "rotate_half":
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin
    if pairing == "adjacent":
        x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)             # [B,H,T,dh/2,2]
        x0, x1 = x_pairs[..., 0], x_pairs[..., 1]
        out0 = x0 * cos - x1 * sin
        out1 = x1 * cos + x0 * sin
        return torch.stack([out0, out1], dim=-1).flatten(-2).to(x.dtype)
    raise ValueError(f"неизвестная конвенция спаривания: {pairing!r}")


# =============================================================================
# KV-кэш (M10)
# =============================================================================
@dataclass
class LayerCache:
    key: torch.Tensor | None = None    # [B, Hkv, S, dh]
    value: torch.Tensor | None = None
    start_pos: int = 0                 # абсолютная позиция первого сохранённого token


class KVCache:
    """Per-layer KV-cache with optional bounded sliding windows.

    ``position`` is the absolute number of tokens already processed. Full-attention
    layers retain the complete prefix; sliding layers retain only their configured
    suffix. This makes the memory contract match the attention mask instead of using
    a grow-only cache for every layer.
    """

    def __init__(self, num_layers: int, windows: list[int | None] | None = None) -> None:
        self.layers = [LayerCache() for _ in range(num_layers)]
        self.windows = list(windows or [None] * num_layers)
        if len(self.windows) != num_layers:
            raise ValueError("windows length must match num_layers")
        self.position = 0

    def __len__(self) -> int:
        return self.position

    def update(self, index: int, key: torch.Tensor, value: torch.Tensor):
        entry = self.layers[index]
        if entry.key is None:
            entry.key, entry.value = key, value
            entry.start_pos = self.position
        else:
            entry.key = torch.cat([entry.key, key], dim=-2)
            entry.value = torch.cat([entry.value, value], dim=-2)
        window = self.windows[index]
        if window is not None and entry.key.shape[-2] > window:
            trim = entry.key.shape[-2] - window
            entry.key = entry.key[..., trim:, :]
            entry.value = entry.value[..., trim:, :]
            entry.start_pos += trim
        return entry.key, entry.value, entry.start_pos

    def advance(self, tokens: int) -> None:
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        self.position += tokens

    def truncate(self, length: int) -> None:
        """Rollback the absolute sequence length, respecting bounded layer windows."""
        if length < 0 or length > self.position:
            raise ValueError(f"invalid cache truncation {length} for position {self.position}")
        for entry in self.layers:
            if entry.key is None:
                continue
            keep = max(0, min(entry.key.shape[-2], length - entry.start_pos))
            if keep == 0:
                entry.key = entry.value = None
                entry.start_pos = length
            else:
                entry.key = entry.key[..., :keep, :]
                entry.value = entry.value[..., :keep, :]
        self.position = length

    def clone(self) -> "KVCache":
        copy = KVCache(len(self.layers), self.windows)
        copy.position = self.position
        for dst, src in zip(copy.layers, self.layers):
            dst.start_pos = src.start_pos
            if src.key is not None:
                dst.key, dst.value = src.key.clone(), src.value.clone()
        return copy

    def nbytes(self) -> int:
        return sum(e.key.numel() * e.key.element_size() * 2
                   for e in self.layers if e.key is not None)

    def dtype(self) -> torch.dtype | None:
        for entry in self.layers:
            if entry.key is not None:
                return entry.key.dtype
        return None


# =============================================================================
# Внимание (M5)
# =============================================================================
def repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    """[B, Hkv, S, dh] -> [B, Hkv·repeats, S, dh]. Дублирование, а не копия смысла:
    GQA экономит именно память кэша, а не арифметику внимания (M5 §3)."""
    if repeats == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, repeats, s, d).reshape(b, h * repeats, s, d)


class Attention(nn.Module):
    """GQA-внимание с опциональными sinks, QK-norm и скользящим окном."""

    def __init__(self, cfg: MiniLLMConfig, layer_index: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_index = layer_index
        self.window = cfg.sliding_window if cfg.layer_types[layer_index] == "sliding" else None
        self.groups = cfg.num_attention_heads // cfg.num_key_value_heads
        self.scale = cfg.head_dim ** -0.5

        self.q_proj = nn.Linear(cfg.hidden_size, cfg.q_dim, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.kv_dim, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.kv_dim, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(cfg.q_dim, cfg.hidden_size, bias=False)

        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps) if cfg.qk_norm else None
        # Sink — обучаемый логит «никуда»: даёт softmax законный способ не смотреть
        # ни на один токен, вместо того чтобы сливать массу в первый (M5 §12).
        self.sinks = nn.Parameter(torch.zeros(cfg.num_attention_heads)) if cfg.attention_sinks else None

    def _causal_mask(self, query_positions: torch.Tensor, key_start: int,
                     key_count: int, segment_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Return a bool *allowed* mask, shape ``[T,S]`` or ``[B,1,T,S]``."""
        if query_positions.ndim == 2:
            q_pos = query_positions[..., None]
        else:
            q_pos = query_positions[:, None]
        k_pos = torch.arange(key_start, key_start + key_count,
                             device=query_positions.device)
        allowed = k_pos <= q_pos
        if self.window is not None:
            allowed = allowed & (q_pos - k_pos < self.window)
        if segment_ids is not None:
            if key_start != 0 or key_count != segment_ids.shape[1]:
                raise ValueError("segment isolation is supported only for uncached packed training")
            same = segment_ids[:, :, None] == segment_ids[:, None, :]
            valid = (segment_ids[:, :, None] >= 0) & (segment_ids[:, None, :] >= 0)
            allowed = allowed[:, None] & same[:, None] & valid[:, None]
        return allowed

    def forward(self, x, cos, sin, positions: torch.Tensor,
                cache: KVCache | None = None,
                segment_ids: torch.Tensor | None = None):
        B, T, _ = x.shape
        cfg = self.cfg

        q = self.q_proj(x).view(B, T, cfg.num_attention_heads, cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)

        q = apply_rope(q, cos, sin, cfg.rope_pairing)
        k = apply_rope(k, cos, sin, cfg.rope_pairing)

        key_start = 0
        if cache is not None:
            k, v, key_start = cache.update(self.layer_index, k, v)

        k = repeat_kv(k, self.groups)
        v = repeat_kv(v, self.groups)
        S = k.shape[-2]
        allowed = self._causal_mask(positions, key_start, S, segment_ids)

        if self.sinks is None:
            # SDPA bool masks use True for allowed attention positions.
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, scale=self.scale)
        else:
            scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            scores = scores.masked_fill(~allowed, float("-inf"))
            sink = self.sinks.to(scores.dtype).view(1, -1, 1, 1).expand(B, -1, T, 1)
            weights = torch.softmax(torch.cat([scores, sink], dim=-1).float(), dim=-1)
            weights = weights[..., :-1].to(q.dtype)   # столбец sink в выход не идёт
            out = torch.matmul(weights, v)

        out = out.transpose(1, 2).reshape(B, T, cfg.q_dim)
        return self.o_proj(out)


# =============================================================================
# FFN (M6 §8–9)
# =============================================================================
def gated_activation(gate: torch.Tensor, up: torch.Tensor, cfg: MiniLLMConfig) -> torch.Tensor:
    """Гейтованная активация с опциональным клампом gpt-oss.

    Чистый SwiGLU: Swish(gate) ⊙ up. Вариант gpt-oss отличается тремя деталями —
    гейт режется сверху, значение с двух сторон, up-ветка сдвигается на единицу,
    а сигмоида несёт коэффициент alpha=1.702 (M6 §9). Кламп — часть обученного
    графа: включать и выключать его после обучения нельзя.
    """
    if cfg.hidden_act == "geglu":
        return F.gelu(gate) * up
    if cfg.hidden_act == "relu":
        return F.relu(gate) * up
    if cfg.swiglu_limit is None:
        return F.silu(gate) * up
    limit = cfg.swiglu_limit
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    if cfg.swiglu_shift_up:
        up = up + 1.0
    return gate * torch.sigmoid(cfg.swiglu_alpha * gate) * up


class FeedForward(nn.Module):
    """Три матрицы, не две. Самая частая ошибка в подсчёте бюджета (M6 §10)."""

    def __init__(self, cfg: MiniLLMConfig, intermediate: int | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        ff = intermediate or cfg.intermediate_size
        self.gate_proj = nn.Linear(cfg.hidden_size, ff, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, ff, bias=False)
        self.down_proj = nn.Linear(ff, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(gated_activation(self.gate_proj(x), self.up_proj(x), self.cfg))


# =============================================================================
# MoE (M7)
# =============================================================================
class MoELayer(nn.Module):
    """Fine-grained MoE: sigmoid-скоринг и aux-loss-free балансировка.

    Ключевая деталь (M7 §8): у каждого эксперта есть смещение b_i, которое
    участвует **только в отборе top-k** и обновляется вне backprop правилом
    «перегружен — вниз, недогружен — вверх». Отдельного auxiliary-gradient в
    языковую цель не добавляется, хотя изменённые маршруты, конечно, меняют,
    какие эксперты получают градиент LM-loss.

    Реализация экспертов — цикл с масками. Это O(E) проходов вместо одного
    grouped GEMM: медленно, зато видно, что происходит. Продовые ядра
    (Megablocks, grouped_gemm) сортируют токены по экспертам — M7 §10.

    Ёмкость (M7 §12) выключена по умолчанию: `capacity_factor=None` означает,
    что эксперт принимает всё. С заданным лимитом лишние назначения
    отбрасываются, и `dropped_fraction` показывает, сколько именно. Это самая
    тихая авария модуля: обучение продолжается, лосс выглядит нормально, а часть
    routed-назначений не вносит expert-вклад. Residual path, attention, shared
    experts и дальнейшие слои сохраняются; долю отброшенных назначений всё равно
    нужно писать в лог наравне с loss.
    """

    def __init__(self, cfg: MiniLLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        E, d, ff = cfg.num_local_experts, cfg.hidden_size, cfg.moe_intermediate_size
        self.router = nn.Linear(d, E, bias=False)
        self.gate_w = nn.Parameter(torch.empty(E, d, ff))
        self.up_w = nn.Parameter(torch.empty(E, d, ff))
        self.down_w = nn.Parameter(torch.empty(E, ff, d))
        self.shared = nn.ModuleList(
            FeedForward(cfg, ff) for _ in range(cfg.num_shared_experts)
        )
        # Не параметры: смещение живёт вне графа, поэтому buffer.
        self.register_buffer("expert_bias", torch.zeros(E), persistent=True)
        self.register_buffer("selected_load", torch.zeros(E), persistent=False)
        self.register_buffer("expert_load", torch.zeros(E), persistent=False)  # kept load
        self.register_buffer("dropped_load", torch.zeros(E), persistent=False)
        self.register_buffer("dropped_fraction", torch.zeros(()), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B, T, d = x.shape
        flat = x.reshape(-1, d)                                    # [N, d]

        logits = self.router(flat)                                 # [N, E]
        scores = (torch.sigmoid(logits) if cfg.router_scoring == "sigmoid"
                  else torch.softmax(logits, dim=-1))
        # Смещение сдвигает только отбор, но не вес — иначе оно попало бы в лосс.
        selection = scores + self.expert_bias
        topk = selection.topk(cfg.num_experts_per_tok, dim=-1).indices   # [N, k]
        selected_counts = torch.bincount(topk.reshape(-1),
                                         minlength=cfg.num_local_experts).float()
        self.selected_load = selected_counts / max(flat.shape[0], 1)
        weights = scores.gather(-1, topk)
        if cfg.router_scoring == "sigmoid":
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-9)

        # --- ёмкость эксперта (M7 §12)
        # Capacity applies only to the routed branch. Even a dropped assignment
        # retains the residual path, attention, shared experts and later layers.
        capacity = cfg.expert_capacity(flat.shape[0])
        kept = torch.ones_like(topk, dtype=torch.bool)
        dropped = 0
        if capacity is not None:
            for e in range(cfg.num_local_experts):
                hit = (topk == e)
                token_idx, slot_idx = hit.nonzero(as_tuple=True)
                overflow = token_idx.numel() - capacity
                if overflow <= 0:
                    continue
                if cfg.drop_policy == "score":
                    # Оставляем назначения с наибольшим гейтом: отбрасываем то,
                    # что и так вносило бы меньший вклад.
                    order = weights[token_idx, slot_idx].argsort(descending=True)
                else:
                    # Классика: кто пришёл раньше по порядку токенов в батче.
                    order = torch.arange(token_idx.numel(), device=x.device)
                drop = order[capacity:]
                kept[token_idx[drop], slot_idx[drop]] = False
                dropped += int(overflow)

            weights = weights * kept
            if cfg.renormalize_on_drop:
                # Без перенормировки выход слоя систематически меньше по норме:
                # гейты суммировались в единицу, а одно слагаемое выпало.
                weights = weights / (weights.sum(-1, keepdim=True) + 1e-9)

        out = torch.zeros_like(flat)
        load = torch.zeros(cfg.num_local_experts, device=x.device, dtype=torch.float32)
        for e in range(cfg.num_local_experts):
            hit = (topk == e) & kept
            if not hit.any():
                continue
            token_idx, slot_idx = hit.nonzero(as_tuple=True)
            load[e] = token_idx.numel()
            xe = flat[token_idx]                                   # [n_e, d]
            he = gated_activation(xe @ self.gate_w[e], xe @ self.up_w[e], cfg)
            ye = he @ self.down_w[e]
            out.index_add_(0, token_idx, ye * weights[token_idx, slot_idx, None])

        self.expert_load = load / max(flat.shape[0], 1)
        self.dropped_load = self.selected_load - self.expert_load
        self.dropped_fraction = torch.tensor(
            dropped / max(topk.numel(), 1), device=x.device)
        out = out.view(B, T, d)
        for expert in self.shared:
            out = out + expert(x)
        return out

    @torch.no_grad()
    def update_expert_bias(self) -> None:
        """Шаг aux-loss-free балансировки. Вызывается тренером после optimizer.step()."""
        target = 1.0 / self.cfg.num_local_experts * self.cfg.num_experts_per_tok
        # Balance the router against pre-capacity selections; post-drop load can
        # hide the very overload that the controller must correct.
        error = target - self.selected_load
        self.expert_bias += self.cfg.router_bias_speed * torch.sign(error)


# =============================================================================
# Блок (M6 §4, §6)
# =============================================================================
class Block(nn.Module):
    """Один слой стека. Размещение нормы — четыре режима одним флагом.

        pre   : x = x + Sub(N(x))                       — распространённая схема
        post  : x = N(x + Sub(x))                       — Transformer-2017
        peri  : x = x + N_post(Sub(N_pre(x)))           — Peri-LN-like
        olmo2 : x = x + N(Sub(x))                       — OLMo2-style
        none  : x = x + Sub(x)                          — для абляции §6
    """

    def __init__(self, cfg: MiniLLMConfig, layer_index: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.attn = Attention(cfg, layer_index)
        self.ffn = MoELayer(cfg) if cfg.moe_layer(layer_index) else FeedForward(cfg)

        def norm():
            return RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

        p = cfg.norm_placement
        self.attn_norm = norm() if p in ("pre", "post", "peri", "olmo2") else None
        self.ffn_norm = norm() if p in ("pre", "post", "peri", "olmo2") else None
        self.attn_post_norm = norm() if p == "peri" else None
        self.ffn_post_norm = norm() if p == "peri" else None

    def forward(self, x, cos, sin, positions=None, cache=None, segment_ids=None):
        if positions is None:
            positions = torch.arange(x.shape[1], device=x.device)
        p = self.cfg.norm_placement
        if p == "pre":
            x = x + self.attn(self.attn_norm(x), cos, sin, positions, cache, segment_ids)
            return x + self.ffn(self.ffn_norm(x))
        if p == "post":
            x = self.attn_norm(x + self.attn(x, cos, sin, positions, cache, segment_ids))
            return self.ffn_norm(x + self.ffn(x))
        if p == "peri":
            x = x + self.attn_post_norm(
                self.attn(self.attn_norm(x), cos, sin, positions, cache, segment_ids))
            return x + self.ffn_post_norm(self.ffn(self.ffn_norm(x)))
        if p == "olmo2":
            x = x + self.attn_norm(self.attn(x, cos, sin, positions, cache, segment_ids))
            return x + self.ffn_norm(self.ffn(x))
        x = x + self.attn(x, cos, sin, positions, cache, segment_ids)
        return x + self.ffn(x)


# =============================================================================
# Модель
# =============================================================================
class MiniLLM(nn.Module):
    """Decoder-only языковая модель, целиком определяемая MiniLLMConfig."""

    def __init__(self, cfg: MiniLLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Пересчёт активаций вместо их хранения снижает пиковую память ценой
        # дополнительного forward-compute. Конкретная надбавка времени зависит
        # от блока, оборудования и реализации; универсального процента нет.
        self.gradient_checkpointing = False
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.rope = RotaryEmbedding(cfg)
        self.layers = nn.ModuleList(Block(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps) if cfg.norm_placement != "none" else None
        if cfg.tie_word_embeddings:
            self.lm_head = None       # логиты считаем весами эмбеддинга
        else:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, MoELayer):
            for p in (module.gate_w, module.up_w, module.down_w):
                nn.init.normal_(p, mean=0.0, std=std)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None,
                cache: KVCache | None = None,
                segment_ids: torch.Tensor | None = None):
        """Run the decoder.

        ``segment_ids`` isolates independently packed SFT dialogues. IDs must be
        non-negative inside real segments and ``-1`` on padding. Positions reset
        within each segment; cached generation intentionally does not accept this mode.
        """
        B, T = input_ids.shape
        if segment_ids is not None:
            if cache is not None or segment_ids.shape != input_ids.shape:
                raise ValueError("segment_ids require uncached input with the same [B,T] shape")
            positions = torch.zeros_like(segment_ids)
            for b in range(B):
                for seg in torch.unique(segment_ids[b]):
                    value = int(seg.item())
                    if value < 0:
                        continue
                    idx = (segment_ids[b] == value).nonzero(as_tuple=True)[0]
                    positions[b, idx] = torch.arange(idx.numel(), device=input_ids.device)
        else:
            offset = len(cache) if cache is not None else 0
            positions = torch.arange(offset, offset + T, device=input_ids.device)
        cos, sin = self.rope(positions)
        cos, sin = cos.to(self.embed_tokens.weight.dtype), sin.to(self.embed_tokens.weight.dtype)

        x = self.embed_tokens(input_ids)
        use_ckpt = self.gradient_checkpointing and self.training and cache is None
        for layer in self.layers:
            if use_ckpt:
                # Capture non-state arguments in a closure; checkpoint receives
                # only the activation tensor whose autograd graph is recomputed.
                def layer_forward(hidden, current_layer=layer):
                    return current_layer(hidden, cos, sin, positions, None, segment_ids)
                x = torch.utils.checkpoint.checkpoint(
                    layer_forward, x, use_reentrant=False)
            else:
                x = layer(x, cos, sin, positions, cache, segment_ids)
        if cache is not None:
            cache.advance(T)
        if self.norm is not None:
            x = self.norm(x)

        logits = (x @ self.embed_tokens.weight.T) if self.lm_head is None else self.lm_head(x)

        loss = None
        if labels is not None:
            # Сдвиг на один токен: предсказываем следующий, а не текущий.
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.cfg.vocab_size).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    # ------------------------------------------------------------------ сервисное
    def update_expert_biases(self) -> None:
        for layer in self.layers:
            if isinstance(layer.ffn, MoELayer):
                layer.ffn.update_expert_bias()

    def num_parameters(self, trainable_only: bool = False) -> int:
        params = self.parameters()
        return sum(p.numel() for p in params if p.requires_grad or not trainable_only)

    def new_cache(self) -> KVCache:
        windows = [self.cfg.sliding_window if kind == "sliding" else None
                   for kind in self.cfg.layer_types]
        return KVCache(self.cfg.num_hidden_layers, windows)
