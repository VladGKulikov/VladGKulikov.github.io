"""MiniLLM configuration: one object controls the entire course architecture.

The capstone principle is that S / M / L profiles differ **only by config**, not
by code branches. If a larger model requires forking the model code, the config
was designed poorly.

Fields deliberately match those in real models' `config.json` files (module 1):
`hidden_size`, `num_key_value_heads`, `rope_theta`, `num_local_experts`. Thus the
module 1 model-passport script reads a student's model with the same code it
uses for gpt-oss, closing the course loop.

Course-module references beside each field group map every setting to its
explanation.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, ClassVar, Literal


def _round_to(value: float, multiple: int) -> int:
    """Round up to a multiple for hardware-tile alignment (M6 §9)."""
    return int(math.ceil(value / multiple) * multiple)


@dataclass
class MiniLLMConfig:
    """Complete description of the course's educational reference architecture.

The defaults form a coherent set of modern mechanisms; they are neither an
industry standard nor the only possible canon.
"""

    # -------------------------------------------------------------- M2: vocabulary
    vocab_size: int = 16384
    tie_word_embeddings: bool = True

    # ---------------------------------------------------------- M6: width and depth
    hidden_size: int = 512
    num_hidden_layers: int = 8

    # --------------------------------------------------------------- M5: attention
    num_attention_heads: int = 8
    num_key_value_heads: int = 2          # < heads → GQA; == 1 → MQA; == heads → MHA
    head_dim: int | None = None           # None → hidden_size // num_attention_heads
    attention_bias: bool = False
    attention_sinks: bool = False         # learned attention sink, as in gpt-oss
    qk_norm: bool = False                 # RMSNorm on q and k before scoring
    sliding_window: int | None = None     # context width for sliding layers
    layer_types: list[str] | None = None  # ["full"|"sliding"] × num_hidden_layers

    # --------------------------------------------------------------- M4: positions
    max_position_embeddings: int = 1024
    rope_theta: float = 10000.0
    rope_pairing: Literal["rotate_half", "adjacent"] = "rotate_half"
    # {"rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 1024,
    #  "beta_fast": 32.0, "beta_slow": 1.0}  |  {"rope_type": "linear", "factor": 4.0}
    rope_scaling: dict[str, Any] | None = None

    # ----------------------------------------------------------- M6: norms and FFN
    norm_placement: Literal["pre", "post", "peri", "olmo2", "none"] = "pre"
    rms_norm_eps: float = 1e-6
    # All variants are gated: "relu" means ReGLU, not a classic two-matrix FFN.
    # Changing φ within the GLU family is the design explored by Shazeer.
    hidden_act: Literal["swiglu", "geglu", "relu"] = "swiglu"
    intermediate_size: int | None = None  # None → aligned 2/3 rule
    ffn_multiple_of: int = 64
    swiglu_limit: float | None = None     # gpt-oss clamp; None → plain SwiGLU
    swiglu_alpha: float = 1.702           # GELU approximation coefficient (M6 §8)
    swiglu_shift_up: bool = False         # gpt-oss's characteristic +1 on the up branch

    # ----------------------------------------------------------------------- M7: MoE
    num_local_experts: int = 0            # 0 → dense FFN
    num_experts_per_tok: int = 2
    num_shared_experts: int = 0
    moe_intermediate_size: int | None = None
    router_scoring: Literal["sigmoid", "softmax"] = "sigmoid"
    router_bias_speed: float = 1e-3       # aux-loss-free balancing step
    first_k_dense_replace: int = 0        # number of initial layers kept dense
    # Expert capacity (M7 §12). None removes the limit; experts accept everything.
    # The number is headroom over perfectly uniform load: 1.0 gives exactly
    # N·k/E assignments per expert; 1.25 adds one quarter for imbalance.
    capacity_factor: float | None = None
    drop_policy: Literal["position", "score"] = "position"
    # Whether to renormalize gates of surviving assignments. False reproduces a
    # classic silent bug: the layer output systematically shrinks in norm.
    renormalize_on_drop: bool = True

    # ------------------------------------------------------------------- metadata
    initializer_range: float = 0.02
    # Reference-runtime parameters are stored in FP32. TrainConfig.compute_dtype
    # controls mixed precision separately, and KV-cache dtype is measured from
    # the tensors actually stored.
    torch_dtype: str = "float32"
    model_type: str = "minillm"

    # ------------------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.head_dim is None:
            if self.hidden_size % self.num_attention_heads:
                raise ValueError("hidden_size не делится на num_attention_heads; "
                                 "задайте head_dim явно")
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads должно делиться на num_key_value_heads")
        if self.intermediate_size is None:
            # 2/3 rule: parameter parity with a classic two-matrix 4d FFN.
            self.intermediate_size = _round_to(8 * self.hidden_size / 3, self.ffn_multiple_of)
        if self.moe_intermediate_size is None:
            self.moe_intermediate_size = self.intermediate_size
        if self.layer_types is None:
            self.layer_types = ["full"] * self.num_hidden_layers
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("длина layer_types должна равняться num_hidden_layers")
        if "sliding" in self.layer_types and self.sliding_window is None:
            raise ValueError("есть sliding-слои, но sliding_window не задан")
        if self.num_local_experts and self.num_experts_per_tok > self.num_local_experts:
            raise ValueError("num_experts_per_tok больше числа экспертов")
        if self.capacity_factor is not None and self.capacity_factor <= 0:
            raise ValueError("capacity_factor должен быть положительным или None")
        if self.torch_dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("torch_dtype описывает storage dtype параметров: float32/bfloat16/float16")

    # --------------------------------------------------------------- derived values
    @property
    def is_moe(self) -> bool:
        return self.num_local_experts > 0

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    def expert_capacity(self, tokens: int) -> int | None:
        """Assignments accepted by one expert in a batch of `tokens` tokens.

        With perfectly uniform routing each expert receives tokens·k/E;
        capacity_factor provides headroom above that number. Nothing is dropped
        while the router remains within the limit—which it almost never does.
        """
        if not self.is_moe or self.capacity_factor is None:
            return None
        import math as _math
        return max(1, _math.ceil(self.capacity_factor * tokens
                                 * self.num_experts_per_tok / self.num_local_experts))

    def moe_layer(self, index: int) -> bool:
        """Whether layer `index` is MoE (the first k layers remain dense)."""
        return self.is_moe and index >= self.first_k_dense_replace

    # ------------------------------------------------ analytical parameter count
    def parameter_report(self) -> dict[str, int]:
        """Exact module 6 formula budget, down to the final parameter.

        The notebook compares this dictionary with actual `model.parameters()`.
        A mismatch means either the lecture formula is wrong or the code does
        not match its own config. Both are bugs that must be visible.
        """
        d, ff = self.hidden_size, self.intermediate_size
        # All three MiniLLM activations are gated: SwiGLU, GeGLU, and ReGLU
        # (`hidden_act="relu"`). There is no ungated two-matrix FFN in the config;
        # it is outside this educational reference stack. The multiplier is
        # therefore always 3. An older factor of 2 for "relu" made the formula
        # differ from the model by 19% of its parameters.
        gates = 3

        embed = self.vocab_size * d
        embeddings = embed if self.tie_word_embeddings else 2 * embed

        attn_per_layer = (
            d * self.q_dim                      # W_q
            + d * self.q_dim                    # W_o (back to the residual stream)
            + 2 * d * self.kv_dim               # W_k, W_v—where GQA saves parameters
        )
        if self.attention_bias:
            # q, k, and v have bias; the o_proj output projection never does.
            attn_per_layer += self.q_dim + 2 * self.kv_dim
        if self.attention_sinks:
            attn_per_layer += self.num_attention_heads
        if self.qk_norm:
            attn_per_layer += 2 * self.head_dim

        dense_ffn = gates * d * ff
        moe_ffn = (
            self.num_local_experts * gates * d * self.moe_intermediate_size
            + self.num_shared_experts * gates * d * self.moe_intermediate_size
            + d * self.num_local_experts        # router
        )

        norms_per_layer = {"pre": 2, "post": 2, "peri": 4, "olmo2": 2, "none": 0}[self.norm_placement] * d

        attention = attn_per_layer * self.num_hidden_layers
        norms = norms_per_layer * self.num_hidden_layers
        ffn = sum(
            moe_ffn if self.moe_layer(i) else dense_ffn
            for i in range(self.num_hidden_layers)
        )
        final_norm = d if self.norm_placement != "none" else 0

        total = embeddings + attention + ffn + norms + final_norm
        report = {
            "embeddings": embeddings,
            "attention": attention,
            "ffn": ffn,
            "norms": norms + final_norm,
            "total": total,
        }
        if not self.is_moe:
            block = attn_per_layer + dense_ffn + norms_per_layer
            report["block"] = block
            report["block_over_d2"] = round(block / d**2, 4)
        return report

    def active_parameters(self) -> int:
        """Active parameters per token; for MoE this differs from total (M7 §5)."""
        if not self.is_moe:
            return self.parameter_report()["total"]
        d, gates = self.hidden_size, 3
        report = self.parameter_report()
        per_expert = gates * d * self.moe_intermediate_size
        routed_active = self.num_experts_per_tok * per_expert
        shared_active = self.num_shared_experts * per_expert
        moe_layers = sum(1 for i in range(self.num_hidden_layers) if self.moe_layer(i))
        dense_layers = self.num_hidden_layers - moe_layers
        active_ffn = (
            moe_layers * (routed_active + shared_active + d * self.num_local_experts)
            + dense_layers * gates * d * self.intermediate_size
        )
        return report["embeddings"] + report["attention"] + active_ffn + report["norms"]

    def flops_per_token(self) -> int:
        """Rough forward-FLOP estimate per token: 2 × active parameters (M1)."""
        return 2 * self.active_parameters()

    # ---------------------------------------------------------- (de)serialization
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_pretrained(self, path: str | Path) -> Path:
        """Write config.json in a Hugging Face-compatible form.

        The module 1 model-passport script must read a student's model with the
        same code it uses for third-party Hugging Face models.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        target = path / "config.json"
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
        return target

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "MiniLLMConfig":
        path = Path(path)
        if path.is_dir():
            path = path / "config.json"
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    # ------------------------------------------------------------------- profiles
    #: Dense profiles: the capstone's three reference sizes.
    DENSE_PRESETS: ClassVar[dict[str, dict[str, Any]]] = {
        "S": dict(vocab_size=8192, hidden_size=256, num_hidden_layers=6,
                  num_attention_heads=8, num_key_value_heads=2, head_dim=32,
                  max_position_embeddings=512, tie_word_embeddings=True),
        "M": dict(vocab_size=16384, hidden_size=512, num_hidden_layers=8,
                  num_attention_heads=8, num_key_value_heads=2, head_dim=64,
                  max_position_embeddings=1024, tie_word_embeddings=True),
        "L": dict(vocab_size=32768, hidden_size=2048, num_hidden_layers=20,
                  num_attention_heads=16, num_key_value_heads=4, head_dim=128,
                  max_position_embeddings=2048, tie_word_embeddings=False,
                  rope_theta=500000.0),
    }

    #: Sparse twins: (experts, top-k, shared, expert width).
    #: Expert width keeps the ACTIVE parameter count close to its dense twin:
    #: (top_k + shared) · ff ≈ d_ffn. Routing, memory traffic, and execution
    #: efficiency still affect token cost; a similar active budget does not
    #: guarantee equal runtime or FLOP counts (M7 §4).
    MOE_PRESETS: ClassVar[dict[str, dict[str, Any]]] = {
        "S-MoE": dict(base="S", num_local_experts=16, num_experts_per_tok=2,
                      num_shared_experts=1, moe_intermediate_size=256),
        "M-MoE": dict(base="M", num_local_experts=32, num_experts_per_tok=2,
                      num_shared_experts=1, moe_intermediate_size=448),
        "L-MoE": dict(base="L", num_local_experts=128, num_experts_per_tok=8,
                      num_shared_experts=1, moe_intermediate_size=640),
    }

    @classmethod
    def profile(cls, name: str, **overrides: Any) -> "MiniLLMConfig":
        """Six profiles: three dense models and three sparse twins.

        Dense:
          S — smoke: `Run all` in free Colab in roughly fifteen minutes.
          M — baseline: 20× Chinchilla-style reference budget ≈ 620M tokens;
              this is a heuristic, not a universal optimum.
          L — optional: 1.02B parameters and a 24+ GB GPU; its full Chinchilla
              budget is unattainable here, and the lecture says so explicitly.

        Sparse profiles (`S-MoE`, `M-MoE`, `L-MoE`) differ from their dense
        twins **only in the FFN**. Vocabulary, depth, attention, and positions
        are identical, so each pair's KV cache per token matches to the byte.
        Active parameter counts differ by a few percent. This gives a comparison
        at similar active parameter budgets, with runtime measured separately.

        For the 10.02B-total `L-MoE`, two FP32 AdamW moment tensors require
        74.66 GiB. FP32 weights, gradients, and both moments total 149.32 GiB
        (160.33 GB), before activations and temporary buffers. This exceeds
        the course's basic hardware budget. Its role is arithmetic—to
        show total/active decoupling at frontier scale while remaining readable
        to the module 1 model-passport script.
        """
        if name in cls.DENSE_PRESETS:
            return cls(**{**cls.DENSE_PRESETS[name], **overrides})
        if name in cls.MOE_PRESETS:
            preset = dict(cls.MOE_PRESETS[name])
            base = cls.DENSE_PRESETS[preset.pop("base")]
            # Keep the first layer dense: routers in early layers consistently
            # fail to specialize, so frontier models keep them dense (M7 §8).
            preset.setdefault("first_k_dense_replace", 1)
            return cls(**{**base, **preset, **overrides})
        available = sorted(cls.DENSE_PRESETS) + sorted(cls.MOE_PRESETS)
        raise KeyError(f"неизвестный профиль {name!r}; доступны {available}")

    @classmethod
    def twin(cls, name: str) -> "MiniLLMConfig":
        """Return a profile's twin: sparse for dense, and dense for sparse."""
        if name in cls.DENSE_PRESETS:
            return cls.profile(f"{name}-MoE")
        if name in cls.MOE_PRESETS:
            return cls.profile(cls.MOE_PRESETS[name]["base"])
        raise KeyError(f"неизвестный профиль {name!r}")


# Convenient ready-made variants for module 5/6/7 ablations.
def gqa_variants(base: MiniLLMConfig) -> dict[str, MiniLLMConfig]:
    """MHA / GQA / MQA, all else equal, for KV-cache measurement (M5 §4)."""
    return {
        "MHA": MiniLLMConfig(**{**base.to_dict(),
                                "num_key_value_heads": base.num_attention_heads}),
        "GQA": base,
        "MQA": MiniLLMConfig(**{**base.to_dict(), "num_key_value_heads": 1}),
    }


def kv_bytes_per_token(cfg: MiniLLMConfig, dtype_bytes: int = 2) -> int:
    """KV-cache bytes/token: 2 · L · H_kv · d_head · sizeof(dtype) (M5 §4, M10 §4).

    Sliding layers are omitted: this formula answers “what does full context
    cost?” rather than “how much actually remains inside the window?”
    """
    return 2 * cfg.num_hidden_layers * cfg.kv_dim * dtype_bytes
