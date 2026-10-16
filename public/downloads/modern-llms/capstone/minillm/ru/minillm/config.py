"""Конфигурация MiniLLM: один объект — вся архитектура курса флагами.

Принцип капстоуна: профили S / M / L отличаются **только конфигом**, а не веткой
кода. Если для «модели побольше» пришлось бы форкнуть модель, значит конфиг
спроектирован плохо.

Поля намеренно названы так же, как в `config.json` живых моделей (модуль 1):
`hidden_size`, `num_key_value_heads`, `rope_theta`, `num_local_experts`. Благодаря
этому паспортный скрипт из модуля 1 читает модель студента тем же кодом, что и
gpt-oss, — курс замыкается в кольцо.

Ссылки на модули курса стоят у каждой группы полей: это карта «что где объяснено».
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, ClassVar, Literal


def _round_to(value: float, multiple: int) -> int:
    """Округление вверх до кратного — выравнивание под тайлы железа (M6 §9)."""
    return int(math.ceil(value / multiple) * multiple)


@dataclass
class MiniLLMConfig:
    """Полное описание учебной reference-архитектуры курса.

Умолчания отражают согласованный набор современных механизмов, но не являются
отраслевым стандартом и не претендуют на единственный «канон».
"""

    # ---------------------------------------------------------------- M2: словарь
    vocab_size: int = 16384
    tie_word_embeddings: bool = True

    # ------------------------------------------------------------- M6: шина, глубина
    hidden_size: int = 512
    num_hidden_layers: int = 8

    # ------------------------------------------------------------------ M5: внимание
    num_attention_heads: int = 8
    num_key_value_heads: int = 2          # < heads → GQA; == 1 → MQA; == heads → MHA
    head_dim: int | None = None           # None → hidden_size // num_attention_heads
    attention_bias: bool = False
    attention_sinks: bool = False         # обучаемый «слив» внимания, gpt-oss
    qk_norm: bool = False                 # RMSNorm на q и k перед скорингом
    sliding_window: int | None = None     # ширина окна для sliding-слоёв
    layer_types: list[str] | None = None  # ["full"|"sliding"] × num_hidden_layers

    # ------------------------------------------------------------------ M4: позиции
    max_position_embeddings: int = 1024
    rope_theta: float = 10000.0
    rope_pairing: Literal["rotate_half", "adjacent"] = "rotate_half"
    # {"rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 1024,
    #  "beta_fast": 32.0, "beta_slow": 1.0}  |  {"rope_type": "linear", "factor": 4.0}
    rope_scaling: dict[str, Any] | None = None

    # ------------------------------------------------------------ M6: норма и FFN
    norm_placement: Literal["pre", "post", "peri", "olmo2", "none"] = "pre"
    rms_norm_eps: float = 1e-6
    # Все варианты гейтованные: "relu" означает ReGLU, а не классический
    # двухматричный FFN. Смена φ внутри GLU-семейства — то, что перебирал Shazeer.
    hidden_act: Literal["swiglu", "geglu", "relu"] = "swiglu"
    intermediate_size: int | None = None  # None → правило 2/3 с выравниванием
    ffn_multiple_of: int = 64
    swiglu_limit: float | None = None     # кламп gpt-oss; None → чистый SwiGLU
    swiglu_alpha: float = 1.702           # коэффициент GELU-аппроксимации (M6 §8)
    swiglu_shift_up: bool = False         # фирменное «+1» на up-ветке gpt-oss

    # ----------------------------------------------------------------------- M7: MoE
    num_local_experts: int = 0            # 0 → dense-FFN
    num_experts_per_tok: int = 2
    num_shared_experts: int = 0
    moe_intermediate_size: int | None = None
    router_scoring: Literal["sigmoid", "softmax"] = "sigmoid"
    router_bias_speed: float = 1e-3       # шаг aux-loss-free балансировки
    first_k_dense_replace: int = 0        # сколько первых слоёв оставить dense
    # Ёмкость эксперта (M7 §12). None — лимита нет, эксперт принимает всё.
    # Число задаёт запас над идеально равномерной нагрузкой: 1.0 — ровно
    # N·k/E назначений на эксперта, 1.25 — четверть сверху на неравномерность.
    capacity_factor: float | None = None
    drop_policy: Literal["position", "score"] = "position"
    # Перенормировать ли гейты выживших назначений. False воспроизводит
    # классическую тихую ошибку: выход слоя систематически падает по норме.
    renormalize_on_drop: bool = True

    # -------------------------------------------------------------------- служебное
    initializer_range: float = 0.02
    # Параметры reference-runtime хранятся в FP32. Смешанная точность обучения
    # задаётся отдельно в TrainConfig.compute_dtype, а dtype KV-cache измеряется
    # по фактически сохранённым тензорам.
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
            # Правило 2/3: паритет по параметрам с классическим 4d-FFN из двух матриц.
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

    # ------------------------------------------------------------------ производные
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
        """Сколько назначений принимает один эксперт на батче из `tokens` токенов.

        При идеально равномерной маршрутизации каждому достаётся tokens·k/E;
        capacity_factor задаёт запас над этим числом. Ничего не отбрасывается,
        пока роутер укладывается в лимит, — а он не укладывается почти никогда.
        """
        if not self.is_moe or self.capacity_factor is None:
            return None
        import math as _math
        return max(1, _math.ceil(self.capacity_factor * tokens
                                 * self.num_experts_per_tok / self.num_local_experts))

    def moe_layer(self, index: int) -> bool:
        """Является ли слой `index` MoE-слоем (первые k слоёв держим dense)."""
        return self.is_moe and index >= self.first_k_dense_replace

    # ------------------------------------------------- аналитический счёт параметров
    def parameter_report(self) -> dict[str, int]:
        """Точный бюджет по формулам модуля 6 — до последнего параметра.

        Ноутбук сверяет этот словарь с фактическим `model.parameters()`: если
        расходится, значит либо формула в лекции неверна, либо код не соответствует
        собственному конфигу. Оба случая — баг, который надо увидеть.
        """
        d, ff = self.hidden_size, self.intermediate_size
        # Все три активации в MiniLLM гейтованные: SwiGLU, GeGLU и ReGLU
        # (`hidden_act="relu"`). Негейтованного двухматричного FFN в конфиге нет —
        # он не входит в учебный reference stack. Поэтому множитель всегда 3; раньше здесь
        # стояло 2 для "relu", и формула расходилась с моделью на 19% параметров.
        gates = 3

        embed = self.vocab_size * d
        embeddings = embed if self.tie_word_embeddings else 2 * embed

        attn_per_layer = (
            d * self.q_dim                      # W_q
            + d * self.q_dim                    # W_o (обратно в шину)
            + 2 * d * self.kv_dim               # W_k, W_v — вот где экономит GQA
        )
        if self.attention_bias:
            # Bias есть у q, k и v; выходная проекция o_proj всегда без него.
            attn_per_layer += self.q_dim + 2 * self.kv_dim
        if self.attention_sinks:
            attn_per_layer += self.num_attention_heads
        if self.qk_norm:
            attn_per_layer += 2 * self.head_dim

        dense_ffn = gates * d * ff
        moe_ffn = (
            self.num_local_experts * gates * d * self.moe_intermediate_size
            + self.num_shared_experts * gates * d * self.moe_intermediate_size
            + d * self.num_local_experts        # роутер
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
        """Активные параметры на токен — для MoE отличается от total (M7 §5)."""
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
        """Грубая оценка forward-FLOP на токен: 2 × активные параметры (M1)."""
        return 2 * self.active_parameters()

    # ------------------------------------------------------------- (де)сериализация
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_pretrained(self, path: str | Path) -> Path:
        """Пишет config.json в HF-совместимом виде.

        Смысл: паспортный скрипт модуля 1 должен уметь прочитать модель студента
        тем же кодом, каким читает чужие модели с Hugging Face.
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

    # --------------------------------------------------------------------- профили
    #: Плотные профили — три опорных размера капстоуна.
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

    #: Разреженные пары: число экспертов, top-k, общие эксперты, ширина.
    #: Ширина даёт близкий к плотной паре активный бюджет:
    #: (top_k + shared) · ff ≈ d_ffn. Маршрутизация, обмен данными и
    #: эффективность исполнения также влияют на стоимость токена;
    #: близость бюджетов не гарантирует равенства времени или FLOP (М7 §4).
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
        """Шесть профилей: три плотных и три разрежённых близнеца.

        Плотные:
          S — smoke: `Run all` в бесплатном Colab минут за пятнадцать.
          M — базовый: 20× Chinchilla-style reference budget ≈ 620M токенов; это эвристика, не универсальный optimum.
          L — опциональный: 1.02B параметров, нужна карта от 24 GB; полный
              Chinchilla-бюджет здесь недостижим, и лекция говорит об этом прямо.

        Разрежённые (`S-MoE`, `M-MoE`, `L-MoE`) отличаются от своих плотных
        близнецов **только FFN**: словарь, глубина, внимание и позиции те же,
        поэтому KV-кэш на токен у пары совпадает до байта. Активных параметров
        столько же с точностью до нескольких процентов — значит, сравнение идёт
        при близком активном параметрическом бюджете; время измеряется отдельно.

        `L-MoE` содержит 10.02 млрд параметров. Два FP32-тензора моментов
        AdamW требуют 74.66 GiB; вместе с FP32-весами и градиентами это
        149.32 GiB (160.33 GB), без активаций и временных буферов. Такой
        объём превышает базовый аппаратный бюджет курса. Роль профиля —
        расчётная: показать различие полного и активного числа параметров
        в масштабе крупных моделей; его конфигурацию читает скрипт М1.
        """
        if name in cls.DENSE_PRESETS:
            return cls(**{**cls.DENSE_PRESETS[name], **overrides})
        if name in cls.MOE_PRESETS:
            preset = dict(cls.MOE_PRESETS[name])
            base = cls.DENSE_PRESETS[preset.pop("base")]
            # Первый слой оставляем плотным: у ранних слоёв роутер стабильно
            # не успевает специализироваться, и фронтир держит их dense (M7 §8).
            preset.setdefault("first_k_dense_replace", 1)
            return cls(**{**base, **preset, **overrides})
        available = sorted(cls.DENSE_PRESETS) + sorted(cls.MOE_PRESETS)
        raise KeyError(f"неизвестный профиль {name!r}; доступны {available}")

    @classmethod
    def twin(cls, name: str) -> "MiniLLMConfig":
        """Близнец профиля: для плотного — разрежённый и наоборот."""
        if name in cls.DENSE_PRESETS:
            return cls.profile(f"{name}-MoE")
        if name in cls.MOE_PRESETS:
            return cls.profile(cls.MOE_PRESETS[name]["base"])
        raise KeyError(f"неизвестный профиль {name!r}")


# Удобные готовые варианты для абляций модуля 5/6/7.
def gqa_variants(base: MiniLLMConfig) -> dict[str, MiniLLMConfig]:
    """MHA / GQA / MQA при прочих равных — для замера KV-кэша (M5 §4)."""
    return {
        "MHA": MiniLLMConfig(**{**base.to_dict(),
                                "num_key_value_heads": base.num_attention_heads}),
        "GQA": base,
        "MQA": MiniLLMConfig(**{**base.to_dict(), "num_key_value_heads": 1}),
    }


def kv_bytes_per_token(cfg: MiniLLMConfig, dtype_bytes: int = 2) -> int:
    """Байты KV-кэша на токен: 2 · L · H_kv · d_head · sizeof(dtype) (M5 §4, M10 §4).

    Sliding-слои в этой оценке не учтены: формула отвечает на вопрос «сколько
    стоит полный контекст», а не «сколько реально живёт в окне».
    """
    return 2 * cfg.num_hidden_layers * cfg.kv_dim * dtype_bytes
