"""Metadata and learner-facing snippets for Bridge Layer Release A."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BridgeSpec:
    id: str
    module: str
    insert_before: int
    title: str
    ru_intro: str
    en_intro: str
    numpy_code: str
    torch_code: str
    ru_contract: str
    en_contract: str


BRIDGES: dict[str, BridgeSpec] = {
    "B02": BridgeSpec(
        "B02", "04", 5, "RoPE rotation",
        "Код для второго прохода. Здесь вход `x[..., T, D]` уже проверен: непустой вещественный тензор, чётная ширина D, целочисленные неотрицательные позиции `[T]`, конечная база больше 1. Соседние координаты образуют пары. Полные проверки входа находятся в ноутбуке (§12).",
        "Implementation route. Assume a validated, nonempty floating tensor `x[..., T, D]`, even D, nonnegative integer IDs `[T]`, and a finite base greater than one. Adjacent coordinates form pairs. The notebook adds input validation (§12).",
        "q_rot = numpy_rope(q, position_ids, base=10_000.0)\nk_rot = numpy_rope(k, position_ids, base=10_000.0)",
        "q_rot = torch_rope(q, position_ids, base=10_000.0)\nk_rot = torch_rope(k, position_ids, base=10_000.0)",
        "Важно согласовать способ спаривания координат и абсолютные позиции кэша. Проверка эквивалентности сравнивает повёрнутые Q/K при одних и тех же входах.",
        "The coordinate-pairing convention and absolute cache positions must match. The parity test compares rotated Q/K for identical inputs.",
    ),
    "B04": BridgeSpec(
        "B04", "05", 5, "Scaled dot-product attention",
        "Здесь Q/K/V уже проверены: это конечные вещественные массивы с непустыми осями и совместимыми формами `[..., T_q, D]`, `[..., T_k, D]`, `[..., T_k, D_v]`. Точки обозначают совместимые ведущие оси, например батч и головы. Для сравнения с API используем один тип данных и одно устройство; булева маска находится на том же устройстве и расширяется до формы логитов. Пример намеренно возвращает ноль для пустой строки. Выбор `error`/`zero` показан отдельно в §12. NumPy раскрывает логиты, маску, softmax и умножение на V; PyTorch API может выбрать оптимизированное ядро.",
        "Assume validated, finite floating Q/K/V with nonempty axes and compatible shapes `[..., T_q, D]`, `[..., T_k, D]`, and `[..., T_k, D_v]`. Leading axes can represent batches and heads. API comparisons use a shared dtype and device, with a Boolean mask on that device broadcastable to the scores. This minimal example deliberately returns zero for an empty row; §12 provides the separate `error`/`zero` choice. NumPy exposes scores, masking, softmax, and the V reduction before the PyTorch API delegates execution.",
        "out_np = numpy_sdpa(q_np, k_np, v_np, allowed_mask=mask_np)",
        "out_t = torch.nn.functional.scaled_dot_product_attention(\n    q, k, v, attn_mask=mask, dropout_p=0.0\n)",
        "В булевой маске SDPA значение `True` означает разрешённое внимание. При оценке `dropout_p` нужно передавать как `0.0`; разные ядра могут немного различаться округлением.",
        "In an SDPA boolean mask, `True` means allowed. Evaluation code must pass `dropout_p=0.0`; kernels may differ slightly through floating-point ordering.",
    ),
    "B05": BridgeSpec(
        "B05", "05", 7, "Grouped-query attention",
        "В этом блоке фиксированы четыре оси: Q `[B,H_q,T_q,D]`, K `[B,H_kv,T_k,D]`, V `[B,H_kv,T_k,D_v]`. Ось голов имеет индекс 1; `H_q` кратно положительному `H_kv`. Конечность значений, общий тип/устройство и булева маска предполагаются проверенными, как в B04. NumPy повторяет K/V-головы явно; PyTorch выражает группировку через `enable_gqa=True`, если выбранная реализация его поддерживает. Формат §12 с объединёнными головами `[T,H*D]` — отдельный интерфейс; перед вызовом нужны reshape и перестановка осей.",
        "This block fixes four axes: Q `[B,H_q,T_q,D]`, K `[B,H_kv,T_k,D]`, and V `[B,H_kv,T_k,D_v]`. Heads occupy axis 1, and `H_q` is divisible by positive `H_kv`. The finite-value, shared dtype/device, and Boolean-mask assumptions from B04 still apply. NumPy repeats K/V heads explicitly; supported PyTorch backends express grouping with `enable_gqa=True`. Section 12's flattened `[T,H*D]` layout is a separate interface: reshape and transpose it before using these functions.",
        "out_np = numpy_gqa(q_np, k_np, v_np, allowed_mask=mask_np)",
        "out_t = torch.nn.functional.scaled_dot_product_attention(\n    q, k, v, attn_mask=mask, dropout_p=0.0, enable_gqa=True\n)",
        "Должно выполняться `Hq % Hkv == 0`. Сравнение проводится вдали от неоднозначных масок и с явно заданными допусками.",
        "The contract requires `Hq % Hkv == 0`. Parity is checked away from ambiguous masking cases with explicit tolerances.",
    ),
    "B07": BridgeSpec(
        "B07", "06", 6, "RMSNorm",
        "Сравниваем конечный вещественный вход `[..., D]`, `D>0`, и необязательный вес `[D]` при конечном `eps>0`. Для сопоставления с API вход и вес имеют общий плавающий тип и устройство. Предполагаем, что квадраты, сумма и результат помещаются в рабочий формат: сам по себе конечный FP64-вход этого не гарантирует. NumPy раскрывает редукцию, PyTorch добавляет автоматическое дифференцирование.",
        "Compare finite floating inputs `[..., D]`, `D>0`, and an optional weight `[D]`, with finite `eps>0`. API comparisons use a common floating dtype and device for input and weight. Squares, their sum, and the output must fit the working format; merely being finite in FP64 does not ensure this. NumPy exposes the reduction, while PyTorch adds autograd.",
        "y_np = numpy_rms_norm(x_np, weight_np, eps=1e-6)",
        "y_t = torch.nn.functional.rms_norm(\n    x, [x.shape[-1]], weight=weight, eps=1e-6\n)",
        "Эквивалентность проверяется и по выходу, и по градиентам. Точность накопления редукции остаётся частью конкретного ядра.",
        "Parity covers both outputs and gradients. Reduction precision remains part of the concrete kernel contract.",
    ),
    "B08": BridgeSpec(
        "B08", "06", 10, "SwiGLU and bounded variants",
        "Обе реализации разделяют gate- и up-ветви. Параметры `alpha`, ограничение диапазона и сдвиг up-ветви задаются явно.",
        "Both implementations keep the gate and up branches explicit. `alpha`, optional clipping, and the up-branch shift are visible parameters.",
        "y_np = numpy_swiglu(g_np, u_np, alpha=1.702,\n    gate_limit=7.0, up_limit=7.0, shift_up=1.0)",
        "y_t = torch_swiglu(g, u, alpha=1.702,\n    gate_limit=7.0, up_limit=7.0, shift_up=1.0)",
        "Проверяются выходы и локальные градиенты вдали от точек излома ограничения. Обычная SwiGLU получается без ограничений и с `shift_up=0`.",
        "Outputs and local gradients are tested away from clamp kinks. Standard SwiGLU is recovered with no limits and `shift_up=0`.",
    ),
    "B09": BridgeSpec(
        "B09", "07", 7, "Top-k MoE routing",
        "Смещение может менять выбор экспертов, а веса смеси по-прежнему вычисляются из чистых оценок. NumPy делает эту границу видимой; PyTorch использует `topk` и `gather`.",
        "A selection bias may change expert choice while mixture weights still come from clean scores. NumPy makes the boundary explicit; PyTorch uses `topk` and `gather`.",
        "idx_np, w_np, clean_np = numpy_topk_router(\n    logits_np, 2, selection_bias=bias_np, activation='sigmoid'\n)",
        "idx_t, w_t, clean_t = torch_topk_router(\n    logits, 2, selection_bias=bias, activation='sigmoid'\n)",
        "PyTorch не гарантирует стабильный порядок индексов при равных значениях. Поэтому проверка эквивалентности использует логиты без равных значений.",
        "PyTorch does not guarantee stable indices for ties. The parity test therefore uses logits with a strict ordering.",
    ),
    "B10": BridgeSpec(
        "B10", "07", 13, "Expert dispatch and combine",
        "Входы имеют формы hidden `[T,D]`, индексы и веса смеси `[T,K]`, матрицы экспертов `[E,D,F]` и `[E,F,D]`. Индексы — целые числа от 0 до E−1; одинаковые номера в разных назначениях допустимы, их вклады складываются. Вещественные массивы конечны и используют совместимый плавающий тип; в PyTorch все тензоры находятся на одном устройстве. Минимальный мост проверяет индексы, затем раскрывает отправку и суммирование; вместимость здесь не ограничена.",
        "Use hidden `[T,D]`, indices and gates `[T,K]`, and expert weights `[E,D,F]` and `[E,F,D]`. Indices are integers from 0 through E−1; repeated assignments are allowed and their contributions add. Floating arrays are finite with compatible dtypes; PyTorch tensors share a device. The minimal bridge validates indices before exposing dispatch and combination, without a capacity limit.",
        "out_np, load_np = numpy_expert_dispatch(\n    x_np, experts_np, gates_np, w1_np, w2_np, capacity=cap\n)",
        "out_t, load_t = torch_expert_dispatch(\n    x, experts, gates, w1, w2, capacity=cap\n)",
        "Политика ёмкости, сохранённые и отброшенные назначения являются частью алгоритмического контракта. На CUDA `index_add_` может быть недетерминированным.",
        "Capacity policy, kept assignments, and dropped assignments are part of the algorithmic contract. CUDA `index_add_` may be nondeterministic.",
    ),
    "B17": BridgeSpec(
        "B17", "10", 13, "Bounded KV cache",
        "Кэш хранит не только K/V, но и абсолютный диапазон позиций. Локальное окно физически отбрасывает старый префикс, а rollback обрезает только сохранённый хвост.",
        "The cache stores K/V together with an absolute position range. A bounded window physically evicts the old prefix, while rollback truncates only the retained suffix.",
        "cache_np = NumpyKVCache(window=128)\ncache_np.append(k_np, v_np)\ncache_np.truncate(absolute_end)",
        "cache_t = TorchKVCache(window=128)\ncache_t.append(k, v)\ncache_t.truncate(absolute_end)",
        "Проверка сравнивает декодирование с кэшем с полным пересчётом и отдельно контролирует `start_pos`, `next_pos` и окно.",
        "The parity test compares cached decode with full recomputation and checks `start_pos`, `next_pos`, and the retained window.",
    ),
    "B18": BridgeSpec(
        "B18", "11", 6, "Exact speculative rejection",
        "NumPy показывает принятую массу и положительный остаток `(p-q)+`. PyTorch использует тот же баланс и `torch.multinomial`; проверки передают отдельный генератор случайных чисел.",
        "NumPy exposes overlap mass and the positive residual `(p-q)+`. PyTorch uses the same balance and `torch.multinomial`; tests pass a dedicated random generator.",
        "accept_np, residual_np, balance_np = numpy_rejection_kernel(p_np, q_np)",
        "accept_t, residual_t, balance_t = torch_rejection_kernel(p, q)\ntoken, accepted = torch_sample_speculative_token(\n    p, q, proposal, generator=gen\n)",
        "Отдельно проверяются `q_t=0`, `p=q` и восстановление методом Монте-Карло распределения основной модели.",
        "The tests cover `q_t=0`, `p=q`, and Monte Carlo recovery of the target distribution.",
    ),
    "B19": BridgeSpec(
        "B19", "11", 14, "Grouped fake quantization",
        "Группы формируются только вдоль последней размерности, поэтому элементы разных экспертов не попадают под один масштаб.",
        "Groups are formed only along the last dimension, so values from different experts never share a scale.",
        "q_np = numpy_grouped_fake_quant(\n    weights_np, grid=grid_np, group_size=16\n)",
        "q_t = torch_grouped_fake_quant(\n    weights, grid=grid, group_size=16\n)",
        "Проверка сравнивает индексы сетки, масштабы и деквантованные значения. Это учебное имитационное квантование, а не побитово точная упаковка NVFP4/MXFP4.",
        "Parity covers grid indices, scales, and dequantized values. This is educational fake quantization, not bit-exact NVFP4/MXFP4 packing.",
    ),
    "B20": BridgeSpec(
        "B20", "13", 6, "Assistant-only labels",
        "Метка содержит реальный идентификатор токена либо `-100`; булева маска только выбирает позиции, участвующие в функции потерь.",
        "A label contains the real token ID or `-100`; the boolean mask only selects positions that contribute to the loss.",
        "labels_np = numpy_assistant_labels(ids_np, assistant_mask_np)",
        "labels_t = torch_assistant_labels(ids, assistant_mask)",
        "Формы `input_ids` и маски обязаны совпадать. Включение EOT и инструментальных сообщений задаётся шаблоном данных, а не самой функцией.",
        "`input_ids` and mask shapes must match. Whether EOT and tool-related messages are targets belongs to the data-template contract.",
    ),
    "B21": BridgeSpec(
        "B21", "13", 7, "Segment-isolated packing",
        "`segment_ids` задают независимые диалоги, `position_ids` сбрасываются внутри каждого сегмента, а булева маска разрешает только причинное внимание внутри одного сегмента.",
        "`segment_ids` identify independent dialogues, `position_ids` reset per segment, and the boolean mask allows only causal attention inside the same segment.",
        "pos_np = numpy_segment_position_ids(segment_ids_np)\nmask_np = numpy_block_causal_mask(segment_ids_np)",
        "pos_t = torch_segment_position_ids(segment_ids)\nmask_t = torch_block_causal_mask(segment_ids)",
        "Регрессионный тест меняет содержимое сегмента A и проверяет, что логиты сегмента B не изменяются. Дополнение обозначается `-1`.",
        "The regression test changes segment A and verifies that segment B is unchanged. Padding uses segment ID `-1`.",
    ),
}


def specs_for_module(module: str) -> list[BridgeSpec]:
    return sorted((b for b in BRIDGES.values() if b.module == module), key=lambda b: (b.insert_before, b.id))


def course_root() -> Path:
    return Path(__file__).resolve().parents[1]
