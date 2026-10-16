"""Render learner-facing RU/EN Bridge Layer blocks from canonical source code."""
from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import Any

from . import attention, cache, moe, normalization, quantization, rope, sft, speculative
from .notebook_cells import TEST_CODE
from .registry import BridgeSpec


def _source(obj: Any, *, strip_docstring: bool = True) -> str:
    source = textwrap.dedent(inspect.getsource(obj)).strip()
    if not strip_docstring:
        return source
    tree = ast.parse(source)
    node = tree.body[0]
    body = getattr(node, "body", [])
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        lines = source.splitlines()
        del lines[body[0].lineno - 1 : body[0].end_lineno]
        source = "\n".join(lines)
    return source


def _sources(objects: tuple[Any, ...]) -> str:
    return "\n\n".join(_source(obj) for obj in objects)


@dataclass(frozen=True)
class Presentation:
    title_ru: str
    title_en: str
    imports: str
    numpy_objects: tuple[Any, ...]
    torch_objects: tuple[Any, ...]
    api_code: str | None
    numpy_ru: str
    numpy_en: str
    torch_ru: str
    torch_en: str
    api_ru: str | None
    api_en: str | None
    table_ru: tuple[tuple[str, str, str], ...]
    table_en: tuple[tuple[str, str, str], ...]
    asset: str
    official_ru: str | None = None
    official_en: str | None = None


PRESENTATIONS: dict[str, Presentation] = {
    "B02": Presentation(
        "RoPE: вращение пар координат", "RoPE: rotating coordinate pairs",
        "import numpy as np\nimport torch",
        (rope.numpy_rope_explicit,), (rope.torch_rope_explicit,), None,
        "NumPy явно строит обратные частоты, углы для абсолютных позиций и две координаты каждого двумерного вращения. Благодаря этому видно, где именно позиция входит в оператор.",
        "NumPy exposes inverse frequencies, absolute-position angles, and both coordinates of every 2D rotation, making the positional contribution visible.",
        "PyTorch вычисляет фазы и вращения как минимум в float32, затем возвращает тип входа. Это предотвращает раннее округление позиций в FP16/BF16. Для больших индексов и точного эталона используется float64; целочисленное хранение само по себе не устраняет погрешность последующего преобразования. Градиенты проходят по входным векторам.",
        "Phase construction and rotation use at least float32 before conversion back to the input dtype. This avoids early FP16/BF16 position rounding. Float64 remains the reference for large indices: keeping IDs as integers does not remove error after conversion to a working float type. Gradients flow through the input vectors.",
        None, None,
        (("Частоты", "`np.arange` + обычный массив", "`torch.arange(..., device=x.device)`"),
         ("Тригонометрия", "`np.cos`, `np.sin`", "`Tensor.cos`, `Tensor.sin`"),
         ("Устройство", "CPU", "CPU/CUDA/MPS"),
         ("Градиенты", "нет", "autograd по входу"),
         ("Точность", "вычисления и выход float64", "фазы float32/64; выход как у входа"),
         ("Абсолютная позиция", "явный `position_ids`", "согласована с позициями ключей в кэше")),
        (("Frequencies", "`np.arange` on a CPU array", "`torch.arange(..., device=x.device)`"),
         ("Trigonometry", "`np.cos`, `np.sin`", "`Tensor.cos`, `Tensor.sin`"),
         ("Device", "CPU", "CPU/CUDA/MPS"),
         ("Gradients", "none", "autograd through the input"),
         ("Precision", "float64 calculation and output", "float32/64 phases; output keeps input dtype"),
         ("Absolute position", "explicit `position_ids`", "must agree with cached-key positions")),
        "m04_rope_bridges.py",
    ),
    "B04": Presentation(
        "SDPA строка за строкой", "SDPA line by line",
        "import math\nimport numpy as np\nimport torch\nimport torch.nn.functional as F",
        (attention.numpy_sdpa_explicit,), (attention.torch_sdpa_explicit,),
        _source(attention.torch_sdpa_optimized),
        "В NumPy видны все стадии: $QK^\\top$, масштаб $1/\\sqrt{d_h}$, расширение маски до нужной формы, softmax по оси ключей и смешивание $V$. `row_has_key` задаёт нулевой выход полностью закрытой строки. Входы FP16 повышаются до FP32 для вычислений, затем выход возвращается в исходный формат. Стандартный NumPy не предоставляет встроенный BF16; этот тип проверяется в PyTorch-варианте.",
        "Every stage is visible in NumPy: $QK^\\top$, the $1/\\sqrt{d_h}$ scale, mask broadcasting, softmax over keys, and the $V$ reduction. `row_has_key` defines the fully masked-row policy: zero output.",
        "`Tensor.transpose`, `dim=-1` и `masked_fill` выражают ту же цепочку с поддержкой autograd. Входы FP16/BF16 переводятся в FP32 до скалярных произведений; выход возвращается в тип Q. Устройство остаётся общим для всех тензоров.",
        "`Tensor.transpose`, `dim=-1`, and `masked_fill` express the same chain with autograd support. FP16/BF16 inputs are widened to FP32 before dot products, and the output returns to Q's dtype. All tensors remain on their shared device.",
        "Однострочный API передаёт тот же оператор диспетчеру PyTorch, который может выбрать math-, memory-efficient-, FlashAttention- или другой доступный backend.",
        "The one-line API hands the same operator to PyTorch's dispatcher, which may select a math, memory-efficient, FlashAttention, or another available backend.",
        (("Транспонирование K", "`np.swapaxes`", "`Tensor.transpose`"),
         ("Ось softmax", "`axis=-1`", "`dim=-1`"),
         ("Булева маска", "задаём явно", "в API `True` означает «разрешено»"),
         ("Пустая строка", "нулевой выход — политика этого примера", "поведение API проверяется тестом"),
         ("Dropout", "нет", "в API нужен явный `dropout_p=0.0` при оценке"),
         ("Backend", "последовательность NumPy-операций", "eager или fused kernel")),
        (("K transpose", "`np.swapaxes`", "`Tensor.transpose`"),
         ("Softmax axis", "`axis=-1`", "`dim=-1`"),
         ("Boolean mask", "implemented explicitly", "API uses `True` for allowed edges"),
         ("Empty row", "course policy: zero output", "API behavior is guarded by regression tests"),
         ("Dropout", "none", "evaluation must pass `dropout_p=0.0`"),
         ("Backend", "a sequence of NumPy operations", "eager or fused kernel")),
        "m05_attention_bridges.py",
        "Официальный контракт: [PyTorch `scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention).",
        "Official contract: [PyTorch `scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention).",
    ),
    "B05": Presentation(
        "GQA: повторение K/V и `enable_gqa`", "GQA: repeating K/V and `enable_gqa`",
        "import numpy as np\nimport torch\nimport torch.nn.functional as F",
        (attention.numpy_gqa_explicit,), (attention.torch_gqa_explicit,),
        _source(attention.torch_gqa_optimized),
        "GQA не меняет формулу attention. NumPy явно повторяет каждую K/V-голову `g = H_q/H_{kv}` раз, после чего использует уже разобранный SDPA.",
        "GQA does not change the attention equation. NumPy explicitly repeats every K/V head `g = H_q/H_{kv}` times and then reuses the SDPA reference.",
        "Явный PyTorch-вариант делает то же через `repeat_interleave`. Это полезный эталон для форм и порядка голов, но он материализует повторённые K/V.",
        "The explicit PyTorch path uses `repeat_interleave`. It is a useful shape and head-order reference, but it materializes repeated K/V tensors.",
        "`enable_gqa=True` выражает группировку без явного повторения в пользовательском коде; доступность и эффективность конкретного пути зависят от backend.",
        "`enable_gqa=True` expresses grouping without explicit user-code repetition; availability and efficiency depend on the backend.",
        (("Условие", "`Hq % Hkv == 0`", "то же"),
         ("K/V", "`np.repeat(..., axis=1)`", "`repeat_interleave(..., dim=1)`"),
         ("Память эталона", "повторённые массивы", "повторённые тензоры"),
         ("Оптимизированный путь", "нет", "`enable_gqa=True`"),
         ("Проверка", "совпадение с MHA после повтора", "явный путь ↔ API")),
        (("Requirement", "`Hq % Hkv == 0`", "same"),
         ("K/V", "`np.repeat(..., axis=1)`", "`repeat_interleave(..., dim=1)`"),
         ("Reference memory", "materialized repeated arrays", "materialized repeated tensors"),
         ("Optimized path", "none", "`enable_gqa=True`"),
         ("Test", "matches MHA after repetition", "explicit path ↔ API")),
        "m05_attention_bridges.py",
        "Официальный контракт: [PyTorch SDPA GQA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention).",
        "Official contract: [PyTorch SDPA GQA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention).",
    ),
    "B07": Presentation(
        "RMSNorm: редукция, dtype и autograd", "RMSNorm: reduction, dtype, and autograd",
        "import numpy as np\nimport torch\nimport torch.nn.functional as F",
        (normalization.numpy_rms_norm,), (normalization.torch_rms_norm_explicit,),
        _source(normalization.torch_rms_norm),
        "NumPy-эталон показывает средний квадрат по последней размерности, добавление $\\epsilon$, обратный корень и покоординатный обучаемый масштаб.",
        "The NumPy reference exposes the last-dimension mean square, $\\epsilon$, reciprocal square root, and the learned per-coordinate scale.",
        "Явная PyTorch-реализация переводит FP16/BF16 в FP32 **до возведения в квадрат**. После умножения на вес она возвращает формат входа; FP64 сохраняется для проверки производных.",
        "The explicit path widens FP16/BF16 to FP32 **before squaring** and restores the input dtype after the gain. FP64 stays FP64 for derivative checks.",
        "`F.rms_norm` скрывает редукцию внутри библиотечного оператора и может использовать оптимизированную реализацию.",
        "`F.rms_norm` moves the reduction into a library operator that may use an optimized implementation.",
        (("Редукция", "`np.mean(..., axis=-1)`", "`mean(dim=-1)`"),
         ("Обратный корень", "`1 / np.sqrt`", "`torch.rsqrt`"),
         ("Вес", "broadcast массива `[D]`", "parameter `[D]` с градиентом"),
         ("Накопление", "float64 в эталоне", "явный путь: минимум FP32; API проверяется отдельно"),
         ("Градиент", "нет", "проверяется `gradcheck`")),
        (("Reduction", "`np.mean(..., axis=-1)`", "`mean(dim=-1)`"),
         ("Reciprocal root", "`1 / np.sqrt`", "`torch.rsqrt`"),
         ("Weight", "broadcast array `[D]`", "learned parameter `[D]`"),
         ("Accumulation", "float64 reference", "explicit path: at least FP32; check API separately"),
         ("Gradient", "none", "checked with `gradcheck`")),
        "m06_block_bridges.py",
        "Официальный API: [PyTorch `rms_norm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.rms_norm.html).",
        "Official API: [PyTorch `rms_norm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.rms_norm.html).",
    ),
    "B08": Presentation(
        "SwiGLU и ограниченный вариант", "SwiGLU and a bounded variant",
        "import numpy as np\nimport torch",
        (normalization.numpy_swiglu,), (normalization.torch_swiglu,), None,
        "Обе ветви видны явно: gate проходит через SiLU-подобную нелинейность и затем умножается на up-ветвь. Параметры `alpha`, ограничения и сдвиг не прячутся в имени функции.",
        "Both branches remain visible: the gate passes through a SiLU-like nonlinearity and multiplies the up branch. `alpha`, clipping, and the shift are explicit parameters.",
        "NumPy вычисляет сигмоиду через `logaddexp`, PyTorch — через `torch.sigmoid`. PyTorch сохраняет граф производных и повышает точность промежуточных вычислений FP16/BF16. В точках излома `clamp` обычная производная не определена; численная проверка проходит по обе стороны границ.",
        "NumPy uses `logaddexp` for a stable sigmoid; PyTorch uses `torch.sigmoid`, retains autograd, and widens half-precision intermediates. Clamp has no ordinary derivative at its kinks, so finite-difference checks use points on either side.",
        None, None,
        (("Нелинейность", "явная сигмоида", "`torch.sigmoid`"),
         ("Ограничение gate", "`np.minimum`", "`Tensor.clamp(max=...)`"),
         ("Ограничение up", "`np.clip`", "`Tensor.clamp(min=..., max=...)`"),
         ("Обычная SwiGLU", "без limits, `shift_up=0`", "тот же контракт"),
         ("Градиенты", "нет", "autograd; точки clamp проверяются отдельно")),
        (("Nonlinearity", "explicit sigmoid", "`torch.sigmoid`"),
         ("Gate bound", "`np.minimum`", "`Tensor.clamp(max=...)`"),
         ("Up bound", "`np.clip`", "`Tensor.clamp(min=..., max=...)`"),
         ("Standard SwiGLU", "no limits, `shift_up=0`", "same contract"),
         ("Gradients", "none", "autograd; clamp kinks are tested separately")),
        "m06_block_bridges.py",
    ),
    "B09": Presentation(
        "Top-k MoE: выбор экспертов и веса смеси", "Top-k MoE: expert selection and mixture weights",
        "import numpy as np\nimport torch",
        (moe.numpy_topk_router_explicit,), (moe.torch_topk_router_explicit,), None,
        "Чистые оценки, приоритеты выбора и веса смеси разделены. Вес нормируется в логарифмической области, поэтому отрицательные логиты около −1000 не дают деления 0/0. Без смещения выбор по логитам избегает искусственных равенств после насыщения sigmoid.",
        "Clean scores, selection priorities, and gates remain separate. Log-space normalization avoids 0/0 at very negative logits. Without bias, ranking logits avoids artificial ties caused by sigmoid saturation.",
        "PyTorch использует log-sigmoid, topk и gather. FP16/BF16 повышается до FP32. Градиент проходит через веса выбранных экспертов; индексы дискретны, а порядок topk при равенствах не гарантирован. При нормированном top-1 вес равен 1, и этот путь не даёт градиента роутеру.",
        "PyTorch uses log-sigmoid, topk, and gather, promoting FP16/BF16 to FP32. Gradients pass through selected gates, not discrete indices; topk tie order is unspecified. A renormalized top-1 gate is identically one and supplies no router gradient through this path.",
        None, None,
        (("Чистые оценки", "log-sigmoid → exp", "log-sigmoid → exp"),
         ("Селекция", "stable `argsort`", "`torch.topk`"),
         ("Вес выбранного эксперта", "`take_along_axis`", "`gather`"),
         ("Bias", "влияет только на выбор", "тот же контракт"),
         ("Ties", "stable sort в эталоне", "порядок не гарантирован")),
        (("Clean scores", "log-sigmoid → exp", "log-sigmoid → exp"),
         ("Selection", "stable `argsort`", "`torch.topk`"),
         ("Selected weight", "`take_along_axis`", "`gather`"),
         ("Bias", "selection only", "same contract"),
         ("Ties", "stable reference sort", "ordering is not guaranteed")),
        "m07_moe_bridges.py",
        "Контракт tie-breaking: [PyTorch `topk`](https://docs.pytorch.org/docs/stable/generated/torch.topk.html).",
        "Tie-breaking contract: [PyTorch `topk`](https://docs.pytorch.org/docs/stable/generated/torch.topk.html).",
    ),
    "B10": Presentation(
        "MoE dispatch и combine", "MoE dispatch and combine",
        "import numpy as np\nimport torch",
        (moe.numpy_expert_dispatch_explicit,), (moe.torch_expert_dispatch_explicit,), None,
        "В этом мосте используются простые двухматричные tanh-эксперты: так видна именно сборка результатов. NumPy проходит по токенам и назначениям, затем прибавляет каждый взвешенный выход. Это не замена SwiGLU основного слоя.",
        "Two-matrix tanh experts keep this bridge focused on dispatch/combine. The token/slot loop adds each weighted expert output. The main notebook layer separately implements SwiGLU.",
        "PyTorch меняет порядок работы: сначала собирает все назначения одного эксперта в пакет, затем возвращает результаты токенам через `index_add_`.",
        "PyTorch changes the execution order: assignments are batched per expert, then results are returned to tokens with `index_add_`.",
        None, None,
        (("Порядок", "token → slot", "expert → assigned tokens"),
         ("Экспертный batch", "нет", "индексация `hidden[token_ids]`"),
         ("Combine", "`output[token] += ...`", "`output.index_add_`"),
         ("Autograd", "нет", "через expert matmul и combine"),
         ("CUDA-детерминизм", "CPU reference", "`index_add_` может зависеть от порядка атомарных сложений")),
        (("Order", "token → slot", "expert → assigned tokens"),
         ("Expert batch", "none", "indexed `hidden[token_ids]`"),
         ("Combine", "`output[token] += ...`", "`output.index_add_`"),
         ("Autograd", "none", "through expert matmul and combine"),
         ("CUDA determinism", "CPU reference", "`index_add_` may depend on atomic-add ordering")),
        "m07_moe_bridges.py",
        "Контракт накопления: [PyTorch `index_add_`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.index_add_.html).",
        "Accumulation contract: [PyTorch `index_add_`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.index_add_.html).",
    ),
    "B17": Presentation(
        "Ограниченный KV-cache: append, окно и rollback", "Bounded KV cache: append, window, and rollback",
        "import numpy as np\nimport torch\nimport torch.nn.functional as F",
        (cache.NumpyKVCacheExplicit,), (cache.TorchKVCacheExplicit,), None,
        "Минимальный кэш показывает три состояния: сохранённые K/V, абсолютную позицию первого сохранённого токена и позицию следующей записи. После сдвига окна создаётся отдельная копия оставшихся данных: срез сам по себе удерживал бы старый буфер.",
        "The minimal cache exposes three pieces of state: retained K/V, the absolute position of the first retained token, and the next append position. After eviction, an owned copy releases this container's reference to the old backing buffer; slicing alone would retain it.",
        "PyTorch-класс изменяет состояние на месте, проверяет тип и устройство тензоров и копирует сохранённое окно. Это контейнер для инференса: входы не должны хранить граф градиентов. Реальный движок использует страницы или заранее выделенную память; временное объединение тензоров здесь может занять больше памяти, чем итоговое окно.",
        "The tensor version validates dtype and device, then owns the retained window. This inference-only container rejects gradient-tracked inputs. A serving engine normally uses pages or preallocated buffers; the temporary concatenation here can exceed the final window's payload.",
        None, None,
        (("Добавление", "`np.concatenate`", "`torch.cat`"),
         ("Физическое окно", "срез + копия", "срез + clone"),
         ("Абсолютные позиции", "Python integers", "Python integers + tensor state"),
         ("Rollback", "обрезает сохранённый хвост", "тот же контракт"),
         ("Выделение памяти", "учебный контейнер с копированием", "в движке обычно буфер или пул блоков")),
        (("Append", "`np.concatenate`", "`torch.cat`"),
         ("Physical window", "slice + owned copy", "slice + clone"),
         ("Absolute positions", "Python integers", "Python integers plus tensor state"),
         ("Rollback", "truncates the retained suffix", "same contract"),
         ("Storage allocation", "reference container with copying", "engines usually use a buffer or block pool")),
        "m10_cache_bridges.py",
    ),
    "B18": Presentation(
        "Точная спекулятивная коррекция", "Exact speculative correction",
        "import numpy as np\nimport torch",
        (speculative.numpy_rejection_kernel_explicit,), (speculative.torch_rejection_kernel_explicit, speculative.torch_sample_speculative_token_explicit), None,
        "NumPy делает видимыми три части доказательства: вероятность принятия `min(1,p/q)`, перекрытие `min(p,q)` и нормированный положительный остаток `(p-q)_+`.",
        "NumPy exposes all three parts of the proof: acceptance `min(1,p/q)`, overlap `min(p,q)`, and the normalized positive residual `(p-q)_+`.",
        "Явное ядро предполагает два неотрицательных нормированных вектора p и q одной одномерной формы, длины не меньше двух. Все значения конечны; PyTorch-векторы имеют общий тип float32 или float64 и находятся на одном устройстве. Остаток нормируется по сумме положительных разностей без отсечения малой массы. Код выборки предназначен для инференса; дискретный выбор не имеет обычного градиента. Параметр `generator` необязателен: без него используется глобальный генератор. В проверках передаём отдельный `torch.Generator` для воспроизводимости. Предложение должно быть выбрано из q, а не задано произвольно.",
        "The explicit kernel assumes matching one-dimensional p and q vectors with at least two entries: finite, nonnegative, and normalized. Torch inputs share one device and a common float32 or float64 dtype. It normalizes positive differences without discarding a small residual. Sampling is an inference operation; discrete choices do not carry an ordinary gradient. The optional `generator` defaults to the global RNG; tests pass a separate `torch.Generator` for reproducibility. The proposal must have been sampled from q, not supplied arbitrarily.",
        None, None,
        (("Носитель q", "явная маска `q>0`", "булев тензор маски"),
         ("Принятие", "`np.minimum`", "`torch.minimum`"),
         ("Остаток", "`np.maximum(p-q,0)`", "`torch.clamp(..., min=0)`"),
         ("Выборка", "вне математического ядра", "`torch.rand` / `multinomial`, генератор необязателен"),
         ("Проверка", "точное восстановление p", "Монте-Карло и проверка баланса")),
        (("Support of q", "explicit `q>0` mask", "boolean tensor mask"),
         ("Acceptance", "`np.minimum`", "`torch.minimum`"),
         ("Residual", "`np.maximum(p-q,0)`", "`torch.clamp(..., min=0)`"),
         ("Sampling", "outside the mathematical kernel", "`torch.rand` / `multinomial` with a generator"),
         ("Test", "exact reconstruction of p", "Monte Carlo plus algebraic parity")),
        "m11_inference_bridges.py",
    ),
    "B19": Presentation(
        "Групповая имитационная квантизация", "Grouped fake quantization",
        "import numpy as np\nimport torch\nimport torch.nn.functional as F",
        (quantization.numpy_grouped_fake_quant_explicit,), (quantization.torch_grouped_fake_quant_explicit,), None,
        "NumPy дополняет только последнюю размерность, разбивает её на группы, вычисляет локальный масштаб и выбирает ближайший уровень сетки. Ведущие оси — например, индекс эксперта — никогда не смешиваются.",
        "NumPy pads only the last dimension, forms groups, computes a local scale, and selects the nearest grid value. Leading axes—such as expert index—never share a group.",
        "Явное ядро принимает конечные вещественные массивы с непустыми осями; для сопоставления используем float32/float64. Размер группы — положительное целое число. Сетка — конечный одномерный массив с точным нулём и хотя бы одним ненулевым уровнем; после приведения к типу входа эти свойства должны сохраняться. Промежуточные масштабы и расстояния должны помещаться в рабочий тип. PyTorch повторяет изменение формы, максимум `amax`, расширение размерностей и выбор `argmin`. Это имитационная квантизация (fake quantization): результат хранится как обычный тензор. Коды не упаковываются в четыре бита. При равном расстоянии argmin выбирает первый уровень; это учебное правило не воспроизводит все аппаратные режимы округления.",
        "The explicit kernel assumes finite real floating arrays with nonempty axes; use float32/float64 for parity. Group size is a positive integer. The grid is a finite one-dimensional array containing exact zero and a nonzero level, including after conversion to the input dtype. Intermediate scales and distances must be representable in that working dtype. PyTorch repeats reshape, `amax`, broadcasting, and `argmin`. This is fake quantization: the result remains an ordinary tensor, without packed bits. Ties select the first grid entry; this teaching rule does not implement every hardware rounding mode.",
        None, None,
        (("Группировка", "reshape последней оси", "тот же reshape"),
         ("Дополнение нулями", "`np.pad`", "`F.pad`"),
         ("Масштаб", "`max_abs / max_grid`", "то же выражение для тензоров"),
         ("Код", "`argmin` по сетке", "`argmin(dim=-1)`"),
         ("Результат", "деквантированный массив", "деквантированный тензор без упаковки битов")),
        (("Grouping", "reshape the last axis", "same reshape"),
         ("Padding", "`np.pad`", "`F.pad`"),
         ("Scale", "`max_abs / max_grid`", "same tensor expression"),
         ("Code", "`argmin` over the grid", "`argmin(dim=-1)`"),
         ("Result", "dequantized array", "dequantized tensor, not packed bits")),
        "m11_inference_bridges.py",
    ),
    "B20": Presentation(
        "Метки только для ответа ассистента", "Assistant-only labels",
        "import numpy as np\nimport torch",
        (sft.numpy_assistant_labels,), (sft.torch_assistant_labels,), None,
        "Булева маска не является самой меткой. NumPy оставляет настоящий идентификатор токена на позициях ответа ассистента и записывает `ignore_index` в остальных позициях.",
        "The boolean mask is not the label itself. NumPy keeps the real token ID on assistant positions and writes `ignore_index` everywhere else.",
        "Для сопоставления используем неотрицательные `np.int64` / `torch.long` идентификаторы и булеву маску той же формы; в PyTorch оба тензора находятся на одном устройстве. `masked_fill` создаёт метки в их собственных позициях. Перед кросс-энтропией нужно ровно один раз выровнять `logits[..., :-1, :]` и `labels[..., 1:]`. Если модель сдвигает метки внутри, ей передают несдвинутые массивы. Наличие целевых токенов и их принадлежность словарю проверяет функция потерь.",
        "Use nonnegative `np.int64` / `torch.long` IDs and a boolean mask of the same shape; Torch inputs share a device. The result stores labels at their own positions. Align `logits[..., :-1, :]` with `labels[..., 1:]` exactly once before cross-entropy. A model that shifts internally must receive unshifted arrays. The loss function checks that active targets exist and belong to its vocabulary.",
        None, None,
        (("Источник метки", "идентификатор токена", "тот же ID"),
         ("Игнорирование", "`np.where`", "`masked_fill`"),
         ("Форма", "совпадает с маской", "проверяется до операции"),
         ("Autograd", "не нужен для целочисленных меток", "потеря дифференцируется по логитам")),
        (("Label value", "real token ID", "same ID"),
         ("Ignore", "`np.where`", "`masked_fill`"),
         ("Shape", "must match the mask", "validated before the operation"),
         ("Autograd", "not applicable to integer labels", "loss differentiates logits, not labels")),
        "m13_sft_bridges.py",
    ),
    "B21": Presentation(
        "Изоляция упакованных диалогов", "Isolating packed dialogues",
        "import numpy as np\nimport torch",
        (sft.numpy_segment_position_ids, sft.numpy_block_causal_mask),
        (sft.torch_segment_position_ids, sft.torch_block_causal_mask), None,
        "NumPy сначала нумерует непрерывные сегменты, затем разрешает внимание только внутри одного сегмента и только назад по времени. `position_ids` сбрасываются на каждой границе.",
        "NumPy first numbers contiguous runs, then permits attention only within one run and only causally. `position_ids` reset at every boundary.",
        "PyTorch строит те же номера непрерывных сегментов `run_ids`, маску и позиции на устройстве исходного тензора. Сравниваются именно непрерывные сегменты: одно и то же числовое обозначение может повториться позже и не должно соединять их внимание.",
        "PyTorch builds the same `run_ids`, mask, and positions on the tensor device. Contiguous runs—not only numeric labels—matter because a segment label may reappear later.",
        None, None,
        (("Граница", "изменение соседнего `segment_id`", "та же проверка границы"),
         ("Позиции", "цикл по T", "цикл по позициям на исходном устройстве"),
         ("Изоляция", "`same_run & causal & valid`", "та же булева маска"),
         ("Дополнение", "сегмент `-1` запрещён", "тот же контракт"),
         ("Регрессия", "изменяем A, B неизменен", "SDPA проверяет выход сегмента B")),
        (("Boundary", "change in adjacent `segment_id`", "same boolean test"),
         ("Positions", "loop over T", "loop over positions on the input device"),
         ("Isolation", "`same_run & causal & valid`", "same boolean mask"),
         ("Padding", "segment `-1` is forbidden", "same contract"),
         ("Regression", "change A; B stays fixed", "SDPA checks B logits/output")),
        "m13_sft_bridges.py",
    ),
}


def _table(rows: tuple[tuple[str, str, str], ...], lang: str) -> str:
    if lang == "ru":
        header = "| Аспект | NumPy | PyTorch |\n|---|---|---|"
    else:
        header = "| Aspect | NumPy | PyTorch |\n|---|---|---|"
    return header + "\n" + "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows)


def render_lecture_block(spec: BridgeSpec, lang: str) -> str:
    if lang not in {"ru", "en"}:
        raise ValueError("lang must be 'ru' or 'en'")
    p = PRESENTATIONS[spec.id]
    title = p.title_ru if lang == "ru" else p.title_en
    intro = spec.ru_intro if lang == "ru" else spec.en_intro
    numpy_explain = p.numpy_ru if lang == "ru" else p.numpy_en
    torch_explain = p.torch_ru if lang == "ru" else p.torch_en
    api_explain = p.api_ru if lang == "ru" else p.api_en
    rows = p.table_ru if lang == "ru" else p.table_en
    demo = textwrap.dedent(TEST_CODE[spec.id]).strip()
    numpy_code = _sources(p.numpy_objects)
    torch_code = _sources(p.torch_objects)
    api_section = ""
    if p.api_code:
        api_heading = "#### 3. Оптимизированный или библиотечный PyTorch API" if lang == "ru" else "#### 3. Optimized or library PyTorch API"
        api_section = f"\n{api_heading}\n\n```python\n{p.api_code}\n```\n\n{api_explain}\n"
    details_heading = "Исполняемая проверка эквивалентности и граничных случаев" if lang == "ru" else "Executable parity and edge-case check"
    numpy_heading = "#### 1. Явная NumPy-реализация" if lang == "ru" else "#### 1. Explicit NumPy implementation"
    torch_heading = "#### 2. Та же математика явными операциями PyTorch" if lang == "ru" else "#### 2. The same mathematics as explicit PyTorch operations"
    full_file = f"Полный исполняемый файл: [`{p.asset}`](../assets/{p.asset})" if lang == "ru" else f"Complete executable file: [`{p.asset}`](../assets/{p.asset})"
    official = p.official_ru if lang == "ru" else p.official_en
    official_text = f"\n\n{official}" if official else ""
    return f'''<!-- BRIDGE:{spec.id}:START -->
### NumPy → PyTorch · {spec.id} — {title}

{intro}

```python
{p.imports}
```

{numpy_heading}

```python
{numpy_code}
```

{numpy_explain}

{torch_heading}

```python
{torch_code}
```

{torch_explain}
{api_section}
{_table(rows, lang)}

<details>
<summary><strong>{details_heading}</strong></summary>

```python
{demo}
```

</details>

{full_file}{official_text}
'''
