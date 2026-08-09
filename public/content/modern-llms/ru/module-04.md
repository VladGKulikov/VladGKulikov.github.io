# Модуль 4. Современные механизмы внимания

*Курс «Modern LLMs» · лекция модуля 4 · редакция 2026.8*

> **О чём этот модуль.** В модуле 3 мы научились сообщать attention порядок токенов. Теперь разберём сам механизм внимания как инженерную систему. У него есть два долга. Первый — **состояние**: во время авторегрессионной генерации нужно хранить ключи и значения прошлых токенов. Второй — **перемещение данных**: при плотном attention матрица попарных скоров растёт квадратично и легко превращает память, а не арифметику, в главное ограничение. MQA, GQA и MLA уменьшают объём сохраняемого состояния. FlashAttention меняет маршрут вычислений без удаления логитов. Скользящие окна и разреженные схемы сокращают число позиций, с которыми взаимодействует запрос. В финале мы соберём эти решения в один паспорт attention-архитектуры.
>
> **Что нужно знать заранее.** Softmax, матричное умножение, causal mask и RoPE из модулей 3 и 18. Представление об уровнях памяти GPU полезно, но необходимые понятия вводятся здесь.

---

## 1. Мотивация: один запрос и два растущих счёта

Пусть модель уже обработала prompt и начала выдавать токены. На шаге $t$ новый query должен сопоставиться со всеми разрешёнными прошлыми ключами, а затем собрать взвешенную сумму значений. Проекции прошлых токенов можно не вычислять повторно: однажды полученные $K$ и $V$ сохраняются в **KV-cache — Key–Value cache**.

Эта экономия вычислений создаёт расход памяти. Для обычного attention объём, добавляемый одним новым токеном, равен

$$
b_{KV}=2L H_{kv}d_h b,
$$

где $L$ — число слоёв, $H_{kv}$ — число KV-heads, $d_h$ — размерность головы, $b$ — байт на значение, а множитель 2 учитывает key и value.

Возьмём открытую конфигурацию gpt-oss-120b: 36 слоёв, 64 query-heads, 8 KV-heads, размерность головы 64, bf16-кэш. Если мысленно сделать **все** слои полно-контекстными, получим

$$
2\cdot36\cdot8\cdot64\cdot2
=73\,728\ \text{байт}
=72\ \text{KiB на токен}.
$$

На 131 072 токенах это ровно 9 GiB. Но реальная схема gpt-oss гибридна: 18 слоёв используют полный контекст, а 18 — окно 128. Поэтому фактическая арифметика другая:

$$
M_{\text{full}}=18\cdot131\,072\cdot2048=4.5\ \text{GiB},
$$

$$
M_{\text{window}}=18\cdot128\cdot2048=4.5\ \text{MiB},
$$

$$
M_{\text{hybrid}}\approx4.504\ \text{GiB}.
$$

Здесь 2048 байт — K+V одного токена в одном GQA-слое. Значение 72 KiB/token остаётся полезным как **эквивалент всех полно-контекстных слоёв (all-full equivalent)**: оно показывает цену, которую имела бы та же геометрия голов без локальных слоёв. Но для оценки конкретной контрольной точки нужен расписанный по слоям паспорт attention.

Второй счёт возникает даже без KV-cache. Плотный attention сопоставляет $N$ запросов с $N$ ключами. Одна fp32-матрица скоров размером $8192\times8192$ занимает 256 MiB на голову. Если реализация пишет её в HBM — High Bandwidth Memory, затем читает обратно для softmax и умножения на $V$, значительная часть времени уходит на обмен данными.

Отсюда два маршрута модуля. Сначала уменьшим **что хранить**. Затем разберём, как уменьшить **что переносить и с чем взаимодействовать**.

## 2. Историческая прогрессия: сжатие состояния и перестройка вычислений

![VIZ m4/01 — две линии развития attention](assets/modern-llms/ru/module-04/m4_01_attention_timeline.svg)

Первая линия начинается с **MHA — Multi-Head Attention**. В классическом Transformer у каждой query-head есть собственные K- и V-проекции. Это даёт каждой голове отдельное подпространство сопоставления, но заставляет хранить $H_q$ наборов ключей и значений.

**MQA — Multi-Query Attention** сохраняет одну K/V-head на все query-heads. Кэш уменьшается в $H_q$ раз относительно MHA, однако в некоторых экспериментах такая агрессивная общность ухудшала качество.

**GQA — Grouped-Query Attention** вводит промежуточное число KV-heads. Обозначим число query-heads через $H_q$, число KV-heads через $H_{kv}$, а размер группы через

$$
g=\frac{H_q}{H_{kv}}.
$$

При $H_{kv}=H_q$ получаем MHA, при $H_{kv}=1$ — MQA. Работа Ainslie et al. показала, что в их uptraining-экспериментах GQA давала качество, близкое к MHA, при скорости, сопоставимой с MQA. Это эмпирический результат конкретной постановки, а не гарантия для любого $g$. [Первоисточник](https://arxiv.org/abs/2305.13245).

**MLA — Multi-head Latent Attention** меняет единицу хранения. Вместо полного набора K/V по головам модель кэширует компактный латент и небольшую позиционную компоненту. DeepSeek-V2 сообщила 93.3% сокращение KV-cache относительно DeepSeek 67B и до 5.76× большей максимальной генерационной пропускной способности в своём сравнении. [Технический отчёт](https://arxiv.org/abs/2405.04434).

Вторая линия не меняет математическую функцию attention. **FlashAttention** организует вычисление тайлами в быстрой памяти и поддерживает online softmax, не записывая полную $N\times N$ матрицу в HBM. Это точная, IO-aware перестановка вычислений, а не разреженная аппроксимация. [FlashAttention](https://arxiv.org/abs/2205.14135).

Третья линия всё же меняет множество взаимодействий. Sliding-window attention использует геометрически заданное локальное окно. StreamingLLM сохраняет начальные реальные позиции как attention sinks. MoBA, NSA и DSA выбирают блоки или токены по содержанию. Эти методы уменьшают не число байтов на сохранённый токен, а число токенов, участвующих в полном сопоставлении.

## 3. Мостик к классике: ядро, распределение и память

Attention удобно рассмотреть через несколько классических конструкций, если не принимать аналогии за буквальное тождество.

### Ядерное сглаживание

Оценка Надарая–Уотсона имеет форму

$$
\hat f(q)=\frac{\sum_i K(q,k_i)v_i}{\sum_i K(q,k_i)}.
$$

Softmax attention структурно похож: экспонента от скалярного логита играет роль положительного ядра, а $v_i$ — наблюдаемые значения. Отличие существенно: проекции $Q$, $K$ и $V$ обучаются совместно, головы взаимодействуют через остаточный поток, а последующие слои изменяют смысл полученного смешения. Поэтому корректнее говорить о **аналогии с ядерной регрессией**.

### Gibbs distribution

Если обозначить score через $s_i$, то

$$
p_i=\frac{e^{s_i/T}}{\sum_j e^{s_j/T}}
$$

имеет форму распределения Гиббса. Если хочется использовать физический термин «энергия», следует положить $E_i=-s_i$, поскольку классическая запись использует $e^{-E_i/T}$.

Масштаб $T=\sqrt{d_h}$ в Scaled Dot-Product Attention прежде всего нормирует дисперсию скалярных произведений. Температурная интерпретация полезна, но не означает, что стандартный делитель выбран как произвольный гиперпараметр остроты.

### Память как состояние динамической системы

KV-cache можно понимать как состояние, достаточное для продолжения авторегрессионного процесса без повторного вычисления прошлого. Разные attention-архитектуры отвечают на вопрос: какой объект является достаточным состоянием?

- MHA хранит K/V каждой головы;
- GQA разделяет K/V между группой query-heads;
- MLA хранит низкоразмерный латент;
- recurrent- и state-space-механизмы могут хранить фиксированный state вместо списка по позициям.

Так статистическая интерпретация приводит нас к системной: архитектура attention задаёт не только функцию смешивания, но и формат состояния при инференсе.

## 4. Формализм: SDPA, маски и масштаб логитов

**SDPA — Scaled Dot-Product Attention** определяется как

$$
\operatorname{Attn}(Q,K,V)
=
\operatorname{softmax}\!\left(
\frac{QK^\top}{\sqrt{d_h}}+M
\right)V.
$$

Здесь $Q\in\mathbb R^{T_q\times d_h}$, $K\in\mathbb R^{T_k\times d_h}$, $V\in\mathbb R^{T_k\times d_v}$, а $M$ — аддитивная маска. Нулевое значение разрешает пару, $-\infty$ запрещает её.

Каузальная маска оставляет $j\le i$. Оконная маска дополнительно требует, чтобы ключ находился в пределах последних $W$ позиций. Cross-attention может использовать другую геометрию $T_q\ne T_k$.

Почему делим на $\sqrt{d_h}$? При идеализированных независимых компонентах с нулевым средним и единичной дисперсией

$$
\operatorname{Var}(q^\top k)=d_h.
$$

Типичный масштаб логита растёт как $\sqrt{d_h}$. Деление возвращает его к порядку единицы и уменьшает риск чрезмерного насыщения softmax. В обученной сети независимость уже не обязана выполняться, поэтому это мотивационный расчёт, а не точное описание всех логитов.

Полностью замаскированная строка требует явной политики. Многие стабильные реализации softmax сначала вычитают максимум. Для строки из одних $-\infty$ выражение $-\infty-(-\infty)$ уже неопределено и может породить NaN. Учебная реализация по умолчанию сообщает ошибку; для padding-строк допускает явно заданный нулевой результат. Learned sink создаёт ещё один конечный score и тем самым делает нормировку определённой.

### NumPy → PyTorch · B04 — SDPA строка за строкой

Прозрачная NumPy-цепочка показывает логиты, маску, softmax и умножение на V; публичный API PyTorch может выбрать оптимизированное ядро.

```python
import math
import numpy as np
import torch
import torch.nn.functional as F
```

#### 1. Явная NumPy-реализация

```python
def numpy_sdpa_explicit(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    allowed_mask: np.ndarray | None = None,
) -> np.ndarray:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale

    if allowed_mask is None:
        row_has_key = np.ones_like(scores[..., :1], dtype=bool)
    else:
        mask = np.broadcast_to(np.asarray(allowed_mask, dtype=bool), scores.shape)
        row_has_key = mask.any(axis=-1, keepdims=True)
        scores = np.where(mask, scores, -np.inf)

    safe_scores = np.where(row_has_key, scores, 0.0)
    shifted = safe_scores - safe_scores.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    weights = exp / exp.sum(axis=-1, keepdims=True)
    weights = np.where(row_has_key, weights, 0.0)
    return np.matmul(weights, v)
```

В NumPy видны все стадии: $QK^\top$, масштаб $1/\sqrt{d_h}$, broadcasting маски, softmax по оси ключей и смешивание $V$. `row_has_key` задаёт политику полностью замаскированной строки: нулевой выход.

#### 2. Та же математика явными операциями PyTorch

```python
def torch_sdpa_explicit(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    allowed_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = q @ k.transpose(-2, -1) * scale

    if allowed_mask is None:
        row_has_key = torch.ones_like(scores[..., :1], dtype=torch.bool)
    else:
        mask = torch.broadcast_to(allowed_mask.to(torch.bool), scores.shape)
        row_has_key = mask.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~mask, -torch.inf)

    safe_scores = torch.where(row_has_key, scores, torch.zeros_like(scores))
    weights = torch.softmax(safe_scores, dim=-1)
    weights = torch.where(row_has_key, weights, torch.zeros_like(weights))
    return weights @ v
```

Математика не изменилась. `Tensor.transpose`, `dim=-1` и `masked_fill` выражают ту же цепочку, но теперь она сохраняет device/dtype и поддерживает autograd.

#### 3. Оптимизированный или библиотечный PyTorch API

```python
def torch_sdpa_optimized(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    allowed_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=allowed_mask,
        dropout_p=dropout_p,
    )
```

Однострочный API передаёт тот же оператор диспетчеру PyTorch, который может выбрать math-, memory-efficient-, FlashAttention- или другой доступный backend.

| Аспект | NumPy | PyTorch |
|---|---|---|
| Транспонирование K | `np.swapaxes` | `Tensor.transpose` |
| Ось softmax | `axis=-1` | `dim=-1` |
| Булева маска | задаём явно | в API `True` означает «разрешено» |
| Пустая строка | нулевой выход — политика курса | для API закрепляется regression-тестом |
| Dropout | нет | в API нужен явный `dropout_p=0.0` при оценке |
| Backend | последовательность NumPy-операций | eager или fused kernel |

<details>
<summary><strong>Исполняемая проверка эквивалентности и граничных случаев</strong></summary>

```python
rng = np.random.default_rng(4)
q_np = rng.standard_normal((1, 2, 4, 8))
k_np = rng.standard_normal((1, 2, 4, 8))
v_np = rng.standard_normal((1, 2, 4, 6))
mask_np = np.tril(np.ones((4, 4), dtype=bool))

np_out = numpy_sdpa_explicit(q_np, k_np, v_np, allowed_mask=mask_np)
q = torch.tensor(q_np, dtype=torch.float64, requires_grad=True)
k = torch.tensor(k_np, dtype=torch.float64, requires_grad=True)
v = torch.tensor(v_np, dtype=torch.float64, requires_grad=True)
mask = torch.tensor(mask_np)
explicit_out = torch_sdpa_explicit(q, k, v, allowed_mask=mask)
api_out = torch_sdpa_optimized(q, k, v, allowed_mask=mask, dropout_p=0.0)

torch.testing.assert_close(explicit_out, torch.from_numpy(np_out), rtol=1e-10, atol=1e-10)
torch.testing.assert_close(api_out, explicit_out, rtol=1e-10, atol=1e-10)

mask_np[0] = False
np_empty = numpy_sdpa_explicit(q_np, k_np, v_np, allowed_mask=mask_np)
t_empty = torch_sdpa_explicit(q, k, v, allowed_mask=torch.tensor(mask_np))
np.testing.assert_array_equal(np_empty[..., 0, :], 0.0)
torch.testing.assert_close(t_empty[..., 0, :], torch.zeros_like(t_empty[..., 0, :]))

explicit_out.sum().backward()
assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
print("B04 explicit NumPy / explicit PyTorch / optimized API: PASS")
```

</details>

Полный исполняемый файл: [`m04_attention_bridges.py`](../assets/m04_attention_bridges.py)

Официальный контракт: [PyTorch `scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention).

## 5. Решённый пример: MHA, MQA, GQA, MLA и реальный гибрид

Начнём с одной модели и меняем только число KV-heads. Для 36 слоёв, $H_q=64$, $d_h=64$ и bf16:

| Схема | $H_{kv}$ | Размер группы $g$ | Кэш на токен, если все слои full |
|---|---:|---:|---:|
| MHA | 64 | 1 | 576 KiB |
| GQA | 8 | 8 | 72 KiB |
| MQA | 1 | 64 | 9 KiB |

На 131 072 токенах это соответственно 72 GiB, 9 GiB и 1.125 GiB. Значение MHA — гипотетический вариант той же глубины и размерности голов. Фраза «не помещается» всегда требует указать устройство: 72 GiB KV-cache почти исчерпывает 80-GiB GPU ещё до весов и рабочих буферов, но распределённая память или CPU offload остаются возможны.

Теперь вернём фактическое расписание gpt-oss. Один GQA-слой добавляет

$$
2\cdot8\cdot64\cdot2=2048\ \text{байт на токен}.
$$

Для 18 полно-контекстных и 18 оконных слоёв получаем около **4.504 GiB**, а не 9 GiB. Половина слоёв перестаёт расти после 128 позиций.

MLA использует другую геометрию. В учебном DeepSeek-подобном сценарии 61 слой хранит латент $d_c=512$ и decoupled RoPE-компоненту 64:

$$
b_{\mathrm{MLA}}=(512+64)\cdot61\cdot2
=70\,272\ \text{байта}
=68.625\ \text{KiB на токен}.
$$

Сравним с **выбранным стандартным MHA-baseline**: 128 голов, $d_h=128$:

$$
b_{\mathrm{MHA}}=2\cdot61\cdot128\cdot128\cdot2
=3\,997\,696\ \text{байт}.
$$

Отношение сырых элементов равно

$$
\frac{b_{\mathrm{MHA}}}{b_{\mathrm{MLA}}}
=56.89.
$$

Это не тот же показатель, что опубликованное сокращение 93.3% относительно DeepSeek 67B. Первый — арифметика двух заданных storage geometries; второй — системное сравнение авторов модели.

![VIZ m4/02 — кэш разных схем](assets/modern-llms/ru/module-04/m4_02_kv_per_token.png)

## 6. MQA и GQA: общий K/V не означает одинаковый query

В MHA каждая query-head имеет собственные K/V. В MQA все query-heads сравниваются с одной K/V-head. GQA делит $H_q$ запросных голов на $H_{kv}$ групп.

Если $H_q=64$ и $H_{kv}=8$, то

$$
g=H_q/H_{kv}=8.
$$

Каждая KV-head обслуживает восемь query-heads. При реализации важно повторять каждую KV-head внутри её группы:

```text
kv0, kv0, ..., kv0, kv1, kv1, ..., kv1
```

а не повторять весь массив блоками:

```text
kv0, kv1, ..., kv7, kv0, kv1, ..., kv7.
```

Обе операции дают правильную форму тензора, но второе размещение связывает query-heads с чужими ключами. Поэтому notebook содержит negative test для `tile` против `repeat`.

Математически корректная GQA-реализация эквивалентна MHA с повторёнными K/V-heads. Слово «эквивалентна» относится к функции. В разных объединённых вычислительных ядрах порядок операций с плавающей точкой может немного отличаться, поэтому побитовое совпадение не является универсальным контрактом.

Выбор $H_{kv}$ — компромисс. Меньше KV-heads означает меньший кэш и трафик; больше — больше независимых K/V-подпространств. Никакого универсального «sweet spot = 8» нет: восемь — распространённая конфигурация нескольких семейств, а не математический оптимум.

### NumPy → PyTorch · B05 — GQA: повторение K/V и `enable_gqa`

В NumPy K/V-головы повторяются явно. PyTorch умеет выполнить тот же контракт через `enable_gqa=True`, если исполняющая реализация его поддерживает.

```python
import numpy as np
import torch
import torch.nn.functional as F
```

#### 1. Явная NumPy-реализация

```python
def numpy_gqa_explicit(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    allowed_mask: np.ndarray | None = None,
) -> np.ndarray:
    if q.shape[1] % k.shape[1]:
        raise ValueError("Hq must be divisible by Hkv")
    group = q.shape[1] // k.shape[1]
    k_repeated = np.repeat(k, group, axis=1)
    v_repeated = np.repeat(v, group, axis=1)
    return numpy_sdpa_explicit(q, k_repeated, v_repeated, allowed_mask=allowed_mask)
```

GQA не меняет формулу attention. NumPy явно повторяет каждую K/V-голову `g = H_q/H_{kv}` раз, после чего использует уже разобранный SDPA.

#### 2. Та же математика явными операциями PyTorch

```python
def torch_gqa_explicit(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    allowed_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if q.shape[1] % k.shape[1]:
        raise ValueError("Hq must be divisible by Hkv")
    group = q.shape[1] // k.shape[1]
    k_repeated = k.repeat_interleave(group, dim=1)
    v_repeated = v.repeat_interleave(group, dim=1)
    return torch_sdpa_explicit(q, k_repeated, v_repeated, allowed_mask=allowed_mask)
```

Явный PyTorch-вариант делает то же через `repeat_interleave`. Это полезный эталон для форм и порядка голов, но он материализует повторённые K/V.

#### 3. Оптимизированный или библиотечный PyTorch API

```python
def torch_gqa_optimized(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    allowed_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    if q.shape[1] % k.shape[1]:
        raise ValueError("Hq must be divisible by Hkv")
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=allowed_mask,
        dropout_p=dropout_p,
        enable_gqa=True,
    )
```

`enable_gqa=True` выражает группировку без явного повторения в пользовательском коде; доступность и эффективность конкретного пути зависят от backend.

| Аспект | NumPy | PyTorch |
|---|---|---|
| Условие | `Hq % Hkv == 0` | то же |
| K/V | `np.repeat(..., axis=1)` | `repeat_interleave(..., dim=1)` |
| Память эталона | повторённые массивы | повторённые tensors |
| Оптимизированный путь | нет | `enable_gqa=True` |
| Проверка | совпадение с MHA после повтора | явный путь ↔ API |

<details>
<summary><strong>Исполняемая проверка эквивалентности и граничных случаев</strong></summary>

```python
rng = np.random.default_rng(5)
q_np = rng.standard_normal((1, 4, 3, 8))
k_np = rng.standard_normal((1, 2, 3, 8))
v_np = rng.standard_normal((1, 2, 3, 7))
np_out = numpy_gqa_explicit(q_np, k_np, v_np)
q, k, v = (torch.tensor(a, dtype=torch.float64) for a in (q_np, k_np, v_np))
explicit = torch_gqa_explicit(q, k, v)
api = torch_gqa_optimized(q, k, v, dropout_p=0.0)
torch.testing.assert_close(explicit, torch.from_numpy(np_out), rtol=1e-10, atol=1e-10)
torch.testing.assert_close(api, explicit, rtol=1e-10, atol=1e-10)
print("B05 explicit repetition / enable_gqa: PASS")
```

</details>

Полный исполняемый файл: [`m04_attention_bridges.py`](../assets/m04_attention_bridges.py)

Официальный контракт: [PyTorch SDPA GQA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention).

## 7. MLA: кэшировать латент, а не развернутые ключи и значения

В обычном attention скрытое состояние $h_t$ проецируется в K и V каждой головы. MLA сначала создаёт общий компактный латент:

$$
c_t^{KV}=W^{DKV}h_t,
$$

а затем использует up-projections для вычисления head-specific компонентов. На инференсе в кэше остаётся $c_t^{KV}$ и небольшая позиционная часть.

Почему полные K/V можно не материализовывать? Линейные операции допускают **матричное поглощение (Matrix Absorption)**. Например, произведение query с восстановленным key можно переписать, поглотив key up-projection в матрицу на стороне query. Аналогично value up-projection можно объединить с выходной проекцией. Это не означает нулевую стоимость: латентные проекции и матричные операции остаются. Но полный K/V-вектор каждого токена не нужно хранить или восстанавливать как отдельный большой тензор.

RoPE создаёт тонкость. Для фиксированной позиции вращение линейно, но позиционно-зависимая матрица $R_m$ в общем случае не коммутирует с произвольной проекцией:

$$
R_mW^{UK}\ne W^{UK}R_m.
$$

Поэтому DeepSeek отделяет небольшую RoPE-ветвь ключа и сохраняет её рядом с latent cache. Проблема не в «нелинейности RoPE», а в невозможности поглотить меняющееся с позицией вращение в одну постоянную матрицу.

Авторы DeepSeek-V2 сообщают, что MLA в их ablation setup сопоставима или сильнее сравниваемых MHA/MQA/GQA-вариантов. Это результат конкретного обучения; raw compression ratio сам по себе качество не гарантирует.

![VIZ m4/04 — MLA и латентный кэш](assets/modern-llms/ru/module-04/m4_04_mla_latent.svg)

## 8. FlashAttention: та же плотная функция, другой IO-граф

Стандартная реализация часто разбивается на три шага:

1. вычислить $S=QK^\top/\sqrt d$;
2. записать $S$ и вероятности softmax в HBM;
3. прочитать их и вычислить $PV$.

Математика плотного attention остаётся квадратичной по длине. Проблема в том, что большие промежуточные матрицы перемещаются между HBM и быстрой on-chip **SRAM — Static Random-Access Memory**.

FlashAttention разбивает Q, K и V на тайлы. Для каждого блока запросов он последовательно проходит по блокам ключей, обновляя три состояния строки:

- текущий максимум $m$;
- нормировочную сумму $\ell$;
- взвешенный числитель или output accumulator $n$.

Если новый блок содержит больший максимум, старые $\ell$ и $n$ пересчитываются через $e^{m_{old}-m_{new}}$. В конце выход равен $n/\ell$. Полная матрица вероятностей не записывается в HBM.

![VIZ m4/05 — online softmax](assets/modern-llms/ru/module-04/m4_05_online_softmax.svg)

Что именно улучшается?

| Свойство | Обычная материализующая реализация | FlashAttention |
|---|---|---|
| Плотные $QK^\top$ FLOPs | $O(N^2d)$ | $O(N^2d)$ |
| Полная $N\times N$ матрица в HBM | да | нет |
| Дополнительная память | квадратичная | линейная по $N$ |
| HBM↔SRAM обмен | высокий | меньше; зависит от SRAM и tile size |

IO-сложность нельзя универсально свести к $O(Nd)$. В аналитической модели FlashAttention число HBM-доступов зависит от ёмкости быстрой памяти $M$ и размеров тайлов; работа доказывает улучшение и оптимальность в определённом диапазоне $M$, а не исчезновение квадратичной арифметики.

FlashAttention называется точным, потому что не отбрасывает позиции и реализует ту же математическую функцию. Конкретные вычислительные ядра могут не совпадать побитово с наивным кодом из-за иного порядка операций, смешанной точности и низкоуровневых приближений.

FlashAttention-4 для B200 сообщает до 1613 TFLOP/s в BF16, около 71% теоретического пика, до 1.3× относительно cuDNN 9.13 и до 2.7× относительно Triton в исследованных конфигурациях. Это результаты авторов на заданном оборудовании, а не коэффициент для любой модели. [FlashAttention-4](https://arxiv.org/abs/2603.05451).

![VIZ m4/06 — память FlashAttention](assets/modern-llms/ru/module-04/m4_06_tiling_memory.svg)

## 9. Sliding-window и гибридное attention

**SWA — Sliding Window Attention** разрешает каждому запросу видеть только последние $W$ ключей. При decode локальный слой хранит не весь контекст, а окно. Для одного слоя память перестаёт расти после $W$ токенов.

Сколько информации может пройти через стек? После одного локального слоя токен зависит от окна. После нескольких слоёв путь зависимости может расшириться примерно на $L(W-1)$ позиций. Это верхняя граница достижимости, а не гарантия, что конкретный факт сохранится без искажений. Глубокая цепочка локальных передач отличается от прямого глобального взаимодействия.

Pure sliding-window stacks существуют: Mistral 7B — документированный пример. Гибридные схемы добавляют global layers, чтобы периодически создавать короткий путь между дальними позициями.

| Семейство | Схема |
|---|---|
| Mistral 7B | sliding-window attention во всех decoder layers |
| gpt-oss | 18 local window-128 + 18 full, чередование 1:1 |
| Gemma 2 | чередование local 4096 и global 8192, фактически 1:1 |
| Gemma 4 | hybrid local/global; финальный слой global, дополнительные детали зависят от варианта |

Для оценки KV-cache важно знать **расписание слоёв**, а не только `sliding_window`. Именно поэтому all-full оценка gpt-oss вдвое завышает фактическое состояние на 131K.

![VIZ m4/03 — гибридное расписание слоёв](assets/modern-llms/ru/module-04/m4_03_layer_alternation.svg)

## 10. Два разных attention sink

Термин `attention sink` используется по меньшей мере для двух связанных, но разных механизмов.

### Ранние реальные токены

StreamingLLM обнаружила, что модели часто направляют заметную массу внимания на первые позиции, даже если они не несут текущей семантической информации. При наивном sliding-window удаление этих K/V ломает распределение attention. Практическое решение — хранить несколько начальных реальных токенов вместе с последним окном. Это **наблюдаемое поведение существующей модели**.

### Learned sink-logit

В gpt-oss у каждой query-head есть обучаемый скаляр $s_h$. Он добавляется как дополнительный столбец score matrix. Затем softmax вычисляется по токенам и sink-слоту, после чего sink-колонка удаляется перед умножением на $V$.

Если все 128 token scores равны нулю, доля sink равна

$$
p_{sink}=\frac{e^{s_h}}{128+e^{s_h}}.
$$

При $s_h=0,2,5$ получаем примерно 0.78%, 5.46% и 53.69%. Масса реальных токенов может суммироваться меньше единицы; sink не добавляет value-вклад.

Learned sink **не эквивалентен обычному фиксированному key-вектору**. Обычный key дал бы $q^\top k_{sink}$, зависящий от query. Точная абстракция — дополнительная постоянный столбец логитов с нулевым значением. Через обычный dot product её можно реализовать лишь при расширении пространства постоянной координатой или специальной поддержке вычислительного ядра.

![VIZ m4/08 — два смысла attention sink](assets/modern-llms/ru/module-04/m4_08_sinks.svg)

## 11. Разреженное внимание: выбирать позиции, а не только формат кэша

MQA, GQA и MLA уменьшают состояние **на позицию**. Sliding window ограничивает позиции геометрически. Content-based sparse attention выбирает подмножество по самому запросу.

Удобно сравнивать методы по единице отбора и статусу свидетельств.

| Метод | Единица отбора | Как выбираются позиции | Требуется специальное обучение | Статус |
|---|---|---|---|---|
| block-sparse pattern | блок | фиксированный шаблон | не всегда | зрелая системная техника |
| MoBA — Mixture of Block Attention | блок | обучаемая маршрутизация блоков | да | статья и открытый код |
| NSA — Native Sparse Attention | compressed/block/token branches | обучаемая иерархическая схема | да | статья; результаты авторов |
| DSA — DeepSeek Sparse Attention | токен | лёгкий индексатор (lightning indexer) + top-$k$ | продолженное обучение | открытый отчёт, веса и вычислительные ядра |
| SubQ | заявленная sparse architecture | детали раскрыты ограниченно | неизвестно полностью | заявление разработчика |

MoBA переносит идею Mixture of Experts на блоки attention. NSA объединяет грубое сжатие, выбор отдельных позиций и локальное окно, проектируя алгоритм вместе с аппаратной реализацией. Их опубликованные преимущества относятся к авторским экспериментам. [MoBA](https://arxiv.org/abs/2502.13189), [NSA](https://arxiv.org/abs/2502.11089).

DSA впервые была представлена 29 сентября 2025 года в DeepSeek-V3.2-Exp, построенной продолженным обучением от V3.1-Terminus. Механизм использует лёгкий indexer и затем вычисляет основной механизм внимания только по выбранным токенам. DeepSeek опубликовала отчёт, веса и ключевые kernels. [Официальный анонс](https://api-docs.deepseek.com/news/news250929).

Для новых закрытых или частично раскрытых систем нужно особенно строго разделять заявленный контекст, заявленную асимптотику и воспроизводимый алгоритмический контракт. Большое context window само по себе не сообщает, как выбираются позиции.

## 12. Код первого уровня: прозрачный SDPA, GQA и окно

Ниже — NumPy-референс. Его цель не конкурировать с GPU-ядром, а сделать видимыми формы тензоров, маску, GQA-группировку и политику полностью замаскированных строк.

```python
from typing import Literal

import numpy as np

EmptyRowPolicy = Literal["error", "zero"]

def make_mask(n: int, window: int | None = None, n_sink: int = 0) -> np.ndarray:
    """Аддитивная causal-mask [n, n]: 0 разрешает пару, -inf запрещает."""
    if n <= 0 or (window is not None and window <= 0):
        raise ValueError("n и window должны быть положительными")
    if not 0 <= n_sink <= n or (n_sink and window is None):
        raise ValueError("начальные опорные позиции имеют смысл только вместе с конечным окном")

    i, j = np.arange(n)[:, None], np.arange(n)[None, :]
    allowed = j <= i
    if window is not None:
        allowed &= (i - j) < window
        allowed |= (j < n_sink) & (j <= i)
    return np.where(allowed, 0.0, -np.inf)

def sdpa(Q, K, V, mask=None, *, sink=None, empty_row: EmptyRowPolicy = "error"):
    """Q [..., Tq, dk], K [..., Tk, dk], V [..., Tk, dv] -> [..., Tq, dv].

    Строка без разрешённого ключа не задаёт softmax. Режим ``error`` выявляет
    сломанную маску, а ``zero`` явно возвращает для padding-строки нулевой выход.
    Конечный sink-логит сам создаёт допустимую опору с нулевым значением-вектором.
    """
    if empty_row not in {"error", "zero"}:
        raise ValueError("empty_row должен быть 'error' или 'zero'")

    Q, K, V = (np.asarray(x) for x in (Q, K, V))
    if min(Q.ndim, K.ndim, V.ndim) < 2:
        raise ValueError("Q, K и V должны иметь хотя бы две оси")
    if Q.shape[-1] != K.shape[-1] or K.shape[-2] != V.shape[-2]:
        raise ValueError("несовместимые формы Q/K/V")
    try:
        np.broadcast_shapes(Q.shape[:-2], K.shape[:-2], V.shape[:-2])
    except ValueError as exc:
        raise ValueError("batch-оси Q/K/V несовместимы для broadcasting") from exc
    if not all(np.isfinite(x).all() for x in (Q, K, V)):
        raise ValueError("Q, K и V должны содержать только конечные значения")

    # Float16-входы переводим в float32 до матричного умножения: иначе
    # переполнение может произойти раньше, чем результат попадёт в широкий аккумулятор.
    dtype = np.result_type(Q.dtype, K.dtype, V.dtype, np.float32)
    Q, K, V = (x.astype(dtype, copy=False) for x in (Q, K, V))
    scale = np.asarray(1.0 / np.sqrt(Q.shape[-1]), dtype=dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        scores = (Q @ np.swapaxes(K, -1, -2)) * scale
    if not np.isfinite(scores).all():
        raise FloatingPointError("Q @ K.T переполнилось; уменьшите масштаб входов")

    if mask is not None:
        mask = np.asarray(mask, dtype=dtype)
        if np.isnan(mask).any() or np.isposinf(mask).any():
            raise ValueError("маска может содержать finite bias и -inf, но не NaN/+inf")
        try:
            mask = np.broadcast_to(mask, scores.shape)
        except ValueError as exc:
            raise ValueError("форма маски несовместима с матрицей scores") from exc
        with np.errstate(over="ignore", invalid="ignore"):
            scores = scores + mask
        if np.isnan(scores).any() or np.isposinf(scores).any():
            raise FloatingPointError("конечный bias маски переполнился; уменьшите его величину")

    has_token = np.isfinite(scores).any(axis=-1, keepdims=True)
    token_max = scores.max(axis=-1, keepdims=True)
    if sink is None:
        if empty_row == "error" and not has_token.all():
            raise ValueError("полностью замаскированная строка attention")
        row_max = np.where(has_token, token_max, np.zeros_like(token_max))
        sink_logits = None
    else:
        sink_logits = np.asarray(sink, dtype=dtype)
        if not np.isfinite(sink_logits).all():
            raise ValueError("sink-логиты должны быть конечными")
        try:
            sink_logits = np.broadcast_to(sink_logits, token_max.shape)
        except ValueError as exc:
            raise ValueError("форма sink несовместима с query-строками") from exc
        row_max = np.maximum(token_max, sink_logits)

    token_exp = np.where(
        np.isfinite(scores), np.exp(scores - row_max), np.zeros_like(scores)
    )
    denominator = token_exp.sum(axis=-1, keepdims=True)
    if sink_logits is not None:
        denominator += np.exp(sink_logits - row_max)  # нормировка без вклада в V

    weights = np.divide(
        token_exp, denominator, out=np.zeros_like(token_exp), where=denominator > 0
    )
    return weights @ V

def gqa(Q, K, V, n_heads: int, n_kv_heads: int, mask=None):
    """Учебный GQA для 2D-проекций; возвращает [T, H*dv]."""
    Q, K, V = (np.asarray(x) for x in (Q, K, V))
    if any(x.ndim != 2 for x in (Q, K, V)):
        raise ValueError("GQA ожидает двумерные projected-последовательности")
    if not (Q.shape[0] == K.shape[0] == V.shape[0]):
        raise ValueError("Q, K и V должны иметь одинаковую длину T")
    if n_heads <= 0 or n_kv_heads <= 0 or n_heads % n_kv_heads:
        raise ValueError("n_heads должен быть положительным кратным n_kv_heads")
    if Q.shape[1] % n_heads or K.shape[1] % n_kv_heads or V.shape[1] % n_kv_heads:
        raise ValueError("ширины Q/K/V должны делиться на соответствующее число голов")

    T = Q.shape[0]
    d_k = Q.shape[1] // n_heads
    if K.shape[1] // n_kv_heads != d_k:
        raise ValueError("размерности query- и key-голов должны совпадать")
    d_v = V.shape[1] // n_kv_heads
    q = Q.reshape(T, n_heads, d_k).transpose(1, 0, 2)
    k = K.reshape(T, n_kv_heads, d_k).transpose(1, 0, 2)
    v = V.reshape(T, n_kv_heads, d_v).transpose(1, 0, 2)
    repeats = n_heads // n_kv_heads
    # repeat даёт [kv0, kv0, ..., kv1, kv1, ...], то есть сохраняет группы.
    k = np.repeat(k, repeats, axis=0)
    v = np.repeat(v, repeats, axis=0)
    z = sdpa(q, k, v, mask)
    return z.transpose(1, 0, 2).reshape(T, n_heads * d_v)
```

В коде `n_sink` означает начальные **реальные** позиции, сохраняемые вместе с окном. Параметр `sink` в `sdpa` — другой механизм: отдельный learned score без value. Это различие специально проверяется в notebook.

## 14. Читаем реальную реализацию gpt-oss

Открытый код gpt-oss позволяет увидеть, как несколько идей модуля соединены в одном блоке.

**GQA.** Query преобразуется в форму

```text
[tokens, H_kv, H_q/H_kv, head_dim],
```

а K/V хранятся как

```text
[tokens, H_kv, head_dim].
```

При вычислении K/V расширяются на размер группы. Это делает отношение $H_q/H_{kv}=8$ явным в формах тензора.

**Hybrid attention.** Для слоёв с чётным индексом применяется `sliding_window=128`, для остальных окно равно нулю, то есть attention полный. Отсюда 18 local и 18 full layers.

**Learned sink.** Параметр `sinks` содержит по скаляру на query-head. Реализация конкатенирует его к score matrix перед softmax и удаляет последний столбец после нормировки, перед умножением на $V$. Это прямое подтверждение score-slot, а не обычного key-вектора.

**RoPE.** Query и key проходят rotary embedding до сопоставления; позиционная геометрия синхронизирована с модулем 3.

![VIZ m4/09 — читаем конфигурацию и код](assets/modern-llms/ru/module-04/m4_09_config_attention.svg)

## 15. Современный ландшафт: читать механизм вместе с основанием

Таблица ниже не пытается назначить одного победителя. Она показывает, какие части паспорта можно восстановить из открытого источника.

| Семейство / работа | Состояние на позицию | Видимые позиции | Основание |
|---|---|---|---|
| Llama 3.x | GQA | full attention | открытый config / implementation |
| Mistral 7B | GQA | sliding window | статья и открытый config |
| DeepSeek-V2/V3 | MLA | full или последующие sparse extensions | технические отчёты |
| gpt-oss | GQA | alternating full/window-128 + learned sink | открытый код и config |
| Gemma 2 | GQA | alternating local/global 1:1 | официальная документация |
| Gemma 4 | hybrid attention | local/global, final global | официальный model card/config |
| MoBA | обычный или совместимый KV | выбранные блоки + local | статья и открытый код |
| NSA | собственная многоветвенная схема | compressed/selective/window | статья, результаты авторов |
| DeepSeek-V3.2-Exp | латентный KV-кэш MLA | DSA top-$k$ + локальная структура | официальный отчёт, веса, kernels |
| закрытая API-модель | неизвестно, если не раскрыто | неизвестно, если не раскрыто | context length недостаточно для вывода |

Эта таблица подчёркивает четыре независимые оси:

1. сколько KV-heads или латентных координат хранится;
2. сколько слоёв full/local/sparse;
3. как выбираются видимые позиции;
4. какое вычислительное ядро исполняет выбранную математику.

Одинаковый номинальный контекст может иметь совершенно разную стоимость. И наоборот, одинаковый механизм внимания может работать по-разному из-за dtype кэша, batch size и реализации.

## 16. Мультимодальность: где встречаются визуальные и текстовые токены

В **early-fusion** архитектуре визуальные embeddings вставляются в последовательность языкового decoder. Тогда каждый визуальный токен получает обычное KV-состояние, определённое геометрией decoder.

Если гипотетический decoder имеет all-full GQA-эквивалент 72 KiB/token, 4096 визуальных токенов добавили бы

$$
4096\cdot72\ \text{KiB}=288\ \text{MiB}
$$

KV-cache. Это учебная оценка нейтральной архитектуры. Опубликованная gpt-oss является только текстовой моделью; выражение `gpt-oss-multimodal` не обозначает официальную контрольную точку.

В **cross-attention** архитектуре визуальный encoder вычисляет features отдельно, а языковые слои обращаются к ним через cross-attention. Визуальный encoder не нужно повторно запускать на каждом decoder layer. Однако разные cross-attention layers могут иметь собственные K/V-проекции и кэш: точная стоимость зависит от реализации.

Таким образом, мультимодальный connector определяет не только совместимость размерностей, но и место, где оплачивается attention state. Этот вопрос подробно развивается в модуле 14.

## 19. Ключевые выводы модуля

![VIZ m4/10 — attention за одну страницу](assets/modern-llms/ru/module-04/m4_10_cheatsheet.svg)

- **KV-cache считается по слоям.** Формула $2LH_{kv}d_hb$ применима к all-full layout; hybrid schedule нужно суммировать отдельно.
- **MQA/GQA уменьшают число KV-heads.** Размер группы — инженерный выбор, а не универсальная константа качества.
- **MLA меняет объект хранения.** Raw ratio 56.89× относится к заданным размерностям; опубликованное 93.3% сокращение — другое сравнение.
- **FlashAttention не делает dense attention линейным.** Он убирает материализацию квадратичной матрицы в HBM и уменьшает IO.
- **Sliding window уменьшает состояние и вычисления локальных слоёв.** Дальняя связь через глубину возможна, но не гарантирована.
- **Attention sink имеет два смысла.** Начальный реальный токен и learned score-slot нельзя смешивать.
- **Разреженное внимание выбирает подмножество позиций.** Сравнивать нужно правило отбора, требования к обучению, вычислительное ядро и статус источника.
- **Длина контекста не является паспортом attention.** Для оценки нужны геометрия кэша, расписание слоёв, формат данных и исполняющая реализация.

## 20. и источники

**Основные первоисточники:**

- Vaswani et al., *Attention Is All You Need* — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need* — [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
- Ainslie et al., GQA — [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- DeepSeek-V2 / MLA — [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
- FlashAttention — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- FlashAttention-4 — [arXiv:2603.05451](https://arxiv.org/abs/2603.05451)
- Mistral 7B — [arXiv:2310.06825](https://arxiv.org/abs/2310.06825)
- StreamingLLM — [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
- MoBA — [arXiv:2502.13189](https://arxiv.org/abs/2502.13189)
- NSA — [arXiv:2502.11089](https://arxiv.org/abs/2502.11089)
- DSA / DeepSeek-V3.2-Exp — [официальный анонс](https://api-docs.deepseek.com/news/news250929)
- gpt-oss — [официальный репозиторий](https://github.com/openai/gpt-oss)

**Углубление.** Системное управление KV-cache, paging, offloading и prefill/decode disaggregation рассматриваются в модуле 9. Архитектурный контекст мультимодального внимания — в модуле 14.

---

*Ландшафт сверен: 5 августа 2026 года. Числа памяти являются воспроизводимыми расчётами заданных конфигураций; внешние коэффициенты производительности сохраняют условия и атрибуцию авторов.*
