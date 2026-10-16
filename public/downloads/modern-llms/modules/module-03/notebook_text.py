"""Independent RU/EN explanations for the M3 notebook."""
TEXT = {
    'ru': {
        'title': r"""# Ноутбук 3. Базовый Transformer""",
        'intro': r"""Здесь мы превращаем формулы лекции в работающую маленькую модель. Сначала вычислим позиции и внимание на готовых векторах, затем соберём блок и обучим его предсказывать короткую периодическую последовательность.

**Первый запуск:** выполните все ячейки сверху вниз; читайте пояснения, формы и результаты. **Второй проход:** изменяйте входы и маски, предсказывая результат до запуска. **Справка:** §2 — внимание и PyTorch, §3 — маска кэша, §4 — головы, §6 — утечка будущего, §7 — обучение.

Обязательный путь работает на CPU без файлов репозитория, ключей API и загрузки данных. Установка отсутствующих PyTorch и Matplotlib требует сети; после установки вычисления выполняются офлайн. Эталонные числа встроены в ноутбук и проверяются по SHA-256. Это учебная реализация: она показывает промежуточные матрицы и проверки, а не измеряет быстродействие промышленного движка.

Формы аргументов: `q [..., T_q, d_k]`, `k [..., T_k, d_k]`, `v [..., T_k, d_v]`. Последняя ось весов внимания — ключи. В маске `True` означает «связь разрешена».""",
        'reference': r"""## 0. Встроенные эталонные числа

Числа принадлежат этому модулю и не зависят от других лекций. Следующая ячейка проверяет, что встроенная запись JSON не изменилась. Это проверка целостности; правильность вычислений проверим отдельно по результатам.""",
        'positional': r"""## 1. Позиции: одинаковый токен в разных местах

Для каждой позиции $p$ строим вектор из пар синуса и косинуса:

$$PE_{p,2i}=\sin(p/10000^{2i/d_{model}}),\qquad PE_{p,2i+1}=\cos(p/10000^{2i/d_{model}}).$$

Позиция начинается с нуля; $i$ нумерует пары координат. Таблица имеет форму `[T,d_model]`, её строка 0 состоит из пар `(0,1)`. Предскажите эту строку до запуска. На графике быстрые и медленные координаты разделены, чтобы сравнение не терялось среди пересечений.""",
        'attention': r"""## 2. Внимание: оценки → веса → смесь

В одной голове запросы $Q$ сравниваются с ключами $K$. Softmax превращает оценки в веса для значений $V$:

$$S=QK^\top/\sqrt{d_k},\qquad A=\operatorname{softmax}_{keys}(S),\qquad Z=AV.$$

Для числового примера запретим будущие ключи до softmax. Проверьте первую строку: доступен только первый ключ, поэтому выход должен совпасть с первым значением.

Если маска запрещает все ключи строки, обычного распределения нет. `empty_row="error"` сообщает об ошибке; явно выбранный `"zero"` возвращает нулевые веса и выход, например для позиции PAD. Остальные строки должны суммироваться в 1. Реализация считает минимум в float32; float64 сохраняется. Она возвращает объект с полями `output`, `weights`, `scores`, `empty_rows`, чтобы можно было осмотреть каждый этап.""",
        'scaling': r"""## 3. Масштаб оценок и координаты маски

Случайные независимые координаты с нулевым средним и единичной дисперсией дают $\operatorname{Var}(q\cdot k)=d_k$. Деление на $\sqrt{d_k}$ уменьшает дисперсию до 1. Выборочная оценка колеблется; проверяем допуск 8% на 50 000 пар. Это модель для объяснения масштаба, а не допущение о любой обученной сети.

При чтении кэша локальная строка запроса 0 может быть позицией 4 всей последовательности. `query_offset=4` разрешает ей пять ключей. Сравните с ошибочным смещением 0. На графике также есть объединение каузального запрета и запрета ключей PAD; строки запросов PAD здесь отдельно не удаляются.""",
        'mha': r"""## 4. Несколько голов и заполнители PAD

Полный вход сначала проходит проекции; затем их результат делится по головам:

```text
[B,T,d_model] → [B,H,T,d_head] → [B,T,d_model].
```

Каждая голова использует все входные признаки. `token_valid [B,T]` отмечает настоящие токены. В этой реализации он запрещает и ключи, и запросы PAD. Для пустых строк выбираем `empty_row="zero"`; выходная проекция не имеет смещения, поэтому сохраняет нулевой выход. Это свойство данного attention-подслоя, а не автоматическая гарантия для полного блока с остаточными связями.""",
        'block': r"""## 5. Собираем блок

LayerNorm вычисляет среднее и дисперсию по признакам одной позиции, с делением на число признаков. Проверим ручной пример, затем добавим две остаточные ветви:

$$x\leftarrow x+\operatorname{Attn}(\operatorname{LN}(x)),\qquad x\leftarrow x+\operatorname{FFN}(\operatorname{LN}(x)).$$

В каждой строке правая часть использует текущее значение `x`; вторая строка получает результат первой. Учебный блок использует pre-norm и GELU, тогда как исходная статья — post-norm и ReLU.""",
        'causality': r"""## 6. Проверка отсутствия утечки будущего

Тест должен обнаруживать ошибку, если убрать маску. Поэтому используем отдельную прозрачную модель: $Q=K=0$, а значения напрямую передают заданные векторы. Внимание равномерно смешивает доступные значения. Изменённый будущий токен гарантированно влияет на ранние выходы без маски; с маской ранние выходы сохраняются.""",
        'tiny_lm': r"""## 7. От обучения к продолжению последовательности

Модель с одним блоком учит период `0,1,2,3,4,5` за 121 шаг. Вход и цель сдвинуты на одну позицию; `cross_entropy` получает логиты, без предварительного softmax. Проверьте снижение NLL, долю правильных предсказаний при известном префиксе и **только новые** сгенерированные токены.

Этот период можно выучить по одному предыдущему токену, поэтому успех не доказывает, что модель научилась дальним зависимостям. Нет отдельной тестовой выборки: задача проверяет работоспособность обучения и генерации. Здесь нет KV-кэша. При превышении окна генератор оставляет последние токены и заново нумерует обучаемые позиции; основной опыт укладывается в окно.""",
        'final': r"""## 8. Арифметика и намеренные ошибки

Сверим softmax и NLL для четырёх логитов, затем посчитаем только веса основных проекций для $d_{model}=512$, $d_{ff}=2048$. Это не полное число параметров модели: смещения, нормы и таблицы эмбеддингов исключены.

Последние проверки подают некорректные формы, тип маски, ширину и идентификатор токена. Ожидаем явную ошибку, а не незаметно изменённый расчёт.""",
        'takeaway': r"""## Что унести в следующий модуль

Сохраните четыре проверки: нормировка по ключам, нулевые веса запрещённых связей, явная политика пустой строки и неизменность ранних выходов при замене будущих токенов. В М4 будем менять способ представления позиций, сохраняя остальной вычислительный путь.""",
        'messages': {'reference': 'Встроенные числа М3 проверены', 'positional': 'Позиционная таблица и строка 0 проверены', 'attention': 'Числовой пример и политики пустой строки проверены', 'scaling': 'Масштаб оценок и смещение запросов проверены', 'mha': 'Формы голов, PAD и нормировка проверены', 'block': 'LayerNorm и блок проверены', 'causality': 'Тест утечки будущего пройден', 'tiny': 'Модель выучила период и продолжила его', 'final': 'NLL, число весов и отрицательные проверки пройдены'},
    },
    'en': {
        'title': r"""# Notebook 3. The Basic Transformer""",
        'intro': r"""This notebook follows the path from token vectors to a working language model. Start with positions and attention, assemble a block, and train it on a short periodic sequence.

**First run:** execute all cells in order and follow the explanations, shapes, and outputs. **Second pass:** change an input or mask and predict the effect before running it. **Reference:** §2 covers attention and PyTorch parity, §3 cached-query positions, §4 heads, §6 causality, and §7 training.

The required path uses CPU and embedded data; no repository files or API keys are needed. Missing PyTorch or Matplotlib packages require a network connection for installation. After that, the computation is offline. SHA-256 checks the embedded numerical reference. This implementation exposes intermediate tensors for learning; it is not a serving-engine benchmark.

Shapes are `q [..., T_q, d_k]`, `k [..., T_k, d_k]`, and `v [..., T_k, d_v]`. Attention weights normalize the last, key axis. Boolean `True` permits a connection.""",
        'reference': r"""## 0. Embedded numerical contract

The reference numbers belong to Module 3 and are independent of lecture prose. Compact
JSON is embedded in the notebook; SHA-256 ties together the builder, answer sheet, and
published `.ipynb`.""",
        'positional': r"""## 1. Sinusoidal positional encoding

For position $p$ and coordinate pair $i$:

\[
PE_{p,2i}=\sin\!\left(p/10000^{2i/d_{model}}\right),\qquad
PE_{p,2i+1}=\cos\!\left(p/10000^{2i/d_{model}}\right).
\]

The table has shape `[T,d_model]`; position zero consists of `(0,1)` pairs.""",
        'attention': r"""## 2. Scaled dot-product attention and an empty row

We compute

\[
S=QK^\top/\sqrt{d_k},\qquad A=\operatorname{softmax}_{keys}(S),\qquad Z=AV.
\]

After masking, a row may lose **all** admissible keys. Softmax over
`[-inf, …, -inf]` is undefined. The educational implementation requires an explicit
policy:

- `empty_row="error"` — the safe default;
- `empty_row="zero"` — zero weights and zero output for an explicitly padded query.

The implementation computes in at least float32 and retains float64 inputs. Its result exposes `output`, `weights`, `scores`, and `empty_rows` so each stage can be inspected.""",
        'scaling': r"""## 3. The $1/\sqrt{d_k}$ scale and causal-mask geometry

Monte Carlo checks only the sanity model of independent standard coordinates:
`Var(q·k) ≈ d_k`, and the scaled variance is near one. It is not a claim that trained
coordinates are independent.

We also test `query_offset`: during cached decode, local query row zero may have global
position four and read five accumulated keys.

The chart also combines padding-key and causal constraints; it does not independently suppress padded query rows.""",
        'mha': r"""## 4. Multi-head reshape and padding

The critical transition is

```text
[B,T,d_model] → [B,H,T,d_head] → [B,T,d_model].
```

`token_valid [B,T]` forbids both padded keys and padded queries. The latter use
`empty_row="zero"` so they cannot create `NaN`.

Each head projects from the entire input width. The output projection has no bias, so it preserves zero attention outputs for PAD queries. A full residual block need not keep those positions zero.""",
        'block': r"""## 5. LayerNorm, the FFN, and a pre-norm decoder block

The hand calculation uses population variance. We then assemble

\[
x\leftarrow x+\operatorname{Attn}(\operatorname{LN}(x)),\qquad
x\leftarrow x+\operatorname{FFN}(\operatorname{LN}(x)).
\]

This is a modern pre-norm decoder block, not the exact post-norm graph of the 2017 paper.""",
        'causality': r"""## 6. A deterministic future-leakage test

The probe sets $Q=K=0$, so attention uniformly mixes the allowed values. The unmasked
control is therefore **guaranteed** to depend on the future, whereas the causal version
is not. The test does not rely on a lucky random initialization.""",
        'tiny_lm': r"""## 7. Train a model, then ask it to continue

A one-block model learns `0,1,2,3,4,5` in 121 updates. Inputs and targets are shifted by one position. Pass logits directly to `cross_entropy`; it handles normalization internally. Check the decrease in NLL, accuracy with known prefixes, and accuracy on **newly generated** tokens only.

One preceding token is enough to predict this period, so success does not demonstrate long-range reasoning or a need for attention. There is no held-out test set: this exercise checks that training and generation work. The generator recomputes the context without a KV cache. Beyond its window it crops old tokens and restarts learned position indices; the required experiment stays within the window.""",
        'final': r"""## 8. Next-token NLL, parameter floor, and negative tests

The final cell verifies softmax/NLL and the bias/norm-free parameter floor for the
original dimensions $d_{model}=512$, $d_{ff}=2048$. It then pins the failure modes:
odd width, an invalid mask shape, incompatible $Q/K/V$, a fully masked row, and an ID
outside the vocabulary.""",
        'takeaway': r"""## Takeaway

Four invariants can now move into every later block:

- softmax normalizes the key axis;
- forbidden links receive zero weight;
- a fully masked row requires an explicit policy;
- changing a future token does not alter earlier states in a causal model.

You may replace details—RoPE, RMSNorm, GQA, SwiGLU, fused SDPA—one at a time while
retaining these checks as a regression net.""",
        'messages': {'reference': 'Embedded Module 3 contract verified', 'positional': 'Sinusoidal table and position zero verified', 'attention': 'Numerical attention and both empty-row policies verified', 'scaling': 'Scaling and cached-decode causal geometry verified', 'mha': 'Multi-head reshape, padding, and normalization verified', 'block': 'LayerNorm and decoder block verified', 'causality': 'Deterministic future-leakage test passed', 'tiny': 'Tiny LM overfit and generated the period', 'final': 'NLL, parameter floor, and negative tests passed'},
    },
}
