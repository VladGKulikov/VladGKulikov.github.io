# Модуль 5. Нормализация, активации и анатомия блока

*Курс «Modern LLMs» · лекция модуля 5 · редакция 2026.8*

> **О чём этот модуль.** В модуле 4 мы разобрали, как attention смешивает информацию между позициями. Но Transformer-блок состоит не только из attention. Между подслоями проходит общий residual-поток, его масштаб контролируют нормализации, а большую часть параметров плотного блока обычно занимает гейтованный FFN. Здесь мы соберём эти детали в единую инженерную картину: сравним LayerNorm и RMSNorm, разведём Post-LN, Pre-LN, Peri-LN и схему OLMo2, разберём Query-Key Normalization, выведем бюджет SwiGLU и до последнего параметра восстановим Llama-3.1-8B. Затем посмотрим, что именно меняют Dynamic Tanh, DyISRU и Manifold-Constrained Hyper-Connections, не смешивая исследовательские идеи с уже устоявшимися производственными рецептами.
>
> **Что нужно знать заранее.** Достаточно материала модулей 1–4: формы attention-проекций, Grouped-Query Attention и базовая арифметика параметров. Все остальные определения вводятся здесь.

---

## 1. Мотивация: почему «мелочи» блока определяют судьбу обучения

В конфигурации модели строки `rms_norm_eps`, `intermediate_size` или `swiglu_limit` легко принять за второстепенные настройки. На деле они описывают три разных механизма устойчивости.

Нормализация определяет, в каком масштабе подслои читают residual-поток. Ширина FFN задаёт большую часть параметрического бюджета плотного блока. Ограничение диапазона активации способно изменить численное поведение модели в низкой точности. Ошибка в любой из этих деталей не обязательно проявится как немедленный `NaN`: чаще она меняет градиентный маршрут, распределение активаций или точный вычислительный граф.

Чтобы увидеть масштабы, мы будем возвращаться к трём воспроизводимым примерам.

- В линейной 36-слойной игрушке residual-поток без нормы вырастает до RMS около 412 тысяч, тогда как Pre-LN даёт 6.42, а Post-LN по определению удерживает выход каждого шага на RMS 1.
- Для Llama-3.1-8B матричные веса одного блока дают ровно $13d^2$, но полный блок содержит ещё две шкалы RMSNorm: $13d^2+2d$.
- В публичной реализации gpt-oss активация SwiGLU ограничивает gate сверху, up-ветвь с двух сторон и использует сдвиг $(u+1)$. Это часть обученного графа, а не настройка декодера.

Модуль будет двигаться от общей residual-шины к норме, затем к FFN и, наконец, к современным попыткам заменить уже саму нормализацию или даже правило residual-смешивания.

## 2. Историческая прогрессия: несколько ответов на одну проблему глубины

![VIZ m5/01 — как менялась внутренняя конструкция блока](assets/modern-llms/ru/module-05/m5_01_block_bus.svg)

**Layer Normalization** появилась в 2016 году как способ нормировать признаки одного примера, не полагаясь на статистику мини-пакета. Оригинальный Transformer использовал **Post-LN**:

$$x_{l+1}=\operatorname{LN}\bigl(x_l+F_l(x_l)\bigr).$$

Позднее широкое распространение получила **Pre-LN**:

$$x_{l+1}=x_l+F_l\bigl(\operatorname{LN}(x_l)\bigr).$$

Работа Xiong et al. объяснила одно важное различие: при инициализации у Post-LN ожидаемые градиенты рядом с выходом могут быть велики, поэтому большая скорость обучения (learning rate) без прогрева (warmup) создаёт нестабильность; у Pre-LN градиенты в их анализе ведут себя лучше, и в экспериментах прогрев удалось убрать. Это не теорема о том, что любой Post-LN без прогрева обязан разойтись. Это объяснение, почему расположение нормы меняет начальную геометрию оптимизации.

**RMSNorm** сохранила только нормировку масштаба и отказалась от вычитания среднего. Исходная работа показала сопоставимое качество в рассмотренных задачах и сокращение времени на 7–64% в конкретных экспериментальных системах. В современных LLM выигрыш обычно меньше, потому что норма занимает лишь часть блока и часто выполняется слитым ядром, но простота RMSNorm сделала её распространённым выбором.

Параллельно развивалась вторая половина блока. GLU-варианты, особенно **SwiGLU**, заменили одиночную активацию мультипликативным взаимодействием двух проекций. Это повысило выразительность FFN при контролируемом бюджете.

Дальше ветви расходятся. **DeepNorm** изменяет масштабирование residual-связи и инициализацию, чтобы стабилизировать очень глубокие Post-LN Transformer. **Peri-LN** размещает нормализацию с обеих сторон подслоя и контролирует масштаб записываемого в residual-поток приращения. **OLMo2** использует другое правило: норма применяется после attention/FFN внутри residual-ветви, то есть $x+\operatorname{Norm}(F(x))$, без предварительной нормы входа подслоя. Эти схемы нельзя объединять под одним названием.

В 2025–2026 годах появились более радикальные предложения. **Dynamic Tanh (DyT)** заменяет редукцию по признакам поэлементной функцией. **Manifold-Constrained Hyper-Connections (mHC)** расширяет residual-поток до нескольких параллельных потоков и ограничивает матрицы их смешивания. Они меняют разные части блока и пока не отменяют основной рецепт RMSNorm + гейтованный FFN.

## 3. Мостик к классике: стандартизация, численное интегрирование и гейты

Три классических аналогии помогают читать блок, если не превращать их в буквальные тождества.

**Нормализация и стандартизация.** LayerNorm вычитает среднее и делит на стандартное отклонение по признакам одного токена. RMSNorm делит только на корень среднего квадрата. Ни одна из них не является декорреляцией (whitening): ковариационная матрица не диагонализируется и пространство признаков не поворачивается. Полезнее думать о норме как о локальном контроле масштаба и частичном предобусловливании (preconditioning): последующие матрицы получают входы в более предсказуемом диапазоне.

**Residual-связь и шаг Эйлера.** Формула

$$h_{l+1}=h_l+F_l(h_l)$$

похожа на дискретизацию дифференциального уравнения. Глубина играет роль времени, а подслой добавляет приращение. Аналогия объясняет интерес к размеру шага и устойчивости, но реальный Transformer не является одной стационарной ODE: функции $F_l$ различаются, а шаг не обязан быть малым.

**Гейтованный FFN и мультипликативные взаимодействия.** В SwiGLU две обученные проекции взаимодействуют поэлементно. Одна ветвь создаёт содержимое, другая регулирует его пропускание. Без нелинейности произведение двух линейных функций было бы квадратичным; с SiLU получается более общий входозависимый мультипликативный механизм. Это роднит GLU с вентилями рекуррентных сетей, но не делает их одинаковыми архитектурами.

## 4. Формализм: residual-поток как общий носитель состояния

Для Pre-LN decoder-блока удобно написать две строки:

$$
\tilde h_l=h_l+\operatorname{Attn}\bigl(N_{l,1}(h_l)\bigr),
$$

$$
h_{l+1}=\tilde h_l+\operatorname{FFN}\bigl(N_{l,2}(\tilde h_l)\bigr).
$$

Residual-поток — это векторное состояние ширины $d_{\text{model}}$, которое проходит через весь стек. Подслои читают из него нормированное представление и записывают поправки через выходные проекции.

У такой конструкции есть прямой тождественный путь. Jacobian шага содержит компонент $I$:

$$
\frac{\partial h_{l+1}}{\partial h_l}
=I+\frac{\partial F_l}{\partial h_l}.
$$

Этот путь помогает передавать сигнал и градиент через глубину. Но из него не следует, что любой обученный слой можно удалить безболезненно. После обучения конкретный подслой может хранить критически важное преобразование; прореживание (pruning) и досрочный выход (early exit) требуют эмпирической проверки или специального обучения.

Ширина residual-потока определяет формы почти всех матриц блока. Поэтому $d$ — главный параметр бюджета, но точное число параметров зависит также от числа KV-heads, ширины FFN, смещений и связанных embedding-весов, а также числа норм.

## 5. LayerNorm и RMSNorm: что сохраняется, а что отбрасывается

Для $x\in\mathbb R^d$ LayerNorm вычисляет

$$
\operatorname{LN}(x)_i=
\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i,
$$

а RMSNorm —

$$
\operatorname{RMSNorm}(x)_i=
\gamma_i\frac{x_i}{\sqrt{d^{-1}\sum_jx_j^2+\epsilon}}.
$$

Возьмём $x=(2,-1,3,0)$, $\gamma=1$, $\beta=0$. Здесь $\mu=1$, $\sigma^2=2.5$, а RMS равен 1.8708. Получаем

$$
\operatorname{LN}(x)\approx(0.632,-1.265,1.265,-0.632),
$$

$$
\operatorname{RMSNorm}(x)\approx(1.069,-0.535,1.604,0).
$$

RMSNorm не центрирует вектор. До применения покоординатной $\gamma$ она сохраняет нули, знаки и отношения координат, меняя общий масштаб. Исходная работа показывает, что в проверенных авторами задачах инвариантности к масштабированию (re-scaling invariance) оказалось достаточно для качества, сопоставимого с LayerNorm. Из этого не следует, что центрирование бесполезно для любой архитектуры.

### Зачем нужен $\epsilon$

Если RMS близок к нулю, деление превращает норму в усилитель шума. Верхняя граница множителя равна

$$
\frac1{\sqrt\epsilon}.
$$

При RMS $10^{-3}$ множитель равен примерно 707 для $\epsilon=10^{-6}$ и 302 для $10^{-5}$. Значение $\epsilon$ выбирают не только по machine epsilon: важны точность редукции, тип активаций, масштаб рабочих сигналов и реализация ядра. Типичный диапазон $10^{-6}$–$10^{-5}$ является инженерным выбором конкретных семейств, а не универсальным выводом из одного dtype.

Многие эталонные реализации временно переводят вход в fp32, вычисляют средний квадрат и `rsqrt`, а затем возвращают результат в исходный dtype. Так устроен, например, публичный PyTorch-код gpt-oss. Объединённые вычислительные ядра (fused kernels) могут использовать другую схему накопления, поэтому точную точность редукции следует проверять в реализации.

### NumPy → PyTorch · B07 — RMSNorm: редукция, dtype и autograd

NumPy показывает редукцию среднего квадрата. PyTorch добавляет autograd, device/dtype-контракт и возможность выбрать fused kernel.

```python
import numpy as np
import torch
import torch.nn.functional as F
```

#### 1. Явная NumPy-реализация

```python
def numpy_rms_norm(x: np.ndarray, weight: np.ndarray | None = None, *, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 0 or not np.isfinite(x).all() or not np.isfinite(eps) or eps <= 0:
        raise ValueError("x must be finite and eps positive")
    work = x.astype(np.float64, copy=False)
    y = work / np.sqrt(np.mean(work * work, axis=-1, keepdims=True) + eps)
    if weight is not None:
        w = np.asarray(weight, dtype=np.float64)
        if w.shape != (x.shape[-1],):
            raise ValueError("weight must have shape [D]")
        y = y * w
    return y.astype(x.dtype, copy=False) if np.issubdtype(x.dtype, np.floating) else y
```

NumPy-эталон показывает средний квадрат по последней размерности, добавление $\epsilon$, обратный корень и покоординатный обучаемый масштаб.

#### 2. Та же математика явными операциями PyTorch

```python
def torch_rms_norm_explicit(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    rms_inv = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    y = x * rms_inv
    return y if weight is None else y * weight
```

Явный tensor-вариант почти строка в строку повторяет формулу. В реальной mixed-precision модели отдельно решают, в каком dtype выполнять редукцию.

#### 3. Оптимизированный или библиотечный PyTorch API

```python
def torch_rms_norm(x: torch.Tensor, weight: torch.Tensor | None = None, *, eps: float = 1e-6) -> torch.Tensor:
    return F.rms_norm(x, [x.shape[-1]], weight=weight, eps=eps)
```

`F.rms_norm` скрывает редукцию внутри библиотечного оператора и может использовать оптимизированную реализацию.

| Аспект | NumPy | PyTorch |
|---|---|---|
| Редукция | `np.mean(..., axis=-1)` | `mean(dim=-1)` |
| Обратный корень | `1 / np.sqrt` | `torch.rsqrt` |
| Вес | broadcast массива `[D]` | parameter `[D]` с градиентом |
| Накопление | float64 в эталоне | зависит от явного/ fused пути |
| Градиент | нет | проверяется `gradcheck` |

<details>
<summary><strong>Исполняемая проверка эквивалентности и граничных случаев</strong></summary>

```python
rng = np.random.default_rng(7)
x_np = rng.standard_normal((3, 5)); w_np = rng.standard_normal(5)
np_out = numpy_rms_norm(x_np, w_np, eps=1e-6)
x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
w = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
explicit = torch_rms_norm_explicit(x, w, eps=1e-6)
api = torch_rms_norm(x, w, eps=1e-6)
np.testing.assert_allclose(np_out, explicit.detach().numpy(), rtol=1e-10, atol=1e-10)
torch.testing.assert_close(api, explicit, rtol=1e-10, atol=1e-10)
assert torch.autograd.gradcheck(lambda a, b: torch_rms_norm_explicit(a, b, eps=1e-6), (x, w))
print("B07 explicit NumPy / PyTorch / F.rms_norm: PASS")
```

</details>

Полный исполняемый файл: [`m05_block_bridges.py`](../assets/m05_block_bridges.py)

Официальный API: [PyTorch `rms_norm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.rms_norm.html).

## 6. Где поставить норму: Post-LN, Pre-LN, Peri-LN и OLMo2

![VIZ m5/02 — четыре размещения нормы](assets/modern-llms/ru/module-05/m5_02_norm_placements.svg)

**Post-LN**:

$$x_{l+1}=N\bigl(x_l+F_l(x_l)\bigr).$$

Норма находится на сквозном пути. Выход каждого шага контролируется по масштабу, но чистого identity path через стек нет.

**Pre-LN**:

$$x_{l+1}=x_l+F_l\bigl(N(x_l)\bigr).$$

Residual stream остаётся нетронутым, что облегчает оптимизацию. Приращения при этом суммируются, и масштаб потока может расти с глубиной.

**Peri-LN**:

$$x_{l+1}=x_l+N_{\text{out}}\!\left(F_l(N_{\text{in}}(x_l))\right).
$$

Вторая норма контролирует сам вклад подслоя перед записью в residual stream. Исследование Peri-LN анализирует более сбалансированный рост дисперсии и градиентов на моделях до 3.2B параметров.

**схема OLMo2**:

$$x_{l+1}=x_l+N\bigl(F_l(x_l)\bigr).
$$

OLMo2 применяет RMSNorm после attention и FFN, а также отдельно нормирует векторы query и key. Здесь нет предварительной нормы входа подслоя, поэтому это не Peri-LN.

### Игрушечный эксперимент

В детерминированной линейной цепочке

$$h\leftarrow h+Wh$$

используются один и тот же $h_0$ и один набор матриц для трёх режимов. После 36 слоёв RMS равен примерно 412 442 без нормы, 6.42 в Pre-LN-подобной схеме и ровно 1 в Post-LN-подобной, потому что каждый шаг явно приводится к единичному RMS.

![VIZ m5/03 — динамика RMS по глубине](assets/modern-llms/ru/module-05/m5_03_depth_variance.png)

Этот опыт ничего не говорит о шуме градиента, оптимизаторе или обучаемой $\gamma$. Он лишь показывает, как размещение нормы меняет прямую динамику масштаба.

## 7. Query-Key Normalization: контролируем масштаб логитов attention

В обычном Scaled Dot-Product Attention score имеет вид

$$
s_{ij}=\frac{q_i^\top k_j}{\sqrt{d_h}}.
$$

Деление на $\sqrt{d_h}$ компенсирует типичный рост дисперсии при фиксированном масштабе компонент. Но нормы $q$ и $k$ обучаемы и могут расти.

Исходная **Query-Key Normalization (QKNorm)** L2-нормирует каждый вектор query и key по размерности головы и заменяет фиксированное деление на обучаемый масштаб. Авторы получили среднее улучшение 0.928 BLEU на пяти низкоресурсных парах перевода в своём протоколе.

Современные модели используют семейство родственных приёмов:

- L2-нормализация query/key;
- RMSNorm для query/key;
- один обучаемый скаляр или покоординатный масштаб;
- soft-capping логитов attention;
- ограничение (clipping) или настройка температуры (temperature tuning).

Их нельзя автоматически называть одной и той же QKNorm. Например, Llama 4 поддерживает L2-нормализацию Q/K на RoPE-слоях, когда она включена в конкретном checkpoint; OLMo2 и Qwen3 применяют RMSNorm к векторам query и key, а Gemma 2 использует soft-capping логитов attention. Конкретный механизм следует читать по коду модели или техническому отчёту.

## 8. FFN: почему одна нелинейность уступила гейтованной паре

Оригинальный Transformer использовал

$$
\operatorname{FFN}(x)=W_2\phi(W_1x),
$$

обычно с промежуточной шириной $4d$. ReLU позднее часто заменяли на GELU или SiLU/Swish, потому что гладкие функции дают ненулевые производные на более широком диапазоне.

GLU-семейство вводит две параллельные проекции:

$$
\operatorname{GLU}(x)=
(W_{\text{up}}x)\odot\phi(W_{\text{gate}}x).
$$

После поэлементного произведения результат возвращается к ширине $d$ через $W_{\text{down}}$. Shazeer сравнил ReGLU, GEGLU и SwiGLU и получил улучшения относительно негейтованных вариантов в исследованной Transformer-постановке.

Главное отличие — не просто третья матрица. Гейт создаёт мультипликативное взаимодействие двух независимо обученных признаков. Это позволяет модели менять пропускание каналов в зависимости от входа.

Численный мост между GELU и сигмоидной формой:

$$
\operatorname{GELU}(z)\approx z\,\sigma(1.702z).
$$

На $[-6,6]$ максимальная абсолютная ошибка в нашем воспроизводимом расчёте равна примерно 0.0203.

![VIZ m5/04 — активации FFN](assets/modern-llms/ru/module-05/m5_04_activations.png)

## 9. SwiGLU: три матрицы, правило паритета и вариант gpt-oss

Стандартная Llama-style формула:

$$
\operatorname{FFN}(x)=
W_{\text{down}}\left[
\operatorname{SiLU}(W_{\text{gate}}x)
\odot W_{\text{up}}x
\right].
$$

Число матричных параметров равно

$$
3d\,d_{\text{ff}}.
$$

Чтобы сравнять его с классическим FFN ширины $4d$, решаем

$$
3d\,d_{\text{ff}}=2d(4d),
$$

откуда

$$
d_{\text{ff}}=\frac83d.
$$

Это параметрический паритет, а не рекомендация для каждой модели. Реальная ширина выбирается как отдельный архитектурный гиперпараметр с учётом качества, бюджета и аппаратного выравнивания.

### Публичный SwiGLU gpt-oss

Эталонный код gpt-oss использует

$$
g\leftarrow\min(g,7),
\qquad
u\leftarrow\operatorname{clip}(u,-7,7),
$$

$$
\operatorname{out}
=g\,\sigma(1.702g)\,(u+1).
$$

Для диагонального примера $g=u=z$:

| $z$ | выход |
|---:|---:|
| -9 | 0.00001 |
| -2 | 0.06434 |
| 0 | 0 |
| 2 | 5.80698 |
| 7 | 55.99963 |
| 9 | 55.99963 |

![VIZ m5/05 — три матрицы и диапазон gpt-oss](assets/modern-llms/ru/module-05/m5_05_swiglu_gate.svg)

![VIZ m5/06 — действие swiglu_limit](assets/modern-llms/ru/module-05/m5_06_clamp_curve.png)

Плато при $z=7$ и $z=9$ возникает потому, что обе ветви уже ограничены. Но утверждение «отрицательный clip бесполезен» неверно: при $z=-9$ gate действительно подавлен сигмоидой, однако up-ветвь всё равно заменяется на $-7$, а затем участвует как $(u+1)=-6$.

Ограничение диапазона совместимо с требованиями низкоточной арифметики и уменьшает выбросы. Но если источник не формулирует мотив явно, лекция не должна приписывать разработчикам единственную причину. Аналогично сдвиг $(u+1)$ можно интерпретировать как сохранение ненулевого value-множителя при $u\approx0$, но это описание эффекта, а не доказанная история проектного решения.

### NumPy → PyTorch · B08 — SwiGLU и ограниченный вариант

Обе реализации разделяют gate- и up-ветви. Параметры `alpha`, ограничение диапазона и сдвиг up-ветви задаются явно.

```python
import numpy as np
import torch
```

#### 1. Явная NumPy-реализация

```python
def numpy_swiglu(
    gate: np.ndarray,
    up: np.ndarray,
    *,
    alpha: float = 1.0,
    gate_limit: float | None = None,
    up_limit: float | None = None,
    shift_up: float = 0.0,
) -> np.ndarray:
    gate, up = np.asarray(gate), np.asarray(up)
    if gate.shape != up.shape or not np.isfinite(gate).all() or not np.isfinite(up).all():
        raise ValueError("gate and up must be finite arrays of the same shape")
    g = gate.astype(np.float64, copy=False)
    u = up.astype(np.float64, copy=False)
    if gate_limit is not None:
        g = np.minimum(g, gate_limit)
    if up_limit is not None:
        u = np.clip(u, -up_limit, up_limit)
    y = g / (1.0 + np.exp(-alpha * g)) * (u + shift_up)
    return y.astype(np.result_type(gate.dtype, up.dtype), copy=False)
```

Обе ветви видны явно: gate проходит через SiLU-подобную нелинейность и затем умножается на up-ветвь. Параметры `alpha`, ограничения и сдвиг не прячутся в имени функции.

#### 2. Та же математика явными операциями PyTorch

```python
def torch_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    alpha: float = 1.0,
    gate_limit: float | None = None,
    up_limit: float | None = None,
    shift_up: float = 0.0,
) -> torch.Tensor:
    if gate.shape != up.shape:
        raise ValueError("gate and up must have the same shape")
    g, u = gate, up
    if gate_limit is not None:
        g = g.clamp(max=gate_limit)
    if up_limit is not None:
        u = u.clamp(min=-up_limit, max=up_limit)
    return g * torch.sigmoid(alpha * g) * (u + shift_up)
```

PyTorch заменяет `np.exp` на `torch.sigmoid` и сохраняет граф производных. `clamp` вводит точки излома, поэтому градиентную parity проверяют вдали от границ.

| Аспект | NumPy | PyTorch |
|---|---|---|
| Нелинейность | явная сигмоида | `torch.sigmoid` |
| Ограничение gate | `np.minimum` | `Tensor.clamp(max=...)` |
| Ограничение up | `np.clip` | `Tensor.clamp(min=..., max=...)` |
| Обычная SwiGLU | без limits, `shift_up=0` | тот же контракт |
| Градиенты | нет | autograd; точки clamp проверяются отдельно |

<details>
<summary><strong>Исполняемая проверка эквивалентности и граничных случаев</strong></summary>

```python
rng = np.random.default_rng(8)
g_np = rng.uniform(-4, 4, (3, 5)); u_np = rng.uniform(-4, 4, (3, 5))
kwargs = dict(alpha=1.702, gate_limit=7.0, up_limit=7.0, shift_up=1.0)
np_out = numpy_swiglu(g_np, u_np, **kwargs)
g = torch.tensor(g_np, dtype=torch.float64, requires_grad=True); u = torch.tensor(u_np, dtype=torch.float64, requires_grad=True)
t_out = torch_swiglu(g, u, **kwargs)
np.testing.assert_allclose(np_out, t_out.detach().numpy(), rtol=1e-10, atol=1e-10)
assert torch.autograd.gradcheck(lambda a, b: torch_swiglu(a, b, **kwargs), (g, u))
print("B08 forward + gradient: PASS")
```

</details>

Полный исполняемый файл: [`m05_block_bridges.py`](../assets/m05_block_bridges.py)

## 10. Решённый пример: Llama-3.1-8B до последнего параметра

Используем

$$
V=128256,
\quad d=4096,
\quad L=32,
\quad d_{\text{ff}}=14336,
$$

$$
H_q=32,
\quad H_{kv}=8,
\quad d_h=128.
$$

Смещения отсутствуют, входная и выходная embedding-матрицы не связаны.

**Attention.** Q и O стоят по $d^2$, K и V вместе — $0.5d^2$:

$$
P_{\text{attn}}=2.5d^2=41\,943\,040.
$$

**FFN.** Три матрицы SwiGLU:

$$
P_{\text{ffn}}=3d\,d_{\text{ff}}
=176\,160\,768.
$$

**Нормы.** Две RMSNorm добавляют

$$
P_{\text{norm}}=2d=8192.
$$

Следовательно, матричная часть блока равна

$$
P_{\text{matrix block}}
=2.5d^2+10.5d^2
=13d^2
=218\,103\,808,
$$

а полный блок:

$$
P_{\text{full block}}
=13d^2+2d
=218\,112\,000.
$$

Полная модель:

$$
2Vd+L(13d^2+2d)+d
=8\,030\,261\,248.
$$

![VIZ m5/07 — точный бюджет блока](assets/modern-llms/ru/module-05/m5_07_param_budget.svg)

FFN занимает около 80.8% полного блока, attention — около 19.2%, нормы — менее 0.004%. Поэтому фраза «блок стоит $13d^2$» полезна как матричная прикидка, но при обещании счёта до последнего параметра нужно добавить $2d$.

## 11. DyT и DyISRU: можно ли заменить редукцию поэлементной функцией

**Dynamic Tanh (DyT)** предлагает

$$
\operatorname{DyT}(x)=\gamma\odot\tanh(\alpha x)+\beta.
$$

Здесь нет редукции по признакам: каждая координата преобразуется независимо. Работа 2025 года показала сопоставимое или лучшее качество в ряде vision- и language-экспериментов, часто без существенной перенастройки.

Эта поэлементность принципиально отличает DyT от RMSNorm. У RMSNorm выход одной координаты зависит от соседей через общий знаменатель; у DyT — нет.

Работа о математической связи нормализации и динамических активаций выводит **Dynamic Inverse Square Root Unit (DyISRU)** как точный поэлементный counterpart RMSNorm после специального decoupling:

$$
\operatorname{DyISRU}(x)
=\gamma\odot\frac{x}{\sqrt{1+(\alpha x)^2}}+\beta.
$$

«Точный» здесь относится к полученной поэлементной конструкции после decoupling, а не к равенству обычной векторной RMSNorm.

В ноутбуке курса отдельно строится **собственная скалярная аппроксимация**: на выборке $x\sim\mathcal N(0,1)$ подбирается $\alpha$ для приближения тождественного преобразования с единичным RMS функцией $\tanh(\alpha x)$. Получается около 1.43. Это число принадлежит учебному эксперименту, а не статье DyT.

![VIZ m5/08 — поэлементные альтернативы нормализации](assets/modern-llms/ru/module-05/m5_08_dyt_family.png)

По состоянию на 6 августа 2026 года в приведённых в модуле открытых конфигурациях крупных моделей DyT не используется как основной рецепт. Это наблюдение по выбранному набору артефактов, а не доказательство отсутствия любой крупной закрытой модели. Работа об optimizer-normalization coupling показывает сильный отрицательный эффект Dynamic Erf с Muon, но DyT в том же эксперименте такого штрафа не показала. Следовательно, эту работу нельзя использовать как аргумент против DyT; она лишь напоминает, что новые нормализаторы нужно проверять вместе с выбранным оптимизатором.

## 12. mHC: ограничиваем не скрытые состояния, а матрицы residual-смешивания

Обычная residual connection переносит один поток:

$$h_{l+1}=h_l+F_l(h_l).$$

**Hyper-Connections** расширяют residual-поток до нескольких параллельных потоков и позволяют обучаемым матрицам смешивать их до и после подслоя. Это повышает выразительность, но свободное смешивание может разрушить identity mapping и сделать глубокую сеть нестабильным.

**Manifold-Constrained Hyper-Connections (mHC)** ограничивает пространство матриц смешивания residual-потоков (residual-mixing matrices). В опубликованной конструкции матрица `comb` проецируется через Sinkhorn–Knopp на множество дважды стохастических матриц, связанное с многогранником Биркгофа (Birkhoff polytope). В DeepSeek-V4 несколько residual-потоков сохраняются внутри блока, а перед выходной головой схлопываются обратно в одну последовательность.

Таким образом, «manifold» относится прежде всего к допустимым матрицам residual-смешивания, а не к гипотезе о том, что траектория скрытых состояний (hidden states) лежит на низкоразмерном многообразии. Это архитектурное изменение самого маршрута сигнала, тогда как Peri-LN сохраняет обычное сложение и нормирует приращение.

После исходной работы появились независимые теоретические и инфраструктурные продолжения, включая альтернативные ограничения и ускорение проекции на многогранник Биркгофа. Они не равны независимому воспроизведению DeepSeek-V4 в полном масштабе, но показывают, что тема уже вышла за пределы одной публикации.

**Углубление.** Контекст применения mHC вместе с Muon и другими решениями претрейна обсуждается в модуле 8.

## 13. Код первого уровня: RMSNorm, SwiGLU и residual block

Ниже остаются три учебных примитива: устойчивая сигмоида, RMSNorm с повышенной точностью редукции и два варианта SwiGLU. Attention передаётся в блок явно, чтобы не повторять код модуля 4.

```python
import numpy as np

def sigmoid(x):
    """Устойчивый sigmoid без переполнения exp на больших отрицательных x."""
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -x))

def rms_norm(x, gamma, eps=1e-6):
    """Нормализация [..., d_model] по последней оси; статистика минимум float32."""
    x = np.asarray(x)
    work_dtype = np.float32 if x.dtype.itemsize < 4 else x.dtype
    x_hi = x.astype(work_dtype, copy=False)
    ms = np.mean(x_hi * x_hi, axis=-1, keepdims=True)
    y = x_hi / np.sqrt(ms + eps)
    return (y * np.asarray(gamma, dtype=work_dtype)).astype(x.dtype, copy=False)

def swiglu(x, W_gate, W_up, W_down):
    """Стандартный SwiGLU: SiLU(gate) ⊙ up, затем down-проекция."""
    gate, up = x @ W_gate, x @ W_up
    return (gate * sigmoid(gate) * up) @ W_down

def swiglu_oss(x, W_gate, W_up, W_down, alpha=1.702, limit=7.0):
    """Публичный gpt-oss-вариант: clamp, alpha=1.702 и сдвиг (up + 1)."""
    gate = np.minimum(x @ W_gate, limit)
    up = np.clip(x @ W_up, -limit, limit)
    return (gate * sigmoid(alpha * gate) * (up + 1.0)) @ W_down

class PreNormBlock:
    """Две residual-ветки; attention-функция передаётся явно."""
    def __init__(self, norm1, norm2, attention, ffn):
        self.norm1, self.norm2 = norm1, norm2
        self.attention, self.ffn = attention, ffn

    def __call__(self, x, mask):
        attn_update = self.attention(self.norm1(x), mask)
        h = x + attn_update
        ffn_update = self.ffn(self.norm2(h))
        return h + ffn_update
```

Проверьте два инварианта. Норма вызывается ровно один раз перед каждым подслоем, а выходная проекция применяется после поэлементного произведения ветвей gate и up. Учебный код намеренно не реализует Peri-LN, схему OLMo2 или mHC: эти варианты требуют явного изменения вычислительного графа, а не одного флага в формуле RMSNorm.

## 15. Читаем публичную реализацию gpt-oss

В `gpt_oss/torch/model.py` хорошо видны три механизма этого модуля.

**RMSNorm.** Вход переводится в fp32, нормируется через `mean(x**2)` и `rsqrt`, затем возвращается в исходный dtype. Значение $\epsilon$ в публичном классе RMSNorm равно $10^{-5}$.

**Активация.** `swiglu(x, alpha=1.702, limit=7.0)` разделяет packed tensor на ветвей gate и up, применяет асимметричные ограничения и возвращает $g\sigma(\alpha g)(u+1)$.

**Порядок блока.** Attention и MLP получают нормированный вход, а их результаты прибавляются к residual-потоку — это Pre-LN.

Важная деталь: в официальном `ModelConfig` есть `hidden_size`, `intermediate_size`, `swiglu_limit`, число экспертов и top-k, но нет поля `hidden_act`. Конкретная функция активации задаётся кодом модели. Поэтому паспорт архитектуры строится из конфигурации **вместе** с реализацией.

```json
{
  "hidden_size": 2880,
  "intermediate_size": 2880,
  "num_experts": 128,
  "experts_per_token": 4,
  "swiglu_limit": 7.0
}
```

![VIZ m5/09 — паспорт блока gpt-oss](assets/modern-llms/ru/module-05/m5_09_config_block.svg)

Равенство `intermediate_size == hidden_size` относится к одному эксперту. Полный бюджет определяется количеством экспертов и маршрутизацией, поэтому его арифметика продолжится в модуле 6.

## 16. Современные блоки: не каталог победителей, а паспорт свидетельств

| Семейство | Нормализация и размещение | FFN / активация | Q/K-стабилизация | Основание |
|---|---|---|---|---|
| Llama 3.x | RMSNorm, Pre-LN | SwiGLU | отдельная QK-норма не указана в открытом config | официальная конфигурация / код модели |
| Llama 4 | RMSNorm, Pre-LN-подобная residual-связь | SiLU-гейтованный MoE/MLP | L2-нормализация Q/K на RoPE-слоях при `use_qk_norm=True`; зависит от checkpoint | официальная конфигурация / код модели |
| Gemma 2 | RMSNorm, нормы до и после подслоя | GELU-tanh gated MLP | soft-capping логитов attention | официальная конфигурация / код модели |
| Gemma 3 | RMSNorm, нормы до и после подслоя | `gelu_pytorch_tanh` gated MLP | RMSNorm для Q и K; soft-capping в стандартной конфигурации отключён | официальная конфигурация / код модели |
| OLMo2 | RMSNorm после attention и FFN внутри residual-ветви | SiLU-гейтованный MLP | отдельная RMSNorm для Q и K | официальная документация / код |
| gpt-oss | RMSNorm, Pre-LN | SwiGLU с clamp с $(u+1)$ | learned attention sinks, не QK-norm | официальный код OpenAI |
| DeepSeek-V4 | RMSNorm внутри sublayers + mHC residual mixing | SwiGLU с clamp в экспертах | архитектурно зависит от типа attention | технический отчёт / открытая реализация |
| Qwen3 | RMSNorm, Pre-LN | SiLU-гейтованный MLP | RMSNorm для Q и K по размерности головы | официальная конфигурация / код модели |

Таблица показывает, почему нельзя переносить свойство между поколениями одной семьи. Llama 3 и Llama 4 различаются по контракту Q/K-нормализации, причём он может зависеть от конкретного checkpoint Llama 4. OLMo2 нельзя называть Peri-LN. Gemma 2 и Gemma 3 имеют разные значения soft-capping по умолчанию. Защитные механизмы оптимизации вроде MuonClip не следует помещать в колонку нормализации блока.

Для закрытой модели отсутствие архитектурных полей означает «не раскрыто», а не «вероятно, как у предыдущего поколения».

## 17. Мультимодальность: тот же блок, но не одна история происхождения

Vision Transformer использует знакомые компоненты: normalization, multi-head attention, FFN и residual connections. Поэтому арифметика блока переносится на vision encoder после подстановки его $d$, числа голов и ширины MLP.

История заимствований при этом не односторонняя. Query-Key Normalization была опубликована в 2020 году для низкоресурсного машинного перевода. ViT-22B позднее стала заметным крупномасштабным примером Q/K-нормализации в компьютерном зрении. Поэтому корректнее говорить не «QK-norm пришла из vision в LLM», а «идея возникла в Transformer-литературе и получила важную масштабную проверку в ViT».

При чтении мультимодальной конфигурации полезно разделять текстовую и визуальную части: они могут использовать разные `hidden_size`, `intermediate_size`, активацию, epsilon и размещение норм.

## 20. Ключевые выводы модуля

![VIZ m5/10 — анатомия блока](assets/modern-llms/ru/module-05/m5_10_cheatsheet.svg)

- Residual connection создаёт прямой путь, но не обещает безболезненного удаления обученного слоя.
- RMSNorm контролирует масштаб без центрирования; $\epsilon$ задаёт поведение на малых активациях.
- Post-LN, Pre-LN, Peri-LN и схема OLMo2 — разные вычислительные графы.
- Query-Key Normalization — семейство механизмов; L2 QKNorm, RMSNorm для Q/K и soft-capping логитов следует различать.
- Gated FFN использует три матрицы и мультипликативное взаимодействие двух ветвей.
- Для Llama-3.1-8B матричная часть блока равна $13d^2$, а полный блок — $13d^2+2d$.
- Публичный gpt-oss использует SwiGLU с clamp, определённый в коде модели, а не полем `hidden_act`.
- DyT и DyISRU являются поэлементными альтернативами; результат $\alpha\approx1.43$ в notebook — собственная аппроксимация курса.
- mHC ограничивает матрицы residual-смешивания нескольких потоков, а не скрытых состояний (hidden states) на низкоразмерном многообразии.

## 21. Ноутбук и первоисточники

**Основные первоисточники:**

- Ba, Kiros, Hinton. [Layer Normalization](https://arxiv.org/abs/1607.06450)
- Zhang, Sennrich. [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- Xiong et al. [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- Wang et al. [DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555)
- Kim et al. [Peri-LN](https://arxiv.org/abs/2502.02732)
- Henry et al. [Query-Key Normalization for Transformers](https://arxiv.org/abs/2010.04245)
- Shazeer. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- Zhu et al. [Transformers without Normalization](https://arxiv.org/abs/2503.10622)
- Stollenwerk. [On the Mathematical Relationship Between Layer Normalization and Dynamic Activation Functions](https://arxiv.org/abs/2503.21708)
- Xie et al. [Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880)
- OpenAI. [gpt-oss reference implementation](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py)
- Документация моделей Hugging Face: [OLMo2](https://huggingface.co/docs/transformers/model_doc/olmo2), [Llama 4](https://huggingface.co/docs/transformers/model_doc/llama4), [Gemma 2](https://huggingface.co/docs/transformers/model_doc/gemma2), [Gemma 3](https://huggingface.co/docs/transformers/model_doc/gemma3), [Qwen3](https://huggingface.co/docs/transformers/model_doc/qwen3) и [DeepSeek-V4](https://huggingface.co/docs/transformers/model_doc/deepseek_v4)
- Официальный код Transformers для [OLMo2](https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmo2/modular_olmo2.py), [Llama 4](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama4.py), [Gemma 3](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modular_gemma3.py) и [Qwen3](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)

**Дальше:** модуль 6 рассматривает Mixture of Experts. Узкий expert gpt-oss с `intermediate_size=2880`, 128 экспертами и top-4 станет отправной точкой для арифметики total и active parameters.

---

*Ландшафт сверен: 6 августа 2026 года. Численные примеры воспроизводятся локальным контрактом; эмпирические результаты сохраняют условия первоисточников; архитектурные свойства привязаны к официальным конфигурациям, коду или техническим отчётам.*
