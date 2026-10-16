# Модуль 6. Нормализация, активации и анатомия блока

*Курс «Modern LLMs» · редакция 11 октября 2026 года*

В М5 мы разобрали, как внимание передаёт информацию между позициями. Теперь посмотрим на весь блок: как он сохраняет уже найденные признаки, добавляет новые и удерживает вычисления в подходящем численном диапазоне. По ходу соберём полный блок и рассчитаем его параметры.

**Предпосылки.** Нужны матричные проекции и остаточная связь из М3, а также головы внимания и GQA из М5. Формулы ниже используют векторы-столбцы; NumPy хранит токены строками и поэтому умножает их на матрицы справа.

**Маршруты чтения**

- **Первый проход:** §1 → §§4–10, пропуская раскрываемые реализации → §§14–15 → §20. Проверьте себя по В1–В6 в §18 и У1–У3 в §19; В7 и У7 используют mHC из §12 и относятся к следующему проходу. В §10 проследите путь от формы каждой матрицы до общего числа параметров. Этого достаточно, чтобы перейти к М7.
- **Второй проход:** вернитесь к B07/B08 в §§5/9, коду §13 и ноутбуку. Проверьте малые и большие входы, ось нормализации и совпадение градиентов; выполните упражнения 1–3 и 6–7.
- **Справочный проход:** §§2–3 связывают материал с историей и классикой; §§11–12 разбирают DyT, DyISRU и несколько остаточных потоков; §§16–17 помогают читать конфигурации реальных моделей. Проекты 4–5 расширяют практику.

## 1. Что делает блок помимо внимания

У каждого токена есть вектор состояния $h\in\mathbb R^d$. Его координаты — признаки, а $d$ — ширина модели. **Остаточный поток (residual stream)** — это состояние, которое проходит через слои и получает добавки от подслоёв. Остаточная связь имеет вид $h+F(h)$: функция $F$ вычисляет поправку той же формы, что и вход.

Внимание смешивает информацию разных позиций. **Полносвязный подслой FFN (feed-forward network)** затем преобразует каждую позицию отдельно, используя одни и те же веса для всех токенов. Нормализация управляет масштабом признаков на входе или выходе этих операций. Сам токен при этом остаётся на своей позиции.

Масштаб удобно измерять среднеквадратичным значением:

$$\operatorname{RMS}(h)=\sqrt{\frac1d\sum_{i=1}^d h_i^2}.$$

В отличие от среднего, RMS не обнуляется при взаимном сокращении положительных и отрицательных координат. Например, у $(1,-1)$ среднее равно нулю, а RMS — единице.

Дальше мы ответим на три вопроса. Как именно нормализация меняет вектор? Где её поставить относительно остаточного сложения? Сколько параметров добавляет FFN, когда одна проекция управляет другой? Для проверки используем численный пример нормы, цепочку из 36 линейных шагов и точный бюджет Llama-3.1-8B.

## 2. Как менялась конструкция блока

Этот раздел можно оставить на справочный проход. Названия ниже обозначают разные изменения блока; их формулы появятся в §§5–12.

LayerNorm (2016) нормирует признаки одного примера без статистики мини-пакета. В исходном Transformer она стояла после остаточного сложения — **Post-LN**. В **Pre-LN** норма перенесена на вход подслоя. Анализ Xiong et al. связал различия начальных градиентов с необходимостью прогрева скорости обучения в исследованной постановке; это не универсальный запрет на Post-LN без прогрева. [Статья](https://arxiv.org/abs/2002.04745).

RMSNorm (2019) убрала вычитание среднего. Авторы получили сопоставимое качество и сокращение времени на 7–64% в своих экспериментальных системах. Переносить этот диапазон на современный блок без измерения нельзя: результат зависит от доли нормализации в вычислениях и её реализации. [RMSNorm](https://arxiv.org/abs/1910.07467).

Параллельно развивались гейтованные FFN: две проекции перемножаются, и одна регулирует вклад другой (§8). DeepNorm меняет масштаб остаточной связи и инициализацию для очень глубоких сетей. Peri-LN нормирует вход и выход подслоя, а OLMo2 — выход подслоя перед сложением (§6). DyT и DyISRU исследуют поэлементные замены нормы (§11); mHC меняет способ соединения нескольких потоков (§12). Эти решения затрагивают разные части вычисления и могут сочетаться.

## 3. Мостик к классике: стандартизация, численное интегрирование и гейты

Три классических аналогии помогают читать блок, если не превращать их в буквальные тождества.

**Нормализация и стандартизация.** LayerNorm вычитает среднее и делит на стандартное отклонение по признакам одного токена. RMSNorm делит только на корень среднего квадрата. Ни одна из них не является декорреляцией (whitening): ковариационная матрица не диагонализируется и пространство признаков не поворачивается. Полезнее думать о норме как о локальном контроле масштаба и частичном предобусловливании (preconditioning): последующие матрицы получают входы в более предсказуемом диапазоне.

**Остаточная связь и шаг Эйлера.** Формула

$$h_{l+1}=h_l+F_l(h_l)$$

похожа на дискретизацию дифференциального уравнения. Глубина играет роль времени, а подслой добавляет приращение. Аналогия объясняет интерес к размеру шага и устойчивости, но реальный Transformer не является одной системой дифференциальных уравнений с неизменной правой частью: функции $F_l$ различаются, а шаг не обязан быть малым.

**Гейтованный FFN и мультипликативные взаимодействия.** В SwiGLU две обученные проекции взаимодействуют поэлементно. Одна ветвь создаёт содержимое, другая регулирует его пропускание. Без нелинейности произведение двух линейных функций было бы квадратичным; с SiLU получается более общий входозависимый мультипликативный механизм. Это роднит GLU с вентилями рекуррентных сетей, но не делает их одинаковыми архитектурами.

## 4. Остаточный поток: два последовательных приращения

Рассмотрим **Pre-LN**: нормализация стоит перед каждым подслоем. Для слоя $l$ обозначим два его нормализатора через $N_{l,1}$ и $N_{l,2}$. Тогда

$$\tilde h_l=h_l+\operatorname{Attn}(N_{l,1}(h_l)),$$
$$h_{l+1}=\tilde h_l+\operatorname{FFN}(N_{l,2}(\tilde h_l)).$$

FFN читает уже обновлённое состояние $\tilde h_l$. Складываются только векторы одинаковой ширины $d$; внутренние проекции внимания и FFN могут иметь другие размеры. Для последовательности обе строки применяются к матрице состояний всех токенов, причём внимание использует маску из М5.

![VIZ m6/01 — Внимание обновляет состояние первым; FFN читает результат этого сложения.](assets/modern-llms/ru/module-05/m6_01_block_bus.svg)
*Внимание обновляет состояние первым; FFN читает результат этого сложения.*

Для одного шага $h^+=h+F(h)$ матрица производных, или **якобиан**, равна

$$J=\frac{\partial h^+}{\partial h}=I+\frac{\partial F}{\partial h}.$$

Здесь $I$ — единичная матрица. Она представляет прямой путь через сложение, однако итоговый градиент зависит от всей суммы. Например, при $F(h)=-h$ получается $J=0$. Поэтому наличие обходного пути помогает проектировать глубокие сети, но само по себе не гарантирует ненулевой градиент или возможность удалить обученный слой. Эта граница будет полезна при сравнении размещений нормы.

## 5. LayerNorm и RMSNorm: две операции над одним вектором

Нормализация действует по признакам **одного токена**, а не по последовательности или пакету. Сведение набора координат к одной статистике — например, среднему — называется **редукцией**. Для $x\in\mathbb R^d$ определим среднее $\mu=d^{-1}\sum_i x_i$ и дисперсию $\sigma^2=d^{-1}\sum_i(x_i-\mu)^2$. Делитель здесь $d$, а не $d-1$: нам нужна статистика данного вектора, а не несмещённая оценка дисперсии популяции.

**LayerNorm** сначала центрирует вектор, затем меняет масштаб:

$$\operatorname{LN}(x)_i=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i.$$

Векторы $\gamma,\beta\in\mathbb R^d$ — обучаемые масштабы и сдвиги, а $\epsilon>0$ защищает знаменатель около нуля. **RMSNorm** использует средний квадрат без центрирования:

$$\operatorname{RMSNorm}(x)_i=\gamma_i\frac{x_i}{\sqrt{d^{-1}\sum_jx_j^2+\epsilon}}.$$

В этой распространённой форме у RMSNorm нет обучаемого сдвига. Для $x=(2,-1,3,0)$, $\gamma_i=1$, $\beta_i=0$ и $\epsilon=10^{-6}$ получаем $\mu=1$, $\sigma^2=2.5$, $\operatorname{RMS}(x)=\sqrt{3.5}\approx1.8708$:

$$\operatorname{LN}(x)\approx(0.632,-1.265,1.265,-0.632),$$
$$\operatorname{RMSNorm}(x)\approx(1.069,-0.535,1.604,0).$$

У первого результата среднее нулевое. Второй до умножения на $\gamma$ сохраняет нули, знаки и отношения ненулевых координат. Обучаемая шкала меняет отношения координат, а при отрицательных значениях — и знаки; нули сохраняются. Постоянный ненулевой вектор LayerNorm без сдвига превратит в нулевой, а RMSNorm сохранит ненулевым — простой способ увидеть, что операции не эквивалентны.

### Малый сигнал, epsilon и точность

Без обучаемой шкалы множитель RMSNorm не превосходит $1/\sqrt\epsilon$. При RMS входа $10^{-3}$ он равен примерно 707.1 для $\epsilon=10^{-6}$ и 301.5 для $10^{-5}$. Это предел **множителя до $\gamma$**, а не универсальный предел всего выхода.

Инвариантность к положительному масштабированию тоже имеет условие. При $\epsilon=0$ и ненулевом $x$ нормированные $x$ и $cx$ совпадают для $c>0$. При фиксированном $\epsilon>0$:

$$
\frac{cx}{\sqrt{c^2\operatorname{RMS}(x)^2+\epsilon}}=
\frac{x}{\sqrt{\operatorname{RMS}(x)^2+\epsilon/c^2}}.
$$

Для приближённой инвариантности нужны оба условия: $\operatorname{mean}(x^2)\gg\epsilon$ и $c^2\operatorname{mean}(x^2)\gg\epsilon$. Очень маленький положительный $c$ может нарушить второе условие даже при большом исходном сигнале. Когда энергия сигнала сравнима с $\epsilon$, разница заметна. В ноутбуке есть и обычный вход, и контрпример около нуля. Нулевой вектор при положительном $\epsilon$ обрабатывается без деления на ноль.

FP16 — 16-битный формат, FP32 — 32-битный. В FP16 квадрат большого конечного числа может переполниться, а малого — округлиться до нуля. Поэтому повышать точность нужно **до возведения в квадрат**, а не только при суммировании. Такой порядок показан ниже; выбор $\epsilon$ и формата накопления остаётся частью контракта конкретной реализации.

**Второй проход: RMSNorm в NumPy, PyTorch и библиотечном API**

### NumPy → PyTorch · B07 — RMSNorm: редукция, dtype и autograd

Сравниваем конечный вещественный вход `[..., D]`, `D>0`, и необязательный вес `[D]` при конечном `eps>0`. Для сопоставления с API вход и вес имеют общий плавающий тип и устройство. Предполагаем, что квадраты, сумма и результат помещаются в рабочий формат: сам по себе конечный FP64-вход этого не гарантирует. NumPy раскрывает редукцию, PyTorch добавляет автоматическое дифференцирование.

```python
import numpy as np
import torch
import torch.nn.functional as F
```

#### 1. Явная NumPy-реализация

```python
def numpy_rms_norm(x: np.ndarray, weight: np.ndarray | None = None, *, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 0 or x.shape[-1] == 0 or not np.isfinite(x).all() or not np.isfinite(eps) or eps <= 0:
        raise ValueError("x must be finite and eps positive")
    work = x.astype(np.float64, copy=False)
    y = work / np.sqrt(np.mean(work * work, axis=-1, keepdims=True) + eps)
    if weight is not None:
        w = np.asarray(weight, dtype=np.float64)
        if w.shape != (x.shape[-1],) or not np.isfinite(w).all():
            raise ValueError("weight must be finite with shape [D]")
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
    if not x.is_floating_point() or x.ndim == 0 or x.shape[-1] == 0:
        raise ValueError("x must be floating with a nonempty last axis")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    work = x.to(torch.float32) if x.dtype in (torch.float16, torch.bfloat16) else x
    rms_inv = torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + eps)
    y = work * rms_inv
    if weight is not None:
        if weight.shape != (x.shape[-1],):
            raise ValueError("weight must have shape [D]")
        y = y * weight.to(work.dtype)
    return y.to(x.dtype)
```

Явная PyTorch-реализация переводит FP16/BF16 в FP32 **до возведения в квадрат**. После умножения на вес она возвращает формат входа; FP64 сохраняется для проверки производных.

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
| Накопление | float64 в эталоне | явный путь: минимум FP32; API проверяется отдельно |
| Градиент | нет | проверяется `gradcheck` |

**Исполняемая проверка эквивалентности и граничных случаев**

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
for a, b in zip(torch.autograd.grad(explicit.square().sum(), (x, w)),
                torch.autograd.grad(api.square().sum(), (x, w))):
    torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-10)
for dtype in (torch.float16, torch.bfloat16):
    large = torch.full((2, 8), 1000.0, dtype=dtype)
    torch.testing.assert_close(torch_rms_norm_explicit(large), torch.ones_like(large))
print("B07 explicit NumPy / PyTorch / F.rms_norm: PASS")
```

Полный исполняемый файл: `m06_block_bridges.py`

Официальный API: [PyTorch `rms_norm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.rms_norm.html).

## 6. Где поставить норму

Обозначим преобразование подслоя через $F$, нормализатор через $N$, вход через $x$, выход через $x^+$. Названия Pre-LN и Post-LN описывают размещение: вместо LayerNorm в этих же схемах может стоять RMSNorm.

| Схема | Один шаг | Что получает нормированный масштаб |
|---|---|---|
| Post-LN | $x^+=N(x+F(x))$ | результат сложения |
| Pre-LN | $x^+=x+F(N(x))$ | вход подслоя |
| Peri-LN | $x^+=x+N_{out}(F(N_{in}(x)))$ | вход подслоя и его добавка |
| Схема OLMo2 | $x^+=x+N(F(x))$ | добавка подслоя |

![VIZ m6/02 — Положение нормы относительно подслоя и сложения задаёт разные вычислительные графы.](assets/modern-llms/ru/module-05/m6_02_norm_placements.svg)
*Положение нормы относительно подслоя и сложения задаёт разные вычислительные графы.*

В Post-LN нормализация лежит на сквозном пути. Для одного шага якобиан содержит произведение $J_N(I+J_F)$. В Pre-LN он имеет вид $I+J_FJ_N$: прямой единичный член сохранён. Это структурное различие, а качество обучения зависит также от инициализации, масштаба весов и оптимизатора.

В Peri-LN вторая норма управляет размером добавки перед сложением. Авторы исследовали такой подход на моделях до 3.2B параметров. В OLMo2 нет соответствующей нормы **на входе подслоя**; отдельная нормализация Q/K внутри внимания выполняет другую работу. [Peri-LN](https://arxiv.org/abs/2502.02732), [OLMo2](https://huggingface.co/docs/transformers/model_doc/olmo2).

### Опыт с общими входом и весами

Возьмём один вектор $h_0$ из стандартного нормального распределения ширины 256 и 36 независимых матриц $W_l$ с элементами из $\mathcal N(0,1/256)$. Для сравнения определим $U(h)=h/\max(\operatorname{RMS}(h),10^{-12})$ и запустим три цепочки на **одних и тех же** данных:

$$h^+=h+W_lh;\qquad h^+=h+W_lU(h);\qquad h^+=U(h+W_lh).$$

Это соответственно режимы без нормы, с нормой перед подслоем и после сложения. При seed 5 RMS на шаге 36 равен примерно **412 442**, **6.42** и **1**.

![VIZ m6/03 — масштаб состояния по глубине](assets/modern-llms/ru/module-05/m6_03_depth_variance.png)
*Ось RMS логарифмическая. Веса различаются между слоями, но повторно используются во всех трёх режимах.*

Единичный результат третьей цепочки задан операцией $U$; это не измерение качества. Опыт показывает динамику прямого прохода без обучения, нелинейностей и обучаемой $\gamma$. В реальной RMSNorm с $\epsilon>0$ даже до обучаемого масштаба RMS лишь близок к единице. Далее ноутбук сравнивает Pre-LN и Peri-LN уже на малом стеке с вниманием и FFN.

## 7. Нормализация Q/K: важна и формула, и ось

Логит внимания, то есть число перед softmax, равен $s_{ij}=q_i^\top k_j/\sqrt{d_h}$. Фиксированный делитель компенсирует рост дисперсии при увеличении ширины головы $d_h$, но не мешает обучаемым проекциям увеличивать длину запросов и ключей. Если оба вектора умножить на $c$, логит вырастет в $c^2$ раз.

Исходная **Query-Key Normalization (QKNorm)** использует единичные по L2 векторы:

$$\hat q=q/\lVert q\rVert_2,\quad \hat k=k/\lVert k\rVert_2,\quad s=\tau\hat q^\top\hat k.$$

Здесь $\tau$ — обучаемый масштаб вместо фиксированного $1/\sqrt{d_h}$. Нулевую длину на практике защищают стабилизатором. Авторы сообщили среднее улучшение 0.928 BLEU на пяти низкоресурсных парах перевода в своём протоколе. BLEU — метрика качества перевода, основанная на совпадении фрагментов с эталонными переводами; приведённый прирост относится именно к этой оценке. [QKNorm](https://arxiv.org/abs/2010.04245).

RMSNorm для Q/K — другая операция. Без $\epsilon$ и обучаемых весов она даёт $\sqrt{d_h}\hat q$. Поэтому после обычного деления на $\sqrt{d_h}$ логит равен $\sqrt{d_h}\cos\theta$, а не просто $\cos\theta$. Подмена L2-нормы RMS-нормой меняет масштаб softmax.

Важна и **ось редукции**. В проверенном коде Qwen3 нормируется каждая голова отдельно по $d_h$ признакам. OLMo2 нормирует весь вектор Q ширины $H_qd_h$ и весь K ширины $H_{kv}d_h$ ещё до разбиения на головы. Если увеличить только одну голову, при общей норме изменятся и остальные. Ноутбук показывает это на двух головах по два признака.

Ещё один механизм — **мягкое ограничение логитов (soft-capping)**: $s\mapsto c\tanh(s/c)$, где $c>0$ задаёт предел. Оно действует на готовые логиты, тогда как Q/K-норма — на векторы. Такие ограничения и методы управления оптимизатором следует проверять отдельно. Примеры реальных реализаций собраны в §16.

## 8. FFN: расширить признаки, преобразовать, вернуть ширину

Для каждой позиции обычный FFN без смещений вычисляет

$$\operatorname{FFN}(x)=W_2\phi(W_1x),\quad W_1\in\mathbb R^{d_{ff}\times d},\quad W_2\in\mathbb R^{d\times d_{ff}}.$$

Промежуточная ширина $d_{ff}$ обычно больше $d$; в исходном Transformer использовалась $4d$. Функция $\phi$ действует поэлементно. Без неё две матрицы можно было бы заменить одной, и дополнительная глубина не дала бы нелинейного преобразования.

Три распространённых варианта:

$$
\operatorname{ReLU}(z)=\max(0,z),\quad
\operatorname{GELU}(z)=z\Phi(z),\quad
\operatorname{SiLU}(z)=z\sigma(z).
$$

Здесь $\Phi$ — функция распределения стандартной нормальной величины, а $\sigma(z)=1/(1+e^{-z})$ — сигмоида. SiLU также называют Swish при коэффициенте 1. GELU и SiLU меняются плавно, но сами по себе не гарантируют ненулевого градиента во всех точках.

В **гейтованном FFN** два набора признаков вычисляются параллельно:

$$u=W_{up}x,\quad g=W_{gate}x,\quad y=W_{down}[u\odot\phi(g)].$$

Символ $\odot$ означает поэлементное произведение. Ветвь $g$ управляет вкладом ветви $u$ для данного входа. Однако при $\phi=\operatorname{SiLU}$ гейт не является вероятностью: его значения не ограничены интервалом $[0,1]$. В семейство входят ReGLU, GEGLU и SwiGLU; эксперимент Shazeer сравнивал их при согласованном бюджете. [GLU Variants](https://arxiv.org/abs/2002.05202).

Связь между двумя активациями можно проверить численно:

$$\operatorname{GELU}(z)\approx z\sigma(1.702z).$$

На сетке из 4001 точки в $[-6,6]$ максимальная абсолютная ошибка равна примерно **0.020335**. Это ошибка на заданной сетке; множитель 1.702 не превращает сигмоидную формулу в точную GELU.

![VIZ m6/04 — активации и ошибка приближения](assets/modern-llms/ru/module-05/m6_04_activations.png)
*Слева — функции, справа — ошибка приближения GELU на той же сетке, что используется в ноутбуке.*

## 9. SwiGLU: бюджет трёх матриц и вариант gpt-oss

У SwiGLU нелинейность гейта — SiLU:

$$\operatorname{FFN}(x)=W_{down}[\operatorname{SiLU}(W_{gate}x)\odot W_{up}x].$$

Матрицы gate и up имеют форму $d_{ff}\times d$, down — $d\times d_{ff}$. Без смещений это **$3dd_{ff}$ параметров**. У обычного FFN ширины $4d$ их $2d(4d)=8d^2$. Для одинакового матричного бюджета:

$$3dd_{ff}=8d^2\quad\Longrightarrow\quad d_{ff}=\frac83d.$$

Это точка равенства бюджетов, а не обязательная ширина. В следующем примере Llama использует $d_{ff}=3.5d$.

### Что вычисляет gpt-oss

В зафиксированном эталонном коде gpt-oss две ветви ограничены по-разному:

$$g_c=\min(g,7),\quad u_c=\operatorname{clip}(u,-7,7),$$
$$a(g,u)=g_c\sigma(1.702g_c)(u_c+1).$$

Это выход активации **до** down-проекции. При $g=u=z$:

| $z$ | $a(z,z)$ |
|---:|---:|
| −9 | 0.000012019 |
| −2 | 0.064341377 |
| 0 | 0 |
| 2 | 5.806975869 |
| 7 | 55.999625026 |
| 9 | 55.999625026 |

![VIZ m6/05 — Гейт проходит через SiLU, умножается на ветвь up, затем результат поступает в down-проекцию.](assets/modern-llms/ru/module-05/m6_05_swiglu_gate.svg)
*Гейт проходит через SiLU, умножается на ветвь up, затем результат поступает в down-проекцию.*

При $g>7$ изменение gate-входа уже не меняет результат: производная по нему равна нулю. В отрицательную сторону gate не обрезается, но подавляется сигмоидой. Ветвь up имеет свой двусторонний предел. На диагонали $z=7$ и $z=9$ обе используемые ветви совпадают, поэтому получается одно плато около 56.

![VIZ m6/06 — гейт и полная активация](assets/modern-llms/ru/module-05/m6_06_clamp_curve.png)
*Выход одного гейта около 7 и выход произведения около 56 — разные величины. Точки справа соответствуют таблице.*

Сдвиг $u_c+1$ оставляет множитель равным единице при $u_c=0$. Ограничения меняют обученный граф; их нельзя произвольно убрать при загрузке весов. Сокращение выбросов полезно для низкоточных вычислений, но код сам по себе не устанавливает единственную причину выбора этой формулы. Источник и размещение упакованных ветвей разобраны в §15.

**Второй проход: устойчивые реализации SwiGLU и проверка производных**

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
    if not np.isfinite(alpha) or not np.isfinite(shift_up):
        raise ValueError("alpha and shift_up must be finite")
    if any(v is not None and (not np.isfinite(v) or v <= 0) for v in (gate_limit, up_limit)):
        raise ValueError("limits must be finite and positive")
    g = gate.astype(np.float64, copy=False)
    u = up.astype(np.float64, copy=False)
    if gate_limit is not None:
        g = np.minimum(g, gate_limit)
    if up_limit is not None:
        u = np.clip(u, -up_limit, up_limit)
    sigmoid = np.exp(-np.logaddexp(0.0, -alpha * g))
    y = g * sigmoid * (u + shift_up)
    dtype = np.result_type(gate.dtype, up.dtype)
    return y.astype(dtype, copy=False) if np.issubdtype(dtype, np.floating) else y
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
    if gate.dtype != up.dtype or not gate.is_floating_point() or gate.device != up.device:
        raise ValueError("gate and up must share a floating dtype and device")
    if not np.isfinite(alpha) or not np.isfinite(shift_up):
        raise ValueError("alpha and shift_up must be finite")
    if any(v is not None and (not np.isfinite(v) or v <= 0) for v in (gate_limit, up_limit)):
        raise ValueError("limits must be finite and positive")
    dtype = torch.float32 if gate.dtype in (torch.float16, torch.bfloat16) else gate.dtype
    g, u = gate.to(dtype), up.to(dtype)
    if gate_limit is not None:
        g = g.clamp(max=gate_limit)
    if up_limit is not None:
        u = u.clamp(min=-up_limit, max=up_limit)
    return (g * torch.sigmoid(alpha * g) * (u + shift_up)).to(gate.dtype)
```

NumPy вычисляет сигмоиду через `logaddexp`, PyTorch — через `torch.sigmoid`. PyTorch сохраняет граф производных и повышает точность промежуточных вычислений FP16/BF16. В точках излома `clamp` обычная производная не определена; численная проверка проходит по обе стороны границ.

| Аспект | NumPy | PyTorch |
|---|---|---|
| Нелинейность | явная сигмоида | `torch.sigmoid` |
| Ограничение gate | `np.minimum` | `Tensor.clamp(max=...)` |
| Ограничение up | `np.clip` | `Tensor.clamp(min=..., max=...)` |
| Обычная SwiGLU | без limits, `shift_up=0` | тот же контракт |
| Градиенты | нет | autograd; точки clamp проверяются отдельно |

**Исполняемая проверка эквивалентности и граничных случаев**

```python
rng = np.random.default_rng(8)
g_np = rng.uniform(-4, 4, (3, 5)); u_np = rng.uniform(-4, 4, (3, 5))
kwargs = dict(alpha=1.702, gate_limit=7.0, up_limit=7.0, shift_up=1.0)
np_out = numpy_swiglu(g_np, u_np, **kwargs)
g = torch.tensor(g_np, dtype=torch.float64, requires_grad=True); u = torch.tensor(u_np, dtype=torch.float64, requires_grad=True)
t_out = torch_swiglu(g, u, **kwargs)
np.testing.assert_allclose(np_out, t_out.detach().numpy(), rtol=1e-10, atol=1e-10)
assert torch.autograd.gradcheck(lambda a, b: torch_swiglu(a, b, **kwargs), (g, u))
extreme = np.array([-1000., -9., 0., 9., 1000.])
with np.errstate(over='raise', invalid='raise'):
    stable = numpy_swiglu(extreme, np.ones_like(extreme), **kwargs)
np.testing.assert_allclose(stable, torch_swiglu(torch.tensor(extreme), torch.ones(5, dtype=torch.float64), **kwargs))
outside = torch.tensor([-9., 9.], dtype=torch.float64, requires_grad=True)
assert torch.autograd.gradcheck(lambda a: torch_swiglu(a, a, **kwargs), (outside,))
print("B08 forward + gradient: PASS")
```

Полный исполняемый файл: `m06_block_bridges.py`

## 10. Решённый пример: Llama-3.1-8B до последнего параметра

Считаем веса плотного декодера Llama-3.1-8B при явно заданной геометрии:

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

Здесь $V$ — число строк словаря, $L$ — число блоков, $H_q$ и $H_{kv}$ — количества голов запросов и K/V, $d_h$ — ширина головы. Предполагаем отсутствие смещений и отдельные входную таблицу эмбеддингов и выходную матрицу логитов. Базовые размеры опубликованы в [таблице 3 отчёта Llama 3](https://arxiv.org/html/2407.21783v3#S3.T3); для точного счёта используем $V=128256$, включая специальные токены, а не округлённые «128K».

У Q ширина выхода $H_qd_h=d$, у K/V — $H_{kv}d_h=d/4$. Выходная проекция O возвращает конкатенацию голов к $d$. Для произвольных размеров без смещений:

$$P_{attn}=2dH_qd_h+2dH_{kv}d_h.$$

Теперь можно подставить размеры.

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

![VIZ m6/07 — точный бюджет блока](assets/modern-llms/ru/module-05/m6_07_param_budget.svg)
*Разделены матричная часть $13d^2$ и полный блок $13d^2+2d$. В итоге получается точное число параметров Llama-3.1-8B.*

FFN занимает около 80.8% полного блока, внимание — около 19.2%, нормы — менее 0.004%. Последнее слагаемое $d$ в формуле модели — финальная RMSNorm. Это число обучаемых параметров; буферы RoPE, активации и состояние оптимизатора в него не входят. Ноутбук сверяет формулу с суммой размеров реально созданных тензоров малого блока, а валидатор — с полной геометрией на устройстве `meta`, без выделения памяти под 8B весов.

## 11. DyT и DyISRU: преобразования без общей статистики

Чтобы вычислить RMSNorm, нужно собрать средний квадрат по признакам — выполнить **редукцию**. Можно ли заменить её функцией, которая работает с каждой координатой независимо? Dynamic Tanh (DyT) предлагает

$$\operatorname{DyT}(x)_i=\gamma_i\tanh(\alpha x_i)+\beta_i.$$

Здесь $\alpha$ — обучаемый скаляр, $\gamma_i$ и $\beta_i$ — покоординатные параметры. Авторы показали сопоставимые или лучшие результаты в рассмотренных задачах зрения и языка. Это экспериментальное основание для изучения замены, а не равенство DyT и RMSNorm. [DyT](https://arxiv.org/abs/2503.10622).

Структурное различие видно на $(1,1)$ и $(1,10)$: RMSNorm изменит первую координату по-разному, поскольку изменился общий знаменатель. Любая фиксированная поэлементная функция даст на первой координате один и тот же ответ.

### Откуда берётся DyISRU

В RMSNorm без обучаемой шкалы и при $\epsilon=0$ выделим одну координату:

$$
\frac{x_i}{\sqrt{d^{-1}\sum_jx_j^2}}
=\sqrt d\,\frac{x_i}{\sqrt{x_i^2+\sum_{j\ne i}x_j^2}}.
$$

Если заменить сумму квадратов **остальных** координат положительным параметром $b$, связь между координатами исчезнет. Получится семейство DyISRU. Поглотив $\sqrt d$ в обучаемый масштаб, будем записывать его так:

$$\operatorname{DyISRU}(x)_i=\gamma_i\frac{x_i}{\sqrt{b+x_i^2}}+\beta_i,\qquad b>0.$$

При $\gamma_i=1$ наклон в нуле равен $1/\sqrt b$. Чтобы сравнить его с DyT при $\alpha=0.5$, зададим $b=4$: начальные наклоны совпадут. Это конкретная параметризация из [математической работы о DyISRU](https://arxiv.org/html/2503.21708v4), где также обсуждается положительная параметризация $b$ через softplus. Для неё обучают свободный скаляр $r$ и задают $b=\operatorname{softplus}(r)=\log(1+e^r)$. В точной арифметике результат положителен при любом конечном $r$. В ноутбуке используется устойчивая библиотечная функция с небольшой положительной добавкой для конечной точности. Точность вывода относится к операции после замены суммы параметром; обычная RMSNorm от этой замены меняется.

В ноутбуке сохранён отдельный опыт курса: для 200 000 чисел $x\sim\mathcal N(0,1)$ подбирается $\alpha$ по 171 точке сетки от 0.3 до 2.0 при seed 5, чтобы $\tanh(\alpha x)$ приблизила $x/\operatorname{RMS}(x_{sample})$. Получаются **$\alpha=1.43$ и MSE около 0.161704**. RMS здесь вычислен по всей синтетической выборке. Это скалярная аппроксимация с фиксированными $\gamma=1,\beta=0$, а не обучение нормализатора Transformer и не подбор параметра DyISRU.

![VIZ m6/08 — поэлементные функции и отдельный подбор DyT](assets/modern-llms/ru/module-05/m6_08_dyt_family.png)
*Слева DyT и DyISRU имеют одинаковый наклон в нуле. Справа показана ошибка самостоятельного подбора DyT по сетке.*

Исследовательские результаты зависят от оптимизатора и масштаба. В работе 2026 года на 1B параметров и 1000 шагах обнаружено отрицательное взаимодействие Dynamic Erf с Muon; контрольный DyT не показал того же штрафа. Это основание совместно проверять нормализатор и оптимизатор, но не общий вывод против DyT. [Протокол и результаты авторов](https://arxiv.org/abs/2604.01563). Оптимизаторы подробно рассматриваются в М9.

## 12. mHC: несколько остаточных потоков

До сих пор у токена был один вектор $h\in\mathbb R^d$. **Hyper-Connections** хранят несколько таких потоков и обучают их смешивание вокруг подслоя. Соберём $n$ потоков строками матрицы $X\in\mathbb R^{n\times d}$. Вход подслоя может быть их взвешенной суммой; его результат распределяется обратно по потокам.

Если матрица переноса $A\in\mathbb R^{n\times n}$ произвольна, уже преобразование $AX$ может сильно увеличить масштаб. В **Manifold-Constrained Hyper-Connections (mHC)** остаточное смешивание ограничивается дважды стохастической матрицей:

$$A_{ij}\ge0,\qquad \sum_jA_{ij}=1,\qquad \sum_iA_{ij}=1.$$

«Дважды» означает нормированные суммы и по строкам, и по столбцам. Каждая выходная координата $AX$ — выпуклая комбинация соответствующих координат входных потоков. Например,

$$
A=\begin{pmatrix}0.75&0.25\\0.25&0.75\end{pmatrix},\qquad
X=\begin{pmatrix}2\\6\end{pmatrix}\quad\Longrightarrow\quad AX=\begin{pmatrix}3\\5\end{pmatrix}.
$$

Смешивание удержало значения внутри исходного диапазона, но не сохранило каждый поток без изменений: дважды стохастическая матрица не обязана быть единичной.

Для получения такой матрицы используется **Sinkhorn–Knopp**: положительные элементы попеременно нормируются по строкам и столбцам. Конечное число итераций даёт приближение, поэтому нужно проверять обе невязки сумм. В исходной работе применялись 20 итераций. Наш небольшой опыт сравнивает 1, 20 и 100 итераций; он не воспроизводит всю модель. [mHC, §§4.2/5.3](https://arxiv.org/html/2512.24880v2).

Ограничение относится к остаточной матрице; коэффициенты чтения и записи ветви задаются отдельно. Кроме того, в реальном mHC коэффициенты зависят от $X$, а подслой добавляет своё преобразование. Поэтому свойство фиксированного $A$ не доказывает ограниченности полного якобиана сети. Множество допустимых матриц называют многогранником Биркгофа; «manifold» в названии работы относится к связям, а не к предположению о низкой размерности скрытых состояний.

DeepSeek-V4 использует mHC вместе с нормализацией внутри подслоёв: это сочетание разных механизмов. [Карточка авторов](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash). Развиваются и последующие варианты ограничений и реализации проекции; сравнивать их нужно по точности, затратам и протоколу обучения. Контекст обучения — в М9.

## 13. Код первого уровня: собрать две остаточные ветви

Здесь вход — матрица токенов `x[T, d]`. Матрицы gate/up имеют форму `[d, d_ff]`, down — `[d_ff, d]`. Они записаны в соглашении NumPy с умножением справа; это транспонированное хранение матриц из формул §9.

RMSNorm вычисляет статистику минимум в FP32, а сигмоида использует устойчивую форму. Две функции FFN различаются только активацией. В `PreNormStep` внимание передаётся явно и должно вернуть массив той же формы; полный GQA-блок находится в ноутбуке.

**Второй проход: полный минимальный листинг**

```python
import math
import numpy as np

def _floating_array(x: np.ndarray) -> np.ndarray:
    """Validate and return a floating NumPy array without changing its shape."""

    array = np.asarray(x)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("normalization expects a floating-point array")
    if array.ndim == 0 or array.shape[-1] == 0:
        raise ValueError("the normalized last axis must be non-empty")
    if not np.isfinite(array).all():
        raise ValueError("inputs must be finite")
    return array

def _statistics_dtype(dtype: np.dtype) -> np.dtype:
    """Use at least float32 for reductions while preserving float64 inputs."""

    return np.dtype(np.float32) if np.dtype(dtype).itemsize < 4 else np.dtype(dtype)

def rms_norm(
    x: np.ndarray,
    gamma: np.ndarray | float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """RMS-normalize `[..., d_model]` over the last axis and preserve dtype."""

    if eps <= 0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and positive")
    array = _floating_array(x)
    work_dtype = _statistics_dtype(array.dtype)
    work = array.astype(work_dtype, copy=False)
    mean_square = np.mean(work * work, axis=-1, keepdims=True)
    result = work / np.sqrt(mean_square + eps)
    if np.shape(gamma) not in ((), (array.shape[-1],)) or not np.isfinite(gamma).all():
        raise ValueError("gamma must be a finite scalar or [D] vector")
    result = result * np.asarray(gamma, dtype=work_dtype)
    return result.astype(array.dtype, copy=False)

def stable_sigmoid(x: np.ndarray | float) -> np.ndarray:
    """Compute sigmoid without `exp` overflow on large negative inputs."""

    z = np.asarray(x, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -z))

def silu(x: np.ndarray | float) -> np.ndarray:
    """Return SiLU/Swish: x * sigmoid(x)."""

    z = np.asarray(x, dtype=np.float64)
    return z * stable_sigmoid(z)

def gpt_oss_swiglu(
    gate: np.ndarray | float,
    up: np.ndarray | float,
    *,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> np.ndarray:
    """Apply the public gpt-oss gate clamp, up clamp, and `(up + 1)` shift."""

    if limit <= 0 or not math.isfinite(limit):
        raise ValueError("limit must be finite and positive")
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    g = np.minimum(np.asarray(gate, dtype=np.float64), limit)
    u = np.clip(np.asarray(up, dtype=np.float64), -limit, limit)
    return g * stable_sigmoid(alpha * g) * (u + 1.0)

def swiglu_mlp(x, w_gate, w_up, w_down):
    """Row-vector SwiGLU: two projections, product, then output projection."""
    return (silu(x @ w_gate) * (x @ w_up)) @ w_down

def swiglu_oss_mlp(x, w_gate, w_up, w_down, alpha=1.702, limit=7.0):
    """Apply the gpt-oss activation before the down projection."""
    return gpt_oss_swiglu(x @ w_gate, x @ w_up, alpha=alpha, limit=limit) @ w_down

class PreNormStep:
    """Inspect two sequential residual updates; attention accepts a mask."""
    def __init__(self, norm1, norm2, attention, ffn):
        self.norm1, self.norm2 = norm1, norm2
        self.attention, self.ffn = attention, ffn

    def __call__(self, x, mask):
        h = x + self.attention(self.norm1(x), mask)
        return h + self.ffn(self.norm2(h))
```

Проследите два инварианта: FFN получает состояние **после** добавки внимания, а down-проекция выполняется **после** произведения gate/up. Граничные входы и отключение ветвей проверяются отдельно. Нулевая добавка даёт тождественный шаг, но это проверка формулы, а не вывод о безвредности удаления обученного слоя.

## 15. Сопоставляем учебный блок с кодом gpt-oss

Для чтения реализации зафиксирована ревизия `7b583341fe16…` файла [gpt_oss/torch/model.py](https://github.com/openai/gpt-oss/blob/7b583341fe16729127f6d5b94a7b09ccae97e1a1/gpt_oss/torch/model.py). Ревизия и SHA-256 записаны в манифесте источников. Это эталонный PyTorch-путь; оптимизированные ядра проверяются отдельно.

Проследите три участка. **RMSNorm** переводит вход в FP32 до среднего квадрата, умножает на обучаемый вектор `scale` и возвращает исходный формат; значение $\epsilon$ по умолчанию равно $10^{-5}$. **SwiGLU** выделяет gate/up как чередующиеся координаты `x[..., ::2]` и `x[..., 1::2]`, затем применяет формулу §9. Разделить тензор на две сплошные половины здесь означало бы другое размещение весов. **Порядок блока** — Pre-LN перед вниманием и экспертным MLP.

Конфигурация gpt-oss-120b содержит, в частности:

```json
{"hidden_size": 2880, "intermediate_size": 2880,
 "num_experts": 128, "experts_per_token": 4, "swiglu_limit": 7.0}
```

В исходном `ModelConfig` нет поля `hidden_act`: функцию задаёт код. Другие библиотеки могут представлять ту же архитектуру своей схемой конфигурации.

![VIZ m6/09 — конфигурация и исполняемые операции](assets/modern-llms/ru/module-05/m6_09_config_block.svg)
*Размеры и число экспертов читаются из конфигурации; точная активация, порядок операций и упаковка ветвей — из кода.*

Валидатор сопоставляет учебные RMSNorm и SwiGLU с этими конкретными функциями на малых входах. Он не запускает всю модель. `intermediate_size=2880` относится к одному эксперту — отдельному FFN. Механизм выбора четырёх экспертов из 128 и его полный бюджет разберём в М7.

## 16. Какие детали искать в современных блоках

Срез на **10 октября 2026 года**. Таблица — справочник к уже разобранным механизмам. MoE означает набор экспертных FFN с выбором части из них для токена; его устройство — тема М7.

| Семейство | Нормализация и размещение | FFN | Обработка Q/K или логитов |
|---|---|---|---|
| Llama 3.1 | RMSNorm, Pre-LN | SwiGLU | отдельной Q/K-нормы в проверенном пути нет |
| Llama 4 | RMSNorm, Pre-LN | гейтованные MLP/MoE | L2 по голове на RoPE-слоях при включённом `use_qk_norm` |
| Gemma 2 | RMSNorm до и после подслоёв | гейтованный GELU-tanh MLP | мягкое ограничение логитов |
| Gemma 3 | RMSNorm до и после подслоёв | гейтованный GELU-tanh MLP | RMSNorm по голове для Q/K; soft-cap зависит от конфигурации |
| Gemma 4, текстовая часть | входные и выходные RMSNorm; дополнительные ветви зависят от размера модели | плотные и, в отдельных вариантах, экспертные ветви | RMSNorm Q/K; при повторном использовании K/V учитывается источник состояний |
| OLMo2 | RMSNorm выхода подслоя перед сложением | SiLU-гейтованный MLP | RMSNorm по всей проекции Q/K до разделения на головы |
| gpt-oss | RMSNorm, Pre-LN | ограниченный SwiGLU с $(u+1)$ | обучаемый дополнительный логит sink из М5 |
| DeepSeek-V4 | RMSNorm внутри подслоёв и mHC между потоками | эксперты с ограниченным SwiGLU | детали зависят от вида внимания |
| Qwen3 | RMSNorm, Pre-LN | SiLU-гейтованный MLP | RMSNorm по размерности каждой головы Q/K |

Основание — конкретные классы `DecoderLayer`, `Attention`, `RMSNorm` и `MLP` в [зафиксированном коде Transformers](https://github.com/huggingface/transformers/tree/536ecc007387a50e77603bb5d92100e9b07514cc/src/transformers/models) и эталонный код gpt-oss из §15. Файлы, контрольные суммы и дополнительные источники перечислены в манифесте.

При переносе весов проверяйте не только имя нормы, но и её ось, $\epsilon$, обучаемую шкалу, порядок приведения типов и место относительно сложения. Настройки отдельной версии модели не являются свойством всех поколений семейства. Для закрытых моделей отсутствующая архитектурная информация остаётся нераскрытой.

## 17. Текстовые и визуальные блоки

Vision Transformer (ViT) превращает участки изображения в последовательность векторов и использует знакомые нормализацию, внимание, FFN и остаточные связи. Подсчёт параметров переносится на него после подстановки размеров и правил смещений. Однако маска внимания в визуальном кодировщике обычно отличается от каузальной маски декодера.

Нормализация Q/K имеет примеры и в языке, и в зрении. QKNorm была опубликована в 2020 году для перевода, а ViT-22B позднее продемонстрировала её родственный механизм в крупной визуальной модели. [QKNorm](https://arxiv.org/abs/2010.04245), [ViT-22B](https://arxiv.org/abs/2302.05442).

В мультимодальной конфигурации отдельно читайте текстовую и визуальную части: ширина, активация, $\epsilon$ и размещение норм могут различаться. Это тот же навык восстановления вычислительного графа, который мы применили к gpt-oss.

## 20. Как прочитать незнакомый блок

![VIZ m6/10 — порядок разбора блока](assets/modern-llms/ru/module-05/m6_10_cheatsheet.svg)
*Сначала восстановите поток данных, затем нормализацию, FFN и точный бюджет.*

Начните с остаточного сложения: какие тензоры в него входят и совпадают ли их формы? Затем найдите каждую норму, её ось, параметры и точность. После этого проследите обе ветви гейта и выходную проекцию. Последний шаг — сумма размеров всех обучаемых тензоров.

Такой порядок разделяет четыре вопроса: **куда идёт информация, как меняется её масштаб, какое преобразование обучается и сколько оно стоит**. Для Llama-3.1-8B получается $13d^2$ матричных параметров и ещё $2d$ параметров норм на блок. Для экспертной модели этот же подсчёт нужно дополнить выбором FFN — к этому переходим в М7.

На справочном проходе добавьте два различия: DyT/DyISRU устраняют общую статистику признаков, а mHC меняет связи между несколькими остаточными потоками. Их следует оценивать на своём вычислительном и экспериментальном уровне.

## 21. первоисточники

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
- OpenAI. [gpt-oss reference implementation](https://github.com/openai/gpt-oss/blob/7b583341fe16729127f6d5b94a7b09ccae97e1a1/gpt_oss/torch/model.py)
- Документация моделей Hugging Face: [OLMo2](https://huggingface.co/docs/transformers/model_doc/olmo2), [Llama 4](https://huggingface.co/docs/transformers/model_doc/llama4), [Gemma 2](https://huggingface.co/docs/transformers/model_doc/gemma2), [Gemma 3](https://huggingface.co/docs/transformers/model_doc/gemma3), [Qwen3](https://huggingface.co/docs/transformers/model_doc/qwen3) и [DeepSeek-V4](https://huggingface.co/docs/transformers/model_doc/deepseek_v4)
- Официальный код Transformers для [OLMo2](https://github.com/huggingface/transformers/blob/536ecc007387a50e77603bb5d92100e9b07514cc/src/transformers/models/olmo2/modeling_olmo2.py), [Llama 4](https://github.com/huggingface/transformers/blob/536ecc007387a50e77603bb5d92100e9b07514cc/src/transformers/models/llama4/modeling_llama4.py), [Gemma 3](https://github.com/huggingface/transformers/blob/536ecc007387a50e77603bb5d92100e9b07514cc/src/transformers/models/gemma3/modeling_gemma3.py) и [Qwen3](https://github.com/huggingface/transformers/blob/536ecc007387a50e77603bb5d92100e9b07514cc/src/transformers/models/qwen3/modeling_qwen3.py)

**Дальше:** модуль 7 рассматривает смесь экспертов (Mixture of Experts). На примере 128 экспертов gpt-oss и выбора четырёх из них разберём общее число параметров и число параметров, участвующих в обработке одного токена.

---

*Источники проверены: 10 октября 2026 года. Численные примеры воспроизводятся локальным контрактом; эмпирические результаты сохраняют условия первоисточников; архитектурные свойства привязаны к официальным конфигурациям, коду или техническим отчётам.*
