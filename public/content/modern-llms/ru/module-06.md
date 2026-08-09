# Модуль 6. Mixture of Experts

*Курс «Modern LLMs» · лекция модуля 6 · редакция 2026.8*

> **О чём этот модуль.** В модуле 5 мы увидели, что большая часть параметров обычного блока Transformer находится в Feed-Forward Network (FFN). **Mixture of Experts (MoE)** использует это обстоятельство буквально: вместо одного большого FFN слой хранит банк экспертов, но для каждого токена запускает только небольшое подмножество. Так общая параметрическая ёмкость растёт быстрее, чем объём весовых матричных операций на токен. В этом модуле мы проследим один токен через весь контур MoE: роутер, top-k, взвешенную сумму, балансировку нагрузки, распределённую отправку и возврат результата. Затем разберём точный подсчёт gpt-oss, реконструкцию DeepSeek-V3, fine-grained и shared experts, LatentMoE и современные варианты адаптивной маршрутизации.
>
> **Что нужно знать заранее.** Полезны модули 1, 4 и 5: общий и активный размер модели, attention и бюджет FFN с SwiGLU. Все необходимые обозначения вводятся заново, поэтому модуль остаётся самодостаточным.

---

## 1. Мотивация: большая модель без полного вычисления на каждом токене

Пусть в обычном блоке Transformer после attention стоит один FFN с SwiGLU. Каждый токен проходит через все его матрицы. Если мы удваиваем ширину FFN, почти удваиваются и параметры, и соответствующая часть вычислений. Параметрическая ёмкость и цена токена растут вместе.

MoE разрывает эту связь не магией, а условным исполнением. Вместо одного FFN мы создаём $E$ экспертов и небольшой **роутер**. Для токена $x$ роутер выбирает $k\ll E$ экспертов, их выходы смешиваются, а остальные экспертные веса на этом шаге не участвуют в матричных умножениях.

Отсюда возникают две разные величины:

- **total parameters** — весь банк весов, который система должна хранить и обслуживать;
- **active parameters** — параметры, реально участвующие в выбранной конвенции вычислений одного токена.

Это не означает, что `total` равен «знаниям», а `active` — точной стоимости. Качество зависит от данных, обучения, маршрутизации и архитектуры; полная стоимость включает attention, общие слои, выходную проекцию, обмены и загрузку оборудования. Но разделение `total/active` даёт первый полезный паспорт разреженной модели.

Открытые примеры показывают масштаб идеи. gpt-oss-120b имеет 117B общих и 5.1B активных параметров, 128 экспертов и top-4. DeepSeek-V3 заявляет 671B/37B. Kimi K3 — 2.8T/104B при выборе 16 из 896 routed experts. Эти числа относятся к различным архитектурам и соглашениям подсчёта; задача модуля — научиться видеть, что именно стоит за каждым паспортом.

MoE экономит вычисления, но создаёт новый долг. Полный банк экспертов должен быть доступен системе, токены нужно доставлять на устройства с выбранными экспертами, нагрузку — удерживать в допустимых границах, а небольшие матричные умножения — собирать в эффективные групповые операции. Поэтому MoE является одновременно архитектурой модели и задачей распределённых систем.

## 2. Историческая линия: от локальных специалистов к триллионным моделям

![VIZ m6/01 — история Mixture of Experts](assets/modern-llms/ru/module-06/m6_01_moe_timeline.svg)

Идея обучаемой смеси локальных специалистов появилась задолго до Transformer. В работе Jacobs и Jordan 1991 года gating network определяла, какой локальный эксперт должен объяснять конкретный пример. Уже там присутствовали две центральные идеи: специализация и конкуренция за данные.

В 2017 году Shazeer и соавторы показали sparsely-gated MoE огромного размера. Слой содержал множество экспертов, но top-k gating активировал лишь несколько. Вместе с масштабом проявилась первая системная болезнь: если роутер слишком рано предпочитает малое число экспертов, остальные получают мало данных и перестают развиваться.

GShard и Switch Transformer перенесли эту схему в крупные Transformer. Появились **Expert Parallelism (EP)**, обмен all-to-all токенов, capacity factor и token dropping. Switch упростил маршрутизацию до top-1; ST-MoE добавил router z-loss для контроля логитов.

Mixtral 8×7B сделал открытый top-2 MoE практически доступным сообществу. DeepSeekMoE затем предложила две связанные идеи: дробить экспертную ёмкость на большее число узких блоков и отделять always-on shared experts от routed experts. DeepSeek-V3 объединила fine-grained experts с sigmoid scores и load balancing без отдельного auxiliary objective.

После этого развитие разветвилось. Одни системы увеличивают число экспертов и совершенствуют распределённое исполнение. Другие уменьшают routed payload — так устроена LatentMoE. Третьи исследуют differentiable или routing-free альтернативы фиксированному top-k. Поэтому современная картина не сводится к одному «победившему рецепту»: сохраняется общий принцип условных вычислений, а контракт роутера и системная реализация различаются.

## 3. Мост к классическим идеям: комитет, условные вычисления и диспетчер

### Смесь локальных моделей

Классическая смесь экспертов имеет вид

$$
y(x)=\sum_{i=1}^{E}g_i(x)f_i(x),
$$

где $f_i$ — эксперты, а $g_i$ — веса gating network. В современном sparse MoE большинство $g_i$ зануляется выбором top-k. Формула остаётся узнаваемой, но вычислительная единица меняется: эксперт — это крупный плотный FFN, удобный для GEMM, а не отдельный скаляр или нейрон.

### Условные вычисления

MoE — один из способов **conditional computation**: модель выбирает, какую часть параметров применить к данному входу. Выигрыш возникает потому, что внутри выбранного эксперта сохраняется плотная матричная арифметика. Это важное отличие от неструктурированной разреженности, где множество нулей не обязательно превращается в быстрый kernel.

### Диспетчер распределённой системы

Роутер напоминает диспетчер очереди. Он принимает решение, но эффективность определяется не только качеством решения: нужны capacity, размещение экспертов, сеть и порядок обработки. Если один эксперт получает в два раза больше токенов, чем остальные, все устройства могут ждать его завершения. Поэтому балансировка нагрузки является частью не только оптимизации модели, но и времени шага.

Эти три взгляда будут повторяться дальше. Статистическая формула объясняет mixture, conditional computation — разницу total/active, а диспетчер — почему низкое число FLOP ещё не гарантирует дешёвый инференс.

## 4. Формализм: routed experts, shared experts и явная конвенция подсчёта

Пусть скрытое состояние токена равно $x\in\mathbb{R}^h$. Роутер вычисляет scores для $E$ routed experts. В варианте с sigmoid:

$$
s_i=\sigma(w_i^\top x).
$$

Если используется selection bias $b_i$, множество выбранных экспертов определяется так:

$$
\mathcal T(x)=\operatorname{TopK}(s_i+b_i,k).
$$

После выбора gate weights могут быть нормированы из **чистых** scores:

$$
g_i=\rho\,\frac{s_i}{\sum_{j\in\mathcal T(x)}s_j},
\qquad i\in\mathcal T(x),
$$

где $\rho$ — дополнительный масштаб конкретной архитектуры. Тогда выход слоя:

$$
\operatorname{MoE}(x)
=
\sum_{i\in\mathcal T(x)}g_i\operatorname{FFN}_i(x)
+
\sum_{j=1}^{S}\operatorname{FFN}^{\text{shared}}_j(x).
$$

Shared experts выполняются для каждого токена. Routed experts проходят через top-k.

![VIZ m6/02 — анатомия слоя MoE](assets/modern-llms/ru/module-06/m6_02_moe_block.svg)

### Почему одной универсальной формулы недостаточно

Количество параметров зависит от архитектурного контракта:

- сколько слоёв dense, а сколько MoE;
- связаны ли input embeddings и output head;
- есть ли bias в attention, роутере и экспертах;
- сколько shared experts;
- одинаковы ли размеры всех экспертов;
- присутствуют ли модули MTP или дополнительные нормы;
- какой тип attention используется.

Поэтому в численном коде модуль использует типизированную спецификацию `MoEModelSpec`. Она требует явно объявить каждую из этих частей.

Для одного эксперта SwiGLU без bias:

$$
P_{\text{expert}}=3h\,i_{\text{exp}}.
$$

Если есть bias у двух входных проекций и выходной проекции:

$$
P_{\text{expert}}
=3h\,i_{\text{exp}}+2i_{\text{exp}}+h.
$$

В active count routed bank входит с множителем $k$, shared bank — целиком. Attention, нормы и роутер обычно являются общими параметрами. Но даже здесь остаётся выбор: считать ли embedding lookup как полную активную матрицу и включать ли output projection. В этом модуле для gpt-oss используется вычислительная конвенция: output head считается полной активной матрицей, а индексный lookup input embedding — нет. Конвенция записывается рядом с числом, а не скрывается внутри формулы.

## 5. Решённый пример A: gpt-oss с параметрами смещения

Открытая реализация позволяет посчитать больше, чем округлённые 117B/5.1B. Для обеих моделей:

- $V=201\,088$;
- $h=i_{\text{exp}}=2880$;
- 64 query-heads, 8 KV-heads, head dimension 64;
- untied input embedding и output head;
- bias в QKV, output attention, router и experts;
- learned attention sink на каждую query-head.

### gpt-oss-120b

Параметры архитектуры:

$$
L=36,
\qquad E=128,
\qquad k=4.
$$

Покомпонентный подсчёт:

| Компонент | Параметры |
|---|---:|
| две словарные матрицы | 1 158 266 880 |
| attention во всех слоях, включая bias | 955 802 880 |
| полный банк experts, включая bias | 114 701 598 720 |
| роутеры | 13 275 648 |
| нормы | 210 240 |
| learned sinks | 2 304 |
| **Итого** | **116 829 156 672** |

Под нашей активной конвенцией один токен использует:

- одну полную словарную проекцию;
- все attention/router/norm/параметры learned sink;
- четыре routed experts в каждом слое.

Итог:

$$
P_{\text{active}}=5\,132\,849\,472.
$$

Отношение:

$$
\frac{P_{\text{total}}}{P_{\text{active}}}
=22.761.
$$

Официальные 117B/5.1B являются корректными округлёнными заявленными округлёнными значениями; наш расчёт — реконструкция под явно объявленной конвенцией.

### gpt-oss-20b

Для $L=24$, $E=32$, $k=4$ получаем:

$$
P_{\text{total}}=20\,914\,757\,184,
$$

$$
P_{\text{active}}=3\,608\,307\,264,
$$

$$
P_{\text{total}}/P_{\text{active}}=5.796.
$$

Меньшая модель сохраняет top-4, но её банк содержит всего 32 эксперта. Поэтому доля активного expert bank значительно выше.

![VIZ m6/03 — total и active parameters](assets/modern-llms/ru/module-06/m6_03_total_active.png)

Приближение $2P_{\text{active}}$ можно использовать для весовой части матричных умножений плотного типа, но отношение 22.76× не является гарантированным ускорением: полный банк весов остаётся в системе, а маршрутизация и all-to-all добавляют собственную цену.

## 6. Роутер крупным планом: числовая трассировка

Пусть для одного токена роутер выдал logits:

```text
[1.2, -0.3, 2.1, 0.4, 1.7, -1.0, 0.9, 2.5]
```

После sigmoid:

```text
[0.7685, 0.4256, 0.8909, 0.5987,
 0.8455, 0.2689, 0.7109, 0.9241]
```

При top-4 выбираются эксперты:

```text
[7, 2, 4, 0]
```

Нормированные gate weights:

```text
[0.2695, 0.2598, 0.2466, 0.2241]
```

Теперь добавим эксперту 3 selection bias $+0.35$. Приоритеты для выбора меняются, и top-4 становится:

```text
[3, 7, 2, 4]
```

Но после выбора веса вычисляются из чистых sigmoid scores. Эксперт 3 получает вес не из $0.5987+0.35$, а из $0.5987$:

```text
[0.1837, 0.2835, 0.2733, 0.2594]
```

![VIZ m6/04 — трассировка роутера](assets/modern-llms/ru/module-06/m6_04_router_trace.svg)

### Групповая предварительная селекция

В DeepSeek-V3 256 routed experts разбиты на восемь групп. Для токена сначала выбираются четыре группы, затем top-8 experts внутри разрешённого множества. Такая двухступенчатая схема ограничивает коммуникационное рассеяние.

При балансировочном bias group score в опубликованной реализации строится из двух лучших скорректированных scores группы. После выбора gate weights снова используют чистые scores и умножаются на routed scale 2.5.

Это важный пример общего принципа: **score для выбора** и **вес вклада** могут быть разными объектами.

### NumPy → PyTorch · B09 — Top-k MoE: выбор экспертов и веса смеси

Смещение может менять выбор экспертов, а веса смеси по-прежнему вычисляются из чистых оценок. NumPy делает эту границу видимой; PyTorch использует `topk` и `gather`.

```python
import numpy as np
import torch
```

#### 1. Явная NumPy-реализация

```python
def numpy_topk_router_explicit(
    logits: np.ndarray,
    top_k: int,
    *,
    selection_bias: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    clean = 1.0 / (1.0 + np.exp(-logits))
    selection = clean if selection_bias is None else clean + selection_bias
    indices = np.argsort(-selection, axis=-1, kind="stable")[:, :top_k]
    weights = np.take_along_axis(clean, indices, axis=-1)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return indices, weights
```

В минимальном NumPy-коде разделены три объекта: чистые scores, scores для селекции после bias и веса смеси, собранные из чистых scores выбранных экспертов.

#### 2. Та же математика явными операциями PyTorch

```python
def torch_topk_router_explicit(
    logits: torch.Tensor,
    top_k: int,
    *,
    selection_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    clean = logits.sigmoid()
    selection = clean if selection_bias is None else clean + selection_bias
    _, indices = torch.topk(selection, top_k, dim=-1)
    weights = clean.gather(-1, indices)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return indices, weights
```

PyTorch выражает тот же маршрут через `sigmoid`, `topk` и `gather`. При равных значениях порядок индексов `topk` не является переносимой гарантией.

| Аспект | NumPy | PyTorch |
|---|---|---|
| Чистые scores | sigmoid NumPy | `Tensor.sigmoid` |
| Селекция | stable `argsort` | `torch.topk` |
| Вес выбранного эксперта | `take_along_axis` | `gather` |
| Bias | влияет только на выбор | тот же контракт |
| Ties | stable sort в эталоне | порядок не гарантирован |

<details>
<summary><strong>Исполняемая проверка эквивалентности и граничных случаев</strong></summary>

```python
logits_np = np.array([[2.1, -0.4, 0.8, 1.3], [0.2, 1.9, -1.1, 0.7]])
bias_np = np.array([0.0, 0.08, -0.03, 0.02])
i_np, w_np = numpy_topk_router_explicit(logits_np, 2, selection_bias=bias_np)
i_t, w_t = torch_topk_router_explicit(torch.tensor(logits_np), 2, selection_bias=torch.tensor(bias_np))
np.testing.assert_array_equal(i_np, i_t.numpy())
np.testing.assert_allclose(w_np, w_t.numpy())
print("B09 explicit top-k parity away from ties: PASS")
```

</details>

Полный исполняемый файл: [`m06_moe_bridges.py`](../assets/m06_moe_bridges.py)

Контракт tie-breaking: [PyTorch `topk`](https://docs.pytorch.org/docs/stable/generated/torch.topk.html).

## 7. Балансировка: от auxiliary objective к управлению маршрутом

Если небольшая случайная флуктуация направляет больше токенов к одному эксперту, он получает больше обучающих примеров и может становиться ещё привлекательнее. Возникает положительная обратная связь — routing collapse.

### Auxiliary load-balancing loss

Цель типа Switch цель сопоставляет долю назначений $f_i$ и среднюю вероятность $P_i$:

$$
L_{\text{aux}}=E\sum_i f_iP_i.
$$

Она создаёт прямой градиент, побуждающий роутер распределять нагрузку равномернее. Если коэффициент слишком велик, балансировочная цель начинает конкурировать с языковой.

### Router z-loss

ST-MoE добавляет штраф

$$
L_z=\mathbb{E}\!\left[\log\sum_i e^{z_i}\right]^2,
$$

который сдерживает рост логитов роутера и помогает численной устойчивости. Это не замена балансировке: z-loss контролирует масштаб logits, а auxiliary loss — распределение назначений.

### Loss-Free Balancing

Вместо отдельной differentiable loss к selection score добавляется bias $b_i$, обновляемый по наблюдаемой нагрузке вне backpropagation. Основное правило статьи использует фиксированный шаг:

$$
b_i\leftarrow b_i+u\,\operatorname{sign}(\bar c-c_i).
$$

Перегруженный эксперт получает bias вниз, недогруженный — вверх. Метод не создаёт **прямого interference gradient** от auxiliary objective. Но утверждать, что обучение вообще не меняется, нельзя: новая маршрутизация определяет, какие эксперты получают градиент языковой модели.

![VIZ m6/05 — балансировка нагрузки](assets/modern-llms/ru/module-06/m6_05_balance.png)

В численном сценарии коэффициент вариации нагрузки начинается с 0.6506. После 400 шагов:

- sign-update при $u=0.001$ даёт 0.0393;
- учебный proportional controller при $u=0.15$ даёт 0.0193.

Вторая кривая не является репликой основного алгоритма DeepSeek. Она показывает работу более гладкого регулятора; пропорциональный вариант также исследовался авторами в отдельных постановках.

### Capacity — ещё одна линия защиты

Если expert может принять не больше

$$
C_e=\left\lceil c\frac{Tk}{E}\right\rceil
$$

назначений, overflow нужно обработать явно: drop, reroute, увеличение буфера или dropless execution. В учебном коде используется отбрасывание по FIFO и перенормировка оставшихся gate weights. Это **политика курса**, а не универсальное правило MoE.

Полностью потерявший routed assignments токен не исчезает из модели: его экспертная ветвь в данном слое равна нулю, но residual path и другие подслои продолжают работу.

## 8. Fine-grained и shared experts: больше комбинационной гибкости

В грубом MoE можно хранить восемь широких experts и выбирать два. DeepSeekMoE предлагает разделить каждый широкий блок на несколько более узких. Если ширина каждого уменьшается пропорционально, можно выбирать больше блоков при сопоставимом expert-compute.

Число возможных неупорядоченных подмножеств иллюстрирует рост гибкости:

$$
\binom{8}{2}=28,
$$

$$
\binom{128}{4}=10\,668\,000.
$$

Но комбинацию нельзя называть отдельным «композитным специалистом». Реальное поведение зависит от непрерывных gate weights, содержимого experts, корреляций и ограничений роутера.

Для DeepSeek-V3 формальное

$$
\binom{256}{8}
$$

является лишь unconstrained upper bound: групповая предварительная селекция запрещает часть подмножеств.

**Shared experts** решают другую проблему. Они выполняются всегда и должны поглощать общую для многих токенов работу, уменьшая необходимость дублировать её в routed bank. Исходная мотивация статьи — common knowledge и снижение redundancy. Это не доказывает, что конкретный shared expert «хранит грамматику» или «арифметику»: такие семантические утверждения требуют интерпретационного эксперимента.

![VIZ m6/06 — fine-grained и shared experts](assets/modern-llms/ru/module-06/m6_06_fine_grained.svg)

### Решённый пример B: реконструкция DeepSeek-V3

С использованием опубликованной конфигурации и явно заданной конвенции модуль получает:

$$
P_{\text{total}}=671\,026\,419\,200,
$$

$$
P_{\text{active}}=36\,625\,618\,432.
$$

Официальный паспорт округляет эти величины до 671B/37B. Отдельно опубликованный Hugging Face package содержит около 685B параметров вместе с 14B модулем MTP. Поэтому в таблицах нужно различать:

- main model;
- full package с MTP;
- active count под выбранной конвенцией.

![VIZ m6/07 — бюджет DeepSeek-V3](assets/modern-llms/ru/module-06/m6_07_v3_budget.svg)

## 9. Специализация: что можно и чего нельзя заключать из маршрутов

MoE создаёт условия для специализации: разные experts видят разные подраспределения токенов. Но путь от маршрутов к понятному ярлыку «эксперт по физике» слишком длинен.

Надёжно измеряются:

- частота выбора по слоям и доменам;
- co-activation разных experts;
- изменение output при ablation или замене эксперта;
- чувствительность роутера к синтаксическим и семантическим признакам;
- устойчивость паттерна на разных корпусах.

Гораздо осторожнее нужно обращаться с интерпретацией. Один эксперт может предпочитать определённую морфологическую конструкцию, тип границы токена или операцию преобразования представления, а не широкую тему. Несколько экспертов могут быть взаимозаменяемы, а shared branch — выполнять разные функции на разных слоях.

Поэтому хороший анализ имеет три ступени:

1. **routing preference** — какие токены чаще выбирают expert;
2. **functional effect** — что меняется при вмешательстве;
3. **semantic label** — человеческое резюме, которое может быть неполным.

Из отсутствия очевидного «эксперта языка X» также не следует, что SFT обязательно повредит старым языкам. Это эмпирический вопрос, зависящий от базы, данных и того, какие параметры обновляются.

Для сервинга важна другая специализация — **hot experts**. Реальный трафик неравномерен, и самые популярные experts могут создавать коммуникационный узкий участок даже у хорошо сбалансированной обучающей смеси. Именно поэтому профиль обслуживания нужно измерять на целевой нагрузке.

## 10. Инференс MoE: память, all-to-all и объединение экспертов по батчу

MoE уменьшает expert-compute токена, но полный банк весов должен быть доступен системе. Это не означает, что все experts обязаны лежать на одной GPU. Возможны:

- Expert Parallelism;
- хранение части весов в CPU DRAM или NVMe;
- репликация горячих experts;
- weight streaming;
- многоуровневый кэш;
- шардирование между узлами.

### Expert Parallelism и all-to-all

При EP experts распределяются по устройствам. После роутинга токены отправляются владельцам выбранных experts, там выполняются grouped GEMM, затем результаты возвращаются. Идеализированный объём активаций в обе стороны:

$$
M_{\text{comm}}=2T h k b.
$$

Для $T=2048$, $h=7168$, $k=8$, bf16:

$$
M_{\text{comm}}=448\ \text{MiB}.
$$

Это только payload: метаданные, padding, topology и protocol overhead не включены.

### Сколько experts затронет батч

При независимом равномерном выборе вероятность, что данный expert не будет выбран ни одним из $N$ токенов:

$$
\left(1-\frac{k}{E}\right)^N.
$$

Ожидаемая доля затронутых experts:

$$
f_{\text{touched}}
=1-\left(1-\frac{k}{E}\right)^N.
$$

Для gpt-oss 128/top-4 и DeepSeek 256/top-8 отношение $k/E$ одинаково, поэтому теоретическая кривая совпадает: при $N=64$ затрагивается около 86.9% experts.

Для Kimi K3 896/top-16:

- $N=16$: около 25.0%;
- $N=64$: около 68.4%.

Это не предсказание реального сервера. Маршруты коррелированы и неравномерны, shared experts всегда активны, а group routing изменяет допустимое множество. Формула — baseline для проверки фактической телеметрии.

![VIZ m6/08 — системная сторона MoE](assets/modern-llms/ru/module-06/m6_08_expert_parallel.svg)

Выгрузка experts не «умирает» автоматически при большом батче. Она становится сложнее, но может использовать перекошенную популярность, временную локальность и предварительную загрузку. Решение определяется задержкой транспорта, частотой reuse и распределением маршрутов.

## 11. LatentMoE и адаптивная маршрутизация

### LatentMoE

Обычный expert parallelism пересылает скрытый вектор ширины $d$. **LatentMoE** сохраняет дискретную маршрутизацию, но перед dispatch проецирует routed payload в более узкую размерность $\ell$:

$$
z=P_{\text{down}}x,
\qquad z\in\mathbb{R}^{\ell},
\quad \ell<d.
$$

Router по-прежнему может читать исходное скрытое состояние и выбирать top-k experts. Сами routed FFN, dispatch и combine выполняются в latent width; затем результат проецируется обратно:

$$
y=P_{\text{up}}\operatorname{MoE}_{\ell}(z).
$$

Если $d=7168$, $\ell=3584$, routed activation payload уменьшается в идеальной оценке в два раза. Сходным образом уменьшается размер expert matrices при соответствующей архитектуре.

Важно: LatentMoE не создаёт непрерывный «виртуальный эксперт» из базисных матриц. Top-k и дискретный expert bank сохраняются. NVIDIA использует этот подход в Nemotron 3 Super/Ultra; Kimi K3 описывает Stable LatentMoE с 896 routed experts, top-16 и двумя shared experts.

### Другие варианты маршрутизации

Несколько работ меняют сам контракт top-k:

| Метод | Как маршрутизирует | Адаптивное число experts | Исходная область |
|---|---|---:|---|
| Soft MoE (2023) | мягкие token-to-slot и slot-to-token смеси | фиксированная слотная схема | vision Transformers |
| ReMoE | differentiable ReLU routing | эффективная динамическая разреженность | language models |
| SoftMoE (2026) | differentiable truncated soft top-k | да, с глобальным бюджетом | LLM |
| Routing-Free MoE | без отдельного центрального router | зависит от метода | исследовательские модели |

Эти направления нельзя объединять в одну линейную «смену стандарта». Они оптимизируют разные свойства: differentiability, бюджет на слой, баланс или отсутствие отдельного роутера. Их эксплуатационная стоимость определяется не только качеством, но и возможностью построить статически эффективные kernels.

## 12. Код первого уровня: прозрачный MoE с явными политиками

Учебный код модуля не пытается имитировать распределённый runtime. Он делает видимыми четыре контракта:

1. clean score и selection score — разные величины;
2. tie-break детерминирован;
3. capacity overflow имеет объявленную политику;
4. active/total count имеет явную архитектурную спецификацию.

```python
import numpy as np

def route_logits(logits, top_k, selection_bias=None):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 1:
        logits = logits[None, :]
    if logits.ndim != 2 or not np.isfinite(logits).all():
        raise ValueError("logits must be finite [T,E]")
    T, E = logits.shape
    if not 1 <= top_k <= E:
        raise ValueError("top_k must satisfy 1 <= k <= E")

    scores = 1.0 / (1.0 + np.exp(-logits))
    bias = np.zeros(E) if selection_bias is None else np.asarray(selection_bias)
    selection = scores + bias[None, :]

    ids = np.arange(E)
    chosen = np.empty((T, top_k), dtype=np.int64)
    for t in range(T):
        chosen[t] = np.lexsort((ids, -selection[t]))[:top_k]

    selected_scores = np.take_along_axis(scores, chosen, axis=1)
    gates = selected_scores / selected_scores.sum(axis=1, keepdims=True)
    return chosen, gates

def loss_free_bias_step(bias, counts, rate):
    """Sign update from the main Loss-Free Balancing rule."""
    counts = np.asarray(counts, dtype=np.float64)
    fractions = counts / counts.sum()
    return np.asarray(bias) + rate * np.sign(fractions.mean() - fractions)
```

Полная версия добавляет stable sigmoid, log-space gate normalization, capacity, shared experts, auxiliary losses и типизированный калькулятор параметров. Встроенные negative tests запрещают молчаливые ошибки форм и единиц.

### NumPy → PyTorch · B10 — MoE dispatch и combine

NumPy выполняет прозрачный цикл по назначениям. PyTorch группирует токены по эксперту и суммирует взвешенные результаты через `index_add_`.

```python
import numpy as np
import torch
```

#### 1. Явная NumPy-реализация

```python
def numpy_expert_dispatch_explicit(
    hidden: np.ndarray,
    expert_indices: np.ndarray,
    gates: np.ndarray,
    w_in: np.ndarray,
    w_out: np.ndarray,
) -> np.ndarray:
    output = np.zeros_like(hidden)
    for token in range(hidden.shape[0]):
        for slot in range(expert_indices.shape[1]):
            expert = int(expert_indices[token, slot])
            mid = np.tanh(hidden[token] @ w_in[expert])
            output[token] += gates[token, slot] * (mid @ w_out[expert])
    return output
```

NumPy проходит по токенам и слотам роутера: выбирает expert weights, вычисляет expert FFN и добавляет результат с gate-весом в выход токена.

#### 2. Та же математика явными операциями PyTorch

```python
def torch_expert_dispatch_explicit(
    hidden: torch.Tensor,
    expert_indices: torch.Tensor,
    gates: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
) -> torch.Tensor:
    output = torch.zeros_like(hidden)
    for expert in range(w_in.shape[0]):
        token_ids, slots = torch.where(expert_indices == expert)
        if token_ids.numel() == 0:
            continue
        mid = torch.tanh(hidden[token_ids] @ w_in[expert])
        expert_out = (mid @ w_out[expert]) * gates[token_ids, slots, None]
        output.index_add_(0, token_ids, expert_out)
    return output
```

PyTorch меняет порядок работы: сначала собирает все назначения одного эксперта в batch, затем возвращает результаты токенам через `index_add_`.

| Аспект | NumPy | PyTorch |
|---|---|---|
| Порядок | token → slot | expert → assigned tokens |
| Экспертный batch | нет | индексация `hidden[token_ids]` |
| Combine | `output[token] += ...` | `output.index_add_` |
| Autograd | нет | через expert matmul и combine |
| CUDA-детерминизм | CPU reference | `index_add_` может зависеть от порядка атомарных сложений |

<details>
<summary><strong>Исполняемая проверка эквивалентности и граничных случаев</strong></summary>

```python
rng = np.random.default_rng(10)
x_np = rng.standard_normal((6, 4))
idx_np = np.array([[0,1],[1,2],[2,0],[0,2],[1,0],[2,1]])
gates_np = np.full((6,2), 0.5)
w1_np = rng.standard_normal((3,4,5)); w2_np = rng.standard_normal((3,5,4))
np_out = numpy_expert_dispatch_explicit(x_np, idx_np, gates_np, w1_np, w2_np)
t_out = torch_expert_dispatch_explicit(torch.tensor(x_np), torch.tensor(idx_np), torch.tensor(gates_np), torch.tensor(w1_np), torch.tensor(w2_np))
np.testing.assert_allclose(np_out, t_out.numpy(), rtol=1e-10, atol=1e-10)
print("B10 token/slot loop vs expert batches: PASS")
```

</details>

Полный исполняемый файл: [`m06_moe_bridges.py`](../assets/m06_moe_bridges.py)

Контракт накопления: [PyTorch `index_add_`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.index_add_.html).

## 14. Чтение открытого кода gpt-oss

Публичная реализация полезна тем, что позволяет увидеть то, что исчезает в кратком `config.json`.

### Роутер

`GptOssTopKRouter` содержит:

- матрицу $W_r\in\mathbb{R}^{E\times h}$;
- bias длины $E$;
- top-k по router logits;
- softmax только по выбранным logits.

Это отличается от sigmoid + selection bias DeepSeek. Обе архитектуры используют top-k, но score и balance contract различаются.

### Expert tensors

Экспертный блок хранит:

- packed gate/up weights;
- gate/up bias;
- down weights;
- down bias.

Именно поэтому matrix-only подсчёт 116.789B был неполным. Параметры смещения добавляют около 40.1M к 120b-модели.

### Expert bank и distributed execution

Эталонный Python-код проходит только по experts, затронутым текущим батчем, и собирает результат через `index_add_`. Это понятная reference implementation. Производственный runtime заменяет цикл grouped GEMM, expert parallelism и fused dispatch.

![VIZ m6/09 — gpt-oss из исходного кода](assets/modern-llms/ru/module-06/m6_09_config_moe.svg)

Главный урок чтения source code: имя `MoE` недостаточно. Нужно проверить score function, bias, нормировку gates, форму expert weights, наличие shared experts, capacity policy и распределённый путь исполнения.

## 15. Современный ландшафт как паспорт свидетельств

Таблица фиксирует срез на 6 августа 2026 года. Она не ранжирует модели; её цель — показать разные MoE-контракты и источник каждого поля.

| Модель | Total / active | Expert contract | Routing / особенности | Основание |
|---|---:|---|---|---|
| gpt-oss-120b | 117B / 5.1B | 128 routed, top-4, без shared | softmax по выбранным logits | официальный анонс, config и код |
| DeepSeek-V3 | 671B / 37B | 256 routed + 1 shared, top-8 | sigmoid, group preselection, loss-free bias | technical report и открытый код инференса |
| Qwen3.5-35B-A3B | 35B / 3B | 256 routed + 1 shared, top-8 | hybrid DeltaNet/attention, MTP | официальный model card и config |
| Kimi K3 | 2.8T / 104B | 896 routed + 2 shared, top-16 | Stable LatentMoE, latent width 3584 | официальный tech report и weights |
| Nemotron 3 Ultra | 550B / 55B | LatentMoE | hybrid Mamba–Attention, MTP, NVFP4 | официальный technical report |
| GLM-5 | 744B / 40B | 256 routed + 1 shared, top-8 | sigmoid/noaux, DSA в attention | официальный model card и config |
| DeepSeek-V4 Pro / Flash | 1.6T / 49B; 284B / 13B | раскрывается в report | MoE + гибридный attention | technical report |

Эти строки нельзя механически сравнивать по total/active ratio. Различаются attention, число плотных слоёв, MTP, модальности, counting conventions и hardware path.

### Qwen3.5 как упражнение по чтению config

Реальный паспорт 35B-A3B:

- hidden size 2048;
- 40 слоёв;
- 256 routed experts;
- top-8 + один shared expert;
- routed и shared intermediate width 512;
- один слой MTP;
- гибрид Gated DeltaNet и full attention.

Это лучше учебного «угадывания» фиктивных 48 слоёв и ширины 768. Обратную задачу имеет смысл строить только после того, как известен фактический контракт.

## 16. Мультимодальность: общий роутер — один из вариантов, а не правило

MoE естественно сочетается с несколькими модальностями, но архитектурный выбор может быть разным.

### Общий routed bank

Текстовые и визуальные токены поступают в один residual stream и используют общий роутер. Тогда эксперты могут разделяться по функциям, модальностям или смешанным признакам. Kimi K3 является примером нативной мультимодальной системы MoE.

### Модально-специфичные experts

Часть experts может быть доступна только vision или audio tokens. Это уменьшает конкуренцию, но усложняет размещение и баланс.

### Отдельный vision encoder и языковой MoE

Изображение сначала кодируется отдельной сетью, а полученные visual tokens проходят через языковой MoE. Тогда router видит уже спроецированное представление, а не исходные patches.

Если каждый токен по-прежнему активирует $k$ routed experts, его expert-compute не растёт только из-за модальности. Но смешанный батч может затронуть более широкий union experts и изменить коммуникационный профиль. Поэтому мультимодальный MoE нужно оценивать по маршрутам, all-to-all и загрузке experts, а не только по total/active паспорту.

## 19. Ключевые выводы модуля

![VIZ m6/10 — шпаргалка Mixture of Experts](assets/modern-llms/ru/module-06/m6_10_cheatsheet.svg)

- MoE увеличивает полный банк параметров быстрее, чем expert-compute одного токена, но не отменяет память и коммуникации.
- `total` и `active` имеют смысл только вместе с counting convention.
- Selection bias может менять top-k, не входя в gate weights.
- Loss-Free Balancing устраняет прямой auxiliary gradient, но всё равно меняет распределение обучающих маршрутов.
- Fine-grained experts увеличивают комбинационную гибкость; $\binom{E}{k}$ не равно числу независимых навыков.
- Shared experts уменьшают стимул дублировать общую работу, но их семантику нужно измерять.
- Большой батч может затронуть значительную часть expert bank; равномерная формула — baseline, а не закон реального роутера.
- LatentMoE сохраняет дискретный top-k и сжимает routed representation.
- Современный ландшафт нужно читать по config/report/code, не перенося свойства между поколениями моделей.

## 20. и источники

**Основные источники:**

- Jacobs et al., *Adaptive Mixtures of Local Experts* — [Neural Computation](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)
- Shazeer et al., *Outrageously Large Neural Networks* — [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)
- GShard — [arXiv:2006.16668](https://arxiv.org/abs/2006.16668)
- Switch Transformer — [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
- ST-MoE — [arXiv:2202.08906](https://arxiv.org/abs/2202.08906)
- DeepSeekMoE — [arXiv:2401.06066](https://arxiv.org/abs/2401.06066)
- Auxiliary-Loss-Free Load Balancing — [arXiv:2408.15664](https://arxiv.org/abs/2408.15664)
- DeepSeek-V3 — [technical report](https://arxiv.org/abs/2412.19437) и [официальный репозиторий](https://github.com/deepseek-ai/DeepSeek-V3)
- gpt-oss — [официальный анонс](https://openai.com/index/introducing-gpt-oss/) и [открытая реализация](https://github.com/openai/gpt-oss)
- LatentMoE — [NVIDIA Research](https://research.nvidia.com/labs/nemotron/LatentMoE/)
- Nemotron 3 Ultra — [официальная страница](https://research.nvidia.com/labs/nemotron/Nemotron-3-Ultra/)
- Kimi K3 — [официальный репозиторий и tech report](https://github.com/MoonshotAI/Kimi-K3)
- Qwen3.5-35B-A3B — [официальная карточка](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base)
- GLM-5 — [официальная карточка и config](https://huggingface.co/zai-org/GLM-5)
- Soft MoE — [arXiv:2308.00951](https://arxiv.org/abs/2308.00951)
- ReMoE — [arXiv:2412.14711](https://arxiv.org/abs/2412.14711)
- SoftMoE 2026 — [arXiv:2606.17952](https://arxiv.org/abs/2606.17952)

**Дальше:** модуль 7 рассматривает архитектуры, которые меняют уже не FFN, а сам механизм перемещения информации по последовательности: рекуррентное состояние, State Space Models и гибриды с attention.

---

*Ландшафт сверен: 6 августа 2026 года. Численные реконструкции курса отделены от округлённых заявлений разработчиков; результаты отдельных работ сохраняют их экспериментальный протокол.*
