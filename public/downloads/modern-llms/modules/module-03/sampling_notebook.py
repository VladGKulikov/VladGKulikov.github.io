"""Bilingual explanation and experiments for the M3 sampling lab."""
from pathlib import Path


INTRO = {
    'ru': r'''## 9. От логитов к следующему токену

Обученная выше модель продолжала период жадно: выбирала ID с максимальным
логитом. Теперь отделим оценки модели от правила выбора. Возьмём один вектор
логитов $[3,2,1,0]$ и оставим его неизменным. Температура меняет относительные
вероятности; top-k и top-p затем ограничивают множество кандидатов. После
исключения кандидатов оставшиеся вероятности нужно снова нормировать.

Функция `token_distribution` делает эти шаги явно. Сначала она проверяет
конечность логитов и допустимость параметров. Перед экспонентой вычитает
максимум: это сохраняет softmax и избегает переполнения. Затем сортирует ID,
при необходимости оставляет top-k и набирает кратчайший префикс с массой не
меньше top-p. При равенстве логитов меньший ID идёт первым. Порядок применения
двух фильтров здесь фиксирован: top-k, затем top-p внутри оставшейся массы.

`sampling_scenario` применяет четыре правила к одним логитам. На рисунке
фильтры используются **по отдельности**, при температуре один. Сначала
предскажите, сколько ID сохранит каждый из них; затем выполните ячейку.''',
    'en': r'''## 9. From logits to the next token

The tiny model above extended its period greedily, choosing the ID with the
highest logit. We can now hold the model's scores fixed and study the choice
rule itself. Start with $[3,2,1,0]$. Temperature changes relative probabilities;
top-k and top-p restrict the candidate set. Removing candidates requires a
second normalization before we can sample.

`token_distribution` exposes each operation. It checks finite logits and
valid parameters, subtracts the maximum before exponentiation to avoid
overflow without changing softmax, then sorts IDs. If requested, it keeps
top-k and takes the shortest prefix whose mass reaches top-p. Equal logits
are ordered by ID. When both filters are enabled, top-k runs first and top-p
measures mass within that retained distribution.

`sampling_scenario` compares four rules on the same scores. In the chart,
the two filters are applied **separately**, both at temperature one. Predict
how many IDs each filter will retain before running the next cell.''',
}

INTERPRET = {
    'ru': r'''На графике top-p сохраняет три ID: масса первых двух, около 0.8808,
ещё не достигает 0.9. После нормировки вероятность ID 0 равна примерно 0.6652,
а не 0.6439 из исходного softmax. Следующая ячейка сравнивает скалярный расчёт
с независимым `torch.softmax` и частотами 20 000 случайных выборов.
Частоты лишь приближают вероятности; фиксированный seed делает эту конкретную
пробу воспроизводимой. Он не превращает распределение в жадный выбор.

Для самостоятельной проверки замените порог 0.9 на 0.8: останутся два ID.
Затем задайте равные логиты и top-p = 0.5. До запуска объясните, почему при
четырёх ID достаточно первых двух и какое значение имеет правило разрешения
равенств.''',
    'en': r'''Top-p retains three IDs: the first two account for about 0.8808,
which is still below 0.9. Renormalization raises the probability of ID 0 to
about 0.6652, compared with 0.6439 in the original softmax. The next cell
checks the scalar calculation against an independent `torch.softmax` result
and the frequencies of 20,000 random draws. Frequencies approximate the
distribution; they need not equal it. A fixed seed reproduces this particular
sample, without turning sampling into greedy decoding.

Try a threshold of 0.8: only two IDs should remain. Then use four equal logits
and top-p = 0.5. Before running the code, explain why two IDs suffice and why
the tie-breaking rule matters.''',
}

STOPPING = {
    'ru': r'''### Завершение по маркеру и по лимиту

Правило выбора токена ещё не задаёт условие остановки. В небольшом опыте
ниже ID 3 обозначает EOS, конец последовательности. `greedy_trace` получает
**заранее подготовленные** векторы оценок для последовательных шагов,
выбирает максимум и останавливается по EOS или по `max_new_tokens`.
Это проверка управления циклом, а не работа обученной модели: оценки не
пересчитываются по полученному префиксу. В реальной генерации такой пересчёт
нужен на каждом шаге, как в `model.generate` выше.

Первый след заканчивается по EOS после двух токенов, второй — по лимиту после
трёх. Сохраните причину остановки рядом с результатом: обрезанный ответ может
выглядеть иначе, чем ответ, который модель завершила сама. Форматы диалога
и маркеры конца реплики подробно появятся в М13.''',
    'en': r'''### End marker versus length cap

A choice rule still needs a stopping rule. In the small experiment below,
ID 3 means EOS, the end of the sequence. `greedy_trace` reads **prepared**
score vectors for consecutive steps, chooses each maximum, and stops at EOS
or `max_new_tokens`. This isolates loop control. It is not a trained model:
the scores are not recomputed from the growing prefix. Actual generation,
including `model.generate` above, must perform that computation at each step.

The first trace ends at EOS after two tokens; the second reaches its cap
after three. Store the reason alongside the tokens: a truncated answer can
behave differently from one the model ended itself. Module 13 will extend
this discussion to dialogue formats and end-of-turn markers.''',
}


def append_sampling_cells(md, code, language):
    md(INTRO[language])
    source = (Path(__file__).parent / 'sampling_lab.py').read_text(encoding='utf-8')
    if language == 'ru':
        source = source.replace('A transparent scalar reference for next-token selection policies.',
                                'Скалярный расчёт распределения для выбора следующего токена.')
        source = source.replace('Return probabilities for one finite logit vector; apply temperature, k, then p.',
                                'Вычислить вероятности: температура, затем top-k и top-p.')
        source = source.replace('Compute four policies on the same illustrative logits, with filters applied separately.',
                                'Сравнить четыре правила на одних логитах; фильтры применяются отдельно.')
    code(source)
    code(f'''scenario = sampling_scenario()
for name, values in scenario.items():
    print(name, [round(value, 6) for value in values])
fig = sampling_plot(scenario, {language!r})
plt.show()
plt.close(fig)''')
    md(INTERPRET[language])
    code('''import random

z = torch.tensor(scenario['logits'], dtype=torch.float64)
expected = torch.softmax(z, dim=0)
torch.testing.assert_close(torch.tensor(scenario['temperature_1'], dtype=torch.float64), expected)
torch.testing.assert_close(torch.tensor(scenario['temperature_2'], dtype=torch.float64), torch.softmax(z / 2, dim=0))
for policy, count in [('top_k_2', 2), ('top_p_09', 3)]:
    truncated = torch.zeros_like(z)
    truncated[:count] = torch.softmax(z[:count], dim=0)
    torch.testing.assert_close(torch.tensor(scenario[policy], dtype=torch.float64), truncated)
assert token_distribution([0., 0., 0., 0.], top_p=0.5) == [0.5, 0.5, 0., 0.]
assert token_distribution([0., 0.], top_k=1) == [1., 0.]

probabilities = scenario['top_p_09']
draws = random.Random(7).choices(range(4), weights=probabilities, k=20_000)
frequencies = [draws.count(i) / len(draws) for i in range(4)]
assert max(abs(a - b) for a, b in zip(frequencies, probabilities)) < 0.015
print({'probabilities': probabilities, 'frequencies': frequencies})''')
    md(STOPPING[language])
    code('''def greedy_trace(prepared_logits, eos_id, max_new_tokens):
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, Integral) or max_new_tokens <= 0:
        raise ValueError('max_new_tokens must be a positive integer')
    if isinstance(eos_id, bool) or not isinstance(eos_id, Integral) or eos_id < 0:
        raise ValueError('eos_id must be a nonnegative integer')
    generated = []
    for scores in prepared_logits:
        probabilities = token_distribution(scores)
        if eos_id >= len(probabilities):
            raise ValueError('eos_id is outside the score vector')
        token = max(range(len(probabilities)), key=probabilities.__getitem__)
        generated.append(token)
        if token == eos_id:
            return generated, 'eos'
        if len(generated) == max_new_tokens:
            return generated, 'length_limit'
    raise ValueError('prepared trace ended before a stopping condition')

eos_result = greedy_trace([[3., 2., 1., 0.], [0., 0., 0., 3.]], eos_id=3, max_new_tokens=3)
limit_result = greedy_trace([[3., 2., 1., 0.]] * 3, eos_id=3, max_new_tokens=3)
assert eos_result == ([0, 3], 'eos')
assert limit_result == ([0, 0, 0], 'length_limit')
print(eos_result, limit_result)''')
