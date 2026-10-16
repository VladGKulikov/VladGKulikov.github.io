"""Localized teaching cells and the shared minimal classifier for M1."""
from pathlib import Path
import json
import ast

ROOT=Path(__file__).resolve().parent

def minimal_source(lang):
    source=(ROOT/'minimal_passport.py').read_text(encoding='utf8')
    original_names=[n.id for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Name)]
    if lang=='ru':
        translations={
          'A minimal model passport from a validated config.json.':'Минимальный паспорт по проверяемым полям config.json.',
          'Read one required positive integer field.':'Прочитать обязательное положительное целое поле.',
          'Read one alias field without hiding conflicting values.':'Прочитать поле-синоним и отклонить противоречие.',
          'Classify MHA (Multi-Head Attention), MQA (Multi-Query Attention), GQA, or MLA from validated fields.':'Различить MHA, MQA, GQA и MLA по поддерживаемым полям.',
          'check the MLA signature first':'сначала проверяем признак MLA',
          'Classify dense/MoE across common field aliases.':'Различить плотную модель и MoE с учётом полей-синонимов.',
          'Return a minimal structural passport without a network call.':'Вернуть структурный паспорт без сетевого вызова.',
          'must be a positive integer':'должно быть положительным целым',
          'conflicting aliases':'противоречащие поля-синонимы',
          'KV heads must divide Q heads':'число KV-голов должно делить число Q-голов',
          'expert settings require an expert count':'настройки экспертов требуют их количества',
          'expert count must be a positive integer':'число экспертов должно быть положительным целым',
          'top-k must be in [1, expert count]':'top-k должно лежать в диапазоне от 1 до числа экспертов',
          'n_shared_experts must be a non-negative integer':'число общих экспертов должно быть неотрицательным целым',
          'experts, top-':'экспертов, top-',
          ' Q heads /':' Q-голов /', ' heads)"':' голов)"', ' shared"':' общих"',
        }
        for a,b in translations.items():source=source.replace(a,b)
    assert original_names==[n.id for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Name)]
    return source


def document_cells(lang):
    fixture=json.loads((ROOT/'data/document_assistant.json').read_text(encoding='utf8'))
    localized=fixture['languages'][lang]
    source=(ROOT/'document_example.py').read_text(encoding='utf8').split('# BEGIN EDUCATIONAL CORE',1)[1].split('# END EDUCATIONAL CORE',1)[0].strip()
    if lang=='ru':
        for a,b in {
          'Check a structured day count and an exact reference passage for one fixed question.':'Проверить срок и точную эталонную цитату для одного фиксированного вопроса.',
          'The required passage is manually labelled for this question; it is not inferred.':'Подтверждающий фрагмент заранее размечен человеком для этого вопроса.',
          'days must be a nonnegative integer':'days должно быть неотрицательным целым',
          'document_id and quote must be strings':'document_id и quote должны быть строками',
        }.items():source=source.replace(a,b)
    introduction={
     'ru':'''## 1. Ответ и источник

Ассистент по документам помогает ответить на вопрос и показать, откуда взяты сведения.
Начнём с двух вымышленных инструкций из отдельного набора `document-assistant-v1`.
Одна устанавливает срок хранения журнала эксперимента, другая — временной выгрузки.
Тексты встроены ниже: редактирование лекции не меняет данные опыта.

В первой ячейке прочитайте вопрос и оба документа. Затем рассмотрим три **подготовленных**
ответа. Нейросеть и поиск здесь не запускаются: мы исследуем правило приёмки результата.
Поле `days` хранит целое число дней, `document_id` — имя источника, `quote` — цитату.
Эта структурированная форма позволяет увидеть проверку без задачи понимания свободного текста.''',
     'en':'''## 1. An answer and its evidence

A document assistant helps a user answer a question and inspect the source. Our first
experiment uses two fictional instructions in the separate `document-assistant-v1`
fixture: one concerns experiment logs, the other temporary exports. The texts are
embedded here, so editing the lecture cannot change the experiment.

Read the question and both documents in the first cell. We then inspect three **prepared**
responses. No neural model or retrieval engine is run; the object of this experiment is
the acceptance rule. Each response has a `days` integer, a `document_id`, and a `quote`.
This structured representation makes the check visible without pretending to understand
arbitrary prose.'''}[lang]
    data=f'DOCUMENTS = {localized["documents"]!r}\nQUESTION = {localized["question"]!r}\n'
    data+='print(QUESTION)\nfor name, passage in DOCUMENTS.items():\n    print(name, passage)\n'
    explanation={
     'ru':'''У ответа могут быть разные дефекты. Срок 7 дней неверен для журнала, даже если
ссылка ведёт к существующему документу A. Срок 30 дней верен, но документ B его не
подтверждает. Поэтому функция возвращает несколько признаков: правильность числа,
наличие цитаты в документе и соответствие заранее размеченному подтверждению.

Поле `required_quote` содержит полный эталонный фрагмент для этого вопроса. Проверка
ниже намеренно узкая: пересказ другими словами она может отклонить. Её нельзя применять
как универсальную проверку смысла ответа.''',
     'en':'''The errors differ. Seven days is wrong for an experiment log even with a real
citation to A. Thirty days is correct, but B does not support it. The checker therefore
returns separate fields for the answer, the presence of the quoted text, and agreement
with the manually labelled evidence.

`required_quote` is the complete reference passage for this question. This deliberately
narrow check may reject a valid paraphrase. It is not a general test of semantic support.'''}[lang]
    run='REFERENCE_ANSWER = '+repr(dict(fixture['reference'],required_quote=localized['documents']['A']))+'\n'
    run+='prepared = '+repr(fixture['prepared_answers'])+'\n'
    run+='answers = [dict(row, quote=DOCUMENTS[row["document_id"]]) for row in prepared]\n'
    run+='checks = [check_prepared_answer(row, DOCUMENTS, REFERENCE_ANSWER) for row in answers]\n'
    run+='labels = '+repr(localized['labels'])+'\n'
    run+='for label, result in zip(labels, checks):\n    print(label, result)\n'
    run+='assert checks == '+repr(fixture['expected'])+'\n'
    conclusion={
     'ru':'''Принимается только первый ответ. Второй ошибается в сроке, третий — в подтверждении.
Попробуйте заменить цитату первого ответа одной буквой или выдуманным предложением:
проверка должна перестать принимать его. В М17 мы построим поиск и проследим происхождение
фрагментов, а в М19 рассмотрим оценку произвольных результатов. Здесь мы установили
смысл одной принятой задачи — знаменателя будущего расчёта стоимости.''',
     'en':'''Only the first response is accepted. The second gets the period wrong; the third
uses evidence about a different object. Try replacing the first response's quote with
one letter or an invented sentence: it should no longer pass. Module 17 will build
retrieval and trace passage provenance; Module 19 will address evaluation more broadly.
For now we have made one accepted task concrete—the denominator in our later cost calculation.'''}[lang]
    return [('markdown',introduction),('code',data),('markdown',explanation),('code',source),('code',run),('markdown',conclusion)]


def resource_cells(lang):
    intro={
     'ru':'''## 2. Первые оценки памяти

Теперь оценим ресурсы условной модели с 8 миллиардами полных и 2 миллиардами активных
параметров. Это вымышленные размеры, выбранные для арифметики. Полное число нужно
умножить на байты одного веса; активное — использовать для грубой оценки матричных
операций при генерации токена. BF16 занимает 2 байта на вес, сырой 4-битный формат —
половину байта. При нечётном числе весов для последнего значения нужен ещё один
байт; код округляет объём вверх. GiB содержит 2³⁰ байт.

В этой ячейке видна вся формула. После запуска измените размер модели и объясните,
какие строки результата должны измениться. Полную память устройства здесь не считаем:
в неё дополнительно войдут состояние контекста и временные буферы.''',
     'en':'''## 2. First resource estimates

Use a hypothetical model with 8 billion total and 2 billion active parameters. These
invented dimensions keep the arithmetic small. Multiply the total count by bytes per
weight to estimate storage; use the active count for a coarse weight-matrix operation
estimate during token generation. BF16 occupies 2 bytes per weight, raw four-bit values
half a byte. An odd weight count needs a final partially filled byte, so the code
rounds storage up. A GiB contains 2³⁰ bytes.

The entire calculation is visible below. Change the model size and predict which outputs
will change. This counts weight payload; retained context state and workspaces would
add to device memory use.'''}[lang]
    code='''total = 8_000_000_000
active = 2_000_000_000
assert type(total) is int and type(active) is int and 0 < active <= total
bf16_bytes = total * 2
raw_4bit_bytes = (total + 1) // 2
weight_flop_per_token = 2 * active
print("BF16, GiB:", bf16_bytes / 2**30)
print("raw 4-bit, GiB:", raw_4bit_bytes / 2**30)
print("GFLOP/token:", weight_flop_per_token / 10**9)
# These numerical checks belong to the stated reference dimensions.
if total == 8_000_000_000:
    assert bf16_bytes == 16_000_000_000
    assert raw_4bit_bytes == 4_000_000_000
if active == 2_000_000_000:
    assert weight_flop_per_token == 4_000_000_000'''
    conclusion={
     'ru':'''При исходных размерах получаются 14,90 GiB для BF16, 3,73 GiB для сырых четырёхбитных весов и около
4 GFLOP на токен для весовых матричных операций. Квантизация дополнительно хранит
масштабы и другие служебные значения; разные тензоры могут иметь разную точность.
Число операций само по себе не задаёт время ответа. Подробный маршрут ниже проверит
такие оценки на сохранённых конфигурациях и объяснит соглашения о подсчёте active.''',
     'en':'''With the original dimensions, the results are 14.90 GiB for BF16, 3.73 GiB for raw four-bit weights, and about
4 GFLOP per token for weight-matrix arithmetic. Quantized formats also store scales and
other metadata, and tensors need not share one precision. An operation count does not
establish response time. The detailed route below applies the accounting to retained
configurations and makes the active-count conventions explicit.'''}[lang]
    return [('markdown',intro),('code',code),('markdown',conclusion)]
