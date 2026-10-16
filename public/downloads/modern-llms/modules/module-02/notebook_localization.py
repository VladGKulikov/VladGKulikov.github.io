"""Localize learner-facing prose without changing the shared numerical program."""
import io
import token
import tokenize

RU = {
    'Change these inputs for your experiment; fixed model examples stay below.': 'Меняйте эти входы для своего опыта; ниже остаются фиксированные примеры моделей.',
    'your configuration': 'ваша конфигурация',
    'Decode one embedded asset and verify its SHA-256 digest.': 'Распаковывает встроенные данные и проверяет их SHA-256.',
    'Apply one non-overlapping merge independently to every corpus chunk.': 'Выполняет одно неперекрывающееся слияние отдельно в каждом чанке.',
    'Configure an exact trainer for a non-negative merge budget.': 'Задаёт целое неотрицательное число шагов слияния.',
    'Learn an ordered merge table while preserving corpus-order tie-breaks.': 'Обучает таблицу слияний; равные частоты разрешаются порядком обхода корпуса.',
    'dict preserves first-appearance order, which preserves the tie-break.': 'dict сохраняет порядок первого появления, а значит, и выбор при равных частотах.',
    'Reconstruct token bytes implied by the learned merge table.': 'Восстанавливает байты токенов по таблице слияний.',
    'Validate an ordered merge table and reconstruct its byte vocabulary.': 'Проверяет упорядоченную таблицу слияний и восстанавливает байтовый словарь.',
    'Return the vocabulary size, including the 256 byte tokens.': 'Возвращает размер словаря, включая 256 байтовых токенов.',
    'Return a defensive copy of the token-id to byte-string vocabulary.': 'Возвращает копию словаря ID → байты, чтобы вызывающий код не менял оригинал.',
    'Encode Unicode text by regex chunks and greedy merge rank.': 'Кодирует Unicode-текст по чанкам, выбирая наименьший ранг слияния.',
    'Join token bytes and decode UTF-8 with an explicit error policy.': 'Объединяет байты токенов и декодирует UTF-8 с явной политикой ошибок.',
    'Serialize the ordered merge table to a JSON-compatible mapping.': 'Представляет упорядоченную таблицу слияний в формате, совместимом с JSON.',
    'Serialize this teaching tokenizer to deterministic JSON text.': 'Сохраняет учебный токенизатор в детерминированный JSON.',
    'Construct a tokenizer from a validated JSON-compatible mapping.': 'Создаёт токенизатор из проверенной структуры данных JSON.',
    'Construct a tokenizer from serialized JSON text.': 'Создаёт токенизатор из строки JSON.',
    'Reconstruct and compare the complete byte vocabulary stored in the artifact.': 'Восстанавливаем и сравниваем весь байтовый словарь сохранённого артефакта.',
    'Fast provenance check: retrain a substantial exact prefix from frozen data.': 'Проверяем происхождение: заново обучаем начальную часть правил на фиксированных данных.',
    'The range contains reserved_200018, which aliases endofprompt at the same ID.': 'reserved_200018 и endofprompt ссылаются на один ID.',
    'Private mappings are inspected deliberately: pin the printed version in reports.': 'Здесь намеренно читаются внутренние поля; фиксируйте выведенную версию в отчёте.',
    'Slow reference: apply every learned merge in training order.': 'Медленный эталон: применяет все слияния по порядку обучения.',
    'A single leading byte of a four-byte UTF-8 sequence is intentionally invalid.': 'Один начальный байт четырёхбайтовой последовательности намеренно нарушает UTF-8.',
    'Normalization changes what can be recovered: an illustrative NFKC example.': 'Нормализация меняет восстанавливаемый текст: пример NFKC.',
    'Build the tokenizer induced by the first merge_count frozen rules.': 'Строит токенизатор по первым merge_count сохранённым правилам.',
    'Return UTF-8 input bytes per emitted token for non-empty output.': 'Делит число байтов UTF-8 на число токенов; результат кодирования должен быть непустым.',
    'Measure UTF-8 bytes per token for the HuggingFace comparison model.': 'Вычисляет байты UTF-8 на токен для сопоставляемой модели HuggingFace.',
    'Return best-of-repeat encode throughput in UTF-8 bytes per second.': 'Возвращает лучшую скорость кодирования из повторов, в байтах UTF-8 в секунду.',
    'Small NumPy embedding lab; the source is embedded in both learner notebooks.': 'Небольшая практика по эмбеддингам на NumPy; код встроен в оба учебных ноутбука.',
    'This is an operation-level example with chosen weights, not a trained language model.': 'Пример показывает операции с заданными весами; языковая модель здесь не обучена.',
    'No files, downloads, GPU, or repository imports are required by this source.': 'Для выполнения не нужны файлы, загрузки, GPU или импорты из репозитория.',
    'Normalize the final axis, subtracting its maximum for numerical stability.': 'Нормирует по последней оси; вычитание максимума обеспечивает численную устойчивость.',
    'Count unique weight parameters and payload, excluding bias and training state.': 'Считает уникальные веса и их байты, без смещения и состояний обучения.',
    'A chosen state, not a computed Transformer output.': 'Состояние задано вручную; Transformer его не вычислял.',
    'Tied projection is not an inverse lookup.': 'Проекция со связанными весами не обращает выбор строки.',
    'One learning step on a simple quadratic objective, not next-token cross-entropy.': 'Шаг обучения с квадратичной потерей; это не кросс-энтропия следующего токена.',
    'Repeated ID 0 must accumulate twice.': 'Два вхождения ID 0 должны дать два вклада.',
    'Advanced indexing writes through a temporary; repeated IDs do not accumulate.': 'Расширенная индексация создаёт временный массив; повторы ID не суммируются.',
    'Independent finite-difference check catches incorrect repeated-index assignment.': 'Независимая проверка конечными разностями обнаруживает ошибку при повторении индексов.',
    'Optional tokenizers installation failed:': 'Не удалось установить необязательный tokenizers:',
    'Versions:': 'Версии:',
    'Snapshot date:': 'Дата снимка:',
    'o200k_base: defined IDs / ID range:': 'o200k_base: определённые ID / диапазон ID:',
    'Qwen3-8B: defined IDs / model rows:': 'Qwen3-8B: определённые ID / строки модели:',
    'tiktoken version:': 'Версия tiktoken:',
    'tokens/sentence:': 'токенов/предложение:',
    'tokens:': 'токенов:',
    'EN-trained': 'Обучен на EN',
    'encode throughput: teaching Python=': 'скорость кодирования: учебный Python=',
    'HF skipped': 'HF пропущен',
    'ratio=': 'отношение=',
    '; training=': '; обучение=',
    'lookup:': 'выбранные строки:',
    'tied / untied logits:': 'логиты: связанные / независимые веса:',
    'tied probabilities:': 'вероятности при связанных весах:',
    'gradient:': 'градиент:',
    'loss before / after:': 'потеря до / после:',
    'num_merges must be a non-negative integer': 'num_merges должен быть целым неотрицательным числом',
    'dimensions and bytes_per_parameter must be positive integers': 'размеры и bytes_per_parameter должны быть положительными целыми числами',
    'tied must be a boolean': 'tied должен быть логическим значением',
}


def localize(source, language):
    if language != 'ru':
        return source
    tokens = []
    prose_types = {tokenize.COMMENT, tokenize.STRING, getattr(token, 'FSTRING_MIDDLE', -1)}
    for item in tokenize.generate_tokens(io.StringIO(source).readline):
        if item.type in prose_types and len(item.string) < 20000:
            value = item.string
            for original, translated in RU.items():
                value = value.replace(original, translated)
            item = item._replace(string=value)
        tokens.append(item)
    return tokenize.untokenize(tokens)
