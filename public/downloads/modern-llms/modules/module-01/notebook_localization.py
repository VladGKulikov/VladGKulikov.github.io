"""Translate explanatory docstrings without changing executable notebook logic."""
import ast

DOCS = {
 'ConfigSchemaError':'Конфигурация нарушает поддерживаемую учебную схему.',
 'ConfigIntegrityError':'Контрольная сумма отличается от явно ожидаемой.',
 '_positive_int':'Положительное целое; логические значения отвергаются.',
 '_non_negative_int':'Неотрицательное целое; логические значения отвергаются.',
 '_finite_positive_real':'Положительное конечное вещественное число.',
 '_string':'Непустая строка без внешних пробелов.',
 '_mapping':'Поверхностная копия словаря со строковыми ключами.',
 '_synonym_value':'Прочитать одно значение полей-синонимов и отклонить конфликт.',
 'strict_json_loads':'Прочитать JSON-объект; отклонить повторные ключи и неконечные константы.',
 'canonical_json':'Детерминированная запись совместимых с JSON данных в UTF-8.',
 'sha256_json':'SHA-256 байтов канонического JSON.',
 'validate_repo_id':'Проверить имя репозитория Hugging Face из двух частей.',
 'validate_commit_revision':'Требовать неизменяемый Git commit из 40 шестнадцатеричных символов.',
 'validate_source_revision':'Проверить ветку, тег, PR ref или commit для необязательной загрузки.\n\n    Принимаются только формы из урока. Пробелы, управляющие символы, повторные\n    разделители пути и неоднозначные конструкции Git отвергаются.',
 'expert_count':'Число маршрутизируемых экспертов с проверкой согласованности полей.',
 'experts_per_token':'Число выбранных экспертов top-k с проверкой полей-синонимов.',
 'shared_expert_count':'Неотрицательное число общих экспертов, активных для каждого токена.',
 'resolved_head_dim':'Прочитать head_dim или вывести его только при точном делении ширины на число голов.',
 'validate_config':'Проверить используемые паспортом поля и вернуть копию.\n\n    Неизвестные поля сохраняются. Строгие ограничения относятся к полям учебных\n    формул: новую архитектуру нельзя молча подставить в формулу знакомого семейства.',
 'SnapshotModel':'Проверенный пример модели и заданное соглашение о подсчёте active.',
 'ReferenceBundle':'Проверенный набор конфигураций, допущений и контекстных окон.',
 'load_reference_bundle':'Проверить сохранённый набор и вернуть типизированные неизменяемые записи.',
 'attention_kind':'Различить MHA, GQA, MQA и MLA после проверки поддерживаемых полей.',
 'moe_kind':'Различить плотную модель и MoE по проверенным полям-синонимам.',
 'context_info':'Описать максимум позиций и необязательное масштабирование RoPE.',
 'ParameterBreakdown':'Подсчёт параметров одного поддерживаемого семейства Transformer.',
 'active':'Активный размер по одному явно указанному соглашению.',
 '_architecture_name':'Первое объявленное имя архитектуры или пустая строка.',
 'attention_parameter_count':'Число параметров внимания в одном слое поддерживаемого семейства.\n\n    Для gpt-oss учитываются смещения QKV и выходной проекции. Для DeepSeek-V3\n    включены масштабы нормализации Q и сжатого KV. Это реконструкция с указанным\n    составом параметров, а не только сумма крупных матриц.',
 'ffn_parameter_count':'Полное и активное число параметров FFN и маршрутизатора одного слоя.',
 'count_parameters':'Полный размер и активные размеры по поддерживаемым соглашениям.',
 'ReconciliationRow':'Вычисленный паспорт и его сверка с заявленными числами карточки.',
 'reconcile_model':'Сверить модель, не выбирая соглашение задним числом под ответ.',
 'CostEstimate':'Память весов и приближённое число операций матричных умножений.',
 'cost_estimate':'Память только весов и приближение 2 × active для матричных операций.',
 'flatten_config':'Развернуть вложенную конфигурацию в пути полей для сравнения.',
 'config_diff':'Детерминированные изменения путей между проверенными конфигурациями.',
 'FetchedConfig':'Проверенная конфигурация, связанная с неизменяемой ревизией репозитория.',
 'SourceConfig':'Конфигурация, исходное изменяемое имя и фактически прочитанный commit.',
 'fetch_config_with_transport':'Загрузить и проверить конфигурацию через переданную функцию чтения.\n\n    Ошибка сети, доступа или ревизии остаётся видимой. Сохранённый набор\n    и живая загрузка — разные операции; автоматической подмены источника нет.',
 'fetch_source_config_with_transports':'Разрешить ветку или тег в commit, затем прочитать эту точную ревизию.\n\n    Имя main само по себе невоспроизводимо. Возвращённый resolved_commit\n    позволяет повторить наблюдение. Ошибки разрешения, загрузки, доступа,\n    схемы и контрольной суммы не заменяют данные сохранённым набором.',
 '_hf_transport':'Загрузить только config.json по точному commit и прочитать UTF-8.',
 '_hf_revision_resolver':'Разрешить имя источника Hub в полный commit SHA.',
 'fetch_hf_config':'Прочитать конфигурацию по точной ревизии без скрытой подмены данных.',
 'fetch_hf_config_from_source':'Прочитать источник и записать фактически использованный commit.\n\n    Ветка main или тег допустимы для явно включённого опыта. Для повторения\n    сохраните resolved_commit и передайте его в fetch_hf_config.',
 'object_pairs':'Собрать объект, отклоняя повторный ключ на его уровне вложенности.',
 'reject_constant':'Отклонить неконечные константы, разрешённые стандартным парсером Python.',
 'task_cost':'Учесть все попытки и каждую принятую задачу один раз; None означает отсутствие успехов.',
 'kv_payload_bytes':'Память K и V одного запроса с отдельной длиной хранения для каждого слоя.',
 'expect_error':'Проверить, что заданный некорректный вход вызывает ожидаемую ошибку.',
 'fake_resolver':'Разрешить изменяемое имя в фиксированный учебный commit.',
 'failing_transport':'Явно сымитировать ошибку загрузки без подмены данных.',
 'failing_resolver':'Явно сымитировать ошибку определения ревизии.',
}

DOCS.update({
 'object_pairs_hook':DOCS['object_pairs'], 'parse_constant':DOCS['reject_constant'],
 'WeightCost':DOCS['CostEstimate'], 'weight_cost':DOCS['cost_estimate'],
 'SourceFetchedConfig':DOCS['SourceConfig'],
 'huggingface_hub_transport':DOCS['_hf_transport'],
 'huggingface_hub_revision_resolver':DOCS['_hf_revision_resolver'],
 'fake_transport':'Вернуть локальную конфигурацию и записать запрошенную точную ревизию.',
 'draw_context':'Логарифмический точечный график заявленных окон; качество здесь не измеряется.',
 'draw_parameters':'Показать total и active на одной оси; их отношение не задаёт задержку.',
})

def russian_explanations(source):
    lines=source.splitlines(keepends=True)
    offsets=[0]
    for line in lines:offsets.append(offsets[-1]+len(line))
    edits=[]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in DOCS and ast.get_docstring(node):
            value=node.body[0]
            a=len(lines[value.lineno-1].encode('utf8')[:value.col_offset].decode('utf8'))
            b=len(lines[value.end_lineno-1].encode('utf8')[:value.end_col_offset].decode('utf8'))
            edits.append((offsets[value.lineno-1]+a,
                          offsets[value.end_lineno-1]+b,
                          '"""'+DOCS[node.name]+'"""'))
    for a,b,text in sorted(edits,reverse=True):source=source[:a]+text+source[b:]
    for a,b in {
      '# These numerical checks belong to the stated reference dimensions.':'# Эти численные проверки относятся к исходным размерам примера.',
      '# Qwen3 applies a learned RMSNorm to Q and K, each with head_dim weights.':'# Qwen3 нормирует Q и K: у каждой RMSNorm по head_dim обучаемых весов.',
      '# retained as an explicit local invariant for readability':'# Явный локальный инвариант помогает читать проверку.',
      '# Independent arithmetic: enumerate the K and V payload of each layer.':'# Независимая проверка: перебрать память K и V каждого слоя.',
      '# Exercise E5: a cheaper token is not enough, and a lower ratio is not a quality gate.':'# У5: цена токена и расходы на успех не заменяют порог качества.',
      'print("eligible:", eligible)':'print("Прошли порог:", eligible)',
      'print("explicit convention:",':'print("Заданное соглашение:",',
      'print("payload-only fit is not a deployment-memory guarantee")':'print("Поместившиеся веса не гарантируют достаточную память для запуска")',
    }.items():source=source.replace(a,b)
    ast.parse(source)
    return source
