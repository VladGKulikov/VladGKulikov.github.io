#!/usr/bin/env python3
"""Export reviewed, canonical course questions. Never modify course repositories.

Modern's module URLs use its current module numbers. RL's platform bank contains both
objective quizzes and open/code tasks; the latter retain their worked answers
for honest self-checking, without executing submitted code or keyword grading.
"""
import argparse
import hashlib
import json
import re
import runpy
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    'modern-llms': Path(r'C:\PersonalProjects\Modern LLMs'),
    'rl-for-llm': Path(r'C:\PersonalProjects\RL for LLM'),
    'information-theory-for-ml': Path(r'C:\PersonalProjects\Information_Theory_for_ML'),
}
LECTURE_TOOLS = runpy.run_path(str(ROOT / 'scripts/refresh-course-content.py'))
QUESTION = re.compile(r'^\*\*(?:Вопрос|Question)\s+(\d+(?:\.[\w]+)+)([^*\n]*)\*\*\s*(.*)$', re.M)
ANSWER = re.compile(r'^\*\*(Правильн(?:ый|ые) ответ[ы]?|Correct answers?|Числовой ответ|Numeric(?:al)? answer):\s*(.*?)\*\*\s*(.*)$', re.M)
OPTION = re.compile(r'^\s*-\s+([A-Z])\)\s+(.+)$', re.M)

def digest(text):
    return hashlib.sha256(text.replace('\r\n', '\n').encode()).hexdigest()

def normalize(text):
    text = re.sub(r'<!--.*?-->', '', text, flags=re.S)
    text = re.sub(r'</?(?:strong|b)>', '**', text, flags=re.I)
    text = re.sub(r'</?(?:em|i)>', '*', text, flags=re.I)
    text = LECTURE_TOOLS['normalize_math_delimiters'](text)
    text = LECTURE_TOOLS['normalize_reader_markup'](text)
    # Heading fragments are rendered inside a question, below its h2.
    text = re.sub(r'^#{1,6}\s+(.+)$', r'**\1**', text, flags=re.M)
    return text.strip()

def parse_questions(source):
    text = source.read_text(encoding='utf-8')
    matches = list(QUESTION.finditer(text))
    result = []
    for i, match in enumerate(matches):
        body = text[match.end():matches[i+1].start() if i+1 < len(matches) else len(text)]
        answer = ANSWER.search(body)
        if not answer:
            raise ValueError(f'Missing answer: {source}:{match[1]}')
        before = body[:answer.start()]
        options = list(OPTION.finditer(before))
        prompt = match[3] + '\n\n' + before[:options[0].start() if options else len(before)]
        # Exclude the next topic heading/footer, but preserve all answer paragraphs.
        explanation = body[answer.start():]
        explanation = re.split(r'^#{1,3}\s+', explanation, maxsplit=1, flags=re.M)[0]
        explanation = re.sub(r'\n---\s*$', '', explanation).strip()
        q = {'id': match[1], 'prompt': normalize(prompt), 'explanation': normalize(explanation)}
        topics = re.findall(r'^##\s+(.+)$', text[:match.start()], flags=re.M)
        if topics: q['topic'] = normalize(topics[-1])
        note = match[2].strip(' .')
        if note:
            q['note'] = note
        if options:
            q.update(type='single', options=[{'value': o[1], 'text': normalize(o[2])} for o in options])
            correct = re.findall(r'\b[A-Z]\b', answer[2])
            assert correct and set(correct) <= {o['value'] for o in q['options']}, (source, q['id'], answer[0])
            q['correct'] = correct
            q['type'] = 'multiple' if len(correct) > 1 else 'single'
        else:
            key = re.search(r'[-+]?\d+(?:[.,]\d+)?', answer[2])
            tolerance = re.search(r'(?:допуск|tolerance)\s*(?:±\s*)?([0-9]+(?:[.,]\d+)?)', answer[3] + '\n' + prompt, re.I)
            assert key and tolerance, (source, q['id'], answer[0])
            q.update(type='numeric', fields=[{'label': 'Ответ' if 'Вопрос' in match[0] else 'Answer',
                'expected': float(key[0].replace(',', '.')), 'tolerance': float(tolerance[1].replace(',', '.'))}])
        result.append(q)
    assert result and len({q['id'] for q in result}) == len(result), source
    return text, result

def plain(text):
    return re.sub(r'[\s*`_]+', '', text).strip('.').casefold()

def parse_rl(records, module, lang, overrides):
    result = []
    for record in records:
        if record['route'] != 'platform':
            continue
        body, solution = record['body'], record['solution']
        spec = overrides.get(f'{module}:{record["id"]}', {})
        q = {'id': record['id'], 'title': record['title'], 'prompt': normalize(body), 'explanation': normalize(solution), 'type': 'self-check'}
        if spec.get('type') == 'numeric':
            q.update(type='numeric', fields=[{'label': f['label'][lang], 'expected': f['expected'], 'tolerance': f['tolerance']} for f in spec['fields']])
            if spec.get('manualReview'):
                q['manualReview'] = True
        elif spec.get('type') == 'matching':
            if 'options' in spec:
                options = [{'value': str(i+1), 'text': label[lang]} for i,label in enumerate(spec['options'])]
            else:
                options = [{'value': m[1], 'text': normalize(m[2])} for m in re.finditer(r'^([A-D])\.\s+(.+?)\s*$', body, re.M)]
            assert options, (module, record['id'], lang)
            q.update(type='matching', options=options, correct=spec['correct'], matchLabels=spec['labels'])
            if spec.get('manualReview'): q['manualReview'] = True
        else:
            checkboxes = list(re.finditer(r'^- \[[ xX]\]\s+(.+)$', body, re.M))
            letters = list(re.finditer(r'^\s*(?:- )?([A-D])[.)]\s+(.+?)\s*$', body, re.M))
            numbered = list(re.finditer(r'^(\d+)\.\s+(.+?)\s*$', body, re.M)) if spec.get('numbered') else []
            options = checkboxes or (letters if spec.get('correct') else []) or numbered
            if options and spec.get('type') != 'self-check':
                labels = [o[1] if checkboxes else o[2] for o in options]
                values = [str(i+1) for i in range(len(options))]
                correct = spec.get('correct')
                if correct is None:
                    correct = [values[i] for i, label in enumerate(labels) if plain(label) in plain(solution)]
                assert correct and set(correct) <= set(values), (module, record['id'], lang, correct)
                # Preserve any post-options instruction, including requests to explain.
                tail = body[options[-1].end():].strip()
                q.update(type='multiple' if len(correct) > 1 else 'single', correct=correct,
                    prompt=normalize(body[:options[0].start()] + ('\n\n' + tail if tail else '')),
                    options=[{'value': v, 'text': normalize(t)} for v, t in zip(values, labels)])
                if spec.get('manualReview'):
                    q['manualReview'] = True
        result.append(q)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', action='append', default=[], metavar='COURSE=PATH')
    args = parser.parse_args()
    roots = dict(DEFAULTS)
    for value in args.source_root:
        key, value = value.split('=', 1); roots[key] = Path(value)
    overrides = json.loads((ROOT / 'scripts/quiz-grading-overrides.json').read_text(encoding='utf8'))
    sys.path.insert(0, str(roots['rl-for-llm'] / 'tools'))
    from assessment_source import load_all_assessments
    manifest = json.loads((roots['modern-llms'] / 'course_manifest.json').read_text(encoding='utf8'))
    modern_ids = {f'module-{m["id"]}': m['id'] for m in manifest['modules']}
    catalog = json.loads((ROOT / 'app/generated/content.json').read_text(encoding='utf8'))
    entries, sources = [], []
    for course in catalog['courses']:
        key, lang = course['course_key'], course['language']
        for module in course['modules']:
            mk = module['module_key']; n = int(module['position'])
            if key == 'modern-llms':
                source = roots[key] / f'modules/module-{modern_ids[mk]}/{lang}/module_{modern_ids[mk]}_tests_{lang.upper()}.md'
            elif key == 'information-theory-for-ml':
                if n == 16:  # Reference appendix has no authored quiz.
                    continue
                source = roots[key] / f'Module_{n:02d}/Module_{n}__Test_questions{"_RU" if lang == "ru" else ""}.md'
            else:
                source = roots[key] / f'modules/module-{n:02d}/Module_{n}_Assessment_{lang.upper()}.md'
            source_text = source.read_text(encoding='utf8')
            if key == 'rl-for-llm':
                questions = parse_rl(load_all_assessments(roots[key], n, lang.upper()), n, lang, overrides)
            else:
                source_text, questions = parse_questions(source)
            if key == 'information-theory-for-ml':
                # The translations number a few optional topics differently.
                # Their question order and grading contracts are aligned below.
                for i, q in enumerate(questions):
                    q['sourceId'], q['id'] = q['id'], f'item-{i+1:03d}'
            relative = source.relative_to(roots[key]).as_posix()
            # New linked assets require an explicit publication/copying decision;
            # never guess a source-repository URL or send them to a lecture page.
            for q in questions:
                for field in ('prompt', 'explanation'):
                    for target in re.findall(r'\[[^\]]+\]\(([^)\s]+)\)', q[field]):
                        assert re.match(r'^(?:https?:|mailto:|#|/)', target), (source, q['id'], 'Unpublished relative asset', target)
            file = f'{key}/{lang}/{mk}.json'
            data = {'courseKey': key, 'moduleKey': mk, 'language': lang, 'questions': questions}
            raw = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
            output = ROOT / 'public/quizzes' / file
            output.parent.mkdir(parents=True, exist_ok=True); output.write_text(raw, encoding='utf8')
            entries.append({'courseKey': key, 'moduleKey': mk, 'language': lang, 'file': file, 'hash': digest(raw),
                'count': len(questions), 'kinds': dict(Counter(q['type'] for q in questions))})
            sources.append({'course': key, 'module': mk, 'language': lang, 'source': relative, 'sha256': digest(source_text), 'count': len(questions)})
    # Fail on translation drift: stable IDs, option values and grading contracts
    # must agree. Translation-specific prose is intentionally excluded.
    for entry in entries:
        counterpart = next(e for e in entries if e['courseKey'] == entry['courseKey'] and e['moduleKey'] == entry['moduleKey'] and e['language'] != entry['language'])
        def contract(e):
            qs = json.loads((ROOT / 'public/quizzes' / e['file']).read_text(encoding='utf8'))['questions']
            return [{'id': q['id'], 'type': q['type'], 'correct': q.get('correct'), 'values': [o['value'] for o in q.get('options', [])],
                'fields': [{k:f[k] for k in ('expected','tolerance')} for f in q.get('fields', [])], 'matchLabels': q.get('matchLabels')} for q in qs]
        a,b = contract(entry), contract(counterpart)
        assert a == b, (entry['file'], 'RU/EN grading contracts differ', [(x,y) for x,y in zip(a,b) if x != y])
        entry['revision'] = digest(json.dumps(a, sort_keys=True))[:16]
    (ROOT / 'app/generated/quizzes.json').write_text(json.dumps({'modules': entries}, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    report = {'status': 'PASS', 'sources': sources, 'counts': {key: sum(e['count'] for e in entries if e['courseKey']==key and e['language']=='en') for key in roots}}
    (ROOT / 'QUIZ_CONTENT_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status': 'PASS', 'localizedModules': len(entries), 'questionsPerLanguage': report['counts']},ensure_ascii=False))

if __name__ == '__main__': main()
