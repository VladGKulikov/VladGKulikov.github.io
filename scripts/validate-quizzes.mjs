import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { quizModules, gradeQuestion, parseNumericAnswer, parseUrl, buildLocation } from '../build-static/pages-render.js';

const counts = { pages: 0, questions: 0, automatic: 0, selfCheck: 0, assertions: 0 };
const checks = (actual, expected, message) => { assert.deepEqual(actual, expected, message); counts.assertions++; };
const contract = q => ({ id: q.id, type: q.type, correct: q.correct, values: q.options?.map(o => o.value), fields: q.fields?.map(({expected, tolerance}) => ({expected, tolerance})), matchLabels:q.matchLabels });
const banks = new Map();
for (const entry of quizModules) {
  const raw = fs.readFileSync(`public/quizzes/${entry.file}`, 'utf8');
  checks(createHash('sha256').update(raw.replaceAll('\r\n', '\n')).digest('hex'), entry.hash, entry.file);
  const bank = JSON.parse(raw); banks.set(entry.file, bank);
  checks(bank.questions.length, entry.count, entry.file);
  checks(new Set(bank.questions.map(q => q.id)).size, entry.count, 'Unique question IDs');
  const url = buildLocation(entry.language, entry.courseKey, entry.moduleKey, 'tests');
  const route = parseUrl(new URL(url, 'https://vladkulikov.com'));
  checks([route.language, route.courseKey, route.moduleKey, route.section], [entry.language, entry.courseKey, entry.moduleKey, 'tests'], url);
  for (const q of bank.questions) {
    counts.questions++;
    assert(q.prompt && q.explanation, `${entry.file}:${q.id} empty question/answer`);
    assert(!/<\/?(?:strong|details|summary|script)\b/i.test(q.prompt+q.explanation), `${entry.file}:${q.id} raw markup`);
    if (q.type === 'self-check') { counts.selfCheck++; checks(gradeQuestion(q, []), 'self-check'); continue; }
    counts.automatic++;
    checks(gradeQuestion(q, []), 'incomplete', q.id);
    if (q.type === 'numeric') {
      assert(q.fields.every(f => Number.isFinite(f.expected) && Number.isFinite(f.tolerance) && f.tolerance >= 0));
      checks(gradeQuestion(q, q.fields.map(f => String(f.expected))), 'correct', q.id);
      checks(gradeQuestion(q, q.fields.map(f => String(f.expected).replace('.', ','))), 'correct', q.id);
      checks(gradeQuestion(q, q.fields.map((f,i) => String(f.expected + (i === 0 ? f.tolerance * 2 + 1 : 0)))), 'incorrect', q.id);
      checks(gradeQuestion(q, q.fields.map((f,i) => i === 0 ? 'NaN' : String(f.expected))), 'incomplete', q.id);
      checks(gradeQuestion(q, q.fields.map((f,i) => i === 0 ? '1/0' : String(f.expected))), 'incomplete', q.id);
      checks(gradeQuestion(q, q.fields.map(f => String(f.expected + f.tolerance))), 'correct', 'Inclusive absolute tolerance');
    } else if (q.type === 'matching') {
      checks(q.matchLabels.length, q.correct.length);
      checks(gradeQuestion(q, q.correct), 'correct');
      checks(gradeQuestion(q, q.correct.map(() => '')), 'incomplete');
      checks(gradeQuestion(q, [...q.correct].reverse()), 'incorrect', 'Matching order matters');
    } else {
      assert(q.options.length >= 2 && q.correct.length >= 1 && new Set(q.options.map(o => o.value)).size === q.options.length);
      assert(q.correct.every(v => q.options.some(o => o.value === v)));
      if (q.type === 'single') checks(q.correct.length, 1);
      checks(gradeQuestion(q, q.correct), 'correct', q.id);
      checks(gradeQuestion(q, [...q.correct].reverse()), 'correct', 'Order-independent selection');
      checks(gradeQuestion(q, [...q.correct, '__unknown__']), 'incorrect', 'Extra answers never pass');
      checks(gradeQuestion(q, [...q.correct, q.correct[0]]), 'incorrect', 'Duplicate answers never pass');
      if (q.type === 'multiple') checks(gradeQuestion(q, q.correct.slice(1)), 'incorrect', 'Partial multi-select never passes');
      for (const option of q.options) checks(gradeQuestion(q, [option.value]), q.correct.length === 1 && q.correct[0] === option.value ? 'correct' : 'incorrect', q.id);
    }
  }
  counts.pages++;
}
for (const entry of quizModules.filter(e => e.language === 'ru')) {
  const other = quizModules.find(e => e.language === 'en' && e.courseKey === entry.courseKey && e.moduleKey === entry.moduleKey);
  assert(other, 'Missing translation');
  checks(entry.revision, other.revision, 'Language-independent progress key');
  checks(banks.get(entry.file).questions.map(contract), banks.get(other.file).questions.map(contract), 'Translation grading parity');
}
for (const [input, expected] of [[' 1,25 ',1.25], ['−2.5',-2.5], ['1/4',.25], ['1e-3',.001], ['.5',.5], ['',null], [' ',null], ['Infinity',null], ['1/0',null], ['0/0',null], ['1+2',null], ['2abc',null], ['1,2,3',null]]) checks(parseNumericAnswer(input),expected,input);
// Independent fixtures from the authored questions and their arithmetic.
const rl = (module, id) => banks.get(`rl-for-llm/en/module-${String(module).padStart(2,'0')}.json`).questions.find(q => q.id === id);
checks(gradeQuestion(rl(2,'P09'), ['3.377','1.19']), 'correct', 'Terminal/truncation fixture');
checks(gradeQuestion(rl(2,'P09'), ['1.19','3.377']), 'incorrect', 'Numeric field order');
checks(gradeQuestion(rl(9,'P09'), [String(Math.sqrt(1.21*.81))]), 'correct', 'GSPO geometric mean');
checks(gradeQuestion(rl(12,'P03'), [String(2*32*8*128*16*8192*2/2**30)]), 'correct', 'KV-cache GiB');
checks(gradeQuestion(rl(12,'P08'), [String(64/28),String(64/28/4)]), 'correct', 'ESS arithmetic');
checks(gradeQuestion(rl(12,'P04'), ['A','B','C','D']), 'correct', 'Architecture matching');
checks(gradeQuestion(rl(5,'P10'), ['1','3','4','2']), 'correct', 'Dataset/method matching');
checks(parseUrl(new URL('https://vladkulikov.com/courses/information-theory-for-ml/module-16/tests/')).notFound, true, 'No invented reference-appendix test');
checks(counts.pages, 96); checks(counts.questions, 1886);
fs.mkdirSync('work', {recursive:true});
fs.writeFileSync('work/quiz-grading-audit.json', JSON.stringify({status:'PASS', counts},null,2));
console.log(JSON.stringify({status:'PASS',counts},null,2));
