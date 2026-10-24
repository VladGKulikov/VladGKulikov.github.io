import { useState, useSyncExternalStore } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { gradeQuestion, type Quiz, type QuizQuestion } from './quizzes';
import { progressModuleKey } from './module-progress';
import { readQuizProgress, writeQuizProgress, subscribeQuizProgress, serverQuizProgress, type QuestionProgress } from './quiz-progress';

export const questionAnchor = (q: QuizQuestion) => `question-${q.id}`;
export const questionLabel = (q: QuizQuestion, ru: boolean) => `${ru ? 'Вопрос' : 'Question'} ${q.sourceId ?? q.id}${q.title ? ` · ${q.title}` : ''}`;

export function QuizMarkdown({ children, inline = false }: { children: string; inline?: boolean }) {
  return <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}
    components={{ ...(inline ? {p: ({children}: {children?: React.ReactNode}) => <span>{children}</span>} : {}), table: ({ children }) => <div className="table-scroll"><table>{children}</table></div> }}>{children}</ReactMarkdown>;
}
const Markdown = QuizMarkdown;

export function QuizReader({ quiz, revision }: { quiz: Quiz; revision: string }) {
  const ru = quiz.language === 'ru';
  const key = `course-quiz:v1:${quiz.courseKey}:${progressModuleKey(quiz.courseKey, quiz.moduleKey)}:${revision}`;
  const progress = useSyncExternalStore(subscribeQuizProgress, () => readQuizProgress(key), serverQuizProgress);
  const [resetPending, setResetPending] = useState(false);
  const update = (id: string, next: QuestionProgress) => writeQuizProgress(key, { ...readQuizProgress(key), [id]: next });
  const automatic = quiz.questions.filter(q => q.type !== 'self-check');
  const checked = automatic.filter(q => progress[q.id]?.checked && gradeQuestion(q, progress[q.id].values) !== 'incomplete');
  const correct = checked.filter(q => gradeQuestion(q, progress[q.id].values) === 'correct').length;
  return <>
    <div className="quiz-intro">
      <p>{ru ? 'Выберите ответы и нажмите «Проверить». Для чисел подходят точка, запятая и простые дроби. Ответы и черновики сохраняются в этом браузере и общие для RU/EN.' : 'Choose your answers and select “Check”. Numbers accept a decimal point, comma, or simple fraction. Answers and drafts are saved in this browser and shared between RU/EN.'}</p>
      {automatic.length < quiz.questions.length && <p>{ru ? 'Открытые вопросы и задания с кодом проверяются самостоятельно по авторскому разбору. Код на сайте не запускается.' : 'Use the author’s worked answers to review open questions and code tasks yourself. Code does not run on this site.'}</p>}
      <noscript><p>{ru ? 'Для автоматической проверки включите JavaScript. Вопросы и раскрываемые ответы доступны ниже.' : 'Enable JavaScript for automatic checking. Questions and expandable answers are available below.'}</p></noscript>
      <div className="quiz-summary">
        <p role="status">{ru ? `Проверено: ${checked.length} / ${automatic.length} · Верно: ${correct}` : `Checked: ${checked.length} / ${automatic.length} · Correct: ${correct}`}</p>
        {resetPending ? <span className="quiz-reset-confirm js-only">
          <span>{ru ? 'Сбросить ответы этого модуля?' : 'Reset this module’s answers?'}</span>
          <button type="button" onClick={() => { writeQuizProgress(key, {}); setResetPending(false); }}>{ru ? 'Сбросить' : 'Reset'}</button>
          <button type="button" onClick={() => setResetPending(false)}>{ru ? 'Отмена' : 'Cancel'}</button>
        </span> : <button type="button" className="js-only" onClick={() => setResetPending(true)}>{ru ? 'Сбросить прогресс' : 'Reset progress'}</button>}
      </div>
    </div>
    <div className="lecture-prose quiz-prose" data-quiz="true">
      {quiz.questions.map(q => {
        const state = progress[q.id] ?? { values: [] };
        const grade = state.checked ? gradeQuestion(q, state.values) : null;
        const correct = grade === 'correct';
        const label = questionLabel(q, ru);
        const anchor = questionAnchor(q);
        const change = (values: string[]) => update(q.id, { ...state, values, checked: false, revealed: false });
        return <section key={q.id} className="quiz-question" data-question-id={q.id} id={anchor}>
          <h2 className="markdown-h2" id={`${anchor}-title`}><Markdown inline>{label}</Markdown></h2>
          {q.topic && <p className="quiz-note"><Markdown inline>{q.topic}</Markdown></p>}
          {q.note && <p className="quiz-note">{q.note}</p>}
          <div className="quiz-prompt"><Markdown>{q.prompt}</Markdown></div>
          <div className="quiz-controls js-only">
            {(q.type === 'single' || q.type === 'multiple') && <fieldset aria-labelledby={`${anchor}-title`}>
              <legend>{q.type === 'multiple' ? (ru ? 'Отметьте все верные варианты' : 'Select all correct options') : (ru ? 'Один правильный вариант' : 'One correct option')}</legend>
              {q.options!.map(option => <label key={option.value} className={`quiz-option ${state.checked && grade !== 'incomplete' ? (q.correct!.includes(option.value) ? 'is-correct' : state.values.includes(option.value) ? 'is-incorrect' : '') : ''}`}>
                <input type={q.type === 'multiple' ? 'checkbox' : 'radio'} name={anchor} value={option.value}
                  checked={state.values.includes(option.value)} onChange={event => change(q.type === 'single' ? [option.value] : event.target.checked ? [...state.values, option.value] : state.values.filter(v => v !== option.value))} />
                <span className="quiz-option-value">{option.value}</span><span className="quiz-option-text"><Markdown inline>{option.text}</Markdown></span>
              </label>)}
            </fieldset>}
            {q.type === 'matching' && <fieldset className="quiz-matching" aria-labelledby={`${anchor}-title`}>
              <legend>{ru ? 'Сопоставьте каждый пункт с ответом' : 'Match each item to an answer'}</legend>
              {q.matchLabels!.map((label,i) => <label key={label}>
                <span>{label}</span><select value={state.values[i] ?? ''} onChange={event => change(q.matchLabels!.map((_,n) => n === i ? event.target.value : state.values[n] ?? ''))}>
                  <option value="">{ru ? 'Выберите…' : 'Choose…'}</option>
                  {q.options!.map(option => <option key={option.value} value={option.value}>{option.value} · {option.text}</option>)}
                </select>
              </label>)}
            </fieldset>}
            {q.type === 'numeric' && <div className="quiz-numeric-fields">{q.fields!.map((field, i) => <label key={i}>
              <span>{field.label}</span><input type="text" inputMode="decimal" autoComplete="off" spellCheck={false}
                aria-describedby={`${anchor}-tolerance-${i}`} value={state.values[i] ?? ''} onChange={event => {
                  const values = q.fields!.map((_, n) => n === i ? event.target.value : state.values[n] ?? ''); change(values);
                }} />
              <small id={`${anchor}-tolerance-${i}`}>{ru ? 'Абсолютный допуск' : 'Absolute tolerance'}: ±{field.tolerance}</small>
            </label>)}</div>}
            {q.type === 'self-check' && <label className="quiz-draft-label">{ru ? 'Ваш ответ или заметки' : 'Your answer or notes'}
              <textarea rows={4} value={state.draft ?? ''} onChange={event => update(q.id, { ...state, draft: event.target.value })} />
            </label>}
            {q.manualReview && <p className="quiz-note">{ru ? 'Автоматически проверяется только выбор или числовая часть. Объяснение сверьте с авторским разбором.' : 'Only your selection or numeric part is checked automatically. Compare your reasoning with the author’s worked answer.'}</p>}
            <div className="quiz-actions">
              {q.type !== 'self-check' && <button type="button" className="quiz-check" onClick={() => update(q.id, { ...state, checked: true })}>{ru ? 'Проверить' : 'Check'}</button>}
              {q.type === 'self-check' && <button type="button" onClick={() => update(q.id, { ...state, revealed: !state.revealed })}>{state.revealed ? (ru ? 'Скрыть разбор' : 'Hide worked answer') : (ru ? 'Сверить с разбором' : 'Compare with worked answer')}</button>}
              {(state.values.length > 0 || state.draft || state.revealed) && <button type="button" onClick={() => update(q.id, { values: [] })}>{ru ? 'Заново' : 'Try again'}</button>}
            </div>
            <p className={`quiz-feedback ${correct ? 'is-correct' : grade === 'incorrect' ? 'is-incorrect' : ''}`} role="status">
              {grade === 'incomplete' ? (ru ? 'Укажите ответ во всех полях. В числовых полях введите конечное число или дробь.' : 'Answer every field. Numeric fields require a finite number or fraction.') : grade === 'correct' ? (q.manualReview ? (ru ? 'Проверяемая часть верна. Объяснение сверьте с разбором.' : 'The checked part is correct. Review your reasoning below.') : (ru ? 'Верно.' : 'Correct.')) : grade === 'incorrect' ? (ru ? 'Пока неверно. Можно изменить ответ или открыть разбор.' : 'Not correct yet. Change your answer or open the worked answer.') : ''}
            </p>
            {q.type !== 'self-check' && state.checked && grade !== 'incomplete' && <details className="quiz-explanation" open={state.revealed ?? false} onToggle={event => {
              if (event.currentTarget.open !== Boolean(state.revealed)) update(q.id, { ...state, revealed: event.currentTarget.open });
            }}><summary>{ru ? 'Ответ и разбор' : 'Answer and explanation'}</summary><Markdown>{q.explanation}</Markdown></details>}
            {q.type === 'self-check' && state.revealed && <div className="quiz-explanation"><p className="quiz-note">{ru ? 'Самостоятельная проверка по авторскому ответу' : 'Self-check against the author’s answer'}</p><Markdown>{q.explanation}</Markdown></div>}
          </div>
          <noscript>
            {q.options && <ol className="quiz-static-options">{q.options.map(option => <li key={option.value}><Markdown>{`${option.value}) ${option.text}`}</Markdown></li>)}</ol>}
            <details className="quiz-explanation"><summary>{ru ? 'Ответ и разбор' : 'Answer and explanation'}</summary><Markdown>{q.explanation}</Markdown></details>
          </noscript>
        </section>;
      })}
    </div>
  </>;
}
