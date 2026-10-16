import catalog from './generated/quizzes.json';

export type QuizQuestion = {
  id: string;
  sourceId?: string;
  title?: string;
  note?: string;
  topic?: string;
  prompt: string;
  explanation: string;
  type: 'single' | 'multiple' | 'numeric' | 'matching' | 'self-check';
  options?: { value: string; text: string }[];
  correct?: string[];
  matchLabels?: string[];
  fields?: { label: string; expected: number; tolerance: number }[];
  manualReview?: boolean;
};
export type Quiz = { courseKey: string; moduleKey: string; language: string; questions: QuizQuestion[] };
export const quizModules = catalog.modules;
export const quizFor = (course: string | null, module: string | null, language: string) =>
  quizModules.find(q => q.courseKey === course && q.moduleKey === module && q.language === language);

export function parseNumericAnswer(value: string): number | null {
  const normalized = value.trim().replaceAll('\u2212', '-').replaceAll(',', '.');
  const number = '[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?';
  if (!new RegExp(`^${number}(?:\\s*/\\s*${number})?$`).test(normalized)) return null;
  const parts = normalized.split('/').map(Number);
  const result = parts.length === 2 ? parts[0] / parts[1] : parts[0];
  return Number.isFinite(result) ? result : null;
}

export function gradeQuestion(question: QuizQuestion, values: string[]): 'correct' | 'incorrect' | 'incomplete' | 'self-check' {
  if (question.type === 'self-check') return 'self-check';
  if (question.type === 'matching') {
    if (values.length !== question.correct!.length || values.some(v => !v)) return 'incomplete';
    return question.correct!.every((value, i) => values[i] === value) ? 'correct' : 'incorrect';
  }
  if (question.type === 'numeric') {
    const parsed = values.map(parseNumericAnswer);
    if (parsed.length !== question.fields!.length || parsed.some(n => n === null)) return 'incomplete';
    return question.fields!.every((field, i) => {
      const roundoff = Number.EPSILON * Math.max(1, Math.abs(field.expected), Math.abs(parsed[i]!)) * 8;
      return Math.abs(parsed[i]! - field.expected) <= field.tolerance + roundoff;
    }) ? 'correct' : 'incorrect';
  }
  if (!values.length) return 'incomplete';
  const unique = new Set(values);
  return unique.size === values.length && values.length === question.correct!.length &&
    question.correct!.every(value => unique.has(value)) ? 'correct' : 'incorrect';
}
