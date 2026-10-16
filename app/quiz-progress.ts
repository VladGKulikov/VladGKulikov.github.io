export type QuestionProgress = { values: string[]; checked?: boolean; revealed?: boolean; draft?: string };
export type QuizProgress = Record<string, QuestionProgress>;
const empty: QuizProgress = Object.freeze({});
const cache = new Map<string, QuizProgress>();
const listeners = new Set<() => void>();
export const serverQuizProgress = () => empty;

export function readQuizProgress(key: string): QuizProgress {
  if (cache.has(key)) return cache.get(key)!;
  let value: QuizProgress = empty;
  try {
    const stored: unknown = JSON.parse(localStorage.getItem(key) ?? '{}');
    if (stored && typeof stored === 'object' && !Array.isArray(stored)) {
      value = Object.fromEntries(Object.entries(stored).filter(([, item]) => item && typeof item === 'object' &&
        Array.isArray(item.values) && item.values.every((v: unknown) => typeof v === 'string') &&
        (!item.draft || typeof item.draft === 'string')));
    }
  } catch { /* Private browsing, storage denied, or an obsolete/corrupt draft. */ }
  cache.set(key, value);
  return value;
}

export function writeQuizProgress(key: string, value: QuizProgress) {
  cache.set(key, value);
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* Keep working in memory if storage is unavailable. */ }
  listeners.forEach(listener => listener());
}

export function subscribeQuizProgress(listener: () => void) {
  listeners.add(listener);
  const onStorage = (event: StorageEvent) => {
    if (event.key?.startsWith('course-quiz:') || event.key?.startsWith('course-practice:')) cache.delete(event.key);
    else if (event.key === null) cache.clear();
    listener();
  };
  window.addEventListener('storage', onStorage);
  return () => { listeners.delete(listener); window.removeEventListener('storage', onStorage); };
}
