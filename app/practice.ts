import catalog from './generated/practice.json';

export type PracticeSection = 'exercises' | 'notebooks';
export type CodeFile = { id: string; filename: string; download: string; bytes: number; sha256: string; source: string; language: string };
export type Exercise = { id: string; title: string; prompt: string; answer: string; answerKind: 'solution' | 'guidance'; codeFiles?: CodeFile[] };
export type Notebook = { id: string; title: string; kind: 'main' | 'project' | 'capstone'; format: 'ipynb'; filename: string; download: string; bytes: number; sha256: string; cells: {id: string; type: string; source: string}[] };
export type Practice = { courseKey: string; moduleKey: string; language: string; section: PracticeSection; introduction?: string; items: Exercise[] | Notebook[]; bundle: {url: string; bytes: number; sha256: string} };
export const practiceModules = catalog.modules;
export const practiceFor = (course: string | null, module: string | null, language: string, section?: string) =>
  practiceModules.find(p => p.courseKey === course && p.moduleKey === module && p.language === language && p.section === section);
export const practiceAnchor = (section: PracticeSection, id: string) => `${section === 'exercises' ? 'exercise' : 'notebook'}-${id}`;
