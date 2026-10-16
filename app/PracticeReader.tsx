/* eslint-disable @next/next/no-img-element -- Portable static course material. */
import { isValidElement, useSyncExternalStore, type ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { CodeBlock } from './CodeBlock';
import { buildLocation, type CourseKey, type Language } from './site';
import { practiceAnchor, type Practice, type Exercise, type Notebook } from './practice';
import { readQuizProgress, writeQuizProgress, subscribeQuizProgress, serverQuizProgress } from './quiz-progress';

export function PracticeMarkdown({children, ru, inline = false}: {children: string; ru: boolean; inline?: boolean}) {
  return <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]} components={{
    ...(inline ? {p: ({children}: {children?: ReactNode}) => <span>{children}</span>} : {}),
    h1: ({children}) => <h3 className="markdown-h3">{children}</h3>,
    h2: ({children}) => <h3 className="markdown-h3">{children}</h3>,
    h3: ({children}) => <h3 className="markdown-h3">{children}</h3>,
    h4: ({children}) => <h3 className="markdown-h4">{children}</h3>,
    h5: ({children}) => <h3 className="markdown-h4">{children}</h3>,
    h6: ({children}) => <h3 className="markdown-h4">{children}</h3>,
    table: ({children}) => <div className="table-scroll"><table>{children}</table></div>,
    a: ({href,children}) => <a href={href} download={href?.startsWith('/downloads/') || undefined} target={/^https?:/.test(href ?? '') ? '_blank' : undefined} rel={/^https?:/.test(href ?? '') ? 'noreferrer' : undefined}>{children}</a>,
    img: ({src,alt}) => <img src={src} alt={alt || (ru ? 'Иллюстрация' : 'Illustration')} loading="lazy" />,
    pre: ({children}) => {
      if (!isValidElement<{children?: ReactNode; className?: string}>(children)) return <pre>{children}</pre>;
      return <CodeBlock source={String(children.props.children ?? '').replace(/\n$/, '')} language={children.props.className?.replace(/^language-/, '') ?? 'text'} ru={ru} />;
    },
  }}>{children}</ReactMarkdown>;
}

const size = (bytes: number, ru: boolean) => `${(bytes / 1024).toFixed(1).replace('.', ru ? ',' : '.')} ${ru ? 'КБ' : 'KB'}`;
export function PracticeReader({ page, revision }: { page: Practice; revision: string }) {
  const ru = page.language === 'ru';
  const key = `course-practice:v1:${page.courseKey}:${page.moduleKey}:${page.section}:${revision}`;
  const progress = useSyncExternalStore(subscribeQuizProgress, () => readQuizProgress(key), serverQuizProgress);
  const toggle = (id: string, open: boolean) => {
    if (Boolean(readQuizProgress(key)[id]?.revealed) !== open) writeQuizProgress(key, {...readQuizProgress(key), [id]: {values: [], revealed: open}});
  };
  return <>
    <div className="practice-intro">
      <p>{page.section === 'exercises' ? (ru ? 'Сначала попробуйте решить упражнение. Ответы, подробные решения и проектные ориентиры раскрываются отдельно и сохраняют своё состояние при переключении RU/EN.' : 'Try each exercise first. Expand worked answers or project guidance when needed; their open state is shared between RU/EN.') : (ru ? 'Здесь можно прочитать пояснения и код имеющихся ноутбуков или скачать файлы. Код на сайте не запускается.' : 'Read the explanations and code in the existing notebooks, or download the files. Code does not run on this site.')}</p>
      {page.introduction && <PracticeMarkdown ru={ru}>{page.introduction.replace(/^#{1,6}\s+(.+)$/gm,'**$1**')}</PracticeMarkdown>}
      <a className="practice-bundle" href={page.bundle.url} download>{ru ? 'Скачать файлы курса (.zip)' : 'Download course files (.zip)'} <small>{size(page.bundle.bytes, ru)}</small></a>
      <p className="practice-note">{page.courseKey === 'information-theory-for-ml' ? (ru ? 'Архив содержит девять Python-файлов решений, общих для RU/EN, с сохранением структуры каталогов.' : 'The archive preserves the folder structure of nine Python solution files shared by RU/EN.') : (ru ? 'Архив содержит файлы на обоих языках и вспомогательные материалы с сохранением структуры каталогов.' : 'The archive preserves the source folder structure, both languages, and supporting files.')}</p>
    </div>
    <div className="lecture-prose practice-prose" data-practice="true">
      {page.section === 'exercises' ? (page.items as Exercise[]).map(item => <section key={item.id} className="practice-card" data-reading-key={`exercise:${item.id}`} id={practiceAnchor('exercises',item.id)}>
        <h2 className="markdown-h2" data-reading-block="true"><PracticeMarkdown ru={ru} inline>{item.title}</PracticeMarkdown></h2>
        <div className="practice-prompt" data-reading-container="true"><PracticeMarkdown ru={ru}>{item.prompt}</PracticeMarkdown></div>
        <details className="practice-answer" data-disclosure={item.id} open={progress[item.id]?.revealed ?? false} onToggle={event => toggle(item.id,event.currentTarget.open)}>
          <summary data-reading-block="true">{item.answerKind === 'guidance' ? (ru ? 'Демо и критерии проекта' : 'Project demos and guidance') : (ru ? 'Показать ответ и решение' : 'Show answer and solution')}</summary>
          <div className="practice-answer-body" data-reading-container="true">
            {item.answer ? <PracticeMarkdown ru={ru}>{item.answer}</PracticeMarkdown> : <p>{ru ? 'Это проектное задание для самостоятельной работы. Используйте демонстрационные ноутбуки модуля как ориентир.' : 'This is an independent project task. Use the module’s demonstration notebooks as guidance.'}</p>}
            {!item.answer && <p><a href={buildLocation(page.language as Language,page.courseKey as CourseKey,page.moduleKey,'notebooks')}>{ru ? 'Ноутбуки модуля' : 'Module notebooks'}</a></p>}
          </div>
          {item.codeFiles?.map(file => <div className="practice-code-solution" key={file.id} data-reading-block="true">
            <p className="practice-note">{ru ? 'Программная часть решения · один исходный файл для RU/EN' : 'Code solution · the same source file for RU/EN'}</p>
            <CodeBlock source={file.source} language={file.language} filename={file.filename} download={file.download} ru={ru} />
          </div>)}
        </details>
      </section>) : (page.items as Notebook[]).map(book => <section key={book.id} className="practice-card notebook-card" data-reading-key={`notebook:${book.id}`} id={practiceAnchor('notebooks',book.id)}>
        <h2 className="markdown-h2" data-reading-block="true"><PracticeMarkdown ru={ru} inline>{book.title}</PracticeMarkdown></h2>
        <p className="practice-note" data-reading-block="true">{book.kind === 'project' ? (ru ? 'Проектная демонстрация' : 'Project demonstration') : book.kind === 'capstone' ? 'MiniLLM · Capstone' : (ru ? 'Основной ноутбук модуля' : 'Main module notebook')}</p>
        <p className="notebook-download" data-reading-block="true"><a href={book.download} download>{ru ? 'Скачать .ipynb' : 'Download .ipynb'}</a> <small>{book.filename} · {size(book.bytes, ru)}</small></p>
        <details className="notebook-preview" data-disclosure={book.id} open={progress[book.id]?.revealed ?? false} onToggle={event => toggle(book.id,event.currentTarget.open)}>
          <summary data-reading-block="true">{ru ? 'Показать текст ноутбука' : 'Show notebook text'}</summary>
          {book.cells.map(cell => <div key={cell.id} className={`notebook-cell notebook-cell-${cell.type}`} data-reading-key={`notebook:${book.id}:cell:${cell.id}`} data-reading-container="true">
            {cell.type === 'code' ? <CodeBlock source={cell.source} language="python" ru={ru} /> : cell.type === 'markdown' ? <PracticeMarkdown ru={ru}>{cell.source}</PracticeMarkdown> : <pre>{cell.source}</pre>}
          </div>)}
        </details>
      </section>)}
    </div>
  </>;
}
