import { useState } from 'react';

const tokens = /#[^\n]*|(?:[rbufRBUF]{0,2})(?:"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|\b(?:False|None|True|and|as|assert|async|await|break|class|continue|def|del|elif|else|except|finally|for|from|global|if|import|in|is|lambda|nonlocal|not|or|pass|raise|return|try|while|with|yield)\b|\b(?:abs|all|any|bool|dict|enumerate|float|int|len|list|map|max|min|open|print|range|round|set|sorted|str|sum|tuple|zip)\b|\b(?:0[xX][0-9a-fA-F]+|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\b/g;
const builtins = /^(?:abs|all|any|bool|dict|enumerate|float|int|len|list|map|max|min|open|print|range|round|set|sorted|str|sum|tuple|zip)$/;

export function highlightPython(source: string) {
  const pieces: { text: string; kind?: string }[] = [];
  let offset = 0;
  for (const match of source.matchAll(tokens)) {
    if (match.index! > offset) pieces.push({text: source.slice(offset, match.index)});
    const value = match[0];
    const kind = value.startsWith('#') ? 'comment' : /^[rbufRBUF]*["']/.test(value) ? 'string' : /^\d/.test(value) ? 'number' : builtins.test(value) ? 'builtin' : 'keyword';
    pieces.push({text: value, kind}); offset = match.index! + value.length;
  }
  if (offset < source.length) pieces.push({text: source.slice(offset)});
  return pieces;
}

export function CodeBlock({ source, language = 'python', filename, download, ru = false }: { source: string; language?: string; filename?: string; download?: string; ru?: boolean }) {
  const [copied, setCopied] = useState(false);
  const [copyFailed, setCopyFailed] = useState(false);
  const python = /^(?:py|python|ipython)$/.test(language);
  const parts: ReturnType<typeof highlightPython> = python ? highlightPython(source) : [{text: source}];
  return <div className="code-block">
    <div className="code-header">
      <span className="code-file-label"><span className="code-language">{python ? 'Python' : language || 'Code'}</span>{filename && <span>{filename}</span>}</span>
      <span className="code-file-actions">
        <button className="js-only" type="button" onClick={async () => {
          try { await navigator.clipboard.writeText(source); setCopied(true); setCopyFailed(false); }
          catch { setCopyFailed(true); }
        }}>{copied ? (ru ? 'Скопировано' : 'Copied') : (ru ? 'Копировать' : 'Copy')}</button>
        {download && <a href={download} download>{ru ? 'Скачать .py' : 'Download .py'}</a>}
      </span>
    </div>
    {copyFailed && <p className="code-copy-status" role="status">{ru ? 'Выделите код и скопируйте его или скачайте файл.' : 'Select and copy the code, or download the file.'}</p>}
    {/* Keyboard focus lets readers scroll long code without a pointer. */}
    {/* eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex */}
    <div className="code-scroll" tabIndex={0} role="region" aria-label={filename ?? (ru ? 'Код' : 'Code')}>
      <div className="code-line-layout">
        <div className="code-line-numbers" aria-hidden="true">{source.split('\n').map((_, i) => <span key={i}>{i + 1}</span>)}</div>
        <pre><code>{parts.map((part, i) => part.kind ? <span key={i} className={`code-token-${part.kind}`}>{part.text}</span> : part.text)}</code></pre>
      </div>
    </div>
  </div>;
}
