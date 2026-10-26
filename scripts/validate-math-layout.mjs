import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const pluginRequire = createRequire(require.resolve('rehype-katex'));
const katexVersion = require('katex/package.json').version;
assert.equal(pluginRequire('katex/package.json').version, katexVersion,
  'KaTeX renderer and stylesheet must use the same version');

const equations = JSON.parse(fs.readFileSync('app/generated/module-20-equations.json', 'utf8'));
const counts = {};
for (const language of ['ru', 'en']) {
  const markdown = fs.readFileSync(`public/content/modern-llms/${language}/module-20.md`, 'utf8').replaceAll('\r\n', '\n');
  const tex = Array.from(markdown.matchAll(/^\$\$\n([\s\S]*?)\n\$\$/gm), match => match[1]);
  assert.deepEqual(Object.keys(equations[language]), tex, `Stale MathML export: ${language}`);
  const prefix = language === 'ru' ? 'ru/' : '';
  const html = fs.readFileSync(`dist-pages/${prefix}courses/modern-llms/module-20/index.html`, 'utf8');
  assert.equal((html.match(/class="reference-equation"/g) ?? []).length, tex.length);
  assert.ok(!html.includes('class="katex-display"'), 'M20 display equations must use the reference MathML');
  for (const [formula, markup] of Object.entries(equations[language])) {
    assert.ok(html.includes(markup), `Reference MathML changed: ${language}: ${formula}`);
  }
  counts[language] = { equations: tex.length, boxedEquations: tex.filter(formula => formula.startsWith('\\boxed')).length };
}
console.log(JSON.stringify({ status: 'PASS', katexVersion, referenceMathML: counts }, null, 2));
