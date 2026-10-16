import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { markdownHeadings } from "../app/markdown-headings.js";

const project = fileURLToPath(new URL("../", import.meta.url));
const root = path.join(project, "public/content");
const catalog = JSON.parse(fs.readFileSync(path.join(project, "app/generated/content.json"), "utf8"));
const quizModules = JSON.parse(fs.readFileSync(path.join(project, 'app/generated/quizzes.json'), 'utf8')).modules;
const practiceModules = JSON.parse(fs.readFileSync(path.join(project, 'app/generated/practice.json'), 'utf8')).modules;
const coursePath = (language, course) => `/${language === 'ru' ? 'ru/' : ''}courses/${course}/`;
const siteRoutes = new Set(catalog.courses.flatMap(course => {
  const base = coursePath(course.language, course.course_key);
  return [base, ...course.modules.map(module => `${base}${module.module_key}/`)];
}));
for (const entry of quizModules) siteRoutes.add(`${coursePath(entry.language, entry.courseKey)}${entry.moduleKey}/tests/`);
for (const entry of practiceModules) siteRoutes.add(`${coursePath(entry.language, entry.courseKey)}${entry.moduleKey}/${entry.section}/`);
const errors = [];
const totals = { documents: 0, headings: 0, tables: 0, lists: 0, listItems: 0, images: 0, links: 0, formulas: 0, codeBlocks: 0 };
const courses = {};
const quizMode = process.argv.includes('--quizzes');
const practiceMode = process.argv.includes('--practice');
const codeFence = source => { const fence = '`'.repeat(Math.max(3, ...(source.match(/`+/g) ?? []).map(s => s.length + 1))); return `${fence}python\n${source}\n${fence}`; };
const documents = practiceMode ? JSON.parse(fs.readFileSync(path.join(project, 'app/generated/practice.json'), 'utf8')).modules.map(entry => {
  const page = JSON.parse(fs.readFileSync(path.join(project, 'public/practice', entry.file), 'utf8'));
  const source = [page.introduction ?? '', ...page.items.flatMap(item => page.section === 'exercises' ? [`## ${item.title}`, item.prompt, item.answer, ...(item.codeFiles ?? []).map(file => codeFence(file.source))] : [`## ${item.title}`, ...item.cells.map(cell => cell.type === 'markdown' ? cell.source : codeFence(cell.source))])].join('\n\n');
  return {content_file: entry.file, course_key: entry.courseKey, language: entry.language, source};
}) : quizMode ? JSON.parse(fs.readFileSync(path.join(project, 'app/generated/quizzes.json'), 'utf8')).modules.map(entry => {
  const quiz = JSON.parse(fs.readFileSync(path.join(project, 'public/quizzes', entry.file), 'utf8'));
  const source = quiz.questions.map(q => `## ${q.id} ${q.title ?? ''}\n\n${q.topic ?? ''}\n\n${q.prompt}\n\n${(q.options ?? []).map(o => `- ${o.text}`).join('\n')}\n\n${q.explanation}`).join('\n\n');
  return {content_file:entry.file, course_key:entry.courseKey, language:entry.language, source};
}) : catalog.documents;

for (const document of documents) {
  const file = path.join(root, document.content_file);
  const source = document.source ?? fs.readFileSync(file, "utf8");
  const headingIndex = markdownHeadings(source);
  const ids = new Set(headingIndex.map((heading) => heading.id));
  const lines = new Set(headingIndex.map((heading) => heading.line));
  const add = (reason, node, detail) => errors.push({
    file: document.content_file, line: node?.position?.start.line, reason, detail,
  });
  if (ids.size !== headingIndex.length) add("duplicate-heading-anchor");
  const raw = (node) => source.slice(node.position.start.offset, node.position.end.offset);
  const text = (node) => node.value ?? (node.children ?? []).map(text).join("");
  const walk = (node) => {
    switch (node.type) {
      case "html":
        add("unconverted-html", node, node.value);
        break;
      case "text": {
        const value = raw(node);
        if (/(?<!\\)(?:\*\*|__|~~)/.test(value)) add("literal-emphasis", node, node.value);
        if (/(?<!\\)\$/.test(value)) add("unparsed-math-delimiter", node, node.value);
        if (/!?\[[^\]\n]+\]\([^)\n]+\)/.test(value)) add("unparsed-link", node, node.value);
        if (/\\(?:frac|sum|begin|end|mathbb|mathbf|text|theta|alpha|beta|log|sqrt|underbrace|overbrace|left|right)\b/.test(value)) {
          add("unparsed-latex", node, node.value);
        }
        if (/\[VIZ\s*:/i.test(value)) add("editorial-directive", node, node.value);
        break;
      }
      case "heading": {
        totals.headings++;
        if (node.depth >= 2 && node.depth <= 4 && !lines.has(node.position.start.line)) {
          add("missing-heading-anchor", node, text(node));
        }
        if (/^(?:\d+(?:\.\d+)*\.?\s*)?(?:and sources|и источники)\s*$/i.test(text(node))) {
          add("truncated-heading", node, text(node));
        }
        break;
      }
      case "paragraph":
        if (/^\s*\|[^\n]*\|/m.test(raw(node))) add("unparsed-table", node, raw(node));
        if (/\n\s*\d{1,2}\.\s+/.test(raw(node))) add("unparsed-numbered-list", node, raw(node));
        break;
      case "list":
        totals.lists++;
        break;
      case "listItem":
        totals.listItems++;
        break;
      case "table": {
        totals.tables++;
        const widths = node.children.map((row) => row.children.length);
        if (widths.some((width) => width !== widths[0])) add("inconsistent-table-columns", node, widths);
        break;
      }
      case "image":
      case "link": {
        if (node.type === "image") totals.images++;
        else totals.links++;
        const url = node.url;
        if (url.startsWith("#")) {
          if (!ids.has(decodeURIComponent(url.slice(1)))) add("missing-internal-anchor", node, url);
        } else if (!/^https?:|^mailto:|^data:/.test(url)) {
          const local = decodeURIComponent(url.split(/[?#]/)[0]);
          if (node.type === 'link' && /^\/(?:ru\/)?courses\//.test(local)) {
            if (!siteRoutes.has(local)) add('missing-site-route', node, url);
            break;
          }
          const target = node.type === "image" || local.startsWith("/")
            ? path.join(project, "public", local)
            : path.resolve(path.dirname(file), local);
          if (!fs.existsSync(target)) add("missing-local-target", node, url);
        }
        break;
      }
      case "inlineMath":
      case "math":
        totals.formulas++;
        break;
      case "code": {
        totals.codeBlocks++;
        const value = raw(node);
        const fence = value.match(/^(?:>\s*)*\s*(`{3,}|~{3,})/m)?.[1];
        if (fence && !(new RegExp(`(?:^|\\n)[ >\\t]*${fence}[ \\t]*\\r?$`)).test(value)) {
          add("unclosed-code-fence", node, value.slice(0, 120));
        }
        break;
      }
    }
    for (const child of node.children ?? []) walk(child);
  };
  const audit = () => (tree) => walk(tree);
  try {
    const html = renderToStaticMarkup(createElement(ReactMarkdown, {
      remarkPlugins: [remarkGfm, remarkMath, audit],
      rehypePlugins: [rehypeKatex],
    }, source));
    if (html.includes("katex-error")) add("katex-error");
  } catch (error) {
    add("render-error", undefined, String(error));
  }
  totals.documents++;
  courses[document.course_key] ??= {};
  courses[document.course_key][document.language] = (courses[document.course_key][document.language] ?? 0) + 1;
}

const result = { status: errors.length ? "FAIL" : "PASS", totals, courses, errors };
const report = process.argv.indexOf("--report");
if (report !== -1) fs.writeFileSync(path.resolve(process.argv[report + 1]), JSON.stringify(result, null, 2) + "\n");
console.log(JSON.stringify(result, null, 2));
if (errors.length) process.exitCode = 1;
