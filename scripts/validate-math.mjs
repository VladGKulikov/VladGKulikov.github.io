import { readFile, readdir } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

const root = fileURLToPath(new URL("../public/content/", import.meta.url));

async function markdownFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(
    entries.map((entry) => {
      const path = join(directory, entry.name);
      return entry.isDirectory() ? markdownFiles(path) : path.endsWith(".md") ? [path] : [];
    }),
  );
  return nested.flat();
}

const files = await markdownFiles(root);
const errors = [];
let renderedFormulas = 0;

for (const file of files) {
  try {
    const markdown = await readFile(file, "utf8");
    const html = renderToStaticMarkup(
      createElement(ReactMarkdown, {
        remarkPlugins: [remarkGfm, remarkMath],
        rehypePlugins: [rehypeKatex],
        children: markdown,
      }),
    );
    renderedFormulas += (html.match(/class="katex(?:-display)?"/g) ?? []).length;
    if (html.includes("katex-error")) {
      errors.push({ file, reason: "katex-error" });
    }
  } catch (error) {
    errors.push({ file, reason: error instanceof Error ? error.message : String(error) });
  }
}

const result = {
  status: errors.length ? "FAIL" : "PASS",
  documents: files.length,
  renderedFormulas,
  errors,
};

console.log(JSON.stringify(result, null, 2));
if (errors.length) process.exitCode = 1;
