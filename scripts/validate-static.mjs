import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { parse } from "parse5";
import { parseUrl, buildLocation, practiceModules } from "../build-static/pages-render.js";

const origin = "https://vladkulikov.com";
const manifest = JSON.parse(fs.readFileSync("dist-pages/route-manifest.json", "utf8"));
const catalog = JSON.parse(fs.readFileSync("app/generated/content.json", "utf8"));
const pages = new Map(manifest.map(page => [page.path, page]));
const titles = new Set(), descriptions = new Set();
const errors = [];
const counts = { pages: 0, lectures: 0, quizzes: 0, exercises: 0, notebooks: 0, links: 0, images: 0, formulas: 0 };
const attr = (node, name) => node.attrs?.find(a => a.name === name)?.value;
const hasClass = (node, value) => (attr(node, "class") ?? "").split(" ").includes(value);
function flatten(node) { return [node, ...(node.childNodes ?? []).flatMap(flatten)]; }
const text = node => node.nodeName === "#text" ? node.value : (node.childNodes ?? []).map(text).join("");
function check(condition, page, message) { if (!condition) errors.push(`${page}: ${message}`); }

for (const page of manifest) {
  const label = page.path;
  const html = fs.readFileSync(path.join("dist-pages", label, "index.html"), "utf8");
  const nodes = flatten(parse(html));
  const head = nodes.find(node => node.tagName === "head");
  const headNodes = flatten(head);
  check(attr(nodes.find(node => node.tagName === "html"), "lang") === page.language, label, "Wrong HTML language");
  check(nodes.filter(node => node.tagName === "h1").length === 1, label, "Expected one h1");
  let headingDepth = 0;
  for (const node of nodes.filter(node => /^h[1-6]$/.test(node.tagName))) {
    const depth = Number(node.tagName[1]);
    check(depth <= headingDepth + 1, label, "Skipped document heading level");
    headingDepth = depth;
  }
  const title = text(headNodes.find(node => node.tagName === "title"));
  const description = attr(headNodes.find(node => attr(node, "name") === "description"), "content");
  check(title === page.title && !titles.has(title), label, "Missing, wrong or duplicate title");
  check(description === page.description && !descriptions.has(description), label, "Missing, wrong or duplicate description");
  titles.add(title); descriptions.add(description);
  const canonical = headNodes.filter(node => attr(node, "rel") === "canonical");
  check(canonical.length === 1 && attr(canonical[0], "href") === origin + label, label, "Wrong canonical");
  check(!headNodes.some(node => Object.values(Object.fromEntries((node.attrs ?? []).map(a => [a.name, a.value]))).some(value => /localhost|127\.0\.0\.1|vladgkulikov\.github\.io|chatgpt\.site/.test(value))), label, "Non-production metadata URL");
  for (const language of ["en", "ru"]) {
    const alternate = headNodes.find(node => attr(node, "hreflang") === language);
    const expected = buildLocation(language, page.courseKey, page.moduleKey, page.section);
    check(attr(alternate, "href") === origin + expected && pages.has(expected), label, "Wrong reciprocal hreflang");
  }
  for (const field of ["og:title", "og:description", "og:url", "og:image", "twitter:card", "twitter:title", "twitter:description", "twitter:image"]) {
    check(headNodes.some(node => (attr(node, "property") ?? attr(node, "name")) === field && attr(node, "content")), label, `Missing ${field}`);
  }
  const initial = nodes.find(node => attr(node, "id") === "page-data");
  const props = JSON.parse(text(initial));
  if (page.moduleKey) {
    if (page.section === 'tests') counts.quizzes++; else if (page.section) counts[page.section]++; else counts.lectures++;
    const prose = nodes.find(node => hasClass(node, "lecture-prose"));
    check(Boolean(prose) && text(prose).length > 1000, label, "Reading content missing from original HTML");
    check(props.initialMarkdown === fs.readFileSync(path.join(page.section === 'tests' ? "public/quizzes" : page.section ? "public/practice" : "public/content", page.contentFile), "utf8"), label, "Published source changed");
    let previous = 1;
    for (const node of flatten(prose).filter(node => /^h[2-6]$/.test(node.tagName))) {
      const depth = Number(node.tagName[1]);
      check(depth <= previous + 1, label, "Skipped heading level");
      previous = depth;
    }
    check(!html.includes("katex-error"), label, "KaTeX error");
  }
  const ids = nodes.map(node => attr(node, "id")).filter(Boolean);
  check(new Set(ids).size === ids.length, label, "Duplicate HTML IDs");
  for (const node of nodes) {
    if (hasClass(node, "katex") || hasClass(node, "reference-equation")) counts.formulas++;
    if (node.tagName === "img") {
      counts.images++;
      const src = attr(node, "src");
      check(Boolean(attr(node, "alt")), label, "Image missing alt text");
      check(src?.startsWith("/") && fs.existsSync(path.join("dist-pages", src)), label, `Missing image ${src}`);
    }
    if (node.tagName !== "a") continue;
    counts.links++;
    const href = attr(node, "href");
    check(Boolean(href), label, "Navigation anchor without href");
    const url = new URL(href, origin + label);
    if (url.origin !== origin) continue;
    check(!url.search, label, `Query selection remains in internal link ${href}`);
    if (url.pathname.startsWith("/downloads/")) {
      check(fs.existsSync(path.join("dist-pages", decodeURIComponent(url.pathname))), label, `Missing download ${href}`);
      check(attr(node, "download") !== undefined, label, `File link missing download attribute ${href}`);
      continue;
    }
    if (["/CITATION.cff", "/PUBLICATION.json"].includes(url.pathname)) {
      const name = url.pathname.slice(1);
      check(fs.existsSync(path.join("dist-pages", name)) && fs.readFileSync(path.join("dist-pages", name), "utf8") === fs.readFileSync(name, "utf8"), label, `Missing or stale publication metadata ${href}`);
      continue;
    }
    if (url.pathname.endsWith(".md")) continue;
    check(pages.has(url.pathname), label, `Missing internal page ${href}`);
    if (url.hash && url.pathname === label) check(ids.includes(decodeURIComponent(url.hash.slice(1))), label, `Missing anchor ${href}`);
  }
  counts.pages++;
}
check(manifest.length === 202 + practiceModules.length && counts.lectures === 98 && counts.quizzes === 96 && counts.exercises === 94 && counts.notebooks === 66, "catalog", "Expected 362 pages: 98 lectures, 96 quizzes, 94 exercise pages and 66 notebook pages");
const sitemap = fs.readFileSync("dist-pages/sitemap.xml", "utf8");
check((sitemap.match(/<loc>/g) ?? []).length === manifest.length, "sitemap", "Wrong URL count");
for (const page of manifest) check(sitemap.includes(`<loc>${origin}${page.path}</loc>`), "sitemap", `Missing ${page.path}`);
check(!/<loc>[^<]*(?:\?|404|github\.io|localhost)/.test(sitemap), "sitemap", "Technical or noncanonical URL");
const errorPage = fs.readFileSync("dist-pages/404.html", "utf8");
check(errorPage.includes('name="robots" content="noindex"') && errorPage.includes('href="/ru/"'), "404", "Missing noindex or library links");
for (const file of ["robots.txt", "llms.txt", "agents.txt"]) check(fs.readFileSync(`dist-pages/${file}`, "utf8").includes(origin), file, "Missing canonical domain");
for (const course of catalog.courses) for (const lecture of course.modules) {
  const legacy = new URL(`/${course.language === "ru" ? "ru/" : ""}?course=${course.course_key}&module=${lecture.module_key}`, origin);
  const parsed = parseUrl(legacy);
  check(parsed.courseKey === course.course_key && parsed.moduleKey === lecture.module_key && parsed.language === course.language, "legacy", legacy.href);
  const source = fs.readFileSync(`public/content/${lecture.content_file}`, "utf8").replaceAll("\r\n", "\n");
  check(createHash("sha256").update(source).digest("hex") === lecture.content_hash, lecture.content_file, "Content hash changed");
}
assert.equal(parseUrl(new URL("/ru/?lang=en&course=modern-llms&module=module-18", origin)).language, "en");
assert.equal(parseUrl(new URL("/missing/", origin)).notFound, true);
const report = { status: errors.length ? "FAIL" : "PASS", counts, errors };
fs.mkdirSync("work", { recursive: true });
fs.writeFileSync("work/static-audit.json", JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
if (errors.length) process.exitCode = 1;
