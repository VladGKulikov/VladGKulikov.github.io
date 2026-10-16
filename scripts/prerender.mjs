import fs from "node:fs";
import path from "node:path";
import { routes, courses, buildLocation, pageMetadata, renderPage, quizModules, practiceModules } from "../build-static/pages-render.js";

const config = JSON.parse(fs.readFileSync("SITE_CONFIG.json", "utf8"));
const publication = JSON.parse(fs.readFileSync("PUBLICATION.json", "utf8"));
const output = "dist-pages";
const shell = fs.readFileSync(`${output}/index.html`, "utf8");
const assets = shell.match(/<(?:script|link)\b[^>]*(?:src|href)="\/assets\/[^>]+>(?:<\/script>)?/g)?.join("\n") ?? "";
if (!assets.includes(".js") || !assets.includes(".css")) throw new Error("Missing compiled reader assets");
const escape = value => String(value).replaceAll("&", "&amp;").replaceAll('"', "&quot;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
const json = value => JSON.stringify(value).replaceAll("<", "\\u003c").replaceAll("\u2028", "\\u2028").replaceAll("\u2029", "\\u2029");
const manifest = [];

function writePage(route) {
  const course = courses.find(c => c.language === route.language && c.course_key === route.courseKey);
  const lecture = course?.modules.find(m => m.module_key === route.moduleKey);
  const quiz = route.section === 'tests' ? quizModules.find(q => q.courseKey === route.courseKey && q.moduleKey === route.moduleKey && q.language === route.language) : null;
  const practice = practiceModules.find(p => p.courseKey === route.courseKey && p.moduleKey === route.moduleKey && p.language === route.language && p.section === route.section);
  const markdown = practice ? fs.readFileSync(path.join('public/practice', practice.file), 'utf8') : quiz ? fs.readFileSync(path.join('public/quizzes', quiz.file), 'utf8') : lecture ? fs.readFileSync(path.join("public/content", lecture.content_file), "utf8") : "";
  const meta = pageMetadata(route);
  const head = [
    '<meta charset="UTF-8">', '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
    `<title>${escape(meta.title)}</title>`, `<meta name="description" content="${escape(meta.description)}">`,
    `<meta name="author" content="Vlad Kulikov">`,
    '<noscript><style>.desktop-left-toggle,.desktop-right-toggle,.mobile-menu,.mobile-right-toggle,.lecture-search,.drawer-close,.js-only{display:none!important}</style></noscript>',
    ...(!route.notFound ? [
      `<link rel="canonical" href="${meta.canonical}">`,
      `<link rel="alternate" hreflang="en" href="${meta.en}">`,
      `<link rel="alternate" hreflang="ru" href="${meta.ru}">`,
      `<link rel="alternate" hreflang="x-default" href="${meta.en}">`,
    ] : ['<meta name="robots" content="noindex">']),
    ...["title", "description", "image"].flatMap(key => [
      `<meta property="og:${key}" content="${escape(meta[key])}">`,
      `<meta name="twitter:${key}" content="${escape(meta[key])}">`,
    ]),
    '<meta name="twitter:card" content="summary_large_image">', '<meta property="og:type" content="website">',
    `<meta property="og:url" content="${meta.canonical}">`,
    `<meta property="og:locale" content="${route.language === "ru" ? "ru_RU" : "en_US"}">`,
    `<meta property="og:locale:alternate" content="${route.language === "ru" ? "en_US" : "ru_RU"}">`,
    '<link rel="alternate" type="text/plain" href="/llms.txt" title="Library guide for language models">',
    ...Object.entries(config.verification).filter(([, value]) => value).map(([provider, value]) =>
      `<meta name="${provider === "google" ? "google-site-verification" : "yandex-verification"}" content="${escape(value)}">`),
    assets,
  ];
  const html = `<!doctype html>\n<html lang="${route.language}"><head>\n${head.join("\n")}\n</head><body><div id="root">${renderPage(route, markdown)}</div><script id="page-data" type="application/json">${json({ initialPage: route, initialMarkdown: markdown })}</script></body></html>\n`;
  const file = route.notFound ? `${output}/404.html` : path.join(output, meta.path, "index.html");
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, html);
  if (!route.notFound) manifest.push({ ...route, path: meta.path, title: meta.title, description: meta.description, contentFile: practice?.file ?? quiz?.file ?? lecture?.content_file ?? null });
}
for (const route of routes) writePage(route);
writePage({ language: "en", courseKey: null, moduleKey: null, notFound: true });
fs.writeFileSync(`${output}/sitemap.xml`, `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${manifest.map(page => `<url><loc>${config.origin}${page.path}</loc></url>`).join("\n")}\n</urlset>\n`);
fs.writeFileSync(`${output}/robots.txt`, `User-agent: *\nAllow: /\n\nSitemap: ${config.origin}/sitemap.xml\n`);
fs.writeFileSync(`${output}/.nojekyll`, "");
fs.writeFileSync(`${output}/route-manifest.json`, JSON.stringify(manifest, null, 2));
for (const name of ["CITATION.cff", "PUBLICATION.json"]) fs.copyFileSync(name, `${output}/${name}`);

const repo = "https://github.com/VladGKulikov/VladGKulikov.github.io";
const license = `${repo}/blob/main/LICENSE-CONTENT.md`;
const guides = [
  "# Vlad Kulikov Open Courses", "",
  "> A bilingual, public AI/ML lecture library by Vlad Kulikov: Modern LLMs, Reinforcement Learning for LLM, and Information Theory for Machine Learning.", "",
  `Canonical website: ${config.origin}/. English home: /. Russian home: /ru/.`,
  `The live lecture text was updated on ${publication.content_updated}. There are 49 modules in each language (98 lecture pages).`,
  "English and Russian versions cover corresponding modules. Module keys are stable identifiers; use the visible reading order rather than sorting module keys numerically.",
  "HTML pages contain complete lectures, formulas, illustrations, and navigation without requiring JavaScript. Public Markdown source files are linked below as optional machine-readable sources.",
  "This website publishes lectures, illustrations and module quizzes with answer checking. Choice and numeric questions are checked locally in the browser; open questions and code tasks have author-written answers for self-checking. Code is not executed. Progress is local to the browser and shared across the two languages. Worked exercises and existing notebooks are also published: answers and notebook text expand on request, original files download without code execution. Information Theory Python examples form part of the corresponding exercise answers. There are 469 exercise/project items and 55 notebooks per language, plus 9 shared Python solution files.",
  "",
  "## Library", `- [English library](${config.origin}/): Author profile and course catalog.`,
  `- [Библиотека на русском](${config.origin}/ru/): Автор и каталог курсов.`,
  `- [Sitemap](${config.origin}/sitemap.xml): All canonical library, course, lecture, quiz, exercise and notebook pages.`,
];
for (const course of courses) {
  guides.push("", `## ${course.title} (${course.language.toUpperCase()})`,
    `- [Course overview](${config.origin}${buildLocation(course.language, course.course_key)}): ${course.description}`,
    ...course.modules.flatMap(module => {
      const quiz = quizModules.find(q => q.courseKey === course.course_key && q.moduleKey === module.module_key && q.language === course.language);
      const practice = practiceModules.filter(p => p.courseKey === course.course_key && p.moduleKey === module.module_key && p.language === course.language);
      return [`- [${module.title}](${config.origin}${buildLocation(course.language, course.course_key, module.module_key)}): Lecture ${String(module.position).padStart(2, "0")}; source: ${config.origin}/content/${module.content_file}`,
        ...(quiz ? [`- [${module.title}: quizzes](${config.origin}${buildLocation(course.language, course.course_key, module.module_key, 'tests')}): ${quiz.count} authored questions; structured source: ${config.origin}/quizzes/${quiz.file}`] : []),
        ...practice.map(p => `- [${module.title}: ${p.section}](${config.origin}${buildLocation(course.language, course.course_key, module.module_key, p.section)}): ${p.count} authored items; structured source: ${config.origin}/practice/${p.file}`)];
    }));
}
guides.push("", "## Attribution and editions",
  `- [Content license](${license}): CC BY-NC-SA 4.0; preserve attribution, noncommercial and share-alike terms.`,
  `- [License map](${repo}/blob/main/LICENSE.md): Site code is Apache-2.0; third-party components retain their licenses.`,
  `- [Archived complete collection, edition ${publication.edition}](https://doi.org/${publication.version_doi}): Released ${publication.release_date}; this DOI identifies the archive, not each live revision.`,
  `- [All archived editions](https://doi.org/${publication.concept_doi}): Collection DOI.`,
  "When citing a live lecture, use its canonical page URL, language and access date. Distinguish it from the archived complete collection.", "");
fs.writeFileSync(`${output}/llms.txt`, guides.join("\n"));
fs.writeFileSync(`${output}/agents.txt`, [
  "# Agent guide: Vlad Kulikov Open Courses", "",
  "This is an informational guide for automated readers, not an access-control standard or a grant of additional rights.",
  `Website: ${config.origin}/`, "Author: Vlad Kulikov / Влад Куликов", "Languages: en, ru",
  "Purpose: Read public AI/ML lectures, take quizzes, study worked exercises and view or download existing notebooks.",
  "Interaction: Ordinary HTTPS pages and files. No agent API, MCP endpoint, authentication, checkout, code execution or user submissions are provided by this library.",
  `Index: ${config.origin}/llms.txt`, `Sitemap: ${config.origin}/sitemap.xml`, `Crawler rules: ${config.origin}/robots.txt`,
  `Content license: CC BY-NC-SA 4.0 (${license})`,
  "This guide does not authorize uses beyond the applicable licenses. External Stepik and Leanpub services have their own terms.",
  "Use canonical vladkulikov.com lecture URLs for references. Use hreflang links to find the corresponding translation; retain module keys even when the reading order changes.",
  "Prefer public HTML or Markdown sources. Formula source is LaTeX in Markdown, with KaTeX-rendered HTML/MathML in lecture pages.",
  `The live texts were updated on ${publication.content_updated}. Archived complete collection ${publication.edition} (${publication.release_date}) has DOI https://doi.org/${publication.version_doi}; it is distinct from the evolving website.`,
  "Module quizzes are published as static HTML and structured JSON under /quizzes/. Choice and numeric answers are checked in the browser. Open questions and code tasks require self-review against the supplied author answer; there is no code execution or server-side grading. Quiz progress is stored only in the reader's browser. Exercise solutions and notebook previews are native expandable HTML; their open state is shared across RU/EN. Structured practice sources are under /practice/ and original files and course archives under /downloads/. Information Theory Python examples are code answers to their exercises, not separate notebooks. Do not invent missing exercises, answer keys, files, or learning outcomes.",
  "Attribute the author, link the source and identify its language and edition or access date when citing. Use the source text to distinguish claims from your own interpretation.", "",
].join("\n"));
console.log(`Generated ${manifest.length} indexable pages, 404, sitemap, robots.txt, llms.txt and agents.txt.`);
