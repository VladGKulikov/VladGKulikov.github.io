# Vlad Kulikov · Open Courses

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21877456.svg)](https://doi.org/10.5281/zenodo.21877456)

Personal website and course library: <https://vladkulikov.com/>

Public bilingual lecture library for three courses:

- Modern LLMs
- RL for LLM
- Information Theory for Machine Learning

The repository contains lecture text and illustrations only. Quizzes, exercises, solutions, project materials, and notebooks are intentionally excluded and remain part of the complete Stepik courses.

## Course editions on Stepik

| Course | English | Russian |
|---|---|---|
| Modern LLMs | <https://stepik.org/297276> | <https://stepik.org/297275> |
| RL for LLM | <https://stepik.org/296686> | <https://stepik.org/296688> |
| Information Theory for ML | <https://stepik.org/295409> | <https://stepik.org/295411> |

## What the complete Stepik courses add

This repository is the open lecture reader. The complete Stepik editions add the assessment and practical routes:

| Course | Additional assessment and practice on Stepik |
|---|---|
| Modern LLMs | 299 auto-graded questions; 137 self-checks; 65 analytical and computational exercises; 36 project exercises; 19 main, 17 project-demo, and 4 Capstone notebooks. The complete course is free on Stepik |
| Reinforcement Learning for LLM | 109 native Stepik tests; 151 self-checks; 77 analytical and computational tasks; 77 programming tasks; 13 notebooks |
| Information Theory for ML | 377 auto-graded questions; all 78 exercises with detailed solutions, including 12 Python exercises. The complete course is free on Stepik |

The canonical authoring repositories are maintained separately. This public repository is a generated, publication-safe copy.

## Static pages and search metadata

`npm run build:pages` creates a portable `dist-pages/` directory for GitHub Pages:
two home pages, six course overviews, 98 complete lecture pages, and a custom
`404.html`. Lectures, KaTeX formulas, illustrations and ordinary navigation are
present in the original HTML. React hydrates that same interface to provide
module search, collapsible panels and preservation of reading position on
language changes. With JavaScript unavailable, native navigation and expandable
module/section indexes remain available.

Permanent addresses use `/courses/<course-key>/<module-key>/` in English and
`/ru/courses/<course-key>/<module-key>/` in Russian. Course overviews omit the
module key. Module keys identify subjects, not reading order: in Modern LLMs,
for example, `module-18` is the Basic Transformer lecture at reading position 3.
Do not renumber keys when changing the catalog order. Previous query-based URLs
are translated to permanent addresses by the browser, preserving section hashes;
GitHub Pages itself redirects the previous domain to the configured custom domain.

The build generates per-page titles, descriptions, canonical URLs, reciprocal
language alternates, social cards, `sitemap.xml`, `robots.txt`, `llms.txt`, and
`agents.txt` from the shared catalog. The two text guides describe the public
library, stable URLs, languages, author, licenses and edition boundaries. They
are informational discovery aids, not permission grants or guarantees about
search ranking or adoption by AI crawlers. `agents.txt` does not advertise any
agent API or capabilities that the site does not provide.

`SITE_CONFIG.json` holds the canonical origin, the optional dismissible Stepik
notice, search-verification values and one optional GA4 integration. All
verification values and analytics identifiers are empty by default; no analytics
script is loaded in that state. To enable GA4, set provider to `ga4` and supply
the real measurement ID. In the GA4 web stream, disable automatic enhanced
measurement (including history-based page views, outbound clicks, site search
and form tracking): the reader sends its own page views and Stepik/Leanpub
outbound events once, with canonical paths and without query strings or hashes.
Do not add a second tag manager or tracker. Search Console and Yandex Webmaster
verification meta tags are emitted only when their actual values are supplied.

The GitHub Pages custom domain is managed in repository settings. The existing
GitHub Actions deployment is preserved; DNS and HTTPS are not reconfigured by
the build. When mirroring onto another static host, retain these page paths and
canonical references to `vladkulikov.com`.

`npm run check:static` validates the already-built HTML, metadata, internal
links, heading hierarchy, formulas, images, legacy route mappings and content
hashes. It runs automatically at the end of `build:pages`. The source markup
and publication/license checks remain separate.

## Planned learning materials

The next passes are authorized to add public assessments with answer checking,
exercises with answers hidden until expanded, and existing notebooks/solutions
as text or downloadable files. Code execution is outside that scope. These
materials have not been copied into this first static-lecture migration; their
publication boundaries and access wording must be updated together when the
actual materials and answer-checking interface are implemented.

## Publication edition

The live lecture texts were updated on **2026-10-15** from the audited bilingual
sources: 20 Modern LLMs modules, 13 RL for LLM modules, and 16 Information Theory
modules per language. Existing module URLs retain their subjects, while the
Modern LLMs reading order follows the revised course.

The archived book edition remains **2026.1**, released on 2026-08-10. Its DOI
identifies that archive; the live website has a separate content update date in
[`PUBLICATION.json`](PUBLICATION.json). Source commits, per-lecture hashes, and
the lecture-only publication checks are recorded in [`CONTENT_BUILD.json`](CONTENT_BUILD.json).

To rebuild from local canonical repositories without changing them:

```powershell
python scripts/refresh-course-content.py
npm ci
npm run check:math
npm run check:publication
npm run build:pages
```

Use `--source-root "modern-llms=C:/path/to/source"` (and the corresponding
`rl-for-llm` / `information-theory-for-ml` options) to override source locations.
The section exclusions in the builder are reviewed for the October 2026 texts;
review them again when changing lecture structure.

## Licensing

This is a mixed-license repository:

- lecture text, course illustrations, course metadata, and author-written
  explanatory prose: **CC BY-NC-SA 4.0**;
- site source code, build tooling, and standalone publication software:
  **Apache-2.0**;
- third-party components: their respective original licenses.

See [`LICENSE.md`](LICENSE.md) for the scope, the
[`LICENSE-CONTENT.md`](LICENSE-CONTENT.md) notice for course content,
[`LICENSE-CODE`](LICENSE-CODE) for the Apache-2.0 text, and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for dependencies.

Restricted Stepik materials that are absent from this repository are not
covered by these licenses.

## Citation

GitHub-compatible citation metadata is provided in
[`CITATION.cff`](CITATION.cff). The DOI for the archived public Reader Edition
`2026.1` is <https://doi.org/10.5281/zenodo.21877457>. Use
<https://doi.org/10.5281/zenodo.21877456> when linking to the collection across
all current and future editions.

## Author

I am a Lead AI/ML specialist focused on LLMs in FinTech. I have worked in ML since 2017 across R&D, ML/GenAI system development, and AI adoption consulting.

- [Stepik profile](https://stepik.org/users/29821475)
- [LinkedIn](https://www.linkedin.com/in/vlad-g-kulikoff/)
- [GitHub](https://github.com/VladGKulikov)
