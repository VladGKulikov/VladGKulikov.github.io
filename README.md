# Vlad Kulikov · Open Courses

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21877456.svg)](https://doi.org/10.5281/zenodo.21877456)

Personal website and course library: <https://vladkulikov.com/>

Public bilingual lecture and quiz library for three courses:

- Modern LLMs
- RL for LLM
- Information Theory for Machine Learning

The repository contains lectures, illustrations, and authored module quizzes with
answer checking and worked answers. Exercises, project materials, and downloadable
notebooks remain a later publication stage.

## Course editions on Stepik

| Course | English | Russian |
|---|---|---|
| Modern LLMs | <https://stepik.org/297276> | <https://stepik.org/297275> |
| RL for LLM | <https://stepik.org/296686> | <https://stepik.org/296688> |
| Information Theory for ML | <https://stepik.org/295409> | <https://stepik.org/295411> |

## What the complete Stepik courses add

The website now includes 943 authored quiz questions per language. The complete
Stepik editions also provide the remaining practical routes:

| Course | Published website quizzes (per language) | Further practice on Stepik |
|---|---|---|
| Modern LLMs | 415 choice questions across 20 modules | Exercises, projects, solutions and notebooks |
| Reinforcement Learning for LLM | 151 platform-bank questions across 13 modules; objective parts checked automatically, open/code tasks reviewed against worked answers | Core exercises, programming tasks, solutions and notebooks |
| Information Theory for ML | 377 choice and numeric questions across 15 modules; module 16 is a reference appendix without an authored quiz | Exercises and detailed solutions |

The canonical authoring repositories are maintained separately. This public repository is a generated, publication-safe copy.

## Static pages and search metadata

`npm run build:pages` creates a portable `dist-pages/` directory for GitHub Pages:
two home pages, six course overviews, 98 complete lecture pages, 96 quiz pages, and a custom
`404.html`. Lectures, KaTeX formulas, illustrations and ordinary navigation are
present in the original HTML. React hydrates that same interface to provide
module search, collapsible panels and preservation of reading position on
language changes. With JavaScript unavailable, native navigation and expandable
module/section indexes remain available. Quiz questions, options, and native
expandable worked answers also remain readable without JavaScript.

Permanent addresses use `/courses/<course-key>/<module-key>/` in English and
`/ru/courses/<course-key>/<module-key>/` in Russian. Course overviews omit the
module key. Module quiz URLs append `tests/` to the lecture URL. Module keys identify subjects, not reading order: in Modern LLMs,
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

## Quiz publication and progress

`python scripts/export-quizzes.py` exports the current authored question files
from the three local canonical repositories without modifying them. It accepts
the same `--source-root COURSE=PATH` overrides as the lecture exporter. Modern
module keys follow `previous_id`; RU/EN question and grading parity is required.
Information Theory uses stable internal question IDs because a few optional
topics have different section numbers in the translations; original source
question numbers are still displayed. File provenance and SHA-256 hashes are
recorded in `QUIZ_CONTENT_AUDIT.json`.

`public/quizzes/**` contains the localized question banks and worked answers.
`app/generated/quizzes.json` is their lightweight route manifest. RL uses only
the current canonical `route="platform"` records, rather than obsolete Stepik
build artifacts. Reviewed numeric/matching contracts and ordinal answer keys
are declared in `scripts/quiz-grading-overrides.json`; original reasoning stays
available in full. Where a prompt also asks for an explanation, the UI explicitly
labels the automatic result as checking only the selection/numeric part.

Checking uses exact sets for choices, ordered values for matching, and displayed
absolute tolerances for finite numeric inputs (including decimal commas and
simple fractions). Open reasoning and code tasks use self-review, never keyword
grading. There is no server-side grading or code execution. All published answer
keys are public study material, not secure examination credentials.

Progress, checked state, expanded answers, and free-form drafts use localStorage
under a language-independent module/revision key. They stay in the browser and
are not sent to a server. If storage is blocked, the current session still works
in memory. Module reset asks for an explicit inline confirmation. Menus and the
reading position use the same navigation/restoration logic as lectures.

`npm run check:quizzes` tests every published key, alternative choice, numeric
parser, tolerance boundary, translation contract, and quiz route. It runs after
HTML validation in `build:pages`, including the existing Pages deployment job.

## Later learning materials

Later passes will add exercises with answers hidden until expanded, and existing
notebooks/solutions as text or downloadable files. Code execution is outside that
scope. Those additional materials have not been copied in this quiz stage.

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
python scripts/export-quizzes.py
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
