# Vlad Kulikov · Open Courses

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21877456.svg)](https://doi.org/10.5281/zenodo.21877456)

Personal website and course library: <https://vladkulikov.com/>

Public bilingual course library for three courses:

- Modern LLMs
- RL for LLM
- Information Theory for Machine Learning

The repository contains lectures, illustrations, module quizzes with answer
checking, exercises with expandable worked solutions, and existing notebooks
that can be read as text or downloaded. No code is executed on the website.

## Course editions on Stepik

| Course | English | Russian |
|---|---|---|
| Modern LLMs | <https://stepik.org/297276> | <https://stepik.org/297275> |
| RL for LLM | <https://stepik.org/296686> | <https://stepik.org/296688> |
| Information Theory for ML | <https://stepik.org/295409> | <https://stepik.org/295411> |

## Published practice

Counts are per language; the library is available in both RU and EN.

| Course | Quiz questions | Exercises / project tasks | Existing notebooks |
|---|---|---|---|
| Modern LLMs | 415 | 116 | 42 (20 main, 18 project demos, 4 MiniLLM) |
| RL for LLM | 151 platform-bank questions | 263 core tasks | 13 |
| Information Theory for ML | 377 | 90 | No `.ipynb`; 9 Python solutions attached to their exercises |

This site is the main home for all three courses. The same materials are also
available free on Stepik for readers who prefer that platform.

The canonical authoring repositories are maintained separately. This public repository is a generated, publication-safe copy.

## Static pages and search metadata

`npm run build:pages` creates a portable `dist-pages/` directory for GitHub Pages:
two home pages, six course overviews, 98 complete lecture pages, 96 quiz pages, 94 exercise pages, 66 notebook pages (362 indexable pages in total), and a custom
`404.html`. Lectures, KaTeX formulas, illustrations and ordinary navigation are
present in the original HTML. React hydrates that same interface to provide
module search, collapsible panels and preservation of reading position on
language changes. With JavaScript unavailable, native navigation and expandable
module/section indexes remain available. Quiz questions, options, and native
expandable worked answers also remain readable without JavaScript.

Permanent addresses use `/courses/<course-key>/<module-key>/` in English and
`/ru/courses/<course-key>/<module-key>/` in Russian. Course overviews omit the
module key. Module quiz URLs append `tests/`, exercise URLs `exercises/`, and notebook URLs `notebooks/` to the lecture URL. Module keys match the current module numbers:
`module-03` is Module 3, the Basic Transformer, and `module-12` is Module 12,
Diffusion LLMs. The old numbering is no longer used for page addresses.
Previous query-based URLs
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
module keys follow the current manifest IDs; RU/EN question and grading parity is required.
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

## Exercises, code solutions and notebooks

`python scripts/export-practice.py` reads authored exercises and solutions from
all three canonical repositories. Modern exercises come from lecture exercise
sections; RL uses only current `route="core"` records; Information Theory joins
exercise prompts to the separate authored solution files, including bonus tasks.
`PRACTICE_CONTENT_AUDIT.json` records provenance, hashes and copied artifacts.

`public/practice/**` holds 160 localized pages of structured practice content;
`app/generated/practice.json` is their route manifest. Existing module keys,
course colours, mobile drawers and bilingual reading restoration are shared with
the lecture and quiz reader. Answers and notebook previews use native `details`
controls and are readable without JavaScript. Their disclosure state is stored
locally under a language-independent module/revision key. Markdown formulas and
tables render normally; long formulas, tables and code scroll within their blocks.

Information Theory's 9 Python examples are the code part of the corresponding
exercise solutions. They appear inside the same expandable answer with syntax
highlighting, line numbers, Copy and Download controls. No notebook section is
invented for this course. Two Modern LLMs project tasks (module-13 E6/E7) have no
complete authored answer: the page says so and links the module's demos as guidance.

The 110 notebook files (55 RU/EN pairs) and existing support files are published
under `public/downloads/**`. Eight Modern LLMs Capstone notebooks have a small
publication adaptation: their setup instructions link to this site's archive,
and their archive loader accepts `course-materials.zip` and its RU/EN package folders. The remaining
notebook cells, IDs, training code and outputs are preserved. Original source
hashes and the adaptation are recorded in `PRACTICE_CONTENT_AUDIT.json`; all
other files are copied byte-for-byte. Each course has a ZIP archive
preserving the source folder hierarchy and license notices. The archive is useful
for notebooks that import sibling modules or load course configuration/data files.
Notebook previews match the downloadable code cells and normalize explanatory Markdown;
they do not execute cells, load kernels, or render active notebook output. Current
source notebooks have no saved outputs; the exporter requires review if this changes.

`npm run check:practice-markup` checks all 160 practice documents for broken
Markdown and math. `npm run check:practice` verifies routes, bilingual item/cell
alignment, original file hashes, download integrity, code preservation and the
Information Theory exercise/code associations; it runs in `build:pages` and CI.

## Publication edition

The live lecture texts were updated on **2026-10-15** from the audited bilingual
sources: 20 Modern LLMs modules, 13 RL for LLM modules, and 16 Information Theory
modules per language. Existing module URLs retain their subjects, while the
Modern LLMs reading order follows the revised course.

The archived complete collection is **2026.3**, released on 2026-10-25:
<https://doi.org/10.5281/zenodo.23649045>. It includes the bilingual lectures, illustrations,
quizzes and answer checker, exercises and worked solutions, existing notebooks
and original files, six PDF and six EPUB books, the static website and source.
The previous 2026.1 lecture archive remains available at
<https://doi.org/10.5281/zenodo.21877457>. The collection DOI
<https://doi.org/10.5281/zenodo.21877456> resolves to the latest archived edition.
Live website updates can continue independently of a fixed archival release.
Publication metadata is in [`PUBLICATION.json`](PUBLICATION.json); source commits
and lecture hashes are in [`CONTENT_BUILD.json`](CONTENT_BUILD.json).

To rebuild from local canonical repositories without changing them:

```powershell
python scripts/refresh-course-content.py
python scripts/export-quizzes.py
python scripts/export-practice.py
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
[`CITATION.cff`](CITATION.cff). Cite the complete bilingual **2026.3** collection
using <https://doi.org/10.5281/zenodo.23649045>. This DOI identifies the collection, including
all six books; it is not a separate DOI for each individual book. Use
<https://doi.org/10.5281/zenodo.21877456> when referring to the evolving collection
without specifying an edition.

## Author

I am a Lead AI/ML specialist focused on LLMs in FinTech. I have worked in ML since 2017 across R&D, ML/GenAI system development, and AI adoption consulting.

- [Stepik profile](https://stepik.org/users/29821475)
- [LinkedIn](https://www.linkedin.com/in/vlad-g-kulikoff/)
- [GitHub](https://github.com/VladGKulikov)

## Modern LLMs text edition — 25 October 2026

All 40 lecture texts were replaced with the author's new RU/EN edition.
Practice, quizzes, notebooks and their downloadable packages are unchanged.
Module numbers in links and illustration paths were mapped by content: the
supplied M1–M8 used legacy routes, whereas M9–M20 already used current routes.
The source inventory is in `LECTURE_EDITION_2026-10-25.json`. For a bounded
refresh run `python scripts/refresh-modern-edition.py <course-source-root>`.
The other courses and their 2026.2 PDF/EPUB books remain unchanged.
