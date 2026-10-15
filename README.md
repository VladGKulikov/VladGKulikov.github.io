# Vlad Kulikov · Open Courses

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21877456.svg)](https://doi.org/10.5281/zenodo.21877456)

Personal website and course library: <https://vladgkulikov.github.io/>

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
