#!/usr/bin/env python3
"""Build a publication-only copy from the three immutable course repositories.

Refresh the public lecture layer after the October 2026 audits. Practice sections
are published on their own pages and linked from the lecture layer. Modern
module IDs follow previous_id so existing reader URLs retain their meaning.
The source repositories are read only; only this site is regenerated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ROOTS = {
    "modern-llms": Path(r"C:\PersonalProjects\Modern LLMs"),
    "rl-for-llm": Path(r"C:\PersonalProjects\RL for LLM"),
    "information-theory-for-ml": Path(r"C:\PersonalProjects\Information_Theory_for_ML"),
}

COURSE_META = {
    "modern-llms": {
        "title": {"en": "Modern LLMs", "ru": "Современные LLM"},
        "description": {
            "en": "Architecture, training, inference, agents, and evaluation of modern language-model systems.",
            "ru": "Архитектура, обучение, инференс, агенты и оценивание современных языковых моделей.",
        },
        "stepik": {"en": "https://stepik.org/297276", "ru": "https://stepik.org/297275"},
    },
    "rl-for-llm": {
        "title": {"en": "RL for LLM", "ru": "RL для LLM"},
        "description": {
            "en": "Reinforcement learning, preference optimization, verifiable rewards, agents, and safety.",
            "ru": "Обучение с подкреплением, оптимизация предпочтений, проверяемые награды, агенты и безопасность.",
        },
        "stepik": {"en": "https://stepik.org/296686", "ru": "https://stepik.org/296688"},
    },
    "information-theory-for-ml": {
        "title": {
            "en": "Information Theory for ML",
            "ru": "Теория информации для машинного обучения",
        },
        "description": {
            "en": "Information theory from entropy and coding to NLP, LLMs, computer vision, and reinforcement learning.",
            "ru": "Теория информации: от энтропии и кодирования до NLP, LLM, компьютерного зрения и обучения с подкреплением.",
        },
        "stepik": {"en": "https://stepik.org/295409", "ru": "https://stepik.org/295411"},
    },
}

H2_RE = re.compile(r"(?m)^##\s+(.+?)\s*$")
H1_RE = re.compile(r"(?m)^#\s+(.+?)\s*$")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
MARKDOWN_IMAGE_RE = re.compile(
    r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+(?:\"[^\"]*\"|'[^']*'))?\)"
)
HTML_IMAGE_RE = re.compile(r"(<img\b[^>]*?\bsrc=[\"'])([^\"']+)([\"'][^>]*>)", re.IGNORECASE)
NOTEBOOK_ROUTE_RE = re.compile(
    r"\.ipynb(?:\b|[?#])|colab\.research\.google\.com|kaggle\.com/(?:code|notebooks)",
    re.IGNORECASE,
)
VIZ_DIRECTIVE_RE = re.compile(r"^\s*\[VIZ\s*:", re.IGNORECASE)
PRIVATE_H2_RE = re.compile(
    r"(?:\bself[- ]?check\b|\bexercises?\b|exercise\s+solutions?|"
    r"вопросы\s+для\s+самопроверки|\bсамопроверка\b|\bупражнения?\b|решения\s+упражнений)",
    re.IGNORECASE,
)


@dataclass
class BuildContext:
    project_root: Path
    content_root: Path
    assets_root: Path
    source_roots: dict[str, Path]
    missing_assets: list[dict[str, str]]
    copied_assets: dict[str, str]
    removed_notebook_blocks: list[dict[str, Any]]
    omitted_sections: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--source-root", action="append", default=[], metavar="COURSE=PATH")
    return parser.parse_args()


def safe_reset_generated_dir(path: Path, project_root: Path) -> None:
    path = path.resolve()
    project_root = project_root.resolve()
    if path.parent != project_root and project_root not in path.parents:
        raise RuntimeError(f"Refusing to reset path outside project: {path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def split_h2(markdown: str) -> tuple[str, list[tuple[str, str]]]:
    matches = list(H2_RE.finditer(markdown))
    if not matches:
        return markdown.strip(), []
    preamble = markdown[: matches[0].start()].strip()
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        title = match.group(1).strip()
        body = markdown[match.end() : end].strip()
        sections.append((title, body))
    return preamble, sections


def strip_number_prefix(title: str) -> str:
    title = re.sub(r"^Lesson\s+\d+(?:\.\d+)*\.\s*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"^\d+(?:\.\d+)*\.\s*", "", title)
    return title.strip()


def public_title(title: str) -> str:
    numbered = re.match(r"^(\d+(?:\.\d+)*\.\s+)(.*)$", title)
    prefix, title = numbered.groups() if numbered else ("", title)
    replacements = (
        (r"(?i)^notebook\s+and\s+sources$", "Sources"),
        (r"(?i)^ноутбук\s+и\s+источники$", "Источники"),
        (r"(?i)^notebook\s+и\s+источники$", "Источники"),
        (r"(?i),?\s*notebook,?\s+and\s+sources", " and sources"),
        (r"(?i),?\s*ноутбук\s+и\s+источники", " и источники"),
        (r"(?i),?\s*notebook\s+и\s+источники", " и источники"),
        (r"(?i),?\s*jupyter-ноутбук\s+и\s+источники", " и источники"),
    )
    value = title
    for pattern, replacement in replacements:
        value = re.sub(pattern, replacement, value)
    value = re.sub(r"(?i)notebook,?\s+and\s+", "", value)
    value = re.sub(r"(?i)(?:Jupyter-)?(?:ноутбук|notebook),?\s+и\s+", "", value)
    value = re.sub(r"\s+,", ",", value)
    value = re.sub(r"\s{2,}", " ", value).strip(" ,")
    return prefix + value


def remove_internal_markup(markdown: str) -> str:
    markdown = HTML_COMMENT_RE.sub("", markdown)
    lines = markdown.splitlines()
    output: list[str] = []
    skip_viz_caption = False
    for line in lines:
        if VIZ_DIRECTIVE_RE.match(line):
            skip_viz_caption = True
            continue
        if skip_viz_caption:
            if not line.strip():
                continue
            if line.lstrip().startswith("*") and line.rstrip().endswith("*"):
                skip_viz_caption = False
                continue
            skip_viz_caption = False
        output.append(line)
    return "\n".join(output)


def normalize_math_delimiters(markdown: str) -> str:
    """Normalize source LaTeX for remark-math without touching code fences.

    Some canonical lectures use ``\\(...\\)`` for inline math, while others
    continue a ``$$`` block with four-space indentation. Markdown interprets
    that indentation as code, leaving the display delimiter unclosed and
    causing KaTeX to consume the rest of the lecture. The public copy uses
    remark-math's dollar delimiters and removes code indentation only while a
    display-math block is open.
    """
    output: list[str] = []
    in_fence = False
    fence_marker = ""
    in_display_math = False
    display_prefix = ""

    for line in markdown.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
            output.append(line)
            continue

        if in_fence:
            output.append(line)
            continue

        line = re.sub(r"\\text\{«([^{}]*)»\}", r"\\text{``\1''}", line)

        if not in_display_math:
            line = re.sub(r"(?<!\\)\\\((.+?)(?<!\\)\\\)", r"$\1$", line)
            if line.count("$$") % 2 == 1:
                before, after = line.split("$$", 1)
                prefix_match = re.fullmatch(r"[ \t]*(?:>[ \t]*)*", before)
                if prefix_match:
                    display_prefix = before
                    output.append(f"{display_prefix}$$")
                    if after.strip():
                        output.append(f"{display_prefix}{after.lstrip()}")
                else:
                    display_prefix = ""
                    output.append(line)
                in_display_math = True
            else:
                output.append(line)
            continue

        body = line[len(display_prefix) :] if display_prefix and line.startswith(display_prefix) else line
        body = re.sub(r"^(?: {4}|\t)", "", body)
        if body.count("$$") % 2 == 1:
            before, after = body.split("$$", 1)
            if before.strip():
                output.append(f"{display_prefix}{before.rstrip()}")
            output.append(f"{display_prefix}$${after}")
            in_display_math = False
            display_prefix = ""
        else:
            output.append(f"{display_prefix}{body}" if display_prefix else body)

    return "\n".join(output)


def normalize_reader_markup(markdown: str) -> str:
    """Repair Markdown delimiters while preserving code and mathematical values."""
    lines = markdown.splitlines()
    output: list[str] = []
    fence: tuple[str, int] | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = (token[0], len(token))
            elif token[0] == fence[0] and len(token) >= fence[1]:
                fence = None
            output.append(line)
            continue
        if fence is not None:
            output.append(line)
            continue

        # A closing emphasis delimiter after punctuation needs a word boundary.
        line = re.sub(r"(\*\*[^*\n]+[—–])\*\*(?=\w)", r"\1** ", line)
        if line.lstrip().startswith("|"):
            # Source tables may wrap one inline formula over several physical
            # lines. GFM requires each table row to occupy a single line.
            dollar_count = lambda value: len(re.findall(r"(?<!\\)\$", value))
            while dollar_count(line) % 2 and index < len(lines):
                continuation = lines[index].strip()
                if not continuation or continuation.startswith("|"):
                    break
                line += " " + continuation
                index += 1

            # Raw vertical bars in math are GFM column separators, even inside
            # dollar delimiters. Use TeX's explicit single-bar command instead.
            line = re.sub(
                r"(?<!\\)\$([^$\n]+)\$",
                lambda match: "$" + re.sub(r"(?<!\\)\|", r"\\vert ", match.group(1)) + "$",
                line,
            )
        output.append(line)
    return "\n".join(output)


def split_blocks(markdown: str) -> list[str]:
    """Split Markdown at blank lines while keeping fenced code intact."""
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    fence_marker = ""
    for line in markdown.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
        if not in_fence and not line.strip():
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def remove_notebook_routes(
    markdown: str,
    *,
    context: BuildContext,
    course_key: str,
    language: str,
    module_key: str,
) -> str:
    kept: list[str] = []
    for block in split_blocks(markdown):
        if NOTEBOOK_ROUTE_RE.search(block):
            context.removed_notebook_blocks.append(
                {
                    "course_key": course_key,
                    "language": language,
                    "module_key": module_key,
                    "preview": re.sub(r"\s+", " ", block)[:240],
                }
            )
            continue
        kept.append(block)
    return "\n\n".join(kept)


def is_external_url(value: str) -> bool:
    return bool(re.match(r"^(?:https?:|data:|//)", value, flags=re.IGNORECASE))


def copy_asset(
    raw_url: str,
    *,
    source_file: Path,
    source_root: Path,
    course_key: str,
    language: str,
    module_key: str,
    context: BuildContext,
) -> str:
    if is_external_url(raw_url) or raw_url.startswith("#"):
        return raw_url
    path_text = urllib.parse.unquote(raw_url.split("#", 1)[0].split("?", 1)[0])
    candidate = (source_file.parent / path_text).resolve()
    source_root = source_root.resolve()
    if source_root not in candidate.parents and candidate != source_root:
        context.missing_assets.append(
            {"source": str(source_file), "asset": raw_url, "reason": "outside_source_repository"}
        )
        return raw_url
    if not candidate.is_file():
        context.missing_assets.append(
            {"source": str(source_file), "asset": raw_url, "reason": "not_found"}
        )
        return raw_url

    source_key = str(candidate).casefold()
    if source_key in context.copied_assets:
        return context.copied_assets[source_key]

    destination_dir = context.assets_root / course_key / language / module_key
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / candidate.name
    if destination.exists() and destination.read_bytes() != candidate.read_bytes():
        digest = hashlib.sha1(str(candidate).encode("utf-8")).hexdigest()[:8]
        destination = destination_dir / f"{candidate.stem}-{digest}{candidate.suffix}"
    shutil.copy2(candidate, destination)
    site_path = destination.relative_to(context.assets_root.parent).as_posix()
    context.copied_assets[source_key] = site_path
    return site_path


def localize_images(
    markdown: str,
    *,
    source_file: Path,
    source_root: Path,
    course_key: str,
    language: str,
    module_key: str,
    context: BuildContext,
) -> str:
    def replace_markdown(match: re.Match[str]) -> str:
        alt, raw_url = match.group(1), match.group(2)
        new_url = copy_asset(
            raw_url,
            source_file=source_file,
            source_root=source_root,
            course_key=course_key,
            language=language,
            module_key=module_key,
            context=context,
        )
        return f"![{alt}]({new_url})"

    def replace_html(match: re.Match[str]) -> str:
        new_url = copy_asset(
            match.group(2),
            source_file=source_file,
            source_root=source_root,
            course_key=course_key,
            language=language,
            module_key=module_key,
            context=context,
        )
        return f"{match.group(1)}{new_url}{match.group(3)}"

    markdown = MARKDOWN_IMAGE_RE.sub(replace_markdown, markdown)
    return HTML_IMAGE_RE.sub(replace_html, markdown)


def sanitize_markdown(
    markdown: str,
    *,
    source_file: Path,
    source_root: Path,
    course_key: str,
    language: str,
    module_key: str,
    context: BuildContext,
) -> str:
    markdown = remove_internal_markup(markdown)
    # ReactMarkdown does not execute raw HTML. Preserve optional explanations
    # and their labels as ordinary Markdown instead of dropping <summary> text.
    def summary_label(match: re.Match[str]) -> str:
        label = re.sub(r"</?strong\b[^>]*>", "", match.group(1), flags=re.IGNORECASE).strip()
        return f"**{label}**"

    markdown = re.sub(r"<summary>(.*?)</summary>", summary_label, markdown, flags=re.DOTALL | re.IGNORECASE)
    markdown = re.sub(r"</?details\b[^>]*>", "", markdown, flags=re.IGNORECASE)
    markdown = normalize_math_delimiters(markdown)
    markdown = normalize_reader_markup(markdown)
    markdown = remove_notebook_routes(
        markdown,
        context=context,
        course_key=course_key,
        language=language,
        module_key=module_key,
    )
    markdown = localize_images(
        markdown,
        source_file=source_file,
        source_root=source_root,
        course_key=course_key,
        language=language,
        module_key=module_key,
        context=context,
    )
    if course_key == "modern-llms":
        manifest = json.loads((source_root / "course_manifest.json").read_text(encoding="utf-8"))
        targets = {
            (source_root / module["lectures"][language]).resolve():
            f"module-{module.get('previous_id') or module['id']}"
            for module in manifest["modules"]
        }
    else:
        targets = {}
    def heading_slug(title: str) -> str:
        title = re.sub(r"[`*_()\[\]{}]", "", title.lower())
        return re.sub(r"[^a-zа-яё0-9]+", "-", title, flags=re.I).strip("-")[:80]
    section_anchors = {
        int(match.group(1)): heading_slug(match.group(0).removeprefix("## "))
        for match in re.finditer(r"(?m)^## (\d+)\. .+$", markdown)
    }
    # Source HTML anchors are not executed by the Markdown reader. Resolve the
    # hand-written diffusion TOC against the retained public headings instead.
    markdown = re.sub(r'<a\s+(?:id|name)=["\'][^"\']+["\']\s*>\s*</a>', "", markdown)
    link_re = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)\)")
    def resolve_link(match: re.Match[str]) -> str:
        label, href = match.groups()
        custom_anchor = re.fullmatch(r"#m\d+-(\d+)", href)
        if custom_anchor:
            section = int(custom_anchor.group(1))
            if section in section_anchors:
                return f"[{label}](#{section_anchors[section]})"
            return f"[{label} (Stepik)]({COURSE_META[course_key]['stepik'][language]})"
        if is_external_url(href) or href.startswith(("#", "mailto:")):
            return match.group(0)
        target = (source_file.parent / urllib.parse.unquote(href.split("#", 1)[0])).resolve()
        if target in targets:
            return f"[{label}]({targets[target]}.md)"
        if course_key != "modern-llms" and target.suffix == ".md" and not re.search(
            r"exercise|solution|test|question|assessment", target.name, re.I
        ):
            module_match = re.search(r"module[-_](\d+)", str(target.relative_to(source_root)) if target.is_relative_to(source_root) else "", re.I)
            if target.is_file() and module_match:
                return f"[{label}](module-{int(module_match.group(1)):02}.md)"
        if re.search(r"exercise|solution|test|question|assessment", target.name, re.I):
            return f"[{label}]({COURSE_META[course_key]['stepik'][language]})"
        return label
    markdown = link_re.sub(resolve_link, markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
    return localize_learning_navigation(markdown, course_key, language, module_key)


def localize_learning_navigation(markdown: str, course_key: str, language: str, module_key: str) -> str:
    """Point learning routes to the published practice pages on either host."""
    # Also remove the pre-practice banner when updating an existing public copy.
    markdown = re.sub(
        r"(?m)^> (?:Открытая лекционная версия\.|Open lecture edition\.).*\n\n",
        "", markdown,
    )
    base = f"/{'ru/' if language == 'ru' else ''}courses/{course_key}/{module_key}/"

    def learning_link(match: re.Match[str]) -> str:
        label, href = match.groups()
        if href != COURSE_META[course_key]['stepik'][language]:
            return match.group(0)
        label = re.sub(r"\s*\(Stepik\)$", "", label)
        if re.search(r"notebook|ноутбук", label, re.I):
            section = "notebooks"
        elif re.search(r"exercise|solution|упражнен|решени", label, re.I):
            section = "exercises"
        elif re.search(r"self[- ]?check|question|test|самопровер|вопрос|тест", label, re.I):
            section = "tests"
        else:
            return match.group(0)
        return f"[{label}]({base}{section}/)"

    markdown = re.sub(r"(?<!!)\[([^\]]+)\]\((https://stepik\.org/\d+)\)", learning_link, markdown)
    if course_key == "rl-for-llm" and module_key == "module-01":
        markdown = markdown.replace(
            "| **A** | Stepik / CPU |",
            "| **A** | Тесты на сайте или Stepik; код — локально на CPU |"
            if language == "ru" else
            "| **A** | Quizzes on this site or Stepik; code locally on CPU |",
        )
    if course_key == "rl-for-llm" and module_key == "module-05":
        old, new = (
            ("Оцениваемая практика содержит 20 заданий основного маршрута и 12 дополнительных платформенных заданий. Условие и подробное решение каждого задания расположены рядом; при переносе на Stepik блок решения следует показывать после попытки или выносить в следующий шаг. Основной набор охватывает:",
             f"Практика содержит [20 заданий основного маршрута]({base}exercises/) и [12 дополнительных вопросов для проверки знаний]({base}tests/). Подробные решения упражнений раскрываются по кнопке. Основной набор охватывает:")
            if language == "ru" else
            ("The graded practice contains 20 core tasks and 12 additional platform tasks. Every prompt is followed by a worked solution; on Stepik, that solution should appear only after an attempt or as the next step. The core route covers:",
             f"Practice contains [20 core tasks]({base}exercises/) and [12 additional quiz questions]({base}tests/). Worked exercise solutions expand on request. The core route covers:")
        )
        markdown = markdown.replace(old, new)
    if course_key == "rl-for-llm" and module_key == "module-06":
        old, new = (
            ("Основной оцениваемый набор содержит 20 заданий: 8 концептуальных, 6 аналитических и 6 программных микрозадач. Ещё 12 дополнительных заданий предназначены для Stepik или другой платформы и разбирают пограничные случаи: семантику остановки, EOS в вероятности последовательности, порядок обработчиков логитов, носители оценивателей и маскирование результатов инструментов. Каждое условие сопровождается подробным решением, которое открывается после попытки.",
             f"Основной набор содержит [20 упражнений]({base}exercises/): 8 концептуальных, 6 аналитических и 6 программных микрозадач. Ещё [12 вопросов для проверки знаний]({base}tests/) разбирают пограничные случаи: семантику остановки, EOS в вероятности последовательности, порядок обработчиков логитов, носители оценивателей и маскирование результатов инструментов. Подробные решения упражнений раскрываются по кнопке.")
            if language == "ru" else
            ("The graded core set contains 20 tasks: 8 conceptual questions, 6 analytical problems, and 6 programming microtasks. Twelve additional platform tasks cover stopping semantics, EOS accounting, processor order, estimator support, and masking of tool observations. Every prompt is paired with a worked solution revealed after an attempt.",
             f"The core set contains [20 exercises]({base}exercises/): 8 conceptual questions, 6 analytical problems, and 6 programming microtasks. Another [12 quiz questions]({base}tests/) cover stopping semantics, EOS accounting, processor order, estimator support, and masking of tool observations. Worked exercise solutions expand on request.")
        )
        markdown = markdown.replace(old, new)
    return markdown.rstrip() + "\n"


def module_document(
    *,
    course_key: str,
    language: str,
    module_key: str,
    position: float,
    title: str,
    source_file: Path,
    source_root: Path,
    markdown: str,
    context: BuildContext,
) -> dict[str, Any]:
    markdown = sanitize_markdown(
        markdown,
        source_file=source_file,
        source_root=source_root,
        course_key=course_key,
        language=language,
        module_key=module_key,
        context=context,
    )
    relative_path = Path(course_key) / language / f"{module_key}.md"
    destination = context.content_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(markdown, encoding="utf-8")
    return {
        "id": f"{course_key}:{language}:{module_key}",
        "course_key": course_key,
        "language": language,
        "module_key": module_key,
        "position": position,
        "title": title,
        "source_file": source_file.relative_to(source_root).as_posix(),
        "content_file": relative_path.as_posix(),
        "content_hash": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
        "markdown": markdown,
        "characters": len(markdown),
        "image_count": len(MARKDOWN_IMAGE_RE.findall(markdown)) + len(HTML_IMAGE_RE.findall(markdown)),
    }


# Reviewed by section number in both audited language versions. The omitted
# sections contain notebook routes, self-checks, or exercises and solutions.
MODERN_PRIVATE_SECTIONS = {
    1: {18, 20, 21}, 2: {7, 14, 15}, 3: {18, 19, 20},
    4: {12, 15, 16}, 5: {13, 17, 18}, 6: {14, 18, 19},
    7: {13, 17, 18}, 8: {13, 17, 18}, 9: {13, 18, 19},
    10: {13, 18, 19}, 11: {14, 18, 19}, 12: {13, 18, 19},
    13: {14, 18, 19}, 14: {14, 18, 19}, 15: {14, 18, 19},
    16: {14, 18, 19}, 17: {14, 18, 19}, 18: {14, 18, 19},
    19: {14, 18, 19}, 20: {17, 18},
}
MODERN_EXPECTED_SECTIONS = {1: 23, 2: 17, 3: 22, 4: 18, 5: 20, 6: 21, 15: 21}


def build_modern(boundary: dict[str, Any], language: str, context: BuildContext) -> list[dict[str, Any]]:
    del boundary
    course_key = "modern-llms"
    root = context.source_roots[course_key]
    manifest = json.loads((root / "course_manifest.json").read_text(encoding="utf-8"))
    documents = []
    for module in manifest["modules"]:
        number = int(module["number"])
        source_file = root / module["lectures"][language]
        raw = source_file.read_text(encoding="utf-8")
        preamble, sections = split_h2(raw)
        expected = MODERN_EXPECTED_SECTIONS.get(number, 20)
        if len(sections) != expected:
            raise RuntimeError(f"Modern {language} module {number}: expected {expected} sections, got {len(sections)}")
        key = f"module-{module.get('previous_id') or module['id']}"
        included = [preamble]
        for title, body in sections:
            prefix = re.match(r"^(\d+)\.\s", title)
            section_number = int(prefix.group(1)) if prefix else None
            if section_number in MODERN_PRIVATE_SECTIONS[number]:
                context.omitted_sections.append({
                    "course_key": course_key, "language": language,
                    "module_key": key, "title": title,
                    "reason": "reviewed_private_assessment_or_notebook_section",
                })
            else:
                included.append(f"## {public_title(title)}\n\n{body}")
        h1 = H1_RE.search(preamble)
        documents.append(module_document(
            course_key=course_key, language=language, module_key=key,
            position=float(number), title=h1.group(1) if h1 else module[f"title_{language}"],
            source_file=source_file, source_root=root,
            markdown="\n\n".join(included), context=context,
        ))
    return documents


def build_rl(boundary: dict[str, Any], language: str, context: BuildContext) -> list[dict[str, Any]]:
    del boundary  # The separate canonical assessment files are never opened.
    course_key = "rl-for-llm"
    root = context.source_roots[course_key]
    suffix = "EN" if language == "en" else "RU"
    documents: list[dict[str, Any]] = []
    for module_number in range(1, 14):
        source_file = (
            root / "modules" / f"module-{module_number:02d}" / f"Module_{module_number}_{suffix}.md"
        )
        raw = source_file.read_text(encoding="utf-8")
        preamble, sections = split_h2(raw)
        included = [preamble]
        for title, body in sections:
            if PRIVATE_H2_RE.search(title):
                context.omitted_sections.append(
                    {
                        "course_key": course_key,
                        "language": language,
                        "module_key": f"module-{module_number:02d}",
                        "title": title,
                        "reason": "private_heading_in_canonical_lecture_file",
                    }
                )
                continue
            included.append(f"## {title}\n\n{body}")
        h1 = H1_RE.search(preamble)
        documents.append(
            module_document(
                course_key=course_key,
                language=language,
                module_key=f"module-{module_number:02d}",
                position=float(module_number),
                title=strip_number_prefix(h1.group(1) if h1 else f"Module {module_number}"),
                source_file=source_file,
                source_root=root,
                markdown="\n\n".join(part for part in included if part.strip()),
                context=context,
            )
        )
    return documents


def information_theory_source(root: Path, module_number: int, language: str) -> Path:
    folder = root / f"Module_{module_number:02d}"
    candidates = [
        path
        for path in folder.glob(f"Module_{module_number}__*.md")
        if "Test_questions" not in path.name
        and "Exercise_solutions" not in path.name
        and (path.stem.endswith("_RU") if language == "ru" else not path.stem.endswith("_RU"))
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Information Theory {language} module {module_number}: expected one canonical source, got {candidates}"
        )
    return candidates[0]


def build_information_theory(
    boundary: dict[str, Any], language: str, context: BuildContext
) -> list[dict[str, Any]]:
    del boundary
    course_key = "information-theory-for-ml"
    root = context.source_roots[course_key]
    documents = []
    # Reviewed exercise section positions in the revised canonical lectures.
    private_sections = {2: 5, 3: 10, 4: 8, 5: 13, 6: 12, 7: 12,
                        8: 11, 9: 10, 10: 11, 11: 11, 12: 11,
                        13: 12, 14: 10, 15: 10}
    for number in range(1, 17):
        source_file = information_theory_source(root, number, language)
        preamble, sections = split_h2(source_file.read_text(encoding="utf-8"))
        included = [preamble]
        for title, body in sections:
            prefix = re.match(r"^(\d+)\.(\d+)\.\s", title)
            section_number = int(prefix.group(2)) if prefix else None
            if section_number == private_sections.get(number, -1):
                context.omitted_sections.append({
                    "course_key": course_key, "language": language,
                    "module_key": f"module-{number:02}", "title": title,
                    "reason": "reviewed_private_exercise_section",
                })
            else:
                included.append(f"## {title}\n\n{body}")
        h1 = H1_RE.search(preamble)
        documents.append(module_document(
            course_key=course_key, language=language,
            module_key=f"module-{number:02}", position=float(number),
            title=strip_number_prefix(h1.group(1) if h1 else f"Module {number}"),
            source_file=source_file, source_root=root,
            markdown="\n\n".join(included), context=context,
        ))
    return documents


def course_catalog(documents: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_course: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for document in documents:
        by_course.setdefault((document["course_key"], document["language"]), []).append(document)
    catalog: list[dict[str, Any]] = []
    for course_key in COURSE_META:
        for language in ("ru", "en"):
            modules = sorted(by_course[(course_key, language)], key=lambda item: item["position"])
            meta = COURSE_META[course_key]
            catalog.append(
                {
                    "course_key": course_key,
                    "language": language,
                    "title": meta["title"][language],
                    "description": meta["description"][language],
                    "stepik_url": meta["stepik"][language],
                    "module_count": len(modules),
                    "characters": sum(module["characters"] for module in modules),
                    "modules": [
                        {
                            key: module[key]
                            for key in ("id", "module_key", "position", "title", "content_file", "content_hash", "characters", "image_count")
                        }
                        for module in modules
                    ],
                }
            )
    return catalog


def validate_output(documents: list[dict[str, Any]], context: BuildContext) -> dict[str, Any]:
    notebook_leaks = []
    private_headings = []
    math_delimiter_issues = []
    html_formatting_leaks = []
    for document in documents:
        if re.search(r"</?(?:summary|details|strong)\b", document["markdown"], re.IGNORECASE):
            html_formatting_leaks.append(document["id"])
        if NOTEBOOK_ROUTE_RE.search(document["markdown"]):
            notebook_leaks.append(document["id"])
        for heading in H2_RE.findall(document["markdown"]):
            if PRIVATE_H2_RE.search(heading):
                private_headings.append({"id": document["id"], "heading": heading})
        if document["markdown"].count("$$") % 2 or re.search(
            r"(?m)^\s{4,}.*\$\$|(?<!\\)\\\(.+?(?<!\\)\\\)",
            document["markdown"],
        ):
            math_delimiter_issues.append(document["id"])
    return {
        "status": "PASS"
        if not notebook_leaks
        and not private_headings
        and not context.missing_assets
        and not math_delimiter_issues
        and not html_formatting_leaks
        else "REVIEW",
        "documents": len(documents),
        "assets": len(context.copied_assets),
        "notebook_route_leaks": notebook_leaks,
        "private_headings": private_headings,
        "missing_assets": context.missing_assets,
        "math_delimiter_issues": math_delimiter_issues,
        "html_formatting_leaks": html_formatting_leaks,
        "removed_notebook_blocks": len(context.removed_notebook_blocks),
        "omitted_private_sections": len(context.omitted_sections),
    }


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    site_root = project_root
    source_roots = dict(DEFAULT_ROOTS)
    for override in args.source_root:
        key, value = override.split("=", 1)
        if key not in source_roots:
            raise ValueError(f"Unknown course: {key}")
        source_roots[key] = Path(value).resolve()
    boundary = {}

    public_root = site_root / "public"
    content_root = public_root / "content"
    assets_root = public_root / "assets"
    safe_reset_generated_dir(content_root, project_root)
    safe_reset_generated_dir(assets_root, project_root)

    context = BuildContext(
        project_root=project_root,
        content_root=content_root,
        assets_root=assets_root,
        source_roots=source_roots,
        missing_assets=[],
        copied_assets={},
        removed_notebook_blocks=[],
        omitted_sections=[],
    )
    documents: list[dict[str, Any]] = []
    for language in ("ru", "en"):
        documents.extend(build_modern(boundary, language, context))
        documents.extend(build_rl(boundary, language, context))
        documents.extend(build_information_theory(boundary, language, context))

    documents.sort(key=lambda item: (item["course_key"], item["language"], item["position"]))
    catalog = course_catalog(documents)
    validation = validate_output(documents, context)
    generated = site_root / "app" / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    client_documents = [
        {key: value for key, value in document.items() if key not in {"markdown", "source_file"}}
        for document in documents
    ]
    (generated / "content.json").write_text(
        json.dumps({"courses": catalog, "documents": client_documents}, ensure_ascii=False),
        encoding="utf-8",
    )
    report = {
        "schema": "course-publication.local-content-build.v2",
        "content_updated": "2026-10-15",
        "source_repositories": {
            key: {"commit": subprocess.check_output([
                "git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), "rev-parse", "HEAD"
            ], text=True).strip()} for key, root in source_roots.items()
        },
        "source_files": {
            document["id"]: {
                "path": document["source_file"],
                "sha256": hashlib.sha256((source_roots[document["course_key"]] / document["source_file"]).read_bytes()).hexdigest(),
            } for document in documents
        },
        "validation": validation,
        "courses": catalog,
        "omitted_sections": context.omitted_sections,
    }
    (project_root / "CONTENT_BUILD.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0 if validation["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
