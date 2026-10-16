#!/usr/bin/env python3
"""Backward-compatible entry point for the deterministic Module 3 builder.

New maintainer code should import :mod:`build_notebook_transformer_shared` directly.
This shim preserves the old ``build(lang, output)`` API without duplicating notebook
source or generation logic.
"""
from __future__ import annotations

from pathlib import Path

import nbformat

from build_notebook_transformer_shared import build_notebook, main as _shared_main


def build(lang: str, output: Path) -> None:
    """Build one language notebook at an explicit path."""

    notebook = build_notebook(lang)
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)


def main() -> int:
    """Build both canonical language notebooks."""

    return _shared_main()


if __name__ == "__main__":
    raise SystemExit(main())
