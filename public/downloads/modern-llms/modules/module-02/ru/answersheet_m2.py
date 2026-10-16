#!/usr/bin/env python3
"""Compatibility entry point for the shared Module-2 answer sheet."""
from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "answersheet_m2.py"),
    run_name="__main__",
)
