#!/usr/bin/env python3
"""Generate Module 12 roofline figure for ru."""
from __future__ import annotations
import sys
from pathlib import Path
MODULE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODULE_ROOT))
from generate_quantitative_figures import generate_roofline  # noqa: E402
if __name__ == "__main__":
    print(generate_roofline("ru"))
