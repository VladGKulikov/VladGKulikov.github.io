#!/usr/bin/env python3
"""Validate or explicitly regenerate the frozen Module 9 numerical contract."""
from __future__ import annotations

import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from reference_calculations import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
