#!/usr/bin/env python3
"""Validate or intentionally regenerate the Module 3 frozen reference payload."""
from pathlib import Path
import sys

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))
from reference_calculations import main

if __name__ == "__main__":
    raise SystemExit(main())
