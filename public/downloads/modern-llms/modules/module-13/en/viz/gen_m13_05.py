"""Compatibility entry point for the shared M13 visual builder."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from build_m13_visuals import build
build('en')
