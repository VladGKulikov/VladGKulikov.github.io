"""Compatibility entry point for the canonical M18 figures."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from build_m18_visuals import build
if __name__=="__main__":build(Path(__file__).resolve().parents[1].name)
