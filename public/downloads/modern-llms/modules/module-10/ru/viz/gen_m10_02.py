"""Compatibility entry point: rebuild the shared bilingual plot definition."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from build_m10_plots import build
if __name__=="__main__": build(Path(__file__).resolve().parents[1].name)
