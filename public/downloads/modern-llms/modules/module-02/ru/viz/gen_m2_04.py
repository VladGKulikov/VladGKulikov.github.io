"""Compatibility entry point: canonical bilingual chart generator."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from build_m2_plots import build
if __name__ == '__main__':
    build('ru', 4)
