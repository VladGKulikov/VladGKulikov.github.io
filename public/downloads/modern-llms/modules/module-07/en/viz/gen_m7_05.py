"""Compatibility entry point; canonical M7 plots share one generator."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from build_m7_plots import build
if __name__ == "__main__":
    build('en')
