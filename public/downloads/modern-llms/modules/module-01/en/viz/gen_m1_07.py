"""Build the canonical Module 1 figure."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from build_m1_plots import build
if __name__ == "__main__":
    build("en", 7)
