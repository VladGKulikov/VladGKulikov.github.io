"""Rebuild canonical M16 figures."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from build_m16_visuals import build
if __name__ == "__main__": build("ru")
