"""Entry point for the shared bilingual figure builder."""
from pathlib import Path
import sys
MODULE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODULE))
from build_m19_visuals import build
if __name__ == "__main__":
    build(Path(__file__).resolve().parents[1].name)
