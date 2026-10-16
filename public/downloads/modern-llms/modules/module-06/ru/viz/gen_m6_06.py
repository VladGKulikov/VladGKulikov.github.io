"""Legacy entry point; plot data and style live in the shared M6 builder."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from build_m6_plots import build
if __name__=="__main__": build('ru','06')
