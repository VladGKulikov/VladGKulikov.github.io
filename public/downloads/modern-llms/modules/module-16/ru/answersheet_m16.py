"""Validate and reproduce the canonical Module 16 answer sheet."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reference_calculations import build_reference_payload
if __name__=="__main__":
    reference=build_reference_payload()
    assert reference==json.loads((ROOT/"data/reference_metrics.json").read_text(encoding="utf-8"))
    print(json.dumps(reference,ensure_ascii=False,indent=2))
