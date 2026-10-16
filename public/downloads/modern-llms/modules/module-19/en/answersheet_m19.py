"""Verify and display the shared M19 numerical contract."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reference_calculations import build_reference_payload,REFERENCE_PATH
if __name__=="__main__":
    data=build_reference_payload()
    assert data==json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    print("Planning assumptions, paired inference, and conditional compute accounting:")
    print(json.dumps(data,indent=2))
