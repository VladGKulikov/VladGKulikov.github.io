"""Inject deterministic self-contained bridge cells into generated learner notebooks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .notebook_cells import bridge_notebook_cells


def _source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def _cell_id(kind: str, source: str) -> str:
    return hashlib.sha1(f"bridge:{kind}:{source}".encode("utf-8")).hexdigest()[:12]


def _cell(kind: str, source: str) -> dict:
    data = {"cell_type": kind, "id": _cell_id(kind, source), "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        data.update({"execution_count": None, "outputs": []})
    return data


def inject_bridge_cells(path: Path, bridge_ids: list[str], lang: str) -> Path:
    """Replace prior bridge cells and insert the current generated cells before exercises."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    cleaned: list[dict] = []
    skip_code = False
    for cell in cells:
        source = _source(cell)
        if cell.get("cell_type") == "markdown" and source.startswith("## PyTorch bridge · B"):
            skip_code = True
            continue
        if skip_code and cell.get("cell_type") == "code":
            skip_code = False
            continue
        skip_code = False
        cleaned.append(cell)

    insert_at = len(cleaned)
    for index, cell in enumerate(cleaned):
        source = _source(cell).lstrip()
        if source.startswith("##") and ("Упражнен" in source[:100] or "Exercise" in source[:100]):
            insert_at = index
            break
    generated = [_cell(kind, source) for kind, source in bridge_notebook_cells(bridge_ids, lang)]
    notebook["cells"] = cleaned[:insert_at] + generated + cleaned[insert_at:]
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path
