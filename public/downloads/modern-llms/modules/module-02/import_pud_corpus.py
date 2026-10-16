"""Rebuild the M2 text fixture from pinned UD PUD 2.15 sources.

Network access is explicit (--download). By default, verify/rebuild derived splits
from the committed aligned text. Dependency annotations are not used or distributed.
The upstream parsing test set is reused for a teaching tokenizer experiment only;
these splits are not an official UD benchmark protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

DATA = Path(__file__).resolve().parent / "data"
SOURCES = {
    "ru": ("UD_Russian-PUD", "7284f9241af8755cbecdc6f6ce3eaf338f0503c4",
           "b2c116faa1295dbfee6461c9d2c6911257eed01e2d688f7d74113f7a53f696c6"),
    "en": ("UD_English-PUD", "6e328e8195c04a297c1e375d21fe663be1ce6178",
           "63eb7eca373bd45912c01f15f1e445d5cdd41245f20da1669474728e484e096d"),
}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def parse_conllu(raw: bytes) -> list[dict]:
    records = []
    document = None
    for block in raw.decode("utf-8").strip().split("\n\n"):
        fields = {}
        for line in block.splitlines():
            if line.startswith("# ") and " = " in line:
                key, value = line[2:].split(" = ", 1)
                fields[key] = value
        document = fields.get("newdoc id", document)
        if not document or "sent_id" not in fields or "text" not in fields:
            raise ValueError("Missing document, sentence ID, or text")
        records.append({"document": document, "id": fields["sent_id"], "text": fields["text"]})
    return records


def is_eval(document: str) -> bool:
    # A whole source document stays in one partition, in both languages.
    return int(digest(document.encode("ascii"))[:8], 16) % 5 == 0


def build(download: bool = False, source_dir: Path | None = None) -> dict:
    aligned_path = DATA / "pud_parallel.jsonl"
    if download or source_dir:
        records = {}
        for lang, (repo, revision, expected) in SOURCES.items():
            base = f"https://raw.githubusercontent.com/UniversalDependencies/{repo}/{revision}"
            name = f"{lang}_pud-ud-test.conllu"
            def read(name: str) -> bytes:
                return ((source_dir / f"{lang}_{name}").read_bytes() if source_dir
                        else urlopen(f"{base}/{name}", timeout=60).read())
            raw = read(name)
            if digest(raw) != expected:
                raise ValueError(f"Upstream checksum mismatch: {lang}")
            records[lang] = parse_conllu(raw)
            (DATA / f"PUD_{lang.upper()}_README.md").write_bytes(read("README.md"))
            (DATA / f"PUD_{lang.upper()}_LICENSE.txt").write_bytes(read("LICENSE.txt"))
        assert len(records["ru"]) == len(records["en"]) == 1000
        rows = []
        for ru, en in zip(records["ru"], records["en"]):
            assert (ru["id"], ru["document"]) == (en["id"], en["document"])
            rows.append({"id": ru["id"], "document": ru["document"],
                         "split": "eval" if is_eval(ru["document"]) else "train",
                         "ru": ru["text"], "en": en["text"]})
        aligned_path.write_bytes(("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8"))
    rows = [json.loads(line) for line in aligned_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len({r["id"] for r in rows}) == 1000
    assert all(r["split"] == ("eval" if is_eval(r["document"]) else "train") for r in rows)
    files = {}
    for split, lang, name in [("train", "ru", "m2_train_ru.txt"),
                              ("eval", "ru", "m2_eval_ru.txt"),
                              ("eval", "en", "m2_eval_en.txt"),
                              ("train", "en", "m2_train_en.txt")]:
        selected = [r for r in rows if r["split"] == split]
        raw = ("\n".join(r[lang] for r in selected) + "\n").encode("utf-8")
        (DATA / name).write_bytes(raw)
        files[name] = {"sha256": digest(raw), "bytes": len(raw), "sentences": len(selected),
                       "documents": len({r["document"] for r in selected})}
    manifest = {
        "name": "Parallel Universal Dependencies (PUD)", "release": "UD 2.15 (r2.15)",
        "license": "CC BY-SA 3.0", "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
        "attribution": "PUD contributors; DFKI translation teams, Google, and the Universal Dependencies community. Full contributor lists and original notices are in PUD_RU_README.md and PUD_EN_README.md.",
        "sources": {lang: {"repository": f"https://github.com/UniversalDependencies/{repo}",
                           "revision": rev, "url": f"https://raw.githubusercontent.com/UniversalDependencies/{repo}/{rev}/{lang}_pud-ud-test.conllu",
                           "sha256": sha} for lang, (repo, rev, sha) in SOURCES.items()},
        "transform": "Extract # text verbatim; retain original order and aligned sentence/document IDs. No Unicode normalization. Join sentences by LF, including a final LF.",
        "split": "eval iff int(SHA256(ASCII(document_id))[:8], 16) % 5 == 0; otherwise train. Same source-document partition in both languages.",
        "scope": "Teaching BBPE only. Upstream labels all sentences as a parsing test set; this is not a parsing benchmark split or evidence of population-level language fairness.",
        "aligned_file": aligned_path.name, "aligned_sha256": digest(aligned_path.read_bytes()),
        "files": files,
    }
    (DATA / "corpus_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.download, args.source_dir)["files"], indent=2))
