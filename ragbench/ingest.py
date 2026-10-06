"""Document ingestion for ragbench-lite.

Walks markdown corpora, splits them into retrievable chunks with
LangChain's RecursiveCharacterTextSplitter, and writes chunks.jsonl.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120


def ingest(roots: list[str], out_path: Path) -> int:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )
    n = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for root in roots:
            rootp = Path(root).expanduser()
            for md in sorted(rootp.rglob("*.md")):
                text = md.read_text(encoding="utf-8", errors="replace")
                for chunk in splitter.split_text(text):
                    if len(chunk.strip()) < 40:
                        continue
                    rec = {
                        "id": n,
                        "source": str(md),
                        "heading": "",
                        "text": chunk.strip(),
                    }
                    fh.write(json.dumps(rec) + "\n")
                    n += 1
    return n


if __name__ == "__main__":
    roots = sys.argv[1:-1] if len(sys.argv) > 2 else sys.argv[1:]
    out = Path(sys.argv[-1]) if len(sys.argv) > 2 else Path(".ragbench/chunks.jsonl")
    count = ingest(roots, out)
    print(f"ingested {count} chunks -> {out}")