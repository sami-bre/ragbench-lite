"""BM25 retrieval over ingested chunks (no external vector store needed)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, chunks_path: Path):
        self.chunks = [json.loads(line) for line in chunks_path.read_text().splitlines() if line.strip()]
        corpus = [tokenize(c["text"]) for c in self.chunks]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int = 5) -> list[dict]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i] for i in ranked]