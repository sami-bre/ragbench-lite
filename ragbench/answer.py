"""RAG answer generation via LiteLLM (any provider, one env var to switch)."""
from __future__ import annotations

import os
from pathlib import Path

from litellm import completion

DEFAULT_MODEL = "openrouter/google/gemini-3-flash-preview"

SYSTEM = (
    "You are a precise technical assistant. Answer ONLY from the provided "
    "context excerpts. If the context is insufficient, say so explicitly. "
    "Be concise (max 6 sentences) and concrete."
)


def answer(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i + 1}] ({Path(c['source']).name}) {c['text']}" for i, c in enumerate(chunks)
    )
    resp = completion(
        model=os.environ.get("RAGBENCH_MODEL", DEFAULT_MODEL),
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.0,
        max_tokens=400,
    )
    return (resp.choices[0].message.content or "").strip()