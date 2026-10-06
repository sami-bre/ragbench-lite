"""Evaluation runner: retrieval metrics + LLM-as-judge quality scores.

Metrics:
- recall@k: gold source document appears in the top-k retrieved chunks
- faithfulness: judge scores whether the answer is supported by the retrieved context
- answer relevance: judge scores the answer against a reference answer
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from litellm import completion

from .answer import DEFAULT_MODEL, answer
from .retrieve import BM25Index

JUDGE_MODEL = os.environ.get("RAGBENCH_JUDGE_MODEL", DEFAULT_MODEL)
TOP_K = 5


def judge_json(question: str, answer_text: str, context: str, reference: str) -> dict:
    faith_prompt = (
        "You are an evaluation judge. Score how FAITHFUL the answer is to the "
        "context: 1.0 if every claim is supported by the context, 0.0 if "
        "contradicted or unsupported. Respond with JSON only: "
        '{"score": <float>, "reason": "<one sentence>"}\n\n'
        f"Context:\n{context}\n\nAnswer:\n{answer_text}"
    )
    rel_prompt = (
        "You are an evaluation judge. Score how relevant and complete the "
        "answer is compared to the reference answer: 1.0 = covers the same "
        "key points, 0.0 = wrong or missing the point. Respond with JSON only: "
        '{"score": <float>, "reason": "<one sentence>"}\n\n'
        f"Question: {question}\nReference answer: {reference}\nAnswer: {answer_text}"
    )
    results = {}
    for name, prompt in (("faithfulness", faith_prompt), ("relevance", rel_prompt)):
        for attempt in range(3):
            try:
                resp = completion(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=200,
                )
                raw = resp.choices[0].message.content
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                results[name] = json.loads(m.group(0)) if m else {"score": 0.0, "reason": f"unparseable: {raw[:120]}"}
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    results[name] = {"score": 0.0, "reason": f"judge error: {exc}"}
                else:
                    time.sleep(2 * (attempt + 1))
    return results


def run(questions_path: Path, chunks_path: Path, out_path: Path) -> dict:
    index = BM25Index(chunks_path)
    questions = [
        json.loads(line)
        for line in questions_path.read_text().splitlines()
        if line.strip()
    ]
    rows = []
    for q in questions:
        t0 = time.time()
        chunks = index.search(q["question"], k=TOP_K)
        gold = q["source"]
        recall = any(gold in c["source"] for c in chunks)
        ans = answer(q["question"], chunks)
        context = "\n\n".join(c["text"] for c in chunks)
        scores = judge_json(q["question"], ans, context, q["answer"])
        rows.append({
            "question": q["question"],
            "gold_source": gold,
            "recall": recall,
            "retrieved": [Path(c["source"]).name for c in chunks],
            "answer": ans,
            "faithfulness": scores["faithfulness"]["score"],
            "relevance": scores["relevance"]["score"],
            "notes": scores,
            "latency_s": round(time.time() - t0, 1),
        })
        status = "ok" if recall else "MISS"
        print(f"[{status}] {q['question'][:70]} (f={scores['faithfulness']['score']:.2f} r={scores['relevance']['score']:.2f})")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"rows": rows}, indent=2))
    summary = {
        "questions": len(rows),
        "recall_at_k": sum(r["recall"] for r in rows) / len(rows),
        "faithfulness": sum(r["faithfulness"] for r in rows) / len(rows),
        "relevance": sum(r["relevance"] for r in rows) / len(rows),
        "judge_model": JUDGE_MODEL,
        "answer_model": os.environ.get("RAGBENCH_MODEL", DEFAULT_MODEL),
        "top_k": TOP_K,
    }
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))