# ragbench-lite

A lightweight RAG evaluation harness: markdown corpus ingestion, BM25
retrieval, answer generation through [LiteLLM](https://docs.litellm.ai), and
automated scoring of retrieval recall and answer quality (faithfulness +
relevance) with an LLM-as-judge, rendering a one-page HTML report.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install langchain-text-splitters litellm rank_bm25 jinja2
export OPENROUTER_API_KEY=sk-...   # or any provider LiteLLM supports
make demo                          # ingest -> evaluate -> report.html
```

Switch the answering/judging model with `RAGBENCH_MODEL` and
`RAGBENCH_JUDGE_MODEL` (any LiteLLM model string).

## Layout

- `ragbench/ingest.py` — chunk markdown corpora with LangChain's RecursiveCharacterTextSplitter
- `ragbench/retrieve.py` — BM25 (rank_bm25) index + top-k search
- `ragbench/answer.py` — grounded answer generation via LiteLLM
- `ragbench/evaluate.py` — recall@k, faithfulness and relevance with an LLM judge
- `ragbench/report.py` — self-contained HTML report
- `eval/questions.jsonl` — golden questions with reference answers and gold source docs

## Metrics

| metric | how |
| --- | --- |
| recall@k | gold source document present in top-k retrieved chunks |
| faithfulness | judge scores answer claims against retrieved context (0–1) |
| answer relevance | judge scores answer vs reference answer (0–1) |