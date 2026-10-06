.PHONY: ingest eval report demo clean

PY = .venv/bin/python
CORPUS = /home/samuel/agent-farm/docs /home/samuel/agent-farm/lessons /home/samuel/lead_ai/klikt_api/docs /home/samuel/lead_ai/docs
INGEST_ARGS = $(CORPUS) .ragbench/chunks.jsonl

ingest:
	$(PY) -m ragbench.ingest $(INGEST_ARGS)

eval: ingest
	$(PY) -m ragbench.evaluate eval/questions.jsonl .ragbench/chunks.jsonl .ragbench/results.json

report:
	$(PY) -m ragbench.report .ragbench/results.json report.html

demo: eval report
	@echo "demo complete: report.html"

clean:
	rm -rf .ragbench report.html