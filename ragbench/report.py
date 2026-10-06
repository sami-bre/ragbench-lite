"""Render evaluation results to a one-page HTML report (jinja2, no JS deps)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from jinja2 import Template

TEMPLATE = Template("""\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ragbench-lite report</title>
<style>
  body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 2rem auto; max-width: 900px; color: #1a1a2e; }
  h1 { font-size: 1.5rem; } .sub { color: #666; margin-bottom: 1.5rem; }
  .cards { display: flex; gap: 1rem; flex-wrap: wrap; }
  .card { border: 1px solid #e2e2ea; border-radius: 10px; padding: 1rem 1.4rem; min-width: 160px; }
  .card .v { font-size: 1.9rem; font-weight: 700; }
  .card .l { color: #666; font-size: .85rem; }
  table { border-collapse: collapse; width: 100%; margin-top: 2rem; font-size: .88rem; }
  th, td { text-align: left; padding: .5rem .7rem; border-bottom: 1px solid #ececf2; vertical-align: top; }
  th { color: #666; font-weight: 600; }
  .ok { color: #0a7d33; font-weight: 600; } .miss { color: #c0392b; font-weight: 600; }
  details { margin-top: .4rem; } summary { cursor: pointer; color: #3b5bdb; }
  pre { white-space: pre-wrap; background: #f7f7fb; padding: .6rem; border-radius: 6px; }
</style>
</head>
<body>
<h1>ragbench-lite evaluation report</h1>
<p class="sub">{{ s.questions }} golden questions &middot; BM25 top-{{ s.top_k }} retrieval &middot;
answer model <code>{{ s.answer_model }}</code> &middot; judge <code>{{ s.judge_model }}</code></p>
<div class="cards">
  <div class="card"><div class="v">{{ "%.0f"|format(s.recall_at_k * 100) }}%</div><div class="l">recall@{{ s.top_k }}</div></div>
  <div class="card"><div class="v">{{ "%.2f"|format(s.faithfulness) }}</div><div class="l">faithfulness (LLM judge)</div></div>
  <div class="card"><div class="v">{{ "%.2f"|format(s.relevance) }}</div><div class="l">answer relevance (LLM judge)</div></div>
</div>
<table>
<tr><th>Question</th><th>Recall</th><th>Faith.</th><th>Relev.</th><th></th></tr>
{% for r in rows %}
<tr>
  <td>{{ r.question }}</td>
  <td><span class="{{ 'ok' if r.recall else 'miss' }}">{{ 'hit' if r.recall else 'MISS' }}</span><br>
      <small>{{ r.retrieved[:3] | join(', ') }}</small></td>
  <td>{{ "%.2f"|format(r.faithfulness) }}</td>
  <td>{{ "%.2f"|format(r.relevance) }}</td>
  <td><details><summary>answer</summary>
      <pre>{{ r.answer }}</pre>
      {% for k, v in r.notes.items() %}<p><small><b>{{ k }}:</b> {{ v.reason }}</small></p>{% endfor %}
  </details></td>
</tr>
{% endfor %}
</table>
</body>
</html>
""")


def report(results_path: Path, out_path: Path) -> None:
    data = json.loads(results_path.read_text())
    out_path.write_text(TEMPLATE.render(s=data["summary"], rows=data["rows"]))
    print(f"report -> {out_path}")


if __name__ == "__main__":
    report(Path(sys.argv[1]), Path(sys.argv[2]))