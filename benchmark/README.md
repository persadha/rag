# ILSA-TableQA — benchmark scaffolding (proposal-stage)

Scaffolding for the proposed **IEA R&D Call 6** follow-on: *table-aware retrieval over
ILSA statistical reports*. It provides (1) the item schema + annotation frame for a new
QA benchmark over statistical exhibits in public ILSA reports, and (2) a working
**numeric-exact** scorer that answers the prior study's biggest evaluation weakness.

This is a **de-risking artifact**, not the benchmark itself: it makes the proposal's
"we can already measure this" claim concrete. The example items carry illustrative,
*unverified* placeholder values (annotator `EXAMPLE`); real gold values are authored in
WP1. See the plan file for how this maps to the proposal.

## Why it exists
The prior PIRLS RAG study could not read tables (text extraction flattens row/column
structure) and its answer-correctness metric was gameable: a 4,000-char chunk dump
scored **0.827** because the metric rewarded fact *coverage* and ignored verbosity
(`reports/reviewer_response_r3.md`, R2-2). For statistical tables that is unacceptable —
a reported percentage or standard error must match to the reported precision. The
scorer here grades **numeric exactness with tolerance**, which cannot be gamed by
padding, and is deterministic and free (no LLM judge).

## What's real vs stub
- **Real, tested:** the numeric-exact-with-tolerance scorer and the answer-kind
  dispatch (`scorer.py`), the schema + validator (`schema.py`), and the CLI harness
  (`score_benchmark.py`). 26 assertions in `smoke_scorer.py` cover ILSA notation
  (SEs in parentheses, significance markers, thousands separators), tolerance, multi-
  value F1, and the anti-coverage-artifact behaviour.
- **Clearly-labelled stub:** table-structure recovery (`grits_content`,
  `grits_topology`, `teds_struct`). A pure-Python grid approximation (GriTS-lite) runs
  today; production should swap in the reference GriTS (arXiv:2203.12555) / TEDS —
  `teds_struct` already uses `apted`+`lxml` if installed.

## Files
| File | Purpose |
|---|---|
| `schema.py` | `TableQAItem` dataclass, controlled vocabularies, `validate()`, JSONL IO |
| `scorer.py` | numeric-exact-with-tolerance (real) + structure recovery (stub) |
| `score_benchmark.py` | CLI: `--validate`, `--emit-template`, and score a system's answers |
| `TAXONOMY.md` | exhibit / reasoning / answer-kind vocabularies (mirrors `schema.py`) |
| `ANNOTATION_GUIDELINES.md` | how to author objective, cell-grounded gold items + IRR |
| `items.example.jsonl` | 7 illustrative items (placeholder values) spanning the taxonomy |
| `predictions.example.csv` | a demo system's answers (some right, some wrong) |
| `smoke_scorer.py` | 26-assertion smoke test |
| `docling_adapter.py` | real exhibit PDF -> table Grid (text-extraction paradigm), feeds the structure metric. **Optional dep: Docling** |
| `gold_exhibit1.example.json` | hand-authored gold grid (header + 6 rows of PIRLS Exhibit 1) for the adapter demo |

## Quickstart
```bash
# 1. verify the scorer works
python benchmark/smoke_scorer.py

# 2. lint an item set
python benchmark/score_benchmark.py --items benchmark/items.example.jsonl --validate

# 3. write a blank CSV annotation template
python benchmark/score_benchmark.py --emit-template benchmark/template.csv

# 4. score a system's answers (id -> answer CSV/JSONL), broken down by type
python benchmark/score_benchmark.py \
    --items benchmark/items.example.jsonl \
    --predictions benchmark/predictions.example.csv
```
The demo (step 4) reports a mean score of 0.667 / exact-match 0.571 over the 7 example
items — two are wrong on purpose (a wrong % and a wrong trend direction) to show the
metric discriminating.

### Real input: the Docling adapter (optional)
`docling_adapter.py` runs the *text-extraction paradigm* (layout-aware parsing) on a
real exhibit PDF and feeds the extracted table into the structure metric. Docling is a
heavy optional dependency (torch + layout/TableFormer models), deliberately kept out of
`requirements-eval.txt`:
```bash
.venv\Scripts\pip install docling
python benchmark/docling_adapter.py \
    --pdf "datasets/Exhibit 1 Years of Schooling.pdf" \
    --gold benchmark/gold_exhibit1.example.json
```
Verified on **PIRLS Exhibit 1 (Years of Schooling)**: Docling detected the 2 page-split
tables (47×3 and 19×3) and reproduced the 3-column structure. Scored against the
independently hand-authored gold (header + 6 country rows), the **region-aligned**
GriTS-lite score is **content 1.000 / topology 1.000** — Docling extracted that region
perfectly.

Two lessons this surfaced, both now handled:
- The naïve **full-table vs partial-gold** score is low (0.261/0.149) purely from size
  mismatch; the adapter therefore also prints a region-aligned score and warns. **Author
  gold at full-table granularity** (or region-align) for a fair number in WP1.
- The console preview truncates long cells (shown with `...`); the JSON output
  (`--out`) is the source of truth. Console is reconfigured to UTF-8 so non-ASCII
  country names (e.g. "Türkiye") don't mojibake on Windows.

## Design decision worth knowing
`numeric_single` is scored by **F1 over the numbers found in the answer**, so extra,
unrelated numbers cost precision — deliberately, to defeat the coverage artifact. One
consequence: an SE-read answer that also restates the estimate ("the average is 520 and
the SE is 2.3") scores <1.0. Choose per `reasoning_type` whether to keep F1 (strict) or
switch to recall-based, and record the choice. See `ANNOTATION_GUIDELINES.md` §6.

## Integration with the existing repo
- Dataset shape generalises the existing `datasets/revision/evaluation_dataset.xlsx`
  (`id / data_type / user_query / reference_answer / reference_context`) by adding the
  table-specific fields (exhibit type, gold cells, numeric targets, tolerance).
- The harness mirrors `scripts/run_eval.py` conventions (resumable CSV append, grouped
  summary) but needs **no judge** — correctness is numeric-exact.
- Pure-Python; no additions to `requirements-eval.txt`.
