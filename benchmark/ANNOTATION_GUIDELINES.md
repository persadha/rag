# ILSA-TableQA — Annotation Guidelines

How to author a benchmark item. The goal is **objective, cell-grounded** gold answers
so that ground truth needs no scarce domain experts — the team can label reliably and
inter-rater agreement is high (this is why the "public data + own raters" scope works
for the table spine specifically).

## 0. One-line workflow
1. Pick a public report/exhibit → 2. write a question with a single defensible answer
→ 3. record the exact gold value(s) **and the cell(s) they come from** → 4. set
`answer_kind` / `gold_unit` / `tolerance` → 5. run the linter → 6. second rater checks.

## 1. Choosing exhibits (sampling)
- Use only **public** materials (international reports + encyclopedia exhibits already
  in `datasets/`, plus TIMSS/ICILS public reports).
- Sample **across** `exhibit_type` and `reasoning_type` (see [TAXONOMY.md](TAXONOMY.md)).
  Deliberately over-sample the hard types (`standard_error_read`, `trend_direction`,
  `significance_read`, `aggregation`) — the prior study was 68% single-fact lookups.
- Record `study`, `cycle`, `source_document` (the PDF filename), `source_page`,
  `exhibit_id`.

## 2. Writing the question
- It must have **one** defensible answer given the exhibit. Avoid ambiguity ("high" →
  say "the High International Benchmark").
- Name the entity and column precisely ("girls", "2021", "the High Benchmark").
- Prefer questions a real secondary analyst would ask.

## 3. Recording the gold answer
- `gold_answer`: the natural human-readable answer (used for text/boolean/direction).
- For numeric items, also fill `gold_values` (list of the target number(s)) and
  `gold_unit`. The scorer reads numbers from a system's free text, so the *value* is
  what's graded, not the phrasing.
- `gold_cells`: where the value comes from — a header-path or coordinate, e.g.
  `"row: Malta | col: High Benchmark %"` or `["Girls Avg", "Boys Avg", "Difference"]`.
  This makes the item auditable and supports the table-structure metric later.
- `reference_context`: paste the serialized table region (or caption) that contains
  the answer — the gold evidence, analogous to the prior dataset's `reference_context`.

## 4. Notation rules (match ILSA conventions)
- **Standard errors** appear in parentheses next to an estimate: `520 (2.3)`. For a
  `standard_error_read` item, `gold_values=[2.3]`, `gold_unit="se"`.
- **Significance markers** (▲ ▼ ↑ ↓ * † ‡) are stripped by the scorer; capture their
  *meaning* in the answer (`direction` = up/down/flat, `boolean` = significant/not).
- **Percentages**: `gold_unit="percent"`, value without the `%` sign (`50.0`).
- **Thousands separators / unicode minus** are handled by the parser — write the gold
  value plainly (`1234.5`, `-3.1`).

## 5. Tolerance policy
- Leave `tolerance: null` to use the unit default (see TAXONOMY.md §4) — recommended.
- Override only with reason (e.g. a value reported to 2 dp → `{"kind":"abs","value":0.005}`).
- Counts/years are exact (tolerance 0).

## 6. Metric-design decisions to be aware of (document, don't fight)
- `numeric_single`/`numeric_multi` are scored by **F1 over extracted numbers**, so an
  answer padded with *extra, unrelated numbers* is penalised on precision. This is
  deliberate — it is what stops a verbose chunk-dump from scoring 1.0 (the "coverage
  artifact" the prior study hit). **Consequence:** for a `standard_error_read` item,
  an answer that also states the estimate ("the average is 520 and the SE is 2.3")
  scores <1.0 because 520 is a spurious number for *that* question. If you want such
  contextual answers to score full marks, either (a) make the item `numeric_multi`
  with both numbers as gold, or (b) treat that reasoning_type as recall-based in the
  harness. Decide this once, per reasoning_type, and record it.

## 7. Quality control
- Run the linter before committing:
  `python benchmark/score_benchmark.py --items <file>.jsonl --validate`
- **Double annotation**: a second rater independently re-derives `gold_values` from
  the source PDF for a random ~15% subset; report exact-agreement as the inter-rater
  reliability (IRR). Because answers are objective cell reads, expect very high IRR;
  investigate any disagreement (usually an ambiguous question — fix the question).
- Never invent a value. If the cell is unreadable from the PDF text layer, that item
  is itself evidence for the vision-vs-text comparison — note it, don't fabricate.

## 8. Item file format
JSON Lines, one `TableQAItem` per line (see [schema.py](schema.py) and
[items.example.jsonl](items.example.jsonl)). Lines starting with `//` are comments.
Generate a flat CSV template with:
`python benchmark/score_benchmark.py --emit-template benchmark/template.csv`
