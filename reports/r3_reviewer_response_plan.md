# Plan: Address remaining reviewer comments (R1 + R2) within 2 days

## Context

Two reviewers commented on the RD4-5 report *"Information Retrieval Using RAG on
PIRLS Documents."* The r3 round already incorporated several comments (dataset
human/synthetic tagging; chunk tuning 1000/100→512/64; the decomposition-dilutes-
context diagnosis fixed via hybrid+reranker; open-vs-closed via gpt-5.4-mini).

The single biggest **unaddressed** ask — and the one point where **both reviewers
converge** — is *"is the RAG actually worth it?"*:

- **R1 §Missing baseline comparison**: measure performance **with vs. without
  retrieval** to prove RAG beats an off-the-shelf LLM. (Plus R1's closing NAEP
  note: GPT-4o scored >80% correct *without* retrieval.)
- **R2 §Value Added by Generation**: add a **retrieval-only** baseline (top-k
  chunks, no LLM synthesis) to isolate what the generator adds over reading chunks.

These combine into **one cheap 3-condition ablation** that answers both:

| Condition | Generation cost | Judge cost (AC only) |
|---|---|---|
| (a) Closed-book — llama3:8b, no retrieval | local llama, free | ~195 rows |
| (b) Retrieval-only — top-k chunks as the answer (no LLM) | **zero** (reuse logged `retrieved_context`) | ~195 rows |
| (c) Full RAG — best r3 config | already have it | already scored |

Outcome: a defensible "value-added" table per the two reviewers, plus point-by-point
editorial responses to the remaining comments. Total new API spend is small
(answer-correctness judge only, gpt-oss-120b on DeepInfra; closed-book generation is free/local).

### Decisions locked with the user
- **Scope:** experiments **+** write-up.
- **Closed-book model:** **llama3:8b only** (free/local). NB: the result speaks to the
  open 8B model specifically — flag this when responding to R1's GPT-4o framing.
- **Human/expert eval:** **future-work text only** (no grading infrastructure built).

### ⚠️ Coordination constraint
Another session is running **E10 (deepseek-r1, Ollama/GPU), E11 (gpt-5.4-mini,
OpenAI), E12 (llama3:8b, Ollama/GPU)** and may take **E13/E14**. Therefore:
- The closed-book run uses **Ollama llama3:8b → contends for the GPU with E10/E12.**
  Run it only when the GPU is idle (after E12) or accept it queues behind them — do
  **not** run it concurrently with another Ollama job.
- Use **distinct output filenames** (below) and never touch the other session's
  `results/_e1*_watchdog.log` / `run_e1*.py` files.
- The DeepInfra key/budget is **shared** with the other session's judge runs — pace AC scoring so it doesn't starve theirs.
- Confirm experiment numbering with the other session; this plan uses **descriptive
  filename tags** (`noretrieval`, `retrievalonly`) rather than E-numbers to avoid collision.

---

## Part 1 — Value-added ablation (experiments)

### A. Closed-book baseline (no retrieval), llama3:8b
Reuse the existing generation harness; add a no-retrieval path rather than a new script.

- **`scripts/run_generation.py`**: add a `--no-retrieval` flag. When set, skip
  `VectorStore`/retriever construction entirely and answer each question directly
  with the generator via a minimal direct-answer prompt (no context). Log
  `retrieved_context="[]"`, `n_chunks=0`, and a distinct suffix.
  - Reuse: dataset loading + column bridge (lines 96–104), resume-by-`row_id`
    (lines 141–144), the per-row try/except + CSV append (lines 148–184),
    `APIConfig.get_generator("ollama")` (line 130).
  - Output: `results/gen_r3_noretrieval_ollama.csv`.
- **`scripts/run_eval.py`**: add a `--metrics` filter (default = all) so closed-book
  can be scored on **answer_correctness only**. Context precision/recall/faithfulness
  are meaningless (and may error) with empty `retrieval_context`; restricting also
  cuts judge cost. `make_metrics()` (lines 82–97) builds a dict keyed by metric name —
  filter that dict by the requested names.
  - Output: `results/eval_r3_noretrieval_ollama_deepeval.csv`.

### B. Retrieval-only baseline (chunks as the answer, no LLM)
No generation needed — transform the **best-config** completed gen CSV (the 195-row
hybrid+rerank standard/llama3:8b run, the program's best per the eval summary).

- **`scripts/make_retrieval_only.py`** (new, tiny): read a gen CSV, emit a
  gen-shaped CSV where `answer` = the joined `retrieved_context` chunks (e.g.
  `"\n\n".join(chunks)`), preserving `retrieved_context`, `answer_ref`,
  `reference_context`, `data_type`, `row_id`. Keep `system`/`generator` columns
  with a `retrievalonly` marker.
  - Output: `results/gen_r3_retrievalonly_ollama_hybrid.csv`.
- Score with `run_eval.py --metrics answer_correctness` (faithfulness here is
  trivially ~1 since answer==context; precision/recall identical to the source RAG run).
  - Output: `results/eval_r3_retrievalonly_ollama_hybrid_deepeval.csv`.

### C. Assemble the value-added table
Compare **answer_correctness** across (a) closed-book, (b) retrieval-only, (c) full
RAG — overall and split by `data_type` (human vs synthetic). Add rows to
`reports/experiments_index.md` and a small summary (extend the existing
`scripts/summarize_r3.py` pattern, or a one-off table). State explicitly whether
RAG beats closed-book (R1) and whether the LLM beats raw chunks (R2).

---

## Part 2 — Editorial responses (write-up)

Produce **`reports/reviewer_response_r3.md`** — a point-by-point response mapping each
comment to "addressed in r3 / new evidence / acknowledged limitation," and list the
report sections to edit. Most items are free (cite existing results) or one tiny script:

| Reviewer comment | Response | Cost |
|---|---|---|
| R2 §Knowledge Base Scale — # chunks / tokens / footprint | New `scripts/kb_stats.py`: count Chroma docs, estimate tokens, on-disk size | free |
| R2 §Chunking — rationale + sensitivity | Cite the E2 512/64-vs-1000/100 result already on record | free |
| R2 minor — embedding model not described | Describe all-mpnet-base-v2 (768d) vs MiniLM-384; cite any swap result | free |
| R2 §Architecture gap — *which* CRAG stage fails | Consolidate `architecture_evolution_analysis.md` (retrieval, not reasoning: faithfulness ≈0.95 everywhere) into the report; the ablation reinforces it | free |
| R2 minor — human vs synthetic metrics split | Surface existing `eval_r3_summary.xlsx` splits in the report | free |
| R2 minor — characterize 300→195 exclusions | Note r3 uses a clean **195-row revised** dataset; brief characterization | free |
| R2 §Deployment — standalone vs server vs networked | Editorial; tie to gpt-5.4-mini (E11) latency + RTX-4090 hardware caveat | free |
| R1 §2.3 Significance — reframe as motivations/hypotheses | Editorial rewrite | free |
| R1 §Question scope/cognitive level — fact vs why/how | **Optional cheap analysis:** one classification pass tagging the 195 questions by type + domain (gpt-oss-120b, ~195 short calls); report distribution + acknowledge reasoning-question coverage | small |
| R1/R2 — human/expert judge calibration | **Future-work text only** (per decision); strengthen limitations | free |
| R2 §Privacy — validate on internal/non-public docs | Acknowledge as a limitation / future work (no data, not feasible in scope) | free |

---

## Critical files
- `scripts/run_generation.py` — add `--no-retrieval`.
- `scripts/run_eval.py` — add `--metrics` filter.
- `scripts/make_retrieval_only.py` — new transform (retrieval-only).
- `scripts/kb_stats.py` — new (KB scale stats).
- `reports/reviewer_response_r3.md` — new (point-by-point response).
- `reports/experiments_index.md` — add ablation rows.

## Verification
1. **Dry run small**: `python scripts/run_generation.py --system standard --generator ollama --no-retrieval --rows 5` → confirm 5 rows, `n_chunks=0`, non-empty answers.
2. `python scripts/run_eval.py --gen results/gen_r3_noretrieval_ollama.csv --rows 5 --metrics answer_correctness` → confirm AC scores populate, no context-metric errors, judge cost printed.
3. `python scripts/make_retrieval_only.py --gen <best hybrid gen csv>` then `run_eval.py --metrics answer_correctness --rows 5` → confirm chunk-as-answer AC scores.
4. **Full runs** (only when GPU idle / not colliding with E10–E12): full 195-row closed-book + retrieval-only; then build the value-added table and sanity-check the human/synthetic split sums match dataset counts (123 human / 72 synthetic).
5. `kb_stats.py` against `chroma_db/` returns a plausible chunk count / footprint.
6. Eyeball `reviewer_response_r3.md` covers every numbered comment in both docx files.
