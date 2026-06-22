# Response to plans.md — all six items

> Status as of 2026-06-12, branch `RAG-2`. Anything that depends on the full r3 evaluation runs
> (deferred to a dedicated session) is explicitly marked **PENDING r3** — no numbers are invented.
> Living technical reference: `context.md`; stage tracker: `context.md` §10.

## 1. CRAG limitations → new 4th architecture (CRAG preserved)

**Done.** CRAG is frozen at the design that produced the report's l2/g2/d2 numbers
(`docs/adr/0001-crag-frozen-at-evaluated-design.md`): sub-questions reuse the original query's
documents, ≤100-word synthesis cap, gemma3:1b doc-grader. Only crash-safety fixes remain
(grader fallback, 2-attempt loop guard, `<think>` stripping, unified state).

The new **CRAG++** (`src/graph_builder/graph_builder_cragpp.py`, `src/nodes/cragpp_nodes.py`,
`src/state/cragpp_state.py`) keeps CRAG's full skeleton and fixes the three diagnosed
limitations, each mapped to a finding:

| CRAG limitation (diagnosed) | CRAG++ change |
|---|---|
| Decompose-but-reuse: sub-questions add LLM calls but zero retrieval effect | Per-sub-question retrieval (the Auto architecture's core strength), with the same grade+fallback per sub-question |
| Identical/duplicated chunks fed repeatedly into generation | Dedup within each sub-question and in the `final_contexts` union (`src/utils/docs.py`) |
| ≤100-word synthesis cap truncated exactly the facts short gold answers need | Word cap removed |
| `retrieved_context` logged as one blob → zeroed contextual precision/recall (r2 artifact) | State exposes `final_contexts` as a chunk list; the harness logs lists for every system |

Behavioral contract proven by `tests/smoke_cragpp.py` (retrieval counts, dedup, no word cap,
loop guard, fallback). Pilot anecdote (not a metric): on the Abu Dhabi schools question, CRAG
answered "225 private and charter schools" (wrongly merged); CRAG++ answered "203 private + 22
charter" — correct, because the counting sub-question retrieved its own context.
**Measured CRAG→CRAG++ deltas (r3, n=195, llama3:8b, gpt-oss-120b judge):** CRAG++ did **not**
improve on CRAG — it regressed. Ctx Precision 0.690→**0.449**, Ctx Recall 0.671→0.650,
Answer Correctness 0.474→**0.334**, Faithfulness 0.948→0.910 (Gold-Ctx Sim ~flat 0.717→0.712).
The per-sub-question retrieval + dedup widened context at the cost of precision, and answer quality
fell with it. The pilot anecdote above did not generalize. See `context.md` §2.0.

## 2. Why the RAGAS evaluation could not be completed

**Diagnosed with notebook-level evidence** — see `reports/ragas_diagnosis.md`. In three lines:
RAGAS did not fail wholesale (the December standard-RAG runs completed with a paid gpt-4o-mini
judge); the January advanced-RAG runs died in a cascade — OpenAI quota 429s mid-run, then
local-Ollama judges colliding with RAGAS's structured-output machinery (600 s executor
timeouts, `fix_output_format` parse failures → silent NaN), then NaN answers from CRAG's
dead-end bug crashing Pydantic validation — aggravated by unpinned ragas 0.4.1/0.4.2 with two
incompatible metric APIs. DeepEval succeeded because it tolerates a local judge (sequential
calls) and skips bad rows instead of crashing.

## 3. Re-run evaluations: Standard vs the new architecture (DeepEval)

**Design (agreed):** 3 systems (Standard, CRAG, CRAG++) × 2 generators × 300 rows, judge fixed
across all runs; metrics = ContextualPrecision, ContextualRecall, Faithfulness (r2-comparable)
+ an answer-correctness GEval (the answer-quality signal r2 lacked). CRAG is included because
the old r2 numbers are artifact-contaminated — without re-running it, "CRAG++ improves on CRAG"
can't be substantiated.

**Proprietary recommendation — Anthropic `claude-haiku-4-5`.** Strong grounded-QA
faithfulness and instruction-following at the lowest cost/latency tier; CRAG++ makes ~10+ LLM
calls per question, which Haiku keeps affordable (~$15 for all proprietary generation).
Choosing Anthropic for generation lets OpenAI **`gpt-4.1` be the fixed judge** — cross-vendor,
so no self-preference bias, and DeepEval's best-validated judge path. (If the judge budget
matters: `gpt-4.1-mini` cuts judging from ~$80–90 to ~$20 with modest reliability loss.)

**Open-source recommendation — Llama 3.1 8B Instruct.** The Llama family was the research's
workhorse and has no `<think>`-tag pollution (DeepSeek-R1) and no 1B-grader weakness (Gemma's
small variants). Served via **Groq** (`llama-3.1-8b-instant`, OpenAI-compatible API, ~$1–2
total) it runs from this CPU-only laptop — or fully offline via the local-Ollama path below.

**Status:** infrastructure complete and pilot-validated (`scripts/run_generation.py`,
`scripts/run_eval.py`, resumable, chunk-list logging). Key-less pilot on local Ollama
(llama3:8b + gemma3:1b): 35 rows across the three systems, **0 empty answers**, all contexts
logged as chunk lists, all 4 metrics numeric through an Ollama judge. Operational latencies
(CPU): 18 s/row Standard, 67 s CRAG, 98 s CRAG++.
**Full-run scores (r3, done 2026-06-20).** Rescoped to **open-source end-to-end** on a revised
195-row dataset: local Ollama **llama3:8b** generation (3 systems × 195, 0 empty answers), judge
**`gpt-oss-120b` via DeepInfra** (Groq's pay-per-token tier was waitlisted; the `compat:` judge in
`run_eval.py` works with any OpenAI-compatible host). A 5th, judge-free **Gold-Context Similarity**
metric (cosine vs the dataset's gold `reference_context`) was added.

| System | Ctx Precision | Ctx Recall | Faithfulness | Answer Correctness | Gold-Ctx Sim |
|---|---|---|---|---|---|
| standard | **0.741** | **0.796** | 0.939 | **0.473** | **0.741** |
| crag | 0.690 | 0.671 | **0.948** | **0.474** | 0.717 |
| cragpp | 0.449 | 0.650 | 0.910 | 0.334 | 0.712 |

**Standard wins or ties on every metric; CRAG ≈ Standard on answer quality; CRAG++ regresses.**
2,925 cells scored, 1 null (99.97%). Per-data_type split and artifacts in `results/eval_r3_summary.xlsx`
+ `results/eval_r3_*_deepeval.csv`. Full discussion: `context.md` §2.0.

## 4. GPU access + retrieval-quality improvements

**GPU: you don't need one for this evaluation.** With API generators (item 3) and an API
judge, every remaining step — embeddings (mpnet), Chroma, the harness — runs on the CPU
laptop; the index rebuild took ~13 min. For future *local-model* work, in order of preference:

1. **Hosted open-weight inference** (Groq, Together, OpenRouter) — pay-per-token, zero setup,
   the same OpenAI-compatible client already wired into `config_api.py`. Almost always beats
   renting a GPU for sub-70B models.
2. **Google Colab Pro/Pro+** — your notebooks and Drive corpus already live there; A100/L4
   on demand; session limits are the known pain (they contributed to the RAGAS timeouts).
3. **GPU VM rental** (RunPod, Lambda, Lightning.ai) — for long unattended runs (e.g., serving
   gpt-oss:20b as a judge); ~$0.3–2/h, you manage the environment.
4. **Local workstation GPU** — only worth it if local-model experiments become routine.

**Retrieval quality, ranked by expected impact per effort** (current baseline: mpnet 768-dim,
1000/100 chunks, top-4 dense retrieval):

1. **Embedding upgrade — done.** Index rebuilt with `all-mpnet-base-v2` (replacing MiniLM-384).
   Pilot retrieval looked sharp (gold chunk in top-4 on spot checks). **r3 (mpnet) retrieval:**
   Ctx Precision 0.741 / Ctx Recall 0.796 on Standard, and a direct Gold-Context Similarity of
   **0.741** (max cosine of retrieved chunks vs the gold passage). A *clean* MiniLM-vs-mpnet A/B
   isn't available (r2 used MiniLM **and** a different dataset/judge), so this isn't an isolated
   embedding delta — but mpnet retrieval is healthy and is no longer the limiting factor; precision
   *use* (architecture) is. See `context.md` §2.0.
2. **Hybrid retrieval (BM25 + dense).** PIRLS questions are entity-heavy (country names,
   programme acronyms like NAPLAN) — exactly where lexical search beats embeddings.
   `rank_bm25` is already in requirements; fuse with reciprocal-rank fusion, keep top-4.
3. **Cross-encoder reranking.** Retrieve top-20 dense, rerank with `flashrank` (already in
   requirements, CPU-fast) to top-4. Replaces the binary 1B doc-grader's role without its
   catastrophic all-rejected failure mode (graded out the answer-bearing doc — 19% vs 56%
   gold-overlap on zero-precision rows).
4. **Metadata enrichment.** Each chunk currently carries only `source`/`page`. Prepend a
   header to the embedded text (`[United States — PIRLS 2021 Encyclopedia, p.3]`) and store
   country/doc-type in metadata for filtered retrieval when the question names a country
   (~most of the dataset). Cheap and likely a large precision win on cross-country confusions.
5. **Chunking experiments.** 1000/100 chars is coarse for 21-word median answers; try 512/64
   with a parent-document retriever (small chunks for matching, parents for context).
6. **Query expansion** (multi-query / HyDE) — only after 2–4; adds latency and the
   decomposition in CRAG++ already covers part of this.

## 5. UI review

**Done — `reports/ui_review.md`.** The committed `streamlit_app_auto.py` could not start
(calls to two nonexistent methods + an empty URL corpus) and could not answer (session-state
self-deletion every rerun, wrong result key) — 13 issues total, each with file:line and fix.
The app is repaired and extended: loads the persisted index, architecture selector
(Standard / CRAG / CRAG++), local-model selector, and the retrieval inspector below.
Verified end-to-end with all three architectures (`tests/smoke_ui.py`).

## 6. Retrieved-context visualization/logging + production readiness

**Was there a mechanism?** Effectively no. Retrieved context only existed as CSV columns in
eval outputs — and inconsistently (Standard as a 4-chunk list, CRAG as one ~8.7k-char blob,
which zeroed its contextual precision/recall in r2). There was no way to *see* what was
retrieved for a question, and no tracing anywhere in `src/`.

**Now:**
- **Logging:** `scripts/run_generation.py` logs `retrieved_context` as a JSON chunk list for
  every system, plus chunk counts and latencies; resumable row-by-row.
- **Visualization:** the Streamlit app shows, for each question, the top chunks with their
  **vector distance to the question** (the semantic-proximity comparison you asked for) and,
  separately, the contexts the selected architecture *actually used* (post-grading,
  post-dedup) — making grader filtering and per-sub-question retrieval directly inspectable.

**Production-readiness recommendations** (ordered; first three are prerequisites, the rest
scale):
1. **Pin and isolate the environment** — started (`requirements-eval.txt`, `.venv`); unpinned
   deps already cost one evaluation framework (item 2).
2. **Real observability instead of `print`** — per-node structured tracing (LangSmith,
   Langfuse, or plain OpenTelemetry): retrieval scores, grader verdicts, attempt counts,
   latencies, token usage. Every diagnosis in this project required archaeology that tracing
   would have made trivial.
3. **Eval in CI** — the smoke tests (`tests/`) on every MR, plus a scheduled 10–15-row pilot
   eval as a quality-regression gate; promote prompts/models only when the pilot holds.
4. **Service decomposition** — ingestion/indexing job, a retrieval+generation API (FastAPI),
   and the UI as a thin client; the graph builders are already cleanly separable.
5. **Managed vector store** — local Chroma files don't survive concurrent access or
   deployments; move to pgvector/Qdrant/Weaviate with versioned, rebuildable indexes
   (`scripts/build_index.py` is the seed of that job).
6. **Resilience around model calls** — timeouts, retries with backoff (the API configs already
   set `max_retries=5`), rate-limit pacing, and graceful degradation (fall back to the
   Standard graph when graders/decomposition fail).
7. **Caching** — embedding cache (corpus is static) and an answer cache keyed on
   (question, architecture, model) for repeated queries.

---

*r3 eval session complete (2026-06-20): measured-results slots in items 1, 3, and 4.1 are filled,
and `context.md` §2.0 carries the r3 tables. Headline: Standard remains the strongest architecture,
CRAG++ regressed against its design hypothesis, and retrieval-vs-gold is healthy (~0.74) — so the
remaining lever is precision/use of context (reranking, hybrid retrieval, metadata filtering per
4.2/4.3), not raw retrieval recall.*
