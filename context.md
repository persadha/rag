# RAG System — Context & Architecture Assessment

> Durable reference for the PIRLS RAG benchmark project. Captures how the system works,
> the evaluation results, the issues found in each RAG architecture, and the prioritized fix roadmap.
> Created as a checkpoint **before** code fixes (committed `src/` has drifted from the
> working Colab runtime; some files may still be syncing).

---

## 0. Terminology (canonical names — updated 2026-06-12)

- **Standard** — `graph_builder.py` / `nodes.py`. Plain retrieve → generate. Eval codes l1/g1/d1.
- **Auto** — `graph_builder_auto.py` / `autorag_nodes.py`. Query decomposition with step-level
  re-retrieval. Never evaluated; kept as-is for reference.
- **CRAG** — `graph_builder_adv.py` / `advrag_nodes.py`. **Frozen at its evaluated design**
  (sub-questions reuse the original query's docs, ≤100-word synthesis cap, gemma3:1b doc-grader)
  with crash-safety fixes only (state unification, grader fallback, loop guard, `<think>`
  stripping). This is the system behind the l2/g2/d2 numbers. Do not add per-sub-question
  retrieval here — see `docs/adr/0001-crag-frozen-at-evaluated-design.md`. (The P1 batch had
  added it in place; reverted on branch RAG-2, see §10.)
- **CRAG++** — `graph_builder_cragpp.py` / `cragpp_nodes.py` / `cragpp_state.py` (4th
  architecture, new). CRAG's skeleton + per-sub-question retrieval, chunk dedup, no word cap.
  All future improvements land here, never in CRAG.

---

## 1. System overview

**Purpose.** A research/R&D **benchmark harness**: run the same question set through
several **RAG architectures × local LLMs**, then score answers with automated metrics to
compare them. Domain = **PIRLS 2021** (Progress in International Reading Literacy Study) —
country "encyclopedia" chapters and statistical exhibits.

**Stack.** Local open models via **Ollama**, orchestrated as **LangGraph** state machines,
run in **Google Colab** (models + data persisted to Google Drive).

**Data (`datasets/`).**
- Corpus: ~70+ PDFs — PIRLS country chapters (UAE, Germany, Canada, Iran, …) + cross-country
  "Exhibit" tables.
- Eval set: `datasets.xlsx` (sheet `final`) — `synthetic` + `human` Q/A pairs, each with a
  reference answer (`answer_ref`) and gold passages (`passage_1_text`, `passage_2_text`) + pages.
- Seed set: `results/Question-Answer Pairs 2.txt`.
- Stores: persisted **Chroma** (`chroma_db/`) and a **FAISS** index (`vectorstore/`).

**Pipeline.**
```
PDFs → DocumentProcessor → VectorStore (Chroma) → retriever → LangGraph (nodes) → answer
```
- `src/document_ingestion/document_processor.py` — `PyPDFLoader` + `RecursiveCharacterTextSplitter` (1000/100).
- `src/vectorstore/vectorstore.py` — `VectorStore` wraps Chroma (create/load/persist/add/retriever).
- `src/config/config.py` (+ `config_llama.py`, `config_gemma.py`) — model/embedding factory.
  Models: `llama3:8b`/`llama3.1:8b`, `gemma3:4b`, `deepseek-r1:8b`, `gpt-oss:20b`;
  small grader `gemma3:1b`; embeddings **HuggingFace `all-MiniLM-L6-v2`** (384-dim). A stronger
  `all-mpnet-base-v2` (768-dim) was later added to the repo but is **not wired in** — see §2.2.

**Three RAG architectures (the thing being compared).**

| Variant | Graph / Nodes / State | Flow |
|---|---|---|
| **Standard** | `graph_builder.py` / `nodes.py` / `rag_state.py` | `retrieve → generate` (one-shot stuff-the-context). |
| **Auto / "agentic"** | `graph_builder_auto.py` / `autorag_nodes.py` / `autorag_state.py` | `planner → sub_planner → retriever → responder → synthesizer` (decompose into 2–3 sub-qs → 3 steps each → retrieve per step → answer → synthesize). |
| **Advanced (CRAG-style)** | `graph_builder_adv.py` / `advrag_nodes.py` / `advrag_state.py` | `retrieve → grade_documents → plan_sub_steps → generate_answers → grade_generation` (self-grades docs + answer, loops on "not useful"). |

There is also `src/nodes/reactnode.py` — a **ReAct** agent (retriever + Wikipedia tools), **orphaned**
(not wired into any graph builder).

**Evaluation (`notebooks/`, `results/`).**
- **DeepEval**: `ContextualPrecision`, `ContextualRecall`, `Faithfulness` — judge = **`gpt-oss:20b`**.
- **RAGAS**: answer correctness, faithfulness, factual correctness, context precision/recall.
- System codes: `l1/g1/d1` = llama/gemma/deepseek + **standard**; `l2/g2/d2` = same + **advanced**.
- Side-by-side results in `results/eval_result_combined_*.xlsx`.

---

## 2. Evaluation results & architecture comparison (as of 2026-01-20)

Six systems = **3 models × 2 architectures** (`l/g/d` = llama3.1 / gemma3 / deepseek-r1 × `1/2` =
standard / advanced), scored on ~195–197 PIRLS questions. All metrics **0–1, higher = better**.
Two evaluation passes exist and disagree on some metrics:
- **Pass 1 — RAGAS** (→ `eval_result_combined_20260118.xlsx` `summary`).
- **Pass 2 — DeepEval "round 2"** (`eval_r2_*`, judge `gpt-oss:20b`, de-duplicated retrieval context) — **the more trustworthy run**.

> ⚠️ **Only 2 of the 3 architectures were ever evaluated.** The scored systems are **Standard** (`1`)
> and **CRAG** (`2` = `graph_builder_adv`). The **Auto** architecture (`graph_builder_auto`) was never
> wired into any notebook and has **no result files** — it is orphaned code (like `reactnode.py`). The
> tables below say nothing about Auto; see **§2.1** for a design-level comparison.

**Pass 1 — RAGAS (n≈197)**

| System | Ctx Precision | Ctx Recall | Faithfulness | Answer Correctness |
|---|---|---|---|---|
| llama / standard | 0.638 | 0.135 | 0.453 | 0.427 |
| gemma / standard | **0.661** | 0.130 | 0.537 | 0.485 |
| deepseek / standard | 0.652 | 0.130 | 0.326 | **0.523** |
| llama / advanced | 0.318 | 0.215 | 0.560 | 0.292 |
| gemma / advanced | 0.328 | 0.091 | 0.412 | 0.289 |
| deepseek / advanced | 0.328 | 0.223 | **0.621** | 0.324 |

**Pass 2 — DeepEval round-2 (n=195; answer correctness from `deepeval_*_ac`)**

| System | Ctx Precision | Ctx Recall | Faithfulness | Answer Correctness |
|---|---|---|---|---|
| llama / standard | 0.618 | 0.539 | 0.978 | 0.408 |
| gemma / standard | **0.644** | **0.564** | 0.974 | 0.427 |
| deepseek / standard | 0.633 | 0.538 | 0.973 | **0.448** |
| llama / advanced | 0.290 | 0.310 | 0.969 | 0.236 |
| gemma / advanced | 0.313 | 0.291 | 0.962 | 0.244 |
| deepseek / advanced | 0.328 | 0.303 | **0.980** | 0.287 |

**Findings.**
- **Standard RAG beats the "advanced" architecture on nearly everything** — context precision ~2× higher
  (~0.62–0.66 vs ~0.29–0.33) and answer correctness clearly higher (0.41–0.52 vs 0.24–0.32), in *both* passes.
  Corroborates the advanced-RAG bugs in §4.3 (context reuse, doc-grading drops relevant chunks, empty answers).
- **Model ranking (standard RAG):** answer correctness **deepseek > gemma > llama** in both passes; precision a
  near-tie. So **deepseek = best generator, llama = weakest** on the standard pipeline.
- **Retrieval is the bottleneck, not generation:** Pass-2 faithfulness ≈ 0.97 for all six systems (answers are
  well-grounded in whatever context they get), but precision caps at ~0.64 and recall at ~0.56.

**Reliability caveats.**
- The two passes disagree sharply on **faithfulness** (RAGAS 0.33–0.62 vs DeepEval ~0.97) and **recall** (RAGAS
  ~0.13 and *inverted* — advanced > standard; DeepEval standard ~0.54 > advanced ~0.30). Only **context precision
  is stable** across both. Likely cause = the duplicate/unparsed `retrieved_context` issue (§5); Pass 2 added
  de-dup + a 128k-context judge. **Trust Pass 2.**
- `n` varies (195–300) partly because advanced RAG's **empty-answer rows get dropped**, which flatters its averages.
- `rag_ver_2_eval` (14 q) and `rag_ver_3_eval` (20 q) are tiny early pilots — ignore for headline conclusions.
- Why Pass 1 (RAGAS) was abandoned mid-way is now diagnosed with notebook-level evidence:
  see `reports/ragas_diagnosis.md` (quota 429 → local-judge timeouts/parse failures → NaN
  crashes from CRAG's dead-end bug; standard-RAG RAGAS runs in December *did* complete).

**Bottom line.** Simple **standard RAG with deepseek-r1 (or gemma3)** is the best-performing configuration today; the
advanced CRAG-style pipeline underperforms it everywhere that matters — consistent with the bugs in §4.3. The biggest
lever is **retrieval quality** (chunking, the `k`-ignored bug in §5, the `all-mpnet-base-v2` embedding upgrade in §2.2,
reranking). This is *not* evidence that decomposition is unnecessary — **~35% of the questions are multi-hop (§2.2)**,
where a *correctly-implemented* decomposition should help; CRAG simply implements it wrong (decompose-but-reuse, no
re-retrieval).

### 2.0 Pass 3 — r3 (open-source end-to-end, revised dataset; 2026-06-20)

A **within-pass controlled** comparison: identical dataset, judge, embeddings, and retrieval —
only the architecture varies. **Not row-comparable to Pass 1/2** (different dataset, judge, and
embeddings), but the standard-vs-CRAG-vs-CRAG++ contrast inside Pass 3 is clean.

**Setup.** 195-question revised dataset (`datasets/revision/evaluation_dataset.xlsx`; 123 human /
72 synthetic). Generator: local Ollama **llama3:8b** (+ gemma3:1b grader for CRAG/CRAG++). Judge:
**`gpt-oss-120b` via DeepInfra** (OpenAI-compatible; replaced Groq, whose pay-per-token tier was
waitlisted). Embeddings `all-mpnet-base-v2` (768-dim), top-4 dense. New 5th metric **Gold-Context
Similarity** = max cosine between the gold `reference_context` and the retrieved chunks
(deterministic, embedding-based, no judge cost). 2,925 metric cells scored, 1 null (99.97%).

**Pass 3 — DeepEval (n=195 per system)**

| System | Ctx Precision | Ctx Recall | Faithfulness | Answer Correctness | Gold-Ctx Sim |
|---|---|---|---|---|---|
| standard | **0.741** | **0.796** | 0.939 | **0.473** | **0.741** |
| crag | 0.690 | 0.671 | **0.948** | **0.474** | 0.717 |
| cragpp | 0.449 | 0.650 | 0.910 | 0.334 | 0.712 |

Answer correctness by data_type (human / synthetic): standard 0.529 / 0.378; crag 0.477 / 0.468;
cragpp 0.309 / 0.376. Full split in `results/eval_r3_summary.xlsx`.

**Findings.**
- **Standard still wins** — best on 4 of 5 metrics; CRAG ties it only on answer correctness (0.474
  vs 0.473). The Pass-1/2 "simple beats advanced" result holds on a fresh dataset and a stronger judge.
- **CRAG++ underperforms both**, contradicting the design hypothesis — Ctx Precision collapses to
  0.449 and answer correctness to 0.334. Its per-sub-question retrieval + dedup widens the context
  but lowers precision (more, noisier chunks), and answer quality falls with it.
- **Gold-Context Similarity is flat (~0.71–0.74) across all three** — retrieval surfaces comparable
  ground-truth overlap regardless of architecture. The differences are in **how each pipeline uses
  context, not raw retrieval vs. gold** — so the lever stays *precision / use of context*, not recall.
- Faithfulness ~0.91–0.95 everywhere (answers stay grounded), consistent with Pass 2.

### 2.1 Architecture comparison — Standard vs Auto vs CRAG

Strong/weak summary (the underlying bugs are detailed in §4):

- **Standard** *(evaluated — winner)* — simplest, cheapest, best scores. No multi-hop capability, no
  quality control, and the forced "think step by step" prompt pollutes the answer field. Wins **despite**
  the dataset's ~35% multi-hop questions (§2.2) — because the multi-hop architectures are broken, not
  because every question is simple.
- **Auto** *(never run)* — **best retrieval design of the three**: genuinely **re-retrieves per
  sub-question/step**, so each sub-question gets fresh, targeted evidence (the bottleneck §2 identifies).
  But it **crashes with `OllamaLLM`** (`.content`, §4.2), has **no quality control**, uses reasoning
  steps as (poor) search queries, has **no parse fallback**, and is the costliest (~8 sequential calls).
- **CRAG** *(evaluated as "advanced")* — **the only one with quality control** (doc-relevance + answer
  grading + retry loop) and the most robust to run (`StrOutputParser`, batched calls, JSON fallback).
  But its **decomposition is cosmetic** (reuses the original query's docs — the per-sub-question
  retrieval node is dead code), its **grading backfires** (gemma3:1b + exact `"yes"` → drops relevant
  docs → empty answers), and it **lost to Standard** empirically (precision 0.32 vs 0.64, AC 0.29 vs 0.48).

**Auto vs CRAG — head-to-head:**

| Dimension | Auto | CRAG |
|---|---|---|
| Re-retrieval per sub-question | ✅ yes (fresh evidence) | ❌ no — reuses original docs |
| Quality control (grading) | ❌ none | ✅ doc + answer grading |
| Self-correction loop | ❌ none | ✅ yes (but unbounded) |
| Runs with provided `OllamaLLM` | ❌ crashes (`.content`) | ✅ runs |
| Parse fallback | ❌ none | ✅ single-sub-q fallback |
| Efficiency | ❌ sequential, ~8 calls | ✅ batched |
| Empirical result | — never run — | underperformed Standard |

Their strengths mirror each other: **Auto has the right retrieval architecture but no guardrails and
won't run; CRAG has the guardrails/robustness but a hollow retrieval design.**

**Is CRAG actually better than Auto? — Not established, and probably the wrong question.**
- *Empirically:* unknown. Auto was never evaluated; the measured comparison is Standard vs CRAG, and Standard won.
- *By design:* CRAG is the more complete/runnable system; Auto has the more correct core idea
  (per-sub-question retrieval — exactly the bottleneck). Both are currently broken and both lose to
  Standard even on the dataset's **~35% multi-hop** share (§2.2), where decomposition *should* win.
- *Best path:* a **hybrid** — Auto's per-sub-question retrieval + CRAG's grading/self-correction, with a
  stronger grader than gemma3:1b, dedup, and the `k`-fix (§5). To settle it for real, fix Auto's P0
  crash and run it through the same harness as a third system (`l3/g3/d3`) — see §7.

### 2.2 Dataset composition & retrieval levers (updated 2026-06-12, after full sync)

**Eval set (`datasets/datasets.xlsx`, sheet `final`) — now present.** 299 questions, mostly factual wh-questions
(what 130, how 84, which/according 15 each, when 13). It is **not purely single-hop**: **~35% (106/300) carry a
second gold passage** (multi-passage/compound — e.g. "how many public, *private, and charter* schools"), and the
median reference answer is **21 words** (only 12% are ≤5 words). Implication: a *correctly-implemented* decomposition
should help on that multi-hop third, so CRAG's loss there indicts its **implementation** (decompose-but-reuse, no
re-retrieval; §4.3), **not** the idea of decomposition.

**Embedding upgrade available but unwired.** `all-mpnet-base-v2/` (768-dim) was added to the repo, but all three
configs still hard-code `all-MiniLM-L6-v2` (384-dim) and nothing references mpnet (`grep` clean). Since retrieval is
the measured bottleneck (§2: precision ~0.64 / recall ~0.56 ceilings), switching is the **top retrieval lever** — but
it needs a config change (`EMBEDDING_MODEL`) **and a full vector-store rebuild** (384-dim and 768-dim indexes are
incompatible; `chroma_db/` must be regenerated). Tracked in §6.

> **Note (full sync, 2026-06-12):** the synced files are **data + the embedding model only** — every `src/*.py` is
> byte-unchanged, so all code findings (§3–§7) and the source-vs-runtime drift (§3) still hold exactly.

---

## 3. Source-vs-runtime drift (read before touching the code)

The committed `src/` is a **partial/older snapshot** and has **drifted from the code that
actually ran in Colab** (`/content/drive/MyDrive/RAG/src`). Evidence:
- The advanced notebook returns dict-style results keyed `sub_question_1` and reads
  `response['final_answer']` / `response['sub_questions']` — which the committed
  `graph_builder_adv.py` (as imported) cannot produce, because it imports the **Pydantic**
  `AdvanceRAGState` while the node code is written for a **TypedDict**.
- The notebook comments ("will use the patched `build` method") imply runtime patching.

**Implication:** fixes should reconcile the committed source *toward the working runtime
version* (the TypedDict state, string-output models). Reconcile source ↔ runtime before
layering new fixes.

---

## 4. Per-architecture issue catalog

Severity: 🔴 blocker/crash · 🟠 major · 🟡 minor · 🟢 cleanup.

> **Update 2026-06-12:** the 🔴 **P0** items below (Auto `.content` + parse fallback; CRAG state-unification,
> empty-result fallback, generation loop guard) are **fixed in code and smoke-tested** — see §7. The **P1** code
> items (think-stripping, retriever `k`, relaxed grading, per-sub-question retrieval, decoupled prompt, embedding
> wiring) are also fixed — see §8. Remaining P1 (eval-harness re-scoring) and P2 are open.

### 4.1 Standard RAG — `graph_builder.py` / `nodes.py` / `rag_state.py`

- 🔴 **CoT forced into the answer.** `nodes.py:36` instructs "walk the user through your
  thought process step by step." Gold answers are terse facts ("218", "March", "Article 4"),
  but every answer begins *"I'll walk you through my thought process step by step…"* (seen in
  notebook output). Wrecks answer-correctness scoring; bloats tokens/latency.
  **Fix:** drop CoT for this extraction task, or return `{"reasoning":…, "answer":…}` and score only `answer`.
- 🟡 **Unterminated string in prompt.** `nodes.py:37`: `say: 'Sorry, I do not know the answer.`
  — quote opened, never closed. **Fix:** close it.
- 🟡 **Nodes return whole new `RAGState`** (`nodes.py:21-24, 51-55`) instead of partial updates;
  resets sibling fields, fragile as graph grows. **Fix:** return partial dicts and let LangGraph merge.
- 🟢 **Dead state fields.** `rag_state.py:9-12` (`context`, `feedback`, `is_hallucination`,
  `is_unsupported`) never used. **Fix:** implement the hallucination check or delete.

### 4.2 Auto / "agentic" RAG — `graph_builder_auto.py` / `autorag_nodes.py` / `autorag_state.py`

- 🔴 **`response.content` on a completion model → `AttributeError`.** `autorag_nodes.py:32,103,114`.
  `Config.get_llm()` returns **`OllamaLLM`** (returns `str`, no `.content`); nodes were written
  for `ChatOllama`. **Fix:** standardize on `ChatOllama`, or pipe through `StrOutputParser()` / use the string.
- 🔴 **No fallback when sub-question parsing fails.** `plan_query` (`:36-47`) only accepts exact
  `Q1:`/`Q2:` lines with `len(key)==2` (`:40`); anything else → **zero** sub-questions → empty answer.
  **Fix:** regex parse + fallback to a single sub-question = original question.
- 🟠 **Reasoning steps misused as retrieval queries.** `plan_sub_steps` (`:52-83`) →
  `retrieve_per_sub_step` (`:86-94`) retrieves on each prose "step" (up to 3×3 = 9 retrievals,
  noisy). **Fix:** retrieve per *sub-question*; use steps to guide generation only.
- 🟠 **Not actually agentic** — fixed linear pipeline, no tool choice/grading/reflection.
  **Fix:** rename to "query-decomposition RAG" or wire in the real agent.
- 🟠 **Cost/latency** — ~8 sequential LLM calls + ~9 retrievals/question. **Fix:** batch generations; cap steps.
- 🟢 **Unused state fields** — `temp_questions`, `sub_steps`, `retrieved_docs`, `revised`,
  `attempts`, `SubQuery.plans`. Trim.

### 4.3 Advanced RAG — `graph_builder_adv.py` / `advrag_nodes.py` / `advrag_state.py`

- 🔴 **Two incompatible `AdvanceRAGState` definitions; committed import is wrong.**
  `advrag_state.py` = Pydantic (`retrieved_docs`, `sub_questions`, `answers`, `final_answer`);
  `advrag_nodes.py:13-18` = TypedDict (`documents`, `generation_grade`).
  `graph_builder_adv.py:3` imports the **Pydantic** one into `StateGraph` (`:68`), but lambdas/nodes
  use **dict access** `state["documents"]` (`:87`, throughout `advrag_nodes.py`) and `retrieve`
  (`:20-27`) writes a `documents` key absent from the Pydantic schema; `run()` (`:107`) constructs
  with `documents=[]`/`generation_grade=…` (Pydantic drops them). Runtime used the **TypedDict**.
  **Fix:** one canonical state (TypedDict), fix imports, one access convention.
- 🔴 **"Grade-all-irrelevant" → silent dead end (observed).** Conditional edge
  `graph_builder_adv.py:84-89` routes to `END` with empty `final_answer` when `grade_documents`
  filters everything. Notebook rows ~2,3 show empty answers + `sub_questions = {}`. With
  `gemma3:1b` grader this misfires often. **Fix:** fallback (keep top-k, re-query, or "insufficient context").
- 🔴 **Generation-grade loop has no termination guard → possible infinite loop.**
  `:96-100` routes `"no" → generate_answers` with the *same* inputs; no attempt counter.
  **Fix:** `max_attempts` (e.g., 2), increment per loop, change something on retry, then exit gracefully.
- 🟠 **Brittle grader matching.** `:165` keeps doc only if `score == "yes"` exactly; 1B models emit
  "Yes."/"**yes**"/"yes, because…". **Fix:** `"yes" in score` / structured boolean.
- 🟠 **Decomposition adds cost, no retrieval benefit.** `plan_sub_steps` (`:61-68`) reuses the
  original query's docs for every sub-question; the real per-sub-q node
  `retrieve_sub_question_documents` (`:72-84`) is **dead code (never added to the graph)**.
  **Fix:** wire real per-sub-question retrieval (batchable).
- 🟠 **No `<think>` stripping (critical for deepseek-r1).** `deepseek-r1:8b` emits
  `<think>…</think>`; nothing strips it before JSON parse (`:50-55`), grading, synthesis, or scoring.
  **Fix:** shared strip-reasoning helper at every parse/output boundary.
- 🟢 **Inconsistent `.batch()` nesting** (`:47` single vs `:129` double `[[…]]`) — trial-and-error residue.

**Why CRAG scored ~2× worse than basic RAG — artifact vs genuine (investigated 2026-06-12).**
The "basic ≫ advanced" result (§2) is *part scoring artifact, part real*:

*Artifact — inflates the precision/recall gap (an eval-harness issue, not the architecture):*
- 🔴 **Context logged as one blob, not a list.** Standard logs `retrieved_context` as **4 separate nodes**
  (~3.6k chars); CRAG logs **1 node** (~8.3–8.9k chars). Contextual precision/recall are *ranking* metrics over a
  list of nodes — a single blob degenerates them to **0**. CRAG hard-zeros on **135/197 rows (69%)** vs standard's
  36%; round-2's blank-line splitter only reached ~1.35 chunks, so the artifact survived (`l2` precision still 0.29).
- **Proven false zeros:** rows whose blob contains **73–100%** of the gold-answer words still scored precision 0.
- **Logical clincher:** CRAG runs the *same retriever on the same query* as standard (then reuses those docs), so
  its context is a *superset* of standard's — scoring the same text 2× worse is serialization, not worse retrieval.

*Genuine — real CRAG weakness, concentrated in answer correctness:*
- 🟠 **gemma3:1b grader over-filters.** Zero-precision rows average only **19%** gold-answer-word overlap (vs 56%
  on non-zero rows) → the answer-bearing doc was often graded out before generation.
- 🟠 **Decompose-against-reused-context → refusal soup.** Sub-questions are narrower than the broad reused docs
  cover, so per-sub-answers are frequently refusals (*"Unfortunately, I don't see any information…"*); synthesis
  blends them into a vague final answer. (`page_content_1 == page_content_2` in **198/199** rows confirms reuse.)
- 🟠 **"≤100 words" synthesis** (`advrag_nodes.py:127`) paraphrases away the exact fact ("218", "March") the terse
  gold answers need.
- *Not broken:* faithfulness is high (0.59 → **0.97** round-2) — CRAG answers *are* grounded, just grounded-and-vague.

**Takeaway:** the **answer-correctness gap is genuine**; the **precision/recall gap is largely a harness artifact**.
For a fair comparison, **re-log CRAG's retrieved context as a list of distinct chunks (ideally the *pre-grading* docs)
and re-score** — see §6 (P1 eval-harness fix).

### 4.4 ReAct node — `reactnode.py` (orphaned)

- 🟠 **Wikipedia tool breaks closed-book eval.** `:50-58` let the agent answer from general
  knowledge → contaminates faithfulness/grounding. **Fix:** drop/gate Wikipedia for the eval.
- 🟢 **Class re-named `RAGNodes`** (collides with `nodes.py`); `retrieve_docs` (`:20-27`) would
  double-retrieve if bolted onto the standard graph. **Fix:** decide if this is a 4th system; if so, own builder + distinct name.

---

## 5. Cross-cutting issues

- 🔴 **Config env-var collision.** `config.py:8-13` — `LLAMA3_MODEL`, `GEMMA3_MODEL`,
  `GPT_OSS_MODEL`, `DEEPSEEK_MODEL` all read the **same `LLM_MODEL`** env var (only fallbacks differ);
  setting `LLM_MODEL` collapses them all to one value. **Fix:** distinct env keys per model, or drop the env indirection.
- 🟡 **`config.py` `get_llm` defaults to deepseek** (`:25`) even though `pirls_rag_with_llama3.ipynb`
  imports this file and calls `Config.get_llm()` after pulling llama. `config_llama.py`/`config_gemma.py`
  fix the default per-model. **Fix:** set the default explicitly at each call site, or rename `config.py`.
- 🟠 **`get_retriever(k)` ignores `k`.** `vectorstore.py:61` — `as_retriever(k=k)` should be
  `as_retriever(search_kwargs={"k": k})`. Likely source of duplicate retrieved contexts
  (eval notebooks compensate with `dedupe=True`). **Fix:** use `search_kwargs`.
- 🟠 **Benchmark fairness.** Systems emit different answer shapes (standard "I'll walk you
  through…", advanced "Based on the provided documents…"). Scoring measures verbosity as much as
  architecture. **Fix:** uniform "extract concise answer" normalization pass before scoring.
- 🟢 **No node-level error handling/timeouts** (notebooks wrap the loop; nodes don't).
- 🟢 **Deprecations.** `vectorstore.persist()` (`:26,53`) is a no-op in current `langchain_chroma`;
  unused `OllamaEmbeddings` import. **Fix:** remove.

---

## 6. Prioritized fix table

> **Status (2026-06-12):** **P0** ✅ (§7), **P1** code ✅ (§8), **P2** code ✅ (§9). Still open: eval-harness
> chunk-logging + uniform answer-extraction (P1) and the `chroma_db` reindex — all need the eval notebooks / a run.

| Pri | Fix | Where |
|---|---|---|
| **P0** | Unify `AdvanceRAGState` to one definition; correct import | `advrag_state.py`, `graph_builder_adv.py:3`, `advrag_nodes.py:13` |
| **P0** | Standardize model interface (ChatOllama or string-output everywhere) — fixes auto `.content` crash | `config*.py`, `autorag_nodes.py` |
| **P0** | Fallbacks: empty-sub-questions (auto), grade-all-irrelevant→END (advanced); `max_attempts` loop guard | `autorag_nodes.py`, `graph_builder_adv.py:84-100` |
| **P1** | Strip `<think>` tags everywhere (shared helper) | all nodes |
| **P1** | Fix `get_retriever` `k`; relax grader matching to `"yes" in score`; wire real per-sub-question retrieval | `vectorstore.py:61`, `advrag_nodes.py` |
| **P1** | **Retrieval upgrade:** wire `all-mpnet-base-v2` (768-dim) via `EMBEDDING_MODEL` + rebuild `chroma_db` (dim mismatch needs a full reindex) — top lever for the precision/recall ceiling (§2.2) | `config*.py`, `chroma_db/` |
| **P1** | Decouple reasoning from answer field; uniform answer-extraction before scoring | `nodes.py:35-44`, eval notebooks |
| **P1** | **Eval harness:** log CRAG `retrieved_context` as a list of distinct chunks (not one ~8.7k-char blob) + log *pre-grading* docs, then re-score — single-blob logging degenerates ranking metrics and unfairly ~halves CRAG precision/recall (§4.3) | eval notebooks |
| **P2** | Config env-var collision; remove dead fields/nodes; drop Wikipedia from ReAct for eval; deprecations/typos | `config.py:8-13`, states, `reactnode.py`, `vectorstore.py` |

---

## 7. P0 fix roadmap (first batch — ✅ applied & smoke-tested 2026-06-12)

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done.

- [x] **Unify `AdvanceRAGState`.** `advrag_state.py` is now a single canonical `TypedDict`
  (`question`, `documents`, `sub_questions`, `final_answer`, `generation_grade`, `attempts`; `total=False`).
  `advrag_nodes.py` and `graph_builder_adv.py` import it; removed the duplicate class; `retrieve` now uses
  `state["question"]` (dict access throughout).
- [x] **Standardize the model interface.** `autorag_nodes.py` now pipes `self.llm | StrOutputParser()`
  (`self.llm_to_str`) and passes plain string prompts — no more `.content`. Works with both `OllamaLLM` and
  `ChatOllama`; standard/advanced paths already used string output and were left untouched.
- [x] **Auto fallback.** `plan_query` falls back to a single sub-question = the original question when none parse.
- [x] **Advanced empty-result fallback.** `grade_documents` keeps the original retrieved docs when the grader
  rejects everything — no more dead-end empty answer.
- [x] **Advanced loop guard.** `grade_generation` increments `attempts`; the edge stops at
  `MAX_GENERATION_ATTEMPTS = 2` (or on "yes"). No more unbounded loop.
- [x] **Bonus blocker fix (listed P2, but blocked Auto entirely):** `from langchain.schema import Document` →
  `from langchain_core.documents import Document` in `autorag_state.py` + `autorag_nodes.py` — the old import no
  longer resolves, so Auto couldn't even import.

**P0 verification — PASSED.** Smoke-tested all three graphs with a programmable fake LLM (no Ollama needed):
- **CRAG:** runs end-to-end; grade-all-irrelevant fallback fires; loop guard caps at `attempts=2`; non-empty answer.
- **Auto:** runs with no `.content` `AttributeError`; empty-parse fallback yields `sub_questions=['Q1']`; non-empty answer.
- **Standard:** still builds + runs (regression check). All four changed files `py_compile` clean.
- *Not yet run against real Ollama models / the full dataset — that is the next validation step.*

**Follow-up after P0 (validation).** Once Auto's `.content` crash is fixed, **evaluate Auto as a third
system (`l3/g3/d3`)** through the same eval harness (Standard vs Auto vs CRAG). This is the only way to
empirically answer whether CRAG actually beats Auto — currently untested (see §2.1).

> Decisions on record: fix scope = **P0 first** (then P1/P2); sequencing = save this doc now,
> **defer code edits until files finish syncing**.

---

## 8. P1 batch — applied & smoke-tested 2026-06-12

**✅ Code items DONE.**
- [x] **`<think>` stripping.** New `src/utils/text.py::strip_reasoning()` removes deepseek-r1 reasoning; applied at
  every LLM-output point in `nodes.py`, `autorag_nodes.py`, `advrag_nodes.py` (answers, JSON parse, both graders).
- [x] **`get_retriever(k)`.** `vectorstore.py` now passes `search_kwargs={"k": k}` so `k` is honoured.
- [x] **Relaxed grader matching.** CRAG `grade_documents`/`grade_generation` use `"yes" in score` (handles
  "Yes.", "**yes**", "yes, relevant").
- [x] **Real per-sub-question retrieval (CRAG).** `AdVRagNodes` now takes the retriever; `plan_sub_steps` retrieves
  fresh docs per sub-question (falls back to the original docs if no retriever/query). Smoke test confirmed the
  retriever is called with each sub-question query, not just the original.
- [x] **Decouple reasoning (Standard).** `nodes.py` prompt no longer forces step-by-step CoT into the answer; asks
  for a direct, concise answer (also fixed the unterminated quote) and strips `<think>`.
- [x] **Embedding upgrade wired.** `config*.py` default `EMBEDDING_MODEL` → `sentence-transformers/all-mpnet-base-v2`
  (768-dim). ⚠️ **Requires a `chroma_db` rebuild** — the old 384-dim index is incompatible; not reindexed here.
- [x] **Bonus:** the `\{` regex `SyntaxWarning` in `advrag_nodes.py` is gone (raw string).

**⏳ Still open (need the eval notebooks + a re-score run, not `src/`):**
- [ ] **Eval harness — log CRAG `retrieved_context` as a list of chunks** (not one blob) + log pre-grading docs,
  then re-score (§4.3). This is the fix that makes the precision/recall comparison fair.
- [ ] **Uniform answer-extraction before scoring** (§5 benchmark fairness).
- [ ] **`chroma_db` reindex** with all-mpnet-base-v2, then re-run the full suite.

**P1 verification — PASSED.** `strip_reasoning` unit tests; `get_retriever` passes `search_kwargs`; all three graphs
run with a fake LLM — CRAG strips `<think>` (answer `"SYNTHESIZED"`), the relaxed grader keeps a "Yes, relevant."
doc, per-sub-question retrieval calls the retriever with each sub-query (`['main q?','alpha sub','beta sub']`);
Standard returns a clean `"218"`. `py_compile` clean on all 9 changed files.

> **⚠️ The §2 evaluation numbers predate the P0+P1 fixes** (they describe the old, broken CRAG and the CoT-polluted
> Standard answers). They must be **regenerated** against real Ollama models before drawing fresh conclusions.

---

## 9. P2 cleanup — applied & smoke-tested 2026-06-12

- [x] **Config env-var collision.** `LLAMA3_MODEL`/`GEMMA3_MODEL`/`GPT_OSS_MODEL`/`DEEPSEEK_MODEL` now read their
  own env vars (were all reading `LLM_MODEL`, so setting it collapsed them) across `config.py`/`config_llama.py`/
  `config_gemma.py`. Removed the unused `OllamaEmbeddings` import.
- [x] **Deprecated `.persist()` guarded.** `vectorstore.py` calls `self.vectorstore.persist()` only when present
  (`hasattr`) — modern `langchain_chroma` auto-persists and dropped the method.
- [x] **Dead fields/node removed.** `rag_state.py`: dropped `context`/`feedback`/`is_hallucination`/`is_unsupported`.
  `autorag_state.py`: dropped `temp_questions`/`retrieved_docs`/`revised`/`attempts` and `SubQuery.plans` (+ the
  `"plans"` keys in `autorag_nodes.py`). `advrag_nodes.py`: removed the dead `retrieve_sub_question_documents`
  placeholder (real per-sub-question retrieval now lives in `plan_sub_steps`, §8).
- [x] **ReAct Wikipedia gated.** `reactnode.py` takes `use_wikipedia=False` (default); the Wikipedia tool and its
  imports are built only when explicitly enabled, so the closed-book eval can't answer from general knowledge.
  Added a closed-book system prompt.

**P2 verification — PASSED.** `py_compile` clean on all 9 files; all three graphs still build + run with a fake LLM
(CRAG `SYNTHESIZED`/attempts=1, Auto two sub-questions, Standard `218`); ReAct closed-book builds only the
`retriever` tool (no Wikipedia import required).

**Remaining (not `src/`):** eval-harness chunk-list logging + uniform answer-extraction (P1), and the `chroma_db`
reindex with all-mpnet-base-v2 — then re-run the suite to regenerate §2.

---

## 10. plans.md execution (branch RAG-2, started 2026-06-12)

Staged execution of `plans.md`; full plan with pass criteria lives in the session plan file.
Decisions: CRAG frozen at evaluated design (ADR 0001); new 4th architecture **CRAG++**; eval =
3 systems × 2 generators (claude-haiku-4-5 / Llama 3.1 8B via Groq) × 300 rows, judge gpt-4.1
(DeepEval, 4 metrics) — full runs deferred to a dedicated session; Streamlit UI
(`streamlit_app_auto.py`) gets review + repair + chunk/score inspector.

- [x] **Stage 1 — CRAG baseline restore.** Reverted P1 per-sub-question retrieval in
  `advrag_nodes.plan_sub_steps` (sub-questions reuse original docs again); removed retriever
  plumbing from `AdVRagNodes.__init__` / `graph_builder_adv.py`. ADR 0001 + §0 terminology added.
- [x] **Stage 2 — CRAG++ architecture.** New `src/state/cragpp_state.py` (adds
  `final_contexts: List[Document]` — log this as `retrieved_context`, chunk list not blob),
  `src/nodes/cragpp_nodes.py`, `src/graph_builder/graph_builder_cragpp.py`,
  `src/utils/docs.py::dedup_documents`. Pipeline = CRAG skeleton + per-subq retrieval (graded
  with fallback per subq) + dedup (within subq and union) + no synthesis word cap. Smoke test
  `tests/smoke_cragpp.py` passes all 6 behavioral criteria.
- [x] **Stage 3 — RAGAS failure diagnosis.** `reports/ragas_diagnosis.md` — four-cause cascade
  (OpenAI quota 429 → local-judge timeout/parse failures → NaN ValidationError from CRAG
  dead-end answers → unpinned 0.4.1/0.4.2 API drift), all claims with notebook::cell refs;
  December standard-RAG RAGAS runs actually completed (gpt-4o-mini judge).
- [x] **Stage 4 — Eval infrastructure.** Venv `.venv/` (pins in `requirements-eval.txt`),
  `src/config/config_api.py` (generators: haiku / llama-groq / **ollama** for key-less local
  runs), `scripts/build_index.py` (chroma_db rebuilt: 2,346 pages → 6,771 chunks, 768-dim mpnet,
  sanity query OK), `scripts/run_generation.py` (resumable, chunk-list `retrieved_context`,
  k=4 parity), `scripts/run_eval.py` (DeepEval 4 metrics incl. answer-correctness GEval, judge
  gpt-4.1 or `ollama:<model>`, per-run cost). **Pilot (local Ollama, per user)**: standard 15
  rows (mean 18 s), CRAG 10 rows (66.6 s), CRAG++ 10 rows (97.5 s) — 0 empty answers; all
  contexts JSON lists (CRAG blob artifact gone); CRAG++ dedup + loop guard + per-subq retrieval
  observed live; 4/4 metrics numeric via Ollama judge, 0 failures. *Deferred to the eval
  session (needs .env keys): haiku / llama-groq generator connectivity + real judge-cost
  extrapolation; rough estimate ≈ $15 Haiku + $1–2 Groq + $80–90 gpt-4.1 judge (top of
  envelope — consider gpt-4.1-mini ≈ $20 if budget matters).*
- [x] **Stage 5 — UI review + repair + inspector.** `reports/ui_review.md` (13 issues with
  file:line — 5 blocking: nonexistent `process_urls`/`create_retriever`, empty `DEFAULT_URLS`,
  session-state deletion every rerun, wrong result key). `streamlit_app_auto.py` rewritten:
  loads persisted chroma_db, sidebar architecture selector (Standard/CRAG/CRAG++) + Ollama
  model selector, answer card with latency/chunk count, retrieved-chunk inspector with vector
  distances + "contexts actually used" view, history capped at 10. Verified: headless boot
  HTTP 200; widget-layer test (AppTest) answers correctly via Standard; CRAG and CRAG++ answer
  via the same app functions (`tests/smoke_ui.py`).
- [x] **Stage 6 — Consolidated advisory write-up.** `reports/plans_response.md` answers all six
  plans.md items (architecture, RAGAS diagnosis, eval design + model recommendations, GPU +
  retrieval-quality roadmap, UI review, context logging + production readiness). Everything
  dependent on the full runs is marked PENDING r3.
- [x] **Stage 7 — full eval runs (r3), done 2026-06-20.** Open-source end-to-end on the revised
  195-row dataset (`datasets/revision/evaluation_dataset.xlsx`): local Ollama llama3:8b generation
  (3 systems × 195 rows, 0 empty), judge `gpt-oss-120b` via **DeepInfra** (Groq waitlisted), new
  Gold-Context Similarity metric. 15-row pilot gate passed first. Results + findings in §2.0; r3
  slots in `reports/plans_response.md` filled. Scores force-added (`results/eval_r3_*_deepeval.csv`,
  `eval_r3_summary.xlsx`). Code: schema bridge + `id` row_id in `run_generation.py`, `compat:` judge
  + cosine metric in `run_eval.py`, `scripts/summarize_r3.py`, CRAG `IndexError` crash-fix in
  `advrag_nodes.py`.
