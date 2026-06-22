# PIRLS RAG — Evaluation & Architecture Evolution (r1 → E11)

**An R&D analysis for reviewers.** How this benchmark's numbers went from untrustworthy to canonical, what each architecture change actually did *mechanically*, and what to build for production.

Branch `RAG-2` · dataset: revised 195-question PIRLS 2021 set · judge held constant at DeepInfra `gpt-oss-120b` for all r3/E runs. Every claim below is backed by a measured number, an error trace, or a concrete row example. Figures are in [`figures/`](figures/) and regenerable via [`figures/make_figures.py`](figures/make_figures.py).

---

## 0. Executive summary

This project ran three evaluation passes and a follow-up experiment series:

- **r1 (RAGAS, Dec 2025 – Jan 2026)** produced *poor and partly meaningless* numbers. The Standard-RAG runs completed but on an immature pipeline; the CRAG runs **never finished** — they died in a four-failure cascade (API quota, local-judge timeouts, a NaN-producing pipeline bug, and unpinned library drift).
- **r2 (DeepEval, Jan 2026)** fixed the harness (it tolerates a local judge and skips bad rows) but exposed a **measurement artifact**: CRAG's retrieved context was logged as one long blob instead of a chunk list, mechanically zeroing its precision/recall.
- **r3 (Jun 2026)** is the **first trustworthy three-way comparison**. With the logging fixed, embeddings upgraded, and the retrieval bug closed, **Standard RAG wins or ties on 4 of 5 metrics**. CRAG matches it on answer quality but loses recall; CRAG++ regresses sharply. The added machinery removes signal rather than adding it.
- **The E-series** found the real levers. **Reranking lifts answer correctness +32% for ~$0**; **hybrid BM25+dense retrieval (E9) adds another +10%** (AC **0.689** on the free local llama); and a **stronger generator on top of hybrid** lifts it further — the generator sweep on hybrid+bge (E10/E11/E12) gives llama 0.689 → deepseek 0.717 → **gpt-5.4-mini 0.774, the program best**. Retrieval still dominates at the open tier (local llama + hybrid 0.689 > gpt-5.4-mini + plain rerank 0.652). An extraction-prompt tweak (E8) and a dense-only reranker swap (E9b) both **backfired** — though under hybrid, bge ≥ MiniLM (E12).
- **The value-added ablation** answers the reviewers' "is RAG worth it?" decisively: vs a closed-book llama (no retrieval, AC 0.202), retrieval adds **+0.488**; the best config is **+0.572**. (Retrieval-only chunks score 0.827 on AC, but that is a coverage-metric artifact, not better answers — see §8.)

**Recommendation:** ship **Standard + hybrid (BM25+dense+RRF) retrieval + bge-reranker-base**. Default to the **local llama3:8b** generator (private, AC 0.689) and offer **gpt-5.4-mini** for maximum accuracy (AC 0.774). It is the best-scoring, simplest, and lowest-latency family. Do **not** ship CRAG/CRAG++ as-is — they cost 4–5× the latency for no quality gain.

![Metric evolution r1→E11](figures/fig1_metric_evolution.svg)

---

## 1. Setup

| | |
|---|---|
| **Dataset** | Revised 195-question set ([`datasets/revision/`](../datasets/revision)) — 123 human, 72 synthetic. Old 300-question set is deprecated (leakage; see §6). |
| **Systems** | **Standard** (retrieve→generate), **CRAG** (graded docs + sub-questions over *shared* context), **CRAG++** (CRAG + per-sub-question retrieval + dedup). Definitions: [`context.md`](../context.md) §0. |
| **Metrics** | Contextual precision, contextual recall, faithfulness, answer correctness, gold-context similarity. All 0–1, higher is better. |
| **Judge** | r1: OpenAI `gpt-4o-mini`. r2/r3/E: DeepInfra `gpt-oss-120b` (held constant for fair comparison). |
| **Generation** | r3 baseline + E2/OSS/E8/E9/E12: local Ollama `llama3:8b` (CPU). E10: local `deepseek-r1:8b`. E4/E11: OpenAI `gpt-5.4-mini` (API). |

The two retrieval metrics measure *the pipeline*; faithfulness and answer correctness measure *the generation*; gold-context similarity is a judge-free embedding check of whether the gold passage was retrieved at all.

---

## 2. r1 (RAGAS) — what went wrong

r1 failed in two distinct ways: the numbers it *did* produce were poor and not forward-comparable, and the numbers it was *supposed* to produce for CRAG never materialized.

### 2.1 The metrics were poor and immature

Standard-RAG with `llama3:8b`, scored by RAGAS (`gpt-4o-mini` judge, n=298, [`results/eval_with_answer_correctness_llama_20251220.csv`](../results/eval_with_answer_correctness_llama_20251220.csv)):

| Metric | r1 value |
|---|---|
| Contextual precision | 0.624 |
| Contextual recall | **0.459** |
| Faithfulness | **0.598** |
| Answer correctness | 0.412 |

Two of these are alarming: recall 0.459 and faithfulness 0.598. Part of this is the pipeline at the time (the `k`-was-ignored retrieval bug and the weaker MiniLM-384 embedding — both fixed by r3, §4.1), and part is RAGAS's own faithfulness definition, which is not comparable to DeepEval's (see §3). The point for reviewers: **r1's absolute numbers cannot be carried forward**; they reflect an immature pipeline measured by a different yardstick.

### 2.2 The CRAG evaluation never finished — a four-cause cascade

The Standard-RAG runs completed (Dec 2025, `gpt-4o-mini`, ~300 rows). The **CRAG** runs in Jan 2026 died in a cascade, documented with notebook cell references in [`ragas_diagnosis.md`](ragas_diagnosis.md):

1. **OpenAI quota exhaustion** mid-run — `Error code: 429 — You exceeded your current quota` at **row 163**, surfaced as `InstructorRetryException`.
2. **Local-Ollama judge rescue collided with RAGAS's structured-output machinery** — `ERROR:ragas.executor: ... TimeoutError()` (600 s) and `Prompt fix_output_format failed to parse output ... including retries`, both degrading to **silent NaN** scores. RAGAS runs metrics concurrently (`max_workers=5`) and demands schema-conformant JSON; sub-30B local judges on shared Colab hardware could not keep up.
3. **CRAG's dead-end bug produced NaN answers** that crashed validation outright — `ValidationError: ClaimDecompositionInput.response — Input should be a valid string [input_value=nan, input_type=float]`. (When all docs were graded irrelevant, the graph returned an empty answer → NaN in the CSV. Fixed by the r3 empty-docs fallback.)
4. **Unpinned, drifting RAGAS** — `!pip install ragas` pulled 0.4.1 in December and 0.4.2 in January, with **two incompatible metric APIs** across the standard vs advanced notebooks. `requirements.txt` pinned neither RAGAS nor DeepEval.

**Takeaway:** r1 is not a result, it is a lesson — pin the evaluation stack, make scoring resilient to a single bad row, and never let an empty pipeline output reach the scorer as NaN.

---

## 3. r2 (DeepEval) — fixed the harness, exposed an artifact

### 3.1 Why DeepEval worked

DeepEval succeeded where RAGAS cascaded — not because its metrics are "better," but because it ran **sequentially** (no 5-way concurrency starving a 20B local judge), **tolerated a local judge**, and **skipped bad rows** instead of raising and killing the whole pass ([`ragas_diagnosis.md`](ragas_diagnosis.md)).

One number must be read carefully: faithfulness jumped **0.598 (r1) → 0.978 (r2)**. This is a **framework artifact**, not an improvement — RAGAS and DeepEval define and prompt faithfulness differently. Do not interpret the jump as the model suddenly grounding its answers better.

### 3.2 The blob-vs-list measurement artifact

In r2, CRAG's retrieved context was serialized as a **single concatenated blob** while Standard's was a **list of separate chunks**. Recomputed directly from the r2 result files:

| | Standard (l1) | CRAG (l2) |
|---|---|---|
| Context stored as | list — first cell starts `[` | one blob — first cell starts with text |
| Representative cell length | 4,012 chars (a list of 4 chunks) | **7,918 chars (one string)** |
| Rows scored **0** contextual precision | **37.9%** | **64.6%** |

Source: [`eval_r2_l1_deepeval_cpr_cr_fa_20260118.xlsx`](../results/eval_r2_l1_deepeval_cpr_cr_fa_20260118.xlsx), [`eval_r2_l2_deepeval_cpr_cr_fa_20260120.xlsx`](../results/eval_r2_l2_deepeval_cpr_cr_fa_20260120.xlsx). The per-chunk precision/recall metrics need a *list* of contexts to rank; given one blob they collapse toward zero. CRAG's r2 precision/recall (≈0.29/0.31) are therefore **invalid**.

**Proof it is an artifact, not worse retrieval:** CRAG runs the *same retriever on the same query* as Standard, so its retrieved text is a **superset** of Standard's. Scoring the same underlying text ~2× worse is a serialization bug, not a retrieval regression ([`reviewer_response_1.md`](reviewer_response_1.md)). This is exactly why r3 re-logged every system's context as a chunk list.

---

## 4. r3 — the first canonical comparison

### 4.1 What changed between r2 and r3

The fixes that make r3 trustworthy ([`context.md`](../context.md)):

- **Chunk-list logging for every system** — kills the §3.2 artifact.
- **Embedding upgrade** to `all-mpnet-base-v2` (768-dim) from MiniLM-384.
- **The `k`-was-ignored retrieval bug fixed** — `as_retriever(search_kwargs={"k": k})` now honored.
- **Relaxed doc-grader parsing**, **CoT decoupled** from the Standard answer prompt, **empty-docs fallback** (never returns an empty answer → no NaN), **generation loop guard** (≤2 attempts), and **`strip_reasoning()`** for `<think>` tags.
- Revised 195-row dataset; judge fixed at `gpt-oss-120b`.

### 4.2 Results — Standard wins

n=195 per system, `llama3:8b`, `gpt-oss-120b` judge (matches [`experiments_index.md`](experiments_index.md) and [`context.md`](../context.md) §2.0):

| System | Ctx precision | Ctx recall | Faithfulness | Answer correctness | Gold-ctx sim |
|---|---|---|---|---|---|
| **Standard** | **0.741** | **0.796** | 0.939 | **0.473** | **0.741** |
| CRAG | 0.690 | 0.671 | **0.948** | **0.474** | 0.717 |
| CRAG++ | 0.449 | 0.650 | 0.910 | 0.334 | 0.712 |

Standard wins or ties on 4 of 5. CRAG ties on answer correctness (0.474 vs 0.473) but loses recall; CRAG++ regresses on every metric except faithfulness. Faithfulness is ~0.91–0.95 everywhere — **all three ground their answers well in whatever context they are given.** The differences are in *how each pipeline selects and uses context*, not in raw grounding.

![Pipeline per architecture](figures/fig2_architecture_flow.svg)

The next three subsections explain the gap with three **independent, measured** mechanisms.

### 4.3 Mechanism 1 — CRAG's grader destroys recall (net-negative filter)

CRAG's gemma3:1b document grader **fires on 87 of 195 questions** (drops at least one chunk). Splitting r3 rows by whether it fired (join of [`eval_r3_standard_ollama_deepeval.csv`](../results/eval_r3_standard_ollama_deepeval.csv), [`eval_r3_crag_ollama_deepeval.csv`](../results/eval_r3_crag_ollama_deepeval.csv), [`gen_r3_crag_ollama.csv`](../results/gen_r3_crag_ollama.csv)):

| | Recall | Precision |
|---|---|---|
| CRAG — grader kept all 4 (108 rows) | 0.819 | 0.771 |
| CRAG — grader dropped chunks (87 rows) | **0.486** | **0.590** |
| Standard (no grader) | 0.796 | 0.741 |

When the grader fires, recall collapses to **0.486** and precision *also* falls to 0.590 — it is **net-negative on both axes**. A weak 1B model is removing answer-bearing chunks and keeping worse ones.

**Concrete example** (row_id 31): *"How does Finland's national core curriculum integrate multiliteracy into early reading instruction in Grades 1–2…?"* — Standard contextual recall **1.000**, CRAG **0.000** on the identical question. The grader graded out the chunk that held the answer.

![CRAG grader recall collapse](figures/fig4_crag_recall_collapse.svg)

### 4.4 Mechanism 2 — CRAG++'s per-sub-question retrieval destroys precision

CRAG++ decomposes (~73% of questions) and retrieves *fresh* chunks per sub-question, then unions them. Mean chunk count rises to **5.88** (vs Standard's 4.00; [`gen_r3_cragpp_ollama.csv`](../results/gen_r3_cragpp_ollama.csv)). Those extra chunks are relevant to *sub-questions*, but precision is scored against the *original* question — so they read as noise:

| System | Rows with precision < 0.5 (retrieval failure) |
|---|---|
| Standard | 40 / 195 |
| CRAG | 50 / 195 |
| **CRAG++** | **107 / 195** |

CRAG++ more than doubles Standard's retrieval-failure rate. **Concrete example** (row_id 31, same Finland question): Standard precision **1.000**, CRAG++ **0.000**.

### 4.5 Mechanism 3 — a shared generation ceiling

Even with good context, `llama3:8b` often produces the wrong answer. In Standard, **53/195 rows (27%)** have good context (precision ≥ 0.7) yet a wrong answer (correctness ≤ 0.3); at the stricter ≤0.2 threshold it is 41/195 (~21%). This ceiling is **independent of architecture** and caps all three near AC ≈ 0.47.

**Concrete example** (row_id 6): *"At what age does compulsory primary education begin in Azerbaijan, and how long does the compulsory education period last?"* — contextual precision **1.000**, answer correctness **0.000**. The model answered *"…lasts 10 years, from age 5 … to age 15"* against a gold answer of *"begins at age 6, and the total compulsory education period lasts nine years."* Perfect context, wrong synthesis.

### 4.6 The complexity tax — latency

The machinery is not free. Mean latency per question (deduped by row, [`gen_r3_*`](../results)):

| System | Mean | vs Standard |
|---|---|---|
| Standard | 19.5 s | 1.0× |
| CRAG | 74.1 s | **3.8×** |
| CRAG++ | 101.8 s | **5.2×** |

CRAG and CRAG++ pay 4–5× the latency (extra grading + per-sub-question generation calls) for **no answer-quality gain**.

![Latency per run](figures/fig3_latency.svg)

---

## 5. The E-series — systematic improvement

All runs are Standard-family, n=195, judge `gpt-oss-120b`.

| Run | Generator | Config | Ctx P | Ctx R | Faith | **AC** | Latency |
|---|---|---|---|---|---|---|---|
| Standard (baseline) | llama3:8b | k=4, 1000/100 | 0.741 | 0.796 | 0.939 | 0.473 | 19.5 s |
| **+ rerank** | llama3:8b | k=20→rerank→4 | 0.837 | 0.837 | 0.940 | **0.625** | 21.9 s |
| E2 chunk-512 | llama3:8b | k=4, 512/64 | 0.723 | 0.759 | 0.939 | 0.488 | 11.3 s |
| OSS deepseek-r1:8b | deepseek-r1:8b | k=4, 1000/100 | 0.742 | 0.804 | 0.983 | 0.595 | 62.7 s |
| **E4 gpt-5.4-mini + rerank** | gpt-5.4-mini | k=20→rerank→4 | 0.837 | 0.847 | 0.981 | **0.652** | 1.6 s* |
| E8 rerank + extract | llama3:8b | rerank, extract prompt | 0.825 | 0.864 | 0.974 | 0.482 | 19.4 s |
| E9b dense + bge-base | llama3:8b | k=20→bge-reranker-base→4 | 0.807 | 0.850 | 0.957 | 0.578 | ~22 s |
| **E9c hybrid + bge-base** | llama3:8b | BM25+dense→RRF→bge-base→4 | 0.855 | 0.909 | 0.953 | 0.689 | ~23 s |
| E12 hybrid + MiniLM | llama3:8b | BM25+dense→RRF→MiniLM→4 | 0.867 | 0.920 | 0.950 | 0.679 | 19.2 s |
| E10 hybrid + bge-base | deepseek-r1:8b | BM25+dense→RRF→bge-base→4 | 0.853 | 0.904 | 0.985 | 0.717 | 62.4 s |
| **E11 hybrid + bge-base** | gpt-5.4-mini | BM25+dense→RRF→bge-base→4 | 0.865 | 0.914 | **0.984** | **0.774** | 7.9 s* |

\* E4/E11 used the OpenAI API (hosted GPU) — not comparable hardware to the local CPU runs.

### 5.1 Reranking — the biggest lever (+32% AC, ~$0)

**Mechanics.** Dense top-4 captures the gold chunk only **63%** of the time, but top-20 captures it **87%** ([`crag_improvement_analysis.md`](crag_improvement_analysis.md) §6). Reranking retrieves a *wide* candidate set (k=20) and rescoring with a cross-encoder promotes the gold chunk into the top-4 the generator actually sees. Cost is ~$0 (a CPU cross-encoder, [`src/vectorstore/rerank.py`](../src/vectorstore/rerank.py)).

**Impact.** Answer correctness **0.473 → 0.625 (+32%)**, precision 0.741 → 0.837, recall 0.796 → 0.837. The win concentrates exactly where it should — on the **40 worst-retrieval rows** (baseline precision < 0.5):

| On the 40 worst rows | Baseline | Rerank |
|---|---|---|
| Mean precision | 0.127 | **0.531** |
| Mean answer correctness | 0.185 | **0.333** |

**Concrete example** (row_id 63): *"What are the four broad-based comprehension processes assessed by PIRLS?"* Baseline answered *"Sorry, I do not know the answer…"* (AC **0.000**); with reranking it returned the four processes verbatim — *"focus on and retrieve … make straightforward inferences … interpret and integrate … evaluate and critique"* (AC **1.000**). The gold chunk existed all along; reranking surfaced it.

![Rerank win on worst-case rows](figures/fig5_rerank_worstcase.svg)

### 5.2 E2 (chunk-512) — marginal, but faster

Reindexing at 512/64 vs 1000/100 lifts AC only marginally (0.473 → 0.488) and slightly *lowers* precision/recall, but it is the **fastest** run (11.3 s) — less context text per generation. Rationale: index chunks average **832 chars** around a gold span of only **363 chars**, so each 1000-char chunk carries ~600 chars of distractor text. Smaller chunks help a little but do not substitute for reranking.

### 5.3 OSS deepseek-r1:8b — attack the generation ceiling

Swapping the generator (no rerank) lifts AC to **0.595 (+26%)** — direct evidence that §4.5's ceiling is a *generator* limit, not a retrieval one. Cost: latency rises to 62.7 s (a reasoning model emitting `<think>` tokens) with occasional large outliers.

### 5.4 E4 (gpt-5.4-mini + rerank) — a stronger generator helps

Combining wide-retrieve-then-rerank with a stronger generator reaches **AC 0.652**, precision 0.837, recall 0.847, faithfulness 0.981 — the best score *until E9*. Confirms the generator is a real second lever, but (see §5.6) it is **not** the dominant one: a local llama with better *retrieval* beats this.

### 5.5 E8 (extraction prompt) — a cautionary tale

Adding a terse "extract the exact fact" prompt on top of reranking **backfired**: AC fell **0.625 → 0.482 (−23%)** even though retrieval metrics stayed high. The brevity instruction confused `llama3:8b` more than it helped. Lesson: prompt tweaks interact strongly with the specific model and must be A/B-tested, never assumed.

### 5.6 E9 (hybrid retrieval + reranker upgrade) — best open/local, AC 0.689

The report's two untested Tier-1.1 retrieval levers — **hybrid BM25+dense fusion (RRF)** and a **reranker upgrade** — were run as two A/Bs on llama3:8b to attribute each. The result splits cleanly:

| Run | Config | Ctx P | Ctx R | **AC** |
|---|---|---|---|---|
| rerank-only (baseline for E9) | dense-20 → MiniLM → 4 | 0.837 | 0.837 | 0.625 |
| **E9b** | dense-20 → **bge-reranker-base** → 4 | 0.807 | 0.850 | 0.578 |
| **E9c** | **BM25+dense → RRF** → bge-base → 4 | **0.855** | **0.909** | **0.689** |

- **Hybrid retrieval is the win: +0.111 AC** (E9c vs E9b). RRF fusion lifts recall to **0.909** — the highest of any run. This is the predicted benefit: lexical BM25 catches exact terms (country names, acronyms like NAPLAN) that the bi-encoder misses, and RRF merges them with dense semantic hits.
- **The reranker "upgrade" backfired *on dense-only*: −0.047 AC.** Swapping MiniLM→bge-base on dense-only (E9b) *underperformed* the 2021 MiniLM (0.625 → 0.578) and dropped precision — so on a dense pool, bge-base is not a better reranker than `ms-marco-MiniLM-L-6-v2`. **This reverses under hybrid** (E12, §5.7): the E9c gain is from hybrid retrieval, but bge-base then becomes the better reranker. (The heavier `bge-reranker-v2-m3` remains untested.)
- **E9c (0.689) is the best *open/local* config — and it runs on the free local llama**, beating E4's gpt-5.4-mini+rerank (0.652). Direct proof that retrieval, not generator size, is the dominant lever here. (A stronger generator on the same hybrid retrieval later set the overall best at 0.774 — E11, §5.7.)

### 5.7 E10/E11/E12 — the generator × reranker sweep on hybrid (caveat resolved)

The E9 attribution gap (hybrid measured only on bge-base) is now closed, and the generator is swept on the fixed best retrieval (hybrid + bge-base):

| Run | Generator | Reranker | Ctx P | Ctx R | **AC** | Latency |
|---|---|---|---|---|---|---|
| E12 | llama3:8b | **MiniLM** | 0.867 | 0.920 | 0.679 | 19.2 s |
| E9c | llama3:8b | **bge-base** | 0.855 | 0.909 | 0.689 | ~23 s |
| E10 | deepseek-r1:8b | bge-base | 0.853 | 0.904 | 0.717 | 62.4 s |
| **E11** | **gpt-5.4-mini** | bge-base | 0.865 | 0.914 | **0.774** | 7.9 s |

- **Reranker, under hybrid: bge-base ≥ MiniLM** (E9c 0.689 ≥ E12 0.679) — the *opposite* of the dense-only result (§5.6, where MiniLM beat bge-base). BM25 fusion reshapes the candidate pool in a way the bge cross-encoder ranks better. So the earlier "bge-base is not a better reranker" conclusion was dense-only; **under hybrid, bge-base is the right choice** — which is why it is the production default.
- **Generator is a real, additive second lever once hybrid is in place:** llama 0.689 → deepseek 0.717 → gpt-5.4-mini 0.774. But **latency diverges by 8×**: gpt-5.4-mini is *both* the most accurate and the fastest (7.9 s, API), while deepseek-r1's +0.028 over llama costs 62 s/query (local reasoning model, well past the user survey's 30 s threshold). Hence the production split: **llama3:8b for private/fast, gpt-5.4-mini for maximum accuracy; deepseek is dominated.**
- **Retrieval still dominates the generator at the open tier:** local llama + hybrid (0.689) beats gpt-5.4-mini + *plain* rerank (E4, 0.652). The closed model only wins once it *also* has hybrid retrieval.

---

## 6. Cross-cutting findings

- **Retrieval is the binding constraint, then the generator.** Reranking gave +32%, hybrid fusion another +10% (E9c, 0.689 on local llama); stacking the strongest generator on top reaches the program best (E11, gpt-5.4-mini + hybrid, **0.774**). Retrieval dominates at the open tier — local llama + hybrid retrieval (0.689) beats gpt-5.4-mini + plain rerank (0.652) — but a strong generator is a real additive lever once hybrid is in place. The value-added ablation confirms the direction: vs closed-book (0.202), retrieval is worth +0.49 to +0.57 (§8).
- **Grounding is not the problem.** Faithfulness ≈ 0.91–0.98 across every system and run — when given the right context, these models do not hallucinate. This argues *against* "weak sub-10B reasoning" as the primary driver and *against* fine-tuning as the first lever.
- **Fine-tuning is premature.** The 195-row eval set is a **100% subset of the old 300-row set**, leaving only **~105 clean rows** — too few to fine-tune without leakage, and expected gains are low-single-digit vs reranking's +32% ([`crag_improvement_analysis.md`](crag_improvement_analysis.md) §5e).
- **Cross-country confusion is real but mild.** Of the 79/195 country-specific questions, 44% have ≥1 wrong-country chunk in the top-4, but only **4%** have a wrong-country chunk at rank 1 — concentrated in positions 2–4, which reranking already cleans up. A hard metadata filter risks the ~60% of questions that name no country (§5d).
- **The report-vs-reality gap (for reviewers).** An earlier write-up described mechanisms that **did not exist in the evaluated code**: there was no per-sub-question retrieval in CRAG (docs were reused in **198/199** rows) and no reranker wired into any graph. The real causes were the grader and the logging artifact ([`reviewer_response_1.md`](reviewer_response_1.md)). Lesson: validate that the evaluated artifact matches its description.

---

## 7. Production recommendation & scale-up

### 7.1 Recommended pipeline

**Standard RAG + hybrid (BM25+dense+RRF) retrieval + bge-reranker-base.** This is the **best-scoring**, **simplest**, and **lowest-latency** family. Default to the **local llama3:8b** generator (private, on-device, AC **0.689**, ~$0); offer **gpt-5.4-mini** for maximum accuracy (AC **0.774**, E11) and lowest latency (7.9 s API). Hybrid + bge reranking is the core; the generator is an additive second lever (open llama → closed gpt-5.4-mini = +0.085 AC). The production UI ([`pirls_rag_ui.py`](../pirls_rag_ui.py)) ships exactly this, with an open/closed model toggle. **deepseek-r1:8b is dominated** — only +0.028 over llama for 3× the latency (62 s).

**Do not ship CRAG or CRAG++ as-is.** They cost 4–5× the latency (§4.6) for no quality gain, because their two distinctive steps each *remove* signal (§4.3, §4.4).

### 7.2 If decomposition is wanted later, fix it first

1. Replace the binary gemma3:1b doc-grader with **reranker-score thresholding** (a real Corrective-RAG signal) — eliminates the recall-killer of §4.3.
2. Rerank the per-sub-question **union against the original question**, capped — re-aligns CRAG++ with how precision is scored (§4.4).
3. **Adaptive decomposition** — only decompose when a question is genuinely multi-hop, so the 73% of single-hop questions are not penalized.

### 7.3 Retrieval roadmap (highest leverage)

- **Hybrid BM25 + dense with RRF**, then rerank — **DONE (E9c), 0.689 on local llama** (+10% over rerank-only, recall 0.909). Ship this.
- **Reranker model choice is retrieval-dependent:** on dense-only, `bge-reranker-base` *underperformed* MiniLM (E9b); but **under hybrid, bge-base ≥ MiniLM** (E9c 0.689 ≥ E12 0.679), so bge-base is the production default. The heavier `bge-reranker-v2-m3` remains the only untested reranker upgrade.
- **Combine with a stronger generator — DONE (E11), the high-water mark at 0.774.** gpt-5.4-mini on hybrid+bge beats every open-model config; deepseek-r1 (E10, 0.717) is dominated on latency. Generator is confirmed as an additive lever on top of hybrid.

### 7.4 Production engineering

- **Pin the eval stack** (the explicit r1 lesson) and make scoring resilient to single-row failures.
- **Observability** (LangSmith / Langfuse) on retrieval hits, latency, and faithfulness.
- **Eval-in-CI** on a held-out, non-leaked set so regressions are caught before release.
- **Managed/persistent vector store**, plus **response and embedding caching** to control cost/latency.
- **Per-tier budgets**: route to a local model for cheap traffic and an API generator (E4-class) where quality matters; the codebase already supports both ([`src/config/config_api.py`](../src/config/config_api.py)).

### 7.5 Guardrails & risks

- Monitor **faithfulness in production** as a hallucination guard (it is the one metric that stays high — keep it that way).
- The **~21% generation ceiling** is real but **better retrieval pushes through it**: E9c reached AC 0.689 on the *base* llama (no stronger generator), because giving the model the right chunk converts many "good-context-wrong-answer" cases. Reach for an E4-class generator only when hybrid retrieval is already in place and AC > 0.69 is needed.
- Any future **fine-tuning needs a clean train/eval split** — the current eval set is leaked against the old training pool.

---

## 8. Value-added ablation & question coverage (reviewer-driven)

Two reviewer asks — R1 "does RAG beat an off-the-shelf LLM?" and R2 "what does the generator add over raw chunks?" — converge on one 3-condition ablation (n=195, AC only, same `gpt-oss-120b` judge). Full point-by-point response: [`reviewer_response_r3.md`](reviewer_response_r3.md).

| Condition | What | AC | human | synthetic |
|---|---|---|---|---|
| (a) Closed-book | llama3:8b, **no retrieval** | 0.202 | 0.150 | 0.290 |
| (b) Retrieval-only | hybrid+bge chunks **as the answer** | 0.827 | 0.852 | 0.785 |
| (c) Full RAG | llama3:8b + hybrid + bge | 0.689 | 0.753 | 0.581 |
| (d) Full RAG (best) | gpt-5.4-mini + hybrid + bge | 0.774 | 0.787 | 0.751 |

- **R1 — RAG is decisively worth it.** Same generator, retrieval adds **+0.488** (0.202 → 0.689); the best config is **+0.572** over closed-book. The open 8B model is near-useless closed-book on this corpus — so "GPT-4o scored >80% without retrieval" (a far larger closed model, different corpus) does not generalize here.
- **R2 — read the artifact carefully.** Retrieval-only (0.827) scoring *above* full RAG is a **GEval coverage artifact**: a ~4,000-char chunk dump contains the gold facts and the metric does not penalize verbosity, while the LLM's concise synthesis is dinged when it compresses a fact. It is **not** evidence that raw chunks are a better answer. The defensible reading: **the system is retrieval-bound, not generation-bound** — once the right chunks are retrieved the facts are present (0.827); the generator's value is concision/usability, which this metric doesn't reward (future work: a usability/concision metric).

**Question coverage (R1-2).** Classifying all 195 questions (`gpt-oss-120b`, [`scripts/classify_questions.py`](../scripts/classify_questions.py)): **68.2% fact-retrieval / 31.8% reasoning** (human 77% fact, synthetic 53% fact). The methodological topics R1 highlighted — sampling (4.6%), plausible values (3.1%), weighting/variance (1.5%) — are **under-sampled (~9%)**, so the eval under-tests reasoning-heavy "why/how" questions; a follow-up set should over-sample them. **KB scale (R2-6):** 6,771 chunks · 108 source docs · 1.5M tokens · 70.9 MB ([`scripts/kb_stats.py`](../scripts/kb_stats.py)).

---

## Appendix A — Full metric tables

**Across passes (Standard system):**

| Pass | Harness | Ctx P | Ctx R | Faith | AC |
|---|---|---|---|---|---|
| r1 | RAGAS (gpt-4o-mini) | 0.624 | 0.459 | 0.598 | 0.412 |
| r2 | DeepEval (gpt-oss-120b) | 0.618 | 0.539 | 0.978 | 0.376 |
| r3 | DeepEval (gpt-oss-120b) | 0.741 | 0.796 | 0.939 | 0.473 |

(r1/r2 not directly comparable to r3 — different harness, embedding, and the `k` bug. r2 Standard is clean; r2 CRAG is contaminated by the §3.2 artifact.)

**r3 + E-series:** see §4.2, §5, §5.6, and §5.7 tables. **Best overall: E11 (gpt-5.4-mini + hybrid + bge-base) AC 0.774. Best open/local: E9c (llama3:8b + hybrid + bge-base) AC 0.689.**

**Latency (mean s/question):** Standard 19.5 · CRAG 74.1 · CRAG++ 101.8 · +rerank 21.9 · E2 11.3 · deepseek 62.7 · E4 1.6 (API) · E8 19.4 · E9b/E9c/E12 ~19–23 · E10 deepseek+hyb 62.4 · E11 gpt-5.4+hyb 7.9 (API).

## Appendix B — Sources

- Diagnosis & forensics: [`ragas_diagnosis.md`](ragas_diagnosis.md), [`crag_improvement_analysis.md`](crag_improvement_analysis.md), [`reviewer_comments_1.md`](reviewer_comments_1.md), [`reviewer_response_1.md`](reviewer_response_1.md), [`plans_response.md`](plans_response.md).
- Canonical results & status: [`experiments_index.md`](experiments_index.md), [`context.md`](../context.md) §0/§2.0.
- Result data: [`results/eval_r3_*_deepeval.csv`](../results), [`results/gen_r3_*.csv`](../results), [`results/eval_r2_l1/l2_*.xlsx`](../results), [`results/eval_with_answer_correctness_llama_20251220.csv`](../results).
- Figures regenerated by [`figures/make_figures.py`](figures/make_figures.py) (numbers verified against the CSVs above).
- External reranking/RAG references are listed in [`crag_improvement_analysis.md`](crag_improvement_analysis.md) (ARAGOG, BM25→Corrective-RAG benchmark, reranker surveys).
