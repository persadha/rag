# Why Standard RAG beat CRAG/CRAG++ — and how to fix it (r3 forensics + SOTA roadmap)

> Evidence base: r3 run (2026-06-20), 195-question revised dataset, local llama3:8b generation,
> gemma3:1b grader, DeepInfra `gpt-oss-120b` judge. Per-row data in `results/eval_r3_*_deepeval.csv`
> + `results/gen_r3_*_ollama.csv`. Headline table in `context.md` §2.0.
>
> **This is the diagnostic + roadmap document (written before the E-series).** For the *outcomes* of the
> roadmap below — the full E1–E12 results, the value-added ablation, and the final recommendation
> (best overall: gpt-5.4-mini + hybrid + bge = AC 0.774; best open/local: llama3:8b + hybrid + bge = 0.689)
> — see the canonical [`experiments_index.md`](experiments_index.md) and
> [`architecture_evolution_analysis.md`](architecture_evolution_analysis.md). §4 below carries an inline
> **Status (2026-06)** callout marking what was done.

## 1. The result

| System | Ctx Precision | Ctx Recall | Faithfulness | Answer Correctness | Gold-Ctx Sim | mean chunks | mean ans len |
|---|---|---|---|---|---|---|---|
| **standard** | **0.741** | **0.796** | 0.939 | **0.473** | 0.741 | 4.0 | 255 |
| crag | 0.690 | 0.671 | 0.948 | 0.474 | 0.717 | 2.9 | 527 |
| cragpp | 0.449 | 0.650 | 0.910 | 0.334 | 0.712 | 5.9 (max 13) | 481 |

Standard wins or ties on 4 of 5 metrics. **The added machinery in CRAG/CRAG++ removes signal rather than adding it.**

## 2. Root cause — three independent mechanisms (all measured)

### 2.1 CRAG's recall loss = the gemma3:1b document grader drops relevant chunks
The binary "yes/no" grader (`advrag_nodes.py` doc-grading; `gemma3:1b`, exact-match `"yes"`) fires on **87/195** questions, and where it drops chunks recall collapses:

| CRAG rows | mean Ctx Recall |
|---|---|
| grader kept all 4 chunks (108 rows) | **0.819** (≈ Standard) |
| grader dropped chunks (87 rows) | **0.486** |

So the grader is a **net-negative filter** — a weak 1B model removing answer-bearing context. CRAG's precision/answer hold near Standard only because its sub-questions *reuse the same documents* (ADR 0001), so no off-topic dilution. The grader is the whole story behind CRAG's recall gap.

### 2.2 CRAG++'s precision collapse = per-sub-question retrieval injects off-topic chunks
CRAG++ retrieves *per sub-question* and logs the deduped union (mean 5.9, up to 13 chunks). Contextual Precision is judged against the **original** question, but the extra chunks were selected for **sub**-questions:

| CRAG++ rows | Ctx Precision | Answer Correctness |
|---|---|---|
| no expansion, ≤4 chunks (52 rows) | 0.538 | 0.260 |
| expanded, >4 chunks (143 rows) | **0.417** | 0.361 |

Decomposition fires on **73%** of questions and tanks precision when it does. CRAG++ has **107/195** rows with precision < 0.5 (retrieval failure) vs **40** for Standard and **50** for CRAG — it more than doubles the retrieval-failure rate. (Note expansion slightly *helps* answer correctness — the chunks aren't garbage, they're relevant to sub-questions but off-topic for the metric's reference question.)

### 2.3 A shared ceiling = the llama3:8b generator misses answers it already has
Even with good context (precision ≥ 0.7), the generator still produces a wrong answer (correctness ≤ 0.2) on:

| System | "good context, wrong answer" rows | total answer failures (≤0.2) |
|---|---|---|
| standard | 41 / 195 | 85 |
| crag | 27 / 195 | 78 |
| cragpp | 21 / 195 | 107 |

~21% of Standard's questions are *generation* failures, not retrieval failures. This caps answer-correctness (~0.47) for **all** architectures and is independent of the CRAG machinery. Answer length is uncorrelated with correctness (r≈0 every system), so verbosity is not the issue.

**Summary:** Standard wins because focused top-4 retrieval stays on-topic for the exact question scored. CRAG bleeds recall via a harmful grader; CRAG++ bleeds precision via noisy sub-question retrieval; everyone shares a generator ceiling.

## 3. SOTA techniques that fit this project (2025–2026)

- **Two-stage retrieve-then-rerank is the dominant pattern.** Hybrid retrieval + neural reranking hits Recall@5 0.816 / MRR@3 0.605, beating single-stage (Hybrid-RRF alone 0.695); reranking alone gives +15–25% relevance. ([benchmark][bm], [reranking guide][rr], [patterns][p2026])
- **Hybrid (BM25 + dense, fused with RRF)** — best recall floor, especially for entity-heavy queries (country names, programme acronyms like NAPLAN/PIRLS). ([atlan][atlan])
- **Corrective-RAG done properly** uses a graded relevance *score* (Correct / Ambiguous / Incorrect) + knowledge refinement / fallback, not a weak binary LLM filter. ([atlan][atlan])
- **Adaptive / selective retrieval (Self-RAG family)** — decide *whether* to decompose/retrieve per query instead of always. ([blueprint][bp])
- **Sentence-window retrieval ranked #1 for precision** in the ARAGOG head-to-head (beat HyDE, multi-query, MMR, Cohere/LLM rerank). ([ARAGOG][aragog]) HyDE underperformed plain dense retrieval — skip it.
- Straightforward retrieval answers ~44% of factual Qs; advanced (rerank+hybrid+adaptive) ~63%. ([reranking patterns][p2026])

## 4. Improvement roadmap (ordered by impact ÷ effort)

Two libraries are **already in `requirements`** — `rank_bm25` (BM25) and a CPU cross-encoder reranker — so the top levers need no new heavy deps.

> **Status (2026-06) — what was actually done.** Reranking was implemented with sentence-transformers
> `CrossEncoder` (`ms-marco-MiniLM-L-6-v2` default; [`src/vectorstore/rerank.py`](../src/vectorstore/rerank.py)),
> **not** `flashrank`. Roadmap outcomes:
>
> | # | Step | Status | Result |
> |---|---|---|---|
> | T1.1 | Cross-encoder reranking | ✅ done | AC 0.473→**0.625** (+32%) |
> | T1.2 | Hybrid BM25+dense+RRF → rerank | ✅ done | E9c, AC **0.689** (recall 0.909) |
> | T2.3 | Reranker-score grader (CRAG) | ❌ not done | CRAG **abandoned**, not fixed (ship Standard) |
> | T2.4 | CRAG++ union-rerank vs original q | ❌ not done | CRAG++ abandoned |
> | T2.5 | Adaptive decomposition | ❌ not done | CRAG++ abandoned |
> | T3.6 | Stronger generator | ✅ done | deepseek 0.595/0.717; gpt-5.4-mini 0.652/**0.774** (best) |
> | T3.7 | CRAG++ synthesize from reranked ctx | ❌ not done | CRAG++ abandoned |
> | T4.8 | Smaller chunks / sentence-window | ⚠️ partial | E2 512/64 done (+3%); sentence-window not |
> | T4.9 | Metadata filtering by country | ❌ not done | deprioritized by §5d analysis |
>
> The 5 not-done items are the **CRAG/CRAG++ fixes (T2.3–2.5, T3.7)** and metadata (T4.9): the program chose
> to **ship Standard and drop CRAG/CRAG++** (architecture_evolution_analysis §7.2) rather than repair them.
> Also un-run: E1 (k_final 4/6/8 sweep), E5 (grader ablation), E6 (bge-reranker-v2-m3 / mxbai), E7 (embedder
> swap). The reranker swap that *was* tested — bge-reranker-base (E9b/E9c/E12) — under-performed MiniLM on
> dense-only but **matches/beats it under hybrid**, so bge-base is the production default.

### Tier 1 — Retrieval (biggest lever; helps ALL three systems)
1. **Cross-encoder reranking.** Retrieve top-20 dense → rerank with a sentence-transformers `CrossEncoder` → keep top-4. Wrap the retriever so every graph inherits it. Expected: precision + recall up across the board; directly attacks the 40–107 "retrieval-failure" rows. *(Done — implemented as [`src/vectorstore/rerank.py`](../src/vectorstore/rerank.py).)*
2. **Hybrid BM25 + dense with RRF**, then rerank (Tier-1.1). Build a BM25 index over the same chunks, reciprocal-rank-fuse with dense, feed the fused top-20 into the reranker. Targets entity questions where dense alone misses.

### Tier 2 — Fix the CRAG-specific defects
3. **Replace the binary gemma3:1b doc grader with reranker-score thresholding (real Corrective-RAG).** Keep chunks above a relevance score; only trigger sub-question correction when the top score is low ("Ambiguous/Incorrect" band). This deletes the recall-killer in §2.1 (0.486 → ~0.82 on the affected 87 rows). The reranker subsumes the grader's job — gemma3:1b can be dropped entirely.
4. **CRAG++: rerank the per-sub-question UNION against the ORIGINAL question, keep top-6.** Re-aligns the used/logged context with what the precision metric scores and caps chunk count — the direct fix for §2.2 (0.449 precision). Files: `src/nodes/cragpp_nodes.py` (after the `final_contexts` union, before generation/logging).
5. **Adaptive decomposition.** Only decompose when the question is genuinely multi-hop (e.g. first-pass reranked top score is low, or a cheap router classifies it). Most PIRLS questions are single-hop; skipping decomposition routes them down the Standard path and stops CRAG++ from harming the easy majority (the 143 expanded rows).

### Tier 3 — Generation (the shared ceiling on answer correctness)
6. **Stronger generator.** llama3:8b misses ~21% of in-context answers. Swap to a larger/instruct local model or a hosted one (the `compat:`/API generator paths already exist in `config_api.py`). This is the only lever that moves answer-correctness past ~0.47 for *every* system.
7. **Synthesize from reranked context, not just sub-answers.** CRAG++ produced wrong answers even with perfect context (precision 1.0, correctness 0.0 — e.g. row 83) because final synthesis uses only the sub-answers. Include the top reranked chunks in the synthesis prompt so facts aren't lost in the relay (`cragpp_nodes.py` synthesis step).

### Tier 4 — Retrieval representation (if Tier 1–2 plateaus)
8. **Sentence-window or smaller chunks** (current 1000/100 is coarse for ~21-word answers) — ARAGOG's top precision technique. 
9. **Metadata-filtered retrieval by country/doc-type** — most questions name a country; cheap precision win on cross-country confusions.

## 5. How to validate (reuse the r3 harness)
The infrastructure already supports clean A/Bs: change retrieval/grading, regenerate with `scripts/run_generation.py`, score with `scripts/run_eval.py --judge compat:openai/gpt-oss-120b`, compare via `scripts/summarize_r3.py`. Recommended sequence, gated on the 15-row pilot each time:
1. Baseline = current r3 (done).
2. + reranking (Tier 1.1) on **Standard** → confirm precision/recall lift before touching CRAG.
3. + hybrid (Tier 1.2).
4. CRAG with reranker-grader (Tier 2.3) and CRAG++ with union-rerank + adaptive decomposition (2.4/2.5).
5. Stronger generator (Tier 3) as a separate axis.
Track all five metrics incl. `gold_context_similarity` (the judge-free retrieval-vs-gold signal), overall + per `data_type`.

## 5b. The gemma3:1b grader is net-negative (measured)

CRAG retrieves identically to Standard *before* grading, so any gap on graded rows is the grader's doing.

| CRAG rows | recall | precision | answer corr | gold-sim |
|---|---|---|---|---|
| grader **dropped** chunks (87/195) | 0.486 | 0.590 | 0.379 | 0.699 |
| grader **kept** all 4 (108/195) | 0.819 | 0.771 | 0.550 | 0.731 |
| Standard (no grader, same retrieval) | 0.796 | 0.741 | 0.473 | 0.741 |

- Over-aggressive: on the 87 dropped-rows it kept **1 chunk on 52 rows**, 2 on 30 → starves the generator.
- Poor judgment, not just strict: precision *falls* on dropped rows (0.590 vs 0.771) — it keeps the wrong chunks.
- Deletes the gold chunk in ~14 rows (gold-sim <0.5, ~2× the kept-row rate).
- Standard (never grades) beats CRAG outright → the grader's "quality control" removes value.

**Fix priority:** (1) replace binary LLM grading with **reranker-score thresholding** (Corrective-RAG proper —
the cross-encoder being added subsumes the grader, more accurate + faster); (2) failing that, soften it
(never drop below top-k) or remove it; (3) cheap quick win: gemma3:1b → gemma3:4b or self-grade with the
generator. The answer-usefulness grader (retry gate) is the same 1B model and suspect for the same reasons.
Validate via a grader-ablation experiment (E5).

## 5c. Embedding & reranker model upgrades (evaluation)

Current models are both the weakest tier: embedder `all-mpnet-base-v2` (2021, 768-dim), reranker
`ms-marco-MiniLM-L-6-v2` (2021, ~22M, the lightweight sentence-transformers option).

**Effect on answer correctness is real but bounded.** AC = retrieval surfaces gold × generator extracts it.
Embedder/reranker improve only retrieval: measured gold-chunk recall@4=63%, recall@20=87% (mpnet) → a
better embedder lifts the candidate-pool ceiling, a better reranker converts more in-pool gold into top-4.
But ~21% of questions already have good context and a wrong answer (generator ceiling, llama3:8b), and the
reranked run already reached P=0.87/R=0.86 — so expect **single-digit AC gains**, not another +30%. After
reranking, the generator is the binding constraint.

**Priority:**
1. **Reranker swap (cheap, no reindex):** MiniLM-L-6 → `bge-reranker-v2-m3` (Apache, <600M, 2026 default)
   or `mxbai-rerank-base-v2` (0.5B, beats Cohere/Voyage on BEIR). One-line change in
   `src/vectorstore/rerank.py`; CPU-runnable. **E6.**
2. **Embedder swap (needs reindex):** `all-mpnet` → `BGE-M3` (MIT workhorse) or `Qwen3-Embedding`; for a
   CPU-cheap test use `nomic-embed-text` (already on the box) or a DeepInfra-hosted embedder. **E7.**
3. The largest remaining AC lever is the **generator**, not retrieval (see E4).

## 5d. Cross-country retrieval confusion (diagnostic) — metadata filtering NOT worth it

Tested whether per-country metadata filtering would help (PIRLS has ~70 near-identical per-country chapters).
For the 79/195 questions naming a single country (115 name none, 1 multi):

| metric | result |
|---|---|
| top-4 source mix | 75% correct-country, **17% other-country**, 8% cross-country exhibit/methodology |
| questions with ≥1 wrong-country chunk in top-4 | 44% |
| questions whose **top-1** chunk is wrong-country | **4%** |

Confusion is **real but mild and concentrated in positions 2–4** (top-1 almost always right). Reranking
(cross-encoder, query-vs-chunk) demotes wrong-country chunks, so it largely subsumes this. Only ~40% of
questions are country-specific; a hard filter would risk the ~60% non-country / cross-country-table questions.
**Verdict: deprioritize metadata filtering** — revisit only if reranked results still show wrong-country
chunks in top-4. Likewise **skip multi-query expansion** (reranking already covers retrieval) and **hold
self-consistency** (E8 prompt + E4 generator are cheaper routes to the same generator-ceiling gain).

## 5e. Fine-tuning assessment (generator / embedder / reranker)

**Verdict: don't fine-tune yet — highest effort, lowest certainty, and the training-data situation is weak.**

**Critical data-leakage finding.** The new 195-row eval set is a **100% exact subset of the old 300-row set**
(`datasets/original/datasets.xlsx`): all 195 eval questions appear verbatim in the 300. So the 300
= the 195 eval questions + 105 others. **The 300 cannot be used as training data** — it contains the entire
test set; training on it would memorize eval answers and invalidate every metric. Only the **~105
non-overlapping rows** are clean — too few for reliable fine-tuning (overfitting risk, little signal).

**Expected gain.** Modest and uncertain. Reranking already lifted Standard AC 0.473→0.625, so the generator
is the main remaining lever — but most of what a generator LoRA would teach (brevity, exact extraction, the
"Sorry, I don't know" behavior) is what the **E8 prompt** achieves for ~$0. With ~105 clean examples, expect
**low-single-digit gains at best, high variance, possibly negative**. Nothing like reranking's +32%.
Embedder/reranker fine-tuning hits diminishing returns too (off-the-shelf reranking already reached P≈0.84).

**If pursued anyway, requirements:**
1. Clean split — ~105 non-eval rows + **synthetic QA generated from the corpus PDFs** (target ~500–1,000);
   the 195 eval set held out, never trained on.
2. Generator training format = **(question + retrieved context → answer)**, not bare (Q→A), to teach
   grounding rather than hallucination.
3. GPU (~16 GB for 8B QLoRA) + tooling (Unsloth / Axolotl / HF PEFT+TRL; sentence-transformers for
   embedder/reranker), LoRA→GGUF merge for Ollama serving, and held-out eval vs the reranked baseline.

**Recommendation:** exhaust the cheap levers first (reranking ✓, E8 prompt, E4 generator swap, off-the-shelf
reranker/embedder upgrades). Consider a generator QLoRA only if a clear generation gap remains after E4/E8
**and** enough clean synthetic training data can be produced — with modest expectations.

## 6. Hyperparameter findings (retrieval-only sweep, no LLM)

Measured on the index + 195-row gold contexts (`reference_context`), 100-row sample for the rank sweep.
Current settings: **chunk_size=1000 / overlap=100** (`config.py:15-16`), **k=4** (`run_generation.py`),
gen temp=0 / num_ctx=8192 / max_tokens=1024.

**Granularity mismatch — chunks are ~2.3× the gold passage:**
| Quantity | mean | p50 | p90 |
|---|---|---|---|
| index chunk (chars) | 832 | 943 | 989 (max 1000) |
| gold `reference_context` (chars) | 363 | 304 | 634 |
| `reference_answer` (words) | 22.7 | 22 | 39 |

Each retrieved 1000-char chunk carries ~600 chars of non-answer text around a ~300-char gold span —
diluting contextual precision and padding the generation prompt with distractor text.

**k=4 is too shallow — gold chunk rank within the question's top-30 (n=100):**
| within top-k | 1 | 2 | 3 | **4** | 6 | 8 | 10 | 15 | 20 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|
| % questions with gold chunk captured | 35 | 50 | 61 | **63** | 67 | 69 | 77 | 85 | 87 | 100 |

- **37% of questions have their gold-bearing chunk *beyond* top-4** — k=4 literally omits the answer
  chunk in over a third of cases. This is a direct, measured cap on both recall and answer correctness.
- recall jumps to **87% at top-20**. That is the precise justification for **retrieve-wide-then-rerank**:
  a wide first stage (k=20) captures the gold chunk 87% of the time; the cross-encoder then promotes it
  into the top-4 the generator sees — recovering ~24 points of gold context that direct top-4 misses,
  without flooding the prompt.
- best-gold cosine mean 0.786 (p10 0.644) — when present, the gold chunk is clearly identifiable, so a
  reranker has strong signal to surface it.

**Hyperparameters ranked by expected answer-correctness impact:**
1. **Retrieval depth + reranking (k_candidates 20 → rerank → k_final).** Closes the 63%→87% gold-capture
   gap. Highest lever; already under test in the Standard rerank A/B. Also sweep **k_final = 4 / 6 / 8**
   (more post-rerank context captures more gold; rerank controls the added noise).
2. **chunk_size 1000→~512, overlap 100→64 (reindex, ~10–25 min once).** Shrinks chunks toward the
   ~300–630-char gold span → higher precision and a cleaner generation prompt. ARAGOG's top precision
   technique. Guard the p90 (634-char gold) with a parent-document retriever (match small, feed parent)
   so multi-sentence answers aren't fragmented.
3. **Generation context size** = k_final above; fold into the rerank sweep.
4. **Not bottlenecks:** temp=0 (keep), num_ctx=8192 and max_tokens=1024 are ample for 4–8 chunks / ~23-word
   answers. The residual ~21% "good-context-but-wrong-answer" rows are a *generator-model* ceiling
   (llama3:8b), addressed by Tier 3, not by any hyperparameter.

**Clean experiments to run (after the current A/B, gated on the 15-row pilot each):**
- E1: k_candidates=20, rerank, **k_final ∈ {4,6,8}** on Standard — pick the knee.
- E2: reindex at **512/64** (+ optional parent-document) vs current 1000/100, same retrieval — isolates chunking.
- E3: best of E1×E2 carried into CRAG/CRAG++.

---
### Sources
- [From BM25 to Corrective RAG: benchmarking retrieval strategies][bm] (arXiv)
- [Top reranking models for RAG][rr] (MachineLearningMastery)
- [Advanced retrieval patterns that actually work in 2026][p2026] (dev.to)
- [12 advanced RAG techniques][atlan] (Atlan)
- [RAG in 2026: a practical blueprint][bp] (dev.to)
- [ARAGOG: Advanced RAG Output Grading][aragog] (arXiv)

[bm]: https://arxiv.org/html/2604.01733v1
[rr]: https://machinelearningmastery.com/top-5-reranking-models-to-improve-rag-results/
[p2026]: https://dev.to/young_gao/rag-is-not-dead-advanced-retrieval-patterns-that-actually-work-in-2026-2gbo
[atlan]: https://atlan.com/know/advanced-rag-techniques/
[bp]: https://dev.to/suraj_khaitan_f893c243958/-rag-in-2026-a-practical-blueprint-for-retrieval-augmented-generation-16pp
[aragog]: https://arxiv.org/pdf/2404.01037
