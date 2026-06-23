# RAG experiments — master index & findings (r3 program)

Single entry point for reviewing this work. Snapshot: 2026-06-23 (E13 CRAG++fixed+mini + closed-book gpt-5.4-mini added; 2×2 value-added table complete). Branch `RAG-3`.
PIRLS RAG benchmark, 195-question revised dataset (`datasets/revision/evaluation_dataset.xlsx`),
local generation, DeepInfra `gpt-oss-120b` judge (held constant across ALL runs for fair comparison),
5 metrics: contextual precision/recall, faithfulness, answer correctness, gold-context similarity.

## Reports (read these)
- **`reports/crag_improvement_analysis.md`** — the main analysis:
  - §1–2 why Standard beat CRAG/CRAG++ (measured, three mechanisms)
  - §5b gemma3:1b grader is net-negative (measured)
  - §5c embedding/reranker model upgrade evaluation
  - §5d cross-country confusion diagnostic (metadata filtering deprioritized)
  - §5e fine-tuning assessment (don't yet; old 300-set is 100% eval leak → only ~105 clean rows)
  - §6 hyperparameter findings (k too small at 4; chunks 2.3× the gold passage)
  - SOTA roadmap + experiment list (E1–E8) + sources
- **`reports/plans_response.md`** — answers to the original six plans.md items; r3 result slots filled.
- **`context.md` §2.0** — the r3 results table + headline findings (in the repo root, not reports/).
- `reports/ragas_diagnosis.md`, `reports/ui_review.md` — earlier-stage reports.

## Result artifacts
- **`results/eval_r3_summary.xlsx`** — per-system metric means, overall + human/synthetic split (3 sheets).
- `results/eval_r3_{standard,crag,cragpp}_ollama_deepeval.csv` — r3 per-row scores (llama3:8b).
- `results/eval_r3_standard_ollama_rerank_deepeval.csv` — reranked Standard scores.
- `results/eval_r3_standard_ollama_chunk512_deepeval.csv` — (E2) 512/64 chunk scores.
- `results/eval_r3_standard_ollama_deepseek_deepeval.csv` — (OSS) deepseek-r1:8b scores.
- `results/eval_r3_standard_openai-mini_rerank_deepeval.csv` — (E4) GPT-5.4-mini + rerank scores.
- `results/eval_r3_standard_ollama_rerank_extract_deepeval.csv` — (E8) rerank + extraction-prompt scores.
- Generation CSVs `results/gen_r3_*.csv` (large, gitignored); phase logs `results/_*.log`.
- `results/_archive_old_dataset/` — superseded old-300-row-dataset pilots.

## Headline result (r3, llama3:8b, n=195/system)
| System | Ctx Prec | Ctx Recall | Faith | Answer Corr | Gold-Ctx Sim |
|---|---|---|---|---|---|
| **standard** | **0.741** | **0.796** | 0.939 | **0.473** | **0.741** |
| crag | 0.690 | 0.671 | **0.948** | **0.474** | 0.717 |
| cragpp | 0.449 | 0.650 | 0.910 | 0.334 | 0.712 |

**Standard wins; CRAG ≈ Standard on answer quality; CRAG++ regressed.** Causes (measured):
CRAG's gemma3:1b grader drops relevant chunks (recall 0.49 vs 0.82); CRAG++'s per-sub-question retrieval
injects off-topic chunks (precision 0.45). Shared ceiling: llama3:8b fails ~21% of rows that HAVE good context.

## Full experiment results (all runs, n=195, judge=gpt-oss-120b)

| Run | Generator | Config | Ctx Prec | Ctx Recall | Faith | Answer Corr | Gold-Ctx Sim |
|---|---|---|---|---|---|---|---|
| Standard (baseline) | llama3:8b | k=4, 1000/100 chunks | 0.741 | 0.796 | 0.939 | 0.473 | 0.741 |
| CRAG | llama3:8b | k=4, gemma3:1b grader | 0.690 | 0.671 | 0.948 | 0.474 | 0.717 |
| CRAG++ | llama3:8b | k=4 per sub-q, dedup | 0.449 | 0.650 | 0.910 | 0.334 | 0.712 |
| **Standard + rerank** | llama3:8b | k=20→rerank→4 | **0.837** | 0.837 | 0.940 | 0.625 | 0.755 |
| E2: chunk-512/64 | llama3:8b | k=4, 512/64 chunks | 0.723 | 0.759 | 0.939 | 0.488 | 0.775 |
| OSS: DeepSeek-r1:8b | deepseek-r1:8b | k=4, 1000/100 chunks | 0.742 | 0.804 | 0.983 | 0.595 | 0.741 |
| **E4: GPT-5.4-mini + rerank** | gpt-5.4-mini (openai) | k=20→rerank→4 | **0.837** | **0.847** | **0.981** | **0.652** | 0.755 |
| E8: rerank + extract prompt | llama3:8b | k=20→rerank→4, extract style | 0.825 | 0.864 | 0.974 | 0.482 | 0.755 |
| E9b: dense + bge-base rerank | llama3:8b | k=20→bge-reranker-base→4 | 0.807 | 0.850 | 0.957 | 0.578 | 0.748 |
| **E9c: hybrid + bge-base rerank** | llama3:8b | BM25+dense→RRF→bge-base→4 | 0.855 | 0.909 | 0.953 | 0.689 | 0.756 |
| E12: hybrid + MiniLM rerank | llama3:8b | BM25+dense→RRF→MiniLM→4 | 0.867 | 0.920 | 0.950 | 0.679 | 0.760 |
| E10: hybrid + bge-base | deepseek-r1:8b | BM25+dense→RRF→bge-base→4 | 0.853 | 0.904 | 0.985 | 0.717 | 0.756 |
| **E11: hybrid + bge-base** | gpt-5.4-mini | BM25+dense→RRF→bge-base→4 | 0.865 | 0.914 | **0.984** | **0.774** | 0.756 |
| **E13: CRAG++fixed + hybrid + bge** | gpt-5.4-mini | CRAG++ w/ 4 fixes (T2.3–T2.5, T3.7) | 0.855 | 0.897 | — | **0.780** | — |

**Best overall: E13 (CRAG++fixed + gpt-5.4-mini + hybrid + bge) at AC=0.780** (+0.006 over E11). Best Standard config: **E11 AC=0.774**. Best open/local: **E9c (llama3:8b + hybrid + bge-base) AC=0.689**. Note: E13's +0.006 margin over E11 is marginal — Standard (E11) remains the recommended production config for its simplicity. Key findings:
- **Reranking is the single biggest lever** (+32% AC over baseline) and free (~$0, CPU cross-encoder).
- **Hybrid retrieval (E9) is the next lever: +0.111 AC** (E9c 0.689 vs E9b 0.578). BM25+dense+RRF lifts recall to 0.909+ — lexical matching catches exact terms (country names, acronyms) dense embeddings miss.
- **Generator is a strong second lever once hybrid is in place (E10/E11).** On identical hybrid+bge retrieval: llama3:8b **0.689** < deepseek-r1:8b **0.717** < gpt-5.4-mini **0.774**. But latency diverges sharply: gpt-5.4-mini 7.9 s (API) vs deepseek 62.4 s (local reasoning) — deepseek's +0.028 over llama costs ~3× the latency.
- **Reranker choice — bge ≥ MiniLM *under hybrid* (E12), the opposite of dense-only (E9b).** On hybrid, MiniLM (E12 0.679) slightly trails bge-base (E9c 0.689); on dense-only, MiniLM (0.625) beat bge-base (E9b 0.578). The BM25 fusion changes the candidate pool bge handles better. Net: **bge-base is the right reranker under hybrid** — resolving the E9 attribution caveat.
- **Retrieval still dominates at the open tier:** local llama + hybrid (0.689) > GPT-5.4-mini + plain rerank (E4, 0.652). The closed model only pulls ahead once it also gets hybrid retrieval (E11 0.774).
- **Extraction prompt (E8) backfires on llama3:8b:** AC drops 0.625→0.482 despite better retrieval metrics — the brevity instruction confuses the model rather than helping it.
- **Smaller chunks (E2) marginal:** AC 0.473→0.488 (+3%), but precision and recall both drop slightly. Gold-context similarity improves (+0.034), suggesting chunks are cleaner but retrieval k=4 is still too shallow.

## Key levers (ranked, for answer correctness)
1. **Hybrid retrieval + reranking + stronger generator (E11) — best overall, AC 0.774.** gpt-5.4-mini on BM25+dense+RRF→bge-base. Best open/local is E9c (llama3:8b) at 0.689.
2. **Hybrid retrieval (E9c)** — BM25+dense+RRF→bge-base lifts AC 0.473→0.689 (+46% over baseline, +10% over rerank-only), recall 0.909, ~$0 (local).
3. **Wider retrieval + reranking** — gold chunk in top-4 only 63%, top-20 87%; reranking recovers it.
   **CONFIRMED (full 195): AC 0.473→0.625 (+32%), P 0.741→0.837 (+13%), R 0.796→0.837 (+5%).** ~$0 cost.
4. **Stronger generator (additive on top of hybrid).** On hybrid+bge: llama 0.689 → deepseek 0.717 → gpt-5.4-mini 0.774. gpt-5.4-mini also fastest (7.9 s API); deepseek slow (62 s, local reasoning).
5. **Smaller chunks** (E2, marginal +3% AC), **grader fix** (CRAG recall), **CRAG++ redesign**.
6. **Does NOT help:** reranker swap MiniLM→bge-base on **dense-only** (−0.047 AC, E9b); extraction prompt on llama3:8b (−23%, E8). Under **hybrid**, bge ≥ MiniLM (E9c 0.689 ≥ E12 0.679).
Deprioritized: metadata-by-country (mild confusion, subsumed by rerank), multi-query, self-consistency, gemma-3-4b generator.

## Value-added ablation (reviewer R1 "is RAG worth it?" + R2 "value added by generation"), n=195, AC only
| Condition | AC | human | synthetic |
|---|---|---|---|
| Closed-book (llama3:8b, no retrieval) | 0.202 | 0.150 | 0.290 |
| **Closed-book (gpt-5.4-mini, no retrieval)** | **0.490** | — | — |
| Retrieval-only (hybrid+bge chunks as answer) | 0.827 | 0.852 | 0.785 |
| Full RAG (llama3:8b + hybrid + bge) | 0.689 | 0.753 | 0.581 |
| Full RAG (gpt-5.4-mini + hybrid + bge) | 0.774 | 0.787 | 0.751 |

**2×2 (model × retrieval):** open-llama closed-book 0.202 → +RAG 0.689 (+0.487); closed-mini closed-book 0.490 → +RAG 0.774 (+0.284). **RAG-llama (0.689) > closed-book-mini (0.490)** — the small open model with retrieval beats the large closed model without it.

- **R1: retrieval is decisive** — +0.488 (llama, vs closed-book), +0.572 at the ceiling. ⚠️ **R2 caveat:** retrieval-only (0.827) > full RAG is a **GEval coverage artifact** (a 4k-char chunk dump contains the gold facts and isn't penalized for verbosity), NOT evidence chunks beat synthesis. The real takeaway: **retrieval-bound, not generation-bound.** Full write-up: [`reviewer_response_r3.md`](reviewer_response_r3.md). Scripts: [`scripts/run_generation.py --no-retrieval`](../scripts/run_generation.py), [`scripts/make_retrieval_only.py`](../scripts/make_retrieval_only.py), [`scripts/build_value_added_table.py`](../scripts/build_value_added_table.py).

## Question classification (reviewer R1-2: cognitive level & domain), n=195
- **68.2% fact-retrieval / 31.8% reasoning** (human 77% fact, synthetic 53% fact). Methodology topics R1 named (sampling 4.6% + plausible-values 3.1% + weighting 1.5% ≈ **9%**) are under-sampled. Script: [`scripts/classify_questions.py`](../scripts/classify_questions.py); data `results/question_classification.csv`. KB scale: [`scripts/kb_stats.py`](../scripts/kb_stats.py) — 6,771 chunks / 108 docs / 1.5M tokens / 70.9 MB.

## Experiment status
| Phase | What | Status |
|---|---|---|
| r3 | 3 systems × 195, llama3:8b | ✅ done |
| Rerank A/B | Standard, retrieve-20→rerank→4 | ✅ done — AC +32%, P +13%, R +5% |
| E2 | chunk-512/64 reindex vs 1000/100 | ✅ done — AC +3%, P/R slightly down |
| OSS: deepseek-r1:8b | Standard, local deepseek-r1:8b, no rerank | ✅ done — AC 0.595 (+26% vs llama3:8b) |
| E4 | GPT-5.4-mini + rerank on Standard | ✅ done — AC 0.652 |
| E8 | rerank + extraction-prompt A/B | ✅ done — AC 0.482 (regressed; extract prompt hurts llama3:8b) |
| E9b | Standard, dense + bge-reranker-base | ✅ done — AC 0.578 (reranker swap regressed vs MiniLM 0.625) |
| E9c | Standard, hybrid BM25+dense+RRF + bge-base | ✅ done — AC 0.689 (best open/local; hybrid +0.111) |
| E12 | Standard, hybrid + MiniLM (isolate reranker under hybrid) | ✅ done — AC 0.679 (bge ≥ MiniLM under hybrid) |
| E10 | Standard, hybrid + bge, deepseek-r1:8b | ✅ done — AC 0.717 (slow, 62 s) |
| E11 | Standard, hybrid + bge, gpt-5.4-mini | ✅ done — AC 0.774 (best Standard config) |
| Value-added ablation | closed-book + retrieval-only baselines (llama) | ✅ done |
| Closed-book gpt-5.4-mini | gpt-5.4-mini, no retrieval | ✅ done — **AC 0.490** (2×2 complete; RAG-llama 0.689 > closed-mini 0.490) |
| E13: CRAG++fixed + mini | CRAG++ w/ 4 fixes, gpt-5.4-mini, hybrid + bge | ✅ done — **AC 0.780, program best** (marginal +0.006 over E11) |
| Question classification | 195 questions by level/domain | ✅ done |
| OSS: gemma-3-4b | Standard, gemma-3-4b (DeepInfra) | ❌ not run — deprioritized |
Available on demand: hybrid + bge-reranker-v2-m3 (heavier reranker); E7 embedder swap; CRAG/CRAG++ + hybrid.

## Model cost estimates (per full 3-system × 195 pass unless noted; judge ~$1–3 extra)
| Model | $/M in | $/M out | Est. cost | Notes |
|---|---|---|---|---|
| GPT-5.4-mini (closed) | 0.75 | 4.50 | ~$5–6 all-3 / **~$1–2 best-pipeline only (E4)** | reasoning model; use `reasoning_effort: low` |
| gemma-3-4b (DeepInfra) | 0.04 | 0.08 | ~$0.11 | exact match to local gemma3:4b |
| DeepSeek-R1-0528 (DeepInfra, 671B) | 0.50 | 2.15 | ~$7–32 | not used — too strong/different vs local 8B |
| R1-Distill-Qwen-32B (DeepInfra) | ~0.15–0.27 | ~0.15–0.27 | ~$1–4 | hosted reasoning distill option (not chosen) |
| deepseek-r1:8b (local) | — | — | $0 | exact local model; Standard-only (~6h CPU) |
Judge `gpt-oss-120b` via DeepInfra: $0.05/M in, $0.45/M out, ~$1–3 per full pass.

## Reproduce / run more
- Generate: `.venv/Scripts/python.exe scripts/run_generation.py --system <standard|crag|cragpp> --generator <ollama|gemma-deepinfra|openai-mini> [--rerank] [--persist-dir chroma_db_512 --tag chunk512] [--prompt-style extract]`
- Judge: `.venv/Scripts/python.exe scripts/run_eval.py --gen results/<genfile>.csv --judge compat:openai/gpt-oss-120b`
- Summarize: `.venv/Scripts/python.exe scripts/summarize_r3.py`
- Helpers: `scripts/select_best_pipeline.py` (auto-pick winner), `src/vectorstore/rerank.py` (cross-encoder).
- Keys in `.env`: `JUDGE_API_KEY` (DeepInfra, also used for gemma-deepinfra), `OPENAI_API_KEY` (E4).
