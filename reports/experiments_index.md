# RAG experiments — master index & findings (r3 program)

Single entry point for reviewing this work. Snapshot: 2026-06-21. Branch `RAG-2`.
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

**Best system: E4 (GPT-5.4-mini + rerank) at AC=0.652.** Key new findings:
- **Reranking is the single biggest lever** (+32% AC over baseline) and free (~$0, CPU cross-encoder).
- **Stronger generator matters too:** DeepSeek-r1:8b (no rerank) beats llama3:8b+rerank (0.595 vs 0.625); GPT-5.4-mini+rerank reaches 0.652.
- **Extraction prompt (E8) backfires on llama3:8b:** AC drops 0.625→0.482 despite better retrieval metrics — the brevity instruction confuses the model rather than helping it.
- **Smaller chunks (E2) marginal:** AC 0.473→0.488 (+3%), but precision and recall both drop slightly. Gold-context similarity improves (+0.034), suggesting chunks are cleaner but retrieval k=4 is still too shallow.

## Key levers (ranked, for answer correctness)
1. **Wider retrieval + reranking** — gold chunk in top-4 only 63%, top-20 87%; reranking recovers it.
   **CONFIRMED (full 195): AC 0.473→0.625 (+32%), P 0.741→0.837 (+13%), R 0.796→0.837 (+5%).** ~$0 cost.
2. **Stronger generator** — DeepSeek-r1:8b adds +26% over llama3:8b with same retrieval; GPT-5.4-mini+rerank is best overall (0.652).
3. **Smaller chunks** (E2, marginal +3% AC), **grader fix** (CRAG recall), **CRAG++ redesign**.
4. **Extraction prompt does not help** with llama3:8b — hurts AC significantly (-23%).
Deprioritized: metadata-by-country (mild confusion, subsumed by rerank), multi-query, self-consistency, gemma-3-4b generator.

## Experiment status
| Phase | What | Status |
|---|---|---|
| r3 | 3 systems × 195, llama3:8b | ✅ done |
| Rerank A/B | Standard, retrieve-20→rerank→4 | ✅ done — AC +32%, P +13%, R +5% |
| E2 | chunk-512/64 reindex vs 1000/100 | ✅ done — AC +3%, P/R slightly down |
| OSS: deepseek-r1:8b | Standard, local deepseek-r1:8b, no rerank | ✅ done — AC 0.595 (+26% vs llama3:8b) |
| E4 | GPT-5.4-mini + rerank on Standard | ✅ done — AC 0.652 (best overall) |
| E8 | rerank + extraction-prompt A/B | ✅ done — AC 0.482 (regressed; extract prompt hurts llama3:8b) |
| OSS: gemma-3-4b | Standard, gemma-3-4b (DeepInfra) | ❌ not run — deprioritized |
Available on demand: gemma-3-4b on any system; E6 reranker swap; E7 embedder swap; CRAG/CRAG++ + rerank (E3); self-consistency.

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
