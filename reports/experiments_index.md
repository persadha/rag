# RAG experiments — master index & findings (r3 program)

Single entry point for reviewing this work. Snapshot: 2026-06-20. Branch `RAG-2`.
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
- `results/eval_r3_standard_{gemma-deepinfra,ollama_deepseek}_deepeval.csv` — (OSS) gemma / deepseek scores.
- `results/eval_r3_standard_openai-mini*_deepeval.csv` — (E4) GPT-5.4-mini scores.
- `results/eval_r3_*_extract_deepeval.csv` — (E8) extraction-prompt scores.
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

## Key levers (ranked, for answer correctness)
1. **Wider retrieval + reranking** — gold chunk in top-4 only 63%, top-20 87%; reranking recovers it.
   **CONFIRMED (full 195): reranked Standard AC 0.473→0.625 (+32%), P 0.741→0.837 (+13%), R 0.796→0.837 (+5%).**
   Worst-retrieval rows (baseline precision<0.5, n=40): precision 0.13→0.53, AC 0.19→0.33. Biggest lever found, ~$0.
2. **Stronger generator** — the ~21% good-context-wrong-answer ceiling (E4 tests GPT-5.4-mini).
3. **Better prompt** (E8), **smaller chunks** (E2, 512/64), **reranker/embedder upgrades** (deferred), **grader fix**.
Deprioritized: metadata-by-country (mild confusion, subsumed by rerank), multi-query, self-consistency.

## Experiment status
| Phase | What | Status |
|---|---|---|
| r3 | 3 systems × 195, llama3:8b | ✅ done |
| Rerank A/B | Standard, retrieve-20→rerank→4 | ✅ done — AC +32%, P +13%, R +5% |
| E2 | chunk-512/64 reindex vs 1000/100 | ⏳ queued |
| OSS gens | gemma-3-4b (DeepInfra) + deepseek-r1:8b (local), Standard-only | ⏳ queued |
| E4 | GPT-5.4-mini on auto-selected best pipeline (open vs closed / privacy) | ⏳ queued |
| E8 | extraction-prompt A/B on best pipeline | ⏳ queued |
Available on demand: gemma-3-4b on CRAG/CRAG++; E6 reranker swap; E7 embedder swap; self-consistency.

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
