# Why the RAGAS evaluation could not be completed — root-cause diagnosis

> plans.md item 2. Every claim below was verified directly in the committed notebooks/results
> (references are `notebook :: cell N`). Date of diagnosis: 2026-06-12.

## TL;DR

RAGAS did not fail for one reason — it failed in a cascade of four, and it did not fail
*everywhere*: the **standard-RAG evaluations completed** in December 2025 with a paid OpenAI
judge. The cascade hit the **advanced-RAG (CRAG) evaluations** in January 2026:

1. **OpenAI quota exhaustion** mid-run (HTTP 429) killed the paid-judge path;
2. the rescue attempt — **local Ollama judges** — collided with RAGAS's structured-output
   machinery (executor timeouts + unparseable judge output → NaN scores);
3. **NaN answers produced by CRAG's dead-end bug** crashed RAGAS's Pydantic validation outright;
4. unpinned, drifting RAGAS versions (0.4.1 vs 0.4.2) with **two incompatible metric APIs** made
   every notebook behave differently.

DeepEval succeeded afterwards not because it is "better" but because its metrics tolerate a
local judge (it ran sequentially with gpt-oss:20b) and skip bad rows instead of crashing.

## Verified timeline

| When | What | Evidence |
|---|---|---|
| 2025-12-20 | Standard-RAG RAGAS evals **completed** with judge `gpt-4o-mini` over the OpenAI API (`llm_factory("gpt-4o-mini", client=AsyncOpenAI(), max_tokens=4096)`). Wall times per metric: 45 min – 1 h 39 min for ~300 rows. | `eval_llama.ipynb :: cells 1, 6, 9, 13, 16`; result files `results/eval_*_llama_20251220.csv` (+ gemma, deepseek twins) |
| 2026-01-05…08 | Advanced-RAG evals with the same OpenAI judge **died on quota**: `Error code: 429 — You exceeded your current quota` at row 163, surfaced as `InstructorRetryException` (RAGAS's answer-correctness/factual-correctness metrics call the judge through `instructor`, which retries then raises). | `eval_adv_llama.ipynb :: cells 9, 13` |
| same runs | Rows where CRAG produced an empty/NaN answer **crashed validation**: `ValidationError: ClaimDecompositionInput.response — Input should be a valid string [input_value=nan, input_type=float]`. (Root cause: the old CRAG dead-end — all docs graded irrelevant → graph ends with empty answer → NaN in the CSV; fixed by the P0 grader fallback, see context.md §4.3/§7.) | `eval_adv_llama.ipynb :: cell 16`; `eval_adv_gemma.ipynb :: cell 21` |
| 2026-01-08 | Rescue attempt: install **Ollama inside the eval Colab** and judge locally (`ChatOllama` + `LangchainLLMWrapper`, ragas 0.4.2, deprecated `ragas.metrics` API with a printed DeprecationWarning). Ollama itself struggled: `Error: 500 Internal Server Error: timed out waiting for llama runner to start - progress 0.28`. | `eval_adv_gemma.ipynb :: cells 0, 3–5` |
| same run | Batch `ragas.evaluate()` with the local judge: `ERROR:ragas.executor:Exception raised in Job[0]/Job[37]: TimeoutError()` plus `ERROR:ragas.prompt.pydantic_prompt:Prompt fix_output_format failed to parse output … including retries` → **NaN scores** visible in the printed `context_precision_score` column. The manual faithfulness loop skipped rows (`Skipping evaluation for index 2 due to: ValueError`). | `eval_adv_gemma.ipynb :: cells 11, 14` |
| same run | AnswerCorrectness still routed through `instructor` → hit the **OpenAI 429 again** (local judge didn't cover the instructor-backed metrics in this configuration). | `eval_adv_gemma.ipynb :: cells 14 (output), 18` |
| mid-January | Dedicated diagnostic notebooks tried **bigger local judges through Ollama's OpenAI-compat `/v1` endpoint** (`gpt-oss:20b`, `gemma3:27b`; `RunConfig(max_workers=5, timeout=600)`). Outcome: `ERROR:ragas.executor:Exception raised in Job[0]: TimeoutError()` — a 20B judge on Colab cannot answer RAGAS's long structured prompts within 600 s at concurrency 5. | `Understanding RAGAS Answer Correctness Metric.ipynb :: cells 4, 6, 8`; `…Ver 2 (gemma 27b).ipynb :: cells 4, 6`; same `/v1` + `gpt-oss:20b` setup in `eval_adv_deepseek.ipynb :: cell 5` |
| 2026-01-18…20 | Switch to **DeepEval** with local judge gpt-oss:20b → the r2 result files (`eval_r2_*_deepeval_cpr_cr_fa_*.xlsx`, `deepeval_*_ac_*.xlsx`) completed for all six systems. | `results/` listing; context.md §2 Pass 2 |

## Ranked root causes

1. **Quota exhaustion of the paid judge (proximate trigger).** The December standard runs
   consumed the OpenAI budget (4 metrics × ~300 rows × 3 models, 45–99 min each); the January
   advanced runs then died at row ~163 with 429s. Everything after this was an attempt to evaluate
   without a paid judge.
2. **RAGAS's structured-output pipeline is hostile to local judges.** Its metrics demand
   schema-conformant JSON (Pydantic/instructor) from the judge and run them concurrently
   (`max_workers=5`). Sub-30B local models on Colab either exceeded the 600 s timeout
   (`TimeoutError()` from the executor) or returned output the parser could not repair
   (`fix_output_format failed … including retries`) — both degrade to NaN scores silently.
   The notebook authors' own `max_tokens=4096` comment in `eval_llama.ipynb :: cell 1`
   ("prevent truncation of structured outputs") shows parser fragility was visible even on the
   paid path.
3. **NaN answers from the (old) CRAG dead-end bug.** Empty answers became `nan` floats in the
   results CSV and crashed `ClaimDecompositionInput` validation — so even with a healthy judge,
   the advanced-RAG answer-correctness pass could not finish until the generator bug was fixed
   (it now is: P0 grader fallback).
4. **Version/API drift (aggravating factor).** `!pip install ragas` unpinned → 0.4.1 in
   December, 0.4.2 in January; standard notebooks use the new `ragas.metrics.collections` API,
   advanced notebooks the deprecated `ragas.metrics` + `LangchainLLMWrapper` path (printed
   DeprecationWarning, `eval_adv_gemma.ipynb :: cell 5`) — different failure modes per notebook
   and no reproducibility. `requirements.txt` pins neither `ragas` nor `deepeval`.

## Why DeepEval worked where RAGAS didn't

- Its metric prompts are simpler per call and were run **sequentially** — no 5-way concurrency
  squeezing a 20B model on shared Colab hardware, so gpt-oss:20b stayed under timeout.
- Failure handling: bad rows are skipped/scored-as-error instead of raising
  `InstructorRetryException`/`ValidationError` and killing the whole pass.
- No dependency on the OpenAI quota: the judge ran on local Ollama natively.

The trade-off: the r2 DeepEval pass has its own artifact (CRAG's `retrieved_context` logged as a
single blob → zeroed contextual precision/recall, context.md §4.3) — so neither pass is a clean
baseline; hence the planned r3 re-run.

## Implications for the r3 re-run (already reflected in the plan)

- **Judge = paid API model (`gpt-4.1`) with confirmed quota**, fixed across runs — eliminates
  causes 1 and 2 at once; cost is gated by a 10–15-row pilot extrapolation before any full run.
- **NaN guard**: the P0 grader fallback removes empty CRAG answers at the source; the new
  harness additionally validates non-empty string answers before scoring (cause 3).
- **Pinned versions** in requirements for the eval stack (cause 4).
- Context logged as **chunk lists** for every system, so the artifact that plagued r2 cannot recur.
