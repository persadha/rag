# Plan: Full r3 evaluation — open-source end-to-end

## Context

The deferred evaluation session from the plans.md staged execution (all six prior stages are
done and pushed on `RAG-2`; tracker in `context.md` §10). The user validated CRAG++ via the UI
and now wants the **full evaluation using open-source models** — rescoping the original design
(claude-haiku-4-5 + gpt-4.1) to an open-source stack.

**Decisions (asked & answered):**
- **Generator: local Ollama `llama3:8b`** (+ `gemma3:1b` grader for CRAG/CRAG++) — free, no
  key, the exact weights of the research's l1 runs, and the 35 pilot rows in `results/` resume
  into the full runs (same generator).
- **Judge: Groq-hosted `openai/gpt-oss-120b`** via deepeval's native
  `LocalModel(model, api_key, base_url)` — same model family as the r2 judge (gpt-oss:20b),
  fast (~1–2 h for ~9,000 judge calls), ~$3–6 on the paid tier. Open-source end-to-end.
- Scope: 3 systems × 300 rows × 4 metrics (ContextualPrecision, ContextualRecall, Faithfulness,
  answer-correctness GEval). The proprietary (Haiku) run is **not** part of this session; the
  harness keeps supporting it for later.

## What the user must prepare (the direct answer to their question)

1. **`GROQ_API_KEY` in `D:\rag\.env`** (copy `.env.example`) — create at console.groq.com.
   Strongly recommend adding a payment method (dev tier): the free tier's token-per-minute cap
   would throttle ~20M judge tokens to a crawl. Expected judge spend: **~$3–6**.
2. **Keep the laptop on and awake** during the generation window (~15 h total: standard
   ~1.4 h, CRAG ~5.4 h, CRAG++ ~7.9 h — sequential, ideally overnight). Plugged in, Windows
   sleep disabled (or at least "sleep never" while plugged in). Interruptions are safe — every
   script resumes row-by-row.
3. Nothing else: index (6,771 mpnet chunks), venv, scripts, and pilot validation are all in
   place from Stage 4.

## Existing pieces to reuse (no new architecture code)

- `scripts/run_generation.py` — `--system {standard,crag,cragpp} --generator ollama`; resumable;
  chunk-list `retrieved_context`; pilot rows already in `gen_r3_*_ollama.csv` (15/10/10).
- `scripts/run_eval.py` — 4 metrics, resumable; `make_judge()` already routes `ollama:` prefixes.
- deepeval 4.0.6 `LocalModel` (verified: `model, api_key, base_url, temperature, format` params).

## Stages & pass criteria

### Stage A — Judge wiring + judge pilot (gate before any spend/long runs)
- Extend `make_judge()` in `scripts/run_eval.py`: `groq:<model>` →
  `LocalModel(model=<model>, base_url="https://api.groq.com/openai/v1",
  api_key=os.getenv("GROQ_API_KEY"))`. Default `--judge` unchanged; this session passes
  `--judge groq:openai/gpt-oss-120b`. LocalModel reports no $ cost — log call counts/tokens
  and estimate cost manually.
- **Delete `results/eval_r3_standard_ollama_deepeval.csv`** — it holds 3 rows scored by the
  gemma3:4b plumbing judge; the real run must not resume into mixed-judge scores.
- Judge-pilot: score the 15 existing standard rows with the Groq judge.
- **Pass criteria:** (a) 4/4 metrics numeric on ≥14/15 rows; (b) scores non-degenerate
  (faithfulness not collapsing to 0 like the gemma3:4b plumbing judge; some variance across
  rows); (c) measured time/row and token volume extrapolate to ≤3 h and ≲$10 for the full set
  — else stop and re-decide.

### Stage B — Generation (local, overnight-friendly)
- Run sequentially in background: `run_generation.py --system standard --generator ollama`,
  then `crag`, then `cragpp` (no `--rows` → full 300; resumes over pilot rows).
- **Pass criteria:** per file: 300 rows; empty answers ≤1%; `retrieved_context` parses as a
  list on every row; latency stats recorded. If interrupted: rerun the same command (resume)
  before proceeding.

### Stage C — Judging (Groq)
- `run_eval.py --gen results/gen_r3_<system>_ollama.csv --judge groq:openai/gpt-oss-120b`
  for the three files (sequential; resumable).
- **Pass criteria:** ≥98% of row×metric cells scored per file; failures listed explicitly.

### Stage D — Summarize, document, commit
- New `scripts/summarize_r3.py`: per-system metric means (+ n, null counts) from the three
  eval CSVs → prints the comparison table and writes `results/eval_r3_summary.xlsx`.
- Update `context.md` §2 with an **r3 section** (table + headline findings: CRAG++ vs CRAG vs
  Standard under identical judge/embeddings/chunk-list logging; comparison caveats vs r1/r2:
  different judge and embeddings). Fill the PENDING-r3 slots in `reports/plans_response.md`
  (items 1, 3, 4.1). Mark the deferred stage done in `context.md` §10.
- Commit on `RAG-2`: docs + `scripts/summarize_r3.py` + **force-add the small score files**
  (`git add -f results/eval_r3_*_deepeval.csv results/eval_r3_summary.xlsx`, ~100 KB each) so
  the scores are versioned; the 3–4 MB generation CSVs stay gitignored. Push.
- **Pass criteria:** tables in both docs match the CSVs; no contradiction with context.md;
  pushed.

## Risks / notes

- **Groq free tier**: if no payment method is added, Stage A's extrapolation will show it —
  stop and report rather than crawl.
- **gpt-oss-120b JSON compliance**: deepeval needs schema-conformant judge output; LocalModel
  supports a `format` kwarg — set `format="json"` if Stage A shows parse failures.
- **Ollama stability over ~15 h**: resumability makes crashes cheap; the harness skips
  completed rows.
- Generation and judging are independent: Stage C for `standard` can start while CRAG++ still
  generates, but keep it simple and sequential unless time presses.

## Verification (end-to-end)

The session is done when: three 300-row generation files and three ≥98%-scored eval files
exist; `summarize_r3.py` reproduces the documented tables from the CSVs; `context.md` §2-r3
answers the headline question (is CRAG++ better than CRAG and Standard, and on which metrics)
with measured numbers; `reports/plans_response.md` has no remaining PENDING-r3 markers; all
committed and pushed on `RAG-2`.
