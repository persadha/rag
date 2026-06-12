# Response to Reviewer Comment 1 — The Performance Gap Between Architectures

The reviewer is asking the right question — *which* stage of the advanced pipeline causes the
degradation. The codebase + data exploration answers it more precisely than the report does, and it
also surfaces that the report's description of the advanced pipeline **doesn't match the evaluated code**.

## First: the evaluated "advanced" pipeline isn't what the report describes

The scored "advanced" system is **CRAG** (`graph_builder_adv.py`). Its actual flow is:

```
retrieve(original query) → grade_documents (gemma3:1b filter) → plan_sub_steps (decompose, REUSE the same docs)
→ generate_answers (answer each sub-q, then synthesize ≤100 words) → grade_generation (loop if "not useful")
```

Against the report's hypothesized failure modes, two of them **don't exist in the code**:

- **"Multi-step retrieval introduced noise"** — there *is no* per-sub-question retrieval. `plan_sub_steps`
  reuses the original query's documents for every sub-question (`page_content_1 == page_content_2` in
  **198/199** rows). The node that would re-retrieve (`retrieve_sub_question_documents`) was dead/unwired.
- **"The reranking stage discarded useful context"** — there is **no reranker** in the evaluated graph.
  `flashrank`/`rank_bm25` are in `requirements.txt` but aren't wired into any graph builder. The closest
  analogue is the gemma3:1b **document grader**.

So the report attributes the gap to mechanisms the evaluated system doesn't have. That alone is worth
correcting before iterating.

## Stage-by-stage attribution (what's actually causing it)

| Reviewer's candidate | What I found | Verdict |
|---|---|---|
| Poor sub-question decomposition | Decomposition runs, but it's **inert** — docs are reused regardless, so sub-questions add LLM calls/latency with **zero retrieval effect**. Not "bad sub-questions," but pointless ones. | Cost, not quality |
| Low-quality retrieval per sub-question | **N/A** — no per-sub-question retrieval exists in CRAG. | Not a cause (absent) |
| Reranking discards relevant material | No reranker. But the **gemma3:1b doc-grader over-filters**: on zero-precision rows the context holds only **19%** of the gold-answer words (vs 56% on non-zero rows) → the answer-bearing doc was graded out. This is the real "useful context discarded" mechanism. | **Real cause** |
| Synthesis fails to integrate sub-answers | **Real** — sub-answers are often refusals (*"Unfortunately, I don't see any information…"*) because each sub-q is answered against the reused broad-query context; the synthesizer blends refusals, and the **"≤100 words"** cap drops the exact fact short gold answers need. | **Real cause** |

## The biggest correction: much of the precision/recall gap is a measurement artifact

The reviewer treats context precision/recall as a real retrieval-quality signal — but a large part of
CRAG's deficit is **how its context was logged for scoring**, not retrieval:

- Standard logs `retrieved_context` as **4 separate nodes**; CRAG logs **one ~8.7k-char blob**. Contextual
  precision/recall are *ranking* metrics over a list — a single blob degenerates them. CRAG hard-zeros on
  **135/197 (69%)** rows vs standard's 36%, and there are proven false zeros (rows whose blob contains
  73–100% of the gold answer still scored 0).
- Logical clincher: CRAG runs the **same retriever on the same query** as standard, so its context is a
  *superset* of standard's — scoring the same underlying text ~2× worse is serialization, not worse retrieval.

So the precision/recall half of the gap is largely artifactual; the **answer-correctness** half is genuine
(the grader + synthesis issues above).

## Answering the meta-reasoning vs. chunking/retrieval question directly

The reviewer's dichotomy — small-model meta-reasoning vs. chunking/retrieval — misses that the dominant
causes are **implementation bugs in the orchestration and the eval harness**, not the model:

- **Faithfulness ≈ 0.97 (Pass 2)** across all systems → the models *do* ground answers in whatever context
  they get. There's no "reasoning collapse" — when given the right context, generation is fine. This argues
  **against** the report's "limited reasoning capacity of sub-10B models" being the primary driver, and
  **against** prompt-engineering/fine-tuning as the highest-leverage fix.
- **Retrieval/chunking is a real ceiling for *both* architectures** (precision caps ~0.64, recall ~0.56) —
  worth investing in (embedding upgrade, chunk size, the `k`-was-ignored bug, an actual reranker) — but it
  won't explain the CRAG-vs-standard *gap*, because both sit on the same retriever.
- Therefore the gap is best explained by, in order: (1) the **blob-vs-list logging artifact**, (2) the
  **over-aggressive 1B doc-grader**, (3) **decompose-but-reuse** making decomposition inert, (4) **synthesis
  dilution + the 100-word cap**.

**Recommended response to the reviewer:** the degradation is not a single failure mode — it's a measurement
artifact plus three implementation bugs, none of which is "the small model can't reason." A fair re-test
requires (a) logging CRAG's context as a chunk list, (b) loosening/replacing the 1B grader, (c) real
per-sub-question retrieval, (d) dropping the word cap — most of which have now been implemented (P1/P2).
Re-running after those isolates whether any residual gap is genuinely about model reasoning.
