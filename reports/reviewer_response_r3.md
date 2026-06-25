# Response to Reviewers (R1 + R2) — r3 revision

*Information Retrieval Using Retrieval-Augmented Generation on PIRLS Documents*
Snapshot 2026-06-23 · branch `RAG-3` · 195-question revised dataset · judge `gpt-oss-120b` (held constant).

This document maps every reviewer comment to one of three dispositions and points to the evidence or
the report edit that addresses it:

- ✅ **Addressed with new evidence** — a new r3 experiment answers it directly.
- ✏️ **Editorial** — clarified/reframed in the report; no new data needed.
- 🔭 **Future work / acknowledged limitation** — out of scope here; strengthened in the limitations section.

We thank both reviewers; the two strongest asks (R1 "missing baseline" and R2 "value added by
generation") **converge on one cheap ablation**, which we ran — and which produced a genuinely
informative, somewhat counter-intuitive result (below).

---

## Status at a glance

| # | Reviewer comment | Disposition |
|---|---|---|
| R1-1 | §2.3 Significance reads as unsupported claims | ✏️ Editorial |
| R1-2 | Scope/cognitive level of the eval questions (fact vs why/how) | ✅ **New evidence** + ✏️ Editorial |
| R1-3 | **Missing baseline: with vs. without retrieval** | ✅ **New evidence** |
| R1-4 | Praise: DeepEval-over-RAGAS rationale | — (noted) |
| R1-5 | Human-in-the-loop / calibrated expert ratings | 🔭 Future work |
| R1-6 | Test open/foundation models; NAEP GPT-4o >80% no-retrieval | ✅ **New evidence** + ✏️ framing |
| R2-1 | Privacy premise — validate on internal/non-public docs | 🔭 Future work |
| R2-2 | **Value added by generation — retrieval-only baseline** | ✅ **New evidence** |
| R2-3 | LLM-as-judge circularity; is 0.62 "good enough"? | 🔭 Future work (+ partial) |
| R2-4 | **Which** stage of the advanced pipeline fails | ✅ Addressed (see also `reviewer_response_1.md`) |
| R2-5 | Deployment model (standalone/server/networked) + latency | ✏️ Editorial |
| R2-6 | **KB scale (chunks/tokens/footprint) + chunking rationale** | ✅ **New evidence** |
| R2-7 | Minor: human vs. synthetic metric split | ✅ Already in results |
| R2-8 | Minor: characterize the 300→195 exclusions | ✏️ Editorial |
| R2-9 | Minor: embedding model not described | ✅ Described + ⚠️ correction |

---

## The headline new result: the value-added ablation (R1-3 + R2-2)

Both reviewers ask, from two angles, *"is the RAG actually worth it?"* — R1 vs. an off-the-shelf LLM
(no retrieval), R2 vs. raw retrieved chunks (no generation). We ran both as one 3-condition ablation on
the 195-row set, scored on answer-correctness with the same `gpt-oss-120b` judge.

| Condition | What it is | Answer Corr | human | synthetic |
|---|---|---|---|---|
| (a) Closed-book | llama3:8b, **no retrieval** | **0.202** | 0.150 | 0.290 |
| (a2) Closed-book | gpt-5.4-mini, **no retrieval** | **0.490** | — | — |
| (b) Retrieval-only | top-4 hybrid+bge chunks **as the answer**, no LLM | **0.827** | 0.852 | 0.785 |
| (c) Full RAG | llama3:8b + hybrid + bge | 0.689 | 0.753 | 0.581 |
| (d) Full RAG (best) | gpt-5.4-mini + hybrid + bge | 0.774 | 0.787 | 0.751 |

### R1-3 — Retrieval is unambiguously worth it ✅
Holding the generator fixed (llama3:8b), **retrieval lifts answer-correctness from 0.202 → 0.689
(+0.488)**; the best configuration reaches 0.774 (**+0.572** over closed-book). On *this* corpus, the
deployable open 8B model is close to unusable without retrieval. This is the direct empirical
justification the reviewer asked for: the investment in RAG is warranted.

### R1-6 — On the NAEP "GPT-4o >80% without retrieval" point ✅✏️
We now have a **direct empirical answer** to this comparison: gpt-5.4-mini **without retrieval scores 0.490** on the same 195-question PIRLS set — well below the llama3:8b **with** retrieval (0.689). This completes the 2×2:

|  | Closed-book | RAG (hybrid + bge) |
|--|-------------|-------------------|
| llama3:8b (open) | 0.202 | **0.689** |
| gpt-5.4-mini (closed) | 0.490 | **0.774** |

Two takeaways: (i) **RAG-llama (0.689) > closed-book-mini (0.490)** — the small open model with retrieval beats the larger closed model without it on this corpus; (ii) the closed-book 0.490 remains well below the NAEP >80% figure, consistent with our original reading: GPT-4o is a far larger model than gpt-5.4-mini, and the corpora and question styles differ. The honest claim remains narrow and defensible: **for the locally-deployable open model this project targets, retrieval is essential**. The 2×2 now provides the "test open models" evidence the reviewer requested.

### R2-2 — Value added by *generation*: read the caveat carefully ✅⚠️
Taken at face value, the table says raw chunks (0.827) **beat** full RAG (0.689 / 0.774) on
answer-correctness — i.e. generation has *negative* value. **This is a metric artifact, not a finding
that chunks are a better answer**, and we will present it as such:

- The retrieval-only "answer" is a ~4,000-character concatenation of 4 chunks. Our answer-correctness
  metric (GEval) rewards **coverage of the reference facts** and is instructed **not to penalize extra
  correct detail**. A verbose blob that *contains* the gold facts therefore scores very high, while the
  LLM's concise synthesis is penalized whenever it compresses away or rephrases a fact.
- It is **not** evidence that a 4,000-character context dump is a usable answer. It games a
  coverage-oriented metric; it says nothing about concision, directness, or the 30-second/usability
  expectations from the user survey.

The substantive, and stronger, conclusion this supports: **the binding constraint is retrieval, not
generation.** Once the right chunks are retrieved, the gold facts are already present (0.827) — the
generator's job is to turn that into a concise, attributable answer, which this metric does not credit.
This dovetails with R2-4 and with the architecture analysis (retrieval-bound, not reasoning-bound).

We also adopt the reviewer's **expert vs. non-expert framing**: for experts who know the PIRLS
documentation, direct chunk retrieval may already suffice (hence the high 0.827); the generation layer's
value is concentrated on non-expert users who need synthesis and plain-language phrasing. We will state
this explicitly as a practical recommendation, and note that a concision/usability metric (not just
fact-coverage) is needed to measure the generator's true contribution — flagged as future work.

---

## R2-6 — Knowledge base scale & chunking strategy ✅

Measured directly from the live Chroma index (`scripts/kb_stats.py`):

| Property | Value |
|---|---|
| Indexed chunks | **6,771** |
| Source documents | **108** |
| Total words / characters | 829,015 / 5,501,267 |
| Total tokens | **1,498,180** (tiktoken `cl100k_base`) |
| Mean chunk | 122 words · 812 chars · 221 tokens |
| Chunk-length (words) p50 / p90 / max | 136 / 161 / 395 |
| On-disk footprint | **70.9 MB** |
| Embedding model | `sentence-transformers/all-mpnet-base-v2` (768-dim) |

**Chunking rationale & sensitivity.** r3 evaluated chunk size directly (experiment **E2**, 512/64 vs.
the report's 1000/100): AC 0.473 → 0.488 (+3%), gold-context similarity +0.034, with context
precision/recall slightly *down*. Takeaway: smaller chunks are marginally cleaner but do not move the
needle, because the dominant retrieval lever is **candidate width + reranking + hybrid lexical matching**
(see below), not chunk granularity. We will fold these numbers into the body so the chunking parameters
are justified rather than merely stated, and retain the grid-search recommendation as a refinement.

## R2-9 — Embedding model description ✅ ⚠️ (correction needed)
The report lists the embedding model as **"embeddinggemma"**, but the evaluated r3 index uses
**`sentence-transformers/all-mpnet-base-v2` (768-dim)** (verified from the index). The report text must
be corrected to the actual model. We will add a short description: all-mpnet-base-v2 is a general-purpose
768-dim sentence encoder; an embedding-model swap is listed as a Tier-1 retrieval lever (alongside
chunking/reranking) since context precision/recall are the constraining metrics. Note that the **hybrid
BM25+dense** addition (E9c) already mitigates a key weakness of any single dense encoder — exact-term
matches (country names, acronyms) that embeddings miss — lifting recall to 0.909.

---

## R2-4 — Which stage of the advanced pipeline fails ✅

Fully analyzed in **`reports/reviewer_response_1.md`** and `architecture_evolution_analysis.md`. Summary:
the degradation is **retrieval/orchestration, not small-model reasoning** — faithfulness is ≈0.94–0.98
across *all* architectures, so models do ground answers when given good context. Stage attribution:

- **CRAG**: the gemma3:1b document-grader over-filters — recall collapses to 0.49 (vs 0.82 for Standard);
  the answer-bearing chunk is graded out.
- **CRAG++**: per-sub-question retrieval injects off-topic chunks — precision falls to 0.45.
- A large part of the *precision/recall* gap is additionally a **logging artifact** (CRAG logged context as
  one blob vs. Standard's chunk list — ranking metrics degenerate on a single blob).

So the reviewer's dichotomy resolves toward **invest in retrieval/orchestration**, which is exactly what
r3 did (reranking, hybrid retrieval) with the largest gains.

---

## Editorial items

**R1-1 — §2.3 Significance ✏️.** Reframe the significance statements as either literature-informed
motivations (with citations) or as explicit research questions/hypotheses the study investigates.

**R1-2 — Question scope & cognitive level ✅✏️.** We classified all 195 questions by cognitive level,
subtype, and domain with the held-constant `gpt-oss-120b` (`scripts/classify_questions.py`). The reviewer's
intuition is **confirmed and now quantified**: the set is dominated by fact-retrieval, and the deep
methodological "why/how" questions they name are thin.

| Cognitive level | Overall | Human (n=123) | Synthetic (n=72) |
|---|---|---|---|
| fact_retrieval | **68.2%** (133) | 77.2% | 52.8% |
| reasoning (why/how/analysis) | **31.8%** (62) | 22.8% | 47.2% |

| Subtype | share | | Domain (top) | share |
|---|---|---|---|---|
| factual_lookup | 60.0% | | assessment_framework | 25.1% |
| explanatory_why | 17.4% | | other | 24.1% |
| procedural_how | 9.2% | | context_questionnaires | 16.9% |
| definitional | 7.7% | | reading_achievement_results | 8.7% |
| comparative_analytical | 5.6% | | *methodology* (sampling+weighting+plausible-values) | **~9.2%** |

Three honest takeaways for the report: (i) **~68% of the set is fact-retrieval**; reasoning-oriented
questions are a real but minority **~32%**. (ii) The specific methodological topics the reviewer cites —
sampling (4.6%), plausible values & scaling (3.1%), weighting & variance (1.5%) — are **under-represented
(~9% combined)**, so the evaluation under-tests exactly the "why implement X / how to use Y" reasoning the
reviewer highlights. (iii) Human-authored questions skew even more to facts (77%) than synthetic ones
(53%). We will add this distribution to the methods section and **explicitly acknowledge in the
limitations** that reasoning-heavy methodological questions are under-sampled — and recommend deliberately
over-sampling them in any follow-up eval set.

**R2-5 — Deployment model ✏️.** Add a short subsection on the intended deployment (standalone workstation
vs. local network server vs. networked service) and its performance implications. Tie to measured latency:
the closed gpt-5.4-mini path is fast (~8 s/query, API), the local reasoning model (deepseek-r1) exceeds
~82 s/query — beyond the 30-second survey threshold — and the ~6 s basic-pipeline figure was on a high-end
RTX-4090. Note that a local *server* deployment could host a larger model to mitigate sub-10B accuracy
limits, at a hardware cost.

**R2-8 — 300→195 exclusions ✏️.** Clarify that r3 uses a **clean, revised 195-row dataset**
(`datasets/revision/evaluation_dataset.xlsx`) with explicit human/synthetic tagging, replacing the earlier
ad-hoc 300→195 filtering; briefly characterize what changed so the set's representativeness is transparent.

**R2-7 — Human vs. synthetic split ✅.** Already produced for every run (123 human / 72 synthetic; see the
per-condition splits in the table above and `results/eval_r3_summary.xlsx`). We will surface these splits
in the report body. (Note: r3's revised set is 123/72; the report's "105 of 300 synthetic" refers to the
superseded dataset.)

---

## Future work / acknowledged limitations

**R1-5 & R2-3 — Human/expert evaluation & judge calibration 🔭.** We keep this as future work (no expert
grading infrastructure was built in scope) and strengthen the limitations section. Two mitigations to
note: (i) r3's best answer-correctness (0.774; precision 0.865) is now **well above the ≤0.5 band** that
prompted R2's "would an expert accept this?" concern, so the acceptability question is less acute than at
report time; (ii) we ran a small **dual-judge cross-check** (`scripts/rejudge_disputed.py`,
`gpt-oss-120b` vs. a second judge) on disputed rows as a lightweight calibration, indicating a portion of
low scores are judge under-credit rather than wrong answers. A calibrated expert study on a representative
subset remains the right next step and is recommended.

**R2-1 — Validate on internal/non-public documents 🔭.** Valid and important: the privacy premise is best
substantiated on the very material it protects. We acknowledge this as the primary external-validity
limitation (the public PIRLS corpus does not exercise the privacy motivation) and frame
internal-document validation as the key future-work item, subject to data access.

---

## Report edits this implies (checklist)
- [ ] §2.3 reframed as motivations/hypotheses with citations (R1-1).
- [ ] Question-scope distribution table (68% fact / 32% reasoning; methodology ~9%) + reasoning-question limitation expanded (R1-2).
- [ ] New "Value-added ablation" subsection with the 3-condition table + the R2-2 caveat (R1-3, R1-6, R2-2).
- [ ] KB-scale table + chunking rationale/sensitivity (E2) in the body (R2-6).
- [ ] **Correct the embedding-model name** to all-mpnet-base-v2 (768-dim) + description (R2-9).
- [ ] Stage-attribution summary for the architecture gap, citing the retrieval-bound finding (R2-4).
- [ ] Deployment-model subsection with latency caveats (R2-5).
- [ ] Dataset clarification: clean 195-row revised set; human/synthetic splits surfaced (R2-7, R2-8).
- [ ] Limitations: expert-calibration + internal-corpus validation as future work (R1-5, R2-1, R2-3).
