# IEA R&D Funds — Call No. 6 (2026) · Proposal Draft (working synthesis)

> **Status:** working synthesis of the ideation in this project's planning conversation,
> mapped to the call's required sections (§6). **Not submission-ready.** Per call §6,
> the intellectual content (theoretical background, research design, potential impact,
> timeline, budget, references) must be authored in the applicant's own words; rewrite
> this draft and disclose AI assistance (brainstorming, source identification, and
> building the accompanying open-source code artifact). Placeholders are in `[BRACKETS]`.
> Funding tier: **Tier 1 (≤ €55,000)**. Start ≥ 1 Feb 2027, complete before end-2027.

---

## Title
**ILSA-TableQA: A Benchmark and Privacy-Constrained Evaluation of Table-Aware
Retrieval-Augmented Generation for International Large-Scale Assessment Reports**

## Short description (~50 words)
IEA reports are dense with statistical exhibits, yet current retrieval-augmented
generation (RAG) tools cannot read them — text extraction flattens tables and answers
fail silently. This project builds the first open question-answering benchmark over ILSA
statistical exhibits, a numeric-exact evaluation protocol, and a comparison of table
extraction paradigms deployable on-premises.

## Key areas and keywords
Reporting and dissemination; machine learning / AI applied to IEA methods; secondary
analysis tooling; data visualization and communication of results. *Keywords:*
retrieval-augmented generation, large language models, table extraction, document
understanding, benchmark, evaluation methodology, PIRLS, TIMSS, ICILS, privacy-preserving
NLP.

---

## Summary (≤ 500 words)
International large-scale assessments (ILSAs) communicate their central findings through
**statistical exhibits** — achievement distributions with standard errors, benchmark
percentages, group-gap and trend tables, and cross-country policy tables. These exhibits
are where the numbers that matter to researchers, policymakers, and the public actually
live. Recent advances in retrieval-augmented generation (RAG) make it possible to query
document collections in natural language, and a prior proof-of-concept demonstrated a
privacy-preserving RAG system over PIRLS 2021 documentation. That work reached a clear,
documented limitation: the system **cannot reliably read tables**. Standard PDF text
extraction flattens row-and-column structure, and a vision-language attempt misassociated
headers, so any question whose answer sits in an exhibit fails — and fails *silently*,
returning a fluent but wrong answer. This is precisely the content ILSA users most need.

This project addresses that gap with a rigorous, reusable, methodology-first contribution
rather than a one-off tool. We will (1) construct **ILSA-TableQA**, the first open
question-answering benchmark built on statistical exhibits from public ILSA international
reports across **at least three studies** (PIRLS, TIMSS, ICILS), with gold answers tied
to specific cells and typed by reasoning demand (lookup, standard-error reads,
group gaps, trends, aggregation, significance); (2) define a **numeric-exact and
structure-aware evaluation protocol** — numeric-match-with-tolerance plus table-structure
recovery — that cannot be gamed by verbose answers, correcting a measurement artifact
observed in the prior study; and (3) run a controlled, **privacy-constrained comparison**
of the two dominant extraction paradigms — layout-aware text parsing (CPU-deployable) and
OCR-free vision retrieval (GPU-dependent) — on identical inputs, quantifying accuracy
against compute, latency, and on-premises deployability.

The work is deliberately scoped for Tier 1: it uses only **public** ILSA materials and
requires **no confidential data**, and because exhibit answers are objective (a value in
a cell) ground truth can be established reliably by the project team without scarce expert
raters. It also builds directly on assets already in hand — an evaluation harness, a
195-item QA set, a hybrid retrieval stack, and (as a de-risking artifact) a working
numeric-exact scorer and a validated table-extraction adapter — reducing delivery risk.

Outputs are open by default: the benchmark dataset, the evaluation code, a reference
table-aware RAG integration with CPU and GPU deployment recipes, and a research report
with practical guidance. The benchmark and protocol are **reusable across IEA studies and
cycles**, giving IEA a durable instrument for assessing whether any future AI tool is
accurate enough to trust with its results — a direct contribution to trustworthy
dissemination and to reducing error in how ILSA findings are retrieved and communicated.

---

## Theoretical background (≤ 1,000 words)
**ILSA reporting and the secondary-analysis burden.** IEA's studies (PIRLS, TIMSS, ICILS,
ICCS) publish international reports whose evidentiary core is tabular: mean scale scores
with standard errors, percentages reaching international benchmarks, achievement by
context variables, sex/subgroup gaps, and cross-cycle trends flagged with significance
markers. Reading and reusing these exhibits is central to the "analyzing data and
reporting results" strand of the Technical Standards for IEA Studies (Martin, Fraillon, &
Sibberns, 2025) and to the study lifecycle's reporting and dissemination phase. Anything
that lowers the cost and error of extracting facts from these exhibits serves IEA's
mission and the wider secondary-analysis community.

**Retrieval-augmented generation and its current limits for tables.** RAG couples a
retriever over a document collection with a generator that answers from retrieved
context. A prior IEA-focused proof-of-concept over PIRLS documentation established two
findings that shape this proposal. First, performance is **retrieval-bound**: once the
correct context is retrieved, the answer is usually present; the binding constraint is
retrieval quality, not generator size. This is independently corroborated by
**T2-RAGBench** (Strich et al., 2025, arXiv:2506.12071), where accuracy with oracle
context reaches **72.3–72.5%** but the best RAG configuration reaches only **41.3–41.7%**
(Number Match) — and where that best configuration is **hybrid BM25+dense retrieval**,
matching the prior project's own architecture choice. Second, elaborate agentic pipelines
(corrective/decomposition RAG) did not beat a simple retrieve-then-generate pipeline on
this material; for the low-compositionality lookups typical of tables, added iteration
mainly adds latency and error. Together these justify concentrating effort on
**retrieval/extraction quality**, not pipeline complexity.

**Document understanding: two paradigms.** Modern table extraction splits into two camps.
(i) **Layout-aware text parsing** detects layout and recovers table structure from the
text/vector layer: *Docling* with its TableFormer structure model (Livathinos et al.,
2025, arXiv:2501.17887), *PaddleOCR/PP-StructureV3* (Cui et al., 2025, arXiv:2507.05595),
and the academic-document parser *Nougat* (Blecher et al., 2023, arXiv:2308.13418). These
run **on CPU** (Docling reports ~0.79 s/page median on CPU; ~0.11 s/page on a modest GPU),
which matters for on-premises deployment. (ii) **OCR-free vision retrieval** embeds page
images directly and matches with late interaction: *ColPali* (Faysse et al., 2024,
arXiv:2407.01449) reports **81.3 vs 67.0 nDCG@5** over the strongest text-extraction
baseline on the ViDoRe benchmark (a **+14.3** gain) and **83.9 nDCG@5** on the
table-specific TabFQuAD set; *VisRAG* (Yu et al., 2024, arXiv:2410.10594) reports **~40%
average** relative improvement over text-based RAG; *M3DocRAG* (Cho et al., 2024,
arXiv:2411.04952) extends this to multi-page scale. Vision approaches are typically
**GPU-dependent (~8–16 GB)**, creating a genuine accuracy-versus-deployability tension for
a privacy-constrained, on-premises setting — the central empirical question here.
Benchmarks such as **OmniDocBench** (Ouyang et al., 2024, arXiv:2412.07626) evaluate
parsers on diverse documents but not on ILSA statistical exhibits.

**Table question answering and the education gap.** Table-QA is an active field, but its
benchmarks target other domains: finance (**FinQA**, Chen et al., 2021,
arXiv:2109.00122; **TAT-QA**, Zhu et al., 2021; **MultiHiertt**, Zhao et al., 2022),
airlines (AIT-QA), science (SciTab), and Wikipedia/industrial tables (WikiTableQuestions,
TableBench). The closest ancestor, **HiTab** (Cheng et al., 2022, arXiv:2108.06712;
10,686 QA pairs over 3,597 hierarchical tables), draws on statistical-agency and Wikipedia
tables — but **no existing benchmark targets international large-scale education
assessment exhibits**, with their characteristic multi-level country/grade/sex headers,
standard errors in parentheses, and significance-marked trend columns. Methods for
table reasoning (Chain-of-Table, Wang et al., 2024, arXiv:2401.04398; TableRAG, Chen et
al., 2024, arXiv:2410.04739) provide open baselines.

**Evaluation and the coverage artifact.** The prior study exposed a measurement trap: a
raw multi-chunk context dump scored 0.827 on a coverage-oriented answer-correctness metric
because that metric rewarded fact coverage and did not penalize verbosity. For statistical
answers this is unacceptable — a reported percentage or standard error must match to the
reported precision. The evaluation literature therefore uses **numeric-exact match with
tolerance** for numeric answers and **structure metrics** such as GriTS (Smock et al.,
2022, arXiv:2203.12555) / TEDS for table-structure recovery. This project adopts and
adapts these to ILSA conventions. Retrieval enhancements with strong evidence and low
implementation cost — Anthropic's **Contextual Retrieval** (2024; up to 49% retrieval-
failure reduction, 67% with reranking) and multi-representation embeddings such as
**BGE-M3** (Chen et al., 2024, arXiv:2402.03216, whose sparse component captures exact
numeric tokens) — are candidate levers, while heavier methods (RAPTOR, Sarthi et al.,
2024; late chunking, Günther et al., 2024) are cited but out of scope to avoid over-
engineering.

---

## Research design (≤ 1,000 words)
**Research questions.**
- **RQ1 (motivation).** How much worse is current text-extraction RAG on table-sourced
  vs prose-sourced questions over ILSA reports? (Quantify the silent-failure gap.)
- **RQ2 (paradigm).** Layout-aware text parsing vs OCR-free vision retrieval — which
  recovers *exact* cell-level answers from ILSA exhibits, and at what compute/latency
  cost?
- **RQ3 (measurement).** What metric stack scores ILSA table-QA correctly (numeric-exact
  with tolerance + structure recovery) without the coverage artifact?
- **RQ4 (deployment).** What reference architecture and guidelines make table-aware RAG
  deployable under IEA's on-premises/privacy constraint (CPU-only vs modest-GPU tiers)?

**Corpus.** Public ILSA international reports and encyclopedia exhibits from **≥3 studies**
(e.g., PIRLS 2021, TIMSS 2023, ICILS 2023), selected to span the exhibit taxonomy
(achievement distributions, benchmark percentages, trend, group-gap, context-relationship,
percentages-and-averages, policy-categorical). Multi-study, multi-country coverage is a
stated call preference.

**Benchmark construction (ILSA-TableQA).** We sample exhibits across types and author
**~300–500** question–answer items. Each item records the study, cycle, source document
and page, exhibit id and type, the question, a reasoning type, the answer kind, the gold
answer, the target numeric value(s) and unit, a numeric tolerance, the gold cell
coordinates, and the serialized gold table region. Questions are deliberately balanced
**away from single-fact lookups** (the prior eval was ~68% lookups) toward harder
reasoning: standard-error reads, group gaps, trend direction, aggregation, and
significance. Because answers are objective cell values, **two annotators** independently
re-derive gold for a random ~15% subset and we report exact-agreement inter-rater
reliability; disagreements typically indicate an ambiguous question, which is then fixed.
Sampling and any coverage caps are logged (no silent truncation). The item schema,
annotation guidelines, and a validating linter already exist as a project artifact.

**Conditions (bounded matrix).** On identical inputs we compare: **(baseline)** the prior
text-flattening pipeline; **(text paradigm)** Docling and PaddleOCR structure extraction →
serialized table chunks → hybrid retrieval → LLM; **(vision paradigm)** ColPali and
ColQwen2 page-image retrieval → vision-language generation; and **(enhancement)**
contextual-retrieval headers plus table serialization on the best text pipeline. Open
table-QA methods (Chain-of-Table) serve as generation baselines. Generation uses a
locally deployable open model, consistent with the privacy motivation; a hosted model is
included only as an upper-reference where licensing permits.

**Metrics (RQ3).** Answer correctness is **numeric-exact-match-with-tolerance** (per-unit
defaults: percent ±0.05, scale score ±0.5, standard error ±0.05; counts exact),
scored as F1 over the numbers extracted from an answer so that a verbose dump cannot
score full marks. Non-numeric answers use direction/boolean/token-set scorers.
Table-structure recovery uses GriTS-style content/topology (a working grid implementation
exists; the reference GriTS/TEDS is the production upgrade). We report retrieval quality
(nDCG@k, recall) separately to locate failures, and always break results down by reasoning
type, exhibit type, and study.

**Deployment analysis (RQ4).** For each paradigm we record accuracy against **latency and
peak VRAM**, distinguishing a CPU-only tier from a modest-GPU tier, and translate the
trade-off into concrete on-premises deployment guidance — the IEA-specific research output
that a generic table-QA study does not provide.

**Reproducibility & rigor.** All runs go through a resumable, judge-free scoring harness
(numeric-exact is deterministic and free). Small score differences are reported with
run-to-run variation rather than over-interpreted. Code, data, and configurations are
released open-access.

**De-risking already completed.** A numeric-exact scorer (26 passing assertions covering
standard-errors-in-parentheses, significance markers, and the anti-coverage-artifact
behavior), the item schema/linter, a scoring harness, and a Docling text-extraction
adapter are already built and were validated on a real PIRLS exhibit (content/topology
1.000 vs an independent gold on the shared region), demonstrating the pipeline end-to-end.

---

## Potential impact (≤ 1,000 words)
**For IEA — a documented blocker removed.** Today, questions answerable only from exhibits
fail silently in RAG tools built over IEA reports. Solving table reading unlocks the
majority of quantitatively meaningful questions a secondary analyst, journalist, or
policymaker would ask ("what percentage reached the High Benchmark", "did achievement rise
since the last cycle", "how large is the sex gap"). This directly serves the reporting and
dissemination phase of the study lifecycle and the "data visualization and communication
of results" interest highlighted in the call.

**A reusable instrument, not a one-off.** ILSA-TableQA and its evaluation protocol are
**study- and cycle-agnostic**. They give IEA a durable way to answer a recurring question
as AI adoption grows: *is a given tool accurate enough to trust with our numbers?* Because
the metric demands exactness rather than plausibility, it provides an honest acceptance
test for any future retrieval or extraction system — a contribution to trustworthiness and
to reducing error in dissemination (a component of total survey error at the reporting
stage).

**Privacy-relevant deployment evidence.** The prior project's motivation was
privacy-preserving, on-premises operation (no transmission of IEA materials to external
services). This project produces the missing evidence on whether the strongest table
methods can meet that constraint: text-parsing paradigms run on CPU, while vision
retrieval needs a GPU. Quantifying the accuracy cost of staying on-premises lets IEA and
its study centers make an informed build decision — practical guidance no external
benchmark offers.

**For the field — a first-of-kind benchmark.** Table-QA has matured in finance and general
domains but has **no** benchmark for international education-assessment statistics.
ILSA-TableQA fills that gap and, being open, becomes shared infrastructure for the
large-scale-assessment and NLP communities, extending IEA's methodological leadership into
an area (applied ML/AI for assessment methods) explicitly encouraged by the call. Results
are suitable for IEA's *Large-scale Assessments in Education* journal and for presentation
at the General Assembly or the International Research Conference.

**Efficiency and access.** By identifying which extraction approach recovers exhibit
semantics at acceptable cost, the project reduces the manual effort of locating and
transcribing values from lengthy PDFs, and lowers the barrier for non-expert users to
obtain correct, source-attributed answers from ILSA reports.

**Low risk, high leverage.** The work depends on no confidential data and reuses existing,
already-validated tooling, so its risk profile is low relative to its potential to change
how ILSA results are queried and communicated.

---

## Envisaged products (≤ 200 words)
1. **ILSA-TableQA** — an open, documented benchmark dataset (~300–500 items across ≥3
   studies), with gold answers tied to cells and typed by reasoning demand.
2. **Open-source evaluation harness** — numeric-exact-with-tolerance and table-structure
   metrics, resumable and judge-free (seeded implementation already exists).
3. **Reference table-aware RAG integration** — extraction + retrieval + generation with
   CPU and modest-GPU deployment recipes.
4. **Research report** (open-access grey literature) with the paradigm comparison,
   deployment guidance, and metric protocol; optionally a peer-reviewed article in
   *Large-scale Assessments in Education*.
All code and data released openly to support reproducibility, per the call's preference.

## Required cooperation with IEA (≤ 500 words)
The project is designed to require **minimal cooperation and no confidential data** — it
uses only public ILSA international reports and encyclopedia exhibits, and establishes
ground truth with the project team (exhibit answers are objective). This keeps feasibility
and data-availability risk low.

Cooperation that would *strengthen* (but is not required for) the work: (a) a brief
**non-duplication check** with the fund managers to confirm IEA does not already maintain
an internal table-extraction pipeline; (b) pointers to the **canonical/authoritative
versions** of relevant public reports and any machine-readable exhibit sources, to reduce
PDF-parsing noise; (c) light **review of the exhibit taxonomy** for fidelity to IEA
reporting conventions; and (d) support for **dissemination** (General Assembly / IRC
presentation, and publication of outputs on the IEA website). Should IEA later wish to
extend validation to non-public or internal documents, that would be a natural follow-on
requiring a separate data agreement, and is explicitly out of scope here.

## Timeline and milestones (~9 months; start 1 Feb 2027)
| Phase | Months | Milestone / deliverable |
|---|---|---|
| WP0 Scoping & literature review | M1 | Corpus selected; exhibit taxonomy finalized; non-duplication confirmed |
| WP1 Benchmark construction | M1–3 | ~300–500 items authored; IRR on 15% subset; linter clean; v1 dataset |
| WP2 Baseline failure quantification | M2–3 | RQ1 result: table-vs-prose gap measured |
| WP3 Paradigm comparison | M3–6 | RQ2/RQ3 results across text & vision paradigms + metrics |
| WP4 Reference integration & deployment analysis | M6–8 | RQ4: reference build + CPU/GPU deployment guidance |
| WP5 Reporting | M8–9 | Draft report/dataset/code released |
| Peer review buffer | +~8 weeks | Mandatory peer review, revisions, final outputs by Dec 2027 |

## Budget, notes, and justification (Tier 1, EUR; ≤ €55,000)
*Daily rate is a placeholder — insert the actual rate(s).*

| Item | Days | Rate (illustrative) | Cost |
|---|---|---|---|
| WP0 scoping & lit review | 8 | €400 | €3,200 |
| WP1 benchmark construction | 38 | €400 | €15,200 |
| WP2 baseline | 8 | €400 | €3,200 |
| WP3 paradigm comparison | 34 | €400 | €13,600 |
| WP4 integration & deployment | 18 | €400 | €7,200 |
| WP5 report & revisions | 14 | €400 | €5,600 |
| **Staff subtotal** | **120** | | **€48,000** |
| Direct: GPU on-demand rental (vision-RAG runs) | | | €800 |
| Direct: open-access APC (if journal route) | | | €2,000 |
| Direct: compute/storage/misc | | | €300 |
| **Total** | | | **~€51,100** |

*Notes.* Overheads kept minimal per call guidance. The main new direct cost is modest-GPU
rental for the vision paradigm. Headroom to €55k is deliberate but modest; if the real
daily rate is higher, reduce benchmark size or drop the APC rather than cut rigor.

## Team
- **Lead researcher:** `[NAME, ROLE, IEA Hamburg / affiliation]` — designed and executed
  the prior PIRLS RAG evaluation and built the evaluation harness and benchmark scaffolding
  this project extends (demonstrated capability; de-risks timeline and team criteria).
- **Team member(s):** `[NAME(S), ROLES — e.g., annotation lead, ML engineer]` (≤250 words
  each bio at submission).

## References (verified 2026-07-02; confirm final formatting before submission)
- Blecher, L. et al. (2023). *Nougat: Neural Optical Understanding for Academic Documents.* arXiv:2308.13418.
- Chen, J. et al. (2024). *M3-Embedding (BGE-M3).* arXiv:2402.03216.
- Chen, S.-A. et al. (2024). *TableRAG: Million-Token Table Understanding with Language Models.* arXiv:2410.04739 (NeurIPS 2024).
- Chen, Z. et al. (2021). *FinQA: A Dataset of Numerical Reasoning over Financial Data.* arXiv:2109.00122 (EMNLP 2021).
- Cheng, Z. et al. (2022). *HiTab: A Hierarchical Table Dataset for QA and NLG.* arXiv:2108.06712 (ACL 2022).
- Cho, J. et al. (2024). *M3DocRAG.* arXiv:2411.04952.
- Cui, C. et al. (2025). *PaddleOCR 3.0 Technical Report.* arXiv:2507.05595.
- Faysse, M. et al. (2024). *ColPali: Efficient Document Retrieval with Vision Language Models.* arXiv:2407.01449 (ICLR 2025).
- Günther, M. et al. (2024). *Late Chunking.* arXiv:2409.04701.
- Livathinos, N. et al. (2025). *Docling.* arXiv:2501.17887.
- Martin, M.O., Fraillon, J., & Sibberns, H. (2025). *Technical Standards for IEA Studies.*
- Ouyang, L. et al. (2024). *OmniDocBench.* arXiv:2412.07626.
- Sarthi, P. et al. (2024). *RAPTOR.* arXiv:2401.18059.
- Smock, B. et al. (2022). *GriTS: Grid Table Similarity metric.* arXiv:2203.12555.
- Strich, J. et al. (2025). *T²-RAGBench.* arXiv:2506.12071.
- Wang, Z. et al. (2024). *Chain-of-Table.* arXiv:2401.04398.
- Yu, S. et al. (2024). *VisRAG: Vision-based RAG on Multi-modality Documents.* arXiv:2410.10594.
- Zhao, Y. et al. (2022). *MultiHiertt.* ACL 2022 (2022.acl-long.454).
- Zhu, F. et al. (2021). *TAT-QA.* ACL 2021 (2021.acl-long.254) / arXiv:2105.07624.
- Anthropic (2024). *Introducing Contextual Retrieval.* (blog, 19 Sep 2024).
- `[Prior project internal report: Information Retrieval Using RAG on PIRLS Documents (2026).]`

## Declarations (complete at submission)
- **Conflicts of interest:** `[none / declare]`.
- **Use of generative AI:** AI tools assisted with preliminary brainstorming, source
  identification, and building the accompanying open-source evaluation code/scaffolding;
  the intellectual proposal content (background, design, impact, timeline, budget,
  references) is the applicant's own. `[Finalize truthfully.]`
- **Agreement to research-agreement terms:** `[confirm]`.
- **Authorized organizational representative:** `[name]`.
