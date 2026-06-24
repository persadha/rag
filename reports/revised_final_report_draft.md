# Information Retrieval Using Retrieval-Augmented Generation on PIRLS Documents

### A comparative evaluation of retrieval and architecture choices for privacy-preserving, locally-hosted question answering

**Prepared by:** Widianto Persadha, Heiko Sibberns, Mohammad S. Thariq, Bettina Wietzorek
**Prepared for:** IEA — R&D Committee
**Revised:** June 2026 *(revision of the January 2026 final report, incorporating reviewer feedback)*

> **Draft status.** All experiments are scored and final, including the rerank-aware *Advanced v3* pipeline (§6.3.2) and the closed-model closed-book baseline (§5.3).

---

## Executive Summary

**What we built.** A proof-of-concept information-retrieval system that answers natural-language questions over IEA's PIRLS 2021 documentation using Retrieval-Augmented Generation (RAG) with **open-weight LLMs running entirely on local hardware** — no data leaves the machine. It meets the three needs the staff survey identified: data privacy, transparent source attribution, and a usable interface.

**The headline finding — retrieval quality is the dominant lever.** Across a controlled benchmark (195 question–answer pairs, a single held-constant LLM judge), the biggest gains in answer quality came not from a more elaborate "agentic" pipeline but from improving **how documents are retrieved**:

| Lever | Answer correctness | Cost |
|---|---|---|
| Baseline (Basic RAG, local llama3:8b) | 0.473 | — |
| **+ cross-encoder reranking** | **0.625** (+32%) | ~$0 (CPU) |
| **+ hybrid (BM25 + dense) retrieval** | **0.689** (+46% over baseline) | ~$0 (local) |
| + stronger generator (closed gpt-5.4-mini) | **0.774** (best overall) | cloud API |

**RAG is decisively worth it.** A reviewer asked us to prove the system beats an off-the-shelf model. Holding the generator fixed, **retrieval lifts answer correctness from 0.202 (no retrieval) to 0.689** — a +0.49 swing; the best configuration reaches 0.774 (+0.57). On this corpus the open 8B model is close to unusable *without* retrieval.

**The "advanced is better" hypothesis did not hold on local hardware.** A more complex, multi-step "agentic" pipeline (decomposition + grading + re-retrieval) **did not improve answer quality** over the simple pipeline, and cost 4–5× the latency. Part of the very large gap reported in January was a **measurement artifact** in how retrieved context was logged; once corrected, the simple and advanced pipelines are close on answer quality, but the advanced machinery still adds latency without adding accuracy (§6.1, §6.3).

**Recommendation.** Deploy **Basic RAG + hybrid retrieval + reranking** with the **local, open llama3:8b** generator (private, ~$0, answer correctness 0.689, ≈20 s/query). Offer the closed **gpt-5.4-mini** as an *optional* higher-accuracy tier **for non-sensitive queries only**, where the privacy guarantee can be relaxed. Do **not** deploy the agentic pipeline as-is.

**Reviewers.** Both reviewers' substantive requests were addressed with new experiments — the RAG-vs-no-retrieval baseline, the retrieval-only baseline, stage-level attribution of the architecture gap, the knowledge-base scale, the embedding-model correction, and a characterisation of the evaluation questions. Items requiring resources beyond this phase (a calibrated human-expert evaluation; validation on genuinely non-public documents) are carried forward as prioritised future work. A point-by-point crosswalk is in **Appendix A**.

---

## 1. Abstract

Knowledge workers increasingly rely on AI to find information faster, but general-purpose tools fall short in professional settings: closed commercial models can return confident but incorrect answers, cannot see an organisation's internal documents, and send sensitive data to external servers. This project develops a proof-of-concept information-retrieval (IR) system that pairs Retrieval-Augmented Generation (RAG) with open-source LLMs running entirely on local hardware, so an organisation can use AI over its own documents without giving up control of the data.

We evaluate the prototype on PIRLS 2021 documentation — text-heavy, domain-specific material representative of IEA's output. Outputs are scored with an LLM-as-a-judge framework (DeepEval) on four quality metrics (context precision, context recall, faithfulness, answer correctness) plus a judge-free gold-context similarity check, and we measure end-to-end response time. We compare a **Basic** (single-pass) RAG pipeline against an **Advanced** (multi-step, agentic) pipeline, and then systematically search for the levers that most improve answer quality.

The central result is that **retrieval quality, not pipeline complexity, is the dominant lever**: cross-encoder reranking improves answer correctness by 32% at near-zero cost, hybrid (sparse + dense) retrieval adds a further 10%, and a value-added ablation shows retrieval is responsible for the bulk of the system's usefulness (answer correctness 0.202 without retrieval → 0.689 with it). The advanced agentic pipeline did not outperform the simple one on sub-10B local models and cost 4–5× the latency. We recommend a Basic-RAG-plus-hybrid-retrieval configuration for local, private deployment, with an optional closed-model tier for non-sensitive, accuracy-critical queries.

**Keywords:** retrieval-augmented generation; local LLMs; document question answering; hybrid retrieval; reranking; PIRLS; evaluation.

---

## 2. Introduction

### 2.1 Problem Statement

IEA's large-scale studies generate extensive documentation — study frameworks, international reports, methodological guidelines, technical reports, and questionnaires — that hold valuable information about educational trends and the methods used to produce them. The volume and complexity of these materials make it hard for researchers to locate specific facts efficiently; keyword search is slow and imprecise.

Large language models offer fluent, context-aware question answering, but they have well-known limitations in a professional setting: they can hallucinate plausible-but-wrong answers, they cannot answer questions about content created after their training cut-off, and — most consequentially for IEA — most commercial LLMs run on cloud infrastructure, raising data-privacy and security concerns for sensitive internal material. A retrieval system that combines the fluency of modern LLMs with strict, local data governance is therefore needed. This project develops a proof-of-concept that runs entirely on local infrastructure and grounds every answer in retrieved source passages.

### 2.2 Scope

We use the 2021 PIRLS dataset for development and evaluation. PIRLS was chosen for the complexity and diversity of its content — statistical methodology, dense policy and assessment frameworks, and country-level encyclopedic information — making it representative of the high-value, unstructured documentation IEA produces.

The phase reported here focuses on extracting and retrieving high-density information from **PDF documents**. Multi-modal and structured sources (SQL, spreadsheets, HTML) and high-fidelity table extraction are out of scope (§5.5 explains why tables were deferred). The results are intended as a benchmark and a basis for a production deployment.

### 2.3 Significance (motivations and research questions)

*Reviewer 1 noted that the original significance statements read as unsupported claims. We reframe them here as the literature-informed motivations and the research questions this study sets out to test.*

A locally-hosted RAG system could benefit IEA in three ways, each of which this study treats as a question to be tested rather than a claim asserted in advance:

1. **Retrieval efficiency.** RAG can let researchers ask natural-language questions and receive concise, source-grounded answers instead of manually searching large document collections. *Research question:* does retrieval materially improve answer quality over an LLM alone? (Tested in §5.3.)
2. **Reliability through grounding.** Grounding answers in retrieved passages is expected to reduce hallucination and improve verifiability (Lewis et al., 2020; Gao et al., 2024). *Research question:* how faithful and how correct are the grounded answers, and where do they fail? (Tested in §5.2, §5.4.)
3. **Privacy and control.** Local deployment keeps all processing on-premises, supporting compliance and internal policy. *Research question:* can an open, locally-deployable model reach acceptable quality without resorting to a cloud service? (Tested in §6.2.3.)

Beyond these, the project lays a foundation for future AI applications at IEA (report summarisation, research assistance, AI-assisted data cleaning); these remain prospective and are not evaluated here.

### 2.4 Contributions

1. Design and implementation of two locally-hosted RAG pipelines — a **Basic** single-pass system and an **Advanced** multi-step (agentic) variant — on a common open-source stack (LangChain/LangGraph, Chroma, Ollama).
2. A PIRLS 2021 evaluation set combining human-authored and LLM-generated question–answer pairs, with explicit provenance tagging and a characterisation of its cognitive coverage (§4.4).
3. A controlled, judge-held-constant evaluation across four LLM-graded metrics plus a judge-free retrieval check, with end-to-end latency on a workstation-class laptop GPU.
4. A systematic improvement study isolating the levers that drive answer quality (reranking, hybrid retrieval, generator choice, chunking), and a **value-added ablation** quantifying what retrieval and generation each contribute.
5. A stage-level error analysis explaining why the advanced pipeline underperformed, and a production recommendation.

### 2.5 Related Work

RAG couples a parametric language model with a non-parametric document store to improve factuality on knowledge-intensive tasks (Lewis et al., 2020). Surveys emphasise that RAG performance depends as much on retrieval quality and document preprocessing as on the generator (Gao et al., 2024). A common motivation for "agentic" RAG is that complex questions benefit from decomposition into sub-problems (Khot et al., 2022) and from iterative self-reflection (Shinn et al., 2023); Corrective RAG (CRAG) adds post-retrieval validation (Yan et al., 2024), and GraphRAG models documents as connected units for multi-hop retrieval (Edge et al., 2024).

On the retrieval side, the **retrieve-then-rerank** pattern — a wide first-stage retrieval followed by a cross-encoder reranker — is now the dominant approach, and **hybrid** retrieval (sparse BM25 fused with dense vectors, e.g. via reciprocal rank fusion) raises recall on entity-heavy queries that dense embeddings alone miss. Evaluation has moved toward LLM-as-a-judge and component-wise metrics (context precision/recall, faithfulness) to diagnose retrieval and generation failures at scale (Zheng et al., 2023). This project adopts DeepEval's reference-based contextual metrics and answer-correctness scoring (Confident AI, 2026).

---

## 3. Methodology

Our approach builds on the RAG framework (Lewis et al., 2020): coupling the parametric memory of an LLM with a non-parametric retriever lets the system answer from an external index that can be updated without retraining the model. Standard RAG is vulnerable to retrieval noise, which motivated an **Advanced** variant that adds query decomposition (Khot et al., 2022), step-wise reasoning, and post-retrieval validation. As the results show, the value of that extra machinery depends heavily on the capacity of the local model running it.

### 3.1 Data Preparation

The knowledge base was built from publicly available PIRLS 2021 PDFs: the *International Results in Reading*, the *Encyclopedia* (education policy and curriculum), the *Methods and Procedures* technical report, the *User Guide for the International Database*, *Countries' Reading Achievement*, and the school/teacher/student questionnaires. PDFs are loaded with `PyPDFLoader`, split into chunks with a recursive character splitter, embedded, and stored in a Chroma vector index.

**Knowledge-base scale** *(Reviewer 2-6).* Measured directly from the live index:

| Property | Value |
|---|---|
| Source documents | 108 |
| Indexed chunks | 6,771 |
| Total tokens (cl100k_base) | ~1.5 million |
| Mean chunk length | 122 words / 812 characters / 221 tokens |
| On-disk footprint | 70.9 MB |
| Embedding model | `sentence-transformers/all-mpnet-base-v2` (768-dim) |

The chunking parameters and the choice of embedding model are discussed in §6.2.4, where we show retrieval quality — not chunk granularity — is the binding constraint.

### 3.2 User Survey

A requirements survey was rolled out to all IEA staff in the Hamburg and Amsterdam offices.

- **3.2.1 Summary.** 40 staff responded, completing Phase 1 of the program. The clear demand was for a highly accurate tool for navigating technical documentation and report details.
- **3.2.2 Participant profile.** 40 respondents across specialised units (International Study Unit largest at 15; plus Sampling, Data Management, Software, Research & Analysis); 33 English / 7 German.
- **3.2.3 Functional requirements.** Navigating complex study information was the top need — technical documentation and specific study results (e.g. PIRLS) were each cited 28 times. Region comparisons and longitudinal analysis were also requested. Most prefer a "medium" level of detail; a segment wants exhaustive explanations or raw data/tables.
- **3.2.4 Non-functional requirements.** Speed matters — **80% expect answers within 30 seconds**, many preferring a few seconds. **Accuracy is the top priority — over 67% rate precise information "extremely important."**
- **3.2.5 Qualitative insights.** Staff stressed **source attribution** for manual verification and trust, reflecting concern about hallucinations. These drove the design: a grounded retriever and a UI that shows the source chunks behind every answer.

### 3.3 System Architecture

Both pipelines run on a local open-source stack: **LangChain** (loading, splitting, prompting), **LangGraph** (stateful multi-step workflows), **Ollama** (local model inference; swap models without code changes), and **Chroma** (vector store).

#### 3.3.1 Basic Architecture (Basic RAG)

![Figure 1 — Basic RAG pipeline](figures/orig/orig_fig1_basic_arch.jpg)
*Figure 1. Basic RAG pipeline: documents are chunked, embedded, and stored; a query is embedded, the top-k chunks are retrieved, and the LLM answers using only the retrieved context.*

A linear retrieve-then-generate path: PDFs are ingested and chunked; chunks are embedded and stored in Chroma; a query retrieves the top-k chunks by similarity; the chunks are injected into a prompt that instructs the model to answer **using only the provided context** and to say so if the answer is absent; the LLM generates the answer. The original implementation retrieved the top chunks by dense similarity; §6.2 replaces this with a wide-retrieve-then-rerank, hybrid retriever.

#### 3.3.2 Advanced Architecture (Advanced RAG)

![Figure 2b — Advanced RAG workflow](figures/orig/orig_fig2_advanced_arch.jpg)
*Figure 2. The intended Advanced RAG workflow: decompose the query, plan sub-steps, retrieve, answer each sub-question, and synthesise.*

The Advanced pipeline uses a stateful graph rather than a linear path. Its intended design decomposes a complex query into sub-questions, validates retrieved documents, answers each sub-question, and synthesises a final answer.

> **Correction (report-vs-implementation).** The January report described the Advanced pipeline as already performing per-sub-question re-retrieval and reranking. The version actually *evaluated* (we call it **Advanced v1**) did neither: its sub-questions **reused the original query's retrieved documents**, and no reranker was wired into the graph. Its only active post-retrieval step was a binary relevance grader (a small `gemma3:1b` model). The intended re-retrieval and reranking were realised later, in the **Advanced v2/v3** iterations (§6.3). We describe each version by what it actually did, and attribute results accordingly.

The three Advanced versions referenced in this report:

| Version | What it adds over Basic | Status |
|---|---|---|
| **Advanced v1** | `gemma3:1b` document grader; decomposition that **reuses** the original docs; ≤100-word synthesis | Evaluated (this is the original "advanced" system) |
| **Advanced v2** | per-sub-question **re-retrieval** + chunk dedup; no word cap | Evaluated (§6.3) |
| **Advanced v3** | reranker-score grader; rerank the sub-question union against the original query; adaptive decomposition; synthesise from reranked context | Evaluated on the improved retriever (§6.3.2) — matches Basic (AC 0.780 vs 0.774), does not surpass it |

---

## 4. Experimental Setup

### 4.1 Hardware

The original prototype (January 2026 report) was engineered and tested on a mobile workstation — **Alienware M18: Intel Core i9-13980HX, NVIDIA RTX 4090 Laptop GPU (16 GB), 64 GB RAM**. We deliberately avoided cloud GPUs (AWS/Lambda) to keep the system fully offline and to stress-test it under realistic local constraints.

The re-evaluation reported here — the r3 runs and the June 2026 improvement experiments (§5–§6) — was carried out on a **different machine**, because access to the RTX 4090 laptop was limited this period: **Intel Core Ultra 9 (16 cores), 32 GB RAM, Intel Arc Pro 140T GPU (16 GB)**. Both are workstation-class but not equivalent — the Arc-based laptop has roughly half the RAM and a different GPU/driver stack.

> **Latency caveat.** Because the two evaluation rounds ran on different hardware, the **response-time figures in this revision are not directly comparable to the latency reported in the January report** (nor to absolute figures from any other machine). They remain valid for *relative* comparison *within* the June runs — Basic vs. Advanced, local vs. API — which is the only way we use them. The **quality metrics** (context precision/recall, faithfulness, answer correctness) are hardware-independent and unaffected.

The deployment implications of a single-machine setup are discussed in §7 (Reviewer 2-5).

### 4.2 Models

Early tests with large models (e.g. Llama-2 70B) were unusable locally — single queries took minutes. We therefore standardised on **sub-10B open-weight models**: **Llama 3 (8B)** as the primary generator, with **Gemma 3 (4B)** and **DeepSeek-R1 (8B)** as comparison points. For the controlled three-architecture comparison (§5) we hold the generator fixed at **llama3:8b**; the generator is then varied as a separate axis on the best retrieval configuration (§6.2.3), where we also add the closed **gpt-5.4-mini** as an open-vs-closed benchmark.

**Judge.** All quality metrics are scored by a single LLM judge — **`gpt-oss-120b`** (an open-weight model, served via an OpenAI-compatible API) — **held constant across every run** so that comparisons are fair. (Earlier passes used a different judge; numbers across passes are therefore not directly comparable — see §6.1.)

### 4.3 Metrics

Four LLM-graded metrics test different stages, plus one judge-free retrieval check:

- **4.3.1 Context Precision** — the signal-to-noise ratio of retrieved chunks: are the relevant chunks ranked above irrelevant ones?
- **4.3.2 Context Recall** — did retrieval capture all the facts the reference answer needs?
- **4.3.3 Faithfulness** — are the answer's claims supported by the retrieved context (a hallucination detector)?
- **4.3.4 Answer Correctness** — does the answer contain the same facts as the reference answer (scored with G-Eval, an LLM-with-reasoning grader), penalising missing or contradicted facts but not phrasing?
- **Gold-context similarity** *(new in this revision)* — the maximum cosine similarity between the gold reference passage and any retrieved chunk; a deterministic, judge-free check of whether retrieval surfaced the right passage at all.

We chose **DeepEval over RAGAS** after RAGAS proved both far slower and unreliable on a local judge (§6.1); the choice was endorsed by both reviewers. All metrics are 0–1, higher is better (lower is better for latency).

### 4.4 Evaluation Dataset Construction

The evaluation set is a **revised, clean 195-pair dataset** with explicit provenance tagging (`datasets/revision/`). It supersedes the earlier ad-hoc construction (see §5.1.1).

- **4.4.1 Human-generated datapoints (123).** Authored by a domain expert; each answer is verifiable against the source documents. These anchor the evaluation.
- **4.4.2 Synthetic datapoints (72).** Generated with a large model from passages in the corpus, used to broaden coverage. Synthetic items may not reflect real user intent, so we report results split by provenance (§4.4.3, §5.2).

#### 4.4.3 Dataset composition: provenance and question coverage *(Reviewers 1-2, 2-7)*

To answer Reviewer 1's question about what the evaluation set actually tests, we classified all 195 questions by cognitive level and subtype (using the same `gpt-oss-120b` model; `scripts/classify_questions.py`).

![Figure 8 — Question coverage](figures/fig8_question_coverage.svg)
*Figure 8. Eval-set question types (n=195).*

| Cognitive level | Overall | Human (n=123) | Synthetic (n=72) |
|---|---|---|---|
| Fact-retrieval | **68.2%** (133) | 77.2% | 52.8% |
| Reasoning (why/how/analysis) | **31.8%** (62) | 22.8% | 47.2% |

The reviewer's intuition is confirmed and quantified: the set is **dominated by fact-retrieval (~68%)**, and the deep methodological "why/how" questions the reviewer highlighted — sampling (4.6%), plausible values & scaling (3.1%), weighting & variance (1.5%) — are **under-sampled (~9% combined)**. We carry this forward explicitly as a limitation (§7) and recommend deliberately over-sampling reasoning-heavy methodological questions in any follow-up evaluation set.

---

## 5. Results and Discussion

### 5.1 Comparative Performance

The controlled comparison holds dataset, judge, embeddings, retrieval, and generator (llama3:8b) constant; only the architecture varies.

![Figure 6 — r3 canonical comparison](figures/fig6_r3_comparison.svg)

**Table 1. Basic vs. Advanced pipelines (llama3:8b, n=195; `gpt-oss-120b` judge).** Higher is better; latency lower is better.

| Architecture | Ctx Precision | Ctx Recall | Faithfulness | Answer Correctness | Gold-Ctx Sim | Latency (s) |
|---|---|---|---|---|---|---|
| **Basic** | **0.741** | **0.796** | 0.939 | **0.473** | **0.741** | 19.5 |
| Advanced v1 | 0.690 | 0.671 | **0.948** | **0.474** | 0.717 | 74.1 |
| Advanced v2 | 0.449 | 0.650 | 0.910 | 0.334 | 0.712 | 101.8 |

**Basic wins or ties on four of five quality metrics.** Advanced v1 matches Basic only on answer correctness (0.474 vs 0.473) while losing recall; Advanced v2 regresses on precision and answer correctness. Faithfulness is high (~0.91–0.95) everywhere — all three pipelines ground their answers well in whatever context they are given. The differences are in *how each pipeline selects and uses context*, not in raw grounding. Latency rises sharply with complexity (3.8× and 5.2× over Basic).

#### 5.1.1 Evaluation constraints and the clean dataset *(Reviewer 2-8)*

The January report analysed a 300-pair set narrowed to 195 after dropping items the advanced pipeline could not score (over-long outputs exceeding the judge's token limit). Because those exclusions could be systematic, this revision uses a **clean, revised 195-pair dataset** built from scratch with explicit human/synthetic tagging and verified reference answers. All results in this report are on that set; the earlier 300→195 filtering no longer applies. (We later confirmed the old 300-set also had a train/eval leakage problem — see §6.2.4 — reinforcing the move to the clean set.)

### 5.2 Metric-by-metric analysis

- **Context precision & recall.** Basic retrieves focused, on-topic context (precision 0.74, recall 0.80). Advanced v1 loses recall (0.67); Advanced v2 loses precision badly (0.45). §5.4 attributes each loss to a specific stage.
- **Faithfulness.** Uniformly high (0.91–0.95). Grounding is *not* the system's problem — when given context, these models rarely contradict it. This argues against "weak reasoning" as the primary failure mode and against fine-tuning as the first lever.
- **Answer correctness.** Bounded at ~0.47 for all three architectures on the base model. §5.4 explains the ceiling; §6.2 shows how better retrieval pushes through it (to 0.689) without changing the model.
- **Response time.** The single-pass Basic pipeline is far faster. The Advanced pipelines run several sequential LLM calls (grading, per-sub-question generation, synthesis), and latency compounds — especially for reasoning models (DeepSeek-R1 exceeds 80 s/query). See Figure 3. *(All June latency figures were measured on the Intel Arc laptop (§4.1) and are not comparable to the January report's RTX-4090 timings; we use them only for relative comparison within the June runs.)*

![Figure 3 — Latency per run](figures/fig3_latency.svg)
*Figure 3. Mean latency per question (seconds), measured on the June test laptop (Intel Arc Pro 140T; §4.1) — not comparable to the January report's RTX-4090 timings. API runs (\*) use hosted GPUs and are not comparable to the local runs either.*

### 5.3 Is RAG worth it? — the value-added ablation *(Reviewers 1-3 and 2-2)*

Both reviewers asked, from two angles, whether the system earns its complexity: Reviewer 1 asked whether RAG beats an off-the-shelf LLM (no retrieval); Reviewer 2 asked what the **generator** adds over simply returning the retrieved chunks. We ran both as one three-condition ablation (answer correctness, n=195, same judge).

![Figure 7 — Value-added ablation](figures/fig7_value_added.svg)

**Table 2. Value-added ablation (answer correctness).**

| Condition | What it is | Overall | Human | Synthetic |
|---|---|---|---|---|
| Closed-book (open model) | llama3:8b, **no retrieval** | 0.202 | 0.150 | 0.290 |
| Closed-book (closed model) | gpt-5.4-mini, **no retrieval** | 0.490 | 0.377 | 0.683 |
| Retrieval-only | top-4 hybrid chunks **as the answer**, no LLM | 0.827 | 0.852 | 0.785 |
| Full RAG (Basic, local) | llama3:8b + hybrid retrieval | 0.689 | 0.753 | 0.581 |
| Full RAG (best) | gpt-5.4-mini + hybrid retrieval | 0.774 | 0.787 | 0.751 |

**Retrieval is unambiguously worth it — for both open and closed generators.** Holding the generator fixed, retrieval lifts answer correctness **0.202 → 0.689 (+0.49)** for the local open model, and **0.490 → 0.774 (+0.28)** for the strong closed model — retrieval adds substantial accuracy *even to the closed model that scores best without it*. On this corpus the deployable open 8B model is close to unusable without retrieval, the direct empirical justification Reviewer 1 asked for. This also reconciles the reviewer's NAEP reference, where a far larger *closed* model (GPT-4o) scored highly without retrieval: our own closed-book closed model is indeed much stronger than the open one (0.490 vs 0.202) — consistent with that observation — yet it *still* gains +0.28 from retrieval here. The defensible claim is twofold: **for the locally-deployable open model this project targets, retrieval is essential; and even for the strongest closed model, retrieval remains clearly worth it on this corpus.**

**On generation, read the caveat carefully.** Taken at face value, retrieval-only (0.827) *beats* full RAG — implying generation has negative value. **This is a scoring artifact, not a finding that chunks are a better answer.** The retrieval-only "answer" is a ~4,000-character concatenation of chunks; the answer-correctness metric rewards *coverage* of the reference facts and does not penalise verbosity, so a blob that contains the gold facts scores very high, while the LLM's concise synthesis is penalised whenever it compresses a fact. A 4,000-character context dump is not a usable answer. The substantive conclusion is stronger and consistent with everything else here: **the system is retrieval-bound, not generation-bound** — once the right chunks are retrieved, the facts are present; the generator's job is to turn them into a concise, attributable answer. The reviewer's expert/non-expert framing applies: expert users who know the documentation may be served well by direct chunk retrieval, while non-experts benefit from the generator's synthesis. Measuring the generator's true value needs a concision/usability metric, not just fact-coverage — noted as future work (§7).

### 5.4 Why answer correctness is bounded, and why the Advanced pipelines lost *(Reviewer 2-4)*

Reviewer 2 asked us to pinpoint *which* stage of the advanced pipeline fails — decomposition, retrieval, reranking, or synthesis — rather than offer competing explanations. The data localises each loss to a specific, measured mechanism, and the answer is **retrieval/orchestration, not small-model reasoning** (faithfulness is high everywhere).

1. **Advanced v1 — the grader destroys recall.** Its `gemma3:1b` document grader fires on 87/195 questions; where it drops chunks, recall collapses to **0.486** (vs 0.819 when it keeps all four) and precision *also* falls — a weak 1B model is removing answer-bearing chunks and keeping worse ones. It is a net-negative filter.

   ![Figure 4 — grader recall collapse](figures/fig4_crag_recall_collapse.svg)

2. **Advanced v2 — per-sub-question retrieval destroys precision.** Decomposing (~73% of questions) and retrieving fresh chunks per sub-question widens the context (mean 5.9 chunks vs 4.0) with material relevant to *sub*-questions but off-topic for the *original* question against which precision is scored — more than doubling the retrieval-failure rate (107/195 rows below 0.5 precision vs 40 for Basic).

3. **A shared generation ceiling.** Even with good context (precision ≥ 0.7), llama3:8b produces a wrong answer on ~21% of rows — independent of architecture, this caps answer correctness near 0.47 for all three. Better retrieval (§6.2) converts many of these by surfacing the exact gold chunk; a stronger generator (§6.2.3) lifts the rest.

A large part of the *precision/recall* gap reported in January was additionally a **logging artifact**, corrected in this revision (§6.1). The net of the reviewer's dichotomy: **invest in retrieval/orchestration** — which is exactly what §6.2 does, with the largest gains.

### 5.5 Challenges in tabular data extraction

The system could not reliably extract information from tables with nested layouts or graphical elements: standard PDF text extraction flattens table structure, losing row/column relationships. Treating tables as images for Vision-Language Models (e.g. LLaVA) also performed poorly — fragments parsed, but headers and categories were misassociated, and a text embedding model is not suited to spatial layout. Given the engineering effort required, high-fidelity table extraction was deferred (§7 lists specialised parsers to evaluate).

### 5.6 Deployable prototype (user interface)

![Figure — production UI overview](figures/orig/orig_fig9_ui.png)
*Figure 9. The IEA–PIRLS Document Search interface. The left sidebar provides a model-type toggle (Open — local & private / Closed — cloud API), a local Ollama model selector, a high-accuracy hybrid+reranker retrieval switch, and a document-upload panel. The main pane shows the generated answer with provenance metadata (model, retrieval mode, response time, chunk count), a collapsible "Sources used" inspector, and a "Recent questions" history panel.*

![Figure — source-chunk inspector expanded](figures/orig/orig_fig8_ui.png)
*Figure 10. The expandable source-chunk inspector showing the exact retrieved passages behind an answer — directly addressing the survey's demand for verifiable, attributable answers. Each chunk can be traced back to its source document and page.*

We built a web interface (Streamlit) that lets staff query the PIRLS corpus and see both the generated answer and the source chunks behind it. The interface is IEA-branded and exposes the key controls researchers need: a choice between a fully local, privacy-preserving model (llama3:8b via Ollama) and an optional cloud API for higher accuracy, a toggle for the hybrid+reranker retrieval mode, and a live document-upload panel for ad-hoc corpora. Every answer is accompanied by its provenance metadata (model name, retrieval strategy, response time, number of chunks) and an expandable panel showing the exact source passages — making the system auditable and suitable for institutional use.

---

## 6. Improving the System

### 6.1 Evaluation hardening (how the numbers became trustworthy)

The benchmark went through three passes; the comparison is only fair within the final one.

- **Pass r1 (RAGAS).** Abandoned mid-run: the RAGAS harness cascaded through API-quota, local-judge timeout, and library-version failures, and never produced complete advanced-pipeline scores. Lesson: pin the evaluation stack and make scoring resilient to single-row failures.
- **Pass r2 (DeepEval).** Fixed the harness but exposed a **measurement artifact**: the Advanced pipeline's retrieved context was logged as one concatenated blob while Basic's was a list of separate chunks. Context precision/recall are *ranking* metrics over a list — a single blob degenerates them toward zero. This **artificially inflated** the precision/recall gap reported in January. (Proof it is an artifact: the Advanced pipeline runs the same retriever on the same query as Basic, so its context is a superset — scoring the same text far worse is a serialisation bug, not worse retrieval.)
- **Pass r3 (this revision).** With every system's context logged as a chunk list, the embedding upgraded, a retrieval bug fixed, and the judge held constant, r3 is the first trustworthy three-way comparison. The corrected result: Basic and Advanced v1 are close on answer quality (the earlier ~2× precision gap was largely the artifact), but the agentic machinery still adds latency without adding accuracy.

Because the harness, embeddings, and judge all changed, **absolute numbers are not comparable across passes**; only within-pass comparisons are valid. The faithfulness "jump" between r1 and r2, for instance, is a framework difference, not a real improvement.

![Figure 1 — metric evolution](figures/fig1_metric_evolution.svg)
*Figure 10. Metric evolution across passes and experiments (Basic-family runs). The shaded region (RAGAS) is not comparable; the canonical comparison begins at r3.*

### 6.2 Retrieval is the dominant lever

A retrieval-only diagnostic explains why: the gold-bearing chunk is in the top-4 only **63%** of the time, but in the top-20 **87%** of the time. The fix is to retrieve wide and then rerank — recovering the gold chunk into the few that the generator sees.

#### 6.2.1 Reranking (+32% answer correctness, ~$0)

Retrieving the top-20 candidates and rescoring them with a CPU cross-encoder to keep the best 4 lifts answer correctness **0.473 → 0.625 (+32%)**, precision 0.741 → 0.837, recall 0.796 → 0.837. The gain concentrates exactly where it should — on the 40 worst-retrieval rows, where mean precision rises 0.127 → 0.531 and answer correctness 0.185 → 0.333. Cost is effectively zero (a local CPU model).

![Figure 5 — rerank win on worst rows](figures/fig5_rerank_worstcase.svg)

#### 6.2.2 Hybrid retrieval (+10% more; best local configuration)

Adding sparse **BM25** retrieval fused with the dense retriever (reciprocal rank fusion), then reranking, lifts answer correctness a further **0.625 → 0.689** and recall to **0.909** — the highest of any run. Lexical BM25 catches exact terms (country names, programme acronyms) that the dense encoder misses. This is the **best open/local configuration (answer correctness 0.689), and it runs on the free local model** — beating a closed model on plain reranking (0.652). Direct evidence that retrieval, not generator size, is the dominant lever here.

#### 6.2.3 Stronger generator (open vs. closed) *(Reviewer 1-6)*

With the best retrieval fixed, varying the generator is a real, additive second lever:

| Generator | Answer correctness | Latency | Notes |
|---|---|---|---|
| llama3:8b (open, local) | 0.689 | ~20 s | **recommended private default** |
| deepseek-r1:8b (open, local) | 0.717 | ~62 s | dominated — small gain, 3× latency |
| gpt-5.4-mini (closed, cloud) | **0.774** | ~8 s | optional accuracy tier, **non-sensitive only** |

The closed gpt-5.4-mini is the most accurate *and* the fastest, but it runs in the cloud and therefore **forfeits the privacy guarantee**. We present it as (i) the open-vs-closed benchmark the reviewer requested and (ii) an optional tier for non-sensitive, accuracy-critical queries — **not** as the recommended private deployment. The recommended private configuration remains the local open model. Note that retrieval still dominates at the open tier: the local model with hybrid retrieval (0.689) beats the closed model on plain reranking (0.652); the closed model only pulls ahead once it *also* has hybrid retrieval.

#### 6.2.4 Chunking, embedding, and a data-leakage note *(Reviewers 2-6, 2-9)*

- **Chunking sensitivity.** Re-indexing at 512/64 (vs the report's 1000/100) changed answer correctness only marginally (0.473 → 0.488) and slightly lowered precision/recall — confirming chunk granularity is *not* the binding lever (reranking + hybrid are). Index chunks average 812 characters around a ~363-character gold span, so smaller chunks are marginally cleaner but do not substitute for better retrieval. A full grid search remains a refinement, not a priority.
- **Embedding-model correction** *(2-9).* The January report listed the embedding model as "embeddinggemma." The evaluated r3 index in fact uses **`sentence-transformers/all-mpnet-base-v2` (768-dim)**, a general-purpose sentence encoder; the report text is corrected accordingly. The hybrid BM25+dense addition (§6.2.2) already mitigates a single dense encoder's main weakness (exact-term matches), so an embedding-model swap is a lower-priority lever than reranking and hybrid retrieval.
- **Data-leakage note.** The old 300-pair set was found to fully contain the 195-pair evaluation set, leaving only ~105 clean rows — too few for fine-tuning without leakage. This is a further reason the clean revised set (§5.1.1) is used throughout, and why fine-tuning is **not** a near-term lever.

### 6.3 Fixing the Advanced architecture (v1 → v2 → v3)

#### 6.3.1 Why v1 and v2 lost

As established in §5.4: **v1's grader** removes answer-bearing chunks (recall collapse), and **v2's per-sub-question retrieval** injects off-topic chunks (precision collapse). Both distinctive steps *remove* signal. This is not evidence that decomposition is useless — roughly a third of the questions are genuinely multi-hop, where correct decomposition should help — but that these two implementations realise it poorly.

#### 6.3.2 Advanced v3 — a rerank-aware redesign

Advanced v3 applies four targeted fixes, each addressing a measured failure, on top of the best retriever (hybrid + reranking):

1. **Replace the binary grader with reranker-score thresholding** — a proper Corrective-RAG signal that no longer drops answer-bearing chunks (fixes the v1 recall collapse).
2. **Rerank the per-sub-question union against the original question** — re-aligns the context with how precision is scored (fixes the v2 precision collapse).
3. **Adaptive decomposition** — only decompose genuinely multi-hop questions, routing the single-hop majority down the fast Basic path.
4. **Synthesise from reranked context, not just sub-answers** — so facts are not lost in the relay.

**Results (n=195, hybrid + bge reranker; ~9.9 s/query mean).**

| Generator | Ctx Precision | Ctx Recall | Faithfulness | Answer Correctness |
|---|---|---|---|---|
| llama3:8b (open, local) | — | — | — | not completed† |
| gpt-5.4-mini (closed) | 0.855 | 0.897 | 0.980 | **0.780** |

† The local-generator variant generated all 195 answers but only 15/195 were scored before the evaluation window closed (judge-budget contention with the closed-generator run); the partial sample is all-synthetic and not representative, so we report only the completed closed-generator run.

> **Final read.** With the strong closed generator, Advanced v3 reaches **answer correctness 0.780** — statistically indistinguishable from the *Basic* pipeline on the same generator and retriever (**0.774**, §6.2.3), at higher latency (~9.9 s vs 7.9 s/query) and materially higher engineering complexity. The four fixes did exactly what §5.4 predicted they should: they **closed the recall/precision gap that v1's grader and v2's per-sub-question retrieval had opened**, lifting the agentic pipeline from a net-negative architecture back up to parity with Basic. But they did **not surpass** it. The interpretation is clean and consistent with the whole report: once retrieval is fixed (hybrid + reranking), the system is generation-bound, and the agentic orchestration adds cost without adding accuracy on this corpus. **The recommendation is unchanged: ship Basic RAG.** The agentic pipeline remains a research track that now *matches* the simple one rather than beating it — useful to keep for genuinely multi-hop workloads, not to deploy as the default.

### Summary of the improvement study

Best **open/local** configuration: **Basic + hybrid retrieval + reranking, llama3:8b — answer correctness 0.689**. Best **overall** (cloud, non-private): the same retrieval with gpt-5.4-mini — 0.774. The full experiment matrix is in **Appendix B**.

---

## 7. Recommendations and Future Work

### 7.1 Recommended deployment

**Ship Basic RAG + hybrid (BM25 + dense) retrieval + cross-encoder reranking**, with the **local, open llama3:8b** generator as the private default (answer correctness 0.689, ~$0, ≈20 s/query on the June test laptop (§4.1) — within the 30-second bar 80% of staff accept; latency will differ on production hardware). Do **not** deploy the agentic pipeline as-is: it adds 4–5× latency without adding accuracy. Offer the closed gpt-5.4-mini only as an opt-in higher-accuracy tier **for non-sensitive queries**.

### 7.2 Deployment model *(Reviewer 2-5)*

The prototype ran on a single RTX 4090 laptop, which sets the latency figures reported here. For organisational use we recommend a **local network server** (rather than per-user installs): a single on-premises GPU server hosts the model and the vector index and serves the existing web UI to staff browsers. This keeps data on-premises (preserving the privacy thesis), centralises index updates, and — importantly — could host a **larger local model** than a laptop can, mitigating the sub-10B accuracy ceiling. Per-user laptop installs are viable for offline/field use but multiply maintenance and constrain model size. The cloud closed-model tier (§6.2.3) is the only option that leaves the premises and is reserved for non-sensitive queries.

### 7.3 Future work (including reviewer items not yet addressed)

- **Calibrated human-expert evaluation** *(Reviewers 1-5, 2-3).* The most important next step: have subject-matter experts grade a representative subset to calibrate the LLM judge against expert judgement and establish whether a given score is "good enough" for professional use. We did not build expert-grading infrastructure in this phase. Two partial mitigations already reduce the concern: the best answer-correctness (0.774; precision 0.865) is now well above the ≤0.5 band that prompted the reviewer's question, and a small dual-judge cross-check suggests a portion of low scores are judge under-credit rather than wrong answers. A formal expert study remains the right next step.
- **Validation on internal, non-public documents** *(Reviewer 2-1).* The privacy premise is best substantiated on the very material it protects, yet the evaluation corpus is entirely public PIRLS documentation. We acknowledge this as the primary external-validity limitation and frame internal-document validation as a key future-work item, subject to data-governance approval.
- **Reasoning-heavy evaluation questions** *(Reviewer 1-2).* The current set is ~68% fact-retrieval; methodological "why/how" questions are under-sampled (§4.4.3). A follow-up set should deliberately over-sample sampling/weighting/plausible-values reasoning questions.
- **A concision/usability metric** to credit the generator's contribution over raw chunks (§5.3).
- **High-fidelity table extraction** via specialised document parsers (e.g. Docling, Marker, PaddleOCR) and multi-modal retrieval (e.g. ColPali), to handle PIRLS's statistical tables (§5.5).
- **Hardware/latency optimisation** (quantisation; efficient serving such as vLLM; prefix caching) to push the local tier below 10 seconds and enable larger models on a server.

### 7.4 Limitations

Single-domain (PIRLS) public corpus; PDF-to-text preprocessing artifacts; an LLM judge not yet calibrated against human experts; a fact-retrieval-heavy question set; and hardware-specific latency — the June re-evaluation ran on a different, less-capable laptop (Intel Arc Pro 140T, 32 GB) than the original RTX-4090 prototype (§4.1), so response times are not comparable across the two reports and are used only for relative comparison within the June runs. These bound the generalisability of the results and motivate the future work above.

---

## 8. Conclusion

A secure, locally-hosted RAG system over PIRLS documentation is feasible on consumer hardware and meets the organisation's core requirements: privacy through local deployment, source-attributed answers, and a usable interface. The central, well-supported finding is that **retrieval quality — not pipeline complexity — drives answer quality**: cross-encoder reranking and hybrid retrieval together lifted answer correctness from 0.47 to 0.69 on the free local model, and a value-added ablation confirms retrieval is decisively worth its cost. The more elaborate agentic pipeline did not beat the simple one: even a purpose-built rerank-aware redesign (Advanced v3), which fixed the failures that had made earlier versions net-negative, only drew level with Basic on the same strong generator (answer correctness 0.780 vs 0.774) while still costing higher latency and far more complexity. We therefore recommend a Basic-RAG-plus-hybrid-retrieval deployment with a local open model, an optional closed-model tier for non-sensitive queries, and a clear, prioritised path — led by a calibrated human evaluation and validation on internal documents — toward a production, privacy-preserving knowledge-management tool.

---

## 9. References

Brown, T. B., et al. (2020). *Language models are few-shot learners.* NeurIPS. arXiv:2005.14165.
Confident AI. (2026). *DeepEval: The LLM evaluation framework.* https://deepeval.com/docs/
DeepSeek-AI. (2025). *DeepSeek-R1.* arXiv:2501.12948.
Edge, D., et al. (2024). *From local to global: A GraphRAG approach to query-focused summarization.* arXiv:2404.16130.
Gao, L., et al. (2024). *Retrieval-augmented generation for large language models: A survey.* arXiv:2312.10997.
Khot, T., et al. (2022). *Decomposed prompting.* arXiv:2210.02406.
Lewis, P., et al. (2020). *Retrieval-augmented generation for knowledge-intensive NLP tasks.* NeurIPS. arXiv:2005.11401.
Robertson, S., & Zaragoza, H. (2009). *The probabilistic relevance framework: BM25 and beyond.* Foundations and Trends in IR.
Shinn, N., et al. (2023). *Reflexion.* arXiv:2303.11366.
Wei, J., et al. (2022). *Chain-of-thought prompting.* NeurIPS. arXiv:2201.11903.
Yan, S. Q., et al. (2024). *Corrective retrieval augmented generation.* arXiv:2401.15884.
Zheng, L., et al. (2023). *Judging LLM-as-a-judge with MT-bench and Chatbot Arena.* arXiv:2306.05685.

*(Full reference list including reranking/hybrid sources to be finalised at conversion.)*

---

## Appendix A — Reviewer-response crosswalk

Each comment is mapped to a disposition — ✅ **new evidence**, ✏️ **editorial fix**, or 🔭 **future work** — and where it is addressed. No comment is left unaddressed; those not implemented are explicitly carried to future work (§7.3).

**Reviewer 1**

| # | Comment | Disposition | Where |
|---|---|---|---|
| 1-1 | §2.3 significance reads as unsupported claims | ✏️ Editorial | §2.3 reframed as motivations + research questions |
| 1-2 | Scope/cognitive level of eval questions (fact vs why/how) | ✅ + ✏️ | §4.4.3 classification; §7.3/§7.4 limitation |
| 1-3 | Missing baseline: with vs. without retrieval | ✅ New evidence | §5.3 value-added ablation |
| 1-4 | Praise for DeepEval-over-RAGAS rationale | — | §4.3, §6.1 |
| 1-5 | Human-in-the-loop / calibrated expert ratings | 🔭 Future work | §7.3 |
| 1-6 | Test open/foundation models; NAEP GPT-4o no-retrieval | ✅ + ✏️ | §6.2.3 open-vs-closed; §5.3 NAEP framing |

**Reviewer 2**

| # | Comment | Disposition | Where |
|---|---|---|---|
| 2-1 | Privacy premise — validate on internal/non-public docs | 🔭 Future work | §7.3 (primary external-validity limitation) |
| 2-2 | Value added by generation — retrieval-only baseline | ✅ New evidence | §5.3 (with the coverage-artifact caveat) |
| 2-3 | LLM-as-judge circularity; is 0.62 "good enough"? | 🔭 Future work (+ partial) | §7.3 (dual-judge cross-check; scores now well above 0.5) |
| 2-4 | Which stage of the advanced pipeline fails | ✅ Addressed | §5.4 stage attribution |
| 2-5 | Deployment model (standalone/server/networked) | ✏️ Editorial | §7.2 |
| 2-6 | KB scale (chunks/tokens/footprint) + chunking rationale | ✅ New evidence | §3.1, §6.2.4 |
| 2-7 | Human vs. synthetic metric split | ✅ Already in results | §4.4.3, §5.2/§5.3 splits |
| 2-8 | Characterise the 300→195 exclusions | ✏️ Editorial | §5.1.1 (now a clean 195-pair set) |
| 2-9 | Embedding model not described / mislabelled | ✅ + ⚠️ correction | §3.1, §6.2.4 (all-mpnet-base-v2) |

## Appendix B — Full experiment matrix

All runs n=195, judge `gpt-oss-120b`. Best open/local in **bold**; best overall in **bold**.

| Run | Generator | Configuration | Ctx P | Ctx R | Faith | Answer Corr | Latency |
|---|---|---|---|---|---|---|---|
| Basic (baseline) | llama3:8b | k=4, 1000/100 | 0.741 | 0.796 | 0.939 | 0.473 | 19.5 s |
| Advanced v1 | llama3:8b | gemma3:1b grader | 0.690 | 0.671 | 0.948 | 0.474 | 74.1 s |
| Advanced v2 | llama3:8b | per-sub-q retrieve + dedup | 0.449 | 0.650 | 0.910 | 0.334 | 101.8 s |
| Basic + rerank | llama3:8b | k=20→rerank→4 | 0.837 | 0.837 | 0.940 | 0.625 | 21.9 s |
| chunk-512 | llama3:8b | k=4, 512/64 | 0.723 | 0.759 | 0.939 | 0.488 | 11.3 s |
| Basic, deepseek | deepseek-r1:8b | k=4, 1000/100 | 0.742 | 0.804 | 0.983 | 0.595 | 62.7 s |
| Basic + rerank, gpt-5.4 | gpt-5.4-mini | k=20→rerank→4 | 0.837 | 0.847 | 0.981 | 0.652 | 1.6 s\* |
| rerank + extract prompt | llama3:8b | extract style | 0.825 | 0.864 | 0.974 | 0.482 | 19.4 s |
| dense + bge reranker | llama3:8b | k=20→bge-base→4 | 0.807 | 0.850 | 0.957 | 0.578 | ~22 s |
| **hybrid + bge (best local)** | **llama3:8b** | **BM25+dense→RRF→bge→4** | **0.855** | **0.909** | 0.953 | **0.689** | ~23 s |
| hybrid + MiniLM | llama3:8b | BM25+dense→RRF→MiniLM→4 | 0.867 | 0.920 | 0.950 | 0.679 | 19.2 s |
| hybrid + bge, deepseek | deepseek-r1:8b | BM25+dense→RRF→bge→4 | 0.853 | 0.904 | 0.985 | 0.717 | 62.4 s |
| **hybrid + bge, gpt-5.4 (best overall)** | **gpt-5.4-mini** | **BM25+dense→RRF→bge→4** | **0.865** | **0.914** | **0.984** | **0.774** | 7.9 s\* |
| Advanced v3 (rerank-aware) | llama3:8b | hybrid + 4 fixes | — | — | — | n/c† | — |
| Advanced v3 (rerank-aware) | gpt-5.4-mini | hybrid + 4 fixes | 0.855 | 0.897 | 0.980 | 0.780 | 9.9 s\* |

\* API runs use hosted GPUs — not comparable hardware to the local runs.
† Local-generator Advanced v3 generated all 195 answers but only 15/195 were scored before the evaluation window closed (judge-budget contention); not reported. See §6.3.2.

**Value-added ablation (answer correctness, no-retrieval baselines; cf. §5.3).**

| Run | Generator | Configuration | Answer Corr |
|---|---|---|---|
| Closed-book | llama3:8b | no retrieval | 0.202 |
| Closed-book | gpt-5.4-mini | no retrieval | 0.490 |
| Retrieval-only | — | top-4 hybrid chunks as answer | 0.827 |

## Appendix C — Evaluation results

Per-row scores and per-system summaries are provided in the accompanying spreadsheet (`InformationRetrieval_RAG_PIRLS_R&D_FinalReport_Appendix_C_Evaluation_Results.xlsx`, updated for r3) and the repository's `results/` directory. The detailed engineering analysis is in `reports/architecture_evolution_analysis.md`, `reports/crag_improvement_analysis.md`, and `reports/experiments_index.md`.
