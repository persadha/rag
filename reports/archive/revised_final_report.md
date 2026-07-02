# Information Retrieval Using Retrieval-Augmented Generation on PIRLS Documents

### A Comparative Evaluation of Retrieval and Architecture Choices for Privacy-Preserving, Locally-Hosted Question Answering

**Prepared by:** Widianto Persadha, Heiko Sibberns, Mohammad S. Thariq, Bettina Wietzorek
**Prepared for:** IEA — R&D Committee
**Revised:** June 2026 *(a revision of the original 2026 final report)*

---

## Executive Summary

We built a proof-of-concept system that answers natural-language questions over IEA's PIRLS 2021 documentation. It uses Retrieval-Augmented Generation (RAG) with open-weight language models that run entirely on local hardware. The prototype meets the three needs our staff survey identified: data privacy, transparent source attribution, and a web-based user interface.

The main finding is that retrieval quality is what drives answer quality instead of pipeline complexity. We ran a controlled benchmark of 195 question–answer pairs scored by a single LLM judge held constant across every run. The largest gains came from improving how documents are retrieved, not from adding a more sophisticated "agentic" reasoning pipeline on top:

| Lever | Answer correctness | Cost |
|---|---|---|
| Baseline (Basic RAG, local llama3:8b) | 0.473 | — |
| + cross-encoder reranking | 0.625 (+32%) | ~$0 (CPU) |
| + hybrid (BM25 + dense) retrieval | 0.689 (+46% over baseline) | ~$0 (local) |
| + stronger generator (closed gpt-5.4-mini) | 0.774 (best overall) | cloud API |

Retrieval is clearly justified. A central question for any such system is whether it outperforms a general-purpose model used without retrieval. Holding the generator fixed, retrieval lifts answer correctness from 0.202 with no retrieval to 0.689, a swing of +0.49; the best configuration reaches 0.774. On this corpus the open 8B model is close to unusable without retrieval, and even the strongest closed model we tested still gains substantially once retrieval is added.

The "advanced is better" hypothesis did not hold on local hardware. A more complex, multi-step pipeline that decomposes the query, grades retrieved documents, and re-retrieves per sub-question did not improve answer quality over the simple pipeline, yet cost four to five times the latency. Part of the large gap reported in the original report was a measurement artifact in how retrieved context was logged; once corrected, the two pipelines were close on quality. A purpose-built redesign (Advanced v3) reached parity with the simple pipeline but did not surpass it.

Our recommendation is to deploy Basic RAG with hybrid retrieval and reranking, using the local open llama3:8b generator as the private default. That configuration is private, costs essentially nothing to run, scores 0.689 on answer correctness, and returns an answer in roughly 20 seconds. The closed gpt-5.4-mini model can be offered as an optional higher-accuracy tier for non-sensitive queries only. We do not recommend deploying the agentic pipeline as it stands.

The extensions identified during this phase were addressed with new experiments, and the changes relative to the original report are summarized in Appendix A. Items that need resources beyond this phase, chiefly a calibrated human-expert evaluation and validation on non-public documents, are carried forward as prioritized future work.

---

## 1. Abstract

Knowledge workers increasingly rely on AI to find information faster, but general-purpose tools fall short in professional settings. Closed commercial models can return confident but incorrect answers, they cannot see an organization's internal documents, and they send sensitive data to external servers. This project develops a proof-of-concept information-retrieval (IR) system that pairs Retrieval-Augmented Generation with open-source LLMs running entirely on local hardware, so an organization can apply AI to its own documents without giving up control of the data.

We evaluate the prototype on PIRLS 2021 documentation, which is text-heavy, domain-specific material representative of IEA's output. Outputs are scored with an LLM-as-a-judge framework (DeepEval) on four quality metrics — context precision, context recall, faithfulness, and answer correctness — plus a judge-free gold-context similarity check, and we measure end-to-end response time. We compare a Basic (single-pass) RAG pipeline against an Advanced (multi-step, agentic) pipeline, and then search systematically for the levers that most improve answer quality.

The central result is that retrieval quality, not pipeline complexity, is the dominant lever. Cross-encoder reranking improves answer correctness by 32% at near-zero cost, hybrid (sparse plus dense) retrieval adds a further 10%, and a value-added ablation shows that retrieval accounts for the bulk of the system's usefulness: answer correctness rises from 0.202 without retrieval to 0.689 with it. The advanced agentic pipeline did not outperform the simple one on sub-10B local models and cost four to five times the latency. We recommend a Basic-RAG-plus-hybrid-retrieval configuration for local, private deployment, with an optional closed-model tier for non-sensitive, accuracy-critical queries.

**Keywords:** retrieval-augmented generation; local LLMs; document question answering; hybrid retrieval; reranking; PIRLS; evaluation.

---

## 2. Introduction

### 2.1 Problem Statement

IEA's large-scale studies generate extensive documentation: study frameworks, international reports, methodological guidelines, technical reports, and questionnaires. These materials hold valuable information about educational trends and about the methods used to produce them, but their volume and complexity make it hard for researchers to find specific facts quickly. Keyword search is slow and imprecise, and much of the knowledge sits in unstructured PDFs that resist conventional querying.

Large language models offer fluent, context-aware question answering (Brown et al., 2020), but they carry well-known limitations in a professional setting. They can hallucinate plausible-but-wrong answers. They cannot answer questions about content created after their training cut-off. And, most consequentially for IEA, most commercial LLMs run on cloud infrastructure, which raises data-privacy and security concerns for sensitive internal material. What is needed is a system that combines the fluency of modern LLMs with strict, local data governance. This project develops a proof-of-concept that runs entirely on local infrastructure and grounds every answer in retrieved source passages, so that each answer can be checked against the documents it came from.

### 2.2 Scope

We use the 2021 PIRLS dataset for development and evaluation. PIRLS was chosen for the complexity and diversity of its content, which spans statistical methodology, dense policy and assessment frameworks, and country-level encyclopedic information. This makes it representative of the high-value, unstructured documentation IEA produces.

The phase reported here focuses on extracting and retrieving high-density information from PDF documents. Multi-modal and structured sources such as SQL databases, spreadsheets, and HTML, along with high-fidelity table extraction, are out of scope; Section 7.5 explains why tables in particular were deferred. The results are intended both as a benchmark in their own right and as the basis for a future production deployment.

### 2.3 Motivations and Research Questions

The original report stated the project's significance as a set of forward claims about what a locally-hosted RAG system would deliver. On reflection, those claims are better framed as motivations drawn from the literature together with the questions this study sets out to test. A locally-hosted RAG system could benefit IEA in three ways, and we treat each as a question to answer with evidence rather than a benefit to assert in advance.

The first is retrieval efficiency. RAG can let researchers ask natural-language questions and receive concise, source-grounded answers instead of manually searching large document collections. The research question is whether retrieval materially improves answer quality over an LLM used alone, which we test in Section 7.3.

The second is reliability through grounding. Grounding answers in retrieved passages is expected to reduce hallucination and improve verifiability (Lewis et al., 2020; Gao et al., 2024). The research question is how faithful and how correct the grounded answers actually are, and where they fail, which we examine in Sections 7.2 and 7.4.

The third is privacy and control. Local deployment keeps all processing on-premises, which supports compliance and internal policy. The research question is whether an open, locally-deployable model can reach acceptable quality without resorting to a cloud service, which we test in Section 8.2.3.

Beyond these, the project lays groundwork for later AI applications at IEA, such as report summarization, research assistance, and AI-assisted data cleaning. Those uses remain prospective and are not evaluated here.

### 2.4 Contributions

This report makes five contributions.

1. The design and implementation of two locally-hosted RAG pipelines, a Basic single-pass system and an Advanced multi-step (agentic) variant, built on a common open-source stack of LangChain, LangGraph, Chroma, and Ollama.
2. A PIRLS 2021 evaluation set that combines human-authored and LLM-generated question–answer pairs, with explicit provenance tagging and a characterization of its cognitive coverage (Section 5.4).
3. A controlled evaluation, with the judge held constant, across four LLM-graded metrics plus a judge-free retrieval check, reporting end-to-end latency on workstation-class laptop hardware.
4. A systematic improvement study that isolates the levers driving answer quality (reranking, hybrid retrieval, generator choice, and chunking), together with a value-added ablation that quantifies what retrieval and generation each contribute.
5. A stage-level error analysis that explains why the advanced pipeline underperformed, and a production recommendation grounded in the results.

### 2.5 Related Work

RAG couples a parametric language model with a non-parametric document store to improve factuality on knowledge-intensive tasks (Lewis et al., 2020). Surveys of the area make the point that RAG performance depends as much on retrieval quality and document preprocessing as on the generator itself (Gao et al., 2024), a theme our own results echo throughout.

A common motivation for "agentic" RAG is that complex questions benefit from being broken into sub-problems (Khot et al., 2022) and from iterative self-reflection (Shinn et al., 2023). Corrective RAG (CRAG) adds a validation step after retrieval that checks whether the retrieved documents are actually relevant before they are used (Yan et al., 2024), and GraphRAG models a corpus as connected units to support multi-hop retrieval and global summarization (Edge et al., 2024). Selective methods such as Self-RAG go further and decide, for each query, whether retrieval is needed at all (Asai et al., 2023), an idea we return to in Section 8.3.2. Our Advanced pipeline draws on the first two ideas, decomposition and post-retrieval validation, and Section 7.4 reports what happened when a small local model was asked to run that machinery.

On the retrieval side, the retrieve-then-rerank pattern is now standard practice: a wide first-stage retrieval is followed by a cross-encoder that re-scores the candidates and keeps the best few (Nogueira & Cho, 2019). Hybrid retrieval, which fuses sparse BM25 scores (Robertson & Zaragoza, 2009) with dense vector similarity (Karpukhin et al., 2020), often using reciprocal rank fusion (Cormack et al., 2009), raises recall on entity-heavy queries that dense embeddings alone tend to miss. Both techniques turn out to be the most effective levers in our study.

Evaluation has moved toward LLM-as-a-judge scoring and component-wise metrics, such as context precision, context recall, and faithfulness, which make it possible to diagnose retrieval and generation failures separately and at scale (Zheng et al., 2023; Es et al., 2023). This project uses DeepEval's reference-based contextual metrics and its answer-correctness scoring (Confident AI, 2026).

---

## 3. Background

This section explains the techniques the rest of the report builds on, in enough depth that the results can be read on their own terms. Readers already familiar with retrieval-augmented generation can skip ahead; the evaluation metrics are defined later, alongside the experimental setup, in Section 5.3.

### 3.1 Retrieval-Augmented Generation

A language model, left to itself, answers from what it absorbed during training. For a corpus like PIRLS that is a poor arrangement. The model may never have seen the documents, and even where it has, it cannot point to the page a fact came from, so a confident-sounding answer cannot be checked. Retrieval-Augmented Generation (Lewis et al., 2020) takes a different route. Instead of asking the model to recall, it hands the model the relevant source passages at the moment of answering, so the model works from text it can actually see and we can trace every claim back to a document.

It works in two phases. The first happens once, before any question is asked. Every document is split into chunks of a few hundred words, and each chunk is passed through an embedding model that turns it into a vector, a long list of numbers positioned so that passages about similar things land near each other in a high-dimensional space. These vectors are stored in an index. The second phase happens at query time. The question is embedded by the same model, and the system compares the question's vector against every chunk's vector using cosine similarity, the cosine of the angle between them:

*cos(q, d) = (q · d) / (|q| · |d|)*

This is 1 when two vectors point the same way, 0 when they are unrelated, so a higher score means a chunk is more likely to be about the same thing as the question. The system keeps the k highest-scoring chunks, places them in the prompt, and instructs the model to answer from that context alone and to say plainly when the answer is not present.

In this bare form the one setting that matters most is k, the number of chunks retrieved. Set it too low and the passage that actually holds the answer can fall just past the cutoff, leaving the model to guess. Set it too high and the prompt fills with loosely related text that buries the useful sentence and, in practice, nudges a small model toward a fluent but wrong answer. We use k = 4. This whole single-pass arrangement, embed the query, take the top four chunks, generate once, is what we call Basic RAG, and it is the baseline against which every later refinement is measured.

### 3.2 Corrective RAG

Basic RAG takes the retriever at its word. Whatever the top-k chunks happen to be, they go into the prompt, and if some of them are off-topic the model simply has to cope with the noise. Corrective RAG (CRAG; Yan et al., 2024) tries to head that off by inserting a checking step between retrieval and generation. Each retrieved chunk is judged for relevance to the question; the ones that fail are dropped, and if too little is left the system can retrieve again or fall back to what it started with. The intent is to tidy the context before the generator ever reads it.

The catch is that this step is only as good as the judge behind it, and it cuts both ways. A reliable judge strips out the genuine distractors and leaves the answer-bearing chunks untouched, and the generator benefits. A poor judge does real damage, because the step can only ever remove chunks, never recover one it wrongly discarded. When it throws away a chunk that held the answer, recall falls and nothing downstream can undo it. Section 7.4 shows that the lightweight grader wired into our first agentic pipeline behaves like the poor judge often enough to make the whole correction step a net loss on this corpus.

### 3.3 Hybrid Retrieval

Dense retrieval, the cosine-similarity search just described, is good at meaning. Ask about "young readers' attitudes toward books" and it will surface passages on reading motivation even when those exact words never appear. That same strength is its weakness. Because it matches on meaning rather than surface form, it can miss the one literal token a question turns on. Country names, study acronyms, and identifiers such as "PIRLS" or a specific benchmark label do not always sit close together in embedding space, so a question that hinges on one of them can come back with plausible neighbours that are nonetheless wrong.

The long-standing remedy is lexical search. BM25 (Robertson & Zaragoza, 2009) scores a chunk by how often the query's terms occur in it, weighting rare and discriminating words more heavily than common ones and discounting chunks that score only because they are long:

*BM25(q, d) = Σ over terms t in q of  IDF(t) · f(t,d) · (k1 + 1) / ( f(t,d) + k1 · (1 - b + b · |d| / avgdl) )*

Here f(t,d) is how often term t appears in chunk d, IDF(t) is higher for rarer terms, |d| is the chunk's length, avgdl is the average chunk length, and k1 and b are small tuning constants. The practical effect is simple: BM25 reliably finds the chunk that literally contains "Singapore" or "plausible values," which is exactly the case dense search can fumble.

Hybrid retrieval runs both searches and merges their results, so a chunk earns a place if either method ranks it highly. We merge with reciprocal rank fusion (RRF; Cormack et al., 2009), which deliberately ignores the raw scores, since a cosine similarity and a BM25 score are not on the same scale, and uses only where each chunk landed in each list:

*RRF(d) = Σ over retrievers r of  1 / (c + rank_r(d))*

where rank_r(d) is the chunk's position in retriever r's ranking and c is a small constant that keeps the very top ranks from dominating. A chunk near the top of either the dense or the lexical list is therefore lifted in the combined order. Adding BM25 to the dense retriever this way is what recovers the entity-heavy questions that dense similarity alone tends to miss (Karpukhin et al., 2020).

### 3.4 Cross-Encoder Reranking

The retrievers above are fast for a specific reason: they look at the question and each chunk separately. The embedding model encodes the query once and every chunk once, ahead of time, and retrieval is then just a nearest-neighbour lookup over millions of pre-computed vectors. This arrangement, a bi-encoder, scales to a large index effortlessly, but it pays for that speed with a blind spot. The query and the chunk are never read together, so the similarity score is only an approximation of how relevant the chunk really is.

A cross-encoder removes that approximation. Rather than encoding the two pieces apart, it feeds the question and a single candidate chunk into the model joined together, lets every word of the question attend to every word of the chunk, and emits one relevance score grounded in both at once (Nogueira & Cho, 2019). It is markedly more accurate, and far too slow to run across the whole index for every query.

The way to get the speed of the first and the judgement of the second is to use them in sequence, and that is the pattern this report adopts. The fast retriever casts a wide net, here the top 20 candidates, and the slow but accurate cross-encoder then re-scores only those 20 and keeps the best 4 for the generator. The expensive model never sees more than a handful of chunks, so the added cost is small, while the chunk that genuinely answers the question is pulled up into the few the generator reads even when it began outside the naive top 4. Section 8.2 measures exactly this effect and shows it is the single largest source of the quality gains reported here.

---

## 4. Methodology

Our approach builds on the RAG framework (Lewis et al., 2020). Coupling the parametric memory of an LLM with a non-parametric retriever lets the system answer from an external index that can be updated without retraining the model. Standard RAG is vulnerable to retrieval noise, which is what motivated an Advanced variant that adds query decomposition (Khot et al., 2022), step-wise reasoning (Wei et al., 2022), and post-retrieval validation. As the results show, the value of that extra machinery depends heavily on the capacity of the local model running it.

### 4.1 Data Preparation

The knowledge base was built from publicly available PIRLS 2021 PDFs: the *International Results in Reading*, the *Encyclopedia* of education policy and curriculum, the *Methods and Procedures* technical report, the *User Guide for the International Database*, *Countries' Reading Achievement*, and the school, teacher, and student questionnaires. Each PDF is loaded with `PyPDFLoader`, split into chunks with a recursive character splitter, embedded with a sentence-transformer model (Reimers & Gurevych, 2019), and stored in a Chroma vector index. Chroma was chosen because it runs in-process with no external service, which keeps the whole pipeline self-contained and offline.

Measured directly from the live index, the knowledge base has the following scale.

**Table 1. Knowledge-base scale (measured from the live index).**

| Property | Value |
|---|---|
| Source documents | 108 |
| Indexed chunks | 6,771 |
| Total tokens (cl100k_base) | ~1.5 million |
| Mean chunk length | 122 words / 812 characters / 221 tokens |
| On-disk footprint | 70.9 MB |
| Embedding model | `sentence-transformers/all-mpnet-base-v2` (768-dim) |

The chunking parameters and the choice of embedding model are discussed in Section 8.2.4, where we show that retrieval quality, rather than chunk granularity, is the binding constraint.

### 4.2 User Survey

Before building anything, we ran a requirements survey across all IEA staff in the Hamburg and Amsterdam offices to make sure the prototype answered a real need. Forty staff responded, completing Phase 1 of the program. They came from across the specialized units, with the International Study Unit the largest group at 15 respondents, alongside colleagues from Sampling, Data Management, Software, and Research and Analysis. Thirty-three responded in English and seven in German.

The functional requirements were clear. Navigating complex study information was the top need, with technical documentation and specific study results, such as those from PIRLS, each cited 28 times. Staff also asked for cross-region comparisons and longitudinal analysis. Most respondents wanted a "medium" level of detail in answers, though a sizeable segment wanted either exhaustive explanations or direct access to the underlying data and tables.

The non-functional requirements shaped the design just as strongly. Speed is a clear requirement, with 80% of respondents expecting an answer within 30 seconds and many preferring a few seconds. Accuracy ranks even higher, with over 67% rating precise information as "extremely important." In the open-ended responses, staff repeatedly stressed the need for source attribution so they can verify an answer manually, which reflects a clear awareness of how confidently such tools can produce incorrect answers. Those two themes, accuracy and verifiability, shaped the system's design, which pairs a retriever that grounds every answer with an interface that shows the source chunks behind it.

### 4.3 System Architecture

Both pipelines run on a local open-source stack. LangChain handles document loading, splitting, and prompting; LangGraph manages the stateful multi-step workflow used by the Advanced pipeline; Ollama serves the local models, so a model can be swapped without code changes; and Chroma stores the vectors.

#### 4.3.1 Basic Architecture (Basic RAG)

![Figure 1. Basic RAG pipeline.](figures/orig/orig_fig1_basic_arch.jpg)

*Figure 1. The Basic RAG pipeline. Documents are chunked, embedded, and stored; a query is embedded, the top-k chunks are retrieved, and the model answers using only the retrieved context.*

The Basic pipeline is a linear retrieve-then-generate path. PDFs are ingested and chunked, the chunks are embedded and stored in Chroma, and at query time the system retrieves the top-k chunks by similarity. Those chunks are injected into a prompt that instructs the model to answer using only the provided context and to say so when the answer is not present. The model then generates the answer. The original implementation retrieved the top chunks by dense similarity alone; Section 8.2 replaces that with a wider retrieval followed by reranking and hybrid fusion, which is where most of our gains come from.

#### 4.3.2 Advanced Architecture (Advanced RAG)

![Figure 2. Intended Advanced RAG workflow.](figures/orig/orig_fig2_advanced_arch.jpg)

*Figure 2. The intended Advanced RAG workflow: decompose the query, plan sub-steps, retrieve, answer each sub-question, and synthesize a final answer.*

The Advanced pipeline runs as a stateful graph rather than a linear path. Its intended design decomposes a complex query into sub-questions, validates the retrieved documents, answers each sub-question, and synthesizes a final answer from those parts.

There is an important correction to make between what the original report described and what was actually evaluated. The original report described the Advanced pipeline as already performing per-sub-question re-retrieval and reranking. The version that was actually measured, which we now call Advanced v1, did neither. Its sub-questions reused the original query's retrieved documents, and no reranker was wired into the graph. Its only active post-retrieval step was a binary relevance grader run by a small `gemma3:1b` model. The intended re-retrieval and reranking were built later, in the Advanced v2 and v3 iterations described in Section 8.3. To keep the analysis honest, we describe each version by what it actually did and attribute results accordingly.

**Advanced v1, as built.** Figure 3 shows the pipeline that was actually evaluated, which runs as a five-stage graph. First, it retrieves the top four chunks for the original query. Second, a small `gemma3:1b` model grades each chunk as relevant or not and drops those it rejects; if it rejects all four, the pipeline keeps the original four rather than proceeding with no context. Third, it decomposes the query into two or three sub-questions, but answers each from the same original chunks, with no fresh retrieval. Fourth, it answers the sub-questions and synthesizes a final answer capped at 100 words. Fifth, the same `gemma3:1b` model grades the answer for usefulness, and an answer judged unhelpful is regenerated up to two times. The two steps that distinguish v1 from Basic RAG, the binary grader and the decomposition, are precisely the ones Section 7.4 finds to be net-negative on this corpus.

![Figure 3. Advanced v1 (CRAG) as built.](figures/fig_adv_v1.svg)

*Figure 3. The Advanced v1 (CRAG) pipeline as evaluated. The gemma3:1b grader can only drop chunks, the sub-questions reuse the original query's documents, and there is no reranker and no per-sub-question retrieval.*

Three Advanced versions appear in this report.

**Table 2. The three Advanced pipeline versions.**

| Version | Additional Features | Status |
|---|---|---|
| Advanced v1 | `gemma3:1b` document grader; decomposition that reuses the original documents; synthesis capped at 100 words | Evaluated; this is the original "advanced" system |
| Advanced v2 | per-sub-question re-retrieval plus chunk de-duplication; no word cap | Evaluated (Section 8.3) |
| Advanced v3 | reranker-score grader; reranks the sub-question union against the original query; adaptive decomposition; synthesizes from reranked context | Evaluated on the improved retriever (Section 8.3.2); matches Basic (0.780 vs 0.774), does not surpass it |

Versions v2 and v3, and the reasoning behind each change, are described in detail, with their as-built diagrams, in Section 8.3.

---

## 5. Experimental Setup

### 5.1 Hardware

The original prototype, documented in the original report, was engineered and tested on a mobile workstation: an Alienware M18 with an Intel Core i9-13980HX, an NVIDIA RTX 4090 Laptop GPU (16 GB), and 64 GB of RAM. We deliberately avoided cloud GPUs to keep the system fully offline and to stress-test it under realistic local constraints.

The re-evaluation reported here, covering the improvement experiments in Sections 7 and 8, was carried out on a different machine, because access to the RTX 4090 laptop was limited during this period. That machine has an Intel Core Ultra 9 (16 cores), 32 GB of RAM, and an Intel Arc Pro 140T GPU (16 GB). Both are workstation-class, but they are not equivalent: the Arc-based laptop has roughly half the RAM and a different GPU and driver stack.

This matters for one thing in particular. Because the two evaluation rounds ran on different hardware, the response-time figures in this revision are not directly comparable to the latency reported in the original report, nor to absolute figures from any other machine. They are still valid for relative comparison within the re-evaluation runs, that is, Basic versus Advanced and local versus API, which is the only way we use them. The quality metrics, namely context precision and recall, faithfulness, and answer correctness, are hardware-independent and are unaffected. The deployment implications of running on a single machine are discussed in Section 9.2.

### 5.2 Models

The choice of models follows from the privacy goal. Since the aim is to keep data on the organization's own hardware, the generator has to be an open-weight model that can run locally; a closed, cloud-hosted model would send every query and every retrieved passage to an external server, which is the situation this project exists to avoid. That ruled the commercial APIs out as the default and the open-weight families in.

Model size was the next constraint. Early tests with large models such as Llama-2 70B were unusable on local hardware, with single queries taking minutes. As recorded during the interim findings, the question "What are the major advances in PIRLS 2021?" took 139.46 seconds to answer, and another, "What are some of the key advantages of the digital assessment in PIRLS 2021, and how were the systems designed?", took 59.2 seconds, still well over the one-minute mark most users would tolerate. We therefore standardized on sub-10B open-weight models, which keep response times usable on a single GPU. Llama 3 (8B) is the primary generator, chosen for its strong general quality at that size. The original report had also evaluated Gemma 3 (4B), where it scored on par with Llama 3 8B (answer correctness 0.42 against 0.41); it was not carried into the re-evaluation, which standardized on a single fixed generator and then probed generator strength only at the extremes — a stronger reasoning model, DeepSeek-R1 (8B) (DeepSeek-AI, 2025), and a strong closed model, gpt-5.4-mini — where a second mid-size model would have added little. For the controlled three-architecture comparison in Section 7 the generator is held fixed at llama3:8b so that only the pipeline varies. It is then treated as a separate axis on the best retrieval configuration in Section 8.2.3, where the closed gpt-5.4-mini is added as an open-versus-closed benchmark, included not as a deployment candidate but to measure how much accuracy the privacy constraint costs.

Two of the agentic pipelines use a second, smaller model as an internal grader that scores each retrieved chunk for relevance (Section 3.2). We use gemma3:1b for that role deliberately. The grader runs once per chunk and would dominate latency if it were large, so a 1B model is the natural fit for a step meant to be a cheap filter. Section 7.4 shows that the economy proves a poor trade on this corpus, but the reasoning behind the choice is the conventional one.

All quality metrics are scored by a single LLM judge, `gpt-oss-120b`, served through an OpenAI-compatible API. Three considerations drove that choice. It is an open-weight model, so the evaluation is reproducible by anyone and does not hinge on a proprietary endpoint that can change or be withdrawn. It is much larger than any system under test, so it grades their outputs from a position of greater capability. And it is held fixed across every run, so a difference in score reflects a difference in the systems rather than drift in the judge. An earlier evaluation pass used a different framework and judge, which is why scores are not comparable across passes; Section 8.1 covers this in detail.

### 5.3 Metrics

We report four LLM-graded metrics, each probing a different stage of the pipeline, plus one judge-free retrieval check. All scores lie between 0 and 1, and higher is better. The definitions follow DeepEval (Confident AI, 2026).

*Context precision* measures the signal-to-noise ratio of the retrieved chunks: are the relevant chunks ranked above the irrelevant ones? It is the rank-weighted mean of precision-at-k,

*Context Precision = ( Σ over ranks k of [ P(k) × rel(k) ] ) / (number of relevant chunks),*

where rel(k) is 1 when the chunk at rank k is relevant and P(k) is the precision over the top k. For example, if relevant chunks sit at ranks 1 and 3 with an irrelevant chunk at rank 2, the score falls below 1.0 because an irrelevant chunk outranks a relevant one, relative to the ideal ordering of ranks 1 and 2.

*Context recall* measures whether retrieval captured all the facts the reference answer needs. An LLM breaks the reference answer into individual statements and checks each against the retrieved context,

*Context Recall = (reference statements supported by the context) / (total reference statements).*

For example, if the reference makes two claims and only one appears in the retrieved chunks, recall is 0.5.

*Faithfulness* is a hallucination detector: it checks that the answer's claims are grounded in the retrieved context. An LLM extracts the atomic claims in the answer and verifies each against the context,

*Faithfulness = (answer claims supported by the context) / (total answer claims).*

For example, an answer that states one supported fact and one fact absent from the context scores 0.5.

*Answer correctness* compares the generated answer with the reference answer for factual and semantic agreement, independent of wording. It is scored with G-Eval (Y. Liu et al., 2023), in which the judge reasons step by step (chain-of-thought) against a fixed rubric and returns a probability-weighted score. For example, "Singapore scored 587" against a reference of "Singapore: 587 points" scores near 1.0 despite the different phrasing, whereas a missing or contradicted figure is penalized.

The fifth check, new in this revision, is gold-context similarity: the maximum cosine similarity between the gold reference passage and any retrieved chunk. It is deterministic and needs no judge, which makes it a cheap, repeatable way to ask whether retrieval surfaced the right passage at all, independent of how the judge scored the final answer.

We chose DeepEval over RAGAS after RAGAS proved both far slower and unreliable on a local judge, as Section 8.1 recounts. For latency, lower is better.

### 5.4 Evaluation Dataset Construction

The evaluation set is the same 195 question–answer pairs the original report scored, now annotated with explicit human/synthetic provenance and held in `datasets/revision/`. Section 7.1.1 describes how these 195 relate to the original 300-item pool.

Of the 195 pairs, 123 are human-generated, authored by a domain expert, with each answer traceable to the source documents; these anchor the evaluation. The remaining 72 are synthetic, generated with a large model from passages in the corpus to broaden coverage. Because synthetic items may not reflect real user intent, we report results split by provenance throughout Sections 7.2 and 7.3.

#### 5.4.1 Dataset Composition: Provenance and Question Coverage

To characterize what the evaluation set actually tests, we classified all 195 questions by cognitive level and subtype, using the same `gpt-oss-120b` model (`scripts/classify_questions.py`).

![Figure 4. Eval-set question types.](figures/fig8_question_coverage.svg)

*Figure 4. Distribution of the 195 evaluation questions by subtype (n=195), colored by whether they are fact-retrieval or reasoning questions.*

**Table 3. Question coverage by cognitive level and provenance.**

| Cognitive level | Overall | Human (n=123) | Synthetic (n=72) |
|---|---|---|---|
| Fact-retrieval | 68.2% (133) | 77.2% | 52.8% |
| Reasoning (why/how/analysis) | 31.8% (62) | 22.8% | 47.2% |

The composition is worth quantifying. The set is dominated by fact-retrieval questions at about 68%, while the deep methodological "why and how" questions are under-sampled, with sampling at 4.6%, plausible values and scaling at 3.1%, and weighting and variance at 1.5%, roughly 9% combined. We carry this forward explicitly as a limitation in Section 9 and recommend deliberately over-sampling reasoning-heavy methodological questions in any follow-up evaluation set.

---

## 6. From the Original Report to the Re-Evaluation

The original report and this revision are two evaluations of the same family of systems, carried out a few months apart with different evaluation machinery. This section connects them: how the experiment program unfolded, what the original report measured, and why the numbers reported here differ from the ones it published.

### 6.1 Experimental Progression

The work did not begin from a blank slate. It started from the prototype and the findings in the original report, and it proceeded in a deliberate order that is worth setting out, because it explains the shape of the experiment matrix in Appendix B.

We first corrected the problems in the original evaluation (Section 8.1 sets them out in detail) and re-ran the Basic pipeline to establish a baseline we could trust. We then applied the same corrections to the Advanced pipeline and re-evaluated it, first as Advanced v1 and then, after reworking its sub-question handling, as Advanced v2. Across that corrected comparison the Basic pipeline still came out ahead (Section 7.1), so we concentrated the improvement effort there rather than on the agentic machinery.

That decision is why the middle of the experiment matrix varies only the Basic pipeline. The sequence of retrieval and generator levers — reranking, then hybrid retrieval, then a stronger generator up to gpt-5.4-mini — was explored on Basic alone, because Basic was the configuration worth improving. Only once those levers had been characterized did we fold the best of them, hybrid retrieval with a cross-encoder reranker, back into the agentic pipeline. That is the last pair of runs, Advanced v3 (rerank-aware, Section 8.3.2), and its purpose was narrow: to test whether a strong retrieval stack could finally be pushed further by the decomposition machinery. It could not — Advanced v3 only drew level with Basic. The progression therefore runs baseline → corrected Advanced (v1, v2) → a Basic-only improvement search → Advanced v3, and Appendix B reads in that order.

### 6.2 The Original Results, and What Changed

The original report evaluated the Basic and Advanced pipelines across three local generators; its headline numbers are reproduced in Table 4.

**Table 4. Results as reported in the original report (averaged over the same 195 items; higher is better; latency on the original RTX-4090 hardware, in seconds).**

| Pipeline | Generator | Ctx Precision | Ctx Recall | Faithfulness | Answer Correctness | Latency (s) |
|---|---|---|---|---|---|---|
| Basic | Llama 3 8B | 0.62 | 0.54 | 0.98 | 0.41 | 7.4 |
| Basic | Gemma 3 4B | 0.64 | 0.56 | 0.97 | 0.42 | 5.9 |
| Basic | DeepSeek-R1 8B | 0.63 | 0.54 | 0.97 | 0.44 | 19.0 |
| Advanced | Llama 3 8B | 0.29 | 0.31 | 0.97 | 0.23 | 22.9 |
| Advanced | Gemma 3 4B | 0.31 | 0.29 | 0.96 | 0.24 | 15.4 |
| Advanced | DeepSeek-R1 8B | 0.33 | 0.30 | 0.98 | 0.28 | 82.6 |

Two things stand out. The original report already found Basic ahead of Advanced on every model, which is the conclusion this revision reaches as well, so the headline direction has not changed. But the absolute numbers differ sharply, most of all for the Advanced pipeline, whose context precision the original report put at 0.29 against the 0.690 we now measure for the same pipeline on the same questions (Table 5).

That gap is not the system behaving differently; it is the evaluation being corrected. The Advanced pipeline's retrieved context had been logged as one concatenated blob rather than a list of separate chunks, and a ranking metric like context precision collapses toward zero when it is handed a single blob, even though the underlying retrieval is identical to Basic's. Section 8.1 explains the artifact in full. Re-scoring the same retrieval as a proper chunk list lifts Advanced precision and recall back into the same band as Basic, which is why the near-doubling of the precision gap that the original report described largely disappears here.

The Basic pipeline was never touched by that logging bug, yet its numbers still moved — context recall from 0.54 to 0.80, precision from 0.62 to 0.74. Those gains come from fixing the prototype and hardening the evaluation rather than from the scoring correction above. The re-evaluation repaired a retrieval defect, set the retrieval depth to k = 4, logged the retrieved context as a proper chunk list, and held a single judge constant across every run (Section 8.1). The embedding model was `all-mpnet-base-v2` in both the original prototype and this re-evaluation; the original report's reference to "embeddinggemma" was a labelling error (Section 8.2.4), not a change to the system. The result is a Basic baseline whose retrieval quality is, for the first time, measured cleanly enough to improve deliberately.

One caveat governs the whole comparison. The two evaluations used different evaluation machinery — a repaired retrieval path, corrected context logging, and a judge held constant across runs — so the absolute values in Table 4 are not directly comparable to the re-evaluation numbers elsewhere in this report. What is comparable, and what matters, is the direction of the result, Basic ahead of Advanced in both evaluations, and a Basic baseline that is now measured accurately enough to build on.

---

## 7. Results and Discussion

### 7.1 Comparative Performance

The controlled comparison holds the dataset, judge, embeddings, retrieval, and generator (llama3:8b) constant, and varies only the architecture.

![Figure 5. Basic vs Advanced comparison.](figures/fig6_r3_comparison.svg)

*Figure 5. The final-evaluation comparison of Basic, Advanced v1, and Advanced v2 across four quality metrics (llama3:8b, n=195).*

**Table 5. Basic vs Advanced pipelines (llama3:8b, n=195, `gpt-oss-120b` judge).** Higher is better; latency is lower-is-better.

| Architecture | Ctx Precision | Ctx Recall | Faithfulness | Answer Correctness | Gold-Ctx Sim | Latency (s) |
|---|---|---|---|---|---|---|
| Basic | 0.741 | 0.796 | 0.939 | 0.473 | 0.741 | 19.5 |
| Advanced v1 | 0.690 | 0.671 | 0.948 | 0.474 | 0.717 | 74.1 |
| Advanced v2 | 0.449 | 0.650 | 0.910 | 0.334 | 0.712 | 101.8 |

The Basic pipeline performs best or ties on four of the five quality metrics. Advanced v1 matches Basic on answer correctness (0.474 against 0.473) but loses recall, and Advanced v2 regresses on both precision and answer correctness. Faithfulness stays high everywhere, between about 0.91 and 0.95, which tells us all three pipelines ground their answers well in whatever context they are handed. The differences between them are in how each pipeline selects and uses context, not in raw grounding. Latency, meanwhile, rises sharply with complexity, reaching 3.8 times and 5.2 times the Basic pipeline's response time.

#### 7.1.1 The Evaluation Set and Its Selection

The original report assembled a 300-item pool and scored the 195 items that both pipelines could complete, setting aside items whose advanced-pipeline outputs exceeded the judge's token limit. This revision works from those same 195 items, so the architecture comparison runs on identical questions; what we added is an explicit human/synthetic provenance label for each item (123 and 72 respectively), which is what allows results to be split by source. We kept the original 300-to-195 selection rather than re-opening it, both for comparability with the original report and because the full 300-item pool turned out to have a train/eval leakage problem: it entirely contains the 195 evaluation items (Section 8.2.4), which makes the surplus items unsuitable as held-out data.

### 7.2 Metric-by-Metric Analysis

On context precision and recall, the Basic pipeline retrieves focused, on-topic context, scoring 0.74 on precision and 0.80 on recall. Advanced v1 loses recall, dropping to 0.67, and Advanced v2 loses precision badly at 0.45. Section 7.4 attributes each of those losses to a specific stage of the pipeline.

Faithfulness is uniformly high, between 0.91 and 0.95, which tells us grounding is not where the system struggles. When these models are given context, they rarely contradict it. That argues against "weak reasoning" as the primary failure mode, and against fine-tuning as the first lever to consider.

Answer correctness sits at roughly 0.47 for all three architectures on the base model. Section 7.4 explains the ceiling, and Section 8.2 shows how better retrieval pushes through it, to 0.689, without changing the model at all.

On response time, the single-pass Basic pipeline is far faster. The Advanced pipelines run several sequential model calls for grading, per-sub-question generation, and synthesis, so the latency compounds, and it is worst for reasoning models, with DeepSeek-R1 exceeding 80 seconds per query. All re-evaluation latencies were measured on the Intel Arc laptop and are not comparable across reports (Section 5.1).

![Figure 6. Mean latency per question.](figures/fig3_latency.svg)

*Figure 6. Mean latency per question, in seconds, measured on the re-evaluation laptop (Intel Arc Pro 140T). These are not comparable to the original report's RTX-4090 timings, and the API runs marked with an asterisk use hosted GPUs that are not comparable to the local runs either.*

### 7.3 The Value of Retrieval: A Value-Added Ablation

Two questions about whether the system earns its complexity are worth settling directly. First, does RAG actually outperform a general-purpose LLM used without retrieval? Second, what does the generator add over simply returning the retrieved chunks? A single ablation answers both, scoring answer correctness across five conditions on the same 195 questions with the same judge.

![Figure 7. Value-added ablation.](figures/fig7_value_added.svg)

*Figure 7. Answer correctness by condition (n=195). Retrieval lifts correctness sharply for both the open and the closed generator. The retrieval-only bar is high only as an artifact of how G-Eval scores answer correctness: G-Eval rewards coverage of the reference facts and does not penalize a verbose, unsynthesized chunk dump, so a roughly 4,000-character blob that happens to contain the gold facts scores well. It is not a usable answer, and it is not comparable to the RAG bars, which are held to producing a concise, synthesized response.*

**Table 6. Value-added ablation (answer correctness).**

| Condition | What it is | Overall | Human | Synthetic |
|---|---|---|---|---|
| Closed-book (open model) | llama3:8b, no retrieval | 0.202 | 0.150 | 0.290 |
| Closed-book (closed model) | gpt-5.4-mini, no retrieval | 0.490 | 0.377 | 0.683 |
| Retrieval-only | top-4 hybrid chunks as the answer, no LLM | 0.827 | 0.852 | 0.785 |
| Full RAG (Basic, local) | llama3:8b + hybrid retrieval | 0.689 | 0.753 | 0.581 |
| Full RAG (best) | gpt-5.4-mini + hybrid retrieval | 0.774 | 0.787 | 0.751 |

Retrieval is beneficial for both the open and the closed generator. Holding the generator fixed, retrieval lifts answer correctness from 0.202 to 0.689 for the local open model, a gain of +0.49, and from 0.490 to 0.774 for the strong closed model, a gain of +0.28. Retrieval therefore adds substantial accuracy even to the closed model that scores best without it. On this corpus the deployable open 8B model is close to unusable without retrieval, which is the empirical justification the design needs. The result also sits comfortably with the often-cited NAEP finding that a much larger closed model (GPT-4o) scored well without retrieval. The two observations are consistent once model scale is accounted for, since our own closed-book closed model is far stronger than the open one, 0.490 against 0.202, yet still gains +0.28 from retrieval here. The defensible claim is twofold. For the locally-deployable open model this project targets, retrieval is essential, and even for the strongest closed model, retrieval remains clearly beneficial on this corpus.

The generation result needs reading with care. Taken at face value, retrieval-only at 0.827 beats full RAG, which would imply the generator has negative value. That reading is wrong, because it reflects a scoring artifact rather than evidence that raw chunks make a better answer. The retrieval-only "answer" is a roughly 4,000-character concatenation of chunks, and answer correctness rewards coverage of the reference facts without penalizing verbosity, so a blob that contains the gold facts scores very high while the model's concise synthesis is penalized whenever it compresses a fact away. A 4,000-character dump is not a usable answer. The substantive conclusion is consistent with the rest of the report, namely that the system is retrieval-bound rather than generation-bound. Once the right chunks are retrieved the facts are present, and the generator turns them into a concise, attributable answer. A usability distinction is worth drawing here. Experts who already know the documentation may be well served by direct chunk retrieval, whereas non-experts benefit from the generator's synthesis. Measuring the generator's contribution properly would need a concision or usability metric rather than fact-coverage alone, noted as future work in Section 9.

### 7.4 Why Answer Correctness Is Bounded, and Why the Advanced Pipelines Underperformed

It is worth pinpointing which stage of the advanced pipeline actually fails — decomposition, retrieval, reranking, or synthesis — rather than settling for competing explanations. The data localizes each loss to a specific, measured mechanism, and the answer is that the problem lies in retrieval and orchestration rather than in the small model's reasoning, since faithfulness is high everywhere.

The first loss is in Advanced v1, where the grader destroys recall. Its `gemma3:1b` document grader fires on 87 of the 195 questions. Where it drops chunks, recall collapses to 0.486, against 0.819 when it keeps all four, and precision falls as well. A weak 1B model is removing answer-bearing chunks and keeping worse ones, which makes it a net-negative filter.

![Figure 8. Grader recall collapse.](figures/fig4_crag_recall_collapse.svg)

*Figure 8. Context recall in Advanced v1, split by whether the gemma3:1b grader dropped chunks. Where the grader fires, recall collapses from 0.819 to 0.486.*

The second loss is in Advanced v2, where per-sub-question retrieval destroys precision. Decomposing roughly 73% of the questions and then retrieving fresh chunks for each sub-question widens the context, from a mean of 4.0 chunks to 5.9, with material that is relevant to the sub-questions but off-topic for the original question against which precision is scored. That more than doubles the retrieval-failure rate: 107 of 195 rows fall below 0.5 precision, against 40 for Basic.

The third factor is a shared generation ceiling. Even with good context, where precision is 0.7 or higher, llama3:8b produces a wrong answer on about 21% of rows. This is independent of architecture, and it caps answer correctness near 0.47 for all three pipelines. Better retrieval, in Section 8.2, converts many of these rows by surfacing the exact gold chunk, and a stronger generator, in Section 8.2.3, lifts the rest.

A large part of the precision and recall gap reported in the original report was, in addition, a logging artifact, which this revision corrects and Section 8.1 explains. The practical upshot is to invest in retrieval and orchestration, which is exactly what Section 8.2 does, and where the largest gains are found.

### 7.5 Challenges in Tabular Data Extraction

The system could not reliably extract information from tables with nested layouts or graphical elements. Standard PDF text extraction flattens table structure and loses the row-and-column relationships, so a figure that is obvious to a human reader becomes an unstructured sequence of numbers to the retriever. Treating tables as images for a vision-language model such as LLaVA (H. Liu et al., 2023) also performed poorly: fragments parsed, but headers and categories were misassociated, and a text embedding model is not built for spatial layout in any case. Given the engineering effort that a proper fix would require, we deferred high-fidelity table extraction; Section 9.3 lists the specialized parsers we would evaluate next.

### 7.6 Deployable Prototype (User Interface)

![Figure 9. Production UI overview.](figures/orig/orig_fig9_ui.png)

*Figure 9. The IEA–PIRLS Document Search interface. The left sidebar offers a model-type toggle (open, local and private, versus closed, cloud API), a local Ollama model selector, a high-accuracy hybrid-plus-reranker retrieval switch, and a document-upload panel. The main pane shows the generated answer with its provenance metadata (model, retrieval mode, response time, and chunk count), a collapsible "Sources used" inspector, and a "Recent questions" history panel.*

![Figure 10. Source-chunk inspector expanded.](figures/orig/orig_fig8_ui.png)

*Figure 10. The expandable source-chunk inspector, showing the exact retrieved passages behind an answer. This directly addresses the survey's demand for verifiable, attributable answers, and each chunk can be traced back to its source document and page.*

We built a web interface in Streamlit that lets staff query the PIRLS corpus and see both the generated answer and the source chunks behind it. The interface carries IEA branding and exposes the controls researchers require: a choice between a fully local, privacy-preserving model (llama3:8b via Ollama) and an optional cloud API for higher accuracy, a toggle for the hybrid-plus-reranker retrieval mode, and a live document-upload panel for ad-hoc corpora. Every answer carries its provenance metadata, including the model name, the retrieval strategy, the response time, and the number of chunks used, alongside an expandable panel that shows the exact source passages. A simple query such as "What is PIRLS?" returns an answer in about 17 seconds on the local model, with the four source chunks accessible in a single click. This combination of a grounded answer and visible sources makes the system auditable and suitable for institutional use.

---

## 8. Improving the System

### 8.1 Evaluation Hardening: How the Numbers Became Trustworthy

The benchmark went through three passes, and the comparison is only fair within the final one. The progression is worth describing, because it explains a discrepancy with the figures in the original report.

The first pass used RAGAS and was abandoned mid-run. The RAGAS harness cascaded through API-quota errors, local-judge timeouts, and library-version failures, and it never produced complete scores for the advanced pipeline. The lesson we took from it was to pin the evaluation stack and make scoring resilient to single-row failures.

The second pass moved to DeepEval. It fixed the harness but exposed a measurement artifact. The Advanced pipeline's retrieved context had been logged as a single concatenated blob, while the Basic pipeline's was logged as a list of separate chunks. Because context precision and recall are ranking metrics computed over a list, a single blob degenerates them toward zero, which artificially inflated the precision and recall gap reported in the original report. That it is an artifact rather than genuinely worse retrieval is easy to show, since the Advanced pipeline runs the same retriever on the same query as Basic, so its context is a superset of Basic's, and scoring the same text far worse can only be a serialization bug.

The third and final pass is the one this report relies on. With every system's context logged as a chunk list, a retrieval bug fixed, and the judge held constant, it is the first trustworthy three-way comparison. The corrected picture is that Basic and Advanced v1 are close on answer quality, since the earlier near-doubling of the precision gap was largely the artifact, while the agentic machinery still adds latency without adding accuracy.

Because the harness, the embeddings, and the judge all changed between passes, absolute numbers are not comparable across passes, and only within-pass comparisons are valid. The jump in faithfulness between the first two passes, for example, is a framework difference rather than a real improvement.

![Figure 11. Metric evolution across passes.](figures/fig1_metric_evolution.svg)

*Figure 11. Metric evolution across the evaluation passes and experiments for the Basic-family runs. The shaded RAGAS region is not comparable; the trustworthy comparison begins at the final pass.*

### 8.2 Retrieval Is the Dominant Lever

A retrieval-only diagnostic shows why retrieval is the lever that matters most. The gold-bearing chunk is in the top 4 only 63% of the time, but it is in the top 20 fully 87% of the time. The fix is to retrieve wide and then rerank, pulling the gold chunk back into the few that the generator actually sees.

#### 8.2.1 Reranking (+32% Answer Correctness, ~$0)

Retrieving the top 20 candidates and re-scoring them with a CPU cross-encoder to keep the best 4 lifts answer correctness from 0.473 to 0.625, a 32% gain. Precision rises from 0.741 to 0.837 and recall from 0.796 to 0.837. The gain concentrates exactly where it should, on the 40 worst-retrieval rows, where mean precision climbs from 0.127 to 0.531 and answer correctness from 0.185 to 0.333. The cost is effectively zero, since the reranker is a local CPU model.

![Figure 12. Rerank gains on the worst rows.](figures/fig5_rerank_worstcase.svg)

*Figure 12. The effect of reranking on the 40 worst-retrieval rows: mean precision and answer correctness both rise substantially, which is where the overall gain comes from.*

#### 8.2.2 Hybrid Retrieval (+10% More; Best Local Configuration)

Adding sparse BM25 retrieval, fused with the dense retriever through reciprocal rank fusion and then reranked, lifts answer correctness a further step from 0.625 to 0.689, and pushes recall to 0.909, the highest of any run. The reason is that lexical BM25 catches exact terms, such as country names and program acronyms, that the dense encoder misses. This is the best open and local configuration at 0.689, and it runs on the free local model. It even beats a closed model using plain reranking, which scores 0.652, which is direct evidence that retrieval, not generator size, is the dominant lever here.

#### 8.2.3 Stronger Generator: Open vs Closed

With the best retrieval held fixed, varying the generator is a real, additive second lever.

**Table 7. Generator comparison on the best retrieval configuration.**

| Generator | Answer correctness | Latency | Notes |
|---|---|---|---|
| llama3:8b (open, local) | 0.689 | ~20 s | recommended private default |
| deepseek-r1:8b (open, local) | 0.717 | ~62 s | dominated: small gain, 3x latency |
| gpt-5.4-mini (closed, cloud) | 0.774 | ~8 s | optional accuracy tier, non-sensitive only |

The closed gpt-5.4-mini is both the most accurate and the fastest, but it runs in the cloud and so forfeits the privacy guarantee. We present it for two reasons. It is the open-versus-closed benchmark introduced in Section 5.2, and it is an optional tier for non-sensitive, accuracy-critical queries. It is not the recommended private deployment. The recommended private configuration remains the local open model. Retrieval still dominates even at the open tier: the local model with hybrid retrieval (0.689) outperforms the closed model on plain reranking (0.652), and the closed model surpasses it only once it also uses hybrid retrieval.

#### 8.2.4 Chunking, Embedding, and a Data-Leakage Note

On chunking sensitivity, re-indexing at 512/64, against the original 1000/100, changed answer correctness only marginally, from 0.473 to 0.488, and slightly lowered precision and recall. That confirms chunk granularity is not the binding lever; reranking and hybrid retrieval are. Index chunks average 812 characters around a gold span of roughly 363 characters, so smaller chunks are a little cleaner but do not substitute for better retrieval. A full grid search remains a refinement rather than a priority.

On the embedding model, a correction is needed. The original report listed the embedding model as "embeddinggemma," but both the original prototype and this re-evaluation in fact use `sentence-transformers/all-mpnet-base-v2`, a 768-dimensional general-purpose sentence encoder from Hugging Face; the earlier label was an error, corrected here. The hybrid BM25-plus-dense addition in Section 8.2.2 already mitigates a single dense encoder's main weakness, exact-term matching, so swapping the embedding model is a lower-priority lever than reranking and hybrid retrieval.

Finally, a data-leakage note. The 300-item pool was found to fully contain the 195 evaluation items, leaving only about 105 genuinely separate rows, too few to fine-tune on without leakage. This is a further reason the evaluation set from Section 7.1.1 is used as-is throughout, and why fine-tuning is not a near-term lever.

### 8.3 Revising the Advanced Architecture (v1 to v3)

#### 8.3.1 Why v1 and v2 Underperformed

Section 7.4 established the diagnosis. Advanced v1's grader removes answer-bearing chunks and collapses recall, and Advanced v2's per-sub-question retrieval injects off-topic chunks and collapses precision. Both of the steps that distinguish those pipelines from Basic end up removing signal. That is not evidence that decomposition is useless, since roughly a third of the questions are genuinely multi-hop and should benefit from correct decomposition, but it is evidence that these two implementations realize it poorly.

**Advanced v2, as built.** Figure 13 shows where v2 departs from v1. It keeps the `gemma3:1b` grader but changes how sub-questions are handled: instead of reusing the original chunks, it retrieves a fresh set of chunks for each sub-question, de-duplicates them, and merges them into a single de-duplicated union that the generator works from; the 100-word synthesis cap is also removed. The intent is reasonable, since a genuine sub-question deserves its own evidence. On this corpus, though, decomposition fires on roughly 73% of questions, and the per-sub-question retrieval widens the context from a mean of four chunks to almost six. Many of those extra chunks answer a sub-question well but are off-topic for the original question, against which context precision is scored, so precision falls from 0.74 to 0.45.

![Figure 13. Advanced v2 (CRAG++) as built.](figures/fig_adv_v2.svg)

*Figure 13. Advanced v2 adds per-sub-question retrieval and a de-duplicated union (highlighted) on top of v1. The extra retrieval widens the context with chunks that are relevant to the sub-questions but off-topic for the original query.*

#### 8.3.2 Advanced v3: A Rerank-Aware Redesign

Advanced v3 applies four targeted fixes, each aimed at a measured failure, on top of the best retriever from Section 8.2.

1. It replaces the binary grader with reranker-score thresholding, a proper Corrective-RAG signal that no longer drops answer-bearing chunks, which fixes the v1 recall collapse.
2. It reranks the per-sub-question union against the original question, re-aligning the context with how precision is scored, which fixes the v2 precision collapse.
3. It uses adaptive decomposition, only breaking down questions that are genuinely multi-hop and routing the single-hop majority down a faster direct path. A question is treated as single-hop when its top reranked chunk scores at least 0.7 and clearly outscores the runner-up; otherwise it is decomposed.
4. It synthesizes from the reranked context rather than from sub-answers alone, so that facts are not lost across the multi-step process.

![Figure 14. Advanced v3 (rerank-aware) as built.](figures/fig_adv_v3.svg)

*Figure 14. Advanced v3 replaces the binary grader with reranker-score thresholding (T2.3), routes single-hop questions straight to a direct answer and decomposes only genuine multi-hop questions (T2.5), reranks the sub-question union against the original query (T2.4), and synthesizes from the reranked context (T3.7). Without a reranker the pipeline falls back to v2.*

The results, on the same 195 questions with the hybrid-plus-bge reranker and a mean latency of about 9.9 seconds per query, are below.

**Table 8. Advanced v3 results (n=195, hybrid + bge reranker).**

| Generator | Ctx Precision | Ctx Recall | Faithfulness | Answer Correctness |
|---|---|---|---|---|
| llama3:8b (open, local) | — | — | — | not completed* |
| gpt-5.4-mini (closed) | 0.855 | 0.897 | 0.980 | 0.780 |

*The local-generator variant generated all 195 answers, but only 15 of 195 were scored before the evaluation window closed, owing to judge-budget contention with the closed-generator run. The partial sample is all-synthetic and not representative, so we report only the completed closed-generator run.

With the strong closed generator, Advanced v3 reaches an answer correctness of 0.780. That is statistically indistinguishable from the Basic pipeline on the same generator and retriever, which scores 0.774 (Section 8.2.3), and it comes at higher latency, about 9.9 against 7.9 seconds per query, and materially more engineering complexity. The four fixes did exactly what Section 7.4 predicted they should: they closed the recall and precision gap that v1's grader and v2's per-sub-question retrieval had opened, and they lifted the agentic pipeline from a net-negative architecture back up to parity with Basic. What they did not do is surpass it. The interpretation is clean and consistent with the rest of the report. Once retrieval is fixed with hybrid fusion and reranking, the system is generation-bound, and the agentic orchestration adds cost without adding accuracy on this corpus. We therefore continue to recommend Basic RAG for deployment. The agentic pipeline merits retention as a research direction for genuinely multi-hop workloads, now that it matches the simple pipeline rather than underperforming it, but it is not recommended for default deployment.

#### 8.3.3 Summary of the Improvement Study

The best open and local configuration is Basic RAG with hybrid retrieval and reranking on llama3:8b, at an answer correctness of 0.689. The best overall configuration, in the cloud and not private, is the same retrieval with gpt-5.4-mini, at 0.774. The full experiment matrix is in Appendix B.

---

## 9. Recommendations and Future Work

### 9.1 Recommended Deployment

We recommend shipping Basic RAG with hybrid (BM25 plus dense) retrieval and cross-encoder reranking, using the local open llama3:8b generator as the private default. That configuration scores 0.689 on answer correctness, costs essentially nothing to run, and returns an answer in roughly 20 seconds on the re-evaluation laptop, which is within the 30-second bar 80% of staff said they would accept. Latency will differ on production hardware. We do not recommend deploying the agentic pipeline as it stands, since it adds four to five times the latency without adding accuracy. The closed gpt-5.4-mini model should be offered only as an opt-in higher-accuracy tier for non-sensitive queries.

### 9.2 Deployment Model

The prototype ran on a single laptop, which is what sets the latency figures in this report. For organizational use we recommend a local network server rather than per-user installs. A single on-premises GPU server would host the model and the vector index and serve the existing web interface to staff browsers. That keeps data on-premises, which preserves the privacy premise; it centralizes index updates; and, importantly, it could host a larger local model than a laptop can, which would help with the sub-10B accuracy ceiling. Per-user laptop installs remain viable for offline or field use, but they multiply maintenance and constrain model size. The cloud closed-model tier from Section 8.2.3 is the only option that leaves the premises, and it is reserved for non-sensitive queries.

### 9.3 Future Work

Several findings point to clear next steps, listed roughly in priority order.

The most important is a calibrated human-expert evaluation. Subject-matter experts should grade a representative subset, both to calibrate the LLM judge against expert judgement and to establish whether a given score is sufficient for professional use. We did not build expert-grading infrastructure in this phase. Two partial mitigations already reduce the concern. The best answer correctness, at 0.774 with precision of 0.865, is now well above the 0.5 threshold that would otherwise call sufficiency into question, and a small dual-judge cross-check suggests that a portion of the low scores are the judge under-crediting a correct answer rather than the answer being wrong. A formal expert study remains the appropriate next step.

Next is validation on internal, non-public documents. The privacy premise is best demonstrated on the very material it is meant to protect, yet the evaluation corpus here is entirely public PIRLS documentation. We treat this as the primary external-validity limitation, and we frame internal-document validation as a key future-work item, subject to data-governance approval.

The evaluation set itself should incorporate more reasoning-heavy questions. The current set is about 68% fact-retrieval, and methodological "why and how" questions are under-sampled (Section 5.4.1). A follow-up set should deliberately over-sample questions about sampling, weighting, and plausible values.

Beyond these, three engineering directions would be valuable. A concision or usability metric would allow the generator's contribution over raw chunks to be credited, which the current fact-coverage metric cannot do (Section 7.3). High-fidelity table extraction, using specialized document parsers such as Docling, Marker, or PaddleOCR, together with multi-modal retrieval such as ColPali (Faysse et al., 2024), would address the table problem from Section 7.5. And hardware and latency optimization, through quantization, efficient serving such as vLLM, and prefix caching, would push the local tier below 10 seconds and make room for larger models on a server.

### 9.4 Limitations

The results are bounded in several ways that should be stated explicitly. The corpus is a single-domain, public one (PIRLS). PDF-to-text preprocessing introduces artifacts. The LLM judge is not yet calibrated against human experts. The question set is weighted toward fact-retrieval. And the latency is hardware-specific. The re-evaluation ran on a different, less capable laptop (the Intel Arc Pro 140T with 32 GB) than the original RTX-4090 prototype, so response times are not comparable across the two reports and are used only for relative comparison within the re-evaluation runs. These bound how far the results generalize, and they motivate the future work above.

---

## 10. Conclusion

A locally-hosted RAG system over PIRLS documentation is feasible on consumer hardware, and it meets the core requirements: privacy through local deployment, source-attributed answers, and a usable interface. The central, well-supported finding is that retrieval quality, not pipeline complexity, drives answer quality. Cross-encoder reranking and hybrid retrieval together lifted answer correctness from 0.47 to 0.69 on the free local model, and a value-added ablation confirms that retrieval justifies its cost. The more elaborate agentic pipeline did not outperform the simple one. Even a purpose-built rerank-aware redesign (Advanced v3), which addressed the failures that had made the earlier versions net-negative, only reached parity with Basic on the same strong generator (0.780 against 0.774), while still incurring more latency and far more complexity. We therefore recommend a Basic-RAG-plus-hybrid-retrieval deployment with a local open model, an optional closed-model tier for non-sensitive queries, and a clear, prioritized path, led by a calibrated human evaluation and validation on internal documents, toward a production, privacy-preserving knowledge-management tool.

---

## 11. References

Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2023). *Self-RAG: Learning to retrieve, generate, and critique through self-reflection.* arXiv:2310.11511.

Brown, T. B., et al. (2020). *Language models are few-shot learners.* NeurIPS. arXiv:2005.14165.

Confident AI. (2026). *DeepEval: The LLM evaluation framework.* https://deepeval.com/docs/

Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). *Reciprocal rank fusion outperforms Condorcet and individual rank learning methods.* SIGIR.

DeepSeek-AI. (2025). *DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning.* arXiv:2501.12948.

Edge, D., et al. (2024). *From local to global: A GraphRAG approach to query-focused summarization.* arXiv:2404.16130.

Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). *RAGAS: Automated evaluation of retrieval augmented generation.* arXiv:2309.15217.

Faysse, M., Sibille, H., Wu, T., Omrani, B., Viaud, G., Hudelot, C., & Colombo, P. (2024). *ColPali: Efficient document retrieval with vision language models.* arXiv:2407.01449.

Gao, L., et al. (2024). *Retrieval-augmented generation for large language models: A survey.* arXiv:2312.10997.

Karpukhin, V., et al. (2020). *Dense passage retrieval for open-domain question answering.* EMNLP. arXiv:2004.04906.

Khot, T., et al. (2022). *Decomposed prompting: A modular approach for solving complex tasks.* arXiv:2210.02406.

Lewis, P., et al. (2020). *Retrieval-augmented generation for knowledge-intensive NLP tasks.* NeurIPS. arXiv:2005.11401.

Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). *Visual instruction tuning.* NeurIPS. arXiv:2304.08485.

Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., & Zhu, C. (2023). *G-Eval: NLG evaluation using GPT-4 with better human alignment.* EMNLP. arXiv:2303.16634.

Nogueira, R., & Cho, K. (2019). *Passage re-ranking with BERT.* arXiv:1901.04085.

Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence embeddings using Siamese BERT-networks.* EMNLP. arXiv:1908.10084.

Robertson, S., & Zaragoza, H. (2009). *The probabilistic relevance framework: BM25 and beyond.* Foundations and Trends in Information Retrieval.

Shinn, N., et al. (2023). *Reflexion: Language agents with verbal reinforcement learning.* arXiv:2303.11366.

Wei, J., et al. (2022). *Chain-of-thought prompting elicits reasoning in large language models.* NeurIPS. arXiv:2201.11903.

Yan, S. Q., et al. (2024). *Corrective retrieval augmented generation.* arXiv:2401.15884.

Zheng, L., et al. (2023). *Judging LLM-as-a-judge with MT-bench and Chatbot Arena.* arXiv:2306.05685.

---

## Appendix A — Changes from the Original Report

This appendix summarizes how the present report differs from the original. Each change is classified as **added evidence** (a new experiment or analysis), an **editorial** revision, or an item **deferred** to future work (Section 9.3).

| Change | Type | Section |
|---|---|---|
| Reframed the significance statements as motivations and explicit research questions | Editorial | §2.3 |
| Corrected the Basic and Advanced architecture descriptions to match what was actually evaluated | Editorial | §4.3.2 |
| Added knowledge-base scale figures and a chunking-sensitivity analysis | Added evidence | §4.1, §8.2.4 |
| Corrected the embedding-model description to `all-mpnet-base-v2` | Correction | §4.1, §8.2.4 |
| Documented the rationale for the generator, grader, and judge choices, and for DeepEval over RAGAS | Editorial | §5.2, §5.3 |
| Classified the evaluation questions by cognitive level and flagged the under-sampling of reasoning questions | Added evidence; editorial | §5.4.1, §9.4 |
| Reported every metric split by human and synthetic provenance | Added evidence | §5.4.1, §7.2–6.3 |
| Clarified the 300-to-195 item selection and reused the same 195 items for comparability | Editorial | §7.1.1 |
| Added a with-versus-without-retrieval baseline (value-added ablation) | Added evidence | §7.3 |
| Added a retrieval-only baseline isolating what the generator contributes | Added evidence | §7.3 |
| Added an open-versus-closed generator comparison and related it to the NAEP no-retrieval finding | Added evidence; editorial | §8.2.3, §7.3 |
| Added a stage-level attribution of where the advanced pipeline loses quality | Added evidence | §7.4 |
| Hardened the evaluation: chunk-list logging, fixed embedding and retrieval, and a single held-constant judge | Added evidence | §8.1 |
| Added a deployment-model discussion (local network server versus per-user installs) | Editorial | §9.2 |
| Deferred calibrated human-expert evaluation, validation on internal documents, and full judge-sufficiency calibration | Future work | §9.3 |

---

## Appendix B — Full Experiment Matrix

All runs are n=195, judged by `gpt-oss-120b`. The best open/local configuration and the best overall configuration are in bold.

| Run | Generator | Configuration | Ctx P | Ctx R | Faith | Answer Corr | Latency |
|---|---|---|---|---|---|---|---|
| Basic (baseline) | llama3:8b | k=4, 1000/100 | 0.741 | 0.796 | 0.939 | 0.473 | 19.5 s |
| Advanced v1 | llama3:8b | gemma3:1b grader | 0.690 | 0.671 | 0.948 | 0.474 | 74.1 s |
| Advanced v2 | llama3:8b | per-sub-q retrieve + dedup | 0.449 | 0.650 | 0.910 | 0.334 | 101.8 s |
| Basic + rerank | llama3:8b | k=20→rerank→4 | 0.837 | 0.837 | 0.940 | 0.625 | 21.9 s |
| chunk-512 | llama3:8b | k=4, 512/64 | 0.723 | 0.759 | 0.939 | 0.488 | 11.3 s |
| Basic, deepseek | deepseek-r1:8b | k=4, 1000/100 | 0.742 | 0.804 | 0.983 | 0.595 | 62.7 s |
| Basic + rerank, gpt-5.4 | gpt-5.4-mini | k=20→rerank→4 | 0.837 | 0.847 | 0.981 | 0.652 | 1.6 s* |
| rerank + extract prompt | llama3:8b | extract style | 0.825 | 0.864 | 0.974 | 0.482 | 19.4 s |
| dense + bge reranker | llama3:8b | k=20→bge-base→4 | 0.807 | 0.850 | 0.957 | 0.578 | ~22 s |
| **hybrid + bge (best local)** | **llama3:8b** | **BM25+dense→RRF→bge→4** | **0.855** | **0.909** | 0.953 | **0.689** | ~23 s |
| hybrid + MiniLM | llama3:8b | BM25+dense→RRF→MiniLM→4 | 0.867 | 0.920 | 0.950 | 0.679 | 19.2 s |
| hybrid + bge, deepseek | deepseek-r1:8b | BM25+dense→RRF→bge→4 | 0.853 | 0.904 | 0.985 | 0.717 | 62.4 s |
| **hybrid + bge, gpt-5.4 (best overall)** | **gpt-5.4-mini** | **BM25+dense→RRF→bge→4** | **0.865** | **0.914** | **0.984** | **0.774** | 7.9 s* |
| Advanced v3 (rerank-aware) | llama3:8b | hybrid + 4 fixes | — | — | — | n/c† | — |
| Advanced v3 (rerank-aware) | gpt-5.4-mini | hybrid + 4 fixes | 0.855 | 0.897 | 0.980 | 0.780 | 9.9 s* |

*API runs use hosted GPUs, which are not comparable hardware to the local runs.
†The local-generator Advanced v3 generated all 195 answers, but only 15 of 195 were scored before the evaluation window closed (judge-budget contention); not reported. See Section 8.3.2.

**Configuration notes.** A few of the runs above warrant a word of explanation.

- *chunk-512* re-indexes the corpus at a 512/64 chunk size and overlap (against the default 1000/100) to test whether smaller, tighter chunks raise retrieval precision. The effect was marginal (answer correctness 0.473 to 0.488), confirming that chunk granularity is not the binding constraint.
- *rerank + extract prompt* keeps the wide-retrieve-then-rerank retriever but switches the generation prompt to an extract-style instruction (pull the exact fact rather than write prose), to test whether prompt style lifts answer correctness. It did not (0.482), which located the reranking gain in retrieval rather than in the prompt.
- *dense + bge reranker* reranks a dense-only candidate set with the bge-base cross-encoder instead of the lighter MiniLM. On dense-only retrieval bge trailed MiniLM (0.578), but under hybrid retrieval it pulled ahead, which is why bge is the production reranker.

**Value-added ablation (answer correctness, no-retrieval baselines; see Section 7.3).**

| Run | Generator | Configuration | Answer Corr |
|---|---|---|---|
| Closed-book | llama3:8b | no retrieval | 0.202 |
| Closed-book | gpt-5.4-mini | no retrieval | 0.490 |
| Retrieval-only | — | top-4 hybrid chunks as answer | 0.827 |

---

## Appendix C — Evaluation Results

The complete evaluation results are provided as an accompanying spreadsheet, `revised_final_report_appendix_C.xlsx`. It has a Summary sheet with one row per run — generator, configuration, the five metrics, the human/synthetic answer-correctness split, and latency — and a per-run detail sheet giving the per-question scores behind each run. The underlying per-row CSVs are in the repository's `results/` directory, and the detailed engineering analysis is in `reports/architecture_evolution_analysis.md`, `reports/crag_improvement_analysis.md`, and `reports/experiments_index.md`.
