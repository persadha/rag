# Information retrieval using retrieval-augmented generation on PIRLS documents

### A comparative evaluation of retrieval and architecture choices for privacy-preserving, locally hosted question answering

**Prepared by:** Widianto Persadha, Heiko Sibberns, Mohammad S. Thariq, Bettina Wietzorek
**Prepared for:** IEA R&D Committee
**Revised:** June 2026 (a revision of the original 2026 final report)

***

## Executive summary

We built a proof of concept system that answers natural-language questions over IEA's PIRLS 2021 documentation. It uses retrieval-augmented generation (RAG) with open-weight language models that run entirely on local hardware. The prototype meets the requirements our staff survey identified for data privacy, transparent source attribution, and a web interface.

The main finding is that retrieval quality drives answer quality much more than pipeline complexity. We ran a controlled benchmark of 195 question and answer pairs scored by a single LLM judge held constant across every run. The largest gains came from improving how documents are retrieved rather than adding a more sophisticated agentic reasoning pipeline on top:

| Lever | Answer correctness | Cost |
|---|---|---|
| Baseline (Basic RAG, local llama3:8b) | 0.473 | N/A |
| + cross-encoder reranking | 0.625 (+32%) | ~$0 (CPU) |
| + hybrid (BM25 + dense) retrieval | 0.689 (+46% over baseline) | ~$0 (local) |
| + stronger generator (closed gpt-5.4-mini) | 0.774 (best overall) | cloud API |

Retrieval is justified here. One question for any such system is whether it outperforms a general-purpose model used without retrieval. Holding the generator fixed, retrieval lifts answer correctness from 0.202 with no retrieval to 0.689. This is a swing of 0.49. The best configuration reaches 0.774. On this corpus, the open 8B model is mostly unusable without retrieval. Even the strongest closed model we tested gains accuracy once retrieval is added.

The hypothesis that advanced pipelines perform better did not hold on local hardware. A more complex, multi step pipeline that decomposes the query, grades retrieved documents, and retrieves fresh context per subquestion did not improve answer quality over the simple pipeline. It also cost four to five times the latency. Part of the large gap reported in the original report was a measurement artifact in how retrieved context was logged. Once we corrected that, the two pipelines were close on quality. A purpose-built redesign (Advanced v3) reached parity with the simple pipeline but did not surpass it.

Our recommendation is to deploy Basic RAG with hybrid retrieval and reranking, using the local open llama3:8b generator as the private default. This configuration is private and practically free to run. It scores 0.689 on answer correctness and returns answers in about 20 seconds. The closed gpt-5.4-mini model can be offered as an optional tier for non-sensitive queries where accuracy is the absolute priority. We do not recommend deploying the agentic pipeline right now.

We addressed the extensions identified during this phase with new experiments. The changes relative to the original report are summarized in Appendix A. Items that need resources beyond this phase are carried forward as prioritized future work, primarily a calibrated human-expert evaluation and validation on internal documents.

***

## 1. Abstract

Knowledge workers rely on AI to find information faster, but general-purpose tools fall short in professional settings. Closed commercial models can return confident but incorrect answers. They cannot read an organization's internal documents, and they send sensitive data to external servers. This project develops a proof of concept information retrieval system that pairs retrieval-augmented generation with open-source LLMs running entirely on local hardware. This lets an organization apply AI to its own documents without giving up control of its data.

We evaluate the prototype on PIRLS 2021 documentation. This is text-heavy, domain-specific material representative of IEA's output. Outputs are scored with an LLM-as-a-judge framework (DeepEval) on four quality metrics: context precision, context recall, faithfulness, and answer correctness. We also use a judge-free gold-context similarity check and measure end to end response time. We compare a Basic (single pass) RAG pipeline against an Advanced (multi step, agentic) pipeline and search systematically for the variables that improve answer quality.

The results show that retrieval quality matters more than pipeline complexity. Cross-encoder reranking improves answer correctness by 32% at near-zero cost. Hybrid (sparse plus dense) retrieval adds a further 10%. A value-added ablation shows that retrieval accounts for the bulk of the system's usefulness, as answer correctness rises from 0.202 without retrieval to 0.689 with it. The advanced agentic pipeline did not outperform the simple one on local models under 10B parameters, and it cost four to five times the latency. We recommend a Basic RAG configuration with hybrid retrieval for local deployment, alongside an optional closed model tier for non-sensitive queries.

***

## 2. Introduction

### 2.1 Problem statement

IEA's large-scale studies generate extensive documentation like study frameworks, international reports, methodological guidelines, technical reports, and questionnaires. These materials hold valuable information about educational trends and the methods used to produce them. Their volume and complexity make it hard for researchers to find specific facts quickly. Keyword search is slow and imprecise. Much of the knowledge sits in unstructured PDFs that resist conventional querying.

Large language models offer fluent, context-aware question answering (Brown et al., 2020), but they carry known limitations in a professional setting. They can hallucinate plausible but wrong answers. They cannot answer questions about content created after their training cut-off. Most consequentially for IEA, commercial LLMs run on cloud infrastructure. This raises data privacy and security concerns for sensitive internal material. We need a system that combines the fluency of modern LLMs with strict local data governance. This project develops a proof of concept that runs entirely on local infrastructure and grounds every answer in retrieved source passages. This ensures users can check each answer against the documents it came from.

### 2.2 Scope

We use the 2021 PIRLS dataset for development and evaluation. We chose PIRLS for the complexity and diversity of its content, spanning statistical methodology, dense policy and assessment frameworks, and country-level encyclopedic information. This makes it representative of the high-value, unstructured documentation IEA produces.

The phase reported here focuses on extracting and retrieving high-density information from PDF documents. Multi modal and structured sources such as SQL databases, spreadsheets, and HTML are out of scope, along with high-fidelity table extraction. Section 7.5 explains why we deferred tables in particular. The results serve as a benchmark in their own right and as the basis for a future production deployment.

### 2.3 Motivations and research questions

The original report stated the project's significance as a set of forward claims about what a locally hosted RAG system would deliver. We now frame those claims as motivations drawn from the literature, alongside the questions this study tests. A locally hosted RAG system could benefit IEA in three ways. We treat each as a question to answer with evidence.

The first is retrieval efficiency. RAG can let researchers ask natural-language questions and receive concise, source-grounded answers instead of manually searching large document collections. The research question is whether retrieval materially improves answer quality over an LLM used alone. We test this in Section 7.3.

The second is reliability through grounding. Grounding answers in retrieved passages is expected to reduce hallucination and improve verifiability (Lewis et al., 2020; Gao et al., 2024). The research question is how faithful and how correct the grounded answers actually are, and where they fail. We examine this in Sections 7.2 and 7.4.

The third is privacy and control. Local deployment keeps all processing on-premises, which supports compliance and internal policy. The research question is whether an open, locally deployable model can reach acceptable quality without resorting to a cloud service. We test this in Section 8.2.3.

Beyond these, the project lays groundwork for later AI applications at IEA, such as report summarization, research assistance, and AI-assisted data cleaning. Those uses remain prospective and are not evaluated here.

### 2.4 Contributions

This report makes five contributions.

1. The design and implementation of two locally hosted RAG pipelines built on a common open source stack of LangChain, LangGraph, Chroma, and Ollama. One is a Basic single pass system, and the other is an Advanced multi step variant.
2. A PIRLS 2021 evaluation set that combines human-authored and LLM-generated question and answer pairs, complete with explicit provenance tagging and a characterization of its cognitive coverage (Section 5.4).
3. A controlled evaluation across four LLM-graded metrics and a judge-free retrieval check, reporting latency on workstation-class laptop hardware.
4. A systematic improvement study that isolates the variables driving answer quality (reranking, hybrid retrieval, generator choice, and chunking), along with a value-added ablation that quantifies what retrieval and generation each contribute.
5. A stage-level error analysis that explains why the advanced pipeline underperformed, followed by a production recommendation grounded in the results.

### 2.5 Related work

RAG couples a parametric language model with a nonparametric document store to improve factuality on knowledge-intensive tasks (Lewis et al., 2020). Surveys of the area point out that RAG performance depends heavily on retrieval quality and document preprocessing, not just the generator itself (Gao et al., 2024). Our results echo this.

People often justify agentic RAG by arguing that complex questions benefit from being broken into subproblems (Khot et al., 2022) and from iterative self-reflection (Shinn et al., 2023). Corrective RAG (CRAG) adds a validation step after retrieval that checks whether the retrieved documents are actually relevant before they are used (Yan et al., 2024). GraphRAG models a corpus as connected units to support multi hop retrieval and global summarization (Edge et al., 2024). Selective methods such as Self-RAG decide for each query whether retrieval is needed at all (Asai et al., 2023). We return to this idea in Section 8.3.2. Our Advanced pipeline draws on decomposition and post-retrieval validation. Section 7.4 reports what happened when we asked a small local model to run that machinery.

On the retrieval side, the retrieve-then-rerank pattern is now standard practice. A wide first-stage retrieval is followed by a cross encoder that rescores the candidates and keeps the best few (Nogueira & Cho, 2019). Hybrid retrieval fuses sparse BM25 scores (Robertson & Zaragoza, 2009) with dense vector similarity (Karpukhin et al., 2020), often using reciprocal rank fusion (Cormack et al., 2009). This raises recall on entity-heavy queries that dense embeddings alone tend to miss. Both techniques turn out to be the most effective variables in our study.

Evaluation has moved toward LLM-as-a-judge scoring and component-wise metrics like context precision, context recall, and faithfulness. These make it possible to diagnose retrieval and generation failures separately and at scale (Zheng et al., 2023; Es et al., 2023). This project uses DeepEval's reference-based contextual metrics and its answer-correctness scoring (Confident AI, 2026).

***

## 3. Background

This section explains the techniques the rest of the report builds on. Readers already familiar with retrieval-augmented generation can skip ahead. The evaluation metrics are defined later alongside the experimental setup in Section 5.3.

### 3.1 Retrieval-augmented generation

A language model left to itself answers from what it absorbed during training. For a corpus like PIRLS, that is a poor arrangement. The model may never have seen the documents. Even where it has, it cannot point to the page a fact came from, so a confident-sounding answer cannot be checked. Retrieval-augmented generation (Lewis et al., 2020) takes a different route. Instead of asking the model to recall, it hands the model the relevant source passages at the moment of answering. The model works from text it can actually see, and we can trace every claim back to a document.

It works in two phases. The first phase happens once before any question is asked. Every document is split into chunks of a few hundred words. Each chunk is passed through an embedding model that turns it into a vector. These are long lists of numbers positioned so that passages about similar things land near each other in a high-dimensional space. These vectors are stored in an index. The second phase happens at query time. The embedding model embeds the question, and the system compares the question's vector against every chunk's vector using cosine similarity, the cosine of the angle between them:

$\cos(q, d) = \frac{q \cdot d}{|q| \cdot |d|}$

This is 1 when two vectors point the same way and 0 when they are unrelated. A higher score means a chunk is more likely to be about the same thing as the question. The system keeps the $k$ highest-scoring chunks, places them in the prompt, and instructs the model to answer from that context alone. It also instructs the model to say plainly when the answer is not present.

In this bare form, the one setting that matters most is $k$, the number of chunks retrieved. If you set it too low, the passage that holds the answer might fall just past the cutoff, leaving the model to guess. Set it too high, and the prompt fills with loosely related text that buries the useful sentence. In practice, this nudges a small model toward a fluent but wrong answer. We use $k = 4$. This whole single pass arrangement (embed the query, take the top four chunks, generate once) is what we call Basic RAG. It is the baseline against which every later refinement is measured.

### 3.2 Corrective RAG

Basic RAG takes the retriever at its word. Whatever the top chunks happen to be, they go into the prompt. If some of them are off-topic, the model simply has to cope with the noise. Corrective RAG (CRAG; Yan et al., 2024) tries to head that off by inserting a checking step between retrieval and generation. Each retrieved chunk is judged for relevance to the question. The ones that fail are dropped. If too little is left, the system can retrieve again or fall back to what it started with. The intent is to tidy the context before the generator ever reads it.

The catch is that this step is only as good as the judge behind it. A reliable judge strips out the genuine distractors and leaves the answer-bearing chunks untouched. A poor judge does real damage because the step can only ever remove chunks. It cannot recover one it wrongly discarded. When it throws away a chunk that held the answer, recall falls and nothing downstream can undo it. Section 7.4 shows that the lightweight grader wired into our first agentic pipeline behaves like the poor judge often enough to make the correction step a net loss on this corpus.

### 3.3 Hybrid retrieval

Dense retrieval, the cosine-similarity search just described, is good at meaning. Ask about "young readers' attitudes toward books" and it surfaces passages on reading motivation even when those exact words never appear. That same strength is its weakness. Because it matches on meaning rather than surface form, it can miss the literal token a question turns on. Country names, study acronyms, and identifiers such as "PIRLS" or a specific benchmark label do not always sit close together in embedding space. A question that hinges on one of them can come back with plausible neighbours that are nonetheless wrong.

The long-standing remedy is lexical search. BM25 (Robertson & Zaragoza, 2009) scores a chunk by how often the query's terms occur in it. It weights rare and discriminating words more heavily than common ones and discounts chunks that score only because they are long:

$BM25(q, d) = \sum_{t \in q} IDF(t) \cdot \frac{f(t,d) \cdot (k_1 + 1)}{f(t,d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{avgdl})}$

Here $f(t,d)$ is how often term $t$ appears in chunk $d$. $IDF(t)$ is higher for rarer terms. $|d|$ is the chunk's length, $avgdl$ is the average chunk length, and $k_1$ and $b$ are small tuning constants. The practical effect is simple. BM25 reliably finds the chunk that literally contains "Singapore" or "plausible values," which is exactly the case dense search can fumble.

Hybrid retrieval runs both searches and merges their results. A chunk earns a place if either method ranks it highly. We merge with reciprocal rank fusion (RRF; Cormack et al., 2009). This deliberately ignores the raw scores since a cosine similarity and a BM25 score are not on the same scale. It uses only where each chunk landed in each list:

$RRF(d) = \sum_{r} \frac{1}{c + rank_r(d)}$

Here $rank_r(d)$ is the chunk's position in retriever $r$'s ranking and $c$ is a small constant that keeps the very top ranks from dominating. A chunk near the top of either the dense or the lexical list is lifted in the combined order. Adding BM25 to the dense retriever this way is what recovers the entity-heavy questions that dense similarity alone tends to miss (Karpukhin et al., 2020).

### 3.4 Cross-encoder reranking

The retrievers above are fast because they look at the question and each chunk separately. The embedding model encodes the query once and every chunk once ahead of time. Retrieval is then just a nearest-neighbour lookup over millions of precomputed vectors. This arrangement, a bi-encoder, scales to a large index effortlessly, but it pays for that speed with a blind spot. The query and the chunk are never read together, so the similarity score is only an approximation of how relevant the chunk really is.

A cross encoder removes that approximation. Rather than encoding the two pieces apart, it feeds the question and a single candidate chunk into the model joined together. It lets every word of the question attend to every word of the chunk, emitting one relevance score grounded in both at once (Nogueira & Cho, 2019). It is markedly more accurate, but far too slow to run across the whole index for every query.

The way to get the speed of the first and the judgement of the second is to use them in sequence. The fast retriever casts a wide net, here the top 20 candidates. The slow but accurate cross encoder then rescores only those 20 and keeps the best 4 for the generator. The expensive model never sees more than a handful of chunks, so the added cost is small. The chunk that genuinely answers the question is pulled up into the few the generator reads even when it began outside the naive top 4. Section 8.2 measures exactly this effect and shows it is the single largest source of the quality gains reported here.

***

## 4. Methodology

Our approach builds on the RAG framework (Lewis et al., 2020). Coupling the parametric memory of an LLM with a nonparametric retriever lets the system answer from an external index that can be updated without retraining the model. Standard RAG is vulnerable to retrieval noise. This motivated an Advanced variant that adds query decomposition (Khot et al., 2022), step by step reasoning (Wei et al., 2022), and post-retrieval validation. As the results show, the value of that extra machinery depends heavily on the capacity of the local model running it.

### 4.1 Data preparation

The knowledge base was built from publicly available PIRLS 2021 PDFs. These include the International Results in Reading, the Encyclopedia of education policy and curriculum, the Methods and Procedures technical report, the User Guide for the International Database, Countries' Reading Achievement, and the school, teacher, and student questionnaires. Each PDF is loaded with PyPDFLoader, split into chunks with a recursive character splitter, embedded with a sentence transformer model (Reimers & Gurevych, 2019), and stored in a Chroma vector index. Chroma was chosen because it runs in-process with no external service. This keeps the whole pipeline self-contained and offline.

Measured directly from the live index, the knowledge base has the following scale.

**Table 1. Knowledge base scale (measured from the live index).**

| Property | Value |
|---|---|
| Source documents | 108 |
| Indexed chunks | 6,771 |
| Total tokens (cl100k_base) | ~1.5 million |
| Mean chunk length | 122 words / 812 characters / 221 tokens |
| On-disk footprint | 70.9 MB |
| Embedding model | `sentence-transformers/all-mpnet-base-v2` (768-dim) |

The chunking parameters and the choice of embedding model are discussed in Section 8.2.4, where we show that retrieval quality is the binding constraint rather than chunk granularity.

### 4.2 User survey

Before building anything, we ran a requirements survey across all IEA staff in the Hamburg and Amsterdam offices to make sure the prototype answered a real need. Forty staff responded, completing Phase 1 of the program. They came from across the specialized units, with the International Study Unit being the largest group at 15 respondents. They were joined by colleagues from Sampling, Data Management, Software, and Research and Analysis. Thirty-three responded in English and seven in German.

The functional requirements were clear. Navigating complex study information was the top need. Technical documentation and specific study results from PIRLS were each cited 28 times. Staff also asked for cross-region comparisons and longitudinal analysis. Most respondents wanted a "medium" level of detail in answers, though a sizeable segment wanted either exhaustive explanations or direct access to the underlying data and tables.

The non-functional requirements shaped the design just as strongly. Speed is a clear requirement, with 80% of respondents expecting an answer within 30 seconds and many preferring a few seconds. Accuracy ranks even higher, with over 67% rating precise information as "extremely important." In the open-ended responses, staff repeatedly stressed the need for source attribution so they can verify an answer manually. Those two themes, accuracy and verifiability, shaped the system's design. It pairs a retriever that grounds every answer with an interface that shows the source chunks behind it.

### 4.3 System architecture

Both pipelines run on a local open source stack. LangChain handles document loading, splitting, and prompting. LangGraph manages the stateful multi step workflow used by the Advanced pipeline. Ollama serves the local models so a model can be swapped without code changes. Chroma stores the vectors.

#### 4.3.1 Basic architecture (Basic RAG)

![Figure 1. Basic RAG pipeline.](figures/orig/orig_fig1_basic_arch.jpg)

*Figure 1. The Basic RAG pipeline. Documents are chunked, embedded, and stored. A query is embedded, the top chunks are retrieved, and the model answers using only the retrieved context.*

The Basic pipeline is a linear retrieve-then-generate path. PDFs are ingested and chunked. The chunks are embedded and stored in Chroma. At query time, the system retrieves the top chunks by similarity. Those chunks are injected into a prompt that instructs the model to answer using only the provided context and to say so when the answer is not present. The model then generates the answer. The original implementation retrieved the top chunks by dense similarity alone. Section 8.2 replaces that with a wider retrieval followed by reranking and hybrid fusion. This is where most of our gains come from.

#### 4.3.2 Advanced architecture (Advanced RAG)

![Figure 2. Intended Advanced RAG workflow.](figures/orig/orig_fig2_advanced_arch.jpg)

*Figure 2. The intended Advanced RAG workflow: decompose the query, plan sub-steps, retrieve, answer each subquestion, and synthesize a final answer.*

The Advanced pipeline runs as a stateful graph rather than a linear path. Its intended design decomposes a complex query into subquestions, validates the retrieved documents, answers each subquestion, and synthesizes a final answer from those parts.

We need to make an important correction between what the original report described and what was actually evaluated. The original report described the Advanced pipeline as already performing per-subquestion reretrieval and reranking. The version that was actually measured, which we now call Advanced v1, did neither. Its subquestions reused the original query's retrieved documents, and no reranker was wired into the graph. Its only active post-retrieval step was a binary relevance grader run by a small `gemma3:1b` model. The intended reretrieval and reranking were built later in the Advanced v2 and v3 iterations described in Section 8.3. We describe each version by what it actually did and attribute results accordingly.

**Advanced v1, as built.** Figure 3 shows the pipeline that was actually evaluated. It runs as a five-stage graph. First, it retrieves the top four chunks for the original query. Second, a small `gemma3:1b` model grades each chunk as relevant or not and drops those it rejects. If it rejects all four, the pipeline keeps the original four rather than proceeding with no context. Third, it decomposes the query into two or three subquestions but answers each from the same original chunks, with no fresh retrieval. Fourth, it answers the subquestions and synthesizes a final answer capped at 100 words. Fifth, the same `gemma3:1b` model grades the answer for usefulness. An answer judged unhelpful is regenerated up to two times. The two steps that distinguish v1 from Basic RAG (the binary grader and the decomposition) are precisely the ones Section 7.4 finds to be net-negative on this corpus.

![Figure 3. Advanced v1 (CRAG) as built.](figures/fig_adv_v1.svg)

*Figure 3. The Advanced v1 (CRAG) pipeline as evaluated. The gemma3:1b grader can only drop chunks, the subquestions reuse the original query's documents, and there is no reranker and no per-subquestion retrieval.*

Three Advanced versions appear in this report.

**Table 2. The three Advanced pipeline versions.**

| Version | Additional Features | Status |
|---|---|---|
| Advanced v1 | `gemma3:1b` document grader; decomposition that reuses the original documents; synthesis capped at 100 words | Evaluated (this is the original "advanced" system) |
| Advanced v2 | per-subquestion reretrieval plus chunk deduplication; no word cap | Evaluated (Section 8.3) |
| Advanced v3 | reranker-score grader; reranks the subquestion union against the original query; adaptive decomposition; synthesizes from reranked context | Evaluated on the improved retriever (Section 8.3.2); matches Basic (0.780 vs 0.774), does not surpass it |

Versions v2 and v3, and the reasoning behind each change, are described in detail in Section 8.3.

***

## 5. Experimental setup

### 5.1 Hardware

The original prototype was engineered and tested on a mobile workstation: an Alienware M18 with an Intel Core i9-13980HX, an NVIDIA RTX 4090 Laptop GPU (16 GB), and 64 GB of RAM. We deliberately avoided cloud GPUs to keep the system fully offline and to stress-test it under realistic local constraints.

The reevaluation reported here was carried out on a different machine because access to the RTX 4090 laptop was limited during this period. That machine has an Intel Core Ultra 9 (16 cores), 32 GB of RAM, and an Intel Arc Pro 140T GPU (16 GB). Both are workstation-class, but they are not equivalent. The Arc-based laptop has roughly half the RAM and a different GPU and driver stack.

Because the two evaluation rounds ran on different hardware, the response-time figures in this revision are not directly comparable to the latency reported in the original report. They are still valid for relative comparison within the reevaluation runs, such as Basic versus Advanced and local versus API. The quality metrics (context precision and recall, faithfulness, and answer correctness) are hardware-independent and are unaffected. The deployment implications of running on a single machine are discussed in Section 9.2.

### 5.2 Models

The choice of models follows from the privacy goal. Since the aim is to keep data on the organization's own hardware, the generator has to be an open-weight model that can run locally. A closed cloud model would send every query and every retrieved passage to an external server. That is the situation this project exists to avoid. That ruled the commercial APIs out as the default and the open-weight families in.

Model size was the next constraint. Early tests with large models such as Llama-2 70B were unusable on local hardware, with single queries taking minutes. The question "What are the major advances in PIRLS 2021?" took 139.46 seconds to answer. We therefore standardized on open-weight models under 10B parameters, which keep response times usable on a single GPU. Llama 3 (8B) is the primary generator, chosen for its strong general quality at that size. For the controlled three-architecture comparison in Section 7, the generator is held fixed at llama3:8b so that only the pipeline varies. We then probe generator strength at the extremes using DeepSeek-R1 (8B) and gpt-5.4-mini in Section 8.2.3. The closed gpt-5.4-mini is included not as a deployment candidate but to measure how much accuracy the privacy constraint costs.

Two of the agentic pipelines use a second, smaller model as an internal grader that scores each retrieved chunk for relevance. We use gemma3:1b for that role deliberately. The grader runs once per chunk and would dominate latency if it were large, so a 1B model is the natural fit for a step meant to be a cheap filter. Section 7.4 shows that the economy proves a poor trade on this corpus, but the reasoning behind the choice is the conventional one.

All quality metrics are scored by a single LLM judge, `gpt-oss-120b`, served through an OpenAI-compatible API. It is an open-weight model, so the evaluation is reproducible by anyone. It is much larger than any system under test, so it grades their outputs from a position of greater capability. And it is held fixed across every run, so a difference in score reflects a difference in the systems rather than drift in the judge.

### 5.3 Metrics

We report four LLM-graded metrics, each probing a different stage of the pipeline, plus one judge-free retrieval check. All scores lie between 0 and 1, and higher is better. The definitions follow DeepEval (Confident AI, 2026).

Context precision measures the signal-to-noise ratio of the retrieved chunks. Are the relevant chunks ranked above the irrelevant ones? It is the rank-weighted mean of precision-at-k:

$Context Precision = \frac{\sum_{k} [ P(k) \times rel(k) ]}{\text{number of relevant chunks}}$

Here $rel(k)$ is 1 when the chunk at rank $k$ is relevant and $P(k)$ is the precision over the top $k$. If relevant chunks sit at ranks 1 and 3 with an irrelevant chunk at rank 2, the score falls below 1.0 because an irrelevant chunk outranks a relevant one.

Context recall measures whether retrieval captured all the facts the reference answer needs. An LLM breaks the reference answer into individual statements and checks each against the retrieved context:

$Context Recall = \frac{\text{reference statements supported by the context}}{\text{total reference statements}}$

If the reference makes two claims and only one appears in the retrieved chunks, recall is 0.5.

Faithfulness is a hallucination detector. It checks that the answer's claims are grounded in the retrieved context. An LLM extracts the atomic claims in the answer and verifies each against the context:

$Faithfulness = \frac{\text{answer claims supported by the context}}{\text{total answer claims}}$

Answer correctness compares the generated answer with the reference answer for factual and semantic agreement. It is scored with G-Eval (Liu et al., 2023), in which the judge reasons step by step against a fixed rubric and returns a probability-weighted score. "Singapore scored 587" against a reference of "Singapore: 587 points" scores near 1.0 despite the different phrasing.

The fifth check is gold-context similarity. This is the maximum cosine similarity between the gold reference passage and any retrieved chunk. It is deterministic and needs no judge. This makes it a cheap, repeatable way to ask whether retrieval surfaced the right passage at all, independent of how the judge scored the final answer.

We chose DeepEval over RAGAS after RAGAS proved slow and unreliable on a local judge, as Section 8.1 recounts.

### 5.4 Evaluation dataset construction

The evaluation set is the same 195 question and answer pairs the original report scored, now annotated with explicit human and synthetic provenance. Of the 195 pairs, 123 are human-generated, authored by a domain expert, with each answer traceable to the source documents. The remaining 72 are synthetic, generated with a large model from passages in the corpus to broaden coverage. Because synthetic items may not reflect real user intent, we report results split by provenance throughout Sections 7.2 and 7.3.

#### 5.4.1 Dataset composition: Provenance and question coverage

To characterize what the evaluation set actually tests, we classified all 195 questions by cognitive level and subtype using the `gpt-oss-120b` model.

![Figure 4. Eval-set question types.](figures/fig8_question_coverage.svg)

*Figure 4. Distribution of the 195 evaluation questions by subtype, colored by whether they are fact-retrieval or reasoning questions.*

**Table 3. Question coverage by cognitive level and provenance.**

| Cognitive level | Overall | Human (n=123) | Synthetic (n=72) |
|---|---|---|---|
| Fact-retrieval | 68.2% (133) | 77.2% | 52.8% |
| Reasoning (why/how/analysis) | 31.8% (62) | 22.8% | 47.2% |

The set is dominated by fact-retrieval questions at about 68%. The deep methodological questions are undersampled, with sampling at 4.6%, plausible values and scaling at 3.1%, and weighting and variance at 1.5%. We carry this forward explicitly as a limitation in Section 9 and recommend deliberately oversampling reasoning-heavy methodological questions in any follow-up evaluation set.

***

## 6. From the original report to the reevaluation

The original report and this revision are two evaluations of the same family of systems, carried out a few months apart with different evaluation machinery. This section explains how the experiment program unfolded, what the original report measured, and why the numbers reported here differ from the ones it published.

### 6.1 Experimental progression

We first corrected the problems in the original evaluation and ran the Basic pipeline to establish a baseline we could trust. We then applied the same corrections to the Advanced pipeline and reevaluated it, first as Advanced v1 and then as Advanced v2. Across that corrected comparison, the Basic pipeline still came out ahead (Section 7.1). We concentrated the improvement effort there rather than on the agentic machinery.

That decision is why the middle of the experiment matrix varies only the Basic pipeline. The sequence of retrieval and generator levers (reranking, hybrid retrieval, and a stronger generator up to gpt-5.4-mini) was explored on Basic alone. Only once those variables had been characterized did we fold the best of them back into the agentic pipeline. That is the last pair of runs, Advanced v3, and its purpose was narrow: to test whether a strong retrieval stack could be pushed further by the decomposition machinery. It could not. Advanced v3 only drew level with Basic. The progression runs baseline, corrected Advanced (v1, v2), a Basic-only improvement search, and finally Advanced v3. Appendix B reads in that order.

### 6.2 The original results, and what changed

The original report evaluated the Basic and Advanced pipelines across three local generators. Its headline numbers are reproduced in Table 4.

**Table 4. Results as reported in the original report (averaged over the same 195 items; higher is better; latency on the original RTX-4090 hardware, in seconds).**

| Pipeline | Generator | Ctx Precision | Ctx Recall | Faithfulness | Answer Correctness | Latency (s) |
|---|---|---|---|---|---|---|
| Basic | Llama 3 8B | 0.62 | 0.54 | 0.98 | 0.41 | 7.4 |
| Basic | Gemma 3 4B | 0.64 | 0.56 | 0.97 | 0.42 | 5.9 |
| Basic | DeepSeek-R1 8B | 0.63 | 0.54 | 0.97 | 0.44 | 19.0 |
| Advanced | Llama 3 8B | 0.29 | 0.31 | 0.97 | 0.23 | 22.9 |
| Advanced | Gemma 3 4B | 0.31 | 0.29 | 0.96 | 0.24 | 15.4 |
| Advanced | DeepSeek-R1 8B | 0.33 | 0.30 | 0.98 | 0.28 | 82.6 |

The original report already found Basic ahead of Advanced on every model. The headline direction was similar. But the absolute numbers differ sharply, most of all for the Advanced pipeline, whose context precision the original report put at 0.29 against the 0.690 we now measure for the same pipeline on the same questions (Table 5).

That gap is present because of the evaluation being corrected. The Advanced pipeline's retrieved context had been logged as one concatenated blob rather than a list of separate chunks. A ranking metric like context precision collapses toward zero when it is handed a single blob. Section 8.1 explains the artifact in full. Rescoring the same retrieval as a proper chunk list lifts Advanced precision and recall back into the same band as Basic.

The Basic pipeline was never touched by that logging bug, yet its numbers still moved. Context recall from 0.54 to 0.80, precision from 0.62 to 0.74. Those gains come from fixing the prototype and hardening the evaluation. The reevaluation repaired a retrieval defect, set the retrieval depth to $k = 4$, logged the retrieved context as a proper chunk list, and held a single judge constant across every run (Section 8.1). The result is a Basic baseline whose retrieval quality is finally measured cleanly enough to improve deliberately.

One important thing to note that the absolute values in Table 4 are not directly comparable to the reevaluation numbers elsewhere in this report since the two evaluations used different evaluation setup. What is comparable is the direction of the result: Basic ahead of Advanced in both evaluations.

***

## 7. Results and discussion

### 7.1 Comparative performance

The controlled comparison holds the dataset, judge, embeddings, retrieval, and generator (llama3:8b) constant, and varies only the architecture.

![Figure 5. Basic vs Advanced comparison.](figures/fig6_r3_comparison.svg)

*Figure 5. The final-evaluation comparison of Basic, Advanced v1, and Advanced v2 across four quality metrics (llama3:8b, n=195).*

**Table 5. Basic vs Advanced pipelines (llama3:8b, n=195, `gpt-oss-120b` judge).** Higher is better; latency is lower-is-better.

| Architecture | Ctx Precision | Ctx Recall | Faithfulness | Answer Correctness | Gold-Ctx Sim | Latency (s) |
|---|---|---|---|---|---|---|
| Basic | 0.741 | 0.796 | 0.939 | 0.473 | 0.741 | 19.5 |
| Advanced v1 | 0.690 | 0.671 | 0.948 | 0.474 | 0.717 | 74.1 |
| Advanced v2 | 0.449 | 0.650 | 0.910 | 0.334 | 0.712 | 101.8 |

The Basic pipeline performs best or ties on four of the five quality metrics. Advanced v1 matches Basic on answer correctness (0.474 against 0.473) but loses recall. Advanced v2 regresses on both precision and answer correctness. Faithfulness stays high everywhere, between about 0.91 and 0.95. This tells us all three pipelines ground their answers well in whatever context they are handed. The differences between them are in how each pipeline selects and uses context, not in raw grounding. Latency rises sharply with complexity, reaching 3.8 times and 5.2 times the Basic pipeline's response time.

#### 7.1.1 The evaluation set and its selection

The original report assembled a 300-item pool and scored the 195 items that both pipelines could complete. This revision works from those same 195 items, so the architecture comparison runs on identical questions. We kept the original 300-to-195 selection for comparability with the original report and because the full 300-item pool turned out to have a train/eval leakage problem: it entirely contains the 195 evaluation items (Section 8.2.4).

### 7.2 Metric by metric analysis

On context precision and recall, the Basic pipeline retrieves focused, on-topic context, scoring 0.74 on precision and 0.80 on recall. Advanced v1 loses recall, dropping to 0.67, and Advanced v2 loses precision badly at 0.45. Section 7.4 attributes each of those losses to a specific stage of the pipeline.

Faithfulness is uniformly high. When these models are given context, they rarely contradict it. That argues against weak reasoning as the primary failure mode and against fine-tuning as the first variable to consider.

Answer correctness sits at roughly 0.47 for all three architectures on the base model. Section 7.4 explains the ceiling, and Section 8.2 shows how better retrieval pushes through it to 0.689 without changing the model at all.

On response time, the single pass Basic pipeline is far faster. The Advanced pipelines run several sequential model calls for grading, per-subquestion generation, and synthesis, so the latency compounds.

![Figure 6. Mean latency per question.](figures/fig3_latency.svg)

*Figure 6. Mean latency per question, in seconds, measured on the reevaluation laptop (Intel Arc Pro 140T).*

### 7.3 The value of retrieval: A value-added ablation

Two questions about whether the system earns its complexity are worth settling directly. First, does RAG actually outperform a general-purpose LLM used without retrieval? Second, what does the generator add over simply returning the retrieved chunks? A single ablation answers both, scoring answer correctness across five conditions on the same 195 questions with the same judge.

![Figure 7. Value-added ablation.](figures/fig7_value_added.svg)

*Figure 7. Answer correctness by condition (n=195).*

**Table 6. Value-added ablation (answer correctness).**

| Condition | What it is | Overall | Human | Synthetic |
|---|---|---|---|---|
| Closed-book (open model) | llama3:8b, no retrieval | 0.202 | 0.150 | 0.290 |
| Closed-book (closed model) | gpt-5.4-mini, no retrieval | 0.490 | 0.377 | 0.683 |
| Retrieval-only | top-4 hybrid chunks as the answer, no LLM | 0.827 | 0.852 | 0.785 |
| Full RAG (Basic, local) | llama3:8b + hybrid retrieval | 0.689 | 0.753 | 0.581 |
| Full RAG (best) | gpt-5.4-mini + hybrid retrieval | 0.774 | 0.787 | 0.751 |

Retrieval is beneficial for both the open and the closed generator. Holding the generator fixed, retrieval lifts answer correctness from 0.202 to 0.689 for the local open model. This is a gain of 0.49. For the strong closed model, it lifts correctness from 0.490 to 0.774, a gain of 0.28. Retrieval therefore adds substantial accuracy even to the closed model that scores best without it. On this corpus, the deployable open 8B model is largely unusable without retrieval.

Taken at face value, retrieval-only at 0.827 beats full RAG. This would imply the generator has negative value, but that reading is wrong. It reflects a scoring artifact rather than evidence that raw chunks make a better answer. The retrieval-only answer is a roughly 4,000-character concatenation of chunks. Answer correctness rewards coverage of the reference facts without penalizing verbosity. A 4,000-character blob that happens to contain the gold facts scores very high, while the model's concise synthesis is penalized whenever it compresses a fact away. A 4,000-character dump is not a usable answer. Measuring the generator's contribution properly requires a concision or usability metric rather than fact-coverage alone.

### 7.4 Why answer correctness is bounded, and why the advanced pipelines underperformed

It is worth pinpointing which stage of the advanced pipeline actually fails: decomposition, retrieval, reranking, or synthesis. The data localizes each loss to a specific, measured mechanism. The problem lies in retrieval and orchestration rather than in the small model's reasoning since faithfulness is high everywhere.

The first loss is in Advanced v1, where the grader destroys recall. Its `gemma3:1b` document grader fires on 87 of the 195 questions. Where it drops chunks, recall collapses to 0.486, against 0.819 when it keeps all four. Precision falls as well. A weak 1B model removes answer-bearing chunks and keeps worse ones.

![Figure 8. Grader recall collapse.](figures/fig4_crag_recall_collapse.svg)

*Figure 8. Context recall in Advanced v1, split by whether the gemma3:1b grader dropped chunks. Where the grader fires, recall collapses from 0.819 to 0.486.*

The second loss is in Advanced v2, where per-subquestion retrieval destroys precision. Decomposing roughly 73% of the questions and retrieving fresh chunks for each subquestion widens the context from a mean of 4.0 chunks to 5.9. This material is relevant to the subquestions but off-topic for the original question against which precision is scored.

The third factor is a shared generation ceiling. Even with good context where precision is 0.7 or higher, llama3:8b produces a wrong answer on about 21% of rows. Better retrieval converts many of these rows by surfacing the exact gold chunk (Section 8.2), and a stronger generator lifts the rest (Section 8.2.3).

A large part of the precision and recall gap reported in the original report was a logging artifact. This revision corrects it. The practical upshot is to invest in retrieval and orchestration.

### 7.5 Challenges in tabular data extraction

The system could not reliably extract information from tables with nested layouts or graphical elements. Standard PDF text extraction flattens table structure and loses the row and column relationships. Treating tables as images for a vision-language model such as LLaVA (Liu et al., 2023) also performed poorly. Given the engineering effort that a proper fix would require, we deferred high-fidelity table extraction. Section 9.3 lists the specialized parsers we would evaluate next.

### 7.6 Deployable prototype (user interface)

![Figure 9. Production UI overview.](figures/orig/orig_fig9_ui.png)

*Figure 9. The IEA PIRLS Document Search interface.*

![Figure 10. Source-chunk inspector expanded.](figures/orig/orig_fig8_ui.png)

*Figure 10. The expandable source-chunk inspector showing the exact retrieved passages behind an answer.*

We built a web interface in Streamlit that lets staff query the PIRLS corpus and see both the generated answer and the source chunks behind it. The interface carries IEA branding and exposes the controls researchers require: a choice between a fully local, privacy-preserving model and an optional cloud API for higher accuracy. Every answer carries its provenance metadata and an expandable panel that shows the exact source passages. A simple query such as "What is PIRLS?" returns an answer in about 17 seconds on the local model.

***

## 8. Improving the system

### 8.1 Evaluation hardening: How the numbers became trustworthy

The benchmark went through three passes, and the comparison is only fair within the final one. The progression explains a discrepancy with the figures in the original report.

The first pass used RAGAS and was abandoned mid-run. The RAGAS harness cascaded through API-quota errors, local-judge timeouts, and library-version failures. It never produced complete scores for the advanced pipeline.

The second pass moved to DeepEval. It fixed the harness but exposed a measurement artifact. The Advanced pipeline's retrieved context had been logged as a single concatenated blob, while the Basic pipeline's was logged as a list of separate chunks. Because context precision and recall are ranking metrics computed over a list, a single blob degenerates them toward zero. This artificially inflated the precision and recall gap reported in the original report.

The third and final pass is the one this report relies on. With every system's context logged as a chunk list, a retrieval bug fixed, and the judge held constant, it is the first trustworthy three-way comparison. The corrected picture is that Basic and Advanced v1 are close on answer quality. The agentic machinery still adds latency without adding accuracy.

![Figure 11. Metric evolution across passes.](figures/fig1_metric_evolution.svg)

*Figure 11. Metric evolution across the evaluation passes and experiments for the Basic-family runs.*

### 8.2 Retrieval is the dominant lever

A retrieval-only diagnostic shows why retrieval matters most. The gold-bearing chunk is in the top 4 only 63% of the time, but it is in the top 20 fully 87% of the time. The fix is to retrieve wide and then rerank.

#### 8.2.1 Reranking (+32% answer correctness, ~$0)

Retrieving the top 20 candidates and rescoring them with a CPU cross encoder to keep the best 4 lifts answer correctness from 0.473 to 0.625. Precision rises from 0.741 to 0.837 and recall from 0.796 to 0.837. The cost is effectively zero since the reranker is a local CPU model.

![Figure 12. Rerank gains on the worst rows.](figures/fig5_rerank_worstcase.svg)

*Figure 12. The effect of reranking on the 40 worst-retrieval rows.*

#### 8.2.2 Hybrid retrieval (+10% more; best local configuration)

Adding sparse BM25 retrieval, fused with the dense retriever through reciprocal rank fusion and then reranked, lifts answer correctness from 0.625 to 0.689 and pushes recall to 0.909. Lexical BM25 catches exact terms that the dense encoder misses. This is the best open and local configuration, and it runs on the free local model. It beats a closed model using plain reranking, which scores 0.652.

#### 8.2.3 Stronger generator: Open vs closed

With the best retrieval held fixed, varying the generator is an additive second lever.

**Table 7. Generator comparison on the best retrieval configuration.**

| Generator | Answer correctness | Latency | Notes |
|---|---|---|---|
| llama3:8b (open, local) | 0.689 | ~20 s | recommended private default |
| deepseek-r1:8b (open, local) | 0.717 | ~62 s | small gain, 3x latency |
| gpt-5.4-mini (closed, cloud) | 0.774 | ~8 s | optional accuracy tier, non-sensitive only |

The closed gpt-5.4-mini is both the most accurate and the fastest, but it runs in the cloud and forfeits the privacy guarantee. The recommended private configuration remains the local open model.

#### 8.2.4 Chunking, embedding, and a data-leakage note

Reindexing at 512/64 changed answer correctness only marginally from 0.473 to 0.488. This confirms chunk granularity is not the binding constraint. Reranking and hybrid retrieval are.

On the embedding model, a correction is needed. Both the original prototype and this reevaluation use `sentence-transformers/all-mpnet-base-v2`. The earlier "embeddinggemma" label in the original report was an error.

Finally, the 300-item pool was found to fully contain the 195 evaluation items. This leaves only about 105 genuinely separate rows, which is too few to fine-tune on without leakage.

### 8.3 Revising the advanced architecture (v1 to v3)

#### 8.3.1 Why v1 and v2 underperformed

Section 7.4 established the diagnosis. Advanced v1's grader removes answer-bearing chunks and collapses recall. Advanced v2's per-subquestion retrieval injects off-topic chunks and collapses precision.

**Advanced v2, as built.** Figure 13 shows where v2 departs from v1. It keeps the `gemma3:1b` grader but retrieves a fresh set of chunks for each subquestion, deduplicates them, and merges them into a single union. The 100-word synthesis cap is also removed. On this corpus, per-subquestion retrieval widens the context from a mean of four chunks to almost six. Many of those extra chunks are off-topic for the original question, so precision falls from 0.74 to 0.45.

![Figure 13. Advanced v2 (CRAG++) as built.](figures/fig_adv_v2.svg)

*Figure 13. Advanced v2 adds per-subquestion retrieval and a deduplicated union on top of v1.*

#### 8.3.2 Advanced v3: A rerank-aware redesign

Advanced v3 applies four targeted fixes on top of the best retriever.

1. It replaces the binary grader with reranker-score thresholding. This no longer drops answer-bearing chunks, fixing the v1 recall collapse.
2. It reranks the per-subquestion union against the original question, fixing the v2 precision collapse.
3. It uses adaptive decomposition, breaking down questions only if they are genuinely multi hop and routing the single hop majority down a faster direct path.
4. It synthesizes from the reranked context rather than from subanswers alone.

![Figure 14. Advanced v3 (rerank-aware) as built.](figures/fig_adv_v3.svg)

*Figure 14. Advanced v3 replaces the binary grader with reranker-score thresholding, routes single hop questions straight to a direct answer, reranks the subquestion union, and synthesizes from the reranked context.*

**Table 8. Advanced v3 results (n=195, hybrid + bge reranker).**

| Generator | Ctx Precision | Ctx Recall | Faithfulness | Answer Correctness |
|---|---|---|---|---|
| llama3:8b (open, local) | N/A | N/A | N/A | not completed* |
| gpt-5.4-mini (closed) | 0.855 | 0.897 | 0.980 | 0.780 |

*The local-generator variant generated all 195 answers, but only 15 were scored before the evaluation window closed due to judge-budget contention. We report only the completed closed-generator run.

With the strong closed generator, Advanced v3 reaches an answer correctness of 0.780. That is statistically indistinguishable from the Basic pipeline on the same generator and retriever, which scores 0.774. The four fixes lifted the agentic pipeline from a net-negative architecture back up to parity with Basic. They did not surpass it. We continue to recommend Basic RAG for deployment.

#### 8.3.3 Summary of the improvement study

The best open and local configuration is Basic RAG with hybrid retrieval and reranking on llama3:8b, at an answer correctness of 0.689. The best overall configuration is the same retrieval with gpt-5.4-mini at 0.774. The full experiment matrix is in Appendix B.

***

## 9. Recommendations and future work

### 9.1 Recommended deployment

We recommend shipping Basic RAG with hybrid (BM25 plus dense) retrieval and cross-encoder reranking, using the local open llama3:8b generator as the private default. That configuration scores 0.689 on answer correctness, costs almost nothing to run, and returns an answer in about 20 seconds. We do not recommend deploying the agentic pipeline right now since it adds latency without adding accuracy. The closed gpt-5.4-mini model should be offered only as an opt-in tier for non-sensitive queries.

### 9.2 Deployment model

For organizational use, we recommend a local network server rather than per-user installs. A single on-premises GPU server hosts the model and the vector index and serves the web interface to staff browsers. This keeps data on-premises, centralizes index updates, and can host a larger local model than a laptop.

### 9.3 Future work

The most important next step is a calibrated human-expert evaluation. Subject-matter experts should grade a representative subset to calibrate the LLM judge against expert judgement.

Next is validation on internal, non-public documents. The evaluation corpus here is entirely public PIRLS documentation. We frame internal-document validation as a key future-work item, subject to data governance approval.

The evaluation set should incorporate more reasoning-heavy questions. A follow-up set should oversample questions about sampling, weighting, and plausible values.

Beyond these, three engineering directions would be valuable. A concision metric would allow the generator's contribution over raw chunks to be credited. High-fidelity table extraction, using specialized document parsers like Docling or PaddleOCR, would address the table problem. Hardware optimization through quantization and efficient serving would push the local tier below 10 seconds.

### 9.4 Limitations

The corpus is a single-domain, public one (PIRLS). PDF to text preprocessing introduces artifacts. The LLM judge is not yet calibrated against human experts. The question set is weighted toward fact-retrieval. And the latency is hardware-specific. The reevaluation ran on a less capable laptop than the original prototype, so response times are not comparable across the two reports and are used only for relative comparison.

***

## 10. Conclusion

A locally hosted RAG system over PIRLS documentation is feasible on consumer hardware. It meets the core requirements for privacy, source-attributed answers, and a usable interface. The central, well-supported finding is that retrieval quality matters more than pipeline complexity. Cross-encoder reranking and hybrid retrieval together lifted answer correctness from 0.47 to 0.69 on the free local model. A value-added ablation confirms that retrieval justifies its cost. The more elaborate agentic pipeline did not outperform the simple one. We therefore recommend a Basic RAG deployment using hybrid retrieval and a local open model, an optional closed model tier for non-sensitive queries, and a clear path toward a calibrated human evaluation and validation on internal documents.

***

## 11. References

Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2023). Self-RAG: Learning to retrieve, generate, and critique through self-reflection. arXiv:2310.11511.

Brown, T. B., et al. (2020). Language models are few-shot learners. NeurIPS. arXiv:2005.14165.

Confident AI. (2026). DeepEval: The LLM evaluation framework.

Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. SIGIR.

DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv:2501.12948.

Edge, D., et al. (2024). From local to global: A GraphRAG approach to query-focused summarization. arXiv:2404.16130.

Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). RAGAS: Automated evaluation of retrieval augmented generation. arXiv:2309.15217.

Faysse, M., Sibille, H., Wu, T., Omrani, B., Viaud, G., Hudelot, C., & Colombo, P. (2024). ColPali: Efficient document retrieval with vision language models. arXiv:2407.01449.

Gao, L., et al. (2024). Retrieval-augmented generation for large language models: A survey. arXiv:2312.10997.

Karpukhin, V., et al. (2020). Dense passage retrieval for open-domain question answering. EMNLP. arXiv:2004.04906.

Khot, T., et al. (2022). Decomposed prompting: A modular approach for solving complex tasks. arXiv:2210.02406.

Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. NeurIPS. arXiv:2005.11401.

Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). Visual instruction tuning. NeurIPS. arXiv:2304.08485.

Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., & Zhu, C. (2023). G-Eval: NLG evaluation using GPT-4 with better human alignment. EMNLP. arXiv:2303.16634.

Nogueira, R., & Cho, K. (2019). Passage re-ranking with BERT. arXiv:1901.04085.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. EMNLP. arXiv:1908.10084.

Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval.

Shinn, N., et al. (2023). Reflexion: Language agents with verbal reinforcement learning. arXiv:2303.11366.

Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. NeurIPS. arXiv:2201.11903.

Yan, S. Q., et al. (2024). Corrective retrieval augmented generation. arXiv:2401.15884.

Zheng, L., et al. (2023). Judging LLM-as-a-judge with MT-bench and Chatbot Arena. arXiv:2306.05685.

***

## Appendix A: Changes from the original report

This appendix summarizes how the present report differs from the original. Each change is classified as added evidence, an editorial revision, or an item deferred to future work.

| Change | Type | Section |
|---|---|---|
| Reframed the significance statements as motivations and explicit research questions | Editorial | 2.3 |
| Corrected the Basic and Advanced architecture descriptions to match what was actually evaluated | Editorial | 4.3.2 |
| Added knowledge base scale figures and a chunking-sensitivity analysis | Added evidence | 4.1, 8.2.4 |
| Corrected the embedding model description to `all-mpnet-base-v2` | Correction | 4.1, 8.2.4 |
| Documented the rationale for the generator, grader, and judge choices, and for DeepEval over RAGAS | Editorial | 5.2, 5.3 |
| Classified the evaluation questions by cognitive level and flagged the undersampling of reasoning questions | Added evidence, editorial | 5.4.1, 9.4 |
| Reported every metric split by human and synthetic provenance | Added evidence | 5.4.1, 7.2 |
| Clarified the 300-to-195 item selection and reused the same 195 items for comparability | Editorial | 7.1.1 |
| Added a with-versus-without-retrieval baseline | Added evidence | 7.3 |
| Added a retrieval-only baseline isolating what the generator contributes | Added evidence | 7.3 |
| Added an open versus closed generator comparison | Added evidence, editorial | 8.2.3, 7.3 |
| Added a stage-level attribution of where the advanced pipeline loses quality | Added evidence | 7.4 |
| Hardened the evaluation: chunk-list logging, fixed embedding and retrieval, and a single judge | Added evidence | 8.1 |
| Added a deployment model discussion | Editorial | 9.2 |
| Deferred calibrated human-expert evaluation, validation on internal documents, and full judge-sufficiency calibration | Future work | 9.3 |

***

## Appendix B: Full experiment matrix

All runs are n=195, judged by `gpt-oss-120b`.

| Run | Generator | Configuration | Ctx P | Ctx R | Faith | Answer Corr | Latency |
|---|---|---|---|---|---|---|---|
| Basic (baseline) | llama3:8b | k=4, 1000/100 | 0.741 | 0.796 | 0.939 | 0.473 | 19.5 s |
| Advanced v1 | llama3:8b | gemma3:1b grader | 0.690 | 0.671 | 0.948 | 0.474 | 74.1 s |
| Advanced v2 | llama3:8b | per-sub-q retrieve + dedup | 0.449 | 0.650 | 0.910 | 0.334 | 101.8 s |
| Basic + rerank | llama3:8b | k=20 to rerank to 4 | 0.837 | 0.837 | 0.940 | 0.625 | 21.9 s |
| chunk-512 | llama3:8b | k=4, 512/64 | 0.723 | 0.759 | 0.939 | 0.488 | 11.3 s |
| Basic, deepseek | deepseek-r1:8b | k=4, 1000/100 | 0.742 | 0.804 | 0.983 | 0.595 | 62.7 s |
| Basic + rerank, gpt-5.4 | gpt-5.4-mini | k=20 to rerank to 4 | 0.837 | 0.847 | 0.981 | 0.652 | 1.6 s* |
| rerank + extract prompt | llama3:8b | extract style | 0.825 | 0.864 | 0.974 | 0.482 | 19.4 s |
| dense + bge reranker | llama3:8b | k=20 to bge-base to 4 | 0.807 | 0.850 | 0.957 | 0.578 | ~22 s |
| hybrid + bge (best local) | llama3:8b | BM25+dense to RRF to bge to 4 | 0.855 | 0.909 | 0.953 | 0.689 | ~23 s |
| hybrid + MiniLM | llama3:8b | BM25+dense to RRF to MiniLM to 4 | 0.867 | 0.920 | 0.950 | 0.679 | 19.2 s |
| hybrid + bge, deepseek | deepseek-r1:8b | BM25+dense to RRF to bge to 4 | 0.853 | 0.904 | 0.985 | 0.717 | 62.4 s |
| hybrid + bge, gpt-5.4 (best overall) | gpt-5.4-mini | BM25+dense to RRF to bge to 4 | 0.865 | 0.914 | 0.984 | 0.774 | 7.9 s* |
| Advanced v3 (rerank-aware) | llama3:8b | hybrid + 4 fixes | N/A | N/A | N/A | n/c† | N/A |
| Advanced v3 (rerank-aware) | gpt-5.4-mini | hybrid + 4 fixes | 0.855 | 0.897 | 0.980 | 0.780 | 9.9 s* |

*API runs use hosted GPUs, which are not comparable hardware to the local runs.
†The local-generator Advanced v3 generated all 195 answers, but only 15 of 195 were scored before the evaluation window closed.

**Configuration notes.**

* *chunk-512* reindexes the corpus at a 512/64 chunk size and overlap to test whether smaller, tighter chunks raise retrieval precision. The effect was marginal.
* *rerank + extract prompt* keeps the wide retriever but switches the generation prompt to an extract-style instruction. It did not lift answer correctness.
* *dense + bge reranker* reranks a dense-only candidate set with the bge-base cross encoder instead of the lighter MiniLM. On dense-only retrieval bge trailed MiniLM, but under hybrid retrieval it pulled ahead.

**Value-added ablation**

| Run | Generator | Configuration | Answer Corr |
|---|---|---|---|
| Closed-book | llama3:8b | no retrieval | 0.202 |
| Closed-book | gpt-5.4-mini | no retrieval | 0.490 |
| Retrieval-only | N/A | top-4 hybrid chunks as answer | 0.827 |

***

## Appendix C: Evaluation results

The complete evaluation results are provided as an accompanying spreadsheet, `revised_final_report_appendix_C.xlsx`. It has a Summary sheet with one row per run and a per-run detail sheet giving the per-question scores behind each run.