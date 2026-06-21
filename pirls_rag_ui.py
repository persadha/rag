"""Streamlit UI for the PIRLS RAG system — experiment playground.

Run:  streamlit run pirls_rag_ui.py

Requires a persisted index (python scripts/build_index.py) and, for the local
generators/graders, an Ollama server (OLLAMA_BASE_URL, default localhost:11434).
The sidebar exposes every r3 / E-series lever live: architecture, generator
(local + API), grader model (CRAG/CRAG++), index/chunking, cross-encoder
reranking, and the extraction-prompt A/B. Every answer shows the retrieved
chunks with their vector distance to the question (retrieval inspection,
plans.md item 6). Bug history: reports/ui_review.md.
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

sys.path.append(str(Path(__file__).parent))
load_dotenv()  # API keys for the hosted generators/graders

from src.config.config import Config
from src.config.config_api import APIConfig
from src.utils.text import strip_reasoning

ARCHITECTURES = ("Standard", "CRAG", "CRAG++")

# label -> (kind, value). kind "local" builds a ChatOllama; "api" routes through
# APIConfig.get_generator. Both yield a `runnable | StrOutputParser()`.
GENERATORS = {
    "llama3:8b (local)": ("local", "llama3:8b"),
    "gemma3:4b (local)": ("local", "gemma3:4b"),
    "deepseek-r1:8b (local)": ("local", "deepseek-r1:8b"),
    "haiku (Anthropic)": ("api", "haiku"),
    "llama-groq (Groq)": ("api", "llama-groq"),
    "gemma-deepinfra (gemma-3-4b)": ("api", "gemma-deepinfra"),
    "openai-mini (gpt-5.4-mini)": ("api", "openai-mini"),
    "GLM-5.2 (DeepInfra)": ("deepinfra", APIConfig.GLM_DEEPINFRA_MODEL),
    "Kimi-K2.7-Code (DeepInfra)": ("deepinfra", APIConfig.KIMI_DEEPINFRA_MODEL),
    "NVIDIA-Nemotron-3-Ultra-550B-A55B (DeepInfra)": ("deepinfra", APIConfig.NEMOTRON_DEEPINFRA_MODEL),
    "DeepSeek-V4-Pro (DeepInfra)": ("deepinfra", APIConfig.DEEPSEEK_V4_DEEPINFRA_MODEL),
}

# Grader (slm) for CRAG/CRAG++ document + generation grading. Default gemma3:1b
# preserves the frozen CRAG design; any generator-class model can stand in.
GRADERS = {
    "gemma3:1b (local)": ("local", "gemma3:1b"),
    "llama3:8b (local)": ("local", "llama3:8b"),
    "gemma3:4b (local)": ("local", "gemma3:4b"),
    "haiku (Anthropic)": ("api", "haiku"),
    "llama-groq (Groq)": ("api", "llama-groq"),
    "gemma-deepinfra (gemma-3-4b)": ("api", "gemma-deepinfra"),
    "openai-mini (gpt-5.4-mini)": ("api", "openai-mini"),
    "GLM-5.2 (DeepInfra)": ("deepinfra", APIConfig.GLM_DEEPINFRA_MODEL),
    "Kimi-K2.7-Code (DeepInfra)": ("deepinfra", APIConfig.KIMI_DEEPINFRA_MODEL),
    "NVIDIA-Nemotron-3-Ultra-550B-A55B (DeepInfra)": ("deepinfra", APIConfig.NEMOTRON_DEEPINFRA_MODEL),
    "DeepSeek-V4-Pro (DeepInfra)": ("deepinfra", APIConfig.DEEPSEEK_V4_DEEPINFRA_MODEL),
}

# index label -> chroma directory (both built by scripts/build_index.py, same embeddings)
INDEXES = {
    "1000 / 100  (chroma_db)": "chroma_db",
    "512 / 64  (chroma_db_512)": "chroma_db_512",
}

RETRIEVER_K = 4
DEFAULT_CANDIDATES = 20

st.set_page_config(page_title="PIRLS RAG Search", page_icon="📚", layout="centered")


def build_llm(kind: str, value: str):
    """Return a string-output runnable for a local Ollama model or an API generator.

    Shared by the generator and grader selectors so there is one code path."""
    if kind == "local":
        from langchain_ollama import ChatOllama
        from langchain_core.output_parsers import StrOutputParser
        chat = ChatOllama(model=value, temperature=0, num_ctx=8192,
                          base_url=Config.OLLAMA_BASE_URL)
        return chat | StrOutputParser()
    if kind == "deepinfra":  # arbitrary DeepInfra model id
        return APIConfig.get_deepinfra_generator(value)
    return APIConfig.get_generator(value)  # named API generator; may raise if key missing


@st.cache_resource
def load_store(persist_dir: str):
    """Load a persisted Chroma index (built once by scripts/build_index.py)."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from src.vectorstore.vectorstore import VectorStore

    full_dir = str(Path(__file__).parent / persist_dir)
    embeddings = HuggingFaceEmbeddings(model_name=Config.DEFAULT_EMBEDDING_MODEL)
    store = VectorStore(embeddings, persist_directory=full_dir)
    store.load_vectorstore()
    return store


@st.cache_resource
def get_graph(architecture: str, gen_kind: str, gen_value: str,
              grader_kind: str, grader_value: str,
              persist_dir: str, rerank: bool, candidates: int):
    """Build the selected graph once per unique lever combination."""
    store = load_store(persist_dir)

    base = store.get_retriever(k=candidates if rerank else RETRIEVER_K)
    if rerank:
        from src.vectorstore.rerank import RerankRetriever
        retriever = RerankRetriever(base, top_k=RETRIEVER_K)
    else:
        retriever = base

    llm = build_llm(gen_kind, gen_value)

    if architecture == "Standard":
        from src.graph_builder.graph_builder import GraphBuilder
        builder = GraphBuilder(retriever=retriever, llm=llm)
    else:
        slm = build_llm(grader_kind, grader_value)  # CRAG/CRAG++ grader
        if architecture == "CRAG":
            from src.graph_builder.graph_builder_adv import GraphBuilder
        else:  # CRAG++
            from src.graph_builder.graph_builder_cragpp import GraphBuilder
        builder = GraphBuilder(retriever=retriever, llm=llm, slm=slm)
    builder.build()
    return builder


def extract(architecture: str, result: dict):
    """(answer, docs actually used) per architecture — keys differ by design."""
    if architecture == "Standard":
        return result.get("answer", ""), result.get("retrieved_docs", [])
    if architecture == "CRAG":
        return result.get("final_answer", ""), result.get("documents", [])
    return result.get("final_answer", ""), (result.get("final_contexts")
                                            or result.get("documents", []))


def main():
    st.title("PIRLS Document Search")
    st.markdown("Ask a question about the PIRLS 2021 corpus.")

    with st.sidebar:
        architecture = st.selectbox("Architecture", ARCHITECTURES,
                                    help="Standard: retrieve→generate. CRAG: graded docs + "
                                         "sub-questions over shared context. CRAG++: CRAG + "
                                         "per-sub-question retrieval, dedup, no word cap.")
        is_standard = architecture == "Standard"

        gen_label = st.selectbox("Generator model", list(GENERATORS),
                                 help="Local models need Ollama; API models need keys in .env.")
        grader_label = st.selectbox("Grader model (CRAG/CRAG++)", list(GRADERS),
                                    disabled=is_standard,
                                    help="Grades retrieved docs + generations. "
                                         "Standard does no grading, so this is ignored there.")
        index_label = st.selectbox("Index / chunking", list(INDEXES))

        rerank = st.checkbox("Cross-encoder reranking",
                             help="Retrieve a wide candidate set, then rerank to the top-4 "
                                  "with a cross-encoder. First run downloads ~80 MB.")
        candidates = st.slider("Rerank candidates", min_value=8, max_value=50,
                               value=DEFAULT_CANDIDATES, step=1, disabled=not rerank,
                               help="First-stage dense pool size before reranking to top-4.")
        extract_on = st.checkbox("Extraction prompt (Standard only)", disabled=not is_standard,
                                 help="E8 A/B: terse fact-extraction prompt. "
                                      "Only affects the Standard generator.")

        st.caption(f"Retriever: top-{RETRIEVER_K} chunks, "
                   f"{Config.DEFAULT_EMBEDDING_MODEL.split('/')[-1]} embeddings.")

    gen_kind, gen_value = GENERATORS[gen_label]
    grader_kind, grader_value = GRADERS[grader_label]
    persist_dir = INDEXES[index_label]

    if "history" not in st.session_state:
        st.session_state.history = []

    try:
        store = load_store(persist_dir)
        if store.vectorstore is None:
            st.error(f"No index found at ./{persist_dir} — build it first:  "
                     "`python scripts/build_index.py`")
            st.stop()
    except Exception as exc:
        st.error(f"Failed to load the vector store: {exc}")
        st.stop()

    with st.form("search_form"):
        question = st.text_input("Enter your question:",
                                 placeholder="e.g. How is reading assessed in Flanders?")
        submitted = st.form_submit_button("Search", type="primary")

    if submitted and question.strip():
        question = question.strip()

        # Extraction prompt is read at generation time by RAGNodes (Standard only).
        os.environ["GEN_PROMPT_STYLE"] = "extract" if (extract_on and is_standard) else "baseline"

        try:
            with st.spinner(f"Running {architecture} with {gen_label}…"):
                graph = get_graph(architecture, gen_kind, gen_value,
                                  grader_kind, grader_value,
                                  persist_dir, rerank, candidates)
                start = time.time()
                result = graph.run(question)
                elapsed = time.time() - start
        except RuntimeError as exc:  # missing API key, etc.
            st.error(str(exc))
            st.stop()
        except Exception as exc:
            st.error(f"Generation failed: {exc}")
            st.stop()

        answer, used_docs = extract(architecture, result)
        answer = strip_reasoning(answer) or "(no answer produced)"

        st.markdown("### Answer")
        st.success(answer)
        levers = [architecture, gen_label]
        if not is_standard:
            levers.append(f"grader {grader_label}")
        levers.append(index_label.split()[0])  # chunk size
        levers.append("rerank" if rerank else "no-rerank")
        if is_standard:
            levers.append("extract" if extract_on else "baseline")
        levers.append(f"{elapsed:.1f}s")
        levers.append(f"{len(used_docs)} chunks")
        st.caption(" · ".join(levers))

        # Retrieval inspection: vector distance of the top dense chunks to the question.
        label = ("Top dense chunks (first stage, before rerank)" if rerank
                 else "Retrieved chunks vs. question (vector distance)")
        with st.expander(label, expanded=True):
            st.caption("Top chunks by embedding distance to the question — "
                       "lower distance = semantically closer."
                       + (" Reranking then reorders these to pick the top-4 actually used."
                          if rerank else ""))
            scored = store.vectorstore.similarity_search_with_score(question, k=RETRIEVER_K)
            for i, (doc, distance) in enumerate(scored, start=1):
                source = Path(doc.metadata.get("source", "?")).name
                page = doc.metadata.get("page", "?")
                st.markdown(f"**[{i}] distance {distance:.3f}** — {source}, p.{page}")
                st.code(doc.page_content, language=None, wrap_lines=True)

        # What the architecture actually used can differ (grading, rerank, per-sub-question retrieval).
        with st.expander(f"Contexts {architecture} actually used ({len(used_docs)} chunks)"):
            for i, doc in enumerate(used_docs, start=1):
                source = Path(doc.metadata.get("source", "?")).name
                page = doc.metadata.get("page", "?")
                st.markdown(f"**[{i}]** {source}, p.{page}")
                st.code(doc.page_content, language=None, wrap_lines=True)

        st.session_state.history.insert(0, {
            "question": question, "answer": answer,
            "architecture": architecture, "model": gen_label, "time": elapsed,
        })
        st.session_state.history = st.session_state.history[:10]

    if st.session_state.history:
        st.markdown("---")
        st.markdown("### Recent searches")
        for item in st.session_state.history:
            st.markdown(f"**{item['question']}**")
            st.markdown(item["answer"][:200] + ("…" if len(item["answer"]) > 200 else ""))
            st.caption(f"{item['architecture']} · {item['model']} · {item['time']:.1f}s")


if __name__ == "__main__":
    main()
