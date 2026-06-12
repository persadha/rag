"""Streamlit UI for the PIRLS RAG system.

Run:  streamlit run streamlit_app_auto.py

Requires the persisted index (python scripts/build_index.py) and a local
Ollama server. Architecture and model are selectable in the sidebar; every
answer shows the retrieved chunks with their vector distance to the question
(retrieval inspection, plans.md item 6). Bug history: reports/ui_review.md.
"""

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.config.config_api import APIConfig
from src.utils.text import strip_reasoning

ARCHITECTURES = ("Standard", "CRAG", "CRAG++")
OLLAMA_MODELS = ("llama3:8b", "gemma3:4b", "deepseek-r1:8b")
RETRIEVER_K = 4

st.set_page_config(page_title="PIRLS RAG Search", page_icon="📚", layout="centered")


@st.cache_resource
def load_store():
    """Load the persisted Chroma index (built once by scripts/build_index.py)."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from src.vectorstore.vectorstore import VectorStore

    persist_dir = str(Path(__file__).parent / "chroma_db")
    embeddings = HuggingFaceEmbeddings(model_name=Config.DEFAULT_EMBEDDING_MODEL)
    store = VectorStore(embeddings, persist_directory=persist_dir)
    store.load_vectorstore()
    return store


@st.cache_resource
def get_graph(architecture: str, model_name: str):
    """Build the selected graph once per (architecture, model)."""
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser

    store = load_store()
    retriever = store.get_retriever(k=RETRIEVER_K)
    llm = ChatOllama(model=model_name, temperature=0, num_ctx=8192) | StrOutputParser()
    slm = APIConfig.get_ollama_slm()  # gemma3:1b grader (CRAG/CRAG++ design)

    if architecture == "Standard":
        from src.graph_builder.graph_builder import GraphBuilder
        builder = GraphBuilder(retriever=retriever, llm=llm)
    elif architecture == "CRAG":
        from src.graph_builder.graph_builder_adv import GraphBuilder
        builder = GraphBuilder(retriever=retriever, llm=llm, slm=slm)
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
        model_name = st.selectbox("Generator model (Ollama)", OLLAMA_MODELS)
        st.caption(f"Retriever: top-{RETRIEVER_K} chunks, "
                   f"{Config.DEFAULT_EMBEDDING_MODEL.split('/')[-1]} embeddings. "
                   "Doc grader (CRAG/CRAG++): gemma3:1b.")

    if "history" not in st.session_state:
        st.session_state.history = []

    try:
        store = load_store()
        if store.vectorstore is None:
            st.error("No index found at ./chroma_db — build it first:  "
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
        try:
            with st.spinner(f"Running {architecture} with {model_name}…"):
                graph = get_graph(architecture, model_name)
                start = time.time()
                result = graph.run(question)
                elapsed = time.time() - start
        except Exception as exc:
            st.error(f"Generation failed: {exc}")
            st.stop()

        answer, used_docs = extract(architecture, result)
        answer = strip_reasoning(answer) or "(no answer produced)"

        st.markdown("### Answer")
        st.success(answer)
        st.caption(f"{architecture} · {model_name} · {elapsed:.1f}s · "
                   f"{len(used_docs)} context chunks")

        # Retrieval inspection: vector distance of the top chunks to the question
        with st.expander("Retrieved chunks vs. question (vector distance)", expanded=True):
            st.caption("Top chunks by embedding distance to the question — "
                       "lower distance = semantically closer.")
            scored = store.vectorstore.similarity_search_with_score(question, k=RETRIEVER_K)
            for i, (doc, distance) in enumerate(scored, start=1):
                source = Path(doc.metadata.get("source", "?")).name
                page = doc.metadata.get("page", "?")
                st.markdown(f"**[{i}] distance {distance:.3f}** — {source}, p.{page}")
                st.code(doc.page_content, language=None, wrap_lines=True)

        # What the architecture actually used can differ (grading, per-sub-question retrieval)
        with st.expander(f"Contexts {architecture} actually used ({len(used_docs)} chunks)"):
            for i, doc in enumerate(used_docs, start=1):
                source = Path(doc.metadata.get("source", "?")).name
                page = doc.metadata.get("page", "?")
                st.markdown(f"**[{i}]** {source}, p.{page}")
                st.code(doc.page_content, language=None, wrap_lines=True)

        st.session_state.history.insert(0, {
            "question": question, "answer": answer,
            "architecture": architecture, "model": model_name, "time": elapsed,
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
