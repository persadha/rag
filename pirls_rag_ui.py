"""Streamlit UI for the PIRLS RAG assistant (production configuration).

Run:  streamlit run pirls_rag_ui.py

Wired to the best architecture combination found across r3 (experiments E1–E12):
**Standard RAG + hybrid (BM25+dense+RRF) retrieval + bge-reranker-base**, the
default, with a local Ollama generator (llama3:8b). The benchmarked underperformers
(CRAG / CRAG++ architectures, the extraction-prompt A/B, the chunk-size and grader
levers) are removed. Users choose between **open** local models (private, on-device
via Ollama) and **closed** cloud models (higher accuracy), and can **attach files**
(PDF / DOCX / TXT) to ask questions over their own documents alongside the corpus.

Requires a persisted index (python scripts/build_index.py); local models need an
Ollama server (OLLAMA_BASE_URL, default localhost:11434); closed models need keys
in .env. Best results to date: gpt-5.4-mini AC 0.774, llama3:8b 0.689 (n=195).
"""

import hashlib
import html as _html
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

sys.path.append(str(Path(__file__).parent))
load_dotenv()  # API keys for the closed (cloud) models

ROOT = Path(__file__).resolve().parent

from src.config.config import Config
from src.config.config_api import APIConfig
from src.utils.text import strip_reasoning
from src.utils.docloader import SUPPORTED_EXTENSIONS, files_to_documents

# --- Production configuration (the winning combination) ---------------------
PERSIST_DIR = "chroma_db"               # 1000/100 index used by the best runs
RETRIEVER_K = 4                          # chunks handed to the generator
CANDIDATES = 20                          # first-stage pool feeding the reranker
RERANKER_MODEL = "BAAI/bge-reranker-base"  # best reranker under hybrid (E9–E12)

# Open = local via Ollama (private, on-device). The selectable models are discovered
# at runtime from the local Ollama server (equivalent to `ollama list`), so any pulled
# model can be chosen. llama3:8b is preferred as the default when present (best open
# model on hybrid retrieval, AC 0.689).
PREFERRED_LOCAL_DEFAULT = "llama3:8b"
# Closed = cloud API (higher accuracy, data leaves the machine). label -> APIConfig name.
CLOSED_MODELS = {
    "GPT-5.4-mini  (OpenAI)": "openai-mini",       # best overall (AC 0.774)
    "Claude Haiku  (Anthropic)": "haiku",
}

st.set_page_config(page_title="IEA • PIRLS Document Search", page_icon="📘", layout="wide")


# --- Local model discovery --------------------------------------------------
@st.cache_data(ttl=30)
def list_ollama_models():
    """Models installed on the local Ollama server (like `ollama list`), via its
    /api/tags endpoint. Cached briefly so newly pulled models appear without a
    restart. Returns [] if Ollama is unreachable."""
    import json
    import urllib.request
    url = Config.OLLAMA_BASE_URL.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.load(resp)
        names = [m["name"] for m in data.get("models", []) if m.get("name")]
        # Drop embedding-only models — they can't generate chat answers, so picking
        # one as the generator would only error. (Heuristic: "embed" in the name.)
        return sorted(n for n in names if "embed" not in n.lower())
    except Exception:
        return []


# --- Cached heavy resources -------------------------------------------------
@st.cache_resource
def load_store(persist_dir: str):
    from langchain_huggingface import HuggingFaceEmbeddings
    from src.vectorstore.vectorstore import VectorStore
    full_dir = str(Path(__file__).parent / persist_dir)
    embeddings = HuggingFaceEmbeddings(model_name=Config.DEFAULT_EMBEDDING_MODEL)
    store = VectorStore(embeddings, persist_directory=full_dir)
    store.load_vectorstore()
    return store


@st.cache_resource
def get_corpus_hybrid(persist_dir: str, k_each: int):
    """Hybrid BM25+dense retriever over the whole corpus (BM25 index built once)."""
    from src.vectorstore.hybrid import HybridRetriever, build_bm25_corpus
    store = load_store(persist_dir)
    corpus = build_bm25_corpus(store)
    return HybridRetriever(store.get_retriever(k=k_each), corpus, k_each=k_each)


@st.cache_resource
def load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


@st.cache_resource
def build_uploaded_retriever(sig: str, _docs, k: int):
    """In-memory dense retriever over the user's uploaded chunks. `sig` (content
    hash) is the cache key; `_docs` is underscore-prefixed so Streamlit doesn't
    try to hash the Document list."""
    from langchain_core.vectorstores import InMemoryVectorStore
    store = load_store(PERSIST_DIR)  # reuse the same embedding model
    vs = InMemoryVectorStore.from_documents(_docs, store.embeddings)
    return vs.as_retriever(search_kwargs={"k": k})


# --- Retriever assembly -----------------------------------------------------
class UnionRetriever:
    """Concatenate + dedup results from several retrievers (corpus + uploaded)."""
    def __init__(self, retrievers):
        self.retrievers = retrievers

    def invoke(self, query: str, config=None, **kwargs):
        seen, out = set(), []
        for r in self.retrievers:
            for d in r.invoke(query):
                if d.page_content not in seen:
                    seen.add(d.page_content)
                    out.append(d)
        return out

    def get_relevant_documents(self, query: str):
        return self.invoke(query)


def build_llm(kind: str, value: str):
    """String-output runnable for a local Ollama model ('local') or API model ('api')."""
    if kind == "local":
        from langchain_ollama import ChatOllama
        from langchain_core.output_parsers import StrOutputParser
        chat = ChatOllama(model=value, temperature=0, num_ctx=8192,
                          base_url=Config.OLLAMA_BASE_URL)
        return chat | StrOutputParser()
    return APIConfig.get_generator(value)  # named API model; raises if key missing


def build_retriever(use_hybrid: bool, uploaded_docs, sig: str, only_uploaded: bool):
    """Assemble the query-time retriever from cached parts. The cross-encoder
    reranker (when used) fairly selects the top-4 across corpus + uploaded chunks."""
    store = load_store(PERSIST_DIR)
    need_wide = use_hybrid or bool(uploaded_docs)  # wide pool only matters if we rerank

    if uploaded_docs and only_uploaded:
        base = build_uploaded_retriever(sig, uploaded_docs, CANDIDATES if need_wide else RETRIEVER_K)
    else:
        if use_hybrid:
            corpus = get_corpus_hybrid(PERSIST_DIR, CANDIDATES)
        else:
            corpus = store.get_retriever(k=CANDIDATES if need_wide else RETRIEVER_K)
        if uploaded_docs:
            up = build_uploaded_retriever(sig, uploaded_docs, CANDIDATES)
            base = UnionRetriever([corpus, up])
        else:
            base = corpus

    if need_wide:
        from src.vectorstore.rerank import RerankRetriever
        return RerankRetriever(base, top_k=RETRIEVER_K,
                               cross_encoder=load_cross_encoder(RERANKER_MODEL))
    return base  # plain dense top-4


def build_graph(retriever, gen_kind: str, gen_value: str):
    from src.graph_builder.graph_builder import GraphBuilder
    builder = GraphBuilder(retriever=retriever, llm=build_llm(gen_kind, gen_value))
    builder.build()
    return builder


def main():
    st.markdown("""
<style>
.block-container { padding-top: 80px; padding-bottom: 4.5rem; }

#header-sticky {
  position: sticky;
  top: 0;
  z-index: 1000;
  background: #ffffff;
  margin: -12px 0 10px 0;
  padding: 10px 24px 8px 24px;
  border-bottom: 1px solid #e5e7eb;
  box-shadow: 0 4px 10px rgba(0,0,0,0.02);
}

.topbar { display: flex; align-items: center; gap: 14px; padding: .4rem .25rem; margin: 0; }
.topbar h3 { margin: 0; font-size: 1.4rem; font-weight: 700; letter-spacing: .2px; color: #111827; }

.pulse-dot {
  display: inline-block; width: 10px; height: 10px; border-radius: 50%;
  background: #22c55e; box-shadow: 0 0 0 rgba(34,197,94,.7);
  animation: pulse 1.5s infinite; margin-right: 6px;
}
@keyframes pulse {
  0%   { box-shadow: 0 0 0 0   rgba(34,197,94,.7); }
  70%  { box-shadow: 0 0 0 10px rgba(34,197,94,0);  }
  100% { box-shadow: 0 0 0 0   rgba(34,197,94,0);   }
}

.footer {
  position: fixed; left: 0; right: 0; bottom: 0; height: 40px;
  background: #fafafa; border-top: 1px solid #e5e7eb;
  display: flex; align-items: center; justify-content: center;
  font-size: .85rem; color: #6b7280; z-index: 9999;
}

[data-testid="stSidebar"] { min-width: 300px !important; }

.source-chunk {
  background: #f8fafc;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  padding: .65rem .9rem;
  margin-bottom: .5rem;
  font-size: .85rem;
  line-height: 1.55;
  color: #374151;
}
.source-chunk .sc-meta { font-weight: 600; font-size: .8rem; color: #6b7280; margin-bottom: .35rem; }
.source-chunk .sc-text { white-space: pre-wrap; word-break: break-word; }

.history-card {
  background: #f9fafb;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: .75rem 1rem;
  margin-bottom: .5rem;
}
.history-card .hq { font-weight: 600; font-size: .95rem; color: #111827; margin-bottom: .3rem; }
.history-card .ha { font-size: .88rem; color: #374151; line-height: 1.5; margin-bottom: .25rem; }
.history-card .hm { font-size: .78rem; color: #9ca3af; }

.answer-card {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-left: 4px solid #2563eb;
  border-radius: 8px;
  padding: 1rem 1.25rem;
  margin: .5rem 0 .75rem 0;
  font-size: 1rem;
  line-height: 1.6;
  color: #111827;
}

[data-testid="stFormSubmitButton"] button {
  background-color: #2563eb !important;
  color: white !important;
  border: none !important;
  font-weight: 600 !important;
  border-radius: 8px !important;
  transition: background-color 0.25s ease-in-out;
}
[data-testid="stFormSubmitButton"] button:hover {
  background-color: #1d4ed8 !important;
}
</style>
""", unsafe_allow_html=True)

    st.markdown(
        '<div id="header-sticky">'
        '<div class="topbar"><h3>IEA • PIRLS Document Search</h3></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        _logo = ROOT / "img" / "logo.png"
        if _logo.exists():
            st.image(str(_logo), use_container_width=True)
        else:
            st.markdown(
                '<div style="text-align:center;margin:2px 0 10px;">'
                '<img src="https://www.iea.nl/sites/default/files/2020-05/IEA_Hamburg_logo_rgb.png" width="160"/>'
                '</div>',
                unsafe_allow_html=True,
            )
        st.divider()
        st.subheader("Model")
        is_open = st.radio(
            "Model type",
            ["Open — local & private", "Closed — cloud API"],
            index=0,
            help="Open models run locally via Ollama; your data never leaves this machine. "
                 "Closed models call a cloud API (higher accuracy, but data is sent to the provider).",
        ).startswith("Open")

        if is_open:
            local_models = list_ollama_models()
            if local_models:
                default_idx = (local_models.index(PREFERRED_LOCAL_DEFAULT)
                               if PREFERRED_LOCAL_DEFAULT in local_models else 0)
                model_label = st.selectbox("Local Ollama model", local_models, index=default_idx,
                                           help="Discovered from the local Ollama server "
                                                "(`ollama list`). Pull more with `ollama pull <model>`.")
                gen_kind, gen_value = "local", model_label
            else:
                st.warning("No local Ollama models found. Is the server running? "
                           "Pull one with `ollama pull llama3:8b`, then refresh.")
                model_label, gen_kind, gen_value = PREFERRED_LOCAL_DEFAULT, "local", PREFERRED_LOCAL_DEFAULT
            if st.button("🔄 Refresh model list"):
                list_ollama_models.clear()
                st.rerun()
        else:
            model_label = st.selectbox("Model", list(CLOSED_MODELS))
            gen_kind, gen_value = "api", CLOSED_MODELS[model_label]

        st.subheader("Retrieval")
        use_hybrid = st.checkbox(
            "High-accuracy retrieval (hybrid + reranker)", value=True,
            help="Hybrid BM25 + dense retrieval, reranked by a cross-encoder (bge-reranker-base). "
                 "The best-performing configuration. Uncheck for faster, lower-accuracy dense-only search.",
        )

        st.subheader("Attach documents")
        uploaded_files = st.file_uploader(
            "PDF, DOCX, TXT", type=SUPPORTED_EXTENSIONS, accept_multiple_files=True,
            help="Ask questions over your own files. They are processed in-memory for this "
                 "session only and never added to the persistent index.",
        )
        only_uploaded = False
        if uploaded_files:
            only_uploaded = st.checkbox("Search only the uploaded files", value=False,
                                        help="Ignore the PIRLS corpus and answer purely from your attachments.")

        st.caption(f"Standard RAG · top-{RETRIEVER_K} chunks · "
                   f"{Config.DEFAULT_EMBEDDING_MODEL.split('/')[-1]} embeddings"
                   + (f" · reranker {RERANKER_MODEL.split('/')[-1]}" if use_hybrid else ""))

    # Parse + hash uploads (cache key) once per render.
    uploaded_docs, sig = None, ""
    if uploaded_files:
        pairs = [(f.name, f.getvalue()) for f in uploaded_files]
        sig = hashlib.md5(b"".join(n.encode() + d for n, d in pairs)).hexdigest()
        with st.spinner("Reading attached documents…"):
            uploaded_docs = files_to_documents(pairs)
        st.sidebar.success(f"{len(uploaded_files)} file(s) → {len(uploaded_docs)} chunks")

    if "history" not in st.session_state:
        st.session_state.history = []

    try:
        store = load_store(PERSIST_DIR)
        if store.vectorstore is None:
            st.error(f"No index found at ./{PERSIST_DIR} — build it first:  "
                     "`python scripts/build_index.py`")
            st.stop()
    except Exception as exc:
        st.error(f"Failed to load the vector store: {exc}")
        st.stop()

    with st.form("search_form"):
        question = st.text_input("Your question:",
                                 placeholder="e.g. How is reading achievement scaled in PIRLS 2021?")
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted and question.strip():
        question = question.strip()
        os.environ["GEN_PROMPT_STYLE"] = "baseline"
        try:
            with st.spinner(f"Answering with {model_label}…"):
                retriever = build_retriever(use_hybrid, uploaded_docs, sig, only_uploaded)
                graph = build_graph(retriever, gen_kind, gen_value)
                start = time.time()
                result = graph.run(question)
                elapsed = time.time() - start
        except RuntimeError as exc:  # missing API key for a closed model, etc.
            st.error(str(exc))
            st.stop()
        except Exception as exc:
            st.error(f"Generation failed: {exc}")
            st.stop()

        answer = strip_reasoning(result.get("answer", "")) or "(no answer produced)"
        used_docs = result.get("retrieved_docs", [])

        st.markdown(
            f'<div class="answer-card">{_html.escape(answer).replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )
        tags = [model_label.split("  ")[0],
                "hybrid+rerank" if use_hybrid else "dense",
                f"{elapsed:.1f}s", f"{len(used_docs)} sources"]
        if uploaded_docs:
            tags.insert(1, "uploaded-only" if only_uploaded else "corpus+uploaded")
        st.caption(" · ".join(tags))

        # Source attribution (transparency: every answer shows the chunks it used).
        with st.expander(f"Sources used ({len(used_docs)} chunks)", expanded=True):
            uploaded_names = {f.name for f in (uploaded_files or [])}
            chunks_html = ""
            for i, doc in enumerate(used_docs, start=1):
                source = Path(doc.metadata.get("source", "?")).name
                page = doc.metadata.get("page")
                badge = "📎 " if doc.metadata.get("source") in uploaded_names else ""
                where = _html.escape(f"{badge}[{i}] {source}" + (f", p.{page}" if page is not None else ""))
                text = _html.escape(doc.page_content)
                chunks_html += (
                    f'<div class="source-chunk">'
                    f'<div class="sc-meta">{where}</div>'
                    f'<div class="sc-text">{text}</div>'
                    f'</div>'
                )
            st.markdown(chunks_html, unsafe_allow_html=True)

        st.session_state.history.insert(0, {
            "question": question, "answer": answer,
            "model": model_label, "time": elapsed,
        })
        st.session_state.history = st.session_state.history[:10]

    if st.session_state.history:
        st.markdown("---")
        st.markdown("### Recent questions")
        cards_html = ""
        for item in st.session_state.history:
            q = _html.escape(item["question"])
            a = _html.escape(item["answer"][:200]) + ("…" if len(item["answer"]) > 200 else "")
            m = _html.escape(f"{item['model'].split('  ')[0]} · {item['time']:.1f}s")
            cards_html += (
                f'<div class="history-card">'
                f'<div class="hq">{q}</div>'
                f'<div class="ha">{a}</div>'
                f'<div class="hm">{m}</div>'
                f'</div>'
            )
        st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown(
        '<div class="footer">© IEA Hamburg · PIRLS Document Search</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
