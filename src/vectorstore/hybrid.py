"""Hybrid retrieval: sparse (BM25) + dense, fused with Reciprocal Rank Fusion.

Dense (bi-encoder cosine) and sparse (BM25 lexical) retrievers fail on different
queries — dense misses exact terms (country names, acronyms like NAPLAN), BM25
misses paraphrase. RRF merges their rankings without needing comparable scores:
each document scores Sum 1/(rrf_k + rank) over the lists it appears in.

EnsembleRetriever is not available in the installed LangChain, so RRF is
implemented here (~10 lines). Exposes `.invoke(query) -> List[Document]` so it is
a drop-in base for RerankRetriever and the graphs (Standard, CRAG, CRAG++).

Typical use (wide candidate pool, then a cross-encoder reranks to top-4):
    corpus = build_bm25_corpus(store)
    hybrid = HybridRetriever(store.get_retriever(k=20), corpus, k_each=20)
    retriever = RerankRetriever(hybrid, top_k=4, model_name="BAAI/bge-reranker-base")
"""

from typing import List

from langchain_core.documents import Document

DEFAULT_RRF_K = 60  # standard RRF constant; dampens the weight of top ranks


def build_bm25_corpus(store) -> List[Document]:
    """Reconstruct the full chunk corpus from a loaded VectorStore (Chroma).

    BM25 indexes in memory, so it needs every chunk's text — Chroma's .get()
    returns all documents + metadata for the collection."""
    if store.vectorstore is None:
        raise ValueError("Vector store not loaded; call load_vectorstore() first.")
    data = store.vectorstore.get()  # {'documents': [...], 'metadatas': [...], ...}
    docs = data.get("documents") or []
    metas = data.get("metadatas") or [{}] * len(docs)
    return [Document(page_content=t, metadata=(m or {}))
            for t, m in zip(docs, metas) if t]


class HybridRetriever:
    def __init__(self, dense_retriever, corpus_docs: List[Document],
                 k_each: int = 20, rrf_k: int = DEFAULT_RRF_K):
        """dense_retriever should already fetch k_each results; BM25 is built here
        over corpus_docs and also returns k_each. Their rankings are RRF-fused."""
        from langchain_community.retrievers import BM25Retriever
        self.dense = dense_retriever
        self.rrf_k = rrf_k
        self.bm25 = BM25Retriever.from_documents(corpus_docs)
        self.bm25.k = k_each

    def invoke(self, query: str, config=None, **kwargs) -> List[Document]:
        dense_docs = self.dense.invoke(query)
        bm25_docs = self.bm25.invoke(query)

        # RRF over both ranked lists, keyed by chunk text (dedups across lists).
        scores: dict = {}
        first: dict = {}
        for ranked in (dense_docs, bm25_docs):
            for rank, doc in enumerate(ranked):
                key = doc.page_content
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
                first.setdefault(key, doc)
        fused = sorted(first.values(), key=lambda d: scores[d.page_content], reverse=True)
        return fused

    # LangChain compatibility alias.
    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.invoke(query)
