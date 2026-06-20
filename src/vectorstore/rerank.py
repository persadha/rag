"""Cross-encoder reranking retriever (Tier-1 retrieval upgrade).

Two-stage retrieve-then-rerank: fetch a wide candidate set from the dense
vectorstore, then re-score each (query, chunk) pair with a cross-encoder and
keep the top-k. Cross-encoders judge query-document relevance jointly, which is
far more precise than the bi-encoder cosine used for first-stage recall.

Uses sentence-transformers' CrossEncoder (already installed via the embedding
stack — no new dependency). The default model is a small CPU-friendly MS-MARCO
reranker (~80 MB, downloaded once).

Exposes `.invoke(query) -> List[Document]` so it is a drop-in for the LangChain
retriever the graphs already call (Standard, CRAG, CRAG++).
"""

from typing import List

from langchain_core.documents import Document

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankRetriever:
    def __init__(self, base_retriever, top_k: int = 4,
                 model_name: str = DEFAULT_RERANK_MODEL):
        """base_retriever should fetch a WIDE candidate set (e.g. k=20); this
        wrapper reranks those down to top_k (e.g. 4)."""
        from sentence_transformers import CrossEncoder
        self.base = base_retriever
        self.top_k = top_k
        self._ce = CrossEncoder(model_name)

    def invoke(self, query: str, config=None, **kwargs) -> List[Document]:
        docs = self.base.invoke(query)
        if not docs:
            return docs
        scores = self._ce.predict([(query, d.page_content) for d in docs])
        ranked = sorted(zip(docs, scores), key=lambda ds: ds[1], reverse=True)
        return [d for d, _ in ranked[:self.top_k]]

    # LangChain compatibility alias (some call sites use the older method name)
    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.invoke(query)
