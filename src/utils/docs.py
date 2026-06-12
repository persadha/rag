"""Document-list helpers shared by graph nodes and the eval harness."""

from typing import List
from langchain_core.documents import Document


def dedup_documents(documents: List[Document]) -> List[Document]:
    """Drop exact-content duplicates, keeping first occurrence order."""
    seen = set()
    unique = []
    for doc in documents:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique.append(doc)
    return unique
