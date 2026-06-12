from typing import List
from pydantic import BaseModel
from langchain_core.documents import Document

class RAGState(BaseModel):
    question: str
    retrieved_docs: List[Document] = []
    answer: str = ""
    context: str = ""
    feedback: str = ""
    is_hallucination: bool = False
    is_unsupported: bool = False
