"""RAG state definition for LangGraph"""

from typing import Dict, List, TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document


# ---- State types ----
class SubQuery(TypedDict):
    query: str
    contexts: List[Document]
    answer: str
    steps: List[str]


class AutoRAGState(BaseModel):
    question: str
    sub_questions: Dict[str, SubQuery] = Field(default_factory=dict)
    sub_steps: Dict[str, List[str]] = Field(default_factory=dict)
    answers: List[str] = Field(default_factory=list)
    final_answer: str = ""