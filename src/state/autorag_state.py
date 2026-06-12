"""RAG state definition for LangGraph"""

from typing import Dict, List, TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document


# ---- State types ----
class SubQuery(TypedDict):
    query: str
    plans: List[str]
    contexts: List[Document]
    answer: str
    steps: List[str]


class AutoRAGState(BaseModel):
    question: str
    temp_questions: List[str] = Field(default_factory=list)
    sub_questions: Dict[str, SubQuery] = Field(default_factory=dict)
    sub_steps: Dict[str, List[str]] = Field(default_factory=dict)
    retrieved_docs: List[Document] = Field(default_factory=list)
    revised: bool = False
    attempts: int = 0
    answers: List[str] = Field(default_factory=list)
    final_answer: str = ""