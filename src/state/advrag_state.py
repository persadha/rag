"""Advanced RAG (CRAG) state definition for LangGraph.

This is the single source of truth for the advanced/CRAG graph state. It is a
``TypedDict`` (not a Pydantic model): the graph builder and nodes read it with dict
syntax (``state["..."]``) and return *partial* dict updates that LangGraph merges.
``total=False`` lets a node return only the keys it changes.
"""

from typing import Dict, List, Literal, TypedDict

from langchain_core.documents import Document


class AdvanceRAGState(TypedDict, total=False):
    question: str                                         # original user question
    documents: List[Document]                             # retrieved (then graded) docs
    sub_questions: Dict[str, dict]                        # {sub_question_x: {query, id, contexts, answer}}
    final_answer: str                                     # synthesized answer
    generation_grade: Literal["yes", "no", "not_useful"]  # answer-usefulness grade
    attempts: int                                         # generation attempts (loop guard)
