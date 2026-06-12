"""State for the CRAG++ architecture (4th architecture, see docs/adr/0001).

CRAG++ = CRAG's skeleton (doc grading with fallback, guarded generation-grading
loop) + per-sub-question retrieval, chunk dedup, and no synthesis word cap.
"""

from typing import Dict, List, Literal, TypedDict
from langchain_core.documents import Document


class CRAGppState(TypedDict, total=False):
    question: str
    documents: List[Document]        # initial broad retrieval (after grading)
    sub_questions: Dict[str, dict]   # key -> {"query", "id", "contexts", "answer"}
    final_contexts: List[Document]   # deduped union of all sub-question contexts;
                                     # log THIS as retrieved_context (chunk list, not a blob)
    final_answer: str
    generation_grade: Literal["yes", "no", "not_useful"]
    attempts: int
