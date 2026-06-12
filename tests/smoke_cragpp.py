"""Smoke test: CRAG++ behavioral contract (Stage 2 pass criteria).

Asserts with a fake LLM and a keyed counting retriever (no Ollama needed):
  (a) retriever invoked once for the main question + once per sub-question;
  (b) a chunk retrieved by two sub-questions appears once per sub-question's
      contexts and once in final_contexts; within-sub-question duplicates deduped;
  (c) the synthesis prompt contains no word cap;
  (d) the regeneration loop terminates at 2 attempts on a persistent 'no';
  (e) all-docs-rejected falls back to the retrieved docs (never empty contexts);
  (f) the graph compiles and run() returns final_answer + final_contexts.

Run:  python tests/smoke_cragpp.py
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import Field
from langchain_core.documents import Document
from langchain_core.language_models.llms import LLM

from src.graph_builder.graph_builder_cragpp import GraphBuilder

MAIN_Q = "What is PIRLS and how was it administered?"
SUB_Q1 = "What is PIRLS?"
SUB_Q2 = "How was PIRLS administered?"

SUB_QUESTIONS_JSON = (
    f'{{"sub_question_1": {{"query": "{SUB_Q1}", "id": "1"}}, '
    f'"sub_question_2": {{"query": "{SUB_Q2}", "id": "2"}}}}'
)


class FakeLLM(LLM):
    grade_generation_as: str = "yes"
    grade_documents_as: str = "yes"
    prompts: List[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        self.prompts.append(prompt)
        if "Break down the following question" in prompt:
            return SUB_QUESTIONS_JSON
        if "Is this document relevant" in prompt:
            return self.grade_documents_as
        if "Is this generated answer useful" in prompt:
            return self.grade_generation_as
        if "Using the following documents" in prompt:
            return "SUBANSWER"
        if "Given the main question" in prompt:
            return "SYNTHESIZED"
        return "yes"


SHARED = Document(page_content="SHARED chunk relevant to both sub-questions")
DOC_A = Document(page_content="DOC-A what PIRLS is")
DOC_B = Document(page_content="DOC-B how PIRLS was administered")
DOC_MAIN = Document(page_content="DOC-MAIN broad context")


class KeyedRetriever:
    """Returns different docs per query; records call order."""

    def __init__(self, mapping: Dict[str, List[Document]]):
        self.mapping = mapping
        self.calls: List[str] = []

    def invoke(self, query: str) -> List[Document]:
        self.calls.append(query)
        return list(self.mapping.get(query, []))


def run_graph(grade_generation_as="yes", grade_documents_as="yes"):
    retriever = KeyedRetriever({
        MAIN_Q: [DOC_MAIN],
        SUB_Q1: [SHARED, DOC_A],
        SUB_Q2: [SHARED, DOC_B, DOC_B],  # within-sub-question duplicate
    })
    llm = FakeLLM()
    slm = FakeLLM(grade_generation_as=grade_generation_as,
                  grade_documents_as=grade_documents_as)
    builder = GraphBuilder(retriever=retriever, llm=llm, slm=slm)
    builder.build()
    result = builder.run(MAIN_Q)
    return retriever, llm, result


def contents(docs):
    return [d.page_content for d in docs]


def main():
    # --- happy path ---
    retriever, llm, result = run_graph()

    # (a) one retrieval for the main question + one per sub-question
    assert retriever.calls == [MAIN_Q, SUB_Q1, SUB_Q2], (
        f"expected main+per-subq retrieval, got {retriever.calls}")
    print("PASS (a): retriever called once for main question + once per sub-question")

    # (b) dedup: shared chunk once per sub-question and once in final_contexts;
    #     within-sub-question duplicate collapsed
    subq = result["sub_questions"]
    assert contents(subq["sub_question_1"]["contexts"]) == contents([SHARED, DOC_A])
    assert contents(subq["sub_question_2"]["contexts"]) == contents([SHARED, DOC_B]), (
        "within-sub-question duplicate must be deduped")
    assert contents(result["final_contexts"]) == contents([SHARED, DOC_A, DOC_B]), (
        f"final_contexts must be the deduped union, got {contents(result['final_contexts'])}")
    print("PASS (b): shared chunk appears once per sub-question and once in final_contexts")

    # (c) synthesis prompt has no word cap
    synthesis_prompts = [p for p in llm.prompts if "Given the main question" in p]
    assert synthesis_prompts, "synthesis prompt not captured"
    assert all("100 words" not in p and "word" not in p.lower().split("limit")[0][-30:]
               for p in synthesis_prompts) and all("100" not in p for p in synthesis_prompts), (
        "synthesis prompt must not contain a word cap")
    print("PASS (c): synthesis prompt contains no word cap")

    # (f) run() returns final_answer + final_contexts
    assert result["final_answer"] == "SYNTHESIZED"
    assert result["final_contexts"], "final_contexts must be populated"
    assert result["generation_grade"] == "yes" and result["attempts"] == 1
    print("PASS (f): graph compiles; run() returns final_answer + final_contexts")

    # --- (d) loop guard: persistent 'no' terminates at 2 attempts ---
    retriever, llm, result = run_graph(grade_generation_as="no")
    assert result["attempts"] == 2, f"loop guard must cap at 2 attempts, got {result['attempts']}"
    assert result["generation_grade"] == "no"
    assert retriever.calls == [MAIN_Q, SUB_Q1, SUB_Q2], "regeneration must not re-retrieve"
    print("PASS (d): persistent 'no' stops after 2 attempts")

    # --- (e) all-docs-rejected falls back (initial grading AND per-sub-question) ---
    retriever, llm, result = run_graph(grade_documents_as="no")
    assert result["final_answer"] == "SYNTHESIZED", "fallback must still produce an answer"
    assert contents(result["sub_questions"]["sub_question_1"]["contexts"]) == contents([SHARED, DOC_A]), (
        "per-sub-question grading must fall back to that sub-question's retrieved docs")
    assert result["final_contexts"], "fallback must keep final_contexts non-empty"
    print("PASS (e): all-rejected grading falls back; contexts never empty")

    print("ALL STAGE-2 SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
