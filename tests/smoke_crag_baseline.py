"""Smoke test: CRAG is frozen at its evaluated design (docs/adr/0001).

Asserts, with a fake LLM and a counting retriever (no Ollama needed):
  1. the graph builds and runs end-to-end;
  2. the retriever is invoked exactly ONCE (only for the original question);
  3. every sub-question's contexts are the original query's documents (doc-reuse);
  4. crash-safety fixes are intact (loop guard caps attempts, grader fallback keeps docs).

Run:  python tests/smoke_crag_baseline.py
"""

import sys
from pathlib import Path
from typing import Any, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document
from langchain_core.language_models.llms import LLM

from src.graph_builder.graph_builder_adv import GraphBuilder

SUB_QUESTIONS_JSON = (
    '{"sub_question_1": {"query": "What is PIRLS?", "id": "1"}, '
    '"sub_question_2": {"query": "How was PIRLS administered?", "id": "2"}}'
)


class FakeLLM(LLM):
    """Routes canned responses by prompt content. `grade_generation_as` lets a test
    force the regeneration loop ('no' forever) to exercise the loop guard."""

    grade_generation_as: str = "yes"
    grade_documents_as: str = "yes"

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        if "Break down the following question" in prompt:
            return SUB_QUESTIONS_JSON
        if "Is this document relevant" in prompt:
            return self.grade_documents_as
        if "Is this generated answer useful" in prompt:
            return self.grade_generation_as
        if "Using the following documents" in prompt:
            return "SUBANSWER"
        if "Provide a comprehensive" in prompt:
            return "SYNTHESIZED"
        return "yes"


class CountingRetriever:
    def __init__(self):
        self.calls = []
        self.docs = [Document(page_content="DOC-A pirls content"),
                     Document(page_content="DOC-B administration content")]

    def invoke(self, query: str) -> List[Document]:
        self.calls.append(query)
        return list(self.docs)


def run_graph(grade_generation_as="yes", grade_documents_as="yes"):
    retriever = CountingRetriever()
    llm = FakeLLM()
    slm = FakeLLM(grade_generation_as=grade_generation_as,
                  grade_documents_as=grade_documents_as)
    builder = GraphBuilder(retriever=retriever, llm=llm, slm=slm)
    builder.build()
    result = builder.run("What is PIRLS and how was it administered?")
    return retriever, result


def main():
    question = "What is PIRLS and how was it administered?"

    # --- happy path: doc-reuse design ---
    retriever, result = run_graph()
    assert retriever.calls == [question], (
        f"retriever must be invoked exactly once with the original question, got {retriever.calls}")
    original_contents = [d.page_content for d in retriever.docs]
    assert len(result["sub_questions"]) == 2, "two sub-questions expected from the rewriter JSON"
    for key, subq in result["sub_questions"].items():
        got = [d.page_content for d in subq["contexts"]]
        assert got == original_contents, (
            f"{key} must reuse the original docs (evaluated CRAG design), got {got}")
    assert result["final_answer"] == "SYNTHESIZED"
    assert result["generation_grade"] == "yes"
    assert result["attempts"] == 1
    print("PASS doc-reuse: retriever called once; both sub-questions reuse original docs")

    # --- loop guard: persistent 'no' terminates at 2 attempts ---
    retriever, result = run_graph(grade_generation_as="no")
    assert result["attempts"] == 2, f"loop guard must cap at 2 attempts, got {result['attempts']}"
    assert result["generation_grade"] == "no"
    assert retriever.calls == [question], "regeneration must not re-retrieve"
    print("PASS loop guard: persistent 'no' stops after 2 attempts")

    # --- grader fallback: all docs rejected -> originals kept, no dead end ---
    retriever, result = run_graph(grade_documents_as="no")
    assert result["final_answer"] == "SYNTHESIZED", "fallback must still produce an answer"
    for key, subq in result["sub_questions"].items():
        got = [d.page_content for d in subq["contexts"]]
        assert got == original_contents, f"{key} must fall back to original docs"
    print("PASS grader fallback: all-rejected falls back to original docs and still answers")

    print("ALL STAGE-1 SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
