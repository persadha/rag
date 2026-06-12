"""Nodes for the CRAG++ architecture.

CRAG++ keeps CRAG's skeleton — document grading with an all-rejected fallback and
a guarded generation-grading loop — and changes three things (docs/adr/0001):
  1. each sub-question retrieves its OWN documents (CRAG reuses the original docs);
  2. retrieved chunks are deduped (within each sub-question and in the union);
  3. the synthesis prompt has NO word cap (CRAG caps at 100 words).
"""

import re
import json

from langchain_core.output_parsers import StrOutputParser

from src.utils.text import strip_reasoning
from src.utils.docs import dedup_documents
from src.state.cragpp_state import CRAGppState


class CRAGppNodes:
    def __init__(self, llm, slm, retriever, retrieval_grader_prompt=None,
                 generation_grader_prompt=None, question_rewriter_prompt=None):
        self.llm = llm
        self.slm = slm
        self.retriever = retriever
        self.retrieval_grader_prompt = retrieval_grader_prompt
        self.generation_grader_prompt = generation_grader_prompt
        self.question_rewriter_prompt = question_rewriter_prompt

        # String output regardless of LLM vs ChatModel, for batch processing
        self.llm_to_str = self.llm | StrOutputParser()
        self.slm_to_str = self.slm | StrOutputParser()

    def _filter_relevant(self, question, documents):
        """Grade docs for relevance with the slm; fall back to the full list if the
        grader rejects everything (never return an empty context)."""
        if not documents:
            return []
        grading_inputs = [
            self.retrieval_grader_prompt.format_messages(question=question, document=doc.page_content)
            for doc in documents
        ]
        grades = self.slm_to_str.batch(grading_inputs)
        kept = [doc for grade, doc in zip(grades, documents)
                if "yes" in strip_reasoning(grade).lower()]
        if not kept:
            print(f"All documents graded irrelevant for '{question}'; keeping originals.")
            return list(documents)
        return kept

    def grade_documents(self, state: CRAGppState) -> CRAGppState:
        """Grade the initial broad retrieval (CRAG skeleton, with fallback)."""
        print("---GRADING DOCUMENTS---")
        documents = state["documents"]
        if not documents:
            print("No documents to grade.")
            return {"documents": []}
        return {"documents": self._filter_relevant(state["question"], documents)}

    def plan_sub_steps(self, state: CRAGppState) -> CRAGppState:
        """Decompose the question; retrieve, grade, and dedup documents PER
        sub-question; expose the deduped union as final_contexts."""
        question = state["question"]
        documents = state["documents"]
        print(f"---PLANNING SUB-STEPS FOR QUESTION: {question}---")

        prompt_messages = self.question_rewriter_prompt.format_messages(question=question)
        responses = self.llm_to_str.batch([prompt_messages])
        sub_questions_str = strip_reasoning(responses[0])

        pattern = r"\{[\s\S]*\}"
        try:
            parsed_sub_questions = json.loads(re.findall(pattern, sub_questions_str)[0])
        except (json.JSONDecodeError, IndexError):
            print(f"Warning: LLM did not return valid JSON for sub-questions: {sub_questions_str}. "
                  "Using original question as a single sub-question.")
            parsed_sub_questions = {"sub_question_1": {"query": question, "id": "1"}}

        final_sub_questions = {}
        for key, subq_data in parsed_sub_questions.items():
            query = subq_data.get("query", "")
            contexts = self.retriever.invoke(query) if query else list(documents)
            if not contexts:
                contexts = list(documents)
            contexts = self._filter_relevant(query or question, contexts)
            contexts = dedup_documents(contexts)
            final_sub_questions[key] = {
                "query": query,
                "id": subq_data.get("id", key),
                "contexts": contexts,
            }

        final_contexts = dedup_documents(
            [doc for subq in final_sub_questions.values() for doc in subq["contexts"]]
        )

        return {"sub_questions": final_sub_questions, "documents": documents,
                "final_contexts": final_contexts}

    def generate_answers(self, state: CRAGppState) -> CRAGppState:
        """Answer each sub-question from its own contexts, then synthesize a final
        answer — without a word cap."""
        print("---GENERATING ANSWERS FOR SUB-QUESTIONS AND FINAL ANSWER---")
        question = state["question"]
        sub_questions = state["sub_questions"]
        all_individual_answers = []

        generation_inputs = []
        ordered_sub_question_keys = []
        for key, subq in sub_questions.items():
            context_joined = "\n\n".join(doc.page_content for doc in subq["contexts"]) \
                if subq["contexts"] else "No specific documents found."
            prompt_content = (f"Using the following documents:\n{context_joined}\n"
                              f"Answer the question: \"{subq['query']}\".")
            generation_inputs.append([{"role": "user", "content": prompt_content}])
            ordered_sub_question_keys.append(key)

        if generation_inputs:
            individual_answers = self.llm_to_str.batch(generation_inputs)
            individual_answers = [strip_reasoning(a) for a in individual_answers]
            for i, key in enumerate(ordered_sub_question_keys):
                sub_questions[key]["answer"] = individual_answers[i]
                all_individual_answers.append(individual_answers[i])
        else:
            print("No sub-questions to generate answers for.")

        if all_individual_answers:
            combined_sub_answers = "\n\n".join(all_individual_answers)
            # No word cap (CRAG's <=100-word cap measurably truncated correct answers)
            synthesis_prompt_content = (
                f"Given the main question: \"{question}\" and the following answers to "
                f"its sub-questions:\n{combined_sub_answers}\n"
                "Provide a complete, direct final answer to the main question, based only "
                "on the sub-answers above."
            )
            synthesis_input = [[{"role": "user", "content": synthesis_prompt_content}]]
            final_answer_responses = self.llm_to_str.batch(synthesis_input)
            final_answer = strip_reasoning(final_answer_responses[0]) \
                if final_answer_responses else "Could not synthesize a final answer."
        else:
            final_answer = "No answers were generated for sub-questions."

        return {"sub_questions": sub_questions, "final_answer": final_answer}

    def grade_generation(self, state: CRAGppState) -> CRAGppState:
        """Grade the final answer; count attempts so the graph can cap regeneration."""
        print("---GRADING GENERATION---")
        grading_input = self.generation_grader_prompt.format_messages(
            question=state["question"], generation=state["final_answer"])
        grade_responses = self.slm_to_str.batch([grading_input])
        grade_str = strip_reasoning(grade_responses[0]).lower()

        attempts = state.get("attempts", 0) + 1
        if "yes" in grade_str:
            print("---GENERATION USEFUL---")
            return {"generation_grade": "yes", "attempts": attempts}
        print(f"---GENERATION NOT USEFUL (attempt {attempts})---")
        return {"generation_grade": "no", "attempts": attempts}
