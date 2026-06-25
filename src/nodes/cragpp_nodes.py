"""Nodes for the CRAG++ architecture.

CRAG++ keeps CRAG's skeleton — document grading with an all-rejected fallback and
a guarded generation-grading loop — and changes three things (docs/adr/0001):
  1. each sub-question retrieves its OWN documents (CRAG reuses the original docs);
  2. retrieved chunks are deduped (within each sub-question and in the union);
  3. the synthesis prompt has NO word cap (CRAG caps at 100 words).
"""

import os
import re
import json

from langchain_core.output_parsers import StrOutputParser

from src.utils.text import strip_reasoning
from src.utils.docs import dedup_documents
from src.state.cragpp_state import CRAGppState

# Direct-answer prompt for the single-hop fast path (mirrors the Standard RAG style:
# concise, extract the exact fact, say so if absent).
DIRECT_ANSWER_PROMPT = (
    "Use the following documents to answer the question. Give a direct, concise answer "
    "with the exact fact(s); if the documents do not contain the answer, say so.\n\n"
    "Documents:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)


class CRAGppNodes:
    def __init__(self, llm, slm, retriever, retrieval_grader_prompt=None,
                 generation_grader_prompt=None, question_rewriter_prompt=None,
                 cross_encoder=None):
        self.llm = llm
        self.slm = slm
        self.retriever = retriever
        self.retrieval_grader_prompt = retrieval_grader_prompt
        self.generation_grader_prompt = generation_grader_prompt
        self.question_rewriter_prompt = question_rewriter_prompt

        # Rerank-aware fixes activate only when a cross-encoder is supplied (i.e.
        # the retriever uses reranking). Without it, behaviour is the legacy path.
        self.cross_encoder = cross_encoder
        # bge-reranker-base emits CALIBRATED [0,1] relevance scores, so the absolute
        # top-1 score is the right single-hop signal (top1>=0.7 ~= 68% single-hop on this
        # set, matching the fact-question share). Margin defaults off; both knobs are env-
        # tunable and apply per reranker model.
        self.grade_threshold = float(os.getenv("CRAGPP_GRADE_THRESHOLD", "0.0"))
        self.singlehop_abs = float(os.getenv("CRAGPP_SINGLEHOP_ABS", "0.7"))
        self.singlehop_margin = float(os.getenv("CRAGPP_SINGLEHOP_MARGIN", "0.0"))
        self.union_topk = int(os.getenv("CRAGPP_UNION_TOPK", "6"))
        self.direct_topk = int(os.getenv("CRAGPP_DIRECT_TOPK", "4"))

        # String output regardless of LLM vs ChatModel, for batch processing
        self.llm_to_str = self.llm | StrOutputParser()
        self.slm_to_str = self.slm | StrOutputParser()

    def _rerank_scores(self, query, documents):
        """[(doc, score)] sorted by cross-encoder relevance, highest first."""
        if not documents:
            return []
        scores = self.cross_encoder.predict([(query, d.page_content) for d in documents])
        return sorted(zip(documents, scores), key=lambda ds: ds[1], reverse=True)

    def _filter_relevant(self, question, documents):
        """Grade docs for relevance, never returning empty.

        With a cross-encoder (T2.3, real Corrective-RAG): keep chunks scoring at/above
        the threshold; if none clear it, fall back to the top-k by score (degrades to
        "best few", not "all"). Without one: the legacy gemma3:1b binary grader, keeping
        originals if it rejects everything."""
        if not documents:
            return []
        if self.cross_encoder is not None:
            ranked = self._rerank_scores(question, documents)
            kept = [doc for doc, score in ranked if score >= self.grade_threshold]
            if not kept:
                kept = [doc for doc, _ in ranked[:self.direct_topk]]
            return kept
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
        """Grade the initial broad retrieval (CRAG skeleton, with fallback).

        With a cross-encoder, also decide single-hop vs multi-hop (T2.5 adaptive
        decomposition): rerank the graded docs against the ORIGINAL question and call
        it single-hop when one chunk clearly dominates (top score clears a floor AND
        beats the runner-up by a margin) — that question is answerable directly, so we
        skip decomposition. <2 docs is trivially single-hop."""
        print("---GRADING DOCUMENTS---")
        documents = state["documents"]
        if not documents:
            print("No documents to grade.")
            return {"documents": []}
        graded = self._filter_relevant(state["question"], documents)
        if self.cross_encoder is None:
            return {"documents": graded}
        ranked = self._rerank_scores(state["question"], graded)
        reranked_docs = [doc for doc, _ in ranked[:self.direct_topk]]
        if len(ranked) < 2:
            is_single_hop = True
        else:
            top1, top2 = ranked[0][1], ranked[1][1]
            is_single_hop = (top1 >= self.singlehop_abs
                             and (top1 - top2) >= self.singlehop_margin)
        print(f"---ADAPTIVE: {'SINGLE-HOP (direct)' if is_single_hop else 'MULTI-HOP (decompose)'}---")
        return {"documents": graded, "reranked_docs": reranked_docs,
                "is_single_hop": is_single_hop}

    def answer_direct(self, state: CRAGppState) -> CRAGppState:
        """Single-hop fast path (T2.5): answer the original question directly from the
        reranked top-k context — no decomposition — like the Standard pipeline. Sets
        final_contexts to that chunk list so it is logged as chunks, not a blob."""
        print("---ANSWERING DIRECTLY (single-hop)---")
        question = state["question"]
        docs = state.get("reranked_docs") or state.get("documents") or []
        context = "\n\n".join(d.page_content for d in docs) if docs else "No documents found."
        prompt = DIRECT_ANSWER_PROMPT.format(context=context, question=question)
        responses = self.llm_to_str.batch([[{"role": "user", "content": prompt}]])
        final_answer = strip_reasoning(responses[0]) if responses else "Could not produce an answer."
        return {"final_answer": final_answer, "final_contexts": docs}

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

        # T2.4: the union is selected for SUB-questions, but precision is judged against
        # the ORIGINAL question — so rerank the union vs the original question and keep the
        # most relevant few. This re-aligns the logged/used context with what's scored.
        if self.cross_encoder is not None and final_contexts:
            ranked = self._rerank_scores(question, final_contexts)
            final_contexts = [doc for doc, _ in ranked[:self.union_topk]]
        if not final_contexts:
            final_contexts = list(documents)

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
            # T3.7: synthesize from the reranked CONTEXT chunks too, not just the relayed
            # sub-answers — facts dropped during per-sub-question answering are recoverable
            # from the chunks. No word cap, but instruct for a direct, exact answer.
            final_contexts = state.get("final_contexts", [])
            if self.cross_encoder is not None and final_contexts:
                context_block = "\n\n".join(d.page_content for d in final_contexts)
                synthesis_prompt_content = (
                    f"Main question: \"{question}\".\n\n"
                    f"Retrieved documents:\n{context_block}\n\n"
                    f"Draft sub-answers:\n{combined_sub_answers}\n\n"
                    "Using the documents and sub-answers, give a direct, concise final answer "
                    "to the main question with the exact fact(s). If they do not contain the "
                    "answer, say so."
                )
            else:
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
