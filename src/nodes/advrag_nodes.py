# /content/drive/MyDrive/RAG/src/nodes/advrag_nodes.py

import os
import re
import json
from typing import List, Literal, TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage
from src.utils.text import strip_reasoning

# Canonical state lives in src/state/advrag_state.py (single source of truth).
from src.state.advrag_state import AdvanceRAGState


class AdVRagNodes:
    def __init__(self, llm, slm, retriever=None, retrieval_grader_prompt=None, generation_grader_prompt=None, question_rewriter_prompt=None):
        self.llm = llm # This llm is typically a ChatModel (e.g., ChatOllama)
        self.slm = slm
        self.retriever = retriever  # P1: used to retrieve fresh docs per sub-question
        self.retrieval_grader_prompt = retrieval_grader_prompt
        self.generation_grader_prompt = generation_grader_prompt
        self.question_rewriter_prompt = question_rewriter_prompt

        # Create a new chain that ensures string output for batch processing
        self.llm_to_str = self.llm | StrOutputParser()
        self.slm_to_str = self.slm | StrOutputParser()

    def plan_sub_steps(self, state: AdvanceRAGState) -> AdvanceRAGState:
        """
        Plans sub-steps (sub-questions) for the given question.
        Uses the LLM to rewrite the original question into a list of sub-questions.
        """
        question = state["question"]
        documents = state["documents"] # Retrieved docs for the original question
        print(f"---PLANNING SUB-STEPS FOR QUESTION: {question}---")

        # Format prompt for the question rewriter, getting a list of BaseMessage
        prompt_messages = self.question_rewriter_prompt.format_messages(question=question)

        # Use batch for consistency and performance; llm_to_str ensures string output
        # batch expects a list of inputs, where each input can be a PromptValue or list of messages
        # Corrected: Removed extra nesting, passing [prompt_messages] instead of [[prompt_messages]]
        responses_str_list = self.llm_to_str.batch([prompt_messages]) # Pass as list of single list of messages
        sub_questions_str = strip_reasoning(responses_str_list[0]) # strip <think> before JSON parse

        pattern = r"\{[\s\S]*\}"
        
        # Assuming sub_questions_str is a JSON string of sub-questions
        try:
            # Example expected format: {"sub_question_1": {"query": "...", "id": "1"}, ...}
            parsed_sub_questions = json.loads(re.findall(pattern, sub_questions_str)[0])
        except json.JSONDecodeError:
            print(f"Warning: LLM did not return valid JSON for sub-questions: {sub_questions_str}. Using original question as a single sub-question.")
            # Fallback: if not JSON, use original question as a single sub-question
            parsed_sub_questions = {"sub_question_1": {"query": question, "id": "1"}}

        # P1: retrieve fresh, targeted documents per sub-question instead of reusing
        # the original query's docs. Fall back to the originals if no retriever/query.
        final_sub_questions = {}
        for key, subq_data in parsed_sub_questions.items():
            query = subq_data.get("query", "")
            if self.retriever is not None and query:
                contexts = self.retriever.invoke(query)
            else:
                contexts = documents
            final_sub_questions[key] = {
                "query": query,
                "id": subq_data.get("id", key),     # Ensure ID exists, default to key
                "contexts": contexts,
            }

        return {"sub_questions": final_sub_questions, "documents": documents} # Propagate documents

    def retrieve_sub_question_documents(self, state: AdvanceRAGState) -> AdvanceRAGState:
        """
        Placeholder function. In an advanced RAG, this would retrieve documents for each sub-question.
        For this simplified graph, main documents are reused as contexts for sub-questions.
        This function is kept if the graph structure expects it, but its logic isn't modified
        per the current task which focuses on LLM calls.
        """
        # In this setup, sub_questions already have 'contexts' from initial retrieval in plan_sub_steps.
        # This node would typically implement specific retrieval for each sub-question.
        # For the current task, no changes are needed here, as the contexts are already populated
        # in `plan_sub_steps` by reusing the main query's retrieved documents.
        print("---RETRIEVING SUB-QUESTION DOCUMENTS (reusing main documents)---")
        return state # Return state as is, as contexts are already set.


    def generate_answers(self, state: AdvanceRAGState) -> AdvanceRAGState:
        """
        Generates answers for each sub-question using their contexts and LLM.
        Then synthesizes a final answer.
        """
        print("---GENERATING ANSWERS FOR SUB-QUESTIONS AND FINAL ANSWER---")
        question = state["question"]
        sub_questions = state["sub_questions"]
        all_individual_answers = []

        # Prepare inputs for batch generation for each sub-question
        generation_inputs = []
        ordered_sub_question_keys = []

        for key, subq in sub_questions.items():
            context_joined = "\n\n".join([doc.page_content for doc in subq["contexts"]]) if subq["contexts"] else "No specific documents found."
            # Create a simple user message prompt for generation
            prompt_content = f"""Using the following documents:\n{context_joined}\nAnswer the question: \"{subq["query"]}\"."""
            generation_inputs.append([{"role": "user", "content": prompt_content}])
            ordered_sub_question_keys.append(key)

        if generation_inputs:
            # Batch process all sub-question answer generations
            individual_answers_strings = self.llm_to_str.batch(generation_inputs) # List of strings
            individual_answers_strings = [strip_reasoning(a) for a in individual_answers_strings]

            # Store individual answers back into the state and collect for synthesis
            for i, key in enumerate(ordered_sub_question_keys):
                sub_questions[key]["answer"] = individual_answers_strings[i]
                all_individual_answers.append(individual_answers_strings[i])
        else:
            print("No sub-questions to generate answers for.")

        # Synthesize final answer from all sub-answers
        final_answer = ""
        if all_individual_answers:
            combined_sub_answers = "\n\n".join(all_individual_answers)
            synthesis_prompt_content = f"""
            Given the main question: \"{question}\" and the following answers to 
            its sub-questions:\n{combined_sub_answers}\nProvide a comprehensive 
            final answer to the main question. Try to give a consice answer not 
            more than 100 words.
            """
            synthesis_input = [[{"role": "user", "content": synthesis_prompt_content}]]

            final_answer_responses = self.llm_to_str.batch(synthesis_input)
            final_answer = strip_reasoning(final_answer_responses[0]) if final_answer_responses else "Could not synthesize a final answer."
        else:
            final_answer = "No answers were generated for sub-questions."

        return {"sub_questions": sub_questions, "final_answer": final_answer}

    def grade_documents(self, state: AdvanceRAGState) -> AdvanceRAGState:
        """
        Grades each retrieved document for relevance to the main question.
        Filters out irrelevant documents.
        """
        print("--GRADING DOCUMENTS---")
        question = state["question"]
        documents = state["documents"]

        if not documents:
            print("No documents to grade.")
            return {"documents": []}

        # Prepare prompts for batch grading
        grading_inputs = []
        for doc in documents:
            # retrieval_grader_prompt expects 'question' and 'document'
            grading_inputs.append(self.retrieval_grader_prompt.format_messages(question=question, document=doc.page_content))

        # Batch grade documents, expecting string output ('yes' or 'no')
        # batch expects a list of inputs, each being a PromptValue or list of messages
        # Corrected: Removed extra nesting, passing grading_inputs directly
        doc_grades_strings = self.slm_to_str.batch(grading_inputs)

        filtered_docs = []
        for doc_grade_str, doc in zip(doc_grades_strings, documents):
            score = strip_reasoning(doc_grade_str).lower() # Normalize score (drop <think>)
            if "yes" in score:
                filtered_docs.append(doc)
            else:
                print(f"Document graded as irrelevant: {doc.metadata.get('title', doc.metadata.get('source'))}")

        # Fallback (P0): if the grader rejected everything, keep the originally
        # retrieved docs instead of dead-ending the graph with an empty answer.
        if not filtered_docs:
            print("All documents graded irrelevant; falling back to original retrieved documents.")
            filtered_docs = documents

        return {"documents": filtered_docs}

    def grade_generation(self, state: AdvanceRAGState) -> AdvanceRAGState:
        """
        Grades the final generated answer for usefulness to the original question.
        """
        print("---GRADING GENERATION---")
        question = state["question"]
        generation = state["final_answer"]

        # Prepare prompt for batch grading (single input here)
        # generation_grader_prompt expects 'question' and 'generation'
        grading_input_messages = self.generation_grader_prompt.format_messages(question=question, generation=generation)

        # Batch grade generation, expecting string output ('yes' or 'no')
        # Corrected: Removed extra nesting, passing [grading_input_messages] instead of [[grading_input_messages]]
        grade_responses = self.slm_to_str.batch([grading_input_messages])
        grade_str = strip_reasoning(grade_responses[0]).lower() # single response, drop <think>

        # Count this attempt so the graph can cap re-generation (loop guard, P0).
        attempts = state.get("attempts", 0) + 1

        if "yes" in grade_str:
            print("---GENERATION USEFUL---")
            return {"generation_grade": "yes", "attempts": attempts}
        else:
            print(f"---GENERATION NOT USEFUL (attempt {attempts})---")
            return {"generation_grade": "no", "attempts": attempts}
