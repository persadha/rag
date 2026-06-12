"""Graph builder for the CRAG++ architecture (4th architecture).

Same shape as CRAG (graph_builder_adv) — retrieve, grade, decompose, generate,
grade-generation with a guarded retry — but sub-questions retrieve their own
deduped documents and the synthesis has no word cap. See docs/adr/0001.
"""

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate

from src.state.cragpp_state import CRAGppState
from src.nodes.cragpp_nodes import CRAGppNodes

MAX_GENERATION_ATTEMPTS = 2


class GraphBuilder:
    def __init__(self, retriever, llm, slm):
        self.retriever = retriever
        self.llm = llm
        self.slm = slm
        self.nodes = CRAGppNodes(llm, slm, retriever)  # prompts set in build()
        self.graph = None

    def set_prompts(self, retrieval_grader_prompt, generation_grader_prompt, question_rewriter_prompt):
        self.nodes.retrieval_grader_prompt = retrieval_grader_prompt
        self.nodes.generation_grader_prompt = generation_grader_prompt
        self.nodes.question_rewriter_prompt = question_rewriter_prompt

    def retrieve(self, state: CRAGppState) -> CRAGppState:
        """Initial broad retrieval for the original question."""
        question = state["question"]
        print(f"---RETRIEVING DOCUMENTS FOR: {question}---")
        documents = self.retriever.invoke(question)
        return {"documents": documents}

    def build(self):
        print("---BUILDING CRAG++ GRAPH---")

        retrieval_grader_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a grader agent. Your task is to assess the
                          relevance of retrieved documents to the user's question.
                       Respond with 'yes' if the document is relevant, and 'no' otherwise.
                       Strictly respond with either 'yes' or 'no'.
                       Question: {question}
                       Document: {document}"""),
            ("human", "Is this document relevant to the question?")
        ])

        generation_grader_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a grader agent. Your task is to assess the usefulness of the generated answer to the user's question.
                       Respond with 'yes' if the answer is useful, and 'no' otherwise.
                       Strictly respond with either 'yes' or 'no'.
                       Question: {question}
                       Generated Answer: {generation}"""),
            ("human", "Is this generated answer useful for the question?")
        ])

        question_rewriter_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a question rewriter. Your task is to break down a complex question into 2 to 3 smaller, answerable sub-questions.
                       Respond with a JSON object where keys are sub_question_X (e.g., sub_question_1) and values are objects containing 'query' and 'id'.
                       If the question is simple, return a single sub-question identical to the original.
                       Example: '{{\"sub_question_1\": {{\"query\": \"first sub-question\", \"id\": \"1\"}}, \"sub_question_2\": {{\"query\": \"second sub-question\", \"id\": \"2\"}}}}'
                       Question: {question}"""),
            ("human", "Break down the following question into sub-questions:")
        ])

        self.set_prompts(retrieval_grader_prompt, generation_grader_prompt, question_rewriter_prompt)

        workflow = StateGraph(CRAGppState)

        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.nodes.grade_documents)
        workflow.add_node("plan_sub_steps", self.nodes.plan_sub_steps)
        workflow.add_node("generate_answers", self.nodes.generate_answers)
        workflow.add_node("grade_generation", self.nodes.grade_generation)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            lambda state: "plan_sub_steps" if state["documents"] else "end",
            {"plan_sub_steps": "plan_sub_steps", "end": END}
        )
        workflow.add_edge("plan_sub_steps", "generate_answers")
        workflow.add_edge("generate_answers", "grade_generation")

        # Stop when the answer is useful OR the attempt cap is reached (loop guard)
        workflow.add_conditional_edges(
            "grade_generation",
            lambda state: "stop" if (
                state.get("generation_grade") == "yes"
                or state.get("attempts", 0) >= MAX_GENERATION_ATTEMPTS
            ) else "retry",
            {"stop": END, "retry": "generate_answers"}
        )

        self.graph = workflow.compile()
        print("---CRAG++ GRAPH COMPILED---")
        return self.graph

    def run(self, question: str):
        initial_state = CRAGppState(question=question, documents=[], sub_questions={},
                                    final_contexts=[], final_answer="",
                                    generation_grade="not_useful", attempts=0)
        return self.graph.invoke(initial_state)
