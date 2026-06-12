"""LangGraph nodes for RAG workflow"""

from src.state.autorag_state import AutoRAGState, SubQuery
from typing import Dict, List, TypedDict
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from src.utils.text import strip_reasoning

class AutoRAGNodes:
    """Contains node function for RAG workflow"""

    def __init__(self, retriever, llm, slm):
        """
        Initializes RAG nodes

        Args:
            retriever: Document retriever
            llm: Language model
        """
        self.retriever = retriever
        self.llm = llm
        self.slm = slm
        # String-output chain so nodes work whether `llm` is a completion model
        # (OllamaLLM -> str) or a chat model (ChatOllama -> AIMessage). Avoids the
        # `.content` AttributeError OllamaLLM would otherwise raise.
        self.llm_to_str = self.llm | StrOutputParser()


    def plan_query(self, state: AutoRAGState) -> AutoRAGState:
        prompt = (
            "You are an expert at breaking down complex questions. "
            f'Given the question: "{state.question}", decompose it into 2-3 specific sub-questions '
            "that will help in answering the main question.\n\n"
            "Answer in the following format:\n"
            "Q1: <first sub-question>\nQ2: <second sub-question>\nQ3: <third sub-question>"
        )
        response = strip_reasoning(self.llm_to_str.invoke(prompt))
        temp_questions = [line.strip() for line in response.splitlines() if line.strip()]

        sub_qs: Dict[str, SubQuery] = {}
        for q in temp_questions:
            # Accept lines like "Q1: ..." / "Q2: ..." (len(key)==2 -> 'Q1','Q2',etc.)
            if ":" in q:
                key, val = q.split(":", 1)
                key = key.strip()
                if len(key) == 2 and key[0].upper() == "Q" and key[1].isdigit():
                    sub_qs[key] = {
                        "query": val.strip(),
                        "contexts": [],
                        "answer": "",
                        "steps": [],
                    }

        # Fallback (P0): if nothing parsed, treat the whole question as a single
        # sub-question rather than leaving sub_questions empty (-> empty final answer).
        if not sub_qs:
            print("No sub-questions parsed; falling back to the original question.")
            sub_qs["Q1"] = {"query": state.question, "contexts": [], "answer": "", "steps": []}

        return state.model_copy(update={"sub_questions": sub_qs})


    def plan_sub_steps(self, state: AutoRAGState) -> AutoRAGState:
        sub_steps_dict: Dict[str, List[str]] = {}

        for key, subq in state.sub_questions.items():
            prompt = (
                "You are an expert reasoner. "
                f'Given the sub question: "{subq["query"]}", break it down into 3 smaller sub-steps to answer it effectively.\n\n'
                "Answer in the following format:\n"
                "**Step 1:** <first step>\n**Step 2:** <second step>\n**Step 3:** <third step>"
            )
            response = strip_reasoning(self.llm_to_str.invoke(prompt))
            # Robust parsing: accept lines that start with **Step
            lines = [ln.strip() for ln in response.splitlines()]
            steps = []
            for ln in lines:
                if ln.lstrip().startswith("**Step"):
                    # remove the leading marker and colon, keep the content
                    parts = ln.split(":", 1)
                    if len(parts) == 2:
                        steps.append(parts[1].strip())
                    else:
                        # fallback: keep the whole line if parsing fails
                        steps.append(ln.replace("**", "").strip())

            # fallback if nothing parsed
            if not steps:
                steps = [response.strip()]

            sub_steps_dict[key] = steps
            state.sub_questions[key]["steps"] = steps  # keep in sync

        return state.model_copy(update={"sub_questions": state.sub_questions, "sub_steps": sub_steps_dict})


    def retrieve_per_sub_step(self, state: AutoRAGState) -> AutoRAGState:
        for key, steps in state.sub_steps.items():
            all_retrieved: List[Document] = []
            for step in steps:
                docs = self.retriever.invoke(step)
                all_retrieved.extend(docs)
            state.sub_questions[key]["contexts"] = all_retrieved

        return state.model_copy(update={"sub_questions": state.sub_questions})

    
    def generate_answers(self, state: AutoRAGState) -> AutoRAGState:
        answers: List[str] = []
        for key, subq in state.sub_questions.items():
            context_joined = "\n\n".join([doc.page_content for doc in subq["contexts"]])
            prompt = f"""Using the following documents:\n{context_joined}\nAnswer the question: "{subq["query"]}"."""
            response = strip_reasoning(self.llm_to_str.invoke(prompt))
            answers.append(response)
            state.sub_questions[key]["answer"] = response  # store individual answer

        return state.model_copy(update={"answers": answers})


    def synthesize_answer(self, state: AutoRAGState) -> AutoRAGState:
        answers_content = "\n\n".join(state.answers)
        prompt = f"""Given the following answers to sub-questions:\n{answers_content}\n
        Synthesize a comprehensive answer to the main question: "{state.question}".
                    Give the answer in a clear and concise manner."""
        response = strip_reasoning(self.llm_to_str.invoke(prompt))

        return state.model_copy(update={"final_answer": response})

