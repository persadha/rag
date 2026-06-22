"""LangGraph nodes for RAG workflow"""

import os

from src.state.rag_state import RAGState
from src.utils.text import strip_reasoning

# Extraction-focused prompt variant (E8 prompt A/B). Selected via GEN_PROMPT_STYLE=extract.
# Few-shot examples are GENERIC (not from the eval set) to avoid answer leakage.
_EXTRACT_PROMPT = """You are a precise question-answering assistant. Using ONLY the context, reply with the exact fact(s) that directly answer the question — as briefly as possible (often just a few words). Do not explain, restate the question, or add detail beyond what is asked. If the answer is not in the context, reply exactly: "Sorry, I do not know the answer."

Example
Context: The national reading assessment is administered every five years, most recently in 2021.
Question: How often is the national reading assessment administered?
Answer: Every five years

Example
Context: The committee met in Geneva to review the curriculum framework.
Question: What was the total programme budget?
Answer: Sorry, I do not know the answer

Now answer.
Context:
{context}

Question: {question}

Answer:"""

_BASELINE_PROMPT = """You are a helpful assistant. Answer the question using ONLY the context below.
Answer directly and concisely. If the answer is not in the context, reply exactly: "Sorry, I do not know the answer."

Context:
{context}

Question: {question}

Answer:"""

class RAGNodes:
    """Contains node function for RAG workflow"""

    def __init__(self, retriever, llm):
        """
        Initializes RAG nodes
        
        """
        self.retriever = retriever
        self.llm = llm

    def retrieve_docs(self, state:RAGState)-> RAGState:
        """
        Retrieve relevant document nodes
        """
        docs=self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )
    
    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate answer from retrieved document nodes
        """

        # Combine retrieved documents into context
        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])

        # Prompt variant: baseline (default) or extract (E8 A/B), via GEN_PROMPT_STYLE env.
        template = _EXTRACT_PROMPT if os.getenv("GEN_PROMPT_STYLE") == "extract" else _BASELINE_PROMPT
        prompt = template.format(context=context, question=state.question)

        # Generate response; strip any <think> reasoning (e.g. deepseek-r1).
        response = self.llm.invoke(prompt)
        answer = strip_reasoning(response)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer
        )

