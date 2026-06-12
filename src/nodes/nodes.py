"""LangGraph nodes for RAG workflow"""

from src.state.rag_state import RAGState
from src.utils.text import strip_reasoning

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

        # Create prompt — ask for a direct, concise answer (no forced step-by-step
        # chain-of-thought, which previously polluted the answer field).
        prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
Answer directly and concisely. If the answer is not in the context, reply exactly: "Sorry, I do not know the answer."

Context:
{context}

Question: {state.question}

Answer:"""

        # Generate response; strip any <think> reasoning (e.g. deepseek-r1).
        response = self.llm.invoke(prompt)
        answer = strip_reasoning(response)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer
        )

