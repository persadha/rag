"""LangGraph nodes for RAG workflow"""

from src.state.rag_state import RAGState

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

        # Create prompt
        prompt = f"""You are a helpful assistant. Use ONLY the context below to answer.
            Before you respond to any query please walk the user through your thought process step by step.
            If the answer is not in the context, say: 'Sorry, I do not know the answer.


        Context:
        {context}

        Question: {state.question}
        """
        
        # Generate response
        response = self.llm.invoke(prompt)

        # Use conversational memory?

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=response
        )

