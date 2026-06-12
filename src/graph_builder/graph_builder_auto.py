"""Graph builder for LangGraph workflow"""

from langgraph.graph import StateGraph, END
from src.state.autorag_state import AutoRAGState
from src.nodes.autorag_nodes import AutoRAGNodes

class GraphBuilder:
    """Builds and manages the LangGraph workflow."""

    def __init__(self, retriever, llm, slm):
        """
        Initializes the GraphBuilder

        Args:
            retriever: Document retriever
            llm: Language model
            slm: Small language model
        """

        self.nodes = AutoRAGNodes(retriever, llm, slm)
        self.graph = None


    def build(self):
        """
        Build the LangGraph workflow
        """
          
        # ---- Graph wiring ----
        builder = StateGraph(AutoRAGState)
        builder.add_node("planner", self.nodes.plan_query)          # Query Planning and Decomposition
        builder.add_node("sub_planner", self.nodes.plan_sub_steps)  # Sub-steps planning
        builder.add_node("retriever", self.nodes.retrieve_per_sub_step)
        builder.add_node("responder", self.nodes.generate_answers)
        builder.add_node("synthesizer", self.nodes.synthesize_answer)

        builder.set_entry_point("planner")
        builder.add_edge("planner", "sub_planner")
        builder.add_edge("sub_planner", "retriever")
        builder.add_edge("retriever", "responder")
        builder.add_edge("responder", "synthesizer")
        builder.add_edge("synthesizer", END)
        self.graph = builder.compile()



        return self.graph
    

    def run(self, question: str) -> dict:
        """
        Run the RAG workflow
        """

        if self.graph is None:
            self.build()

        initial_state = AutoRAGState(question=question)
        return self.graph.invoke(initial_state)    