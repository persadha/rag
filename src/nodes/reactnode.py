from typing import List, Optional
from src.state.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

class RAGNodes:
    """Contains the node functions for RAG workflow"""

    def __init__(self, retriever, llm, use_wikipedia: bool = False):
        self.retriever=retriever
        self.llm=llm
        # Closed-book by default: only the retriever tool. Enable Wikipedia explicitly
        # for open-book use — it lets the agent answer from general knowledge, which
        # contaminates faithfulness/grounding metrics in a closed-corpus eval.
        self.use_wikipedia=use_wikipedia
        self._agent=None

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Classic retriever node"""

        docs=self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )
    
    ## Build tools##

    def _build_tools(self) -> List[Tool]:
        "Build the retriever tool (+ Wikipedia only when use_wikipedia is set)"

        def retriever_tool__fn(query:str)->str:
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found."
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n\n".join(merged)

        retriever_tool=Tool(
            name="retriever",
            description="Fetch passages from indexed vectorstore",
            func=retriever_tool__fn
        )
        tools = [retriever_tool]

        if self.use_wikipedia:
            # Imported lazily so the closed-book path doesn't require the wikipedia deps.
            from langchain_community.utilities import WikipediaAPIWrapper
            from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
            wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="en"))
            tools.append(Tool(
                name="wikipedia",
                description="Search Wikipedia for general knowledge.",
                func=wiki.run,
            ))

        return tools

    
    ## Build agent
    def _build_agent(self):
        """ReAct agent with tools"""

        tools=self._build_tools()
        if self.use_wikipedia:
            system_prompt=(
                "You are a helpful RAG agent. "
                "Prefer 'retriever' for user-provided docs; use 'wikipedia' for general knowledge. "
                "Return only the final useful answer."
            )
        else:
            system_prompt=(
                "You are a helpful RAG agent. Answer ONLY from the 'retriever' tool's passages; "
                "if they do not contain the answer, say you don't know. "
                "Return only the final useful answer."
            )
        self._agent = create_react_agent(self.llm, tools=tools, prompt=system_prompt)

    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate answer using ReAct with retriever * wikipedia
        """

        if self._agent is None:
            self._build_agent()
            
        result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})

        messages = result.get("messages", [])
        answer: Optional[str] = None
        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", None)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate answer."
        )


