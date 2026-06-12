"""Streamlit UI for Agentic RAG System"""

import streamlit as st
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder_adv import GraphBuilder

# Page configuration
st.set_page_config(
    page_title="RAG Search",
    page_icon="RAG",
    layout="centered"
)

# Simple CS
st.markdown("""
    <style>
            .stButton > button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            }
    </style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""

if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None

if "rag_system" in st.session_state:
    del st.session_state["rag_system"]

if 'initialized' not in st.session_state:
    st.session_state.initialized = False

if 'history' not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def initialize_rag():
    """Intialized the RAG system (cached)"""
    try:
        # Initialized components
        llm = Config.get_llm()

        slm = Config.get_slm()
        embedding = Config.get_embedding_model()
        doc_processor = DocumentProcessor(embedding)
        vector_store = VectorStore(embedding)

        # Use default URLs
        urls = Config.DEFAULT_URLS
        #pdfs = Config.PDF_PATHS

        # Process documents
        documents = doc_processor.process_urls(urls)
        #documents = doc_processor.process_pdf(pdfs)

        # Create vector store
        vector_store.create_retriever(documents)

        # Build graph
        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm,
            slm=slm
        )
        graph_builder.build()

        return graph_builder, len(documents)
    except Exception as e:
        st.error(f"Failed to intialized: {str(e)}")
        return None, 0
    
def main():
    """Main application"""
    init_session_state()

    # Title
    st.title("PIRLS Document Search")
    st.markdown("Ask question about the loaded documents.")

    # Initialized system
    if not st.session_state.initialized:
        with st.spinner("Loading system..."):
            rag_system, num_chunks = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"System ready! ({num_chunks} document chunks loaded)")
    st.markdown("---")

    # Search interface
    with st.form("search_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know?"
        )
        submit = st.form_submit_button("Search")

    # Process search
    if submit and question:
        if st.session_state.rag_system:
            with st.spinner("Searching..."):
                start_time = time.time()

                # Get answer
                result = st.session_state.rag_system.run(question)

                elapsed_time = time.time() - start_time

                # Add to history
                st.session_state.history.append({
                    'question': question,
                    'answer': result['final_answer'],
                    'time': elapsed_time
                })

                # Display answer
                st.markdown("### Answer")
                st.success(result['final_answer'])

                # Show retrieved docs in expander
                with st.expander("Source Documents"):
                    for i, doc in enumerate(result['retrieved_docs'], 1):
                        st.text_area(
                            f"Document {i}",
                            #doc.page_content[:3000] + "...",
                            doc.page_content,
                            height=100,
                            disabled=True
                        )

                st.caption(f"Response time: {elapsed_time:.2f} seconds")

    # Show history
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### Recent Searches")

        for item in reversed(st.session_state.history[-3:]):
            with st.container():
                st.markdown(f"Question: {item['question']}")
                st.markdown(f"Answer: {item['answer'][:100]}...")
                st.caption(f"Time: {item['time']:.2f}s")
                st.markdown("")

if __name__ == "__main__":
    main()