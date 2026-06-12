
import os
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

class VectorStore:
    def __init__(self, embedding_model, persist_directory: str = "/content/drive/MyDrive/RAG/chroma_db", collection_name: str = "rag_collection"):
        self.embeddings = embedding_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vectorstore = None

    def create_vectorstore(self, documents: List[Document]):
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        print(f"Chroma vector store created and persisted to {self.persist_directory}")

    def save_vectorstore(self):
        if self.vectorstore:
            # Chroma with a persist_directory auto-persists; older versions exposed
            # an explicit .persist(). Call it only when present.
            if hasattr(self.vectorstore, "persist"):
                self.vectorstore.persist()
            print(f"Vector store data ensured to be saved in {self.persist_directory}")
        else:
            print("No vector store to save.")

    def load_vectorstore(self):
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            print(f"Loading vector store from {self.persist_directory}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            print("Vector store loaded successfully.")
        else:
            print(f"Vector store not found at {self.persist_directory}. A new one will be created if documents are processed.")
            self.vectorstore = None

    def load_or_create_vectorstore(self, documents: List[Document]):
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            self.load_vectorstore()
        else:
            self.create_vectorstore(documents)

    def add_documents(self, documents: List[Document]):
        if self.vectorstore:
            self.vectorstore.add_documents(documents)
            if hasattr(self.vectorstore, "persist"):
                self.vectorstore.persist()
            print(f"Added {len(documents)} documents to the vector store and persisted.")
        else:
            print("Vector store not initialized. Creating a new one with the provided documents.")
            self.create_vectorstore(documents)

    def get_retriever(self, k: int = 5):
        if self.vectorstore:
            return self.vectorstore.as_retriever(search_kwargs={"k": k})
        else:
            raise ValueError("Vector store not initialized.")
