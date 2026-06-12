
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.documents = []

    def load_documents(self, file_path: str):
        if os.path.isdir(file_path):
            for filename in os.listdir(file_path):
                if filename.endswith('.pdf'):
                    full_file_path = os.path.join(file_path, filename)
                    if os.path.isfile(full_file_path):
                        loader = PyPDFLoader(full_file_path)
                        self.documents.extend(loader.load())
        elif os.path.isfile(file_path) and file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            self.documents.extend(loader.load())
        else:
            raise ValueError(f"Invalid file_path: {file_path}. Must be a PDF file or a directory containing PDFs.")

    def split_documents(self, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        self.documents = text_splitter.split_documents(self.documents)
        return self.documents

    def process(self, file_path: str):
        self.load_documents(file_path)
        self.split_documents()
        return self.documents
