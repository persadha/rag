# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings # Added this import
import os

class Config:
    DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "llama3:8b")
    DEFAULT_SLM_MODEL = os.getenv("SLM_MODEL", "gemma3:1b")
    LLAMA3_MODEL = os.getenv("LLAMA3_MODEL", "llama3:8b")
    GEMMA3_MODEL = os.getenv("GEMMA3_MODEL", "gemma3:4b")
    GPT_OSS_MODEL = os.getenv("GPT_OSS_MODEL", "gpt-oss:20b")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-r1:8b")
    DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")  # 768-dim; rebuild chroma_db after switching (index dim mismatch)
    # DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
    # Where the Ollama server lives. Default is the local daemon; inside Docker
    # Compose this is set to http://ollama:11434 so the app reaches the service.
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    

    # Default URLs and TXTs (replace with your actual data if needed)
    DEFAULT_URLS = []
    DEFAULT_TXTS = []

    @staticmethod
    def get_llm(model_name: str = DEEPSEEK_MODEL):
        return OllamaLLM(model=model_name)

    @staticmethod
    def get_slm(model_name: str = DEFAULT_SLM_MODEL):
        return OllamaLLM(model=model_name)

    @staticmethod
    def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
        return HuggingFaceEmbeddings(model=model_name)

    @staticmethod
    def get_chunk_size():
        return Config.DEFAULT_CHUNK_SIZE

    @staticmethod
    def get_chunk_overlap():
        return Config.DEFAULT_CHUNK_OVERLAP
