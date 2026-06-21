# PIRLS RAG UI — app image. Pairs with docker-compose.yml (app + Ollama service).
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake the HF models so the first query is instant and the image runs offline:
#   - all-mpnet-base-v2          (embeddings, ~420 MB)
#   - ms-marco-MiniLM-L-6-v2     (cross-encoder reranker, ~80 MB)
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-mpnet-base-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# App code, package, and both prebuilt indexes (~162 MB).
COPY src/ ./src/
COPY pirls_rag_ui.py ./
COPY chroma_db/ ./chroma_db/
COPY chroma_db_512/ ./chroma_db_512/

EXPOSE 8501

# Local generators/graders reach Ollama via OLLAMA_BASE_URL (set in compose to
# http://ollama:11434). API generators read keys from the env_file (.env).
CMD ["streamlit", "run", "pirls_rag_ui.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
