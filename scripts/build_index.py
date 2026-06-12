"""Rebuild the Chroma index from the PIRLS PDFs with the configured embeddings.

The old chroma_db was indexed with all-MiniLM-L6-v2 (384-dim) while the config
now defaults to all-mpnet-base-v2 (768-dim) — mixing them breaks retrieval, so
the index must be rebuilt once. Runs on CPU; ~10–25 min for the full corpus.

Usage:
    python scripts/build_index.py [--datasets-dir datasets] [--persist-dir chroma_db]
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_huggingface import HuggingFaceEmbeddings

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore

ADD_BATCH = 1000  # Chroma rejects very large single batches; add in chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", default=str(ROOT / "datasets"))
    parser.add_argument("--persist-dir", default=str(ROOT / "chroma_db"))
    parser.add_argument("--collection", default="rag_collection")
    parser.add_argument("--chunk-size", type=int, default=Config.DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=Config.DEFAULT_CHUNK_OVERLAP)
    args = parser.parse_args()

    if Path(args.persist_dir).exists() and any(Path(args.persist_dir).iterdir()):
        sys.exit(f"{args.persist_dir} already exists and is not empty — delete it first to rebuild.")

    print(f"Embedding model: {Config.DEFAULT_EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=Config.DEFAULT_EMBEDDING_MODEL)
    dim = len(embeddings.embed_query("dimension probe"))
    print(f"Embedding dimension: {dim}")

    t0 = time.time()
    processor = DocumentProcessor(embeddings)
    processor.load_documents(args.datasets_dir)
    print(f"Loaded {len(processor.documents)} PDF pages in {time.time() - t0:.0f}s")

    chunks = processor.split_documents(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    print(f"Split into {len(chunks)} chunks (size={args.chunk_size}, overlap={args.chunk_overlap})")

    t0 = time.time()
    store = VectorStore(embeddings, persist_directory=args.persist_dir, collection_name=args.collection)
    store.create_vectorstore(chunks[:ADD_BATCH])
    for i in range(ADD_BATCH, len(chunks), ADD_BATCH):
        store.vectorstore.add_documents(chunks[i:i + ADD_BATCH])
        done = min(i + ADD_BATCH, len(chunks))
        elapsed = time.time() - t0
        print(f"  embedded {done}/{len(chunks)} chunks "
              f"({elapsed:.0f}s, ~{elapsed / done * (len(chunks) - done):.0f}s left)")
    print(f"Index built in {time.time() - t0:.0f}s")

    # Sanity checks
    count = store.vectorstore._collection.count()
    hits = store.get_retriever(k=4).invoke("What is PIRLS 2021?")
    print(f"Collection count: {count}")
    print(f"Sanity query returned {len(hits)} chunks; first chunk starts with:")
    print("  " + hits[0].page_content[:200].replace("\n", " "))
    assert count == len(chunks), "collection count != number of chunks"
    assert hits, "sanity query returned nothing"
    print("BUILD OK")


if __name__ == "__main__":
    main()
