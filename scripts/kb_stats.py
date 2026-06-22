"""Knowledge-base scale stats for the reviewer response (R2 §Knowledge Base Scale).

Reports the number of indexed chunks, token/word volume, source-document spread,
chunk-length distribution, the embedding model + dimensionality, and the on-disk
footprint of the Chroma index. Reads the Chroma collection directly (no embedding
model loaded), so it's fast and free.

Usage:
    python scripts/kb_stats.py
    python scripts/kb_stats.py --persist-dir chroma_db --collection rag_collection
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chromadb

from src.config.config import Config


def dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024


def count_tokens(texts):
    """Prefer tiktoken (cl100k_base); fall back to a chars/4 heuristic."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return sum(len(enc.encode(t)) for t in texts), "tiktoken/cl100k_base"
    except Exception:
        return sum(max(1, len(t) // 4) for t in texts), "heuristic (chars/4)"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--persist-dir", default=str(ROOT / "chroma_db"))
    ap.add_argument("--collection", default="rag_collection")
    args = ap.parse_args()

    persist = Path(args.persist_dir)
    if not persist.exists():
        sys.exit(f"No Chroma dir at {persist}")

    client = chromadb.PersistentClient(path=str(persist))
    col = client.get_collection(args.collection)
    n = col.count()
    data = col.get(include=["documents", "metadatas"])
    docs = [d or "" for d in data["documents"]]
    metas = data["metadatas"] or []

    char_lens = [len(d) for d in docs]
    word_lens = [len(d.split()) for d in docs]
    total_chars = sum(char_lens)
    total_words = sum(word_lens)
    n_tokens, tok_method = count_tokens(docs)

    # source documents (best-effort: common metadata keys)
    src_keys = ("source", "file_name", "filename", "document", "doc", "path")
    sources = set()
    key_used = None
    for m in metas:
        if not m:
            continue
        for k in src_keys:
            if k in m and m[k]:
                sources.add(str(m[k]))
                key_used = key_used or k
                break

    def pct(vals, p):
        s = sorted(vals)
        return s[min(len(s) - 1, int(p / 100 * len(s)))] if s else 0

    print("=" * 60)
    print("KNOWLEDGE BASE STATS  (R2 §Knowledge Base Scale)")
    print("=" * 60)
    print(f"Persist dir        : {persist}")
    print(f"Collection         : {args.collection}")
    print(f"Embedding model    : {Config.DEFAULT_EMBEDDING_MODEL}")
    print(f"Indexed chunks     : {n:,}")
    if sources:
        print(f"Source documents   : {len(sources)} (metadata key '{key_used}')")
    else:
        print("Source documents   : (no source metadata found on chunks)")
    print(f"Total words        : {total_words:,}")
    print(f"Total characters   : {total_chars:,}")
    print(f"Total tokens       : {n_tokens:,}  [{tok_method}]")
    if n:
        print(f"Mean chunk         : {total_words / n:.0f} words / "
              f"{total_chars / n:.0f} chars / {n_tokens / n:.0f} tokens")
        print(f"Chunk words p50/p90/max: {pct(word_lens,50)}/{pct(word_lens,90)}/{max(word_lens)}")
    print(f"On-disk footprint  : {human(dir_size_bytes(persist))}")
    print("=" * 60)


if __name__ == "__main__":
    main()
