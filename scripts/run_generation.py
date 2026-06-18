"""Generate answers for the r3 evaluation: one (system, generator) pair per run.

Systems:    standard | crag | cragpp        (see context.md §0 for definitions)
Generators: haiku | llama-groq              (see src/config/config_api.py)

Logs retrieved_context as a JSON LIST of chunks for every system — never a
concatenated blob (the blob is what zeroed CRAG's contextual precision/recall
in the r2 pass, context.md §4.3). Output is appended row-by-row, so an
interrupted run resumes where it stopped.

Usage:
    python scripts/run_generation.py --system cragpp --generator haiku --rows 15
    python scripts/run_generation.py --system standard --generator llama-groq   # full dataset
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings

from src.config.config import Config
from src.config.config_api import APIConfig
from src.vectorstore.vectorstore import VectorStore

SYSTEMS = ("standard", "crag", "cragpp")
RETRIEVER_K = 4  # parity with the original runs (old get_retriever ignored k; Chroma default = 4)


def build_graph(system: str, retriever, llm, slm):
    if system == "standard":
        from src.graph_builder.graph_builder import GraphBuilder
        builder = GraphBuilder(retriever=retriever, llm=llm)
    elif system == "crag":
        from src.graph_builder.graph_builder_adv import GraphBuilder
        builder = GraphBuilder(retriever=retriever, llm=llm, slm=slm)
    elif system == "cragpp":
        from src.graph_builder.graph_builder_cragpp import GraphBuilder
        builder = GraphBuilder(retriever=retriever, llm=llm, slm=slm)
    else:
        raise ValueError(f"Unknown system '{system}'")
    builder.build()
    return builder


def extract(system: str, result: dict):
    """Return (answer, list_of_context_chunks) from a graph's final state."""
    if system == "standard":
        answer = result.get("answer", "")
        docs = result.get("retrieved_docs", [])
    elif system == "crag":
        answer = result.get("final_answer", "")
        docs = result.get("documents", [])  # sub-questions reuse these (frozen design)
    else:  # cragpp
        answer = result.get("final_answer", "")
        docs = result.get("final_contexts") or result.get("documents", [])
    return (answer or "").strip(), [d.page_content for d in docs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=SYSTEMS)
    parser.add_argument("--generator", required=True, choices=APIConfig.GENERATOR_NAMES)
    parser.add_argument("--rows", type=int, default=None, help="limit to first N rows (pilot)")
    parser.add_argument("--dataset",
                        default=str(ROOT / "datasets" / "revision" / "evaluation_dataset.xlsx"))
    parser.add_argument("--sheet", default="final")
    parser.add_argument("--persist-dir", default=str(ROOT / "chroma_db"))
    parser.add_argument("--delay", type=float, default=0.0,
                        help="seconds to sleep between rows (rate-limit pacing)")
    args = parser.parse_args()

    df = pd.read_excel(args.dataset, sheet_name=args.sheet)
    # Revision dataset schema bridge: map the new column names to what the pipeline
    # expects, and keep `id` as a stable, resume-safe row identifier.
    df = df.rename(columns={"user_query": "question", "reference_answer": "answer_ref"})
    for col in ("id", "question", "answer_ref", "reference_context", "data_type"):
        if col not in df.columns:
            sys.exit(f"Dataset sheet '{args.sheet}' has no '{col}' column; found: {list(df.columns)}")
    if args.rows:
        df = df.head(args.rows)

    embeddings = HuggingFaceEmbeddings(model_name=Config.DEFAULT_EMBEDDING_MODEL)
    store = VectorStore(embeddings, persist_directory=args.persist_dir)
    store.load_vectorstore()
    if store.vectorstore is None:
        sys.exit(f"No index at {args.persist_dir} — run scripts/build_index.py first.")
    retriever = store.get_retriever(k=RETRIEVER_K)

    llm = APIConfig.get_generator(args.generator)
    # API generators grade with the same model; local Ollama runs grade with the
    # small gemma3:1b grader for parity with the original CRAG design.
    slm = APIConfig.get_ollama_slm() if args.generator == "ollama" else llm
    graph = build_graph(args.system, retriever, llm, slm)

    out_path = ROOT / "results" / f"gen_r3_{args.system}_{args.generator}.csv"
    out_path.parent.mkdir(exist_ok=True)
    done = set()
    if out_path.exists():
        done = set(pd.read_csv(out_path, usecols=["row_id"])["row_id"])
        print(f"Resuming: {len(done)} rows already in {out_path.name}")

    model_id = {"haiku": APIConfig.ANTHROPIC_GENERATOR_MODEL,
                "llama-groq": APIConfig.OPENSOURCE_GENERATOR_MODEL,
                "ollama": APIConfig.OLLAMA_GENERATOR_MODEL}[args.generator]
    empty, latencies = 0, []
    for _, row in df.iterrows():
        row_id = int(row["id"])  # stable id from the dataset, not the positional index
        if row_id in done:
            continue
        t0 = time.time()
        result = graph.run(str(row["question"]))
        latency = time.time() - t0
        answer, chunks = extract(args.system, result)
        if not answer:
            empty += 1
            print(f"  WARNING row {row_id}: empty answer")
        record = pd.DataFrame([{
            "row_id": row_id,
            "data_type": row["data_type"],
            "question": row["question"],
            "answer_ref": row["answer_ref"],
            "reference_context": row["reference_context"],
            "answer": answer,
            "retrieved_context": json.dumps(chunks, ensure_ascii=False),
            "n_chunks": len(chunks),
            "latency_s": round(latency, 2),
            "system": args.system,
            "generator": args.generator,
            "model": model_id,
        }])
        record.to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
        latencies.append(latency)
        print(f"row {row_id}: {latency:.1f}s, {len(chunks)} chunks, answer {len(answer)} chars")
        if args.delay:
            time.sleep(args.delay)

    n = len(latencies)
    if n:
        print(f"\nDONE {args.system}/{args.generator}: {n} new rows, "
              f"{empty} empty answers, mean latency {sum(latencies) / n:.1f}s -> {out_path}")
    else:
        print("Nothing to do (all rows already generated).")


if __name__ == "__main__":
    main()
