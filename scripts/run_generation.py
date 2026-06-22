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
import os
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

# Closed-book baseline (reviewer value-added ablation): answer from the model's own
# knowledge, no retrieval. Minimal, neutral prompt so the comparison isolates retrieval.
NO_RETRIEVAL_PROMPT = (
    "Answer the following question as accurately and concisely as you can, using only "
    "your own knowledge. If you are unsure, give your best answer.\n\n"
    "Question: {q}\n\nAnswer:"
)


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
    parser.add_argument("--rerank", action="store_true",
                        help="Tier-1: retrieve --candidates then cross-encoder rerank to k=4 "
                             "(writes to a *_rerank.csv so the baseline is untouched)")
    parser.add_argument("--candidates", type=int, default=20,
                        help="first-stage candidate pool size when --rerank/--hybrid is set (k per retriever)")
    parser.add_argument("--hybrid", action="store_true",
                        help="Tier-1.1: BM25 + dense retrieval fused with RRF, then cross-encoder "
                             "rerank to k=4 (writes to a *_hybrid.csv). Implies reranking.")
    parser.add_argument("--reranker-model", default=None,
                        help="cross-encoder model for --rerank/--hybrid (default: ms-marco-MiniLM-L-6-v2; "
                             "e.g. BAAI/bge-reranker-base)")
    parser.add_argument("--tag", default="",
                        help="extra suffix for the output filename (e.g. chunk512) to keep A/B runs separate")
    parser.add_argument("--prompt-style", choices=["baseline", "extract"], default="baseline",
                        help="standard-RAG generation prompt variant (E8 A/B); 'extract' = few-shot extraction prompt")
    parser.add_argument("--no-retrieval", action="store_true",
                        help="closed-book baseline: skip retrieval entirely and answer each question "
                             "directly from the generator (writes gen_r3_noretrieval_<generator>.csv). "
                             "For the reviewer value-added ablation.")
    args = parser.parse_args()
    os.environ["GEN_PROMPT_STYLE"] = args.prompt_style  # read by RAGNodes.generate_answer

    df = pd.read_excel(args.dataset, sheet_name=args.sheet)
    # Revision dataset schema bridge: map the new column names to what the pipeline
    # expects, and keep `id` as a stable, resume-safe row identifier.
    df = df.rename(columns={"user_query": "question", "reference_answer": "answer_ref"})
    for col in ("id", "question", "answer_ref", "reference_context", "data_type"):
        if col not in df.columns:
            sys.exit(f"Dataset sheet '{args.sheet}' has no '{col}' column; found: {list(df.columns)}")
    if args.rows:
        df = df.head(args.rows)

    if args.no_retrieval:
        # Closed-book: no index, no retriever, no graph — just the bare generator.
        llm = APIConfig.get_generator(args.generator)
        graph = None
        print(f"Closed-book ON: no retrieval; {args.generator} answers from its own knowledge.")
    else:
        embeddings = HuggingFaceEmbeddings(model_name=Config.DEFAULT_EMBEDDING_MODEL)
        store = VectorStore(embeddings, persist_directory=args.persist_dir)
        store.load_vectorstore()
        if store.vectorstore is None:
            sys.exit(f"No index at {args.persist_dir} — run scripts/build_index.py first.")
        rr_kwargs = {"top_k": RETRIEVER_K}
        if args.reranker_model:
            rr_kwargs["model_name"] = args.reranker_model
        if args.hybrid:
            from src.vectorstore.hybrid import HybridRetriever, build_bm25_corpus
            from src.vectorstore.rerank import RerankRetriever
            corpus = build_bm25_corpus(store)
            base = HybridRetriever(store.get_retriever(k=args.candidates), corpus, k_each=args.candidates)
            retriever = RerankRetriever(base, **rr_kwargs)
            print(f"Hybrid ON: BM25+dense (k_each={args.candidates}, corpus={len(corpus)}) "
                  f"-> RRF -> {rr_kwargs.get('model_name', 'MiniLM')} -> top-{RETRIEVER_K}")
        elif args.rerank:
            from src.vectorstore.rerank import RerankRetriever
            retriever = RerankRetriever(store.get_retriever(k=args.candidates), **rr_kwargs)
            print(f"Reranking ON: top-{args.candidates} candidates -> "
                  f"{rr_kwargs.get('model_name', 'MiniLM')} -> top-{RETRIEVER_K}")
        else:
            retriever = store.get_retriever(k=RETRIEVER_K)

        llm = APIConfig.get_generator(args.generator)
        # API generators grade with the same model; local Ollama runs grade with the
        # small gemma3:1b grader for parity with the original CRAG design.
        slm = APIConfig.get_ollama_slm() if args.generator == "ollama" else llm
        graph = build_graph(args.system, retriever, llm, slm)

    # Closed-book runs are labelled by "noretrieval" instead of the (irrelevant) system,
    # so they group cleanly in the eval and never collide with retrieval runs.
    sys_label = "noretrieval" if args.no_retrieval else args.system
    suffix = ("_rerank" if args.rerank else "") + ("_hybrid" if args.hybrid else "") \
             + (f"_{args.tag}" if args.tag else "") \
             + (f"_{args.prompt_style}" if args.prompt_style != "baseline" else "")
    out_path = ROOT / "results" / f"gen_r3_{sys_label}_{args.generator}{suffix}.csv"
    out_path.parent.mkdir(exist_ok=True)
    done = set()
    if out_path.exists():
        done = set(pd.read_csv(out_path, usecols=["row_id"])["row_id"])
        print(f"Resuming: {len(done)} rows already in {out_path.name}")

    model_id = APIConfig.model_id_for(args.generator)
    empty, failed, latencies = 0, 0, []
    for _, row in df.iterrows():
        row_id = int(row["id"])  # stable id from the dataset, not the positional index
        if row_id in done:
            continue
        t0 = time.time()
        try:
            # One row must never take down an unattended run: a hung/timed-out
            # generation is logged and skipped (left un-done so a resume retries it).
            if args.no_retrieval:
                answer = (llm.invoke(NO_RETRIEVAL_PROMPT.format(q=str(row["question"]))) or "").strip()
                chunks = []
            else:
                result = graph.run(str(row["question"]))
                answer, chunks = extract(args.system, result)
        except Exception as exc:
            failed += 1
            print(f"  row {row_id} FAILED ({time.time()-t0:.0f}s): {type(exc).__name__}: {str(exc)[:120]}")
            continue
        latency = time.time() - t0
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
            "system": sys_label,
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
        print(f"\nDONE {sys_label}/{args.generator}: {n} new rows, "
              f"{empty} empty answers, {failed} failed/skipped, "
              f"mean latency {sum(latencies) / n:.1f}s -> {out_path}")
        if failed:
            print(f"  {failed} rows skipped on error — re-run the same command to retry them.")
    else:
        print(f"Nothing to do (all rows already generated). {failed} failed/skipped this pass.")


if __name__ == "__main__":
    main()
