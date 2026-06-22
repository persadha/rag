"""Retrieval-only baseline (reviewer R2 §Value Added by Generation): emit a
gen-shaped CSV where the "answer" is just the retrieved chunks concatenated — no
LLM synthesis. Scoring this on answer_correctness and comparing to the full RAG
run (same retrieved chunks) isolates what the generator adds over reading chunks.

Retrieval is generator-independent, so the chunks in the best full-RAG run (E11:
hybrid+bge) ARE the chunks a reader would see. We transform that run so the
value-added comparison is on identical retrieval.

Usage:
    python scripts/make_retrieval_only.py \\
        --gen results/gen_r3_standard_openai-mini_hybrid_bgebase.csv \\
        --tag hybrid_bgebase
-> results/gen_r3_retrievalonly_<tag>.csv
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen", required=True, help="source full-RAG generation CSV (best retrieval config)")
    ap.add_argument("--tag", default="hybrid", help="suffix for the output filename")
    ap.add_argument("--sep", default="\n\n", help="separator joining chunks into the pseudo-answer")
    args = ap.parse_args()

    src = Path(args.gen)
    df = pd.read_csv(src)
    out_rows = []
    empty = 0
    for _, row in df.iterrows():
        chunks = json.loads(row["retrieved_context"]) if pd.notna(row["retrieved_context"]) else []
        answer = args.sep.join(str(c) for c in chunks)
        if not answer:
            empty += 1
        out_rows.append({
            "row_id": row["row_id"],
            "data_type": row.get("data_type", None),
            "question": row["question"],
            "answer_ref": row["answer_ref"],
            "reference_context": row.get("reference_context", ""),
            "answer": answer,
            "retrieved_context": row["retrieved_context"],  # preserved (identical retrieval)
            "n_chunks": len(chunks),
            "latency_s": 0.0,
            "system": "retrievalonly",
            "generator": "retrievalonly",
            "model": "(chunks-as-answer)",
        })
    out = ROOT / "results" / f"gen_r3_retrievalonly_{args.tag}.csv"
    pd.DataFrame(out_rows).to_csv(out, index=False)
    print(f"Wrote {len(out_rows)} rows ({empty} empty) -> {out}")
    print(f"Source: {src.name} (retrieval = generator-independent, so chunks match any same-retrieval run)")


if __name__ == "__main__":
    main()
