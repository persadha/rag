"""Assemble the reviewer value-added table (R1 'is RAG worth it?' + R2 'value added
by generation'). Reads the eval CSVs for each condition and reports answer_correctness
overall and split by data_type (human vs synthetic). Missing/partial files are shown
as such, so it can be run before all judging completes.

  (a) closed-book      llama3:8b, no retrieval
  (b) retrieval-only   hybrid+bge chunks as the answer (no LLM synthesis)
  (c) full RAG (llama)      hybrid+bge + llama3:8b      [same generator as (a) -> isolates retrieval]
  (d) full RAG (best, mini) hybrid+bge + gpt-5.4-mini   [program best -> ceiling]

Usage: python scripts/build_value_added_table.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd

CONDITIONS = [
    ("(a) closed-book (llama, no retrieval)", "eval_r3_noretrieval_ollama_deepeval.csv"),
    ("(b) retrieval-only (chunks as answer)", "eval_r3_retrievalonly_hybrid_bgebase_deepeval.csv"),
    ("(c) full RAG  llama + hybrid + bge",    "eval_r3_standard_ollama_hybrid_bgebase_deepeval.csv"),
    ("(d) full RAG  gpt-5.4-mini + hybrid+bge", "eval_r3_standard_openai-mini_hybrid_bgebase_deepeval.csv"),
]


def stats(path):
    p = ROOT / "results" / path
    if not p.exists():
        return None
    d = pd.read_csv(p)
    if "answer_correctness" not in d.columns or not len(d):
        return None
    g = d.groupby("data_type")["answer_correctness"].mean()
    return {
        "n": d["row_id"].nunique(),
        "overall": d["answer_correctness"].mean(),
        "human": g.get("human", float("nan")),
        "synthetic": g.get("synthetic", float("nan")),
    }


def main():
    print("=" * 78)
    print("VALUE-ADDED ABLATION — answer_correctness (judge: gpt-oss-120b)")
    print("=" * 78)
    print(f"{'Condition':<42}{'n':>5}{'overall':>9}{'human':>8}{'synth':>8}")
    print("-" * 78)
    vals = {}
    for label, path in CONDITIONS:
        s = stats(path)
        vals[label] = s
        if s is None:
            print(f"{label:<42}{'—':>5}{'(pending/missing)':>25}")
        else:
            done = "" if s["n"] >= 195 else f"  [partial {s['n']}/195]"
            print(f"{label:<42}{s['n']:>5}{s['overall']:>9.3f}{s['human']:>8.3f}{s['synthetic']:>8.3f}{done}")
    print("-" * 78)

    a = vals[CONDITIONS[0][0]]
    b = vals[CONDITIONS[1][0]]
    c = vals[CONDITIONS[2][0]]
    d = vals[CONDITIONS[3][0]]
    print("\nReviewer takeaways:")
    if a and c:
        print(f"  R1  RAG vs closed-book (same llama):  {c['overall']:.3f} vs {a['overall']:.3f}  "
              f"= {c['overall']-a['overall']:+.3f} from retrieval")
    if a and d:
        print(f"  R1  best RAG vs closed-book:          {d['overall']:.3f} vs {a['overall']:.3f}  "
              f"= {d['overall']-a['overall']:+.3f} (ceiling)")
    if b and c:
        print(f"  R2  generation value (llama):         {c['overall']:.3f} vs {b['overall']:.3f} chunks  "
              f"= {c['overall']-b['overall']:+.3f} from the LLM")
    if b and d:
        print(f"  R2  generation value (gpt-5.4-mini):  {d['overall']:.3f} vs {b['overall']:.3f} chunks  "
              f"= {d['overall']-b['overall']:+.3f} from the LLM")
    print("=" * 78)


if __name__ == "__main__":
    main()
