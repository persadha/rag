"""Pick the best open-source pipeline (architecture + retrieval config) by mean
answer_correctness across the llama3:8b eval runs, and emit the run_generation
flags + expected gen filename to reproduce it with a different generator.

Used by the E4 (closed-model GPT-5.4-mini) phase to auto-select the winner.
Writes results/_e4_winner.sh (sourceable by the E4 bash job).
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# candidate eval file -> (system, run_generation flags, config-suffix on the gen filename)
CANDIDATES = {
    "eval_r3_standard_ollama_deepeval.csv":          ("standard", "", ""),
    "eval_r3_standard_ollama_rerank_deepeval.csv":   ("standard", "--rerank", "_rerank"),
    "eval_r3_standard_ollama_chunk512_deepeval.csv": ("standard", "--persist-dir chroma_db_512 --tag chunk512", "_chunk512"),
    "eval_r3_crag_ollama_deepeval.csv":              ("crag", "", ""),
    "eval_r3_cragpp_ollama_deepeval.csv":            ("cragpp", "", ""),
}


def main():
    rows = []
    for fname, (system, flags, suffix) in CANDIDATES.items():
        p = ROOT / "results" / fname
        if not p.exists():
            continue
        d = pd.read_csv(p)
        rows.append((system, flags, suffix,
                     d["answer_correctness"].mean(),
                     d["contextual_precision"].mean(),
                     d["contextual_recall"].mean(), len(d)))
    if not rows:
        sys.exit("select_best_pipeline: no eval files found")

    rows.sort(key=lambda r: r[3], reverse=True)  # by answer_correctness
    print("Pipeline ranking by answer_correctness (llama3:8b runs):")
    for system, flags, suffix, ac, cp, cr, n in rows:
        print(f"  {system:8s} {flags or '(baseline)':40s} AC={ac:.3f} P={cp:.3f} R={cr:.3f} n={n}")

    system, flags, suffix, ac, *_ = rows[0]
    genfile = f"results/gen_r3_{system}_openai-mini{suffix}.csv"
    out = ROOT / "results" / "_e4_winner.sh"
    out.write_text(f'E4_SYSTEM={system}\nE4_FLAGS="{flags}"\nE4_GENFILE={genfile}\n')
    print(f"\nWINNER: {system} {flags or '(baseline)'} (answer_correctness={ac:.3f})")
    print(f"  E4 will run GPT-5.4-mini on this pipeline -> {genfile}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
