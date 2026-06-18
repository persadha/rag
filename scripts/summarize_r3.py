"""Summarize the r3 evaluation: per-system metric means across the three
eval CSVs, reported overall AND split by data_type (human vs synthetic).

Reads results/eval_r3_<system>_ollama_deepeval.csv for system in (standard,
crag, cragpp), prints the comparison tables, and writes a multi-sheet
results/eval_r3_summary.xlsx (one sheet 'overall', one per data_type).

Usage:
    python scripts/summarize_r3.py
    python scripts/summarize_r3.py --generator ollama --systems standard crag cragpp
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.run_eval import METRIC_COLUMNS


def load(systems, generator):
    frames = []
    for system in systems:
        path = ROOT / "results" / f"eval_r3_{system}_{generator}_deepeval.csv"
        if not path.exists():
            print(f"  (skip) missing {path.name}")
            continue
        df = pd.read_csv(path)
        df["system"] = system  # authoritative, in case the CSV column drifted
        frames.append(df)
    if not frames:
        sys.exit("No eval CSVs found — run scripts/run_eval.py first.")
    return pd.concat(frames, ignore_index=True)


def means_table(df):
    """Per-system mean of each metric, plus n and null counts."""
    rows = []
    for system, g in df.groupby("system", sort=False):
        row = {"system": system, "n": len(g)}
        for col in METRIC_COLUMNS:
            row[col] = round(g[col].mean(), 3) if col in g else None
            row[f"{col}_null"] = int(g[col].isna().sum()) if col in g else None
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", default="ollama")
    parser.add_argument("--systems", nargs="+", default=["standard", "crag", "cragpp"])
    args = parser.parse_args()

    df = load(args.systems, args.generator)
    out_path = ROOT / "results" / "eval_r3_summary.xlsx"

    sheets = {"overall": means_table(df)}
    if "data_type" in df.columns:
        for dt, g in df.groupby("data_type", sort=False):
            sheets[str(dt)] = means_table(g)

    for name, table in sheets.items():
        print(f"\n=== {name} (n={table['n'].sum()}) ===")
        print(table.to_string(index=False))

    with pd.ExcelWriter(out_path) as writer:
        for name, table in sheets.items():
            table.to_excel(writer, sheet_name=name[:31], index=False)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
