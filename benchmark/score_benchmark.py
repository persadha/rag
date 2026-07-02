"""Score a system's answers against the ILSA-TableQA benchmark.

Mirrors scripts/run_eval.py conventions: argparse CLI, row-by-row resumable CSV
append, grouped summary at the end. Unlike run_eval.py this needs NO LLM judge —
correctness is numeric-exact-with-tolerance (see benchmark/scorer.py), which is
deterministic, free, and cannot be gamed by verbose answers.

Inputs
------
--items        benchmark/items.example.jsonl  (TableQAItem JSONL; see schema.py)
--predictions  a CSV or JSONL mapping item id -> the system's answer string.
               CSV needs columns: id, answer[, system]. JSONL needs {"id","answer"}.

Usage
-----
    # lint the item set
    python benchmark/score_benchmark.py --items benchmark/items.example.jsonl --validate

    # write a flat CSV annotation template to fill in
    python benchmark/score_benchmark.py --emit-template benchmark/template.csv

    # score a system's answers, broken down by reasoning/exhibit type
    python benchmark/score_benchmark.py --items benchmark/items.example.jsonl \
        --predictions results/tableqa_docling_llama.csv
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark import schema
from benchmark.scorer import score_item


def load_predictions(path: Path) -> dict:
    """id -> answer string. Accepts CSV (cols id,answer) or JSONL ({"id","answer"})."""
    preds = {}
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                preds[str(d["id"])] = str(d.get("answer", ""))
    else:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                preds[str(row["id"])] = str(row.get("answer", ""))
    return preds


def emit_template(path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(schema.CSV_COLUMNS)
        w.writerow(["EX-001", "PIRLS", "2021", "Exhibit 1 Years of Schooling.pdf",
                    "Exhibit 1", "policy_categorical",
                    "<question>", "lookup_single", "text",
                    "<gold answer>", "<serialized table region>",
                    "[]", "none", "", "[]", "", "<annotator>", ""])
    print(f"template -> {path}  (columns: {', '.join(schema.CSV_COLUMNS)})")


def cmd_validate(items) -> int:
    bad = 0
    seen_ids = set()
    for it in items:
        problems = schema.validate(it)
        if it.id in seen_ids:
            problems.append("duplicate id")
        seen_ids.add(it.id)
        if problems:
            bad += 1
            print(f"  [{it.id}] " + "; ".join(problems))
    print(f"\n{len(items)} items, {bad} with problems.")
    return 1 if bad else 0


def summarise(rows: list[dict]) -> None:
    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    scores = [r["score"] for r in rows]
    exacts = [1.0 if r["exact"] else 0.0 for r in rows]
    print(f"\nSUMMARY  n={len(rows)}")
    print(f"  numeric/answer score (mean) : {mean(scores):.3f}")
    print(f"  exact-match rate            : {mean(exacts):.3f}")
    for group in ("reasoning_type", "exhibit_type", "study"):
        by = defaultdict(list)
        for r in rows:
            by[r[group]].append(r["score"])
        print(f"  by {group}:")
        for k in sorted(by):
            print(f"      {k:24s} n={len(by[k]):3d}  mean={mean(by[k]):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=Path, help="TableQAItem JSONL")
    ap.add_argument("--predictions", type=Path, help="id->answer CSV/JSONL")
    ap.add_argument("--validate", action="store_true", help="lint items and exit")
    ap.add_argument("--emit-template", type=Path, help="write a CSV annotation template and exit")
    args = ap.parse_args()

    if args.emit_template:
        emit_template(args.emit_template)
        return
    if not args.items:
        ap.error("--items is required (unless --emit-template)")
    items = schema.read_jsonl(args.items)

    if args.validate:
        sys.exit(cmd_validate(items))

    if not args.predictions:
        ap.error("--predictions is required to score (or use --validate)")
    preds = load_predictions(args.predictions)

    out_path = args.predictions.with_name(f"score_{args.predictions.stem}.csv")
    done = set()
    if out_path.exists():
        done = {r["id"] for r in csv.DictReader(open(out_path, encoding="utf-8"))}
        print(f"Resuming: {len(done)} rows already scored in {out_path.name}")

    rows: list[dict] = []
    missing = 0
    with open(out_path, "a", encoding="utf-8", newline="") as fh:
        writer = None
        for it in items:
            if it.id in done:
                continue
            if it.id not in preds:
                missing += 1
                print(f"  no prediction for {it.id} — skipping")
                continue
            res = score_item(it, preds[it.id])
            record = {"id": it.id, "study": it.study, "exhibit_type": it.exhibit_type,
                      "reasoning_type": it.reasoning_type, **res.as_row(),
                      "detail": json.dumps(res.detail, ensure_ascii=False)}
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(record))
                if not done:
                    writer.writeheader()
            writer.writerow(record)
            rows.append(record)
            print(f"  {it.id}: score={res.score:.3f} exact={res.exact} ({res.kind})")

    all_rows = list(csv.DictReader(open(out_path, encoding="utf-8")))
    for r in all_rows:  # normalise types read back from CSV
        r["score"] = float(r["score"])
        r["exact"] = str(r["exact"]).lower() == "true"
    summarise(all_rows)
    if missing:
        print(f"  ({missing} items had no prediction and were skipped)")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
