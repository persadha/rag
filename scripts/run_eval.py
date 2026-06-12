"""Score a generation output file with DeepEval (judge: gpt-4.1 by default).

Metrics (r2-comparable trio + the answer-quality metric r2 lacked):
  - ContextualPrecision, ContextualRecall, Faithfulness
  - Answer Correctness (GEval over input/actual/expected)

Reads the CSVs produced by scripts/run_generation.py (retrieved_context is a
JSON list of chunks). Appends scored rows one at a time, so an interrupted
run resumes where it stopped. Prints total judge cost at the end.

Usage:
    python scripts/run_eval.py --gen results/gen_r3_cragpp_haiku.csv
    python scripts/run_eval.py --gen results/gen_r3_standard_haiku.csv --rows 15   # pilot
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from deepeval.metrics import (ContextualPrecisionMetric, ContextualRecallMetric,
                              FaithfulnessMetric, GEval)
from deepeval.test_case import LLMTestCase
try:  # deepeval >= 4 renamed the params enum
    from deepeval.test_case import SingleTurnParams as TestCaseParams
except ImportError:
    from deepeval.test_case import LLMTestCaseParams as TestCaseParams

from src.config.config_api import APIConfig

METRIC_COLUMNS = ("contextual_precision", "contextual_recall", "faithfulness", "answer_correctness")


def make_judge(judge: str):
    """'gpt-4.1' -> OpenAI judge (needs OPENAI_API_KEY); 'ollama:<model>' -> local
    Ollama judge (key-less pilots; weaker judge, plumbing validation only)."""
    if judge.startswith("ollama:"):
        from deepeval.models import OllamaModel
        return OllamaModel(model=judge.split(":", 1)[1], temperature=0)
    return judge


def make_metrics(model):
    common = dict(model=model, include_reason=False, async_mode=False)
    return {
        "contextual_precision": ContextualPrecisionMetric(**common),
        "contextual_recall": ContextualRecallMetric(**common),
        "faithfulness": FaithfulnessMetric(**common),
        "answer_correctness": GEval(
            name="Answer Correctness",
            criteria=("Determine whether the actual output answers the question with the same "
                      "facts as the expected output. Penalize missing or contradicted facts; "
                      "do not penalize phrasing differences or extra correct detail."),
            evaluation_params=[TestCaseParams.INPUT,
                               TestCaseParams.ACTUAL_OUTPUT,
                               TestCaseParams.EXPECTED_OUTPUT],
            model=model, async_mode=False),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen", required=True, help="generation CSV from run_generation.py")
    parser.add_argument("--rows", type=int, default=None, help="limit to first N rows (pilot)")
    parser.add_argument("--judge", default=APIConfig.JUDGE_MODEL)
    args = parser.parse_args()

    gen_path = Path(args.gen)
    df = pd.read_csv(gen_path)
    if args.rows:
        df = df.head(args.rows)

    out_path = gen_path.with_name(f"eval_r3_{gen_path.stem.replace('gen_r3_', '')}_deepeval.csv")
    done = set()
    if out_path.exists():
        done = set(pd.read_csv(out_path, usecols=["row_id"])["row_id"])
        print(f"Resuming: {len(done)} rows already scored in {out_path.name}")

    metrics = make_metrics(make_judge(args.judge))
    total_cost, failures = 0.0, 0

    for _, row in df.iterrows():
        if row["row_id"] in done:
            continue
        answer = str(row["answer"]) if pd.notna(row["answer"]) else ""
        if not answer:
            print(f"  WARNING row {row['row_id']}: empty answer — scoring anyway")
        test_case = LLMTestCase(
            input=str(row["question"]),
            actual_output=answer,
            expected_output=str(row["answer_ref"]),
            retrieval_context=json.loads(row["retrieved_context"]),
        )
        scores = {}
        for name, metric in metrics.items():
            try:
                metric.measure(test_case)
                scores[name] = metric.score
                total_cost += metric.evaluation_cost or 0.0
            except Exception as exc:  # judge/parse failure: record and continue
                failures += 1
                scores[name] = None
                print(f"  row {row['row_id']} {name} FAILED: {type(exc).__name__}: {str(exc)[:120]}")
        record = pd.DataFrame([{
            "row_id": row["row_id"], "system": row["system"], "generator": row["generator"],
            **scores,
        }])
        record.to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
        printable = {k: (f"{v:.2f}" if v is not None else "FAIL") for k, v in scores.items()}
        print(f"row {row['row_id']}: {printable}")

    scored = pd.read_csv(out_path)
    print(f"\nSUMMARY {gen_path.stem} (n={len(scored)}, judge={args.judge}):")
    for col in METRIC_COLUMNS:
        print(f"  {col:22s} mean={scored[col].mean():.3f}  null={scored[col].isna().sum()}")
    print(f"  metric failures this run: {failures}")
    print(f"  judge cost this run: ${total_cost:.2f}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
