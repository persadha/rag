"""Re-judge the disputed low-answer-correctness rows with two judges to separate
generator failure from judge under-credit.

For every row in a run whose ORIGINAL answer_correctness <= --ac-max and
contextual_precision >= --p-min (the "disputed band"), this re-scores GEval
answer-correctness with `include_reason=True` under:
  - judge A: the original judge (default gpt-oss-120b via DeepInfra), re-run fresh
  - judge B: a stronger judge (default gpt-5.5 via OpenAI)
Both use the SAME GEval config as run_eval.py, so the judge is the only variable.

It then reports judge-vs-judge disagreement and, crucially, how many rows the
stronger judge LIFTS from <0.5 to >=0.5 — a direct measure of judge under-credit.

Resumable (appends per row). Start with a tiny pilot to check cost/plumbing:
    python scripts/rejudge_disputed.py --rows 3 --judge-b compat:openai/gpt-oss-120b
Then the real run (uses OpenAI credit; ~87 rows x 1 metric):
    python scripts/rejudge_disputed.py --judge-b gpt-5.5
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase
try:
    from deepeval.test_case import SingleTurnParams as TestCaseParams
except ImportError:
    from deepeval.test_case import LLMTestCaseParams as TestCaseParams

from src.config.config_api import APIConfig
from scripts.run_eval import make_judge  # reuse compat:/ollama:/openai judge dispatch

# Same criteria/params as run_eval.make_metrics — only include_reason differs.
CRITERIA = ("Determine whether the actual output answers the question with the same "
            "facts as the expected output. Penalize missing or contradicted facts; "
            "do not penalize phrasing differences or extra correct detail.")
PARAMS = [TestCaseParams.INPUT, TestCaseParams.ACTUAL_OUTPUT, TestCaseParams.EXPECTED_OUTPUT]


def geval(judge_spec):
    # GEval always emits metric.reason (no include_reason kwarg, unlike the other metrics).
    return GEval(name="Answer Correctness", criteria=CRITERIA,
                 evaluation_params=PARAMS, model=make_judge(judge_spec),
                 async_mode=False)


def score_one(metric, question, answer, reference):
    tc = LLMTestCase(input=str(question), actual_output=str(answer),
                     expected_output=str(reference))
    metric.measure(tc)
    return metric.score, (metric.reason or ""), (metric.evaluation_cost or 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen", default=str(ROOT / "results" / "gen_r3_standard_ollama.csv"))
    ap.add_argument("--eval", default=str(ROOT / "results" / "eval_r3_standard_ollama_deepeval.csv"))
    ap.add_argument("--ac-max", type=float, default=0.6, help="select rows with original AC <= this")
    ap.add_argument("--p-min", type=float, default=0.5, help="and contextual_precision >= this")
    ap.add_argument("--judge-a", default="compat:openai/gpt-oss-120b", help="original judge (re-run fresh)")
    ap.add_argument("--judge-b", default="gpt-5.5", help="stronger judge (OpenAI plain name, or compat:/ollama:)")
    ap.add_argument("--rows", type=int, default=None, help="limit to first N disputed rows (pilot)")
    ap.add_argument("--out", default=str(ROOT / "results" / "rejudge_disputed_standard.csv"))
    args = ap.parse_args()

    # Pre-flight: a plain OpenAI judge name needs a key.
    if not args.judge_b.startswith(("compat:", "ollama:")) and not os.getenv("OPENAI_API_KEY"):
        sys.exit("judge-b looks like an OpenAI model but OPENAI_API_KEY is not set in .env")

    ev = pd.read_csv(args.eval)
    gen = pd.read_csv(args.gen).drop_duplicates("row_id", keep="last").set_index("row_id")

    disputed = ev[(pd.to_numeric(ev["answer_correctness"], errors="coerce") <= args.ac_max)
                  & (pd.to_numeric(ev["contextual_precision"], errors="coerce") >= args.p_min)]
    ids = [r for r in disputed["row_id"].tolist() if r in gen.index]
    if args.rows:
        ids = ids[:args.rows]
    print(f"Disputed rows to re-judge: {len(ids)}  (AC<={args.ac_max}, P>={args.p_min})")

    out = Path(args.out)
    done = set()
    if out.exists():
        done = set(pd.read_csv(out, usecols=["row_id"])["row_id"])
        print(f"Resuming: {len(done)} already done in {out.name}")

    mA, mB = geval(args.judge_a), geval(args.judge_b)
    orig_ac = dict(zip(ev["row_id"], pd.to_numeric(ev["answer_correctness"], errors="coerce")))
    cost = 0.0
    for rid in ids:
        if rid in done:
            continue
        g = gen.loc[rid]
        q, ans, ref = g["question"], g["answer"], g["answer_ref"]
        try:
            a_s, a_r, a_c = score_one(mA, q, ans, ref)
            b_s, b_r, b_c = score_one(mB, q, ans, ref)
        except Exception as exc:
            print(f"  row {rid} FAILED: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        cost += a_c + b_c
        pd.DataFrame([{
            "row_id": rid, "orig_ac": orig_ac.get(rid),
            "judge_a": args.judge_a, "a_score": round(a_s, 3), "a_reason": a_r,
            "judge_b": args.judge_b, "b_score": round(b_s, 3), "b_reason": b_r,
            "question": q, "answer": ans, "answer_ref": ref,
        }]).to_csv(out, mode="a", header=not out.exists(), index=False)
        flag = "LIFT" if (a_s < 0.5 <= b_s) else ("DROP" if (b_s < 0.5 <= a_s) else "")
        print(f"row {rid}: A={a_s:.2f} B={b_s:.2f} (orig {orig_ac.get(rid):.2f}) {flag}")

    # ---- summary ----
    df = pd.read_csv(out)
    df = df[df["row_id"].isin(ids)]
    n = len(df)
    if n:
        a, b = df["a_score"], df["b_score"]
        lifted = int(((a < 0.5) & (b >= 0.5)).sum())
        dropped = int(((b < 0.5) & (a >= 0.5)).sum())
        print(f"\n=== SUMMARY (n={n}, judge A={args.judge_a}, judge B={args.judge_b}) ===")
        print(f"  mean AC: original={df['orig_ac'].mean():.3f}  A(fresh)={a.mean():.3f}  B={b.mean():.3f}")
        print(f"  mean |A-B| disagreement: {(a - b).abs().mean():.3f}")
        print(f"  correlation A vs B: {a.corr(b):.3f}")
        print(f"  rows B LIFTS  <0.5 -> >=0.5: {lifted}/{n} ({100*lifted/n:.0f}%)  <- judge under-credit")
        print(f"  rows B LOWERS >=0.5 -> <0.5: {dropped}/{n} ({100*dropped/n:.0f}%)")
        print(f"  => of the disputed 'failures', ~{100*lifted/n:.0f}% look like judge under-credit, "
              f"the rest are genuine generator misses.")
        print(f"  judge cost this run: ${cost:.2f} (note: $0/inaccurate if a model id is unknown to deepeval)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
