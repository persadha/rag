"""Classify the 195 evaluation questions by cognitive level, subtype, and domain
(reviewer R1-2: the set appears weighted toward fact-retrieval; quantify how much,
and surface reasoning/why-how coverage). Uses the same gpt-oss-120b DeepInfra model
held constant across r3. Row-by-row append keyed by row_id, so it resumes.

Usage:
    python scripts/classify_questions.py            # classify all 195
    python scripts/classify_questions.py --rows 10  # pilot
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

DOMAINS = ["sampling_design", "weighting_and_variance", "plausible_values_and_scaling",
           "assessment_framework", "reading_achievement_results", "context_questionnaires",
           "trends_and_methodology", "test_administration", "data_and_codebooks",
           "general_about_pirls", "other"]
SUBTYPES = ["factual_lookup", "definitional", "procedural_how", "explanatory_why",
            "comparative_analytical"]

SYS = (
    "You classify evaluation questions about the PIRLS 2021 international reading-assessment "
    "technical documentation. Return ONLY a compact JSON object with three keys:\n"
    "  cognitive_level: 'fact_retrieval' (asks for a specific fact, figure, name, or definition "
    "that can be looked up) OR 'reasoning' (requires explanation, justification, a procedure, or a "
    "comparison — typically why/how/analysis questions).\n"
    f"  subtype: one of {SUBTYPES}.\n"
    f"  domain: one of {DOMAINS}.\n"
    "No prose, no markdown — just the JSON object."
)


def parse(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--dataset", default=str(ROOT / "datasets" / "revision" / "evaluation_dataset.xlsx"))
    ap.add_argument("--sheet", default="final")
    args = ap.parse_args()

    df = pd.read_excel(args.dataset, sheet_name=args.sheet)
    df = df.rename(columns={"user_query": "question", "reference_answer": "answer_ref"})
    if args.rows:
        df = df.head(args.rows)

    out = ROOT / "results" / "question_classification.csv"
    done = set()
    if out.exists():
        done = set(pd.read_csv(out, usecols=["row_id"])["row_id"])
        print(f"Resuming: {len(done)} already classified")

    api_key = os.getenv("DEEPINFRA_API_KEY") or os.getenv("JUDGE_API_KEY")
    base_url = os.getenv("JUDGE_BASE_URL", "https://api.deepinfra.com/v1/openai")
    if not api_key:
        sys.exit("Need DEEPINFRA_API_KEY or JUDGE_API_KEY in .env")
    # gpt-oss-120b is a reasoning model: it spends tokens on internal reasoning before
    # emitting the JSON, so a tight cap returns empty/truncated content. Give it room.
    llm = ChatOpenAI(model="openai/gpt-oss-120b", base_url=base_url, api_key=api_key,
                     temperature=0, max_tokens=2000, max_retries=4, timeout=90)

    failed = 0
    for _, row in df.iterrows():
        rid = int(row["id"])
        if rid in done:
            continue
        try:
            resp = llm.invoke([("system", SYS), ("user", str(row["question"]))]).content
            obj = parse(resp)
            if not obj:
                raise ValueError(f"unparseable: {resp[:100]}")
        except Exception as exc:
            failed += 1
            print(f"  row {rid} FAILED: {type(exc).__name__}: {str(exc)[:100]}")
            continue
        rec = pd.DataFrame([{
            "row_id": rid,
            "data_type": row.get("data_type", None),
            "cognitive_level": obj.get("cognitive_level"),
            "subtype": obj.get("subtype"),
            "domain": obj.get("domain"),
            "question": str(row["question"])[:300],
        }])
        rec.to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"row {rid}: {obj.get('cognitive_level')} / {obj.get('subtype')} / {obj.get('domain')}")

    n = pd.read_csv(out)["row_id"].nunique() if out.exists() else 0
    print(f"\nDONE: {n} classified, {failed} failed this pass -> {out}")


if __name__ == "__main__":
    main()
