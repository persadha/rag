"""Smoke test for the ILSA-TableQA scorer and schema.

Run:  python benchmark/smoke_scorer.py
Exits non-zero on the first failed assertion (CI-friendly, like tests/smoke_*.py).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark import schema
from benchmark.scorer import (Estimate, boolean_score, direction_score,
                              extract_estimates, extract_numbers, grits_content,
                              grits_topology, numeric_score, score_item, text_f1,
                              within_tolerance)

checks = 0


def ok(cond, label):
    global checks
    checks += 1
    if not cond:
        print(f"FAIL: {label}")
        sys.exit(1)
    print(f"  ok: {label}")


print("== number / SE parsing (ILSA notation) ==")
est = extract_estimates("Average 512 (2.3) ▲")           # value + SE + sig marker
ok(est == [Estimate(512.0, 2.3)], "parse 'estimate (se)' with significance marker")
ok(extract_numbers("Girls 522 (2.1), Boys 507 (2.6)") == [522.0, 507.0],
   "two estimates, SEs dropped for value list")
ok(extract_numbers("1,234.5 and −3.1") == [1234.5, -3.1],
   "thousands separator + unicode minus")
ok(extract_numbers("45.2%") == [45.2], "percent glyph stripped")

print("== tolerance ==")
ok(within_tolerance(45.24, 45.2, {"kind": "abs", "value": 0.05}, "percent"),
   "45.24 within abs 0.05 of 45.2")
ok(not within_tolerance(45.4, 45.2, {"kind": "abs", "value": 0.05}, "percent"),
   "45.4 NOT within abs 0.05 of 45.2")
ok(within_tolerance(519.6, 520.0, None, "scale_score"), "default scale_score tol (0.5)")

print("== numeric scoring: verbose dump cannot game it ==")
good = numeric_score("about 50%", [50.0], {"kind": "abs", "value": 0.5}, "percent")
ok(good.exact and good.score == 1.0, "concise correct answer -> exact 1.0")
wrong = numeric_score("around 62 percent", [50.0], {"kind": "abs", "value": 0.5}, "percent")
ok(not wrong.exact and wrong.score == 0.0, "wrong number -> 0.0")
# A blob containing the right number among many others gets recall but low precision:
dump = numeric_score("11 22 33 44 50 66 77 88", [50.0], {"kind": "abs", "value": 0.5}, "percent")
ok(dump.detail["recall"] == 1.0 and dump.detail["precision"] < 0.2 and not dump.exact,
   "chunk-dump: recall 1.0 but precision low -> not exact (anti-coverage-artifact)")

print("== multi-value / direction / boolean / text ==")
multi = numeric_score("45% of students, average 520", [45.0, 520.0],
                      {"kind": "abs", "value": 0.5})
ok(multi.exact and multi.score == 1.0, "both gold values recovered -> exact")
half = numeric_score("45% of students", [45.0, 520.0], {"kind": "abs", "value": 0.5})
ok(0.0 < half.score < 1.0 and not half.exact, "one of two -> partial F1")
ok(direction_score("achievement declined", "decrease").exact, "direction: declined==decrease")
ok(not direction_score("it increased", "decrease").exact, "direction: increased!=decrease")
ok(boolean_score("Yes, statistically significant", "yes (significant)").exact, "boolean yes")
ok(boolean_score("the change was not significant", False).exact, "boolean 'not significant'->False")
ok(text_f1("Arabic and English", "English and Arabic").exact, "text F1: token-set order-invariant")

print("== score_item dispatch on example items ==")
items = schema.read_jsonl(ROOT / "benchmark" / "items.example.jsonl")
ok(len(items) == 7, "loaded 7 example items (// comments ignored)")
by_id = {it.id: it for it in items}
ok(score_item(by_id["EX-003"], "the standard error is 2.3").exact, "EX-003 SE read -> exact")
ok(not score_item(by_id["EX-002"], "around 62 percent").exact, "EX-002 wrong % -> not exact")
ok(score_item(by_id["EX-005"], "reading achievement decreased").exact, "EX-005 trend -> exact")

print("== schema validation ==")
ok(all(schema.validate(it) == [] for it in items), "all example items validate clean")
bad = schema.TableQAItem(id="B", study="NASA", cycle="2021", source_document="x",
                         exhibit_id="E1", exhibit_type="policy_categorical", question="q?",
                         reasoning_type="lookup_single", answer_kind="numeric_single",
                         gold_answer="1", reference_context="c", gold_values=[])
probs = schema.validate(bad)
ok(any("study" in p for p in probs) and any("gold_values" in p for p in probs),
   "validator catches bad study + empty numeric gold_values")

print("== table-structure stub (GriTS-lite) ==")
gold_grid = [["Country", "Avg", "SE"], ["Malta", "520", "2.3"]]
ok(grits_content(gold_grid, gold_grid) == 1.0, "content: identical grid -> 1.0")
ok(grits_topology(gold_grid, gold_grid) == 1.0, "topology: identical grid -> 1.0")
partial = grits_content([["Country", "Avg", "SE"], ["Malta", "999", "2.3"]], gold_grid)
ok(0.0 < partial < 1.0, "content: one wrong cell -> partial")

print(f"\nALL {checks} CHECKS PASSED")
