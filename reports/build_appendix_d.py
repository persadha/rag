"""Build reports/revised_final_report_appendix_D.xlsx — the per-question generated
answers and retrieved contexts for every r3 run.

This is the qualitative companion to Appendix C (which holds the per-question
scores). The two cross-reference by (run, row_id): same run names, same ordering.

Layout:
  - "README"   : what the workbook is + the run list.
  - "All runs" : one combined, filterable sheet (run + row_id + per-row text).
  - one sheet per run, mirroring Appendix C's detail sheets.

Source: reports/../results/gen_r3_*.csv  (one file per run).
Re-run to regenerate deterministically:  python reports/build_appendix_d.py
"""
import ast
import csv
import json
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

csv.field_size_limit(10 ** 8)

ROOT = Path(__file__).parent
RESULTS = ROOT.parent / "results"
OUT = ROOT / "revised_final_report_appendix_C.xlsx"
CELL_MAX = 32000  # Excel hard limit is 32767; leave headroom

# gen CSV -> run name, in Appendix C order (names match its detail sheets).
RUNS = [
    ("gen_r3_standard_ollama.csv",                       "Basic (baseline)"),
    ("gen_r3_crag_ollama.csv",                           "Advanced v1"),
    ("gen_r3_cragpp_ollama.csv",                         "Advanced v2"),
    ("gen_r3_standard_ollama_rerank.csv",                "Basic + rerank"),
    ("gen_r3_standard_ollama_chunk512.csv",              "chunk-512"),
    ("gen_r3_standard_ollama_deepseek.csv",              "Basic, deepseek"),
    ("gen_r3_standard_openai-mini_rerank.csv",           "Basic + rerank, gpt-5.4"),
    ("gen_r3_standard_ollama_rerank_extract.csv",        "rerank + extract prompt"),
    ("gen_r3_standard_ollama_rerank_bgebase.csv",        "dense + bge reranker"),
    ("gen_r3_standard_ollama_hybrid_bgebase.csv",        "hybrid + bge (best local)"),
    ("gen_r3_standard_ollama_hybrid_minilm.csv",         "hybrid + MiniLM"),
    ("gen_r3_standard_ollama_hybrid_deepseek_bgebase.csv","hybrid + bge, deepseek"),
    ("gen_r3_standard_openai-mini_hybrid_bgebase.csv",   "hybrid + bge, gpt-5.4 (best overall)"),
    ("gen_r3_cragpp_ollama_hybrid_cragfix.csv",          "Advanced v3, llama3-8b"),
    ("gen_r3_cragpp_openai-mini_hybrid_cragfix.csv",     "Advanced v3, gpt-5.4-mini"),
    ("gen_r3_noretrieval_ollama.csv",                    "Closed-book, llama3-8b"),
    ("gen_r3_noretrieval_openai-mini.csv",               "Closed-book, gpt-5.4-mini"),
    ("gen_r3_retrievalonly_hybrid_bgebase.csv",          "Retrieval-only"),
]

COLS = ["row_id", "data_type", "question", "reference_answer",
        "reference_context", "generated_answer", "retrieved_context",
        "n_chunks", "latency_s"]
WIDTHS = {"run": 26, "row_id": 8, "data_type": 10, "question": 42,
          "reference_answer": 42, "reference_context": 50, "generated_answer": 50,
          "retrieved_context": 72, "n_chunks": 9, "latency_s": 9}
WRAP = {"question", "reference_answer", "reference_context",
        "generated_answer", "retrieved_context"}
HEAD_FILL_FONT = Font(bold=True)
TOP = Alignment(vertical="top")
TOPWRAP = Alignment(vertical="top", wrap_text=True)


def parse_chunks(raw):
    """retrieved_context is stored as a list-literal string -> list of chunks."""
    raw = (raw or "").strip()
    if not raw or raw in ("[]", "''", '""'):
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            v = loader(raw)
            if isinstance(v, (list, tuple)):
                return [str(c) for c in v]
            return [str(v)]
        except Exception:
            pass
    return [raw]  # fall back to the raw string


def render_context(raw, n_chunks, is_closedbook):
    chunks = parse_chunks(raw)
    if is_closedbook or (not chunks and str(n_chunks).strip() in ("0", "", "0.0")):
        return "— (no retrieval)"
    return "\n\n".join(f"[{i}] {c}" for i, c in enumerate(chunks, 1))


def clip(s):
    s = "" if s is None else str(s)
    return s if len(s) <= CELL_MAX else s[:CELL_MAX] + " … [truncated]"


def read_run(fname, run_name):
    path = RESULTS / fname
    is_closedbook = "noretrieval" in fname
    is_retrieval_only = "retrievalonly" in fname
    rows = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            gen = r.get("answer", "")
            if is_retrieval_only:
                gen = ("[No LLM generation — the retrieved chunks below were "
                       "returned directly as the answer (see §6.3).]")
            rows.append({
                "row_id": r.get("row_id", ""),
                "data_type": r.get("data_type", ""),
                "question": clip(r.get("question", "")),
                "reference_answer": clip(r.get("answer_ref", "")),
                "reference_context": clip(r.get("reference_context", "")),
                "generated_answer": clip(gen),
                "retrieved_context": clip(render_context(
                    r.get("retrieved_context", ""), r.get("n_chunks", ""), is_closedbook)),
                "n_chunks": r.get("n_chunks", ""),
                "latency_s": r.get("latency_s", ""),
            })
    return rows


def style_sheet(ws, header):
    for j, col in enumerate(header, 1):
        c = ws.cell(row=1, column=j)
        c.font = HEAD_FILL_FONT
        c.alignment = TOP
        ws.column_dimensions[get_column_letter(j)].width = WIDTHS.get(col, 16)
    ws.freeze_panes = "A2"
    for row in ws.iter_rows(min_row=2):
        for c in row:
            col = header[c.column - 1]
            c.alignment = TOPWRAP if col in WRAP else TOP


def write_sheet(wb, title, header, records):
    ws = wb.create_sheet(title=title[:31])
    ws.append(header)
    for rec in records:
        ws.append([rec.get(col, "") for col in header])
    style_sheet(ws, header)
    return ws


def main():
    all_rows = []
    per_run = []
    for fname, run_name in RUNS:
        rows = read_run(fname, run_name)
        per_run.append((run_name, rows))
        for r in rows:
            all_rows.append({"run": run_name, **r})

    wb = openpyxl.Workbook()
    wb.remove(wb.active)  # drop default sheet

    # README
    readme = wb.create_sheet("README")
    readme.column_dimensions["A"].width = 110
    lines = [
        "Appendix D — Generated Answers and Retrieved Contexts",
        "",
        "Per-question generated answers and the chunks retrieved for every r3 run.",
        "Qualitative companion to Appendix C (per-question scores); cross-reference by (run, row_id).",
        "",
        "Sheets:",
        "  • All runs — every run in one filterable sheet (filter on 'run' and 'row_id').",
        "  • One sheet per run, in the same order and with the same names as Appendix C.",
        "",
        "Notes:",
        "  • Closed-book runs use no retrieval; their retrieved_context reads '— (no retrieval)'.",
        "  • Retrieval-only returns the raw top-k chunks as the answer (no LLM); see §6.3.",
        "  • retrieved_context lists the chunks shown to the generator, marked [1], [2], …",
        "",
        "Runs included:",
    ]
    for _, run_name in RUNS:
        lines.append(f"  • {run_name}")
    for ln in lines:
        readme.append([ln])
    readme["A1"].font = Font(bold=True, size=14)

    # combined + per-run
    write_sheet(wb, "All runs", ["run"] + COLS, all_rows)
    for run_name, rows in per_run:
        write_sheet(wb, run_name, COLS, rows)

    wb.save(OUT)
    print(f"wrote {OUT}  ({len(all_rows)} rows across {len(per_run)} runs)")


if __name__ == "__main__":
    main()
