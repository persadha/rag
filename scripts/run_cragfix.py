"""Full 195-row confirmation run for the FIXED CRAG++ pipeline with the strong
generator: gpt-5.4-mini (OpenAI), hybrid(BM25+dense+RRF)+bge-reranker-base, judge
gpt-oss-120b. Self-healing + detached + keep-awake (same pattern as run_e11.py).

Tests the pilot signal (n=15: CRAG++fixed+mini AC 0.833 > Standard+mini E11 0.727)
at full scale vs E11's 0.774. The four fixes (T2.3 reranker-score grading, T2.4
union-rerank vs original question, T2.5 adaptive decomposition, T3.7 synthesis-from-
context) activate because the retriever carries a cross-encoder. Resumes the pilot's
15 rows (gen + eval) rather than redoing them.

Needs OPENAI_API_KEY + JUDGE_API_KEY in .env. Launch DETACHED:
    Start-Process -WindowStyle Hidden D:\\rag\\.venv\\Scripts\\python.exe `
      -ArgumentList 'scripts\\run_cragfix.py'
"""
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
TOTAL = 195
MAX_NO_PROGRESS = 6
LOG = ROOT / "results" / "_cragfix_watchdog.log"
ENV = dict(os.environ)  # OPENAI_API_KEY / JUDGE_API_KEY from .env via config_api

GEN = "results/gen_r3_cragpp_openai-mini_hybrid_cragfix.csv"
EV = "results/eval_r3_cragpp_openai-mini_hybrid_cragfix_deepeval.csv"

GEN_CMD = [PY, "-u", str(ROOT / "scripts" / "run_generation.py"),
           "--system", "cragpp", "--generator", "openai-mini", "--hybrid",
           "--reranker-model", "BAAI/bge-reranker-base", "--tag", "cragfix"]
JUDGE_CMD = [PY, "-u", str(ROOT / "scripts" / "run_eval.py"),
             "--gen", str(ROOT / GEN), "--judge", "compat:openai/gpt-oss-120b"]


def keep_awake():
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def count(path):
    p = ROOT / path
    if not p.exists():
        return 0
    try:
        return pd.read_csv(p, usecols=["row_id"])["row_id"].nunique()
    except Exception:
        return 0


def run_until(label, cmd, target_path):
    no_progress = 0
    while True:
        keep_awake()
        done = count(target_path)
        if done >= TOTAL:
            log(f"{label}: COMPLETE {done}/{TOTAL}")
            return
        log(f"{label}: {done}/{TOTAL} — launching (resumes)")
        proc = subprocess.run(cmd, cwd=str(ROOT), env=ENV)
        after = count(target_path)
        log(f"{label}: exited rc={proc.returncode}; now {after}/{TOTAL} (+{after - done})")
        if after >= TOTAL:
            log(f"{label}: COMPLETE {after}/{TOTAL}")
            return
        no_progress = no_progress + 1 if after == done else 0
        if no_progress >= MAX_NO_PROGRESS:
            log(f"{label}: GIVING UP at {after}/{TOTAL} after {MAX_NO_PROGRESS} no-progress relaunches.")
            return
        time.sleep(5)


def main():
    keep_awake()
    log("=== cragfix watchdog started (CRAG++ FIXED + gpt-5.4-mini + hybrid + bge) ===")
    run_until("cragfix gen", GEN_CMD, GEN)
    if count(GEN) >= TOTAL:
        run_until("cragfix judge", JUDGE_CMD, EV)
    log("=== cragfix watchdog: DONE ===")


if __name__ == "__main__":
    main()
