"""Closed-book baseline with the STRONG generator: gpt-5.4-mini, NO retrieval.
Completes the model x retrieval 2x2 (reviewer R1 'is RAG worth it?' / NAEP point):

                   closed-book      RAG (hybrid+bge)
    llama3:8b        0.202              0.689
    gpt-5.4-mini      THIS              0.774

Headline test: does small-open + retrieval (RAG-llama 0.689) beat large-closed
without retrieval (this run)? Generation is OpenAI (fast, ~$0.5). AC-only judge on
gpt-oss-120b, gated to start only AFTER the in-flight cragfix judge finishes, so
we don't run two concurrent DeepInfra judge streams.

Self-healing + detached + keep-awake. Launch DETACHED:
    Start-Process -WindowStyle Hidden D:\\rag\\.venv\\Scripts\\python.exe `
      -ArgumentList 'scripts\\run_closedbook_mini.py'
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
LOG = ROOT / "results" / "_closedbook_mini_watchdog.log"
ENV = dict(os.environ)  # OPENAI_API_KEY / JUDGE_API_KEY from .env

GEN = "results/gen_r3_noretrieval_openai-mini.csv"
EV = "results/eval_r3_noretrieval_openai-mini_deepeval.csv"
# Gate: don't start judging until this in-flight judge has finished (avoid 2 DeepInfra streams).
CRAGFIX_EV = "results/eval_r3_cragpp_openai-mini_hybrid_cragfix_deepeval.csv"

GEN_CMD = [PY, "-u", str(ROOT / "scripts" / "run_generation.py"),
           "--system", "standard", "--generator", "openai-mini", "--no-retrieval"]
JUDGE_CMD = [PY, "-u", str(ROOT / "scripts" / "run_eval.py"),
             "--gen", str(ROOT / GEN), "--metrics", "answer_correctness",
             "--judge", "compat:openai/gpt-oss-120b"]


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
    log("=== closed-book-mini watchdog started (gpt-5.4-mini, no retrieval) ===")
    run_until("closedbook-mini gen", GEN_CMD, GEN)
    if count(GEN) >= TOTAL:
        waited = 0
        while count(CRAGFIX_EV) < TOTAL and waited < 7200:  # gate on cragfix judge (cap 2h)
            keep_awake()
            log(f"gen done; waiting for cragfix judge ({count(CRAGFIX_EV)}/{TOTAL}) to free DeepInfra…")
            time.sleep(60)
            waited += 60
        run_until("closedbook-mini judge", JUDGE_CMD, EV)
    log("=== closed-book-mini watchdog: DONE ===")


if __name__ == "__main__":
    main()
