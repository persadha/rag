"""Judge the two value-added ablation baselines, answer_correctness ONLY, with the
gpt-oss-120b DeepInfra judge (held constant with the rest of r3). Self-healing +
detached + keep-awake. Judges sequentially (one DeepInfra stream), so it doesn't
pile concurrent judge load on the in-flight E12 judge.

  (a) closed-book   -> results/gen_r3_noretrieval_ollama.csv
  (b) retrieval-only -> results/gen_r3_retrievalonly_hybrid_bgebase.csv

Context metrics are skipped on purpose: meaningless for closed-book (no context)
and trivial for retrieval-only (answer==context). gold_context_similarity is still
written (free).

Launch DETACHED:
    Start-Process -WindowStyle Hidden D:\\rag\\.venv\\Scripts\\python.exe `
      -ArgumentList 'scripts\\run_ablation_judge.py'
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
LOG = ROOT / "results" / "_ablation_judge.log"
ENV = dict(os.environ)

JOBS = [
    ("closed-book", "results/gen_r3_noretrieval_ollama.csv",
     "results/eval_r3_noretrieval_ollama_deepeval.csv"),
    ("retrieval-only", "results/gen_r3_retrievalonly_hybrid_bgebase.csv",
     "results/eval_r3_retrievalonly_hybrid_bgebase_deepeval.csv"),
]


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


def judge(label, gen, ev):
    cmd = [PY, "-u", str(ROOT / "scripts" / "run_eval.py"),
           "--gen", str(ROOT / gen), "--metrics", "answer_correctness",
           "--judge", "compat:openai/gpt-oss-120b"]
    no_progress = 0
    while True:
        keep_awake()
        done = count(ev)
        if done >= TOTAL:
            log(f"{label}: COMPLETE {done}/{TOTAL}")
            return
        log(f"{label}: {done}/{TOTAL} — judging (resumes)")
        proc = subprocess.run(cmd, cwd=str(ROOT), env=ENV)
        after = count(ev)
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
    log("=== ablation judge started (AC-only: closed-book + retrieval-only) ===")
    for label, gen, ev in JOBS:
        judge(label, gen, ev)
    log("=== ablation judge: DONE ===")


if __name__ == "__main__":
    main()
