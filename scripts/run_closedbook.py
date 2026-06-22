"""Closed-book baseline generation (reviewer value-added ablation): llama3:8b,
NO retrieval. Generation-only, self-healing + detached + keep-awake (same pattern
as run_e10/e11/e12). Judging is deliberately NOT done here — it's batched later
(AC-only, alongside retrieval-only) so it doesn't run a 3rd concurrent DeepInfra
judge stream against E11/E12.

Ollama contention: this uses llama3:8b, so it must NOT overlap E10 (deepseek) or
an E12 gen. Launch it only in an idle-Ollama window (E12 gen done, E10 paused).

Launch DETACHED:
    Start-Process -WindowStyle Hidden D:\\rag\\.venv\\Scripts\\python.exe `
      -ArgumentList 'scripts\\run_closedbook.py'

Judge afterwards (AC only), when E11/E12 judges have drained:
    python scripts/run_eval.py --gen results/gen_r3_noretrieval_ollama.csv \\
      --metrics answer_correctness --judge compat:openai/gpt-oss-120b
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
LOG = ROOT / "results" / "_closedbook_watchdog.log"
ENV = dict(os.environ, OLLAMA_GENERATOR_MODEL="llama3:8b", OLLAMA_TIMEOUT="300")

GEN = "results/gen_r3_noretrieval_ollama.csv"
GEN_CMD = [PY, "-u", str(ROOT / "scripts" / "run_generation.py"),
           "--system", "standard", "--generator", "ollama", "--no-retrieval"]


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
    log("=== closed-book watchdog started (llama3:8b, no retrieval) ===")
    run_until("closedbook gen", GEN_CMD, GEN)
    log("=== closed-book watchdog: DONE (gen only; judge AC separately) ===")


if __name__ == "__main__":
    main()
