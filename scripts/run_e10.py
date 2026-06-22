"""E10: Standard + hybrid(BM25+dense+RRF) + bge-reranker-base, generator
deepseek-r1:8b (local), judge gpt-oss-120b. Self-healing + detached + keep-awake,
so the slow (~6h) unattended run survives the shell reaping that kills plain
background jobs (see judge_watchdog.py — same pattern, extended to generation).

Runs generation to completion (resuming the CSV after any kill), then judging.
deepseek is a reasoning model, so OLLAMA_TIMEOUT is raised to 600s (a 510s row
occurred in the earlier deepseek run) — true hangs still get caught per-row.

Launch DETACHED so it outlives the parent shell, e.g. from PowerShell:
    Start-Process -WindowStyle Hidden D:\\rag\\.venv\\Scripts\\python.exe `
      -ArgumentList 'scripts\\run_e10.py'
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
LOG = ROOT / "results" / "_e10_watchdog.log"

# deepseek via the local ollama generator (OLLAMA_GENERATOR_MODEL), longer timeout.
ENV = dict(os.environ, OLLAMA_GENERATOR_MODEL="deepseek-r1:8b", OLLAMA_TIMEOUT="600")

GEN = "results/gen_r3_standard_ollama_hybrid_deepseek_bgebase.csv"
EV = "results/eval_r3_standard_ollama_hybrid_deepseek_bgebase_deepeval.csv"

GEN_CMD = [PY, "-u", str(ROOT / "scripts" / "run_generation.py"),
           "--system", "standard", "--generator", "ollama", "--hybrid",
           "--reranker-model", "BAAI/bge-reranker-base", "--tag", "deepseek_bgebase"]
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
    log("=== E10 watchdog started (deepseek-r1:8b + hybrid + bge-base) ===")
    run_until("E10 gen", GEN_CMD, GEN)
    if count(GEN) >= TOTAL:
        run_until("E10 judge", JUDGE_CMD, EV)
    log("=== E10 watchdog: DONE ===")


if __name__ == "__main__":
    main()
