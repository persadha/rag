"""Self-healing judge runner: keeps re-launching run_eval (which resumes from the
output CSV) until every target reaches the full row count, so an externally
killed shell auto-recovers instead of stalling the experiment.

Also holds the system awake (Windows SetThreadExecutionState, no admin needed)
so the machine won't sleep mid-run, and logs flushed progress to
results/_judge_watchdog.log so it can be monitored with the Read tool alone.

Launch DETACHED so it survives the parent shell being reaped, e.g. from PowerShell:
    Start-Process -WindowStyle Hidden D:\\rag\\.venv\\Scripts\\python.exe `
      -ArgumentList 'scripts\\judge_watchdog.py'
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
JUDGE = "compat:openai/gpt-oss-120b"
TOTAL = 195
MAX_NO_PROGRESS = 6  # consecutive relaunches with zero new rows -> give up (safety)
LOG = ROOT / "results" / "_judge_watchdog.log"

TARGETS = [
    ("results/gen_r3_standard_ollama_hybrid_bgebase.csv",
     "results/eval_r3_standard_ollama_hybrid_bgebase_deepeval.csv", "E9c hybrid"),
    ("results/gen_r3_standard_ollama_rerank_bgebase.csv",
     "results/eval_r3_standard_ollama_rerank_bgebase_deepeval.csv", "E9b rerank"),
]


def keep_awake():
    """ES_CONTINUOUS | ES_SYSTEM_REQUIRED — prevent sleep while we run (Windows)."""
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


def scored(eval_path: Path) -> int:
    if not eval_path.exists():
        return 0
    try:
        return pd.read_csv(eval_path, usecols=["row_id"])["row_id"].nunique()
    except Exception:
        return 0


def judge_target(gen, eval_path, label):
    gen, eval_path = ROOT / gen, ROOT / eval_path
    no_progress = 0
    while True:
        keep_awake()
        done = scored(eval_path)
        if done >= TOTAL:
            log(f"{label}: COMPLETE {done}/{TOTAL}")
            return
        log(f"{label}: {done}/{TOTAL} scored — launching run_eval (attempt resumes)")
        proc = subprocess.run(
            [PY, "-u", str(ROOT / "scripts" / "run_eval.py"),
             "--gen", str(gen), "--judge", JUDGE],
            cwd=str(ROOT))
        after = scored(eval_path)
        log(f"{label}: run_eval exited rc={proc.returncode}; now {after}/{TOTAL} (+{after - done})")
        if after >= TOTAL:
            log(f"{label}: COMPLETE {after}/{TOTAL}")
            return
        no_progress = no_progress + 1 if after == done else 0
        if no_progress >= MAX_NO_PROGRESS:
            log(f"{label}: GIVING UP after {MAX_NO_PROGRESS} relaunches with no progress "
                f"(stuck at {after}/{TOTAL}) — investigate manually.")
            return
        time.sleep(5)  # brief backoff before resuming


def main():
    keep_awake()
    log("=== judge watchdog started ===")
    for gen, ev, label in TARGETS:
        judge_target(gen, ev, label)
    log("=== judge watchdog: ALL TARGETS DONE ===")


if __name__ == "__main__":
    main()
