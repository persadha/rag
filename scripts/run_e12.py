"""E12: Standard + hybrid(BM25+dense+RRF) + MiniLM reranker (the 2021 default
ms-marco-MiniLM-L-6-v2), generator llama3:8b (local), judge gpt-oss-120b.

Motivation: E9b showed bge-reranker-base (AC 0.578) actually LOST to the MiniLM
rerank baseline (0.625) on a dense top-20 pool. So E9c's 0.689 win came from the
*hybrid* retrieval, not the bge reranker. E12 swaps bge -> MiniLM under the same
hybrid pipeline as E9c (same llama3:8b generator, same judge) to A/B the reranker
within hybrid. If E12 > E9c, hybrid+MiniLM is the true best-retrieval config and
becomes the UI default.

Same self-healing + detached + keep-awake pattern as run_e10.py. llama3:8b is fast
(~22-30 s/row), so OLLAMA_TIMEOUT=300 is ample; true hangs still caught per-row.

To avoid Ollama model-reload thrash, this is meant to run while E10 (deepseek) is
PAUSED. When E12 finishes, it auto-relaunches run_e10.py detached so E10 resumes
from where it left off — unless E10 is already complete.

Launch DETACHED so it outlives the parent shell:
    Start-Process -WindowStyle Hidden D:\\rag\\.venv\\Scripts\\python.exe `
      -ArgumentList 'scripts\\run_e12.py'
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
LOG = ROOT / "results" / "_e12_watchdog.log"

# llama3:8b via the local ollama generator; fast model, standard timeout.
ENV = dict(os.environ, OLLAMA_GENERATOR_MODEL="llama3:8b", OLLAMA_TIMEOUT="300")

GEN = "results/gen_r3_standard_ollama_hybrid_minilm.csv"
EV = "results/eval_r3_standard_ollama_hybrid_minilm_deepeval.csv"

# No --reranker-model => RerankRetriever uses its default ms-marco-MiniLM-L-6-v2.
GEN_CMD = [PY, "-u", str(ROOT / "scripts" / "run_generation.py"),
           "--system", "standard", "--generator", "ollama", "--hybrid",
           "--tag", "minilm"]
JUDGE_CMD = [PY, "-u", str(ROOT / "scripts" / "run_eval.py"),
             "--gen", str(ROOT / GEN), "--judge", "compat:openai/gpt-oss-120b"]

# E10 output (to decide whether to auto-resume it after E12).
E10_GEN = ROOT / "results" / "gen_r3_standard_ollama_hybrid_deepseek_bgebase.csv"
E10_EV = ROOT / "results" / "eval_r3_standard_ollama_hybrid_deepseek_bgebase_deepeval.csv"


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
    p = path if isinstance(path, Path) else ROOT / path
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


def resume_e10():
    """Relaunch run_e10.py detached so deepseek's E10 resumes, unless it's done."""
    if count(E10_GEN) >= TOTAL and count(E10_EV) >= TOTAL:
        log("E10 already complete (gen+judge); not relaunching.")
        return
    DETACHED = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    subprocess.Popen([PY, str(ROOT / "scripts" / "run_e10.py")],
                     cwd=str(ROOT), creationflags=DETACHED, close_fds=True)
    log(f"E10 relaunched detached to resume (gen {count(E10_GEN)}/{TOTAL}).")


def main():
    keep_awake()
    log("=== E12 watchdog started (llama3:8b + hybrid + MiniLM) ===")
    run_until("E12 gen", GEN_CMD, GEN)
    if count(GEN) >= TOTAL:
        run_until("E12 judge", JUDGE_CMD, EV)
    log("=== E12 watchdog: DONE ===")
    resume_e10()


if __name__ == "__main__":
    main()
