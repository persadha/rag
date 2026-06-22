"""Hourly progress monitor for long-running generation / eval passes.

Standalone — reads the append-only CSVs that run_generation.py and run_eval.py
write, so it never touches the run itself. Each tick prints a timestamped status
line (and appends it to results/_progress.log): rows done / total, rows since the
last tick, rows/hour, ETA, mean latency, and — if --eval is given — the running
mean answer-correctness and contextual precision.

Run it in a second terminal alongside a generation:
    python scripts/report_progress.py --gen results/gen_r3_standard_ollama_hybrid_bgebase.csv --total 195
One-off snapshot (cron-friendly):
    python scripts/report_progress.py --gen results/gen_r3_standard_ollama_hybrid_bgebase.csv --once
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "results" / "_progress.log"


def count_done(path: Path) -> int:
    """Unique row_ids written so far (resume-safe), or 0 if the file isn't there yet."""
    if not path.exists():
        return 0
    try:
        return pd.read_csv(path, usecols=["row_id"])["row_id"].nunique()
    except Exception:
        return 0


def gen_stats(path: Path):
    try:
        df = pd.read_csv(path).drop_duplicates("row_id", keep="last")
        return df["latency_s"].mean()
    except Exception:
        return None


def eval_means(path: Path):
    if not path or not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return (df["answer_correctness"].mean(), df["contextual_precision"].mean(), len(df))
    except Exception:
        return None


def emit(msg: str):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def tick(gen_path, eval_path, total, prev_done, prev_t):
    done = count_done(gen_path)
    now = time.time()
    rate = None
    if prev_t is not None and now > prev_t:
        rate = (done - prev_done) / ((now - prev_t) / 3600.0)  # rows/hour

    parts = [f"gen {done}/{total} ({100*done/total:.0f}%)"]
    if rate is not None:
        parts.append(f"+{done - prev_done} since last, {rate:.0f} rows/h")
        remaining = total - done
        if rate > 0 and remaining > 0:
            parts.append(f"ETA ~{remaining / rate:.1f} h")
    lat = gen_stats(gen_path)
    if lat is not None and not pd.isna(lat):
        parts.append(f"mean {lat:.1f}s/row")
    ev = eval_means(eval_path)
    if ev is not None:
        ac, p, n = ev
        parts.append(f"| eval {n}/{total}: AC={ac:.3f} P={p:.3f}")
    emit(" · ".join(parts))
    return done, now


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen", required=True, help="generation CSV to track")
    ap.add_argument("--eval", default=None, help="optional eval CSV to track running metrics")
    ap.add_argument("--total", type=int, default=195)
    ap.add_argument("--interval", type=float, default=3600, help="seconds between reports")
    ap.add_argument("--once", action="store_true", help="single snapshot then exit")
    args = ap.parse_args()

    gen_path = Path(args.gen)
    eval_path = Path(args.eval) if args.eval else None

    prev_done, prev_t = None, None
    try:
        while True:
            prev_done, prev_t = tick(gen_path, eval_path, args.total,
                                     prev_done if prev_done is not None else 0, prev_t)
            if args.once or prev_done >= args.total:
                if prev_done >= args.total:
                    emit(f"COMPLETE: {prev_done}/{args.total} rows generated.")
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        emit("progress monitor stopped (Ctrl-C).")


if __name__ == "__main__":
    main()
