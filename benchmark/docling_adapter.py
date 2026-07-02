"""Docling adapter — real ILSA exhibit PDF -> table Grid.

Exercises the *text-extraction paradigm* (layout-aware parsing) end-to-end on real
input, and feeds the extracted grid into the structure-recovery metric
(benchmark/scorer.py grits_*). This is the concrete counterpart to the vision-RAG
paradigm the proposal compares against.

Docling is an OPTIONAL, heavy dependency (pulls torch + layout/table models; first
run downloads model weights). It is NOT in requirements-eval.txt on purpose. Install
into the project venv:

    .venv\\Scripts\\pip install docling

If Docling is absent, this module imports fine and `is_available()` returns False —
the rest of the benchmark (scorer, schema, harness) does not depend on it.

Usage
-----
    # extract tables from a real exhibit and print grids
    python benchmark/docling_adapter.py --pdf "datasets/Exhibit 1 Years of Schooling.pdf"

    # also dump grids to JSON and the full markdown (text-extraction paradigm output)
    python benchmark/docling_adapter.py --pdf "<pdf>" --out grids.json --markdown

    # score the FIRST extracted grid against a gold grid (list-of-lists JSON) with GriTS-lite
    python benchmark/docling_adapter.py --pdf "<pdf>" --gold gold_grid.json
"""

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.scorer import grits_content, grits_topology

Grid = list  # list[list[str]]


def is_available() -> bool:
    try:
        import docling  # noqa: F401
        return True
    except ImportError:
        return False


_INSTALL_HINT = (
    "Docling is not installed. It is an optional dependency.\n"
    "  Install into the project venv:  .venv\\Scripts\\pip install docling\n"
    "  (first conversion downloads layout + TableFormer model weights)."
)


@lru_cache(maxsize=1)
def _converter():
    from docling.document_converter import DocumentConverter
    return DocumentConverter()


def _table_to_grid(table, doc=None) -> Grid:
    """Convert a Docling TableItem to a list-of-lists grid (header row + data rows).
    Uses export_to_dataframe(); falls back to the raw cell grid if needed."""
    try:
        try:
            df = table.export_to_dataframe(doc=doc) if doc is not None else table.export_to_dataframe()
        except TypeError:  # older docling: no doc kwarg
            df = table.export_to_dataframe()
        header = [str(c) for c in df.columns]
        rows = [[("" if v is None else str(v)) for v in row] for row in df.values.tolist()]
        # Only prepend the header row if it isn't just a default 0..n RangeIndex.
        if not all(str(c) == str(i) for i, c in enumerate(df.columns)):
            return [header] + rows
        return rows
    except Exception:
        data = getattr(table, "data", None)
        grid = getattr(data, "grid", None)
        if grid is None:
            return []
        return [[getattr(cell, "text", "") or "" for cell in r] for r in grid]


def extract_grids(pdf_path) -> list[Grid]:
    """Run Docling on a PDF and return one Grid per detected table (may be empty:
    a graphic-only exhibit that the text/layout path cannot recover is itself a
    finding for the text-vs-vision comparison)."""
    if not is_available():
        raise RuntimeError(_INSTALL_HINT)
    result = _converter().convert(str(pdf_path))
    doc = result.document
    return [_table_to_grid(t, doc) for t in getattr(doc, "tables", [])]


def extract_markdown(pdf_path) -> str:
    """Full document as markdown — the text-extraction paradigm's serialized output."""
    if not is_available():
        raise RuntimeError(_INSTALL_HINT)
    return _converter().convert(str(pdf_path)).document.export_to_markdown()


def _print_grid(grid: Grid, max_rows: int = 12, max_cols: int = 8) -> None:
    if not grid:
        print("    (empty grid)")
        return
    for r in grid[:max_rows]:
        cells = [(str(c)[:20] + "..." if len(str(c)) > 22 else str(c)) for c in r[:max_cols]]
        print("    | " + " | ".join(cells) + (" | ..." if len(r) > max_cols else " |"))
    if len(grid) > max_rows:
        print(f"    ... (+{len(grid) - max_rows} more rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", type=Path, required=True, help="path to an exhibit PDF")
    ap.add_argument("--out", type=Path, help="write extracted grids to this JSON file")
    ap.add_argument("--markdown", action="store_true", help="also print document markdown")
    ap.add_argument("--gold", type=Path, help="gold grid JSON (list-of-lists) to score against")
    args = ap.parse_args()

    try:  # UTF-8 console so cell text (e.g. "Türkiye") and previews don't mojibake on Windows
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if not is_available():
        print(_INSTALL_HINT)
        sys.exit(2)
    if not args.pdf.exists():
        sys.exit(f"no such file: {args.pdf}")

    print(f"Docling: converting {args.pdf.name} ...")
    grids = extract_grids(args.pdf)
    print(f"  {len(grids)} table(s) detected.")
    for i, g in enumerate(grids):
        shape = f"{len(g)}x{max((len(r) for r in g), default=0)}"
        print(f"\n  table[{i}]  ({shape}):")
        _print_grid(g)

    if args.out:
        args.out.write_text(json.dumps(grids, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  grids -> {args.out}")

    if args.markdown:
        print("\n--- document markdown (text-extraction paradigm) ---")
        print(extract_markdown(args.pdf)[:2000])

    if args.gold:
        gold = json.loads(args.gold.read_text(encoding="utf-8"))
        if not grids:
            print("\n  cannot score: Docling detected no table (text/layout path missed it).")
            sys.exit(1)
        pred = grids[0]
        print(f"\n  structure-recovery vs {args.gold.name} (GriTS-lite, stub):")
        print(f"    full table[0] vs gold : content={grits_content(pred, gold):.3f} "
              f"topology={grits_topology(pred, gold):.3f}")
        if len(pred) != len(gold):
            # A partial gold (e.g. a header+few-rows slice) makes the full-table score
            # look bad purely from size mismatch; also report a fair region-aligned score.
            sliced = pred[:len(gold)]
            print(f"    region-aligned (pred[:{len(gold)}]) : content={grits_content(sliced, gold):.3f} "
                  f"topology={grits_topology(sliced, gold):.3f}")
            print("    note: full-vs-partial-gold is size-mismatch-limited; author gold at "
                  "full-table granularity for a fair score (WP1).")


if __name__ == "__main__":
    main()
