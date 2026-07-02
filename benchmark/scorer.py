"""ILSA-TableQA scorers — numeric-exact (real) + table-structure recovery (stub).

Why this exists
---------------
The prior PIRLS RAG study hit a *coverage artifact*: a 4,000-char chunk dump scored
0.827 on the answer-correctness metric because that metric rewarded fact *coverage*
and was told not to penalise extra detail (reports/reviewer_response_r3.md, R2-2).
For statistical tables, "close" is wrong — a reported percentage or standard error
must match to the reported precision. This module scores answers on *numeric
exactness with tolerance*, which cannot be gamed by verbosity.

Two families of metric:

1. Answer correctness (`score_item`) — the load-bearing, fully-implemented metric.
   Dispatches on the item's `answer_kind`:
     - numeric_single / numeric_multi -> numeric-exact-match-with-tolerance
     - direction                      -> trend direction (up/down/flat)
     - boolean                        -> yes/no, significant/not-significant
     - set / text / category          -> normalised token-set F1
   Handles ILSA table conventions: percentages, standard errors in parentheses
   "512 (2.3)", significance markers (u25b2 u25bc * ** dagger), thousands
   separators, unicode minus/dashes.

2. Table-structure recovery (`grits_content`, `grits_topology`, `teds_struct`) —
   a CLEARLY-LABELLED STUB. A working grid-based approximation (GriTS-lite) is
   implemented in pure Python so the harness runs end-to-end today; production
   should swap in the reference GriTS (Smock et al., arXiv:2203.12555) / TEDS.
   `teds_struct` uses the optional `apted`+`lxml` deps if present, else falls back.

Pure-Python, no new dependencies (fits requirements-eval.txt / Windows / py3.13).
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field

# --- ILSA notation constants -------------------------------------------------

# Significance / footnote markers to strip before numeric parsing.
_SIG_MARKERS = "▲▼↑↓†‡§*°"  # ▲ ▼ ↑ ↓ † ‡ § * °
_MINUSES = "−–—‐‑"                          # − – — ‐ ‑  (all -> '-')

# Default numeric tolerances by unit (used when an item sets none). abs unless noted.
DEFAULT_TOLERANCE = {
    "percent": {"kind": "abs", "value": 0.05},      # reported to 1 dp; catch rounding
    "scale_score": {"kind": "abs", "value": 0.5},   # PIRLS/TIMSS scale points
    "se": {"kind": "abs", "value": 0.05},           # standard error
    "years": {"kind": "abs", "value": 0.0},         # integers, exact
    "count": {"kind": "abs", "value": 0.0},         # integers, exact
    "none": {"kind": "abs", "value": 0.0},
}

# Direction vocabulary for trend questions.
_UP = {"up", "increase", "increased", "higher", "rose", "rise", "gain", "improved", "more"}
_DOWN = {"down", "decrease", "decreased", "lower", "fell", "fall", "decline", "declined", "less"}
_FLAT = {"flat", "no change", "nochange", "unchanged", "stable", "same",
         "no significant change", "not significant", "no difference"}

_YES = {"yes", "true", "significant", "statistically significant", "y"}
_NO = {"no", "false", "not significant", "notsignificant", "no", "n",
       "not statistically significant", "insignificant"}


# --- number parsing ----------------------------------------------------------

# An estimate: optional sign, digits with optional thousands separators, optional decimals.
_NUM = r"[-+]?\d{1,3}(?:[,   ]\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?|[-+]?\.\d+"
# Estimate optionally followed by a standard error in parentheses: "512 (2.3)".
_EST_SE = re.compile(rf"({_NUM})\s*(?:\(\s*({_NUM})\s*\))?")


def _clean(text: str) -> str:
    """Normalise unicode, unify minus signs, drop significance markers and % signs
    (percent handling is by `unit`, not by the literal glyph)."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    for m in _MINUSES:
        s = s.replace(m, "-")
    s = "".join(" " if ch in _SIG_MARKERS else ch for ch in s)
    return s


def normalize_number(token: str):
    """A single token like '1,234.5' / '45.2%' / '(2.3)' -> float, or None."""
    if token is None:
        return None
    s = _clean(token).replace("%", "")
    s = re.sub(r"[,   ](?=\d{3}\b)", "", s)  # strip thousands separators
    s = s.strip().strip("()")
    try:
        return float(s)
    except ValueError:
        return None


@dataclass
class Estimate:
    value: float
    se: float | None = None


def extract_estimates(text: str) -> list[Estimate]:
    """All numeric estimates in `text`, each with an optional parenthesised SE.
    'girls 520 (2.1), boys 505 (2.6)' -> [Est(520, 2.1), Est(505, 2.6)]."""
    s = _clean(text)
    out: list[Estimate] = []
    for m in _EST_SE.finditer(s):
        val = normalize_number(m.group(1))
        if val is None:
            continue
        se = normalize_number(m.group(2)) if m.group(2) is not None else None
        out.append(Estimate(val, se))
    return out


def extract_numbers(text: str) -> list[float]:
    """Just the estimate values (SEs dropped)."""
    return [e.value for e in extract_estimates(text)]


# --- tolerance & matching ----------------------------------------------------

def within_tolerance(pred: float, gold: float, tolerance: dict | None, unit: str = "none") -> bool:
    tol = tolerance or DEFAULT_TOLERANCE.get(unit, {"kind": "abs", "value": 0.0})
    kind, value = tol.get("kind", "abs"), float(tol.get("value", 0.0))
    diff = abs(pred - gold)
    if kind == "rel":
        return diff <= value * abs(gold)
    return diff <= value


@dataclass
class ScoreResult:
    score: float                 # [0,1] headline (F1 for multi/set, 1/0 for single)
    exact: bool                  # strict: every gold value/token recovered, no spurious
    kind: str                    # which sub-scorer ran
    detail: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return {"score": round(self.score, 4), "exact": self.exact, "kind": self.kind}


def _set_match(pred_vals, gold_vals, tolerance, unit) -> ScoreResult:
    """Greedy set match of predicted numbers to gold values within tolerance.
    Returns precision/recall/F1 so a verbose dump of many numbers cannot score 1.0
    unless it contains exactly the right ones."""
    remaining = list(pred_vals)
    hits = 0
    for g in gold_vals:
        for i, p in enumerate(remaining):
            if within_tolerance(p, g, tolerance, unit):
                hits += 1
                remaining.pop(i)
                break
    recall = hits / len(gold_vals) if gold_vals else 0.0
    precision = hits / len(pred_vals) if pred_vals else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    exact = recall == 1.0 and precision == 1.0
    return ScoreResult(f1, exact, "numeric",
                       {"recall": round(recall, 4), "precision": round(precision, 4),
                        "gold_n": len(gold_vals), "pred_n": len(pred_vals), "hits": hits})


def numeric_score(predicted: str, gold_values, tolerance=None, unit="none") -> ScoreResult:
    """Numeric-exact-with-tolerance. `gold_values` is a list of target numbers."""
    preds = extract_numbers(predicted)
    return _set_match(preds, list(gold_values), tolerance, unit)


def _token_set(text: str) -> set[str]:
    s = _clean(text).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    stop = {"the", "a", "an", "of", "in", "and", "or", "to", "is", "are", "for", "on", "at"}
    return {t for t in s.split() if t and t not in stop}


def text_f1(predicted: str, gold: str) -> ScoreResult:
    """Normalised token-set F1 for categorical / free-text answers."""
    p, g = _token_set(predicted), _token_set(gold)
    if not g:
        return ScoreResult(0.0, False, "text", {"reason": "empty gold"})
    inter = len(p & g)
    precision = inter / len(p) if p else 0.0
    recall = inter / len(g)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return ScoreResult(f1, p == g, "text",
                       {"recall": round(recall, 4), "precision": round(precision, 4)})


def _direction(text: str) -> str | None:
    s = _clean(text).lower().strip()
    if any(w in s for w in _FLAT):
        return "flat"
    if any(w in s for w in _UP):
        return "up"
    if any(w in s for w in _DOWN):
        return "down"
    return None


def direction_score(predicted: str, gold: str) -> ScoreResult:
    p, g = _direction(predicted), _direction(gold)
    ok = p is not None and p == g
    return ScoreResult(1.0 if ok else 0.0, ok, "direction", {"pred": p, "gold": g})


def _boolean(text: str) -> bool | None:
    s = _clean(text).lower().strip().rstrip(".")
    # check multiword negatives first ("not significant")
    if any(s.startswith(w) or w in s for w in ("not significant", "not statistically",
                                               "insignificant", "no difference")):
        return False
    toks = _token_set(s)
    if toks & _YES:
        return True
    if toks & _NO or s in _NO:
        return False
    return None


def boolean_score(predicted: str, gold) -> ScoreResult:
    g = gold if isinstance(gold, bool) else _boolean(str(gold))
    p = _boolean(predicted)
    ok = p is not None and p == g
    return ScoreResult(1.0 if ok else 0.0, ok, "boolean", {"pred": p, "gold": g})


def score_item(item, predicted_answer: str) -> ScoreResult:
    """Dispatch on `item.answer_kind`. `item` is a benchmark.schema.TableQAItem
    (or any object/dict with answer_kind, gold_values, gold_answer, gold_unit,
    tolerance)."""
    get = (lambda k, d=None: item.get(k, d)) if isinstance(item, dict) else (lambda k, d=None: getattr(item, k, d))
    kind = get("answer_kind", "text")
    unit = get("gold_unit", "none")
    tol = get("tolerance", None)
    if kind in ("numeric_single", "numeric_multi"):
        return numeric_score(predicted_answer, get("gold_values", []) or [], tol, unit)
    if kind == "direction":
        return direction_score(predicted_answer, get("gold_answer", ""))
    if kind == "boolean":
        return boolean_score(predicted_answer, get("gold_answer", ""))
    return text_f1(predicted_answer, get("gold_answer", ""))


# --- table-structure recovery (STUB: GriTS-lite / TEDS interface) ------------
# Grid = list[list[str]] of cell text (empty string for blank cells).
Grid = list  # list[list[str]]


def _norm_cell(c: str) -> str:
    return re.sub(r"\s+", " ", _clean(c).lower()).strip()


def grits_content(pred: Grid, gold: Grid) -> float:
    """GriTS-lite (STUB) content score: position-aligned cell-content F1 over the
    overlapping grid region, penalised by shape mismatch. The real GriTS solves an
    optimal 2D alignment allowing spans (Smock et al., arXiv:2203.12555); swap it in
    for production. This positional approximation is adequate for smoke/CI and for a
    proposal-stage 'we can already measure structure' demonstration."""
    if not gold:
        return 0.0
    gr, gc = len(gold), max((len(r) for r in gold), default=0)
    pr, pc = len(pred), max((len(r) for r in pred), default=0)
    matches = 0
    total_gold = sum(1 for row in gold for c in row if _norm_cell(c))
    total_pred = sum(1 for row in pred for c in row if _norm_cell(c))
    for i in range(min(gr, pr)):
        for j in range(min(gc, pc)):
            g = _norm_cell(gold[i][j]) if j < len(gold[i]) else ""
            p = _norm_cell(pred[i][j]) if j < len(pred[i]) else ""
            if g and g == p:
                matches += 1
    precision = matches / total_pred if total_pred else 0.0
    recall = matches / total_gold if total_gold else 0.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


def grits_topology(pred: Grid, gold: Grid) -> float:
    """GriTS-lite (STUB) topology score: agreement on grid shape and which cells are
    occupied vs blank (a proxy for row/column/header structure). Real GriTS-Top
    compares cell adjacency relations including spans."""
    if not gold:
        return 0.0
    gr, gc = len(gold), max((len(r) for r in gold), default=0)
    pr, pc = len(pred), max((len(r) for r in pred), default=0)
    rows, cols = min(gr, pr), min(gc, pc)
    agree = 0
    denom = max(gr, pr) * max(gc, pc)
    for i in range(rows):
        for j in range(cols):
            g_occ = bool(_norm_cell(gold[i][j])) if j < len(gold[i]) else False
            p_occ = bool(_norm_cell(pred[i][j])) if j < len(pred[i]) else False
            if g_occ == p_occ:
                agree += 1
    shape_ok = agree
    return shape_ok / denom if denom else 0.0


def teds_struct(pred_html: str, gold_html: str) -> float:
    """TEDS (structure-only) interface. Uses the optional `apted`+`lxml` reference
    path if installed; otherwise raises so callers fall back to grits_* on grids.
    Kept as an interface so production can drop in the real metric without touching
    the harness."""
    try:
        import apted  # noqa: F401
        import lxml  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dep
        raise NotImplementedError(
            "teds_struct needs `apted` + `lxml`; install them for the reference TEDS, "
            "or use grits_content/grits_topology on Grid inputs (the built-in stub)."
        ) from exc
    # Reference implementation intentionally omitted from the stub; wire up
    # https://github.com/ibm-aur-nlp/PubTabNet TEDS here for production.
    raise NotImplementedError("Wire the reference TEDS implementation here for production.")


def is_close(a: float, b: float, tol: float = 1e-9) -> bool:
    return math.isclose(a, b, abs_tol=tol)
