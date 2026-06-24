"""Generate the static SVG figures for reports/architecture_evolution_analysis.md.

Pure stdlib (no matplotlib). All numbers are hardcoded from the verified eval
results with `# source:` comments; re-run to regenerate deterministically:

    .venv/Scripts/python.exe reports/figures/make_figures.py
"""
from pathlib import Path

OUT = Path(__file__).parent

# ---- palette -------------------------------------------------------------
INK = "#222222"
MUTED = "#666666"
GRID = "#dddddd"
AXIS = "#999999"
SHADE = "#f0ede6"
COL = {
    "precision": "#1D9E75",
    "recall": "#534AB7",
    "faithfulness": "#D85A30",
    "answer_correctness": "#185FA5",
    "gold_context_similarity": "#BA7517",
}
LOCAL = "#185FA5"
API = "#BA7517"
HARM = "#C0392B"
OKBOX = "#E6F1FB"
OKBORDER = "#185FA5"
HARMBOX = "#FCEBEB"

FONT = ("font-family=\"-apple-system,BHF,Segoe UI,Roboto,Helvetica,Arial,"
        "sans-serif\"")


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def txt(x, y, s, size=13, anchor="start", fill=INK, weight="400", rot=None):
    tr = f' transform="rotate({rot} {x} {y})"' if rot is not None else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
            f'text-anchor="{anchor}" fill="{fill}" font-weight="{weight}"{tr}>'
            f'{esc(s)}</text>')


def line(x1, y1, x2, y2, stroke=AXIS, w=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{w}"{d}/>')


def rect(x, y, w, h, fill, stroke="none", sw=1, rx=0):
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" rx="{rx}"/>')


def header(w, h, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" {FONT}>\n'
            f'<rect width="{w}" height="{h}" fill="white"/>\n'
            f'{txt(20, 28, title, size=16, weight="600")}\n')


def write(name, body):
    (OUT / name).write_text(body + "</svg>\n", encoding="utf-8")
    print("wrote", name)


# ==========================================================================
# Figure 1 — metric evolution r1 -> E8
# ==========================================================================
def fig1():
    W, H = 880, 440
    L, Rm, T, B = 60, 150, 50, 90
    plot_w, plot_h = W - L - Rm, H - T - B
    runs = ["r1\n(RAGAS)", "r2\n(DeepEval)", "r3", "r3\n+rerank",
            "E2\n512", "OSS\ndeepseek", "E4\ngpt-5.4", "E8\nextract", "E9c\nhybrid",
            "E12\nhyb+MiniLM", "E10\ndeepseek+hyb", "E11\ngpt5.4+hyb"]
    n = len(runs)
    # source: verify_numbers.py output + E9–E12 eval CSVs (None = metric absent that pass)
    series = {
        "precision":               [0.624, 0.618, 0.741, 0.837, 0.723, 0.742, 0.837, 0.825, 0.855, 0.867, 0.853, 0.865],
        "recall":                  [0.459, 0.539, 0.796, 0.837, 0.759, 0.804, 0.847, 0.864, 0.909, 0.920, 0.904, 0.914],
        "faithfulness":            [0.598, 0.978, 0.939, 0.940, 0.939, 0.983, 0.981, 0.974, 0.953, 0.950, 0.985, 0.984],
        "answer_correctness":      [0.412, 0.376, 0.473, 0.625, 0.488, 0.595, 0.652, 0.482, 0.689, 0.679, 0.717, 0.774],
        "gold_context_similarity": [None,  None,  0.741, 0.755, 0.775, 0.741, 0.755, 0.755, 0.756, 0.760, 0.756, 0.756],
    }
    labels = {"precision": "Ctx precision", "recall": "Ctx recall",
              "faithfulness": "Faithfulness", "answer_correctness": "Answer correctness",
              "gold_context_similarity": "Gold-ctx sim"}

    def X(i): return L + (plot_w * i / (n - 1))
    def Y(v): return T + plot_h * (1 - (v - 0.3) / (1.0 - 0.3))

    s = header(W, H, "Figure 1 — Metric evolution, r1 → E11 (Standard-family runs)")
    # shaded r1 region (RAGAS, not directly comparable)
    s += rect(L - 18, T, (X(0.5) - (L - 18)), plot_h, SHADE)
    s += txt(L - 12, T + plot_h + 64, "RAGAS — not comparable", size=10, fill=MUTED)
    # gridlines + y labels
    for g in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        y = Y(g)
        s += line(L, y, L + plot_w, y, stroke=GRID, w=1)
        s += txt(L - 8, y + 4, f"{g:.1f}", size=11, anchor="end", fill=MUTED)
    # divider before r3 ("canonical comparison begins")
    xdiv = (X(1) + X(2)) / 2
    s += line(xdiv, T, xdiv, T + plot_h, stroke=MUTED, w=1, dash="4,4")
    s += txt(xdiv + 4, T + 12, "canonical →", size=10, fill=MUTED)
    # x labels
    for i, r in enumerate(runs):
        for j, ln in enumerate(r.split("\n")):
            s += txt(X(i), T + plot_h + 18 + j * 13, ln, size=10, anchor="middle", fill=INK)
    # series
    for key, vals in series.items():
        pts = [(X(i), Y(v)) for i, v in enumerate(vals) if v is not None]
        d = " ".join(f"{'M' if k == 0 else 'L'}{x:.1f} {y:.1f}" for k, (x, y) in enumerate(pts))
        s += f'<path d="{d}" fill="none" stroke="{COL[key]}" stroke-width="2.2"/>\n'
        for x, y in pts:
            s += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{COL[key]}"/>'
    # legend
    ly = T + 6
    for key in series:
        s += rect(W - Rm + 6, ly - 9, 14, 4, COL[key])
        s += txt(W - Rm + 24, ly - 4, labels[key], size=11)
        ly += 20
    s += txt(W - Rm + 6, ly + 6, "AC: 0.41→0.77", size=10, fill=MUTED)
    s += txt(W - Rm + 6, ly + 20, "(best: E11 gpt5.4+hyb)", size=10, fill=MUTED)
    s += txt(20, H - 8, "Faithfulness jump r1→r2 is a framework change (RAGAS→DeepEval), not a real gain.",
             size=10, fill=MUTED)
    write("fig1_metric_evolution.svg", s)


# ==========================================================================
# Figure 2 — architecture flow per system
# ==========================================================================
def fig2():
    W, H = 760, 530
    s = header(W, H, "Figure 2 — Pipeline per architecture (red = the steps that cost quality/latency)")
    col_x = [20, 285, 530]
    col_w = 210
    titles = ["Basic  (19.5 s)", "Advanced v1  (74.1 s)", "Advanced v2  (101.8 s)"]
    steps = [
        [("Retrieve  k=4", "ok"), ("Generate", "ok"), ("Answer", "ok")],
        [("Retrieve  k=4", "ok"), ("Grade docs (gemma3:1b)", "harm"),
         ("Decompose (reuse docs)", "harm"), ("Generate per sub-q", "ok"),
         ("Grade gen  (loop ≤2)", "ok"), ("Synthesize  ≤100 words", "harm"),
         ("Answer", "ok")],
        [("Retrieve  k=4", "ok"), ("Grade docs (gemma3:1b)", "harm"),
         ("Decompose", "ok"), ("Per-sub-q retrieve k=4 + dedup", "harm"),
         ("Generate per sub-q", "ok"), ("Grade gen  (loop ≤2)", "ok"),
         ("Synthesize  (no cap)", "ok"), ("Answer", "ok")],
    ]
    box_h, gap = 38, 16
    for ci, (cx, title, col_steps) in enumerate(zip(col_x, titles, steps)):
        s += txt(cx + col_w / 2, 56, title, size=13, anchor="middle", weight="600")
        y = 72
        for k, (label, kind) in enumerate(col_steps):
            fill = HARMBOX if kind == "harm" else OKBOX
            border = HARM if kind == "harm" else OKBORDER
            tcol = HARM if kind == "harm" else "#0C447C"
            s += rect(cx, y, col_w, box_h, fill, stroke=border, sw=1.4, rx=6)
            # wrap long labels
            if len(label) > 26:
                parts = label.split("  ")
                s += txt(cx + col_w / 2, y + box_h / 2 - 2, parts[0], size=11, anchor="middle", fill=tcol, weight="500")
                s += txt(cx + col_w / 2, y + box_h / 2 + 12, "  ".join(parts[1:]), size=10, anchor="middle", fill=tcol)
            else:
                s += txt(cx + col_w / 2, y + box_h / 2 + 4, label, size=11, anchor="middle", fill=tcol, weight="500")
            if k < len(col_steps) - 1:
                ay = y + box_h
                s += line(cx + col_w / 2, ay, cx + col_w / 2, ay + gap, stroke=AXIS, w=1.4)
                s += (f'<path d="M{cx + col_w/2 - 4} {ay + gap - 5} L{cx + col_w/2} {ay + gap} '
                      f'L{cx + col_w/2 + 4} {ay + gap - 5}" fill="none" stroke="{AXIS}" stroke-width="1.4"/>')
            y += box_h + gap
    s += txt(20, H - 14, "All three share one retriever; the Advanced versions add grading + decomposition that remove "
             "signal without adding answer quality.", size=10, fill=MUTED)
    write("fig2_architecture_flow.svg", s)


# ==========================================================================
# Figure 3 — latency per run
# ==========================================================================
def fig3():
    W, H = 760, 380
    L, Rm, T, B = 60, 20, 50, 80
    plot_w, plot_h = W - L - Rm, H - T - B
    # source: verify_numbers.py latency means
    data = [("Basic", 19.5, "local"), ("Adv-v1", 74.1, "local"),
            ("Adv-v2", 101.8, "local"), ("Basic+rerank", 21.9, "local"),
            ("chunk-512", 11.3, "local"), ("deepseek", 62.7, "local"),
            ("gpt-5.4*", 1.6, "api"), ("extract", 19.4, "local")]
    vmax = 110
    n = len(data)
    bw = plot_w / n * 0.62
    s = header(W, H, "Figure 3 — Mean latency per question (seconds)")

    def Y(v): return T + plot_h * (1 - v / vmax)
    for g in [0, 25, 50, 75, 100]:
        y = Y(g)
        s += line(L, y, L + plot_w, y, stroke=GRID)
        s += txt(L - 8, y + 4, str(g), size=11, anchor="end", fill=MUTED)
    for i, (label, v, kind) in enumerate(data):
        cx = L + plot_w * (i + 0.5) / n
        x = cx - bw / 2
        y = Y(v)
        fill = API if kind == "api" else LOCAL
        s += rect(x, y, bw, T + plot_h - y, fill, rx=3)
        s += txt(cx, y - 6, f"{v:.1f}", size=11, anchor="middle", weight="500")
        for j, ln in enumerate(label.split(" ")):
            s += txt(cx, T + plot_h + 16 + j * 12, ln, size=10, anchor="middle")
    s += rect(L, H - 30, 12, 10, LOCAL); s += txt(L + 18, H - 21, "local Ollama (CPU)", size=10)
    s += rect(L + 160, H - 30, 12, 10, API); s += txt(L + 178, H - 21, "* API (hosted GPU — not comparable hardware)", size=10)
    write("fig3_latency.svg", s)


# ==========================================================================
# Figure 4 — CRAG grader recall/precision collapse
# ==========================================================================
def fig4():
    W, H = 620, 380
    L, Rm, T, B = 55, 20, 50, 80
    plot_w, plot_h = W - L - Rm, H - T - B
    # source: verify_numbers.py — fired(87)/kept(108) + standard baseline
    cats = ["Basic\n(no grader)", "Advanced v1 — grader\nkept all 4 (108)", "Advanced v1 — grader\ndropped chunks (87)"]
    recall = [0.796, 0.819, 0.486]
    precision = [0.741, 0.771, 0.590]
    n = len(cats)
    s = header(W, H, "Figure 4 — Advanced v1's gemma3:1b grader is net-negative when it fires")

    def Y(v): return T + plot_h * (1 - v / 1.0)
    for g in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = Y(g); s += line(L, y, L + plot_w, y, stroke=GRID)
        s += txt(L - 8, y + 4, f"{g:.1f}", size=11, anchor="end", fill=MUTED)
    grp = plot_w / n
    bw = grp * 0.3
    for i in range(n):
        base = L + grp * (i + 0.5)
        for off, vals, col, lbl in [(-bw * 0.55, recall, COL["recall"], "recall"),
                                    (bw * 0.55, precision, COL["precision"], "precision")]:
            v = vals[i]; x = base + off - bw / 2; y = Y(v)
            s += rect(x, y, bw, T + plot_h - y, col, rx=2)
            s += txt(base + off, y - 5, f"{v:.3f}", size=10, anchor="middle")
        for j, ln in enumerate(cats[i].split("\n")):
            s += txt(base, T + plot_h + 16 + j * 12, ln, size=10, anchor="middle")
    s += rect(L, H - 28, 12, 10, COL["recall"]); s += txt(L + 18, H - 19, "recall", size=10)
    s += rect(L + 90, H - 28, 12, 10, COL["precision"]); s += txt(L + 108, H - 19, "precision", size=10)
    write("fig4_crag_recall_collapse.svg", s)


# ==========================================================================
# Figure 5 — rerank win on worst-case rows
# ==========================================================================
def fig5():
    W, H = 560, 380
    L, Rm, T, B = 55, 20, 50, 70
    plot_w, plot_h = W - L - Rm, H - T - B
    # source: verify_numbers.py — baseline P<0.5 rows (n=40)
    groups = ["Ctx precision", "Answer correctness"]
    baseline = [0.127, 0.185]
    rerank = [0.531, 0.333]
    n = len(groups)
    s = header(W, H, "Figure 5 — Reranking on the 40 worst-retrieval rows (baseline P<0.5)")

    def Y(v): return T + plot_h * (1 - v / 0.6)
    for g in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        y = Y(g); s += line(L, y, L + plot_w, y, stroke=GRID)
        s += txt(L - 8, y + 4, f"{g:.1f}", size=11, anchor="end", fill=MUTED)
    grp = plot_w / n
    bw = grp * 0.26
    for i in range(n):
        base = L + grp * (i + 0.5)
        for off, vals, col, lbl in [(-bw * 0.6, baseline, MUTED, "baseline"),
                                    (bw * 0.6, rerank, LOCAL, "rerank")]:
            v = vals[i]; x = base + off - bw / 2; y = Y(v)
            s += rect(x, y, bw, T + plot_h - y, col, rx=2)
            s += txt(base + off, y - 5, f"{v:.3f}", size=11, anchor="middle", weight="500")
        s += txt(base, T + plot_h + 18, groups[i], size=12, anchor="middle")
    s += rect(L, H - 26, 12, 10, MUTED); s += txt(L + 18, H - 17, "baseline (top-4 dense)", size=10)
    s += rect(L + 180, H - 26, 12, 10, LOCAL); s += txt(L + 198, H - 17, "rerank (top-20→cross-encoder→4)", size=10)
    write("fig5_rerank_worstcase.svg", s)


# ==========================================================================
# Figure 6 — r3 canonical 3-system comparison (Basic vs Advanced v1 vs v2)
# ==========================================================================
def fig6():
    W, H = 720, 400
    L, Rm, T, B = 55, 20, 50, 80
    plot_w, plot_h = W - L - Rm, H - T - B
    # source: experiments_index.md headline table (llama3:8b, n=195)
    groups = ["Ctx precision", "Ctx recall", "Faithfulness", "Answer correctness"]
    systems = [("Basic", COL["answer_correctness"]),
               ("Advanced v1", COL["faithfulness"]),
               ("Advanced v2", COL["recall"])]
    data = {  # per group: [basic, adv-v1, adv-v2]
        "Ctx precision":      [0.741, 0.690, 0.449],
        "Ctx recall":         [0.796, 0.671, 0.650],
        "Faithfulness":       [0.939, 0.948, 0.910],
        "Answer correctness": [0.473, 0.474, 0.334],
    }
    n = len(groups)
    s = header(W, H, "Figure 6 — r3 canonical comparison (llama3:8b, n=195)")

    def Y(v): return T + plot_h * (1 - v / 1.0)
    for g in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = Y(g); s += line(L, y, L + plot_w, y, stroke=GRID)
        s += txt(L - 8, y + 4, f"{g:.1f}", size=11, anchor="end", fill=MUTED)
    grp = plot_w / n
    bw = grp * 0.22
    for i, gname in enumerate(groups):
        base = L + grp * (i + 0.5)
        offs = [-bw * 1.15, 0, bw * 1.15]
        for j, (sname, col) in enumerate(systems):
            v = data[gname][j]; x = base + offs[j] - bw / 2; y = Y(v)
            s += rect(x, y, bw, T + plot_h - y, col, rx=2)
            s += txt(base + offs[j], y - 5, f"{v:.2f}", size=9, anchor="middle")
        s += txt(base, T + plot_h + 18, gname, size=11, anchor="middle")
    lx = L
    for sname, col in systems:
        s += rect(lx, H - 26, 12, 10, col); s += txt(lx + 16, H - 17, sname, size=10)
        lx += 120
    s += txt(20, H - 2, "Basic wins/ties 4 of 5 metrics; Advanced v2 (CRAG++) regresses on precision and AC.",
             size=10, fill=MUTED)
    write("fig6_r3_comparison.svg", s)


# ==========================================================================
# Figure 7 — value-added ablation (is RAG worth it?)
# ==========================================================================
def fig7():
    W, H = 680, 400
    L, Rm, T, B = 55, 20, 50, 96
    plot_w, plot_h = W - L - Rm, H - T - B
    # source: experiments_index.md value-added ablation (AC, n=195)
    bars = [("Closed-book\nllama3:8b (no retr.)", 0.202, MUTED, False),
            ("Closed-book\ngpt-5.4-mini (no retr.)", 0.490, MUTED, False),
            ("Retrieval-only\n(chunks as answer)", 0.827, HARM, True),
            ("Full RAG\n(Basic, local)", 0.689, LOCAL, False),
            ("Full RAG\n(best, gpt-5.4-mini)", 0.774, API, False)]
    n = len(bars)
    bw = plot_w / n * 0.5
    s = header(W, H, "Figure 7 — Is RAG worth it? Answer correctness by condition (n=195)")

    def Y(v): return T + plot_h * (1 - v / 1.0)
    for g in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = Y(g); s += line(L, y, L + plot_w, y, stroke=GRID)
        s += txt(L - 8, y + 4, f"{g:.1f}", size=11, anchor="end", fill=MUTED)
    for i, (label, v, col, caveat) in enumerate(bars):
        cx = L + plot_w * (i + 0.5) / n
        x = cx - bw / 2; y = Y(v)
        s += rect(x, y, bw, T + plot_h - y, col, rx=3)
        s += txt(cx, y - 6, f"{v:.3f}", size=12, anchor="middle", weight="600")
        if caveat:
            s += txt(cx, y - 22, "⚠ coverage artifact", size=9, anchor="middle", fill=HARM)
        for j, ln in enumerate(label.split("\n")):
            s += txt(cx, T + plot_h + 16 + j * 12, ln, size=10, anchor="middle")
    # retrieval arrow annotation
    s += txt(20, H - 30, "Retrieval lifts AC +0.49 (open) and +0.28 (closed) over closed-book — RAG is worth it for both.",
             size=10, fill=INK)
    s += txt(20, H - 14, "Retrieval-only > Full RAG is a GEval coverage artifact (a 4k-char chunk dump contains the gold "
             "facts, unpenalised for verbosity) — not better answers.", size=10, fill=MUTED)
    write("fig7_value_added.svg", s)


# ==========================================================================
# Figure 8 — eval-set question coverage (cognitive level / subtype)
# ==========================================================================
def fig8():
    W, H = 680, 360
    L, Rm, T, B = 200, 60, 50, 70
    plot_w, plot_h = W - L - Rm, H - T - B
    # source: scripts/classify_questions.py — results/question_classification.csv (n=195)
    rows = [("factual_lookup", 60.0, "fact"),
            ("explanatory_why", 17.4, "reason"),
            ("procedural_how", 9.2, "reason"),
            ("definitional", 7.7, "fact"),
            ("comparative_analytical", 5.6, "reason")]
    COLF, COLR = COL["precision"], COL["recall"]
    vmax = 65
    n = len(rows)
    s = header(W, H, "Figure 8 — Eval-set question types (n=195): 68% fact / 32% reasoning")
    bh = plot_h / n * 0.62

    def X(v): return L + plot_w * v / vmax
    for g in [0, 20, 40, 60]:
        x = X(g); s += line(x, T, x, T + plot_h, stroke=GRID)
        s += txt(x, T + plot_h + 16, f"{g}%", size=10, anchor="middle", fill=MUTED)
    for i, (label, v, kind) in enumerate(rows):
        cy = T + plot_h * (i + 0.5) / n
        col = COLF if kind == "fact" else COLR
        s += rect(L, cy - bh / 2, X(v) - L, bh, col, rx=2)
        s += txt(L - 8, cy + 4, label, size=11, anchor="end")
        s += txt(X(v) + 6, cy + 4, f"{v:.1f}%", size=10, fill=INK)
    s += rect(L, H - 24, 12, 10, COLF); s += txt(L + 16, H - 15, "fact-retrieval", size=10)
    s += rect(L + 110, H - 24, 12, 10, COLR); s += txt(L + 126, H - 15, "reasoning (why/how/analysis)", size=10)
    s += txt(20, T + plot_h + 40, "Methodology topics the reviewer named (sampling 4.6% + plausible values 3.1% + "
             "weighting 1.5% ≈ 9%) are under-sampled.", size=10, fill=MUTED)
    write("fig8_question_coverage.svg", s)


# ==========================================================================
# Architecture flowcharts — the Advanced pipelines AS BUILT (v1, v2, v3)
# source of truth:
#   src/nodes/advrag_nodes.py            (v1 = CRAG)
#   src/nodes/cragpp_nodes.py            (v2 = CRAG++, v3 = rerank-aware)
#   src/graph_builder/graph_builder_adv.py, graph_builder_cragpp.py
# ==========================================================================
import math

STEPFILL = "#ffffff"        # ordinary step
NEWFILL = "#E6F1FB"          # new / changed step (blue highlight)
NEWBORDER = "#185FA5"


def _fbox(x, y, w, h, title, sub=None, fill=STEPFILL, border=INK, tcol=INK):
    s = rect(x, y, w, h, fill, stroke=border, sw=1.4, rx=6)
    if sub:
        s += txt(x + w / 2, y + 18, title, size=11.5, anchor="middle", fill=tcol, weight="600")
        s += txt(x + w / 2, y + 34, sub, size=9.5, anchor="middle", fill=MUTED)
    else:
        s += txt(x + w / 2, y + h / 2 + 4, title, size=11.5, anchor="middle", fill=tcol, weight="600")
    return s


def _vdown(x, y1, y2, color=AXIS):
    s = line(x, y1, x, y2 - 6, stroke=color, w=1.5)
    s += f'<path d="M{x-4:.1f} {y2-6:.1f} L{x+4:.1f} {y2-6:.1f} L{x:.1f} {y2:.1f} Z" fill="{color}"/>'
    return s


def _arrow(x1, y1, x2, y2, color=AXIS, dash=None, label=None):
    s = line(x1, y1, x2, y2, stroke=color, w=1.5, dash=dash)
    ang = math.atan2(y2 - y1, x2 - x1)
    L, wd = 7.0, 3.5
    bx1 = x2 - L * math.cos(ang) + wd * math.sin(ang)
    by1 = y2 - L * math.sin(ang) - wd * math.cos(ang)
    bx2 = x2 - L * math.cos(ang) - wd * math.sin(ang)
    by2 = y2 - L * math.sin(ang) + wd * math.cos(ang)
    s += f'<path d="M{bx1:.1f} {by1:.1f} L{bx2:.1f} {by2:.1f} L{x2:.1f} {y2:.1f} Z" fill="{color}"/>'
    if label:
        s += txt((x1 + x2) / 2, (y1 + y2) / 2 - 5, label, size=9, anchor="middle", fill=MUTED)
    return s


def _retry_loop(bx, bw, gy, ty, label="retry ≤ 2"):
    """Dashed loop on the right margin from a grade box (gy) back up to a
    generate box (ty)."""
    rx = bx + bw + 34
    s = line(bx + bw, gy, rx, gy, stroke=MUTED, w=1.3, dash="4,3")
    s += line(rx, gy, rx, ty, stroke=MUTED, w=1.3, dash="4,3")
    s += line(rx, ty, bx + bw + 6, ty, stroke=MUTED, w=1.3, dash="4,3")
    s += f'<path d="M{bx+bw+6:.1f} {ty-4:.1f} L{bx+bw+6:.1f} {ty+4:.1f} L{bx+bw:.1f} {ty:.1f} Z" fill="{MUTED}"/>'
    s += txt(rx + 5, (gy + ty) / 2, label, size=9, anchor="start", fill=MUTED, rot=0)
    return s


def fig_adv_v1():
    W, bx, bw = 600, 140, 320
    cx = bx + bw / 2
    steps = [
        ("Query", None),
        ("Retrieve top-k  (k = 4)", "dense similarity over the index"),
        ("Grade documents", "gemma3:1b binary yes / no  ·  keep-all fallback"),
        ("Decompose into 2–3 sub-questions", "reuse the original query's documents"),
        ("Answer each sub-question", "llama3:8b"),
        ("Synthesize final answer", "≤ 100 words"),
        ("Grade answer (usefulness)", "gemma3:1b"),
        ("Answer", None),
    ]
    gap = 24
    hts = [32 if sub is None else 46 for _, sub in steps]
    ys, y = [], 50
    for h in hts:
        ys.append(y); y += h + gap
    H = y + 30
    s = header(W, H, "Figure — Advanced v1 (CRAG), as built")
    for i in range(len(steps) - 1):
        s += _vdown(cx, ys[i] + hts[i], ys[i + 1])
    for (title, sub), yy, hh in zip(steps, ys, hts):
        s += _fbox(bx, yy, bw, hh, title, sub)
    # retry loop: grade answer (idx 6) -> answer each sub-question (idx 4)
    s += _retry_loop(bx, bw, ys[6] + hts[6] / 2, ys[4] + hts[4] / 2)
    s += txt(20, H - 10, "No reranker and no per-sub-question retrieval; the grader can only drop chunks.",
             size=10, fill=MUTED)
    write("fig_adv_v1.svg", s)


def fig_adv_v2():
    W, bx, bw = 620, 150, 320
    cx = bx + bw / 2
    steps = [
        ("Query", None, False),
        ("Retrieve top-k  (k = 4)", "dense similarity", False),
        ("Grade documents", "gemma3:1b binary yes / no", False),
        ("Decompose into 2–3 sub-questions", "llama3:8b", False),
        ("Retrieve fresh documents per sub-question", "new in v2  ·  then de-duplicate", True),
        ("Union of sub-question contexts", "de-duplicated", False),
        ("Answer each sub-question", "llama3:8b", False),
        ("Synthesize final answer", "no word cap", False),
        ("Grade answer (usefulness)", "gemma3:1b", False),
        ("Answer", None, False),
    ]
    gap = 22
    hts = [32 if sub is None else 46 for _, sub, _ in steps]
    ys, y = [], 50
    for h in hts:
        ys.append(y); y += h + gap
    H = y + 30
    s = header(W, H, "Figure — Advanced v2 (CRAG++), as built")
    for i in range(len(steps) - 1):
        s += _vdown(cx, ys[i] + hts[i], ys[i + 1])
    for (title, sub, hot), yy, hh in zip(steps, ys, hts):
        s += _fbox(bx, yy, bw, hh, title, sub,
                   fill=(NEWFILL if hot else STEPFILL), border=(NEWBORDER if hot else INK))
    # retry loop: grade answer (idx 8) -> answer each sub-question (idx 6)
    s += _retry_loop(bx, bw, ys[8] + hts[8] / 2, ys[6] + hts[6] / 2)
    s += txt(20, H - 10, "Per-sub-question retrieval widens the context with off-topic chunks, lowering precision.",
             size=10, fill=MUTED)
    write("fig_adv_v2.svg", s)


def fig_adv_v3():
    W, H = 760, 760
    cx = 360
    s = header(W, H, "Figure — Advanced v3 (rerank-aware), as built")
    # --- shared top column (centered) ---
    cw = 320; cbx = cx - cw / 2
    s += _fbox(cbx, 50, cw, 32, "Query")
    s += _vdown(cx, 82, 108)
    s += _fbox(cbx, 108, cw, 46, "Retrieve wide  (top-20)", "dense or hybrid (BM25 + dense, RRF)")
    s += _vdown(cx, 154, 192)
    s += _fbox(cbx, 192, cw, 46, "Rerank + score-grade  (T2.3)",
               "cross-encoder (bge); keep score ≥ threshold", fill=NEWFILL, border=NEWBORDER)
    s += _vdown(cx, 238, 280)
    # decision box (wide)
    dw = 420; dbx = cx - dw / 2
    s += _fbox(dbx, 280, dw, 48, "Adaptive route  (T2.5)",
               "single-hop if top-1 score ≥ 0.7 and margin ≥ 0, else multi-hop",
               fill=NEWFILL, border=NEWBORDER)
    # --- branch ---
    # left: single-hop
    lcx, lw = 190, 230; lbx = lcx - lw / 2
    # right: multi-hop
    rcx, rw = 545, 280; rbx = rcx - rw / 2
    s += _arrow(cx - 70, 328, lcx, 392, label="single-hop")
    s += _arrow(cx + 70, 328, rcx, 392, label="multi-hop")
    s += _fbox(lbx, 392, lw, 48, "Answer directly", "from the reranked top-4")
    s += _fbox(rbx, 392, rw, 46, "Decompose + retrieve per sub-question", "then de-duplicate the union")
    s += _vdown(rcx, 438, 470)
    s += _fbox(rbx, 470, rw, 48, "Rerank union vs the original query  (T2.4)",
               "keep top-6", fill=NEWFILL, border=NEWBORDER)
    s += _vdown(rcx, 518, 550)
    s += _fbox(rbx, 550, rw, 48, "Synthesize from reranked context (T3.7)",
               "context + sub-answers; no word cap", fill=NEWFILL, border=NEWBORDER)
    # converge into grade
    gcx, gw = 360, 240; gbx = gcx - gw / 2
    s += _arrow(lcx, 440, gcx - 40, 632)
    s += _arrow(rcx, 598, gcx + 40, 632)
    s += _fbox(gbx, 632, gw, 32, "Grade answer (usefulness)")
    s += _vdown(gcx, 664, 694)
    s += _fbox(gbx, 694, gw, 32, "Answer")
    # path-aware retry loop on far right
    rx = rbx + rw + 22
    s += line(gcx + gw / 2, 648, rx, 648, stroke=MUTED, w=1.3, dash="4,3")
    s += line(rx, 648, rx, 415, stroke=MUTED, w=1.3, dash="4,3")
    s += line(rx, 415, rbx + rw + 6, 415, stroke=MUTED, w=1.3, dash="4,3")
    s += f'<path d="M{rbx+rw+6:.1f} {411:.1f} L{rbx+rw+6:.1f} {419:.1f} L{rbx+rw:.1f} {415:.1f} Z" fill="{MUTED}"/>'
    s += txt(rx + 4, 540, "retry ≤ 2", size=9, anchor="start", fill=MUTED)
    s += txt(20, H - 12, "Blue = the four rerank-aware fixes (T2.3 grader, T2.5 routing, T2.4 union-rerank, "
             "T3.7 synthesis). Without a reranker the pipeline falls back to v2.", size=10, fill=MUTED)
    write("fig_adv_v3.svg", s)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7(); fig8()
    fig_adv_v1(); fig_adv_v2(); fig_adv_v3()
    print("done")
