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
    titles = ["Standard  (19.5 s)", "CRAG  (74.1 s)", "CRAG++  (101.8 s)"]
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
    s += txt(20, H - 14, "All three share one retriever; CRAG/CRAG++ add grading + decomposition that remove signal "
             "without adding answer quality.", size=10, fill=MUTED)
    write("fig2_architecture_flow.svg", s)


# ==========================================================================
# Figure 3 — latency per run
# ==========================================================================
def fig3():
    W, H = 760, 380
    L, Rm, T, B = 60, 20, 50, 80
    plot_w, plot_h = W - L - Rm, H - T - B
    # source: verify_numbers.py latency means
    data = [("Standard", 19.5, "local"), ("CRAG", 74.1, "local"),
            ("CRAG++", 101.8, "local"), ("Std+rerank", 21.9, "local"),
            ("E2 512", 11.3, "local"), ("OSS deepseek", 62.7, "local"),
            ("E4 gpt-5.4*", 1.6, "api"), ("E8 extract", 19.4, "local")]
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
    cats = ["Standard\n(no grader)", "CRAG — grader\nkept all 4 (108)", "CRAG — grader\ndropped chunks (87)"]
    recall = [0.796, 0.819, 0.486]
    precision = [0.741, 0.771, 0.590]
    n = len(cats)
    s = header(W, H, "Figure 4 — CRAG's gemma3:1b grader is net-negative when it fires")

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


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("done")
