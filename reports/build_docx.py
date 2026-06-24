"""Build reports/revised_final_report.docx from revised_final_report.md.

House style matched to the original January report: Times New Roman body,
heading colour 0F4761 (H1/H2/H3 = 20/16/14 pt), Title 28 pt, US Letter, 1" margins.
SVG figures are embedded from their rasterised PNGs in reports/figures/png/.
"""
import re
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = Path(__file__).parent
MD = ROOT / "revised_final_report.md"
OUT = ROOT / "revised_final_report.docx"
HEADCLR = RGBColor(0x0F, 0x47, 0x61)
CONTENT_DXA = 9360  # 6.5" at 1" margins on US Letter

# ---- image width (inches) by filename keyword ----
def img_width(path):
    n = path.lower()
    if any(k in n for k in ("fig_adv_v1", "fig_adv_v2", "fig_adv_v3")):
        return 4.8
    if "orig_fig1_basic" in n or "orig_fig2_advanced" in n:
        return 5.4
    if "ui.png" in n or "orig_fig8" in n or "orig_fig9" in n:
        return 6.2
    return 5.8  # data charts

def map_image(path):
    """figures/foo.svg -> figures/png/foo.png ; others unchanged."""
    if path.endswith(".svg"):
        name = path.split("/")[-1][:-4] + ".png"
        return "figures/png/" + name
    return path

# ---- inline formatting ----
TOKEN = re.compile(r'\*\*(.+?)\*\*|`([^`]+)`|\[([^\]]+)\]\(([^)]+)\)|\*(.+?)\*')

def add_runs(par, text, base_italic=False, size=None):
    pos = 0
    def style(r):
        if size: r.font.size = Pt(size)
        if base_italic: r.italic = True
    for m in TOKEN.finditer(text):
        if m.start() > pos:
            style(par.add_run(text[pos:m.start()]))
        if m.group(1) is not None:
            r = par.add_run(m.group(1)); r.bold = True; style(r)
        elif m.group(2) is not None:
            r = par.add_run(m.group(2)); r.font.name = "Consolas"
            r.font.size = Pt((size or 11) - 1)
        elif m.group(3) is not None:
            r = par.add_run(m.group(3)); style(r)
        elif m.group(5) is not None:
            r = par.add_run(m.group(5)); r.italic = True; style(r)
        pos = m.end()
    if pos < len(text):
        style(par.add_run(text[pos:]))

def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd'); shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:fill'), fill); tcPr.append(shd)

def set_cell_width(cell, dxa):
    cell.width = Inches(dxa / 1440)
    tcPr = cell._tc.get_or_add_tcPr()
    w = OxmlElement('w:tcW'); w.set(qn('w:w'), str(dxa)); w.set(qn('w:type'), 'dxa')
    tcPr.append(w)

def add_page_number_footer(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("Page ").font.size = Pt(9)
    fld = OxmlElement('w:fldSimple'); fld.set(qn('w:instr'), 'PAGE')
    run = OxmlElement('w:r'); rpr = OxmlElement('w:rPr')
    sz = OxmlElement('w:sz'); sz.set(qn('w:val'), '18'); rpr.append(sz)
    run.append(rpr); fld.append(run)
    p._p.append(fld)

# ---- document setup ----
doc = Document()
normal = doc.styles['Normal']
normal.font.name = "Times New Roman"
normal.font.size = Pt(11)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.15
for lvl, sz in (("Title", 28), ("Heading 1", 20), ("Heading 2", 16), ("Heading 3", 14)):
    s = doc.styles[lvl]
    s.font.name = "Times New Roman"
    s.font.bold = True
    s.font.size = Pt(sz)
    if lvl != "Title":
        s.font.color.rgb = HEADCLR

sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.top_margin = sec.bottom_margin = sec.left_margin = sec.right_margin = Inches(1)
add_page_number_footer(sec)

# ---- parse markdown ----
lines = MD.read_text(encoding="utf-8").split("\n")
i = 0
seen_h2 = False           # have we passed the title block's first ## ?
title_block_done = False

def is_caption_or_formula(s):
    return s.startswith("*") and s.rstrip().endswith("*") and s.count("*") == 2

while i < len(lines):
    raw = lines[i]
    line = raw.rstrip()
    s = line.strip()

    if not s:
        i += 1; continue

    # horizontal rule: first one ends the title block (page break); others skipped
    if re.match(r'^-{3,}$', s):
        if not title_block_done:
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
            title_block_done = True
        i += 1; continue

    # table block
    if s.startswith("|"):
        block = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            block.append(lines[i].strip()); i += 1
        # header + separator + rows
        def cells(row):
            parts = [c.strip() for c in row.split("|")]
            return parts[1:-1] if parts and parts[0] == "" else parts
        header = cells(block[0])
        body = [cells(r) for r in block[2:]]
        ncol = len(header)
        tbl = doc.add_table(rows=1, cols=ncol)
        tbl.style = "Table Grid"
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        colw = CONTENT_DXA // ncol
        for j, h in enumerate(header):
            c = tbl.rows[0].cells[j]
            c.paragraphs[0].text = ""
            add_runs(c.paragraphs[0], h, size=10)
            for r in c.paragraphs[0].runs: r.bold = True
            shade(c, "D5E8F0"); set_cell_width(c, colw)
            c.paragraphs[0].paragraph_format.space_after = Pt(2)
        for row in body:
            cs = tbl.add_row().cells
            for j in range(ncol):
                cell = cs[j]
                cell.paragraphs[0].text = ""
                add_runs(cell.paragraphs[0], row[j] if j < len(row) else "", size=10)
                set_cell_width(cell, colw)
                cell.paragraphs[0].paragraph_format.space_after = Pt(2)
        doc.add_paragraph().paragraph_format.space_after = Pt(4)
        continue

    # image
    mimg = re.match(r'^!\[(.*?)\]\((.+?)\)\s*$', s)
    if mimg:
        path = map_image(mimg.group(2))
        fp = ROOT / path
        if fp.exists():
            doc.add_picture(str(fp), width=Inches(img_width(path)))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            doc.paragraphs[-1].paragraph_format.space_before = Pt(6)
            doc.paragraphs[-1].paragraph_format.space_after = Pt(2)
        i += 1; continue

    # headings
    mh = re.match(r'^(#{1,4})\s+(.*)$', s)
    if mh:
        level = len(mh.group(1)); txt = mh.group(2)
        if level == 1:
            p = doc.add_paragraph(style="Title"); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_runs(p, txt)
        elif level == 3 and not seen_h2 and not title_block_done:
            # subtitle (before any ## and before title block ends)
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_runs(p, txt, base_italic=True, size=13)
        else:
            if level == 2: seen_h2 = True
            style = {2: "Heading 1", 3: "Heading 2", 4: "Heading 3"}[level]
            p = doc.add_paragraph(style=style)
            add_runs(p, txt)
        i += 1; continue

    # caption / display formula (whole line italic) -> centred italic
    if is_caption_or_formula(line):
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        inner = line.strip()[1:-1]
        is_fig = inner.startswith("Figure")
        add_runs(p, inner, base_italic=True, size=9.5 if is_fig else 11)
        if is_fig:
            for r in p.runs: r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
        i += 1; continue

    # ordered list item (keep literal number, hanging indent)
    if re.match(r'^\d+\.\s', s):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.first_line_indent = Inches(-0.3)
        add_runs(p, s)
        i += 1; continue

    # bullet
    if s.startswith("- "):
        p = doc.add_paragraph(style="List Bullet")
        add_runs(p, s[2:])
        i += 1; continue

    # title-block meta lines (centered) before title block ends
    if not title_block_done and (s.startswith("**Prepared") or s.startswith("**Revised")):
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_runs(p, s)
        i += 1; continue

    # normal paragraph
    p = doc.add_paragraph()
    add_runs(p, s)
    i += 1

doc.core_properties.title = "Information Retrieval Using RAG on PIRLS Documents"
doc.core_properties.author = "Widianto Persadha, Heiko Sibberns, Mohammad S. Thariq, Bettina Wietzorek"
doc.save(str(OUT))
print("wrote", OUT)
