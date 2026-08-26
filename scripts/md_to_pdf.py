"""Minimal, robust Markdown -> PDF for this project's notes.

Uses fpdf2 (2.7+) + system DejaVu fonts (the only working toolchain here;
tectonic/pandoc/pdflatex are all broken per CLAUDE.md). Handles: #/##/### headings,
--- rules, - bullets, **bold** inline (fpdf markdown), and | pipe | tables via the
auto-wrapping table() API (avoids the O9 fixed-width-cell truncation trap).

Usage: python scripts/md_to_pdf.py <in.md> <out.pdf> ["Title"]
"""
import sys
import re
from fpdf import FPDF

FONT_DIR = "/usr/share/fonts/dejavu-sans-fonts"


def _clean(s):
    # glyphs DejaVuSans lacks / that render as tofu -> safe substitutes
    return s.replace("⚠", "!").replace("→", "->").replace("≈", "~").replace("≥", ">=") \
            .replace("≤", "<=").replace("≪", "<<").replace("≫", ">>")


def _md_inline(s):
    # keep **bold** for fpdf markdown=True; drop backticks (render as plain)
    return _clean(s).replace("`", "")


class PDF(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-12)
        self.set_font("DejaVu", "", 7)
        self.set_text_color(140)
        self.cell(0, 8, f"{self.page_no()}", align="C")
        self.set_text_color(0)


def build(md_path, pdf_path, title=None):
    with open(md_path) as f:
        lines = f.read().split("\n")

    pdf = PDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(16, 14, 16)
    pdf.add_font("DejaVu", "", f"{FONT_DIR}/DejaVuSans.ttf")
    pdf.add_font("DejaVu", "B", f"{FONT_DIR}/DejaVuSans-Bold.ttf")
    pdf.add_font("DejaVu", "I", f"{FONT_DIR}/DejaVuSans-Oblique.ttf")
    pdf.add_page()

    def para(txt, size=9.5, style="", gap=1.6, color=(0, 0, 0)):
        pdf.set_font("DejaVu", style, size)
        pdf.set_text_color(*color)
        pdf.multi_cell(0, size * 0.52, _md_inline(txt), markdown=True)
        pdf.ln(gap)

    i = 0
    while i < len(lines):
        ln = lines[i].rstrip()
        s = ln.strip()

        # tables: a run of lines starting with '|'
        if s.startswith("|"):
            tbl = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl.append(lines[i].strip())
                i += 1
            rows = []
            for r in tbl:
                cells = [c.strip() for c in r.strip("|").split("|")]
                if all(re.fullmatch(r":?-{2,}:?", c or "-") for c in cells):
                    continue  # separator row
                rows.append(cells)
            if rows:
                ncol = max(len(r) for r in rows)
                fs = 8.0 if ncol <= 4 else (7.2 if ncol <= 6 else 6.4)
                pdf.set_font("DejaVu", "", fs)
                pdf.set_draw_color(180)
                with pdf.table(markdown=True, first_row_as_headings=True,
                               line_height=fs * 0.95, width=pdf.epw) as table:
                    for ri, r in enumerate(rows):
                        row = table.row()
                        r = r + [""] * (ncol - len(r))
                        for c in r:
                            row.cell(_md_inline(c))
                pdf.ln(2.2)
            continue

        if not s:
            pdf.ln(1.4)
            i += 1
            continue

        if s.startswith("### "):
            para(s[4:], size=10.5, style="B", gap=1.0)
        elif s.startswith("## "):
            pdf.ln(1.5)
            para(s[3:], size=12.5, style="B", gap=1.2, color=(20, 40, 90))
        elif s.startswith("# "):
            para(s[2:], size=15, style="B", gap=2.2, color=(10, 20, 60))
        elif s in ("---", "***", "___"):
            pdf.set_draw_color(200)
            y = pdf.get_y()
            pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
            pdf.ln(2.5)
        elif s.startswith("- ") or s.startswith("* "):
            pdf.set_font("DejaVu", "", 9.5)
            pdf.set_x(pdf.l_margin + 3)
            pdf.multi_cell(0, 9.5 * 0.52, "•  " + _md_inline(s[2:]), markdown=True)
            pdf.ln(0.6)
        else:
            para(s)
        i += 1

    pdf.output(pdf_path)
    print(f"[md_to_pdf] wrote {pdf_path} ({pdf.page_no()} pages)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: python scripts/md_to_pdf.py <in.md> <out.pdf> [title]")
    build(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
