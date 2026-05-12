# Word/Docx Conventions

For docx generators (python-docx). See also [guardrails.md](guardrails.md) — rules T1, D3, F1-F4 apply to docx too.

## Header

- Logo on the **left** margin
- Project/institution name on the **right** margin via tab stop
- 8pt Calibri, gray italic

```python
text_width = section.page_width - section.left_margin - section.right_margin
hp.paragraph_format.tab_stops.add_tab_stop(text_width, WD_TAB_ALIGNMENT.RIGHT)
hp.add_run("\t")
run = hp.add_run("Your Project Name")
```

## Cover Page

1. Two spacer paragraphs at top
2. Logo centered, 2.0–2.5 inches — use `doc.add_picture()` (not `run.add_picture()`)
3. Title, 48pt bold, primary color
4. Subtitle, 22pt
5. Author name, 14pt, centered
6. Metadata key-value pairs (centered)
7. Page break

## Table of Contents

Use a **standard Word TOC field**, not manual hyperlinks — because manual lists go stale and don't update on page-number shifts. The XML is verbose but mandatory; python-docx has no high-level API for it:

```python
paragraph = doc.add_paragraph()
run = paragraph.add_run()
# begin field
fld_begin = OxmlElement('w:fldChar')
fld_begin.set(qn('w:fldCharType'), 'begin')
run._r.append(fld_begin)
instr = OxmlElement('w:instrText')
instr.set(qn('xml:space'), 'preserve')
instr.text = ' TOC \\o "1-2" \\h \\z \\u '
run._r.append(instr)
# separate
fld_sep = OxmlElement('w:fldChar')
fld_sep.set(qn('w:fldCharType'), 'separate')
run._r.append(fld_sep)
# placeholder entries (shown until Word updates the field)
for title in toc_titles:
    placeholder = paragraph.add_run(title + "\n")
# end field
fld_end_run = paragraph.add_run()
fld_end = OxmlElement('w:fldChar')
fld_end.set(qn('w:fldCharType'), 'end')
fld_end_run._r.append(fld_end)
```

Set `updateFields` so Word auto-populates on open:

```python
settings = doc.settings.element
update_fields = OxmlElement('w:updateFields')
update_fields.set(qn('w:val'), 'true')
settings.append(update_fields)
```

Push the TOC heading down with 3 zero-height spacer paragraphs (avoids it sitting flush against the top margin).

## Page Breaks After Tables

When a table is the last element before a page break, use a tight break — otherwise python-docx inserts a default-height empty paragraph that renders as a visible blank gap:

```python
def _tight_page_break(doc):
    last_p = doc.paragraphs[-1]
    last_p.add_run().add_break(WD_BREAK.PAGE)
```

## General Rules

- Font: Calibri throughout
- A4 page size (21.0 × 29.7 cm), 2.5 cm side margins, 2.0 cm top/bottom
- Page numbers in footer (centered, 8pt gray)
- Body text: 11pt minimum
- Table cells: 10pt minimum
- Heading hierarchy with color cascade (primary → lighter shades for H2, H3, H4)
- Table styling: dark header row with white text, zebra striping, 8–10pt cell padding
- Code blocks: light grey background (`#f5f5f5`)

## Color Template

Replace with project brand colors:

```python
C_PRIMARY = "#1A3A5C"   # H1 headings, header bar
C_H2      = "#2A5A8C"   # H2 headings
C_H3      = "#3A6A9C"   # H3 headings
C_GREEN   = "#157524"   # Success/positive
C_RED     = "#721C24"   # Failure/negative
C_TABLE_HEADER = "#1a3a5c"
C_TABLE_STRIPE = "#f0f4f8"
```
