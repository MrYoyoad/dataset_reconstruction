# Document & Presentation Style Guide

Universal formatting rules for Word documents, PowerPoint presentations, LaTeX documents, and analytical plots.

---

## Table of Contents

1. [Word/Docx Conventions](#worddocx-conventions)
2. [PPTX Conventions](#pptx-conventions)
3. [Plot Styling](#plot-styling)
4. [LaTeX / Thesis Conventions](#latex--thesis-conventions)
5. [Visual Quality Guardrails](#visual-quality-guardrails)
6. [Cross-Format Rules](#cross-format-rules)

---

# Word/Docx Conventions

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
2. Logo centered, 2.0-2.5 inches — use `doc.add_picture()` (not `run.add_picture()`)
3. Title, 48pt bold, primary color
4. Subtitle, 22pt
5. Author name, 14pt, centered
6. Metadata key-value pairs (centered)
7. Page break

## Table of Contents

Use a **standard Word TOC field** (not manual hyperlinks):

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

Push the TOC heading down slightly from the top of the page (3 zero-height spacer paragraphs).

## Page Breaks After Tables

When a table is the last element before a page break, use a tight break to avoid a visible blank gap:

```python
def _tight_page_break(doc):
    last_p = doc.paragraphs[-1]
    last_p.add_run().add_break(WD_BREAK.PAGE)
```

## General Docx Rules

- Font: Calibri throughout
- A4 page size (21.0 x 29.7 cm), 2.5 cm side margins, 2.0 cm top/bottom
- Page numbers in footer (centered, 8pt gray)
- Body text: 11pt minimum
- Table cells: 10pt minimum
- Heading hierarchy with color cascade (primary → lighter shades for H2, H3, H4)
- Table styling: dark header row with white text, zebra striping (light alternate rows), 8-10pt cell padding
- Code blocks: light grey background (`#f5f5f5`)

### Color Template (Docx)

Replace these with your project's brand colors:

```python
C_PRIMARY = "#1A3A5C"   # H1 headings, header bar
C_H2      = "#2A5A8C"   # H2 headings
C_H3      = "#3A6A9C"   # H3 headings
C_GREEN   = "#157524"   # Success/positive
C_RED     = "#721C24"   # Failure/negative
C_TABLE_HEADER = "#1a3a5c"  # Table header background
C_TABLE_STRIPE = "#f0f4f8"  # Table zebra stripe
```

---

# PPTX Conventions

**Dependency**: `python-pptx`

## Color Palette Template

Define your brand palette. Here's a dark-theme template:

| Name | Hex | Usage |
|------|-----|-------|
| BG | `#0D1B2A` | Deep background (all slides) |
| ACCENT1 | `#00B4D8` | Primary accent, section headers |
| ACCENT2 | `#E06C75` | Emphasis, highlights |
| GREEN | `#4CAF50` | Success, positive metrics |
| GOLD | `#FFD54F` | Attention items, special callouts |
| WHITE | `#FFFFFF` | Primary text |
| LGRAY | `#AAAAAA` | Secondary text, slide numbers |
| MGRAY | `#666666` | Borders, dividers |
| BG_LIGHT | `#152A40` | Card backgrounds |
| BG_LIGHTER | `#1A3550` | Hover/active states |
| RED | `#F44336` | Negative metrics, failures |
| DRED | `#B71C1C` | Severe failures |

## Layout Constants

```python
SL_W = Inches(13.333)    # 16:9 widescreen width
SL_H = Inches(7.5)       # 16:9 height
MX   = Inches(0.6)       # Horizontal margin
MY   = Inches(0.4)       # Top margin
CT   = Inches(1.45)      # Content top (below title + accent line)
CW   = Inches(12.13)     # Content width (SL_W - 2*MX)
CH   = Inches(5.55)      # Content height
```

## Typography

- **Font**: Calibri throughout (matches docx)
- **Slide title**: 32pt bold white
- **Body text**: 16pt white
- **Bullet text**: 15pt white
- **Small text**: 9pt LGRAY (slide numbers, footnotes)
- **Speaker notes**: plain text (not shown on slides)

## Helper Functions Pattern

| Function | Purpose |
|----------|---------|
| `new_slide()` | Create blank slide with background color |
| `add_title(slide, text)` | Add title at top |
| `add_accent_line(slide)` | Horizontal rule below title |
| `add_text(slide, text, ...)` | Add text box with positioning |
| `add_bullets(slide, items, ...)` | Add bullet list |
| `add_rich_text(slide, runs, ...)` | Mixed formatting with per-run control |
| `add_image(slide, path, ...)` | Add image with positioning |
| `add_rect(slide, ...)` | Add colored rectangle (cards, backgrounds) |
| `add_slide_num(slide, n)` | Add slide number (bottom-right) |
| `set_notes(slide, text)` | Add speaker notes |
| `_finish()` | End-of-slide wrapper (handles animations, numbering) |

## Animation Rules

The `add_animations()` function creates click-to-advance OOXML entrance animations.

**Critical rules:**

1. **`para_build` must default to `False`** — Only set `True` explicitly when per-paragraph bullet builds are desired. When `True`, multi-paragraph shapes in entry groups get hidden instead of visible on entry.

2. **Card animations must include ALL child shapes** — When creating card layouts (rect + text overlays), collect every shape into the same `anim_groups` entry:
   ```python
   card_shapes = []
   card_shapes.append(add_rect(...))      # background
   card_shapes.append(add_text(...))      # title
   card_shapes.append(add_text(...))      # body
   anim_groups.append(card_shapes)
   ```
   Never put only the rectangle in `anim_groups` — text will float independently.

3. **Animation group order = click order** — Group 0 is visible on entry, group 1 on first click, etc. Left-side content should generally be group 0.

4. **OOXML 3-level par nesting is non-negotiable:**
   ```xml
   Level 1 par: delay="indefinite"  (waits for click)
     Level 2 par: delay="0"
       Level 3 par: presetID="1", presetClass="entr", nodeType="clickEffect"
   ```
   Do not attempt to simplify this XML. Even small deviations silently break in PowerPoint.

## Modular Generator Architecture

Recommended structure for large presentation generators:

```
generate_presentation.py         # Slim orchestrator (~90 lines)
presentation/
├── config.py                    # Paths, colors, layout constants, image dictionary
├── helpers.py                   # Slide setup, text, shapes, animations, _finish()
├── slides_introduction.py       # Section 1
├── slides_methodology.py        # Section 2
├── slides_results.py            # Section 3
└── slides_conclusions.py        # Section 4
```

Each slide is a `slide_*()` function. Auto-numbering via a counter. The orchestrator defines the ordered builder list.

---

# Plot Styling

## Plot Defaults

- **DPI**: 200 (standard), 250 for radar/detail charts
- **Figure size**: (10, 6) standard, (14, 6) for dual-panel, (16, 10) for dashboards
- **Style base**: seaborn + custom rcParams

## Color Palette

Plots use **matplotlib defaults** for analytical charts:

```python
blue    = "#1f77b4"   # Primary
orange  = "#ff7f0e"   # Secondary
green   = "#2ca02c"   # Success/positive
red     = "#d62728"   # Failure/negative
purple  = "#9467bd"   # Tertiary
```

**Exception**: Dark-theme plots embedded in PPTX use the brand palette for visual consistency.

## Visual Conventions

- Remove top and right spines: `ax.spines["top"].set_visible(False)`
- White bar edges: `edgecolor="white", linewidth=1.5`
- Insight annotation boxes: light yellow background, rounded corners
- Value labels: bold, 11pt, above/on bars
- Overfitting zones: red shading with `alpha=0.08`
- Font sizes: title 14-15pt, axes 12-13pt, labels 10-11pt

---

# LaTeX / Thesis Conventions

## Weizmann Institute Thesis Proposal

Weizmann Institute of Science provides an official thesis template. Key conventions:

- **Document class**: Use the Weizmann thesis class if available, otherwise `report` or `book` class
- **Page size**: A4 (standard for Israeli/European institutions)
- **Font**: Computer Modern (LaTeX default) or Times New Roman (if required by department)
- **Line spacing**: 1.5 or double spacing for the main body (check department requirements)
- **Margins**: Typically 2.5 cm on all sides (or as specified by department)
- **Language**: English (primary) with Hebrew abstract. Use `babel` or `polyglossia` for bilingual support
- **Bibliography**: BibTeX/BibLaTeX with a standard style (e.g., `apalike`, `plainnat`, or department-specified)

## Thesis Proposal Structure (Typical)

1. Title page (institution name, department, title, author, advisor, date)
2. Abstract (English + Hebrew)
3. Table of Contents
4. Introduction / Background
5. Literature Review
6. Proposed Research / Methodology
7. Preliminary Results (if any)
8. Expected Contributions
9. Timeline
10. Bibliography

## LaTeX Best Practices

- Use `\input{}` to split chapters into separate files
- Keep figures in a dedicated `figures/` directory
- Use `\label{}` and `\ref{}` consistently (never hardcode figure/table numbers)
- Use `booktabs` for professional tables (`\toprule`, `\midrule`, `\bottomrule`)
- Use `siunitx` for consistent number and unit formatting
- Use `cleveref` for smart cross-references (`\cref{fig:x}` → "Figure 1")
- Use `hyperref` for clickable cross-references in PDF output
- Compile with `latexmk -pdf` for reliable builds

## Beamer Presentations (Academic)

If making presentation slides in LaTeX:

- **Theme**: Metropolis (modern, clean) or your department's preferred theme
- **Font**: Fira Sans (Metropolis default)
- Use custom commands for repeated patterns (metric boxes, comparison layouts, etc.)
- Section divider slides between major sections help audience track progress
- Appendix slides are a "pressure valve" — move detailed content there for Q&A backup

---

# Visual Quality Guardrails

Hard-won rules from 250+ presentation remarks. Every developer touching generator code or creating new slides must follow these.

**Docx applicability**: Rules T1 (body 11pt min, table 10pt min), D3 (avoid consecutive tables without text), and F1-F4 (data freshness) also apply to docx generators.

## 5.1 Occlusion Prevention (O1-O8)

Elements hiding each other is the most common visual defect.

**O1 — Column widths must not exceed CW.**
Multi-column layouts must satisfy: `n * col_w + (n-1) * gap <= CW` (12.13"). A 3-column layout once totaled 12.4" and clipped the right column off-slide.

**O2 — Tables must not overlap downstream elements.**
After placing a table, verify: `table_y + (num_rows + 1) * row_height < next_element_y`. Fix: reduce `row_height` (0.7" -> 0.48") and font size (12pt -> 11pt).

**O3 — Images must not cover subtitle text.**
When an image follows a subtitle, the image top must be >= `subtitle_y + subtitle_h + Inches(0.15)`. Grey subtitle text on dark backgrounds is invisible if even partially covered.

**O4 — Footer safe zone.**
- Main content must end by **y=6.1"**
- Footnotes/references by **y=6.45"**
- Nothing below **y=7.12"** (page numbers live there)

**O5 — Cleared slides must re-apply background.**
When clearing all shapes from a slide (for rebuild), explicitly reset `slide.background.fill`. Otherwise text renders invisible.

**O6 — Callout boxes must be wide enough for their text.**
Minimum callout width: `Inches(2.2)`.

**O7 — Connectors and arrows must be mathematically centered.**
Never eyeball placement. Use: `arrow_x = target_x + target_w / 2 - arrow_w / 2`.

**O8 — Rebuilt slides must clear shapes first or use absolute positioning.**
When rebuilding slide content programmatically, clear existing shapes first. Never rely on "below previous" relative positioning.

## 5.2 Density Limits (D1-D6)

Overcrowded slides were the second most common remark category.

**D1 — Max 5 bullet points per slide (prefer 3-4).**
Each bullet should be one line at the target font size. If you need more, split into two slides.

**D2 — Max 12 words per bullet point.**
Bullets are signposts, not sentences. Move explanatory detail to speaker notes.

**D3 — Max 1 primary table per slide.**
If a slide needs two tables, the second goes to speaker notes or a separate slide. Exception: small key-value stat tables (<=3 rows) alongside a main table.

**D4 — Max 2 statistical measures visible per slide.**
Pick the most important metric. Move the rest to speaker notes.

**D5 — One concept per slide.**
If a slide needs "Part 1" and "Part 2" labels, it should be two slides.

**D6 — Redundant tables go to speaker notes.**
If information is already conveyed by a chart or card layout, do not also show it in a table on the same slide.

## 5.3 Text Readability (T1-T7)

Projector distance reduces effective readability by ~50%.

**T1 — Minimum font sizes by element type:**

| Element | Minimum | Recommended |
|---------|---------|-------------|
| Slide title | 28pt | 32pt |
| Body text / bullets | 14pt | 15-17pt |
| Table cell text | 12pt | 13pt |
| Card description text | 12pt | 13pt |
| Callout box text | 11pt | 12pt |
| Footnotes / references | 8pt | 9pt |
| Annotations on slides | 11pt | 12pt |
| Plot axis labels (matplotlib) | 12pt | 13pt |
| Plot value labels (matplotlib) | 10pt | 11pt |

**T2 — Never use light gray text smaller than 11pt.**
Grey on dark backgrounds has inherently low contrast. Below 11pt it becomes unreadable on projectors. Prefer white for any text below 13pt.

**T3 — Every text frame must set `word_wrap=True` and `auto_size=None`.**
```python
tf = tb.text_frame
tf.word_wrap = True
tf.auto_size = None  # Prevents pptx from auto-shrinking text
```

**T4 — Font sizes must be consistent across similar elements within a slide.**
All card titles on one slide must use the same size. All table cells must use the same size.

**T5 — Plot text must be readable at presentation scale.**
Matplotlib defaults (10pt) are too small. Set `rcParams` for titles >= 14pt, axes >= 12pt, tick labels >= 10pt. Test embedded plots at 50% zoom (simulates projector distance).

**T6 — Split combined dual-axis plots into separate panels.**
A single matplotlib figure with dual axes, small text, and multiple series is always unreadable in presentations. Split into two separate images at larger size.

**T7 — Border and header weight must not overpower content.**
Header bars max 0.32" height. Borders max Pt(1.5). Use fills rather than heavy outlines.

## 5.4 Spacing & Layout (S1-S7)

Breathing room is what separates a professional slide from a wall of text.

**S1 — Minimum 0.15" vertical gap between stacked elements.**
Between cards, between table and text, between image and caption.

**S2 — Minimum 0.3" two-column gutter.**

**S3 — Images and text must share width proportionally.**
Standard split: ~5.6" text + ~0.3" gap + ~5.9" image. Custom ratios are allowed, but text area >= 3.4" and image >= 4.0".

**S4 — Content top starts at CT=1.45".**
This reserves space for title and accent line. Never place content above CT.

**S5 — Content bottom zones.**
- Main content: end by **y=6.1"**
- Footnotes and references: **y=6.45"** max

**S6 — Use named gap variables, never magic numbers.**
```python
row_gap = Inches(0.12)
col_gap = Inches(0.43)
card_h  = Inches(1.05)
```

**S7 — Layout formulas must be self-documenting.**
```python
step = box_w + h_gap + arrow_w + h_gap
total_w = n * box_w + (n - 1) * (h_gap + arrow_w + h_gap)
start_x = MX + (CW - total_w) / 2  # center the grid
```
Never hardcode pixel positions.

## 5.5 Data Freshness (F1-F4)

Stale numbers undermine credibility and caused the most multi-file fix commits.

**F1 — Regenerate embedded images after any metric/threshold change.**
Plots with threshold lines, percentages, or boundaries go stale when source data changes. After modifying canonical numbers, always re-run the relevant plot generators AND regenerate the presentation.

**F2 — All slide text numbers must trace to a single source of truth.**
Canonical numbers should live in one place (e.g., CLAUDE.md, a config file, or a data file). When numbers change, grep all generator files for the old value.

**F3 — Speaker notes must be updated alongside visible slide content.**
When slide text changes, speaker notes for that slide must also be reviewed.

**F4 — After updating a metric threshold, run a full-file grep audit.**
Search for the old threshold value across ALL generator files, markdown docs, and LaTeX source. A single surviving stale reference undermines the entire update.

## 5.6 Animation Structure (A1-A5)

OOXML animation bugs are silent — PowerPoint doesn't warn, it just renders wrong.

**A1 — `para_build` must default to `False`.**

**A2 — Animation groups must include ALL child shapes.**
```python
card_shapes = []
card_shapes.append(add_rect(...))   # background
card_shapes.append(add_text(...))   # title
card_shapes.append(add_text(...))   # body
anim_groups.append(card_shapes)
```

**A3 — Animation group order = visual narrative order.**
Group 0 is visible on slide entry. Group 1 appears on first click. Left-side content = group 0.

**A4 — One animation group per logical unit.**

**A5 — OOXML 3-level par nesting is non-negotiable.**
```xml
Level 1 par: delay="indefinite"  (waits for click)
  Level 2 par: delay="0"
    Level 3 par: presetID="1", presetClass="entr", nodeType="clickEffect"
```

---

# Cross-Format Rules

| Rule | Detail |
|------|--------|
| **Brand palette** | Use consistently for PPTX and Beamer (define once, reference everywhere) |
| **Matplotlib defaults** | OK for standalone analytical plots (not embedded in slides) |
| **Dark-theme plots** | Required when plot is embedded in a dark-background presentation |
| **DPI** | 200 standard, 250 for radar/detail charts |
| **Font** | Calibri for PPTX/docx, Fira Sans for Beamer, Computer Modern for LaTeX thesis |
| **Numbers** | All formats must use same canonical numbers from a single source of truth |
| **Guardrails** | PPTX: all rules (O/D/T/S/F/A). Docx: T1, T3, D3, F1-F4. LaTeX: F1-F4, D1-D5 |
