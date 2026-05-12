# Visual Quality Guardrails

Hard-won rules from 250+ presentation remarks plus a May 2026 side-by-side deck review. Every developer touching generator code or creating new slides must follow these.

**Applicability**: All rules apply to PPTX. For docx, T1 (font minimums), D3 (consecutive tables), and F1-F4 (data freshness) apply. For LaTeX/Beamer, D1-D5 (density) and F1-F4 apply. PDF generators (e.g. fpdf2): O9 and O10 are the most important.

## 5.1 Occlusion Prevention (O1-O10)

Elements hiding each other is the most common visual defect.

**O1 — Column widths must not exceed CW.**
Multi-column layouts must satisfy `n * col_w + (n-1) * gap <= CW` (12.13"). A 3-column layout once totaled 12.4" and clipped the right column off-slide.

**O2 — Tables must not overlap downstream elements.**
After placing a table, verify `table_y + (num_rows + 1) * row_height < next_element_y`. Why: python-pptx doesn't enforce z-order, so an overflowing table renders on top of (or under) the next shape with no warning. Fix: reduce `row_height` (0.7" → 0.48") and font size (12pt → 11pt).

**O3 — Images must not cover subtitle text.**
When an image follows a subtitle, image top must be `≥ subtitle_y + subtitle_h + Inches(0.15)`. Grey subtitle text on dark backgrounds disappears under even partial cover.

**O4 — Footer safe zone.**
- Main content must end by **y=6.1"**
- Footnotes/references by **y=6.45"**
- Nothing below **y=7.12"** (page numbers live there)

**O5 — Cleared slides must re-apply background.**
When clearing all shapes from a slide for rebuild, explicitly reset `slide.background.fill`. Why: clearing shapes can drop the slide-level fill reference, and white text on default-white background is invisible.

**O6 — Callout boxes must be wide enough for their text.**
Minimum callout width: `Inches(2.2)`. Narrower callouts force ugly wraps or silent clipping.

**O7 — Connectors and arrows must be mathematically centered.**
Never eyeball placement: `arrow_x = target_x + target_w / 2 - arrow_w / 2`. Manual placement looks fine on the editing slide and 4px off on the projector.

**O8 — Rebuilt slides must clear shapes first or use absolute positioning.**
When rebuilding programmatically, clear existing shapes first. Relative "below previous" positioning breaks when prior shapes have been edited.

**O9 — Fixed-width cell methods silently truncate.**
`fpdf2.cell()` and similar fixed-width cell APIs clip text exceeding column width — no warning, no wrap, just invisible truncation. Before generating a table: (1) shorten header labels ("Confidence" → "Conf."), (2) abbreviate long cell values, (3) test the widest value in each column fits at the chosen font size, (4) for 5+ column tables, reduce font to 9pt with compact headers. Always verify the rendered output — don't trust proportional `col_widths`.

**O10 — Reset draw state after colored elements.**
Callout boxes, warning boxes, and highlighted rects set `draw_color` and `line_width` to custom values. Subsequent tables and shapes inherit those colors (e.g., green table borders from a green callout). After any element that changes draw state, explicitly call `set_draw_color(0,0,0)` and `set_line_width(0.2)`.

## 5.2 Density Limits (D1-D6)

Overcrowded slides were the second most common remark category. The audience reads slides in 5-10 seconds; anything beyond that load fails.

**D1 — Max 5 bullet points per slide (prefer 3-4).**
Each bullet one line at the target font size. More bullets → split slide.

**D2 — Max 12 words per bullet point.**
Bullets are signposts, not sentences. Detail goes to speaker notes.

**D3 — Max 1 primary table per slide.**
Second table → speaker notes or separate slide. Exception: small key-value stat tables (≤3 rows) alongside a main table.

**D4 — Max 2 statistical measures visible per slide.**
Pick the most important metric. The rest go to speaker notes. Why: every metric on screen competes for attention; >2 and the audience can't track which is the headline.

**D5 — One concept per slide.**
If a slide needs "Part 1" and "Part 2" labels, it should be two slides.

**D6 — Redundant tables go to speaker notes.**
If a chart or card layout already conveys the information, the table is duplication, not redundancy-for-clarity.

## 5.3 Text Readability (T1-T7)

Projector distance reduces effective readability by ~50% versus laptop view.

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
Grey on dark backgrounds has inherently low contrast; below 11pt it becomes unreadable on projectors. Prefer white for any text below 13pt.

**T3 — Every text frame must set `word_wrap=True` and `auto_size=None`.**
```python
tf = tb.text_frame
tf.word_wrap = True
tf.auto_size = None  # Prevents pptx from auto-shrinking text
```
Why: auto-shrink silently turns 16pt body into 9pt unreadable mush when content overflows.

**T4 — Font sizes must be consistent across similar elements within a slide.**
All card titles same size. All table cells same size. Inconsistency reads as a bug, not as emphasis.

**T5 — Plot text must be readable at presentation scale.**
Matplotlib defaults (10pt) are too small. Set `rcParams` for titles ≥14pt, axes ≥12pt, tick labels ≥10pt. Test embedded plots at 50% zoom (simulates projector distance).

**T6 — Split combined dual-axis plots into separate panels.**
A single figure with dual axes, small text, and multiple series is always unreadable in presentations.

**T7 — Border and header weight must not overpower content.**
Header bars max 0.32" height. Borders max `Pt(1.5)`. Use fills rather than heavy outlines.

## 5.4 Spacing & Layout (S1-S7)

Breathing room is what separates a professional slide from a wall of text.

**S1 — Minimum 0.15" vertical gap between stacked elements.**
Cards, table-to-text, image-to-caption.

**S2 — Minimum 0.3" two-column gutter.**

**S3 — Images and text must share width proportionally.**
Standard split: ~5.6" text + ~0.3" gap + ~5.9" image. Custom ratios allowed, but text ≥ 3.4" and image ≥ 4.0".

**S4 — Content top starts at CT=1.45".**
Reserves space for title and accent line. Never place content above CT.

**S5 — Content bottom zones.**
- Main content: end by **y=6.1"**
- Footnotes and references: **y=6.45"** max

**S6 — Use named gap variables, never magic numbers.**
```python
row_gap = Inches(0.12)
col_gap = Inches(0.43)
card_h  = Inches(1.05)
```
Why: magic numbers in three places become inconsistent in three different ways the moment someone tweaks one.

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
Plots with threshold lines, percentages, or boundaries go stale when source data changes. Always re-run plot generators AND regenerate the presentation.

**F2 — All slide text numbers must trace to a single source of truth.**
Canonical numbers live in one place (CLAUDE.md, a config file, a data file). When numbers change, grep all generators for the old value.

**F3 — Speaker notes must be updated alongside visible slide content.**
When slide text changes, review notes too — they cite numbers the slide may no longer show.

**F4 — After updating a metric threshold, run a full-file grep audit.**
Search the old value across ALL generators, markdown docs, LaTeX source. A single surviving stale reference undermines the entire update.

## 5.6 Animation Structure (A1-A5)

OOXML animation bugs are silent — PowerPoint doesn't warn, it just renders wrong. **Full rules and XML structure live in [pptx.md §Animation Rules](pptx.md#animation-rules-canonical-home--guardrails-56-just-points-here).** Summary for awareness:

- **A1**: `para_build` defaults to `False`
- **A2**: animation groups include ALL child shapes of a logical unit
- **A3**: group order = click order = visual narrative order
- **A4**: one group per logical unit
- **A5**: OOXML 3-level `par` nesting is non-negotiable

## 5.7 Build-and-Review Discipline (B1-B12)

Process rules (orthogonal to layout). Each maps to a concrete failure mode observed in real side-by-side deck reviews; ignoring any of them lets a "successful save" ship a broken slide.

**B1 — Visually inspect every touched slide; programmatic checks lie.**
A successful save means the bytes are valid, not that the slide looks right. Bbox sweeps miss: content overflow into adjacent shapes, callout borders bisecting text, captions overwritten by text above, bullets stomping page numbers, callouts clipped below the slide edge. After every build batch, render the touched slides to images (or run an audit script) and look. Caveat: if your renderer (e.g. LibreOffice) disagrees with the user's renderer (e.g. PowerPoint), the user wins — see B10.

**B2 — Semantic colors are reserved keywords.**
Once a color carries meaning in a deck (e.g. green=TRUST/PASS, yellow=REVIEW, red=FAIL), reserve it. Never paint the literal word "Green" in cyan. Never decoratively highlight unrelated text in a tier color. Use a neutral accent (cyan typical) for non-semantic "look here" emphasis. Tier/status badges: solid fill + high-contrast text, not transparent fill with colored borders.

**B3 — Canonical terminology wins over draft sources.**
In an existing deck/project, the live deck is authoritative for naming. If a source/draft file uses an older term that the deck has already replaced, the deck wins. Inventory 5-10 representative slides first, build a short dictionary of canonical terms, and don't deviate.

**B4 — Show full evidence when the slide's thesis depends on full evidence.**
If a slide claims "X preserved, Y lost," the slide must let the reader verify both. Truncating to a fragment transfers proof to the speaker's voice — fine in a live talk, dangerous in screenshots/async review, worst for legal/security audiences. The data is almost always already in your source files; pull it.

**B5 — Color-code table *cell values* by tier, not the header row.**
Tables that show how a metric varies across rows × columns should encode that variance in every cell (green ≥80%, yellow 50–80%, red <50%). Headers stay muted. Co-locate auxiliary stats (n=…, std, p) inside the same cell — don't split into adjacent count columns that force left-right eye-tracking.

**B6 — Prefer native shapes/tables over rasterized chart screenshots.**
For data ≤10 rows, build a native table. For bar-shape encodings, use shape rectangles with widths proportional to value. matplotlib PNGs on slides are pixelated, don't scale, and can't be edited. Reserve raster for photos, real video stills, UI screenshots, and complex multi-series scatter plots.

**B7 — Inline footnote markers via run-split, not paragraph append.**
`para.add_run('*')` appends to the paragraph end — wrong placement for inline markers like `65%*`. Correct sequence: locate the target run, split it at the marker position, clone the run, set the clone to contain only the marker (with marker styling), insert the clone immediately after the original; optionally add a third run for the text after.

**B8 — Shared helper library; no per-slide XML gymnastics.**
If more than three places construct styled runs from raw XML/lxml, extract a `deck_helpers.py` with `styled_run`, `styled_paragraph`, color constants, animation timing blocks. Treat raw run construction in a slide builder as a code smell. Fix rendering bugs in the helper and re-render all slides.

**B9 — Reorder slides via slide-list manipulation, not add-then-delete.**
`add_slide()` always appends. To insert at an earlier position: add, capture the slide ID, remove from the slide-id list, insert at the target index. Companion gotcha in python-pptx: use `prs.part.related_part(rId)` (singular function call), not `prs.part.related_parts[rId]` (bracket form is wrong).

**B10 — The user's renderer is authoritative; yours is approximate.**
LibreOffice's PDF render and PowerPoint's native render disagree on font metrics, antialiasing, and sometimes theme color inheritance. When the user says a slide looks wrong, re-render at higher DPI (220+), crop tight, look hard. If it still looks fine to you, ask which element specifically — don't push back from your screenshot.

**B11 — Build for the audience's worst-case interpretation.**
A presentation is judged by what the audience can attack. For legal/security/compliance audiences, polarity inversion (negation flips), entity hallucination, and unhedged numeric claims are the worst failure modes — and must be demonstrated explicitly, not hidden. Add a "What we claim / What we do not claim" framing slide where stakes are high. Negative space (what's not promised) is as load-bearing as positive claims.

**B12 — Decks longer than ~30 slides need an explicit Thesis slide.**
At position 2 or 3, place a single-large-sentence Thesis slide (36–44pt, plenty of whitespace) stating the central claim. Follow with "What we claim / What we do not claim" if the audience is external. The thesis should be defensible at a whiteboard with no slides; every subsequent section should be checkable against "does this support or qualify the thesis?"

**Source:** Distilled from a 12-section side-by-side deck review (May 2026). The biggest single one is B1 — most of B2 through B6 would have been caught by visually inspecting every slide instead of trusting that the save succeeded.
