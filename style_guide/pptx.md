# PPTX Conventions

For python-pptx generators. See also [guardrails.md](guardrails.md) — all O/D/T/S/F/A/B rules apply.

## Color Palette (Dark Theme Template)

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

These positions are calibrated for the project's slide template — don't adjust without re-validating the guardrails (footer safe zone, content-top reservation):

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
- **Speaker notes**: plain text

For full minimum font tables and contrast rules see [guardrails.md §T1–T2](guardrails.md#53-text-readability-t1-t7).

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

## Animation Rules (canonical home — guardrails §5.6 just points here)

`add_animations()` constructs click-to-advance OOXML entrance animations. PowerPoint validates the XML loosely; broken animations render silently wrong rather than erroring. Hence the strictness below.

### A1 — `para_build` must default to `False`

When `True`, multi-paragraph shapes in entry groups get hidden instead of visible on entry. Only set `True` explicitly when per-paragraph bullet builds are desired.

### A2 — Animation groups must include ALL child shapes

Card layouts (rect + text overlays) appear unified on screen but are independent shapes in the file. If only the rect is in the anim group, the text renders immediately while the rect waits for click:

```python
card_shapes = []
card_shapes.append(add_rect(...))      # background
card_shapes.append(add_text(...))      # title
card_shapes.append(add_text(...))      # body
anim_groups.append(card_shapes)
```

### A3 — Group order = click order

Group 0 is visible on slide entry, group 1 appears on first click, etc. Left-side / earlier-in-reading-order content should generally be group 0 so the audience reads naturally.

### A4 — One animation group per logical unit

Splitting a single concept across two clicks fragments attention. Combining two concepts into one click overloads it.

### A5 — OOXML 3-level `par` nesting is non-negotiable

Even small deviations silently break in PowerPoint. Do not attempt to simplify:

```xml
Level 1 par: delay="indefinite"  (waits for click)
  Level 2 par: delay="0"
    Level 3 par: presetID="1", presetClass="entr", nodeType="clickEffect"
```

## Modular Generator Architecture

For decks >10 slides, split:

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

Each slide is a `slide_*()` function. Auto-numbering via a counter. The orchestrator defines the ordered builder list. Why: monolithic `generate_presentation.py` files become unreviewable past ~20 slides, and slide-specific bugs become hard to isolate.
