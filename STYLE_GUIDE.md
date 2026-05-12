# Document & Presentation Style Guide

Project-specific formatting rules. Generic toolchain knowledge (matplotlib idioms, LaTeX commands) is **not** restated here.

## How to use this guide

Read only what the current task needs:

| You are generating… | Read |
|---------------------|------|
| Word docx | [style_guide/docx.md](style_guide/docx.md) |
| PPTX slides | [style_guide/pptx.md](style_guide/pptx.md) + [style_guide/guardrails.md](style_guide/guardrails.md) |
| Matplotlib plots | [style_guide/plots.md](style_guide/plots.md) |
| LaTeX thesis or Beamer | [style_guide/latex.md](style_guide/latex.md) + guardrails.md |
| PDF reports (fpdf2 etc.) | guardrails.md (O9, O10 are critical) |

When in doubt, also read guardrails.md — it applies to every output format.

## Cross-Format Rules

| Rule | Detail |
|------|--------|
| **Font** | Calibri for PPTX/docx, Fira Sans for Beamer, Computer Modern for LaTeX thesis |
| **Palettes** | Each format defines its own (dark PPTX, light docx, matplotlib defaults). They are not unified — don't force consistency across formats. |
| **DPI** | 200 standard, 250 for radar/detail plots |
| **Numbers** | All formats must use the same canonical numbers from a single source of truth (e.g., CLAUDE.md or a config file). When numbers change, grep all generators for the old value before commit. |
| **Guardrails** | PPTX: all O/D/T/S/F/A/B rules. Docx: T1, T3, D3, F1-F4. LaTeX/Beamer: D1-D5, F1-F4. PDF: O9, O10. |

## What changed in this guide vs. earlier versions

- **Split by format (May 2026)**: progressive disclosure — only the relevant file loads for any given task.
- **Section 5.7 Build-and-Review Discipline (B1-B12)** added to guardrails, distilled from a 12-section side-by-side deck review.
- **Animation Rules** consolidated in `style_guide/pptx.md`; guardrails §5.6 is now just a pointer.
- **Generic LaTeX and matplotlib content removed** — those are not project decisions.
- **O9 generalized** from fpdf2-specific to all fixed-width cell APIs; **O10** added for draw-state reset.
