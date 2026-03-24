---
description: Generate publication-quality figures, plots, and visualization grids for the thesis
disable-model-invocation: true
---

## Existing figures
!`ls figures/`

## Existing results
!`ls results/*.csv | head -20`

## Task
$ARGUMENTS

## Process
1. **Understand what to visualize** — what story does this figure tell? What comparison?
2. **Load the data** — read from `results/` CSVs or `.pth` tensor files.
3. **Generate the plot** — use matplotlib with publication-quality settings.
4. **Save to `figures/`** — always save as both PDF (for LaTeX) and PNG (for slides/docs).
5. **Update docs** if the figure changes any narrative in STATUS.md or presentations.

## Style rules (read STYLE_GUIDE.md for full details)
- Font size ≥ 10pt for all text
- Use colorblind-friendly palettes
- Include axis labels with units
- Legend outside plot area if >3 series
- For reconstruction grids: show ground truth | reconstruction | control side-by-side
- Always include both best AND worst examples
- Save at 300 DPI minimum
