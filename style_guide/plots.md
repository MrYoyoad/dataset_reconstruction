# Plot Styling

For matplotlib plots in this project. Generic matplotlib idioms (spine removal, edgecolor, etc.) aren't documented here — only project-specific decisions.

## Defaults

- **DPI**: 200 (standard), 250 for radar/detail charts. Below 200 plots look soft when embedded in slides.
- **Figure size**: `(10, 6)` standard, `(14, 6)` for dual-panel, `(16, 10)` for dashboards.
- **Style base**: seaborn + custom rcParams (see [guardrails.md §T5](guardrails.md#53-text-readability-t1-t7) for required font sizes).

## Color Palette

Standalone analytical plots use matplotlib defaults:

```python
blue    = "#1f77b4"   # Primary
orange  = "#ff7f0e"   # Secondary
green   = "#2ca02c"   # Success/positive
red     = "#d62728"   # Failure/negative
purple  = "#9467bd"   # Tertiary
```

**Exception**: plots embedded in dark-theme PPTX must use the [pptx.md brand palette](pptx.md#color-palette-dark-theme-template) — matplotlib defaults clash with the dark background and look washed out at projection scale.

## Project Conventions

These are choices, not generic matplotlib practice:

- Overfitting / out-of-spec zones: red shading with `alpha=0.08`
- Insight annotation boxes: light yellow background, rounded corners
- Value labels on bars: bold, 11pt, above or on bars (per [guardrails T1](guardrails.md#53-text-readability-t1-t7))
