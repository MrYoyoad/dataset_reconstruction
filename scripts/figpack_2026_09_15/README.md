# `scripts/figpack_2026_09_15/` — editable sources for the 2026-09-15 figure pack

Imported 2026-09-17 from the `figure_pack_editable` and `LoRA_visual_figures` bundles inside
`Thesis_files_2026-09-17.zip`. Rendered output lives in `figures/gal_2026-09/` (PNG + `vector/*.pdf`), and the
panel-by-panel provenance, including three corrections found when the panels were traced back to rows, is in
`notes/figure_provenance_2026-09-15.md`.

## Files

- `01_architecture.tex` … `07_proposed_local_families.tex` — one TikZ page each, the pack as shown.
- `05_image_family.tex`, `07_mnist_coverage.tex` — the two pages the later pack dropped, kept because the MNIST
  coverage page is the one that shows chart dependence.
- `style.tex` — shared colours and macros. Green = searched family coordinates, blue = frozen model,
  orange = adapter/certificate residual.
- `build.py` — assembles the pack. **Needs pdflatex.**
- `fill_results_template.py` — fills the results page from rendered assets. **Shells out to `kpsewhich`.**
- `validate.py` — two halves. **The algebra half runs and passes here** (checked 2026-09-17, CPU, seconds):
  seed 713, `n=9, r=6, m=4, N=2`, 100 SGD steps, asserting `‖BP‖ < 1e-10` and `‖P(A − A0)‖ < 1e-10` at *every*
  step, the `T=1` identities `A_1 = A_0` and `B_1 = −η D (A_0H)^T`, then `rank B = q`, `rank(PA) = r − q`,
  `‖PAH‖ < 1e-10`, and the free-coefficient gauge identity `ΛV^T = (ΛM^{-T})(VM)^T`. **The layout half fails here
  by design**: from line 59 it expects exactly eight `NN_*.tex` pages with their compiled `.pdf`/`.log`/`.png`
  beside them and a built `LoRA_figures_only.pdf`, none of which exists in this directory (we keep nine pages,
  merged from two packs, and the renders live in `figures/gal_2026-09/`). An `AssertionError` at line 59 therefore
  means the algebra passed. The same invariants are checked against *real* releases by
  `experiments/exact_inversion/certificate.py` and `experiments/multilayer_cert/theory_checks.py`.
- `assets/` — the rendered panels the pages embed (letters, motorcycles, Fashion, MNIST, the architecture figure).
- `compositors/` — the six scripts that cut each published panel out of its source raster. They are the only
  record of the crop boxes, strides and polarity used. `adopt_figure.py` and `build_audit_figures.py` carry
  hard-coded `/workspace/scratch/...` paths and need repathing before they run.

## Building

**Neither pdflatex nor kpsewhich exists on WEXAC** (see CLAUDE.md, "Markdown → PDF on WEXAC"), so the pack does not
rebuild there. Build on Overleaf or a local TeX install. The prebuilt PDFs and PNGs are committed in
`figures/gal_2026-09/`, so nothing needs rebuilding in order to be read or shown.

## Before reusing a panel

Read `notes/archive_evidence_map_2026-09-17.md` first. Two numbers on these pages are stale (the Fashion-MNIST
"4 of 8" is a 150-start cell superseded by 6 of 8 at 400 starts), one panel needs a label it does not have (the
MNIST "private-built reference" chart is oracle-built and not attacker-available), and one quoted correlation is
untraced.
