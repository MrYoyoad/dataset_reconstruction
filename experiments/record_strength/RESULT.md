# Record strength vs recovery, inside a fixed chart — 2026-09-06

Script `experiments/record_strength/record_strength.py` (no canonical script modified). Jobs: **255095** (letters,
hgn41), **255098** (confident digits, hgn36), **255107** (plots), **279342** (σ decomposition + replot).
Everything FP64 except the releases deliberately trained in fp32 / bf16 / fp16. SSIM is `kornia.metrics.ssim`,
window 3, on [0,1].

**Reproduction gate — every cell reproduces its original job exactly.** Landing counts per image are identical to
job 760909 (letters fp64 and fp32 at the noise-matched tolerance), 764976 (letters fp32/bf16 at tol 1e-12), and
706721 (confident digits at k = 6, 8, 10); the objective at each truth in the confident cell matches 706721 to every
printed digit. The re-run is the same experiment with per-start outputs saved.

Definitions, per private example i, all in the release's own arithmetic:
`a_i = A₀h_i`; `ã_i` = `a_i` projected off `span{a_j : j≠i}` and normalised; **record strength `σ_i = ‖B_T ã_i‖`**;
`u_i = Σ_t η‖p_{t,i} − y_i‖₂` (accumulated softmax error, the applied per-image step is `u_i/N`); imprint `‖C_i‖`
from the traced release; certificate residual at the truth in two forms, `‖Ch‖/(‖C‖‖h‖)` (as specified) and the
search objective `‖Ch‖/‖A_T h‖`; recovery from the certificate search re-run start for start.

## What the theorem predicts, and what the data shows

**Prediction (exact arithmetic): a threshold.** If example i's direction is in `row(B_T)`, the certificate
annihilates it exactly whatever the gradient size, and its chart coordinates are identified; if not, there is no
constraint at all.

**Measured: in FP64 the threshold holds over eleven orders of magnitude of σ_i.** Across 32 example-cells from
FP64-trained releases, every recorded example is recovered to an image error of 1e-15 … 2e-5 against a landing
threshold of 1e-2, and every unrecorded one is at 0.48 … 0.65 — no intermediate case. The recovery error does not
track σ_i: the four weakest recorded examples (σ_i = 4e-10 … 5e-5) come back at errors 4e-8 … 1e-12, better than or
equal to examples five orders stronger. What varies with σ_i is not the error but the **basin**, and even that only
loosely (Spearman ≈ 0.6 within a cell; the k = 6 example at σ_i = 0.43 draws 414 of 2000 starts, the one at
σ_i = 1.2e-5 draws 11).

**Graded behaviour appears only when the release is not exact.** Same eight letters, same chart, releases trained in
different formats:

| training format | certificate residual at the truths | landings (500 starts) | closest approach |
|---|---|---|---|
| fp64 | 2e-13 … 7e-12 | 192, all 8 found | 1e-12 … 6e-11 |
| fp32 (tol 1e-12) | 3e-7 … 1e-5 | 164, all 8 found | 1e-6 … 8e-5 |
| fp32 (noise-matched tol) | 1e-4 … 3e-3 | 81, 6 of 8 | 6e-4 … 2e-2 |
| bf16 | 3e-2 … 3e-1 | 0 | 0.12 … 0.78 |
| fp16 | 1e-2 … 2e-1 | 0 | 0.04 … 0.77 |

The step is a step in the **arithmetic**, not in σ_i: σ_i changes by at most 15% between formats (0.069 … 0.24 in all
four), while the residual at the truth rises by eleven orders and the landings go 192 → 0. Within the fp32 cell the
per-example residual does order with σ_i (the two lowest-σ letters are the two lost at the coarse tolerance), which
is the closest thing to a graded effect in the data.

**Backup figure A's invisible example is the threshold's other side.** The confident digit the base model already
classified (imprint 2e-25, σ_i = 6e-16, u_i = 3e-29) has certificate residual 0.04 and is never approached: it is not
a weak recovery, it is no constraint. Three more examples cross the same line at k = 10 as the release records fewer
directions (N' = 5).

**σ_i is predicted by the accumulated update.** `σ_i` against `u_i` is monotone across all 32 points and spans 30
orders of magnitude together, so the softmax residual an example accumulates is what sets how strongly it is
recorded. Two points per decade over ten decades; no curve fitted, and none should be.

## Caveats measured, not assumed (job 279342)

`σ_i` is not a pure own-example quantity: writing `B_T = Σ_j C_j`, the cross term `Σ_{j≠i} C_j ã_i` is 0.3–37% of
σ_i in the confident cells and 28–42% in the letters cell, because A drifts during training and `ã_i` is only
orthogonal to the *initial* readings `A₀h_j`. The own term alone gives the same ordering. `‖a_i^⊥‖/‖a_i‖` is
0.51–0.82, so σ_i also carries a mild A₀-geometry factor. For the one invisible example the own term is ~1e-30 and
the cross term dominates, which is why its σ_i sits at 4e-16 rather than at zero.

## Table and figures

Full per-example table: `figures/record_strength/table.md` (release, i, label, recorded, σ_i, imprint, u_i, both
certificate residuals, landings, basin, closest approach vs chart and vs raw, SSIM, chart floor). Decomposition:
`results/record_strength/sigma_decomposition.json`.

- `figures/record_strength/recovery_error_vs_sigma.png` — the threshold: two horizontal bands, nothing between.
- `figures/record_strength/basin_vs_sigma.png` — basin size against σ_i.
- `figures/record_strength/sigma_vs_u.png` — σ_i against the accumulated update.
- `figures/record_strength/letters_precision_overlay.png` — the same eight letters at four training precisions.

## Three sentences

The theorem predicts a threshold, and in FP64 that is exactly what the data shows: eleven orders of magnitude of
record strength give the same recovery error, and the only example that fails is the one the model never had to
learn. A graded effect appears only when the release itself is inexact — fp32 recovers everything at a degraded
residual, bf16 and fp16 recover nothing — so gradedness is a property of the arithmetic, not of the gradient size.
With 32 example-cells and a bimodal outcome this supports the threshold reading and rules out a smooth law over the
measured range; it is not enough to characterise the transition region, which the fp32 cell only brackets.
