# Row audit — CIFAR headline table (`experiments/cifar/RESULT.md`)

Auditor: yoado-c6 (owns the two published pages, the .tex, the coverage measurement — audits nothing of its own here).
Assigned by R1/cross-check in `notes/math_rulings_2026-09-06.md`. Everything below is read from the job rows
(`experiments/cifar/charts/*/result.json`) or recomputed from the saved tensors
(`experiments/cifar/k32_onchart/release_and_search.pt`), never from adjacent prose.

## F1 — CONFIRMED: the pixel-layer headline contradicts its own row. Fix the headline.

`charts/L1_ae_lm_onchart_k32/result.json`: `starts 400, landed 30, images_found 2,
landings_per_image [0,29,0,1,0,0,0,0], top20_by_residual_landed 20/20, collapse 0.861`.

The §2 table (line 44) and `figures/cifar_charts/table.md` line 3 are right at the row. The headline
(line 15) "0 images; every minimiser is a blend" is wrong on both halves: 2 images are recovered, and
13.9% of the collapse figure is not collapse. R1's flag is upheld.

## F2 — NEW: §1 and §2 are two different runs, stitched as one cell.

§1's "379 of 400" is **not** the L1 chart-study row. It is the original replica,
`experiments/cifar/k32_onchart/release_and_search.pt`, which I recomputed:

| | original replica (§1) | chart study L1 (§2 table) |
|---|---|---|
| starts | 400 | 400 |
| landed | **14** | 30 |
| images found | **1** (image 7 only) | 2 |
| collapsed within 1e-2 of the medoid | **379 (94.8%)** | `collapse` 0.861 |

Both numbers are individually correct; the paragraph is wrong to join them. §1 should name its run.

## F3 — NEW, and it inverts the paragraph: the reached attractor is NOT at the truths' residual.

§1: *"the blend's own certificate residual is 3e-15 — as low as the truths'."* Recomputed on the
replica's own tensors, with the run's own objective `‖Cx‖/‖A_T x‖`:

```
IDEAL blend (exact linear combination of the privates) : 2.86e-14   <- equals the truths; this is the lemma
ATTRACTOR the solver actually reached (mean of the 379): 5.90e-03   <- 2.1e+11 times the truths'
truths                                                 : 2.86e-14
```

The attractor is only **98.9%** blend (`1-(‖r‖/‖x‖)² = 0.9886`; the reported 99.4% is not reproduced —
`1-‖r‖/‖x‖` gives 0.893, so no normalisation I tried yields 99.4). Its coefficients reproduce exactly:
`(-0.70, 0.337, -0.06, 0.135, -0.067, 0.458, 0.176, 0.177)`, sum 0.455, three negative — non-convex,
as stated.

The consequence is the opposite of what §1 argues. In this cell the residual **does** separate:
the 14 landings sit at `R = 7.9e-07 … 1.1e-06`, the 379 collapsed at median `5.83e-04` — ~600×. That
is why `top20_by_residual_landed` is 20/20 in the L1 row rather than 0/20, and why images found is
1–2 rather than 0. The solver stalls *near* the blend subspace without reaching it.

**What survives, and it is the important half.** R1's lemma is a proof and needs no measurement: the
ideal blends are exact zeros (confirmed above at 2.86e-14, identical to the truths), so the truths are
not isolated at any k. That is untouched. What does not survive is the empirical sentence used to carry
it — the checks are not defeated here by a blend sitting at the floor; the run never got to the floor.
**Correct form:** the pixel layer costs *coverage* (1–2 of 8), not *precision* (top-20 clean). The
non-isolation is a theorem about the zero set, not an observed indistinguishability.

## F4 — a one-sided test, offered as two-sided (R1's isolation test)

`J = C·DΨ(z*)`, `rank J = k` ⟹ `z*` isolated: sound. The converse as written — "`rank J < k` ⟹ the zero
set is locally positive-dimensional" — does not follow: a degenerate isolated zero (vanishing first
order, isolated at higher order) has `rank J < k` too. The test certifies isolation and never
certifies its absence. Worth stating one-sided, since the attacker-side value is entirely in the
positive direction.
