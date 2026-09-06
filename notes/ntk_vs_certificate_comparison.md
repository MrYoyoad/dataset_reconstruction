# The linearized route and the certificate: the same zero set, a different search (2026-09-06)

The thesis spent its first phase on a **linearized reconstruction**: assume the base weights are public, observe the
weight change, and fit candidate images to it through a first-order model of training. This note replaces an earlier
draft that presented the two routes as competitors and reported which recovered more images. That framing is
withdrawn. The routes are not competitors, and the reason is algebraic rather than empirical.

Nothing below claims one route beats the other. Two of the three main statements are proved; the third is measured
and is a statement about solvers, not about information.

## 1. The LoRA-aware linearized model is never mis-specified

Write the release as the two factors `(A_T, B_T)`, the private inputs to the adapted layer as `H`, and the model as
`B_T ≈ Σ_i r_i (A_T h_i)^T` with the coefficients `r_i` free. This is the linearized route written in the released
factors rather than in the merged weight change, and it is the form an attacker can actually write, because at the
first step the released down-projection *is* the initialization.

**The model fits exactly at every step count, not only at the first.** By the closure lemma the released `B_T` has
the form `Q_T (A_0U)^T`, so every row of it lies in `S = col(A_0U)`. The design `A_T H` spans that same `S`. Target
rows and design columns therefore span the same subspace and an exact coefficient matrix exists for every `T`.

Measured, as a confirmation rather than as the evidence (job 308859, MNIST letter a, PCA chart at k=32; the floor is
the closed-form least squares at the true images):

| | model floor, LoRA-aware form | model floor, merged-weight form |
|---|---|---|
| T = 1 | 2.89e-16 | 9.64e-01 |
| T = 400 | 8.05e-16 | 7.15e-01 |

So there is no regime boundary in the linearization to find, and the earlier plan to measure one is abandoned. The
merged-weight form, which fits `ΔW ≈ Σ_i r_i φ(x_i)^T`, *is* mis-specified at every T, because a LoRA step moves the
adapter by the gradient composed with the adapter rather than by the gradient. That form is the original Experiment
B written against a merged release, and it should not be used as the comparison's linearized arm.

## 2. The two routes have the same zero set

Fitting the released factor exactly requires the target's row space to sit inside the design's column space. With
exactly `N` candidates the design spans at most `N` dimensions while the target's row space has dimension exactly
`N`, so containment forces equality, and every individual candidate reading must lie in that subspace. The projector
onto its orthogonal complement composed with the released `A_T` is precisely the certificate. Hence:

> the free-coefficient linearized fit is exact **if and only if** every candidate is a zero of the certificate *and*
> the candidate readings are linearly independent.

The certificate is the per-candidate form of that condition; the linearized representer is its joint form. I checked
this identification at the algebra at the request of the reviewer who derived it, and it is definitional rather than
contingent, given one hypothesis: it needs `rank B_T = N`, that is, all `N` images recorded. **Where that fails the
"only if" direction breaks specifically**: with `N' < N` recorded, an exact fit requires only that the design cover
an `N'`-dimensional subspace, so candidates need not be certificate zeros at all, and the independence clause cannot
be satisfied by `N` candidates spanning `N'` dimensions.

Two consequences worth stating plainly.

- **The equations cannot be the difference between the routes.** Whatever one route can identify, so can the other.
- **The blend degeneracy is shared.** `N` independent blends of the private images satisfy both conditions, so the
  representer does not escape the superposition problem; it only excludes duplicate candidates.

## 3. What differs is the search, and it cuts both ways

Since the zero sets coincide, every measured gap is a property of the solver and of how many images one start must
place. Measured on the same release, chart, starts and budget:

| | linearized, LoRA-aware, free coefficients | certificate |
|---|---|---|
| residual reached | 1.05e-02 | 4.39e-14 |
| its own model floor | 2.89e-16 | — |
| ratio to floor | 3.6e13 | — |
| images recovered, 20 starts | 0 of 8 | 4 of 8 |

Residual far **above** its floor with wrong images is a search failure, not an alias. The linearized solve does not
converge at this budget; it is not carrying less information.

**The certificate's advantage is separability.** One certificate start solves for one image and the other `N−1` never
enter it, so coverage accumulates over starts. One linearized start must place all `N` at once, and its descent
direction is driven by the joint residual, so a candidate that would have landed alone can be dragged off by the
others.

**The representer's advantage is distinctness, and it is real.** Its exactness condition requires the candidate
readings to be independent, which the per-candidate certificate condition does not: the certificate happily accepts
`N` copies of the same image. Coverage is exactly the certificate's measured bottleneck — 53 of 89 landings on a
single image in one cell, and 1 to 2 of 8 on the pixel layer.

That is a division of labour rather than a ranking, and it names a construction: generate candidates with the
certificate, which is separable and cheap, then use the representer's independence condition to select a spanning
subset. That is the chaining step earlier plans named without specifying.

## 4. Solver fairness, since the comparison rests on it

Two corrections were needed before any of the numbers above could be quoted, both found in review.

- **The joint solver was handicapped at initialization.** With the coefficients starting at zero and the model being
  their product with the candidate features, the latents receive exactly zero gradient on the first step. Measured
  cost on the same cell: a factor of 19 at T=1 and 7 at T=400.
- **The fix is also the right algorithm.** The model is linear in the coefficients, so they are eliminated in closed
  form and the search runs over the latents alone. The objective becomes the release projected off the span of the
  candidate readings, which is the same shape as the certificate's objective, and the unknowns drop by a third. Both
  solvers are reported per cell so the handicap stays a measured row.

## 5. What the earlier route measured, kept for the record

These numbers are from the pre-certificate work and are unchanged; they are the reason the direction moved. Each is
labelled by what the attacker had to know.

- Best full fine-tuning results, SSIM 0.9999, need **oracle coefficients** computed from the true images.
- In the realistic free-coefficient mode, recognizable recovery is an **N=2 phenomenon**: mean per-image SSIM 0.922,
  0.605, 0.536, 0.252 at N = 2, 4, 6, 10, against trivial mean-image baselines of 0.763, 0.674, 0.606, 0.564. It
  beats the baseline only at N=2.
- Direct weight inversion collapses the same way while knowing the entire recipe, and the gradient-bridge decoder's
  cosine never converts into pixels.

## 6. Open

The remaining question is not which route identifies more. It is whether the joint solve can be made to converge —
it is beatable by a better solver, and that objection should stand rather than be argued around — and whether the
hybrid in section 3 buys the coverage the certificate lacks. An N=1 cell is running: with one image there is no
coupling, no arity gap and, in the LoRA-aware form, a zero model floor, so a failure there would be conditioning or
parameterization rather than anything scientific.
