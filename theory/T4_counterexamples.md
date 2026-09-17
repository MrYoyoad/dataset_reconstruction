# T4 — Counterexamples and the repairs they force

Every entry is a construction that breaks a *stated* version of the multilayer theorem, followed by the extra
hypothesis that repairs it. Counterexamples are results here, not failed proofs.

## C1 — Rank death by training-span inflation (breaks "the certificate survives at depth")

**Construction.** One adapted layer with `r = 4`, `N = 1`, `T = 8`, upstream adapters driving a drift whose
directions `Delta_{l,t}` are in general position. Then `N' = dim Hcal = min(n_l, 8) = 8 > r`, so
`row(B_{l,T}) = R^r`, `C_full = 0`, `rank C_full = 0`.

**What it kills.** The reading of Prop. A as "depth is free". Exactness is free; *rank* is not. A deep certificate
has `r - N'` rows, and `N'` grows with the number of steps times the number of independent drift directions.

**Repair.** The theorem must be stated with the hypothesis `N' < r`, and the practical programme must bound `N'`.
`N'` is bounded by `N * (1 + #{t : Delta_t has a new orthogonal direction})`, so what is needed is not small drift
but **low-rank drift**: drift confined to a few directions (or to `col(H^0)` itself, T3.2) leaves `N'` small even
when `||Delta||` is large. **This inverts the brief's small-drift hypothesis into a small-*rank*-drift hypothesis**,
which is a different and more achievable regime — and it is directly measurable as the numerical rank of
`[Delta_{l,0} ... Delta_{l,T-1}]`.

## C2 — Contamination when `rank B_T < N'` (breaks exactness itself)

**Construction.** Duplicate two of the `N` private examples, or include one already fit at initialisation
(`D_{l,t}` column zero). Then the accumulated residual directions are deficient, `rank B_T = q < N'`,
`row(B_T) subsetneq col(X')`, and `C_full H_l^0 != 0` with **no small parameter** controlling the size.

**What it kills.** Any claim that the deep certificate degrades gracefully. Below the rank hypothesis it is not
approximately right, it is wrong by `O(1)`.

**Repair + attacker-side detectability.** This is inherited verbatim from the single-layer theory (Rev-10
hypothesis 3) and the same test applies: `rank B_T` is observable. The new content at depth is that the target is
`N'`, which the attacker **cannot** observe, so `rank B_T = N` no longer certifies anything — it is equally
consistent with "no drift, exact" and "drift, contaminated". *This is the single most important practical warning
of the track.* The candidate resolution is the singular spectrum: a clean `N`-plus-decaying-tail spectrum with a
large `gap` indicates the first, a filled-in spectrum the second.

## C3 — Drift inside the private span costs nothing (breaks the choice of small parameter)

**Construction.** Choose the upstream adapter so that its update maps `col(H_l^0)` into itself — e.g. a linear
network whose upstream LoRA has `col(sB_{l-1}A_{l-1}H_{l-1}^0) subset col(H_l^0)`. Then `col(H_{l,t}) = col(H_l^0)`
for all `t`, `N' = N`, and the certificate is exact at full rank `r - N` **however large `||Delta_{l,t}||` is** —
the drift can be 100%.

**What it kills.** `eps = ||Delta||/||H^0||` as the small parameter. See T3.2.

**Repair.** State everything in `eps^perp`. Note this also has an experimental consequence: M1 must measure the
orthogonal drift and the drift *rank*, not the drift norm. Measuring `||Delta||` alone would produce a scatter plot
with no relationship and invite the wrong conclusion.

## C4 — Depth cannot beat the first layer (breaks naive rank additivity)

**Construction.** Any network in which the first adapted layer's feature map has a rank-deficient Jacobian:
`rank(J_{Phi_1}(x*) J_G(z*)) = k_1 < k`. Since `Phi_l^0 = psi_l . Phi_{l-1}^0`, every deeper Jacobian factors
through it, so `ker(J_{Phi_1}J_G) subset ker(J_{Phi_l}J_G)` for all `l` and

    rank J_F  <=  k_1   for ANY number of stacked layers.

**What it kills.** "Stack enough layers and you will reach `k`". Adding depth can never recover a chart direction
that the first adapted layer is already blind to.

**Repair.** The additivity statement (T5) must be stated *inside* `row(J_{Phi_1}J_G)`, with `k_1` and not `k` as
the saturation level, and the experimental target for M3 is `min(k_1, sum_l q_l)`, not `min(k, sum_l q_l)`.

## C5 (searched, not found) — a cancellation making the truncation error quadratic

Searched in T3 by four routes (projection structure, LoRA gauge symmetry, Gaussian averaging, slow-drift limit).
All four fail, and T3.1 exhibits a nonvanishing first-order coefficient in closed form. Recorded here so the
search is not repeated.

## Status

C1, C2, C3, C4: **explicit constructions, PROVED as counterexamples** (C1/C3/C4 fully explicit; C2 inherited from
the single-layer theory with the new non-detectability observation, which is the part needing numerical
confirmation). C5: **no counterexample to my own negative result found**, consistent with T3.
