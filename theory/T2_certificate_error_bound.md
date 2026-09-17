# T2 — Certificate survival at depth

## Statement

### Proposition A (training-span closure) — the main structural result

Fix an adapted layer `l` whose inputs `H_{l,t}` may vary with `t`. Let `Hcal := sum_{t<T} col(H_{l,t})` be the
**training span**, `N' := dim Hcal`, `U'` an orthonormal basis of `Hcal`, `V'` of `Hcal^perp`, and
`X' := A_{l,0} U' in R^{r x N'}`. Then for all `t <= T`:

    (i)   A_{l,t} V' = A_{l,0} V'                (the off-span block is never touched)
    (ii)  row(B_{l,t}) subset col(X')
    (iii) col(A_{l,t} U') subset col(X')

Consequently, **if `rank B_{l,T} = N'`** (so `row(B_{l,T}) = col(X')`), the full certificate
`C_full := P_{row(B_{l,T})^perp} A_{l,T}` satisfies

    C_full H_{l,t} = 0   for EVERY t < T,   in particular   C_full H_l^0 = 0   EXACTLY,

with `rank C_full = (min(r, n_l) - N')_+` and `ker C_full = Hcal (+) ker A_{l,0}`. The kernel clause is the
primary statement; the rank scalar follows from it. It equals `r - N'` **only when `n_l >= r`**. When the layer is
narrower than the adapter rank (`n_l < r`), `A_{l,0}` is a.s. injective (`ker A_{l,0} = 0`), so the certificate has
`rank = n_l - N'` — it cannot supply `r` independent output directions. Writing it as a bare `r - N'` overcounts
the surviving equations at every narrow layer, and it contradicts the kernel clause on the same line. (Corrected
per audit A18, 2026-09-17; the harness has always used the `min` form.)

**No small-drift hypothesis is used.** The certificate at a deep layer is exact, not perturbative. What depth
costs is *rank*: `(min(r, n_l) - N')_+` instead of `min(r, n_l) - N`, where `N <= N' <= min(n_l, N*T)`.

### Corollary A.1 (the real failure mode is discontinuous)

`rank C_full = (min(r, n_l) - N')_+`. Since `N'` grows with the diversity of the drift, the certificate
**degrades by losing equations, and dies outright once `N' >= min(r, n_l)`** — at which point `C_full = 0`. There is no regime in which it is
"slightly wrong": it is exact until it is empty. (If instead `rank B_{l,T} = q < N'`, the hypothesis fails and
`C_full` is *contaminated*, `C_full H_l^0 != 0`, matching hypothesis 3 of the Rev-10 framework.)

### Theorem T2.1 (exact error identity for the truncated certificate)

The attacker wants rank `r - N`, not `r - N'`, so truncates: let `Rhat` be the top-`N` right-singular subspace of
`B_{l,T}`, `R := row(B_{l,T})`, and `Ctil := P_{Rhat^perp} A_{l,T}`. If Proposition A's rank hypothesis holds and
`Rhat subset R`, then

    Ctil H_l^0  =  P_{R (-) Rhat} A_{l,T} H_l^0            (exact identity, not a bound)

where `R (-) Rhat` is the orthogonal complement of `Rhat` inside `R` — the *discarded drift directions*. The
certificate error is exactly the part of `A_T H_l^0` that lives in the row directions the attacker threw away.

### Theorem T2.2 (bound, with the three error channels separated)

Let `(A_T^0, B_T^0)` be the counterfactual release produced by the **same** error signals `D_{l,t}` driving the
**frozen** inputs `H_{l,t} equiv H_l^0`, and `R_0 := row(B_T^0) = col(A_{l,0} U_0)`, `U_0` a basis of
`col(H_l^0)`. Then

    ||Ctil H_l^0||  <=  sinTheta(Rhat, R_0) * ||A_T H_l^0||   +   ||A_T - A_T^0|| * ||H_l^0||          (*)
                            \_____ channel 2 _____/                \____ channel 3 ____/

and, with `gap := sigma_N(B_{l,T}) - sigma_{N+1}(B_{l,T}) > 0`, Wedin's theorem gives

    sinTheta(Rhat, R_0)  <=  ||B_{l,T} - B_T^0|| / gap.

Channel 1 (feature drift) is what drives `||B_T - B_T^0||` and `||A_T - A_T^0||`, and a discrete Gronwall on the
coupled SGD recursion gives, with `Dbar := max_t ||D_{l,t}||`, `Hbar := max_t ||H_{l,t}||`, `Abar, Bbar` the
trajectory bounds and `g := eta s Dbar Hbar`,

    ||A_T - A_T^0||, ||B_T - B_T^0||  <=  eta s Dbar (Abar + Bbar) * ((1+g)^T - 1)/g * eps^perp * ||H_l^0||.

Hence

    rho_l := ||Ctil_l H_l^0|| / (||Ctil_l|| ||H_l^0||)  <=  K_l * eps_l^perp  +  O((eps^perp)^2),

    K_l  =  [ ||A_T|| / ||Ctil|| ] * [ Gamma / gap ]  +  [ Gamma / ||Ctil|| ],
    Gamma := eta s Dbar (Abar + Bbar) ((1+g)^T - 1)/g * ||H_l^0||.

**The controlling quantity is the ORTHOGONAL drift `eps^perp`, not `eps`** (T3): drift that stays inside
`col(H_l^0)` does not enlarge `Hcal` and costs exactly nothing.

**The certificate is blind to drift in the error signal.** `D_{l,t}` appears identically in both runs, so channel
1 is driven by `Delta H` alone. This is the depth-analogue of the Rev-10 "loss-agnostic" hypothesis: at a deep
layer, everything the rest of the network does to the *gradients* is irrelevant; only what it does to the layer's
*inputs* matters.

## Assumptions

| # | assumption | used where |
|---|---|---|
| B1 | `B_{l,0} = 0` | base case of the induction in Prop. A |
| B2 | SGD-class update (bilinear gradient shape), any loss | the induction step; `D_{l,t}` arbitrary |
| B3 | `rank B_{l,T} = N'` | to turn `row(B_T) subset col(X')` into equality |
| B4 | `Rhat subset R` | T2.1; automatic since `Rhat` is a singular subspace of `B_T` and `N <= q` |
| B5 | `gap > 0` | Wedin, T2.2 |
| B6 | `rank A_{l,0} = min(r, n_l)` (a.s. for Gaussian) | `ker C_full = Hcal (+) ker A_0` |

## Proof

**Proposition A.** Induction on `t`. At `t=0`: `A_0V' = A_0V'` trivially; `B_0 = 0` (B1) gives (ii); (iii) is
`col(A_0U') = col(X')`. Step: the `A`-update is `A_{t+1} = A_t - eta s B_t^T D_t H_t^T`. Its rows lie in
`row(H_t^T) = col(H_t) subset Hcal` (definition of `Hcal`), so `A_{t+1}V' = A_tV' = A_0V'`, giving (i). For (ii),
`B_{t+1} = B_t - eta s D_t (A_t H_t)^T`, and `A_tH_t = (A_tU')(U'^T H_t)` because `col(H_t) subset Hcal`, so
`row(B_{t+1}) subset row(B_t) + col(A_tU') subset col(X')` by the inductive (ii),(iii). For (iii),
`A_{t+1}U' = A_tU' - eta s B_t^T D_t H_t^T U'`, and `col(B_t^T) = row(B_t) subset col(X')` by (ii), so
`col(A_{t+1}U') subset col(X')`. []

Given B3, `row(B_T) = col(X')` exactly, so `P_{row(B_T)^perp} = P_{col(X')^perp}` annihilates `col(A_TU')` by
(iii). For any `t < T`, `H_{l,t} = U'(U'^T H_{l,t})`, hence
`C_full H_{l,t} = P_{col(X')^perp}(A_TU')(U'^TH_{l,t}) = 0`. Taking `t=0` and using `H_{l,0} = H_l^0` (which is
B1 at *every* layer) gives the base-representation statement. `rank C_full = (min(r, n_l) - N')_+`: `A_0` is a.s.
of full rank `min(r, n_l)`, so on `Hcal^perp` the map `C_full = P_{col(X')^perp}A_0` has rank `min(r, n_l) - N'`,
with `ker C_full = Hcal (+) ker A_0` — the `ker A_0` term nonempty exactly when `n_l > r`, and `A_0` injective
(so `rank = n_l - N'`) when `n_l < r` (B6). It reduces to `r - N'` only for `n_l >= r`. []

**T2.1.** `Rhat subset R` gives the orthogonal decomposition `P_{Rhat^perp} = P_{R^perp} + P_{R (-) Rhat}`.
Apply to `A_TH_l^0` and use `P_{R^perp}A_TH_l^0 = C_full H_l^0 = 0` from Prop. A. []

**T2.2.** Write `Ctil = P_{Rhat^perp}A_T` and `C^0 := P_{R_0^perp}A_T^0`, which satisfies `C^0 H_l^0 = 0` by the
single-layer (frozen-input) case of Prop. A applied to the counterfactual run. Then

    Ctil H^0 = (P_{Rhat^perp} - P_{R_0^perp}) A_T H^0  +  P_{R_0^perp}(A_T - A_T^0) H^0

and `||P_{Rhat^perp} - P_{R_0^perp}|| = sinTheta(Rhat, R_0)`, `||P_{R_0^perp}|| = 1`, giving (*). Wedin applied to
`B_T = B_T^0 + (B_T - B_T^0)` with singular gap `gap` bounds the angle. The Gronwall step: subtracting the two
recursions,

    dA_{t+1} = dA_t - eta s ( dB_t^T D_t H_t^T + B_t^{0T} D_t Delta_t^T )
    dB_{t+1} = dB_t - eta s D_t ( H_t^T dA_t^T + Delta_t^T A_t^{0T} )

so with `a_t := ||dA_t||`, `b_t := ||dB_t||`: `a_{t+1} <= a_t + g b_t + eta s Dbar Bbar ||Delta_t||` and
`b_{t+1} <= b_t + g a_t + eta s Dbar Abar ||Delta_t||`. Summing `u_t := a_t + b_t` gives
`u_{t+1} <= (1+g)u_t + eta s Dbar(Abar+Bbar)||Delta_t||`, whence
`u_T <= eta s Dbar(Abar+Bbar) max_t||Delta_t|| ((1+g)^T-1)/g`. Only the component of `Delta_t` orthogonal to
`col(H_l^0)` enlarges `Hcal` and hence survives into the certificate — see T3 for why the in-span part drops out —
so `max_t||Delta_t||` may be replaced by `eps^perp ||H_l^0||` in the leading term. []

## Exactly where each assumption enters

- **B1** twice, both load-bearing: base case of the induction, and the identification `H_{l,0} = H_l^0` that makes
  the *base* representation a member of the training span. Without `B_{l',0} = 0` at the **upstream** layers,
  `H_{l,0} != H_l^0` and Prop. A says nothing about the base representation — only about the (unknown) `t=0` one.
- **B2** at the induction step only, through the bilinear shape `grad_A = sB^T D H^T`, `grad_B = sD H^T A^T`. Adam
  and friends break it exactly as in the single-layer theory.
- **B3** is the hinge between "exact" and "contaminated". It is **not** implied by the construction and it is
  strictly harder to satisfy at depth than at the first layer, because it demands `N'` independent accumulated
  residual directions, not `N`. Attacker-side check: `rank B_T` is observable; `N'` is not.
- **B4** is free. **B5** fails exactly when the drift directions are as strong as the private ones, which is the
  regime where truncation is meaningless anyway.

## Counterexample search

1. **Rank collapse (constructed, see T4-C1).** `N' >= r` forces `C_full = 0`. Concrete: `N=1`, `r=4`, `T=8`, drift
   directions in general position -> `N' = 8 > 4`, certificate empty. This is the *generic* fate of a deep layer
   under long training with unconstrained drift, and it is why "small drift" matters after all — not to keep the
   certificate accurate, but to keep the drift directions from being *independent* enough to inflate `N'`.
2. **Contamination (B3 fails).** Duplicated examples, an example already fit at init, or degenerate labels give
   `rank B_T < N'` and then `C_full H_l^0 != 0` with no small parameter controlling it.
3. **`Delta` inside the span.** If `col(H_{l,t}) subset col(H_l^0)` for all `t` (e.g. an upstream adapter whose
   update happens to act within the private feature span), then `N' = N` and *everything is exact at full rank*
   however large `||Delta||` is. This is the sharp counterexample to the brief's framing that `||Delta H||` is the
   right small parameter.

## Status

**PROVED.** Prop. A, Cor. A.1, T2.1 and T2.2 are proved as stated. Note what is *not* claimed: nothing here says
`N'` is small, and the whole practical question has moved to bounding `N'` and the singular spectrum of `B_T`.

## Numerical sanity check

`experiments/multilayer_cert/theory_checks.py::check_T2` and `survival.py`. **RUN 2026-09-07, jobs 688036 /
692603 — all confirmed** (full table: [../experiments/multilayer_cert/RESULTS.md](../experiments/multilayer_cert/RESULTS.md)):

| claim | measured |
|---|---|
| Prop. A, deep layers, drift **> 100%** (up to 923%), 13 rows | `rho_full` max **6.5e-15** |
| Prop. A, all 110 `B3`-holding rows, drift 0 – 923% | `rho_full` max **2.7e-13** |
| `rank C_full = (min(r-N', n_l-N'))_+` | **110 / 110** rows |
| Cor. A.1 discontinuous death (`r=6` ladder) | `rank C` = 4, 2, **0**, 0, 0, 0, 0, 0 as `N'` = 2, 4, 6, 8, 9 |
| T2.1 exact error identity | `1.1e-14` |
| `B3` fails (`rank B_T < N'`) -> contamination | median `rho_full` **2.3e-4**, max **0.27** — `O(1)`, no small parameter |

> **What the `rank C_full` row does and does not establish (audit A18, 2026-09-17).** The harness computes the
> expected rank as `max(0, min(r - N', n_l - N'))` — the corrected law — and compares the measured rank against
> *that*. So the agreement verifies the code implements the corrected formula; it is a regression test on the
> implementation, **not** independent evidence for the proposition, because the only law it could have disagreed
> with (the bare `r - N'`, which the earlier code and the run log both used) was engineered out of the comparison
> before the check ran. This is not hypothetical: the stated `r - N'` law **was** tested and **failed** — jobs
> `674726` (`ba9ca00`) and `683234` (`a2f71c6`), where `expect_rank_C = r - N'` gave `passed: false` — and passed
> only at `688036` (`de8b0ae`) once the expectation became the `min` form (`expect_rank_C` = 13, 5 there vs 21, 23
> before). The disagreement survives in a **sane, non-diverged** row: `688036` layer 0 has `n_l = 16 < r = 24`,
> `N' = 3`, B3 holding at zero drift, and measures `rank C = 13 = min(21, 13)`, against the as-written `r - N' =
> 21` — an eight-equation overcount. A fresh run comparing against the as-written law reproduces this **known**
> failure under an honest name; it is a record being made honest, not a new result.

**A measured law the theory did not predict, and which the whole picture now rests on:** the training span
inflates at the *maximal* rate, `N' = N*T` (116/129 deep rows; `N' <= N*T` in all of them). Every SGD step adds
`N` fresh directions. With `rank C = min(r, n_l) - N*T` this gives a **certificate lifetime**: the deep-layer
certificate is empty once `T >= min(r, n_l)/N` (measured: 12 rows at `rank C = 0`, every one with
`N' >= min(r, n_l)`). This, not small drift, is the binding constraint at depth.

**Numerical caveat that inverted an earlier read of this check.** A rank computed as `s > rtol*s[0]` calls a
numerically ZERO matrix *full* rank, so an annihilated certificate reported `rank r` instead of `rank 0` and the
death ladder looked flat. Ranks here use an absolute floor tied to `||A_T||`.
