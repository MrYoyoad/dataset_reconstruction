# Multilayer certificate track — measured results

Runs of 2026-09-07. Checks: job **688036** (`results/multilayer_cert/theory_checks_688036.jsonl`).
Sweep: job **692603** (`survival_692603.jsonl`, 216 rows = 54 configs x 4 layers, 3 seeds x lr in
{0.01…3.0} x T in {2,4,8}). FP64, CPU. Theory: [../../theory/](../../theory/).

**Reading rule.** 11 of 54 configs diverge (the network explodes, drift `>1e100`); they are recorded with
`diverged: true` or excluded by a magnitude filter, and every number below is over the **43 sane configs / 172
rows**. A diverged cell is data about the edge of the regime, not a measurement of the certificate.

## 1. The certificate is EXACT at depth, at any drift — Prop. A

The brief expected a perturbative `O(eps)` statement. What the algebra and the measurement both say is that the
full certificate is *exact*, with no small parameter, wherever its rank hypothesis `rank B_T = N'` holds.

| rows | condition | `rho_full = ||C_full H_l^0||/(||C_full|| ||H_l^0||)` |
|---|---|---|
| 110 | `rank B_T = N'`, all layers, all drifts | max **2.7e-13** |
| 67 | deep layers only (`l >= 1`), drift 0.2% – 923% | max **2.7e-13** |
| 13 | deep layers at drift **> 100%**, up to **923%** | max **6.5e-15** |

The check job confirms the same on a separate net at 62% and 213% drift: `rho_full` = 1.2e-15 and 8.8e-16, with
`max_t rho = 3.2e-14` over *every* training step, not just `t=0`.

**There is no drift dependence to fit.** The residual sits at machine precision across four decades of drift.

## 2. What depth actually costs: rank, and a hard lifetime

`rank C_full = (min(r - N', n_l - N'))_+` held in **110 / 110** rows. And the training span inflates at the
**maximal** rate:

    N' = N * T   in 116 / 129 deep rows,   and   N' <= N * T   in ALL of them.

Every SGD step contributes `N` fresh directions to the span. Combining the two gives the operational law:

> **Certificate lifetime.** A deep-layer certificate has `rank = min(r, n_l) - N*T`, so it is empty once
> `T >= min(r, n_l)/N`. Here (`r=16, N=3`) it dies at `T = 6`: **12 rows have `rank C = 0`, and every one of
> them has `N' >= min(r, n_l)`.**

The death is discontinuous, exactly as Cor. A.1 predicts. The ladder from the check job (`r=6`, layer 1):

| T | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| `N'` | 2 | 4 | 6 | 8 | 9 | 9 | 9 | 9 |
| `rank C_full` | 4 | 2 | **0** | 0 | 0 | 0 | 0 | 0 |
| predicted | 4 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |

**This replaces "small drift" as the governing condition.** The regime that matters is not small `||Delta H||`;
it is a small *training span*, i.e. few steps, few examples, or drift confined to few directions.

## 3. When the rank hypothesis fails, the failure is O(1)

In the 62 rows where `rank B_T < N'` (in this net, the last layer, whose width `m = 8` is below `N'`), the
certificate is **contaminated, not weakened**: median `rho_full` = 2.3e-4, max 0.27. There is no small parameter.
This is theory/T4-C2, and its practical sting is that `rank B_T = N` is **not** an attacker-side certificate of
exactness at depth, because the attacker cannot observe `N'`.

## 4. The truncated certificate: first order, no cancellation

The attacker who wants rank `r - N` back must truncate, and then the error is genuinely perturbative:

    rho_trunc / delta_perp  =  0.082 (median)   -- flat across three decades of drift:
    0.098 @ 2e-3   0.095 @ 2e-2   0.100 @ 6e-2   0.075 @ 0.20   0.074 @ 0.42   0.055 @ 0.98   0.051 @ 6.7

so `K_l ~ 0.1` in this net, with a mild sub-linear bend above 20% drift. The closed-form coefficient of T3.1
matches to **0.45%** and the log-log slope is **1.0004** (an `O(eps^2)` rate would read 2.00). Drift confined to
the base feature span costs nothing: at `||delta|| = ||h||` (100% drift, in-span) the residual is 4.7e-14.

**`delta_perp`, not `delta`, is the right axis** — measuring the drift norm alone would have produced a scatter
plot with no law in it.

## 5. Depth adds information, additively, up to two ceilings

Stacked chart-Jacobian rank, `k = 20`, four layers, `r - N = 9` per layer:

| layers stacked | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| `rank J_F` | 9 | 18 | 20 | 20 |
| `min(k_1, sum q_l)` | 9 | 18 | 20 | 20 |

Exact agreement — **but only in the rank-preserving regime, and this cell does NOT discriminate the two candidate
laws** (updated 2026-09-17). This is a random FP64 MLP with widths >= k, so `d_j = rank M_j = k_1` at every layer,
and there T5.2's `min(k_1, sum q_l)` and F11's corrected `min_j(d_j + sum_{l<j} q_l)` coincide exactly. The M4
cell confirms additivity **in its valid regime**; it is evidence for neither law over the other. **T5.2 is FALSE
as stated** (audit F11): the equality assumes the per-layer row spaces spread independently inside `row(M_1)`,
which nesting (T5.1) forbids. See `notes/m4_additivity_verification_2026-09-17.md`.

### 5b. Real-encoder discrimination — the corrected law confirmed, T5.2 refuted (2026-09-17)

`experiments/multilayer_cert/real_encoder_ranklaw.py` runs the two-law test on a **real** frozen 15-layer MNIST MLP
(`mnist_mlp_d15w1000.pth`), zero-drift certificates, `d_j`/`q_l`/stacked-rank on one row with a config key, across
a tolerance ladder. **THEORY test, not an attack config** (the pixel arm's `k_1` is ~10x the k<=66 identifiability
cap). Job **354535** (attested: `git 59cbe42-dirty`, `script_sha dd81201f5399`), `r=108` (margin 100), `N=8`.

Adapting from layer 3 through the encoder's rank cliff (`d_j = 784, 687, 217, 167, 139, 104, 84, 73`):

| layers stacked L | 5 | 6 | 7 | 8 |
|---|---|---|---|---|
| T5.2 `min(k_1, sum q_l)` | 500 | 600 | 684 | 756 |
| corrected `min_j(d_j + sum_{l<j} q_l)` | **417** | **417** | **417** | **417** |
| measured rank, ladder 1e-6/8/10/12 | 326/372/402/418 | 326/373/403/418 | 326/372/402/418 | 326/372/402/418 |

**Two claims, and the first must NOT carry the second.**

**(1) T5.2 overpredicts the USABLE rank ~2× at real scale (rung-independent); the EXACT-rank refutation is the
synthetic F11, not this measurement (see the §5c caveat, 6e's exact-vs-effective distinction).** The measured
effective rank is `saturated_below_k1: True` and lands at 326/372/402/418 across the ladder — every rung far below
T5.2's 756, which is 96% of the ambient 784. So at any meaningful tolerance the usable rank is ~2× smaller than
T5.2 predicts, and T5.2's extra directions (if exact-real) sit below the fp64 floor with no usable information.
This does NOT depend on which rung is read. The structure is the nesting ceiling: `d_3 + q_1 + q_2 = 217 + 200 =
417` traps everything from the fourth adapted layer on inside the 217-dim row space, so layers 5-8 add nothing (as
an effective-rank statement — whether 417 is an exact rank is the pending gap question). The rank-preserving
control (first adapted layer at the pixel input, `d_j = 784`) saturates at both laws' common value — the positive
control that makes this a live test.

**(2) The corrected law's VALUE (417) is consistent in direction but NOT confirmed.** The harness reports
`matches_corrected: False` **and** `matches_t52: False` — it matches neither. The apparent ±1 agreement is **rung
selection, not tolerance**: the `matches_corrected` field compares the **1e-10** rung (402), while the 418 quoted
above is the **1e-12** rung — the agreement came from reading a different rung than the verdict uses. And the
ladder has **not converged**: 326 → 372 → 402 → 418 is monotone increasing with no plateau — a 28% climb, still
rising at the finest rung — so 418 is not an elbow, it is the last point before the rungs ran out. The increments
(46, 30, 16) decay by ~half per step and extrapolate to a limit of **~430–443** (geometric tail), *above* 417; the
trend points *past* the corrected value, not at it. Per the ladder rule
(an elbow that moves with the cut is a property of the cut), the result is the elbow's STABILITY across the
ladder, and there is none yet. **Value confirmation is deferred to the A100 run 354537, which must extend the
ladder to rungs finer than 1e-12 and record the singular spectrum around the elbow** to show a plateau (or its
absence): a plateau near 417 confirms the corrected law; a plateau near 430 means it *under-predicts* on real data
— a different, more interesting result.

**At attack scale (`chart66` arm, k<=66) depth adds NOTHING, and this bounds what claim (1) means for an attack.**
All 16 chart66 rows: `discriminates: False`, `ladder_spread: 0`, everything saturating at `k_1 = 66` with
`q_l = 66 = k_1` — **one adapted layer already saturates the chart, so stacking more layers buys zero at buildable
width.** The two laws therefore coincide there and the rank-law distinction is invisible. So the two statements
must travel together: T5.2 is *mathematics* and is refuted wherever tested (claim 1), but for an *attacker*
restricted to a buildable chart (`k <= 66`, the identifiability cap), depth is free and the whole re-costing is a
theory statement at ~10x the reachable width. The planner acts on the second. **Open (see below): at what `k*`
does depth stop being free — i.e. where does the nesting begin to bind between k=66 and k=784?** That is a
`k`-sweep, the same axis as the chart-window question. **Provenance:** attested, `script_sha dd81201f5399`; 354535
ran on a shared GPU, so sub-1e-10 rungs are provisional independently of the convergence point above.

### 5c. The k-sweep — where depth stops being free, and what depth buys below it (2026-09-17)

> **⚠ PENDING the gap / tau→0 check (2026-09-18): the RANK numbers below are provisional.** 6e showed a real
> 15-layer φ can have NO spectral gap (smooth decay), in which case "rank" is a choice of threshold, not a
> property — and below the FP64 noise floor the count saturates to the ambient dimension `min(rows, in_dim)` by
> construction. So a law's value is real only if the count APPROACHES it (a gap) rather than CROSSING it toward
> ambient. My own ladder (`325·371·402·418·421·428` at 1e-6…1e-14) is still climbing through 417, which suggests
> **k\*=417 and the 1.8× ratio are CROSSINGS (tolerance choices), not integer ranks** — the honest object is then
> the effective rank at a stated tolerance, which is **theory/T5.4's own caveat arriving for the real encoder
> rather than only under drift** (the conjecture flagged this before it was measured — theory doing its job).
> **6e's depth sweep (C10, jobs 355792/355795) measured the mechanism:** with the training budget controlled, the
> law is EXACT wherever evaluable (nullity = predicted at depths 2/4/8, gaps 1e7–1e9), and the gap COLLAPSES ~7
> orders between depth 8 and 12 (depth 12/16: gap 1.48/1.26, no measurable rank). Depth destroys the
> *measurability*, not the law. **But C10 measures the RELEASE-route Jacobian (trained, head-only) — a different
> object from this stacked zero-drift CERTIFICATE Jacobian, so it does NOT predict mine** (recording it as
> "expected" would be the same cross-object inference struck elsewhere, in reverse). What it supplies is a
> candidate MECHANISM: through enough layers φ's own Jacobian spectrum spreads, and both routes differentiate
> through the same φ (the certificate is linear in the features, ≈ `C · J_φ · V`, inheriting J_φ's spectrum). So an
> absent gap here would be *consistent with* C10 with a named mechanism — but not *expected*. **355778 tests
> whether the mechanism reaches the certificate route; both outcomes are informative and neither is predicted:** an
> absent gap confirms k\*=417 is a crossing / effective-rank statement; a PRESENT gap at depth 15 (where the
> trained-release Jacobian loses it) is a real frozen-vs-trained asymmetry, with 6e's depth sweep as control.
>
> **What survives, and its exact scope (6e's exact-vs-effective distinction).** T5.2 and the corrected law are
> **exact-arithmetic** rank claims. On the SYNTHETIC net their exact ranks are well-defined (F11: real gap, 0/12
> vs 12/12 on integer ranks) and **T5.2 is refuted there**. On THIS real deep φ the exact rank is **not
> fp64-measurable** — 329 of T5.2's directions, if exact-real, would sit below the fp64 floor, and my measurement
> cannot reach them. So the real-net statement is about the **effective/usable** rank: T5.2's 756 is 96% of the
> ambient 784, while the effective rank at a meaningful tolerance (1e-10) is ~402 ≈ 51% — **T5.2 overpredicts the
> usable rank ~2×, and its extra directions, if they exist, carry no usable information.** Practically identical to
> a refutation; theoretically weaker — the exact-rank refutation is the synthetic F11, NOT this measurement. Do
> not write "T5.2 refuted at real scale" as an exact-rank claim. The clean-FP64 A100 run (355778:
> full spectrum + gap_at_corrected + ambient) settles approached-vs-crossed; the k\*/discrimination text below is
> restated per its verdict.
>
> **VERDICT IN (plain-GPU look 355781): NO GAP.** `gap_at_corrected ≈ 1.08–1.15` at every discriminating cell
> (k=512/692/784), `real_rank=False`, the spectrum decaying smoothly through index 417 (5e-11 → 1e-12, no jump)
> above the noise floor and the count climbing toward the ambient dimension below it (rank@1e-16 = 446/464/482). So
> **k\*=417 and the 1.8× ratio are CONFIRMED crossings / effective-rank statements, not integer ranks**, and there
> is **NO frozen-vs-trained asymmetry** — the frozen zero-drift certificate loses the gap just as 6e's trained
> release does, so C10's mechanism reaches the certificate route (the degradation is a property of φ's depth, not
> the route — a negative but real finding). A100 355778 confirms the deep profile below the plain-GPU floor; the
> gap verdict does not depend on it.
>
> **SAME-NET control (355835), read as SHAPES not scalars** (the `gap_at_corrected` ratios are taken at different
> indices per arm — 400 vs 48 — and are not comparable; the full spectra are). One depth-15 net, L=4, k=784,
> `first_adapted` swept 1→12 (frozen layers below = 0/3/7/11), everything else fixed. The spectra differ
> QUALITATIVELY at each arm's own scale: `first_adapted=1` (0 frozen below, d_j≈[784,784,784,687]) delivers all
> ~400 supplied conditions within ~4 orders of the top then a **cliff to machine zero** (a clean rank);
> `first_adapted=4/8/12` (3/7/11 frozen below, d_j contracting to [48,30,24,19]) give **smooth spectra spanning
> 11+ orders with no cliff**. So on ONE network the certificate's gap survives when its input is shallow and
> collapses as the frozen path below it deepens and its rank profile contracts — nothing else varying. **This is
> NOT the null** (the shallow arm has a genuine gap), so the collapse is **frozen-path-driven, not route- or
> training-driven**, and the separately-trained shallow net (355825) is not needed as a control. The variable is
> the frozen path's rank profile; depth is how you move it.

`real_encoder_ranklaw.py --ks 16..784` (job **355531**, attested `script_sha 01043d6a5fec`), `r=108`, `N=8`, real
15-layer MNIST encoder. The question M6 opened: at what chart width `k` does depth stop being free?

**k\* = 417, and it is the nesting ceiling EXACTLY, not approximately.** Below it the corrected law's `j=1` term
(`= k`) is the minimum, so both laws return `k` and coincide; above it the `j=3` term (`d_3 + q_1 + q_2 = 217 + 200
= 417`) binds and T5.2 does not see it. Measured transition brackets 417: `k=384` `discriminates: False`, `k=512`
`discriminates: True`.

**k\* is a threshold in `k` AND `L`, not `k` alone.** Discrimination needs `Σq_l > 417` as well as `k > 417`. At
`k=512, first=3` it is False for `L<=4` (`Σq<=400`) and flips at `L=5` (`Σq=500`). A shallow adapted stack never
reaches the ceiling and the laws coincide regardless of `k`.

**What depth does below the ceiling: it BUYS chart-constraint width, additively** (correcting an earlier "depth is
free at buildable width" recorded here and withdrawn 2026-09-17 — that confused the two laws COINCIDING with depth
adding nothing). The stacked certificate rank is `min(k, Σq_l, ceiling)` and `Σq_l` grows with `L`, so a wider
chart becomes fully constrained as layers are added: measured (first=1, `q_l≈100` at `r=108`) `k=128` is reached at
**L=2**, `k=256` at **L=3**, `k=384` at **L=4**, saturating only at the rank-dependent architectural nesting
ceiling (417 at this `r`). So depth raises the chart width the certificate can pin; `k*` is where that ceiling
binds and the two laws diverge. **The rise and `k*` are from the rank PREDICTIONS** (`d_j`,`q_l` measured as clean
1e-10 ranks), so they are robust; the stacked-rank VALUES at `k>=512` stay provisional (`ladder_converged: False`,
pending A100 355537).

**Arms:** the two `k=692` rows are the two `first_adapted` arms (1 and 3), not a duplicate — corrected **616**
(first=1) vs **416** (first=3); different encoder profiles.

**MNIST chart-error ladder, STANDALONE — do NOT compare to the CIFAR two-walls gate (different dataset, different
intrinsic dimension):** 0.421 (k=16) · 0.332 · 0.236 (k=66) · 0.188 · 0.170 · 0.133 · 0.090 · 0.038 (k=384) ·
**0.0030 (k=512)** · 1.6e-5 (k=692). MNIST's small intrinsic dimension lets a wide PCA chart capture it almost
exactly; the CIFAR releases' gate (0.0124) and PCA error (0.109 at k=384) were measured on a different release and
must never be quoted against these.

**The capacity RISES with depth: `cap(L) = min(L·(r−N) + (m−1), nesting ceiling)`.** (This REPLACES an earlier
"the depth-helps and well-posed windows do not intersect" claim, WITHDRAWN 2026-09-17 — it mistook a ratio for a
gap; the cap is not fixed, it rises with `L`.) At `L=1` this is the capacity line `k < m+r−N`, which is DERIVED (from
the `B_T`-variety dimension `N(m−1+r−N)`) and confirmed sharp to one unit of `k` at **r=16** (jobs 467914/469120,
varying `k` and `N`). **`r` was never varied in those jobs** (verified at the rows, all r=16), so evaluating the
line at `r=108` is an extrapolation along the untested `r`-axis — derivation-supported but NOT measured at r=108.
So 6e's `110` is the capacity line's `L=1` value r-extrapolated to 108, not a settled measurement there. The split is not a fit: `STATUS.md` records `B_T` on the rank-`N`, zero-column-sum variety
of dimension `N(m−1+r−N)`, so depth adds more `(r−N)` rank parts but only ONE `(m−1)` head part — only the head
carries the simplex constraint. **Positive headline:** depth buys usable chart width, from the single-layer cap
toward the architectural ceiling as layers are added. The exact deployed-`r` (8–64) gain is NOT quoted here,
because the ceiling itself scales with `r` through `q_l`, so it must be measured at deployed `r`, not extrapolated
from this `r=108` run.

**The window question is OPEN, blocked on one measurement.** Identifiability reaches `k ≈ ceiling` with enough
depth; whether that suffices depends on FIDELITY at that `k` **in this setting**. The MNIST chart-error ladder
above gives 0.038 at k=384 and 0.003 at k=512, but the only landing gate measured is CIFAR, and applying it to an
MNIST ladder is the cross-construction error struck from §5b/§5c. **A landing gate measured ON MNIST — the chart
error at which recovery actually stops on this net and these images — is what closes the window, and it is the run
queued after `idxverify`.**

**Per-layer budget check, pre-registered for the next run.** `q_l` here is MEASURED as `rank(Ctil_l M_l)`; the
formula value T5.2 assumes is `min(r_l−N', d_j)`. They can differ where the encoder contracts (measured `q_l =
84, 72` at the deep layers, against `r−N=100`). The next run emits BOTH per layer: agree → `cap(L)` grounded;
disagree → the per-layer budget and every cap-ladder number move. Both outcomes pre-registered; disagreement is a
finding, not a failure.

**Join note (6e):** the genuinely shared column is `chart_error` (depends only on checkpoint/image/k, not r). The
two nullity columns (`cert_route` r=108 zero-drift; `release_route` r=16 trained-release, carries drift) share the
`k` axis but describe DIFFERENT adapters at DIFFERENT `r` — a reader taking them as two views of one system is
wrong. The r-split is by necessity both ways: `release_route` is infeasible at r=108 (~34 GB), and the depth
discrimination is vacuous at r=16 (margins of 8, `Σq` never reaches the cliff). Never differenced, never compared
across `r`. Structural-emptiness cross-check when 355541 joins: does `stacked_rank / N` per `k` track the cap?

## 6. What did NOT replicate — my own prediction, refuted

I predicted in T5 that **tying the adapter initialisations across layers (R2) would collapse additivity**, and
offered it as a defence. It does not: shared-seed and independent both give `[9, 18, 20, 20]`. The certificates
still differ through the layer-specific feature span `U_l` and Jacobian `M_l`, so a shared `A_0` is **not** a
defence on its own. T5's defence claim is downgraded to a conjecture about tying the *whole* adapter.

## 7. Status of the four headline outputs the brief asked for

| output | state |
|---|---|
| (i) perturbative survival curve | **superseded** — there is no curve for `C_full`; it is machine zero everywhere. The curve that exists is `rho_trunc` vs `delta_perp`, section 4 |
| (ii) theory bound vs measured residual | partially: `K_l ~ 0.082` measured; the T2.2 Gronwall constant is not yet evaluated numerically for comparison |
| (iii) information rank vs number of layers | **done**, section 5 |
| (iv) reconstruction/replay improvement vs layers | **not started** (M4). Section 5 is the precondition the brief set for it, and it is met |

## Open / next

1. **M4** — certificate-only reconstruction with 1/2/4/8 layers on the real charts, then multilayer vs
   single-layer guidance inside replay.
2. **Evaluate the T2.2 constant** and compare against the measured `K_l ~ 0.082` (output ii).
3. **Real backbone.** Everything here is a random FP64 MLP. The span-inflation law `N' = N*T` is the claim most
   likely to change on a trained network with structured features, and it is the one the lifetime bound rests on.
4. **Metric ruling (from yoado-76, E1B/A1).** Where `N' > N` the certificate constrains the *span*, not the
   individual images, so a per-image norm is not the legitimate metric there — principal angles are. This has not
   yet been applied to any number above; sections 1-5 report residuals and ranks, not per-image recovery, so none
   of them is affected, but M4 must adopt it from the start.
