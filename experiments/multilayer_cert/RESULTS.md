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
> is **NO frozen-vs-trained asymmetry in the depth-DEPENDENCE** — both routes lose the gap as the frozen path
> deepens: the trained release keeps a gap through depth 10 at width 1000 and won't train past 12 (6e, jobs
> 355831/355844), and my frozen zero-drift certificate loses it as `first_adapted` grows (same-net control below).
> There is no matched depth-15 release point (the release won't train that deep at width 1000), so the shared claim
> is the depth-dependence, not a single-depth head-to-head — C10's mechanism reaches the certificate route (the
> degradation is a property of the frozen-path depth/rank-profile, not the route — a negative but real finding).
> A100 355778 confirms the deep profile below the plain-GPU floor; the gap verdict does not depend on it.

**SAME-NET control (355835): the gap collapse is frozen-path-driven, and the clean-rank arm is the one nobody
deploys.** Read as spectral SHAPES, not scalars — `gap_at_corrected` is taken at each arm's own corrected index
(400 for first_adapted=1, 48 for first_adapted=12), and a ratio high up a short spectrum is not comparable to one
deep in a long one, so the scalar is abandoned for the comparison and must not be reintroduced as a convenient
summary; the full spectra are the comparable object. One depth-15 net, L=4, k=784, everything fixed but where
adaptation starts:

| first_adapted | frozen layers below | d_j | spectrum shape | clean rank? | **deployed?** |
|---|---|---|---|---|---|
| 1 | 0 | [784,784,784,687] | ~400 conditions within ~4 orders, then a cliff to machine zero | **yes (~400)** | **NO — raw-input adaptation; §19: real LoRA adapts attention/MLP blocks, never the input layer** |
| 4 | 3 | [687,217,167,139] | smooth over 11+ orders, no cliff | no | yes |
| 8 | 7 | [104,84,73,61] | smooth, machine zero by ~104 | no | yes |
| 12 | 11 | [48,30,24,19] | smooth over 11+ orders, no cliff | no | yes |

The shallow arm having a genuine gap means the comparison had a live alternative and rejected it — this is not one
of the checks that could not fail. So on ONE network the gap survives shallow input and collapses as the frozen
path below it deepens and its rank profile contracts, nothing else varying: **frozen-path-driven, not route- or
training-driven** (the confounded shallow-trained net 355825 was not needed). The variable is the frozen path's
rank profile; depth is how you move it.

> **DEPLOYMENT HEADLINE: the configuration in which the certificate has a clean rank is the one nobody deploys.**
> The only arm with a genuine gap is adaptation on the raw input layer; everywhere adapters actually go
> (attention/MLP blocks, first_adapted ≥ 4) `d_j` contracts to a few dozen directions (19–48) with a spectrum
> smooth over 11+ orders — **no rank, only an effective rank at a stated tolerance**. It compounds with deployment
> precision: at fp16/bf16 roundoff the usable count is read far up that 11-order spectrum, a small fraction of an
> already tiny `d_j`. **MEASURED (job 365681, k=784, first=3, dead_jacobian=False):** T5.2 predicts **756**, the
> corrected law 417, but the usable certificate rank is **fp16 = 142** (~5.3× below T5.2) and **bf16 = 36** (~21×
> below) — against the fp64-tolerance count of ~402 (~1.9×). So at the precision adapters actually ship in the
> overprediction is 5–21×, not the fp64 ~2×, and this is measured directly, not the earlier ~3× extrapolation.
> (gap 1.58, no gap — the verdict holds on this run too.) **Two independent routes reach this same deployment story,
> neither built to test the other:** §19b by pixel count (~7% of the image at deployed rank) and this by the
> spectrum (no clean rank at deployed depth) — the convergence is the result.

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

## 6-CNN. CNN: conv rank law and T arm (job 355907) — 2026-09-18

*(The plan's "§6, CNN"; numbered 6-CNN here because §6 above already exists and is not edited.)*

`experiments/multilayer_cert/conv_encoder_ranklaw.py` (port of `real_encoder_ranklaw.py`, same row format / ladder / gap
diagnostic) on the **bottleneck CNN** `models/exact_inversion/mnist_conv_bottleneck.pth`: conv 1→64→128→**8**→256
(k=3, s=2, p=1, GELU; 28→14→7→4→2) → dense 1024→1000 (GELU) → head 10. Six adaptable modules, `r=256`, `N=8`,
`first ∈ {1,3,5}`, charts PCA `k ∈ {16,32,66,128,256,384,512}` + pixel 784, T arm `T ∈ {1,5,20,100,400}` (lr 0.01,
full batch, `B_0=0`, FP64). Job **355907** (long-gpu, hgn55 A40, 61 min, 654 rows; attested `git 377d841-dirty`,
`script_sha e4792c7e2149`); rows `results/multilayer_cert/conv_ranklaw_355907.jsonl`; smoke 355881. Truths = test
indices `[723, 923, 2619, 3739, 5981, 4186, 6644, 913]` (seed+7, the k-sweep's join key). **WP0 gate on the row:**
train 99.742% / CE 9.10e-3, test 98.63% (checkpoint values and re-measured at load agree; `fully_trained_gate: true`).
**THEORY test at the truth; no solve, no attack.** Numbers provisional until a second session reads the rows.

**Symbols (audit 2026-09-18).** `p_l` = patch dimension `C_in·9` (dense: input width); `P_l` = positions; `N'_l` =
`patch_span_rank`, the rank of the `N·P_l` base patch vectors at the truths; certificate rank `min(r,p_l) − N'_l`;
`d_j = rank M_j` (Jacobian of `vec(P_j)` w.r.t. the chart); `q_l = rank((C_l ⊗ I_{P_l}) J_l)`. The certificate is
applied at **every** position, so T5.2's per-layer budget under weight sharing is `q_l^{(i)} = min(rank C_l · P_l, d_l)`;
the audit's dense-style estimate was `q_l^{(ii)} = min(rank C_l, d_l)`. Both were pre-registered in the header
because they disagree on whether this spec discriminates: under (i) conv 2 alone (`24·49 = 1176 ≥ k`) pins any chart
and the laws coincide everywhere; under (ii) `first=1` gives corrected ≈ 291 vs T5.2 ≈ 459 at k=784.

**Pre-registered outcomes** (harness docstring, before any row): DISCRIMINATION / FALSIFIED / CONTROL (`first=5`,
dense+head, must saturate at the common value) / VACUOUS (no config has `corrected < t52`) / CONV-VACUOUS per module
(`N'_l ≥ min(r, p_l)`); T arm: conv modules — does `N'_l(T)` move at all; dense modules — `N·T` growth vs plateau;
dense-first `N' = N` invariant; no gap → ladder and spectrum, never an integer.

### Measured patch spans and zero-drift certificates (r = 256)

| module | kind | `p_l` | `P_l` | `N·P_l` | width bound on `d_j` | `N'_l` | rank `C_l` = `min(r,p_l)−N'` (measured = formula) | conditions/image `rank C·P_l` | residual at truth |
|---|---|---|---|---|---|---|---|---|---|
| 1 | conv | 9 | 196 | 1568 | 784 | 9 | **0** (conv-vacuous: `N' = p_l`) | 0 | 0 |
| 2 | conv | 576 | 49 | 392 | 784 | 232 | 24 | 1176 | 1.5e-15 |
| 3 | conv | 1152 | 16 | 128 | 784 | 117 | 139 | 2224 | 1.5e-15 |
| 4 | conv | 72 | 4 | 32 | **128** | 32 | 40 | 160 | 6.4e-16 |
| 5 | dense | 1024 | 1 | 8 | 128 | 8 | 248 | 248 | 5.9e-16 |
| 6 | head | 1000 | 1 | 8 | 128 | 8 | 248 | 248 | 7.3e-16 |

Only conv 1 is conv-vacuous; every other module carries a certificate that annihilates the base patches to ~1e-15.
The dense invariant `N' = N` holds at both dense modules at zero drift.

### `q_l` against both formulas — the fork is settled by the rows

`d_j` measured = `[k, k, k, 128, 128, 128]` for every `k ≥ 128` (per-image 126–128 at modules 4–6): the 8-channel
bottleneck contracts exactly to its width, upstream modules are rank-preserving. `q_l` measured at `first=1`, `L=6`:

| k | `q_l` measured | `q_l^{(i)}` weight-sharing | `q_l^{(ii)}` dense-style | matches |
|---|---|---|---|---|
| 128 | `[0,128,128,96,128,128]` | `[0,128,128,128,128,128]` | `[0,24,128,40,128,128]` | (i) except conv 4 |
| 256 | `[0,256,256,96,128,128]` | `[0,256,256,128,128,128]` | `[0,24,139,40,128,128]` | (i) except conv 4 |
| 384 | `[0,384,384,96,128,128]` | `[0,384,384,128,128,128]` | same | (i) except conv 4 |
| 512 | `[0,512,512,96,128,128]` | `[0,512,512,128,128,128]` | same | (i) except conv 4 |
| 784 | `[0,784,784,96,128,128]` | `[0,784,784,128,128,128]` | same | (i) except conv 4 |

**The weight-sharing count (i) is the real one**: conv 2's 24 certificate rows × 49 positions give `q_2 = k` up to the
pixel space, conv 3 likewise. The dense-style count (ii) is refuted at convs 2 and 3 by a factor `P_l`. **Exception:
conv 4 measures `q_4 = 96` at every `k ≥ 128`**, below formula (i)'s `min(40·4, 128) = 128` (and above (ii)'s 40);
`q_l_measured_matches_formula: false` on those rows. Not explained here (see NOT shown).

### The two laws: `corrected` vs T5.2 vs measured (zero drift)

| config | k | `k_1` | `Σq` | T5.2 `min(k_1,Σq)` | corrected `min_j(d_j+Σ_{l<j}q_l)` | measured ladder bf16/fp16/1e-4/…/1e-16 | rows | outcome |
|---|---|---|---|---|---|---|---|---|
| first=1, L=6 | 128 | 128 | 608 | 128 | 128 | 128 at all 12 rungs | 18176 | coincide |
| first=1, L=6 | 256 | 256 | 864 | 256 | 256 | 256 at all rungs | 18176 | coincide |
| first=1, L=6 | 384 | 384 | 1120 | 384 | 384 | 384 at all rungs | 18176 | coincide |
| first=1, L=6 | 512 | 512 | 1376 | 512 | 512 | 512 at all rungs | 18176 | coincide |
| first=1, L=6 | 784 | 784 | 1920 | 784 | 784 | 767 (bf16) / 784 at fp16 and below | 18176 | coincide |
| first=3, L=4 | 128 | 128 | 480 | 128 | 128 | 128 at all rungs | 5632 | coincide |
| first=3, L=4 | 256 | 256 | 608 | 256 | 256 | 256 at all rungs | 5632 | coincide |
| first=3, L=4 | 384 | 384 | 736 | 384 | 384 | 384 at all rungs | 5632 | coincide |
| first=3, L=4 | 512 | 512 | 864 | 512 | 512 | 508 (bf16) / 512 below | 5632 | coincide |
| first=3, L=4 | 784 | 784 | 1136 | 784 | 784 | 739 (bf16) / 784 below | 5632 | coincide |
| **first=5, L=2 (control)** | 128 | 128 | 256 | 128 | 128 | 83/106/115/123/126/127/128×6 | 512 | at common value |
| first=5, L=2 (control) | 256 | 128 | 256 | 128 | 128 | 94/117/123/126/127/128×6/**199** | 512 | at common value, gap **5.7e7** |
| first=5, L=2 (control) | 384 | 128 | 256 | 128 | 128 | 100/121/125/127/127/128×6/228 | 512 | gap 6.9e7 |
| first=5, L=2 (control) | 512 | 128 | 256 | 128 | 128 | 104/124/126/127/127/128×6/273 | 512 | gap 7.5e7 |
| first=5, L=2 (control) | 784 | 128 | 256 | 128 | 128 | 114/125/126/127/127/128×6/321 | 512 | gap 7.1e7 |

Per-depth at `first=1`, k=784: `L=1` → 0 (conv 1 vacuous, 0 rows), `L=2` → **784** (conv 2 alone), `L=3..6` → 784.
Small charts `k ∈ {16,32,66}`: every config saturates at `k` at every rung (`q_l = k` at each non-vacuous module).
Chart errors (MNIST, standalone): 0.421 / 0.332 / 0.236 / 0.170 / 0.090 / 0.038 / 0.0030 / 0 for k = 16…784.

**Verdict: VACUOUS — the pre-registered outcome that arithmetic (i) predicted.** `discriminates: false` on every one
of the 3 configs × 8 charts × 6 depths; `corrected == t52` because the first non-vacuous conv module already
delivers `q = k`. The **control behaves**: `first=5` sits at `k_1 = 128` from 1e-8 through 1e-15 with
`gap_at_corrected` 5.7e7–7.5e7 (`real_rank_at_corrected: true`) — a *real* rank, the 1e-16 rung crossing toward the
ambient (`ladder_converged: false` is that crossing, not an unconverged elbow). The conv stacks have `gap: null`
because `corrected = k` = the full column count (nothing to gap against) and `ladder_converged: true`.

### T arm — drifted certificates `C_l = P_{row(B_{l,T})^⊥} A_{l,T}` on the BASE patches

Loss 9.8e-2 → 1.7e-5…3.6e-5 (first=1/3) and → 4.3e-4 (first=5) at T=400, batch accuracy 1.0, no divergence.
`N'_l(T) = rank B_{l,T}` (1e-12 rel.), rank `C_l`, median residual `‖C_l h‖/‖A_{l,T} h‖` on base patches:

| config | module | zero-drift `N'` | T=1 | T=5 | T=20 | T=100 | T=400 | note |
|---|---|---|---|---|---|---|---|---|
| first=1 | 1 conv | 9 | 9 / C=0 / 0 | 9 / 0 / 0 | 9 / 0 / 0 | 9 / 0 / 0 | 9 / 0 / 0 | first adapted: invariant (`|C_T−C_zd|/|A_0|` 6e-14) |
| first=1 | 2 conv | 232 | **128** / 128 / 1.9e-5 | 128 / 128 / 1.5e-3 | 128 / 128 / 1.6e-3 | 128 / 128 / 1.6e-3 | 128 / 128 / 2.8e-3 | `rank B_T` = `C_out` = 128 **width-capped**; not a certificate |
| first=1 | 3 conv | 117 | **8** / 248 / 0.87 | 8 / 248 / 0.87 | 8 / 248 / 0.86 | 8 / 248 / 0.87 | 8 / 248 / 0.86 | `C_out = 8` width cap; residual O(1) — dead |
| first=1 | 4 conv | 32 | 28 / 52 / 3.2e-9 | 53 / 45 / 3.2e-7 | 59 / 45 / 6.3e-7 | 69 / 44 / 4.4e-7 | **72** / 43 / 3.4e-10 | `N'` moves with T, saturates at `p_4 = 72` |
| first=1 | 5 dense | 8 | **7** / 249 / 6.1e-10 | 15 / 241 / 1.5e-8 | 18 / 238 / 1.9e-8 | 21 / 235 / 1.6e-8 | 26 / 230 / 2.5e-8 | grows, ≪ `N·T` (plateau-like) |
| first=1 | 6 head | 8 | 7 / 249 / 2.8e-11 | 9 / 247 / 5.7e-2 | 9 / 247 / 5.5e-2 | 9 / 247 / 4.1e-2 | 9 / 247 / 4.1e-2 | capped at `m−1 = 9`; drift cost 4–6e-2 |
| first=3 | 3 conv | 117 | 8 / 248 / 0.87 | 8 / 248 / 0.87 | 8 / 248 / 0.87 | 8 / 248 / 0.86 | 8 / 248 / 0.84 | first adapted AND width-capped: invariant in T, residual O(1) |
| first=3 | 4 conv | 32 | 28 / 53 / 3.6e-9 | 51 / 45 / 1.1e-7 | 53 / 45 / 3.4e-7 | 62 / 44 / 1.2e-6 | 68 / 44 / 1.6e-6 | |
| first=3 | 5 dense | 8 | 7 / 249 / 6.6e-10 | 15 / 241 / 1.2e-8 | 18 / 238 / 3.4e-7 | 20 / 236 / 4.5e-8 | 26 / 230 / 5.5e-9 | |
| first=3 | 6 head | 8 | 7 / 249 / 2.7e-11 | 9 / 247 / 2.4e-2 | 9 / 247 / 2.9e-2 | 9 / 247 / 2.6e-2 | 9 / 247 / 4.2e-2 | |
| first=5 | 5 dense | 8 | **7** / 249 / 6.8e-10 | 7 / 249 / 7.8e-10 | 7 / 249 / 1.0e-9 | **8** / 248 / 1.9e-13 | 8 / 248 / 4.1e-13 | first adapted (frozen input): see imprint below |
| first=5 | 6 head | 8 | 7 / 249 / 3.1e-11 | 8 / 248 / 1.0e-4 | 8 / 248 / 2.8e-4 | 9 / 247 / 5.0e-4 | 9 / 247 / 7.1e-4 | |

Stacked ladder with the drifted certificates (k=784): `first=1, L=6` → 784 at every T and every rung (`q_l =
[0,784,784,110→126,128,128]`); `first=5, L=2` → 128 at 1e-10 with gap 7–8e7 at every T. Same shape as zero drift.

**Imprint diagnostic (why the dense-first `N'` is 7, not 8, at T ≤ 20).** `B_1 = −lr Σ_i δ_i (A_0 h_i)^T` with `δ_i ∝`
the softmax residual `p_i − y_i` of the frozen base. Per-image base residual relative to the max:
`[1.2e-4, 1.4e-13, 4.6e-7, 3.2e-6, 1.0, 2.4e-8, 4.5e-4, 1.6e-5]` (base batch loss 9.8e-2, 7/8 correct) — image 2
(test index 923) imprints at 1.4e-13, below the certificate's 1e-12 cut, so **7 images are recorded**
(`n_images_imprinting_above_1e12: 7`, `dense_n_prime_matches_imprint_count: true` at T=1). At T=100 the eighth
imprint has grown past the cut: `N' = 8`, residual 1.9e-13, `|C_T − C_zd|/|A_0| = 3.8e-7`, `first_layer_invariant_ok:
true`. The dense-first invariant therefore holds in the form *N' = number of images whose imprint clears the tolerance*,
which equals `N` once every image imprints — a weakly-recorded image, not drift and not a harness fault (a conv first
layer's invariant value is `min(N'_zd, r, C_out)`: 9 at conv 1, 8 at conv 3, both held at every T).

**NOT shown.** No attack, no solve, no start — algebraic rows at the truth. The law rows are ZERO DRIFT (the T arm
reads drifted certificates on base patches; it does not test T5.4). Single seed (1), single batch (8 images, 3 classes
repeated), single trained net; `r=256` only; `N'` counted at one tolerance (1e-12; the 1e-10 count is on the row and
is lower at the drifting modules, e.g. dense 21 vs 26 at T=400). Conv 4's `q_4 = 96 < 128` is **unexplained** (the
padded 2×2 patch geometry is a candidate, not checked). The corrected law is not *tested* here in the sense of §5b —
no config separates it from T5.2 — so this section neither confirms nor refutes it; it locates why a CNN offers no
discriminating regime at this r. Residuals for width-capped modules are reported but those modules have no
certificate in the ledger's sense. The T-arm residuals at deeper modules are medians over `N·P_l` patch columns; the
weakly-imprinted image's own residual is in `cert_residual_max`, not quoted here.

**Plain reading.** On a CNN the certificate of an adapted conv layer is applied at every spatial position, so a
24-row certificate at conv 2 becomes 1176 conditions per image and by itself pins the entire chart up to the pixel
space (`q_2 = k` for all `k ≤ 784`); at zero drift the stacked rank is `k` from the first non-vacuous conv on, depth
adds nothing, and the two depth laws coincide in every configuration (the dense-only control saturates at the
bottleneck width 128 with a real gap, so the harness would have shown a difference had there been one). Under
training the picture inverts: `rank B_{l,T}` is capped by the layer's output width, so the 128-channel conv records
128 of its 232 patch directions and the 8-channel bottleneck records 8 of 117 — its drifted "certificate" has
residual ~0.87 at the truth and is dead — while the dense modules record 7→26 images-worth of directions over
T = 1→400 (far below `N·T`) and the first adapted layer's count never moves.

## P1-CNN. Rank r for the whole network — the two depth laws discriminated on a real CNN (job 366149) — 2026-09-18

*Plan: `notes/plan_2026-09-18_multilayer_parameter_program.md` P1 (audit items 5, 6, 9). Harness
`conv_encoder_ranklaw.py` with the program flags (commit 12c5927), bottleneck CNN `mnist_conv_bottleneck.pth` (gate PASS),
`r ∈ {8, 16, 32, 64, 128, 256}` at every module, `first ∈ {1, 3, 5}`, prefix stacks, `k ∈ {32, 128, 384, 784}`, **three
seeds** (different eight truths per seed), zero drift, no T arm. 864 RANKLAW rows, 314 s wall (short-gpu, A40). Rows
`results/multilayer_cert/ranklaw_p1_cnn_366149.jsonl`. Single read; provisional until a second session reads the rows.*

**Live-module set per r (pre-registered from `N'_l = [9, 232, 117, 32, 8, 8]`, confirmed by the CONVLAYER rows).**
Certificate rank per module `min(r, p_l) − N'_l`, seed 1:

| r | conv 1 | conv 2 | conv 3 | conv 4 | dense | head |
|---|---|---|---|---|---|---|
| 8 | 0 | 0 | 0 | 0 | 0 | 0 |
| 16 | 0 | 0 | 0 | 0 | 8 | 8 |
| 32 | 0 | 0 | 0 | 0 | 24 | 24 |
| 64 | 0 | 0 | 0 | 32 | 56 | 56 |
| 128 | 0 | 0 | 11–15 (seed) | 40 | 120 | 120 |
| 256 | 0 | 24 | 139 | 40 | 248 | 248 |

r = 8 tests nothing (control, every stack `no_gap_vacuous` at rank 0). r ≤ 32 is dense + head only.

**Stacked rank at k = 784, `first = 1`, L = 6, three seeds** (`rank@1e-10` · gap at the corrected index · `cond_at_1e10`):

| r | corrected law | T5.2 | measured (s1 / s2 / s3) | gap | cond@1e-10 | outcome |
|---|---|---|---|---|---|---|
| 16 | 16 | 16 | 16 / 16 / 16 | real | 8–10 | coincide |
| 32 | 48 | 48 | 48 / 48 / 48 | real | 50 | coincide |
| **64** | **128 / 127 / 128** | 208 | **128 / 127 / 128** | 8.5e7 / 2.3e4 / 4.6e9 | 2.7e7 / 4.5e4 / 5.2e5 | **compare → corrected, 3/3** |
| **128** | **304 / 335 / 368** | 512 / 544 / 576 | **304 / 335 / 368** | 1.2e8 / 2.0e4 / 5.3e9 | 3.8e7 / 7.6e4 / 9.2e5 | **compare → corrected, 3/3** |
| 256 | 784 | 784 | 784 / 784 / 784 | n/a (= k) | 4e2–6e2 | coincide (conv 2 pins, §6-CNN) |

`first = 3` (L = 4) gives the same numbers at every r; `first = 5` (dense + head) saturates at the bottleneck width
128 from r = 128 on, with a real gap (its r = 64 cell reads 112 = 2·56, both laws, gap real, cond 3e3–7e4).

**Reading.** (1) At r = 64 and r = 128 the first non-vacuous module is conv 4 (r = 64) or conv 3 (r = 128), whose few
certificate rows (32; 11–15) times few positions (4; 16) do NOT pin the chart, so the nesting ceiling binds and the two
laws separate by 80–210 ranks. **The measured rank equals the corrected law's integer in 6 of 6 cells, with a real gap
in every cell (2e4 to 5e9), and T5.2 over-predicts in 6 of 6.** The per-seed spread of the prediction (304 / 335 / 368)
comes from conv 3's patch-span rank at each seed's eight images (rank C_3 = 11 / 13 / 15 → q_3 = 176 / 208 / 240),
and the measurement follows it seed by seed — this is the discriminating regime the r = 256 run (§6-CNN, VACUOUS)
did not have. It is the first exact-integer confirmation of the corrected law on a real network with a gap; the d15
MLP confirmation (§5b/5c) was an effective-rank statement without one. (2) **Conditioning is non-monotone in r:**
10 → 50 (dense-only pinning) → 3e7–5e5 (bottleneck-side conv pinning at r = 64/128) → 5e2 (conv 2 pinning at
r = 256). The badly conditioned regime is exactly the discriminating one: a chart pinned by a few certificate rows
through the 8-channel bottleneck. Seed-to-seed the condition number moves by three orders at fixed r
(2.7e7 / 4.5e4 / 5.2e5), so a single-seed conditioning number is not a property of the architecture.
(3) `cond_at_1e10` at r = 64, k = 128 reads 2e8 with `no_gap_vacuous` because corrected = k there (nothing to gap
against) — the same "= k" caveat as §6-CNN.

**NOT shown.** Zero drift only (the T arm was not run at these r; §6-CNN's T arm is r = 256). One trained net, one
class composition (three classes repeated). k = 32/128/384 cells coincide by construction (`q ≥ k` from the first
live module) and are not listed. No solve, no attack. The corrected law's *value* is confirmed here as an integer;
its under-prediction on the d15 MLP (§5b, ~6%) is a different net and a no-gap regime, not contradicted.

## P2(i). Plain deep conv without the bottleneck (job 366150) — VACUOUS for the law test, as pre-registered; conditioning stays low — 2026-09-18

*`conv_encoder_ranklaw.py --spec deep --ckpt models/exact_inversion/mnist_conv_deep_full.pth` (gate PASS twin, job
355840), r = 256 at every module, `first ∈ {1, 3}`, patterns `prefix` and `alternate`, k ∈ {16 … 784}, seed 1, zero
drift, no T arm. 104 RANKLAW rows, 374 s wall (short-gpu). Rows `results/multilayer_cert/ranklaw_p2i_cnn_366150.jsonl`.
Single read, provisional.*

Modules `(p_l, P_l, N'_l)`: conv 1 (9, 196, **9** — conv-vacuous), conv 2 (576, 49, 232), conv 3 (1152, 16, 117),
conv 4 (2304, 4, 32), head (1024, 1, 8). Certificate ranks 0 / 24 / 139 / 224 / 248, residual at the truth ≤ 1e-15.

| stack | k = 66 | k = 384 | k = 784 |
|---|---|---|---|
| prefix [1], conv 1 only | 0 (vacuous) | 0 | 0 |
| prefix [1, 2] | 66, cond 4 | 384, cond 2e1 | 784, **cond 1e6** |
| prefix [1 … 5] | 66, cond 1e1 | 384, cond 2e2 | 784, cond 5e2 |
| alternate [1, 3] | 66, cond 5 | 384, cond 4e1 | 784, cond 9e1 |
| alternate [1, 3, 5] | 66, cond 2e1 | 384, cond 2e2 | 784, cond 6e2 |
| first = 3, any pattern | 66 | 384 | 784, cond 9e1–6e2 |

Every cell: `corrected == t52 == k`, `rank@1e-10 == rank@1e-16 == k`, `rank_test_outcome = no_gap_vacuous` (nothing to
gap against when the prediction equals the column count). **Reading.** Without the 8-channel bottleneck there is no
contracting `d_j`, so the first live conv (conv 2: 24 rows × 49 positions = 1176 conditions per image; conv 3: 139 × 16)
pins the chart up to the pixel space at any width, and the two depth laws never separate — the discriminating regime
of P1-CNN needed the bottleneck's `d_j = 128` AND a small live-module row count (r = 64/128). Layer choice (prefix vs
alternate) changes nothing in rank and little in conditioning. The one conditioning feature: conv 2 alone pinning 784
columns is at 1e6 (1176 conditions barely covering 784 unknowns through one layer); any additional layer brings it to
the hundreds. Compared with the bottleneck CNN at r = 256 (P1-CNN: cond 4e2–6e2 at k = 784) and the d15 MLP (1e10–1e12
at eight stacked layers), **conv stacks are the best-conditioned certificate Jacobians measured so far**.

**NOT shown.** Single seed; r = 256 only (the r-ladder that made P1-CNN discriminating was not run on this net — at
r ∈ {64, 128} conv 2 and conv 3 are vacuous (N' = 232, 117) and conv 4 (N' = 32) would be the first live module: that
is the cell to run if a discriminating regime on this net is wanted); no T arm; no solve.

## P2(ii). ResNet-18 on CIFAR-10, LoRA on the stage-3 convs (jobs 366181 train, 366250 smoke) — every conv is r-side VACUOUS at N = 8 for r ≤ 512 — 2026-09-18

*Plan P2 + audit item 7. `experiments/exact_inversion/train_resnet_backbone.py` (torchvision resnet18, 3×3 stem, no
maxpool, BN eval = affine fold checked to 5e-15), gate **PASS** (train 99.806% / CE 8.04e-3 / test 93.81%, epoch 98,
FP64 re-measure in `results/base_training_gate.jsonl`). `experiments/multilayer_cert/resnet_ranklaw.py` (port of the
conv harness, imports only), rows `results/multilayer_cert/resnet_ranklaw_smoke_366250.jsonl`, design in
`RESNET_NOTES.md`. Stage = torchvision `layer3` (input 128 ch at 8×8: `P_l = 64` positions; `p_l` = 1152 for the
stride-2 conv, 2304 for the other three). Smoke: r ∈ {64, 512, 1024}, k ∈ {32, 128}, L ≤ 2, seed 1. Single read.*

| conv | `p_l` | `P_l` | `N·P_l` | `N'_l` (measured) | rank C at r = 64 / 512 / 1024 | residual at truth |
|---|---|---|---|---|---|---|
| layer3.0.conv1 (s2) | 1152 | 64 | 512 | **512** | 0 / 0 / 512 | 1e-15 … 2e-15 |
| layer3.0.conv2 | 2304 | 64 | 512 | **512** | 0 / 0 / 512 | same |
| layer3.1.conv1 | 2304 | 64 | 512 | **512** | 0 / 0 / 512 | same |
| layer3.1.conv2 | 2304 | 64 | 512 | **512** | 0 / 0 / 512 | same |

**The 512 base patch vectors (8 images × 64 positions) are linearly independent at every conv**, so `N'_l = N·P_l`
exactly and the certificate rank is `min(r, p_l) − 512`: **zero for every r ≤ 512**, i.e. at every deployed rank.
This is the vacuity the audit predicted for ViT tokens (item 7), arriving on the ResNet from the r side rather than
the `p_l` side (the bottleneck CNN's conv 1 died by `N' = p_l = 9`). Live only at r = 1024 (rank C = 512 per conv,
32 768 conditions per image), where one conv pins any chart: stacked rank = k at every rung for k ∈ {32, 128},
`cond_at_1e10` = 10 (k = 32) / 21–23 (k = 128), `no_gap_vacuous` (= k, the CNN picture).

**Reading for deployment.** On a conv stage with `P_l` positions per image, the certificate exists only while
`r > N·P_l`: 512 here at N = 8, i.e. above every LoRA rank anyone ships. The conv harness's "one conv pins the chart"
result is therefore a large-r statement; at deployed r the conv certificate is empty on a ResNet stage, and the head
(or a dense layer, `P_l = 1`) is where the certificate lives. Whether `N·P_l < r` can be restored by fewer images or
a coarser stage (`layer4`: 4×4 = 16 positions → `N·P = 128` at N = 8, live from r = 129) is the next cell.

**Full stage (job submitted 2026-09-18, A100):** r ∈ {64, 256, 1024} at N = 8 (64/256 as vacuous controls) and
r ∈ {256, 512, 1024} at N = 4 (`N' = 256`, live from r = 257), k up to 3072, L ≤ 4. **NOT shown:** ReLU is the
activation here (P5 axis, not a confound for exactness); single seed; no T arm; no solve; `layer4` not run.

## P1-MLP. Rank r for the whole network on the 15-layer MNIST MLP twin (job 366148) — additivity exact with a gap up to the FP64 wall; wherever the laws separate there is no gap and the effective rank sits 1–6% under the corrected law — 2026-09-18

*Plan P1 (audit items 5, 6, 10). `real_encoder_ranklaw.py` (commit 12c5927) on **`mnist_mlp_d15w1000_full`** (the
gate-PASS twin; 365681/355531 used the failing original), `r ∈ {8, 16, 32, 64, 108, 256}` at every adapted layer,
`first ∈ {1, 3}`, prefix stacks L = 1…8, `k ∈ {32, 128, 384, 784}`, three seeds, zero drift. 1152 RANKLAW rows,
56 min wall (long-gpu A40). Rows `results/multilayer_cert/ranklaw_p1_mlp_366148.jsonl`. Single read, provisional.*

**r = 8 rows are DEAD and must not be read.** With N = 8 the certificate rank is r − N' = 0, but the harness reports
`cert_rank = 8` and a flat ladder at 8·L: `sigma_max_median` = 1.4e-15 (vs 1.07 at r = 16) — an annihilated
certificate called full rank by the relative-only `matrix_rank(rtol=1e-10)` at `real_encoder_ranklaw.py:307` and by
the stacked ladder's `numrank` without `ref`, exactly LESSONS 2026-09-07 ground rule 3; the 1e-25 absolute
dead-Jacobian guard is 10 orders too low to catch it. Fix (absolute floor tied to ‖A_0‖, guard relative to
σ₁(A_0 J_φ V)) is queued behind the jobs still running against the file; the rows stay in the file with this note.
No other cell is affected (r − N' ≥ 8 everywhere else, σ_max ≈ 1).

**Additivity, k = 784, `first = 1` (`first = 3` identical in rank, 1–2 orders worse in cond):** stacked rank =
`L·(r − 8)`, capped at k and at the nesting ceiling, three seeds identical, gaps 1e10–1e15:

| r | L = 1 | L = 2 | L = 4 | L = 8 | notes |
|---|---|---|---|---|---|
| 16 | 8 · cond 1 | 16 · 2–3 | 32 · 40 | 64 · 4e4 | gap 1e15 → 5e10; `compare/both` at every L |
| 32 | 24 · 1 | 48 · 3 | 96 · 2e2 | 192 · 2e6 | gap 1e15 → 3e9; both laws |
| 64 | 56 · 2 | 112 · 7 | 224 · 1e3 | **435 / 418 / 435 vs corrected 448 / 441 / 444** · 7e9 | L = 8: gap 3e3 / 1 / 1 — the wall |
| 108 | 100 · 2 | 200 · 10 | 400 · 6e3 | **613 / 613 / 615 vs corrected 624 / 617 / 620** (T5.2 779) · 6e9 | no gap (1–2) |
| 256 | 248 · 4 | 496 · 1e2 | 784 · 4e4 | 784 · 3e5 | pinned at k from L = 4 |

`first = 3`, k = 784, L = 8: r = 64 → 324 / 306 / 326 vs corrected 336 / 329 / 332 (T5.2 448), no gap; **r = 108 →
414 / 414 / 416 vs corrected 424 / 417 / 420 (T5.2 713), gap 1–2** — the k* = 417 cell of 365681 reproduced on the
gate-PASS twin (original: 402 at 1e-10, gap 1.6); r = 256, L ≥ 4 → 688 / 678 / 672 vs corrected 720 / 713 / 716 (T5.2
784), no gap, cond 9e9.

**Readings.** (1) **Depth adds `r − N` per layer exactly, with a real gap, for as long as the conditioning stays
below ≈ 1e9–1e10**; every `compare` cell reads `both` (laws coincide) with the measured integer equal to the
prediction, 3/3 seeds. (2) **The conditioning grows geometrically in L** (≈ ×3 per layer at small r, ×10 at r ≥ 64)
and with r at fixed L, and it is the conditioning, not the rank prediction, that ends the measurable regime: every
cell at cond ≥ 6e9 is `no_gap_vacuous`, and those are precisely the cells where the two laws separate. (3) In that
no-gap regime the effective rank at 1e-10 sits **1–6% below the corrected law and far below T5.2** in every cell
(twin and original alike), never above the corrected law — consistent with §5b/5c and with the CNN's exact
confirmation (P1-CNN) where the bottleneck supplies a clean `d_j` and a gap. **The MLP never delivers a gapped cell
that separates the laws**: at r = 64, L = 8, `first = 1`, seed 1 the gap is 3e3 (barely `compare`) and the verdict
is `neither` (435 vs 448 vs 448) — one marginal cell, not a refutation. (4) The gate-PASS twin changes nothing about
the original d15's picture (bridge cell above), so the earlier no-gap verdict was not a training artefact.

**NOT shown.** Zero drift; k = 32/128 cells (pinned by the first layer at r ≥ 108, or coinciding); a solve; the
truncated certificate; r = 8 (dead, above). The `first = 1` arm adapts the raw input layer, which nobody deploys
(§5c) — the deployable arm is `first = 3`.

## P4-CNN. Layer subsets on the bottleneck CNN (job 366147) — the stacked rank is set by the SHALLOWEST live conv in the set, whatever the pattern; conditioning is set by WHICH conv pins, not by shallowness — 2026-09-18

*Plan P4 (audit items 6, 8). `conv_encoder_ranklaw.py --layers prefix suffix:2 suffix:4 middle:2 middle:4 alternate
random:2:3 random:4:3 single:1 single:3 single:5 single:8` on `mnist_conv_bottleneck.pth`, r = 256, `first ∈ {1, 3, 5}`,
k ∈ {128, 384, 784}, three seeds, zero drift. 441 RANKLAW rows, 66 min wall. Rows
`results/multilayer_cert/ranklaw_p4_cnn_366147.jsonl`. Single read, provisional.*

k = 784, stacked rank at 1e-10 (three seeds; random draws listed by the set they drew) · `cond_at_1e10`:

| chosen modules (1–4 conv, 5 dense, 6 head) | rank | gap | cond | outcome |
|---|---|---|---|---|
| [1] (conv 1 alone; `single:1`, prefix L = 1) | 0 | – | – | conv-vacuous |
| [1, 2] (prefix L = 2: conv 2 pins) | 784 | = k | **9e5 / 5e5 / 1e9** | no_gap_vacuous |
| any set containing conv 3 and no conv 2: [3], [1, 3], [3, 4], [3, 5], [3, 6], [1, 3, 5], [3, 4, 5, 6], [2, 3, 4, 5] … | 784 | = k | **1e2 – 9e2** | no_gap_vacuous |
| [2, 4, 5, 6] (conv 2 + bottleneck side) | 784 | = k | 1e4 | no_gap_vacuous |
| sets starting at conv 4 or later: [4, 5], [4, 6], [5], [5, 6] (`suffix:2`, `middle:2` at first = 3, `single:5`, all of first = 5) | **128 / 127 / 128** | real (`compare`) | 5e4 – 4e7 | compare |

**Readings.** (1) **Nesting confirmed by every pattern**: the rank is `min(k, d_j of the shallowest chosen module,
Σq)` — 784 when conv 2 or conv 3 is in the set, 128 (the bottleneck width) when the set starts at conv 4 or later, 0
for conv 1 alone; alternating, middle, suffix, random and prefix sets with the same shallowest live module give the
same rank in every seed. Adding deeper modules to a set never raises the rank above the shallowest module's `d_j`.
(2) **The pre-registered conditioning ordering is REFUTED**: a set that skips conv 2 in favour of conv 3 is *better*
conditioned by 3–7 orders ([1, 3] at 1e2 vs [1, 2] at 5e5–1e9), and every conv-3-containing set sits in the hundreds
regardless of what else is stacked. Conditioning is set by which module does the pinning — conv 3 (139 certificate
rows × 16 positions = 2224 conditions on 784 unknowns) conditions far better than conv 2 (24 × 49 = 1176, barely
covering) — not by how shallow the set starts. The bottleneck-side sets (rank 128, gap real) carry the worst
conditioning of the live cells (5e4–4e7, three orders across seeds, as in P1-CNN). (3) At k = 128 and 384 every set
with a live conv reads `rank = k`; the bottleneck-side sets read 128 at every k.

**NOT shown.** r = 256 only (P1-CNN's discriminating r = 64/128 regime was not crossed with patterns); zero drift;
no solve; MLP patterns are job 366146 (running at write time).

## P3 first cell — under drift the TRUNCATED certificate beats the full one: exactness is worth less than rank (job 366221, twin, target layer 2, seed 3) — 2026-09-18

*Plan P3 (audit items 1–3). `drift_cert.py`: the layer BELOW the target is adapted too, so the target's input moves.
`mnist_mlp_d15w1000_full`, target layer 2, lower-layer rank ∈ {4, 16, 64}, target rank 64, N = 8, k = 32 PCA,
on-chart privates, 200 LM starts, both certificates solved on the same starts. 60 rows (15 zero-drift controls +
45 drift cells). ONE cell of fifteen; provisional, single seed.*

**Zero-drift controls:** `rho_full` at the FP64 floor, rank C = 56, 8/8 images, every r_lower and T. Harness sound.

**The two failure modes separate exactly as pre-registered, and they hit different certificates.**

- **Full certificate — dies by RANK.** `rank C_full = r − rank B_T`, and `rank B_T` climbs with the training span
  until it saturates at r, where the certificate is empty. Measured: `N'` grows ≈ `N·T` until it saturates at the
  layer width; `rank B_T` follows it to 64; `rank C` falls 56 → 47 → 31 → 6 → 0. Recovery stops when `rank C < k`,
  not when the residual degrades. The decisive cell is `r_lower = 4, T = 20, lr = 0.01`: rank C = 31 ≈ k = 32, the
  solver reaches the certificate's exact zero (objective 1e-27, and the truth's own residual is 6e-6), yet the
  returned images sit 0.34 from the truth. **Exact zeros, wrong images — non-identifiability, not a search failure.**
- **Truncated certificate — dies by ERROR, and much later.** It keeps rank 56 at every drift and pays
  `rho_trunc ≈ K_l · eps_perp` with `K_l` = 0.006–0.031, essentially constant over three decades of drift and over
  all three lower ranks (the pre-registered single-curve form; `K_l` is ~3× smaller here than the synthetic 0.082).
  In the same cell it lands 132/200 starts, **7/8 images inside the exact bar and 8/8 identified top-1 by SSIM and
  by features**, against the full certificate's 0/8.

**Recovery tracks `rho_trunc` alone, across every (r_lower, T, lr) combination:** 8/8 images while `rho_trunc ≲ 1e-4`,
partial (3–7) at 1e-4 … 7e-4, nothing beyond ~1.5e-3. Through `K_l ≈ 0.015` that is a drift threshold of roughly
2–5 % orthogonal movement of the layer's input. **This is the predictive rule the plan asked for**, and it is stated
in a quantity the attacker can compute from the release plus the public model.

**Excitation confirmed where it holds.** One cell has `rank B_T = N' = 40` (no contamination) at 6 % drift, and there
`rho_full` = 2.9e-15 — machine zero, against ~1e-7 at neighbouring cells. Exactness at arbitrary drift when the
excitation hypothesis holds, in a real network. But that same cell recovers 0/8 with the full certificate (rank
C = 24 < k) and 3/8 with the truncated one: **exactness without rank is useless.**

**Reporting bug to fix (does not affect the numbers).** The per-image `verdict` is overwritten by the cell-level
`contaminated` flag, so a cell where 7 images landed inside the bar reports `{'contaminated': 8}` instead of
`recovered`/`alias`. The counts (`landed`, `images_found`, `err`) are correct; only the label is wrong. This merges
outcomes the ground rules require kept apart — fix before the remaining cells are read as verdicts.

**NOT shown.** One target layer, one seed, one chart width, one model; the other 14 cells are running. Privates are
on-chart (so the chart is not the limit here). No momentum/weight decay yet.
