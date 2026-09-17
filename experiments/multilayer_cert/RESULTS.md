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
| measured rank (1e-12 rung) | 418 | 418 | 418 | 418 |

**The measured rank saturates at 418 = the corrected law's 417 (to +/-1), and stays 1.8x below T5.2's 756.** The
binding term is `d_3 + q_1 + q_2 = 217 + 200`: everything from the fourth adapted layer on is trapped inside the
217-dim row space of that layer's encoder, so layers 5-8 add nothing — the nesting ceiling, measured on a real
network. The rank-preserving control (first adapted layer at the pixel input, few layers, `d_j = 784`) saturates
at both laws' common value, confirming the harness can produce the non-discriminating outcome.

**At attack scale (`chart66` arm, k<=66) the two laws COINCIDE** (`d_j = 66 = k_1` everywhere, both give 66): the
rank-law distinction is invisible at a buildable chart width. So "a deep adapted stack is trapped at ~418, not the
naive 756" is a statement about the rank LAW; the attack-side re-costing at `k<=66` is a separate claim.

**PROVISIONAL rung (per the ladder discipline):** the exact elbow (418) resolves only at the 1e-12 rung; 1e-10
undercounts to 402/403. 354535 ran on a shared GPU, so every rung below ~1e-10 is provisional until re-run on the
exclusive A100 (job 354537). The **qualitative** result — corrected confirmed, T5.2 refuted by 1.8x — is robust
across every rung (even 1e-6 reads ~326, far below 756).

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
