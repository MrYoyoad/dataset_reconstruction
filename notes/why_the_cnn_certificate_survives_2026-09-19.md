# Why the CNN certificate is not saturated by its own equations (2026-09-19)

**Question (Yoad, 19 Sept):** the CNN cells recover with chart + certificate. Given a conv certificate acts at
every spatial position, why do the equations not saturate it?

**Answer: at the first conv layer they do — completely. From the second on, they cannot.** Two facts do the work,
both measured in job 355907 (`results/multilayer_cert/conv_ranklaw_355907.jsonl`, `part=CONVLAYER`/`PATCH_SPAN`),
net = `mnist_conv_bottleneck` (1→64→128→8→256, dense 1024→1000, head), `r=256`, `N=8`, FP64, zero drift.

| layer | kind | `p_l` patch dim | `P_l` positions | patch vectors `N·P_l` | measured span | `rank C` |
|---|---|---|---|---|---|---|
| 1 | conv | **9** | 196 | **1568** | **9** | **0 — VACUOUS** |
| 2 | conv | 576 | 49 | 392 | 232 | 24 |
| 3 | conv | 1152 | 16 | 128 | 117 | 139 |
| 4 | conv | 72 | 4 | 32 | 32 | 40 |
| 5 | dense | 1024 | 1 | 8 | 8 | 248 |
| 6 | head | 1000 | 1 | 8 | 8 | 248 |

## 1. At conv layer 1 the certificate is exactly zero

A 3×3 patch on one input channel lives in 9 dimensions. Eight images at 196 positions supply 1568 patch vectors,
and their span is **9 — the whole space** (`spans_input_dim: true`). So `rank C = min(r,p_l) − N' = 9 − 9 = 0`.

This is the saturation the question is about, and it is total. Adapting **only** the first conv layer gives
`sum_q = 0` and leaves the entire chart free — nullity 16 at k=16, 128 at k=128, 784 at pixel width. Nothing is
identified. The intuition is right; it just applies to exactly one layer.

## 2. From layer 2 the geometry reverses, so saturation becomes impossible

Going deeper, the patch dimension **grows** with the channel count (9 → 576 → 1152) while the number of positions
**shrinks** (196 → 49 → 16). The patch count `N·P_l` therefore falls (1568 → 392 → 128) as the space it must fill
grows. At layer 2, 392 vectors cannot span 576 dimensions — measured span 232, leaving `rank C = 24`.

The private data physically cannot saturate a deep conv layer: it does not have enough patches.

## 3. Why 24 surviving rows pin a 784-dimensional chart — weight sharing

A conv certificate row is a constraint **at every spatial position**. The per-layer budget is

    q_l = min( rank(C_l) · P_l , d_l )          [conv, weight-shared]
    q_l = min( rank(C_l)      , d_l )          [dense — the audit's estimate, REFUTED]

At layer 2 that is `min(24·49, 784) = min(1176, 784) = 784`, not 24. The dense-style count predicts 24 < 128 and
so "not identifiable at k=128"; the conv count predicts identifiable. **Measured: rank 128 = k, nullity 0.** The
factor `P_l` is the whole difference, and it is why the CNN beat the count we had. (Ledger Q3: "the dense-style
count is refuted by the factor `P_l`".)

Adding one conv layer flips the case in 3 above from dead to fully determined:

| first adapted | layers | k | stacked rows | rank | nullity |
|---|---|---|---|---|---|
| 1 | 1 (conv1 only) | 128 | 0 | 0 | **128 — nothing identified** |
| 1 | 2 (+conv2) | 128 | 12544 | 128 | **0 — fully identified** |
| 1 | 2 (+conv2) | 784 (pixels) | 12544 | 784 | **0** |

## 4. Where the equations genuinely do run out: no weight sharing

Adapting dense+head only (`first_adapted=5`, `P_l = 1`) removes the position multiplier. `rank C = 248` per layer
and nothing amplifies it: at k=256 nullity 57, k=384 nullity 156, k=512 nullity 239, k=784 nullity 559. This is
the regime where equation count is the binding constraint — and it is a *dense* regime, not a conv one.

## 5. What this does and does not say

- It is an **algebraic rank at the truth** — `claim_class` on every row is "theory rank-law test, no solve, no
  attack". Identifiability, not recovery.
- The recovery cells are separate and they work: `cifar_bottle`, CNN base, public PCA `k=16`, **cert found 8/8**,
  cert landed 183–186 of 200 at T = 20/100/400 (jobs 355946/355948/355950, `experiments/cifar/RESULT.md` §7).
  Breadth falls with chart width — 6/8 at k=32, 1/8 at k=48 — as the capacity line predicts.
- **Reading hazard in that table**: the columns are `cert found | cert landed | NTK found | NTK best resid |
  floor | NTK verdict`. The "2/8" and "search failure" on the CNN rows are the **NTK route's**, not the
  certificate's. Do not read that verdict as the certificate's.
- One net, one seed, one `r`, zero drift. Conv-4's `rank C = 40` against `p_l = 72` and span 32 is consistent;
  Q3 flags a separate unexplained `q = 96 < 128` at conv 4 in the stacked arm.
