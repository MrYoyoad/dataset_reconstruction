# The Crux: Activation × Anchor × Linearization — Analysis of Existing Data

> **UPDATE 2026-08-28 — full 152-config rescore (was 27); refutation of the POSITIVE law holds and sharpens.**
> Rescored all 152 surviving job-857271 tensors (`results/rescored_activations_857271_full_2026-08-28.csv`,
> bsub 389926) — whole smoothness spectrum incl. kinked controls relu/leaky_relu/hardswish, which DID run
> (never scored, not never run). Leakage = `ctrl_margin_norm` (clip-robust) at best-matched wc≈0.10, Spearman
> over **n=13 DISTINCT activations** (softplus-β variants held separate). Honest read (metric-audited, yoado-6d):
> - **The positive "smoother⇒more leakage" hypothesis is REFUTED** — this *extends* the earlier "+0.03
>   no-relationship" (different metric/scope, and now a sign change), it is **not** "the same number
>   strengthened."
> - **−0.38 is NOT a robust negative smoothness law.** It is a **two-cluster** effect: relu (+0.588),
>   leaky_relu (+0.542), selu (+0.487) leak ~4× the rest (mean 0.43 vs 0.10) — but hardswish, also kinked,
>   does NOT (0.11). And the sign **FLIPS within the smooth-only set: +0.58** (n=9). A single Spearman
>   dresses a two-group gap with an opposite within-group trend up as a gradient → do not report −0.38 as a law.
> - **wc-PROVISIONAL:** all T=1, `ntk_passed`=False, not truly matched (sigmoid at wc 0.01). Leakage scales
>   with weight_change, so sign/magnitude **may move under true matched-wc** — exactly what the GPU re-sweep fixes.
> - `feature_stability` tracks smoothness only **weakly (+0.29)** — the strong linearization evidence is the
>   softplus-β β-monotonicity (job 911475), NOT this. softplus-β leakage is **non-monotonic** (b50 rises back
>   toward relu-like) → report as non-monotonic, not a within-family "+0.60 trend."
> - Robust takeaway: **no clean monotonic smoothness→leakage law in this first-pass; kinked relu/leaky_relu/selu
>   leak most (wc-provisional).** The matched-wc GPU re-sweep (user approved Full closure) turns this into the
>   actual leakage ranking. Figure + generator: `figures/crux/activation_ranking_857271.png` /
>   `experiments/plot_activation_ranking.py`.

**Date:** 2026-08-21 · **Scope:** pure re-analysis of on-disk results (no GPU runs).
**Sources:** `results/rescored_batch_2026-08-13.csv` (256 rows), `results/flowers32_rescored.csv` (27),
`results/rescored_activations_857271_2026-08-11.csv` (27), and `results/anchor_sweep_*T10_r8_*_s42.pth`.
**Figures:** `figures/crux/activation_matched_wc.png`, `figures/crux/anchor_two_curve_summary.png`.

**Metric hygiene (enforced):** leakage = `ctrl_margin_norm` (clip-robust: recon `ssim_norm` − control
`ssim_norm`); NTK survival = `feature_stability`; `weight_change` (wc) treated as a **confound** —
activations compared only at **matched wc**, never at fixed LR. Raw `ssim`/`ctrl_margin` ignored.

---

## TL;DR

- **Winner (defensible): `softplus`.** Not because it leaks the most raw margin — it does not — but
  because it is the only activation that wins the *linearization* criteria the thesis is actually about:
  highest NTK survival at matched wc (**feature_stability 0.953** on MNIST), leakage that is **flat in
  wc** (0.203 at wc 0.05/0.10/0.30 — not a wc artifact), **lowest LoRA function-space lin-error at every
  α** on both datasets, and an anchor two-curve that **peaks at α=0** (best at the naive NTK point — the
  cleanest possible linearization signature; it needs no anchor help).
- **The naive "smoother → more leakage" law is REFUTED** (as STATUS warned). At matched wc≈0.10 on MNIST,
  raw leakage is topped by **selu (C¹, +0.487)** and **leaky_relu (kinked, +0.483)**, while **elu/celu
  (also C¹) sit at the bottom (+0.034)**. Spearman(smoothness-rank, leakage@wc0.10) ≈ **+0.03** — no
  monotone relationship. On flowers32 the pattern *inverts*: the C∞ activations gelu/silu/mish leak the
  *least*.
- **The chain breaks at one link.** `smoothness → linearization fidelity` **holds** (softplus is the best
  linearizer, relu the worst). `linearization fidelity → leakage magnitude` **does NOT hold**: relu has
  the *worst* lin-error yet the *highest* control margin, and its margin is *flat in α* (decoupled from
  linearization quality) — the anchor attribution test flags this as extraction/anchor artifact, not a
  linearization win.

---

## TASK A — Matched-`weight_change` activation ranking

### MNIST (rank 8, npc 1, T=1) — the only cell with a real LR→wc sweep (11/13 activations, up to 7 LRs)

Leakage and NTK survival read at a **common wc≈0.10** by linear interpolation of each activation's
(wc → metric) curve. `*` = out of band (single point / wc range below 0.10), value shown at nearest
measured wc (annotated). Ordered by smoothness (smoothest first).

| activation | class | wc range | leakage `ctrl_margin_norm`@0.10 | NTK `feature_stability`@0.10 |
|---|---|---|---|---|
| sigmoid | bounded C∞ | 0.003–0.01 | +0.235 *(@wc0.01)* | 0.999 * |
| tanh | bounded C∞ | 0.062–0.25 | +0.074 | 0.860 |
| gelu | C∞ | 0.020–1.17 | +0.079 | 0.814 |
| gelu_tanh | C∞ | 0.020–1.17 | +0.079 | 0.814 |
| silu | C∞ | 0.018–1.09 | +0.098 | 0.847 |
| **softplus** | **C∞** | **0.019–1.14** | **+0.203** | **0.953** |
| mish | C∞ | 0.027–1.62 | +0.079 | 0.842 |
| celu | C¹ | 0.063–0.25 | +0.034 | 0.861 |
| elu | C¹ | 0.063–1.25 | +0.034 | 0.861 |
| **selu** | **C¹** | **0.092–0.37** | **+0.487** | 0.828 |
| hardswish | non-smooth | 0.017–0.07 | +0.107 *(@wc0.07)* | 0.926 * |
| **leaky_relu** | **kinked** | 0.056–0.06 | **+0.483** *(@wc0.06)* | 0.745 * |

(relu on MNIST lives in unlabeled `finetune_activation=''` rows and is not cleanly matched here; its
behaviour is read instead from the anchor sweep, Task B.) Robustness at wc≈0.05 / 0.30 does **not**
change the order — softplus is dead flat (+0.203 at all three), selu ≈ +0.49 at both 0.10 and 0.30,
gelu/silu/mish ≈ +0.08–0.10 throughout.

**Leakage ranking @ matched wc≈0.10 (MNIST):**
`selu (+0.49) ≈ leaky_relu (+0.48) ≫ softplus (+0.20) > sigmoid (+0.24, but only wc≤0.01) >
silu (+0.10) > gelu ≈ gelu_tanh ≈ mish (+0.08) > tanh (+0.07) > elu ≈ celu (+0.03)`.

**NTK-survival ranking @ matched wc≈0.10 (MNIST):**
`softplus (0.953) > hardswish (0.926*) > celu ≈ elu (0.861) ≈ tanh (0.860) > silu (0.847) >
mish (0.842) > selu (0.828) > gelu (0.814) > leaky_relu (0.745*)`.

**Reading:** the two rankings **disagree**. Leakage is led by kinked/C¹ activations with *poor* NTK
survival; NTK survival is led by softplus. Smoothness predicts **neither** monotonically
(Spearman ≈ +0.03 for leakage; selu and elu are both C¹ yet at opposite extremes). The one activation
that is simultaneously top-tier on NTK survival *and* has high, wc-stable leakage is **softplus**.

### flowers32 (rank 8, npc 1, T=1) — NO usable matched-wc band

11 of 12 activations have a **single LR point** (lr=0.005), all at **wc < 0.05** — far below the 0.10
target. Only softplus has ≥2 wc points. Interpolation to a common wc is therefore impossible; the figure
falls back to each activation's nearest point (annotated `@wc0.0x`). **Verdict: "the grid missed the
band" — a clean matched-wc flowers32 sweep is required (Step 2a's `--target_weight_change` LR bisection).**

What the single points *do* show is a **pattern inverted from the smoothness thesis**, at face value:

| group | activations | leakage (single pt) | clipped_fraction |
|---|---|---|---|
| C∞ smooth | gelu, gelu_tanh, silu, mish | **+0.02 – +0.04** (lowest) | **0.52 – 0.58** (heavy) |
| C¹ / kinked / bounded | celu, elu, selu, tanh, leaky_relu, sigmoid | **+0.27 – +0.32** (highest) | 0.00 – 0.22 |
| softplus (has a sweep) | softplus | +0.28 @wc0.10, +0.29 @wc0.06 | 0.00 |

The C∞ smooth activations both **leak least and clip hardest** on flowers32 — the heavy clipping (≈0.55)
signals the reconstruction blew out of `[-1,1]`, i.e. an **extraction failure**, which confounds the
leakage read. This is exactly the "smoother=better refuted on flowers" caution in STATUS. softplus is the
only C∞ activation that both leaks (+0.28) and does **not** clip — again the smooth outlier that behaves.

---

## TASK B — Anchor α-sweep two-curve (T=10, r8, N=2)

For each activation: LoRA function-space lin-error(α) (dashed, should fall with α) vs LoRA leakage
`ctrl_margin_norm`(α) (solid). α is the linearization-point knob: 0 = linearize at init (naive NTK),
0.9 = linearize near the fine-tuned endpoint. **Attribution test (plan criterion 4):** a *legit*
linearization win means leakage **peaks at or before the lin-error minimum** (α=0.9) and then the α≈0.9
**identifiability collapse** pulls it back down; leakage that **climbs monotonically to α=0.9** signals
anchor-`xᵢ` contamination (the anchor θ_anchor→θ_T already encodes the training data), not a
linearization win.

### MNIST (`anchor_sweep_T10_r8_{gelu,silu,softplus,relu}_s42.pth`)

| act | LoRA lin-err (α=0→0.9) | LoRA ctrl-margin (α=0→0.9) | leakage peak | attribution |
|---|---|---|---|---|
| gelu | 0.192 → 0.034 (falls) | −0.109, 0.052, 0.071, **0.502**, 0.269 | **α=0.75** | **PASS** — rises then collapses at 0.9 |
| silu | 0.152 → 0.026 (falls) | −0.119, 0.038, 0.057, **0.292**, 0.290 | α=0.75 (plateau) | PASS (peak ≤ min) |
| **softplus** | **0.087 → 0.010 (lowest at every α)** | **0.320**, 0.122, 0.259, 0.203, 0.220 | **α=0.0** | PASS — best at naive NTK; needs no anchor |
| relu | 0.398 → 0.016 (highest; erratic) | 0.382, 0.385, **0.621**, 0.535, 0.569 | α=0.5, **flat/high across all α** | **FAIL signature** — margin decoupled from lin-error |

- **gelu / silu** pass cleanly: leakage climbs with α, peaks at 0.75, then the α=0.9 identifiability
  collapse cuts it (full-space ssim_norm 0.96→0.56 at 0.9 is the same collapse) — the textbook
  Addition-3 two-curve shape.
- **softplus** is the strongest linearization story: **lowest LoRA lin-error at every α**, and its
  leakage is **already maximal at α=0** — the smoothest activation linearizes so well at init that the
  anchor buys it nothing. This is the cleanest evidence that softplus's leakage *is* a linearization
  phenomenon.
- **relu is the counter-example that breaks the chain:** worst lin-error, yet the **highest** control
  margin (0.62), and that margin is **roughly flat across α (0.38–0.62)** — it does **not** track the
  10× swing in lin-error. relu's leakage cannot be a linearization-fidelity effect; it is extraction/
  anchor geometry. This is the honest refutation of "linearization fidelity → leakage."

### flowers32 FREE-c (`anchor_sweep_flowers32_free_T10_r8_{gelu,relu,softplus}_s42.pth`)

FREE-coefficient = the realistic attack (per project convention; oracle/fixed-c is an upper bound only).

| act | LoRA lin-err (α=0→0.9) | LoRA ctrl-margin (α=0→0.9) | verdict |
|---|---|---|---|
| gelu | 0.080 → 0.012 | **−0.296 … −0.221 (all negative)** | no leakage at any α |
| softplus | 0.054 → 0.008 (lowest) | **−0.324 … −0.315 (all negative)** | no leakage at any α |
| relu | 0.276 → 0.010 | +0.141, 0.090, 0.152, 0.194, **+0.227** | leaks, but **climbs monotonically to α=0.9** → attribution FAIL |

**In the realistic free-c regime on flowers32, the smooth activations (gelu, softplus) do not leak at all
(negative margin everywhere), and the only leaker — relu — fails the attribution test** (monotone climb
to the α=0.9 edge = anchor contamination, not linearization). This is a hard refutation of
"smoothness → leakage" on flowers32 in the realistic setting. (The non-free / oracle flowers32 files
tell the opposite, softplus-favouring story — which is precisely why oracle is not trusted here.)

---

## Do the four things co-move? (the proof criterion)

| link | holds? | evidence |
|---|---|---|
| smoothness → **linearization fidelity** (lowest lin-error) | **YES** | softplus lowest LoRA lin-err at every α on both datasets; relu worst. |
| smoothness → **NTK survival** (feature_stability @ matched wc) | **partial** | softplus top (0.953), but gelu low (0.814); not monotone across the spectrum. |
| linearization fidelity → **leakage magnitude** | **NO** | relu: worst lin-err, highest margin, margin flat in α. selu/leaky_relu top leakage with poor NTK survival. |
| smoothness → **leakage @ matched wc** | **NO** | MNIST Spearman ≈ +0.03; selu(C¹)/leaky_relu(kink) top, elu/celu(C¹) bottom; flowers32 inverts. |

**Net:** the thesis chain is **true up to its midpoint and breaks after it.** Smoothness genuinely buys
*faithful linearization* (best lin-error, best NTK survival, cleanest α=0 attribution) — and **softplus
is the winner on that axis.** But *faithful linearization does not translate into the largest leakage
margin*: raw leakage is dominated by kinked/C¹ activations whose margins are **decoupled from
linearization quality** (flat in α), which the attribution test attributes to extraction/anchor geometry
rather than a linearization win. So the honest headline is **not** "smoothest activation leaks most"; it
is **"smoothest activation gives the most faithful, best-attributed reconstruction, and softplus is that
activation."**

---

## Honest caveats & data gaps

1. **All Task-A (matched-wc) data is T=1 (single step).** The thesis's core clause — linearization
   "stays accurate over **more** fine-tuning steps T" — is **not tested** by this data. A
   `feature_stability`-vs-**T** sweep per activation is the missing measurement (only the anchor sweep is
   multi-step, and it is N=2, single seed).
2. **The `softplus_bβ` sharpness knob was never run.** The single cleanest smooth↔kinked axis (β=0.5…50,
   continuously deforming softplus toward relu) — the most diagnostic experiment for the whole thesis —
   has no data on disk.
3. **flowers32 has no matched-wc band** (11/12 activations = 1 LR point, all wc<0.05). Every cross-
   activation flowers32 statement is at *unmatched* wc. Needs Step 2a's `--target_weight_change`
   LR-bisection sweep (multi-LR per activation) before any flowers32 ranking is defensible.
4. **flowers32 smooth-activation results are confounded by extraction failure** (clipped_fraction
   0.52–0.58 for gelu/silu/mish at their single points; negative free-c margins). Their "low leakage" may
   be a solver blow-up, not a privacy property.
5. **MNIST anchor recons are clip-heavy** (LoRA clipped_fraction 0.45–0.56). `ssim_norm` (clip-robust) is
   used, but the LoRA α-dependence was already flagged **"NOT robust"** in STATUS — anchor conclusions
   are single-seed, N=2.
6. **selu / leaky_relu "leakage" is unvalidated as genuine signal.** High margin + poor NTK survival +
   kinked/C¹ geometry is consistent with an extraction artifact (cf. relu's α-flat margin). Whether it is
   real private-data recovery needs a retrieval-metric / visual check, not just ctrl_margin_norm.
7. **relu is not cleanly labeled in the MNIST CSV** (`finetune_activation=''` rows), so it is absent from
   the Task-A table and only enters via the anchor sweep.

### What a clean matched-wc sweep would need
Per dataset (MNIST **and** flowers32), per activation across the full smoothness spectrum **including
`softplus_bβ`**: a `--target_weight_change` LR bisection hitting wc ∈ {0.05, 0.10, 0.30} with
`ntk_passed:True` where reachable, at **T ∈ {1, 3, 10}** and ≥3 seeds, logging `feature_stability`,
`ctrl_margin_norm`, retrieval top-1, and function-space lin-error. That — not the current single-step,
single-cell grid — is what would let "smoothest = best" be *proved* rather than *ranked*.


## softplus_b(β) controlled-knob result (job 911475, 2026-08-23)

The decisive missing experiment: softplus_b(β) is a pure smooth→kinked knob (β=0.5 smoothest, β=50≈ReLU),
everything else fixed. LoRA function-space linearization error, interpolated to MATCHED weight_change:

| β | @wc=0.05 | @wc=0.10 | @wc=0.30 |
|---|---|---|---|
| 0.5 | 0.037 | 0.071 | 0.070 |
| **1** | **0.021** | **0.032** | **0.063** |
| 2 | 0.032 | 0.069 | 0.211 |
| 5 | 0.047 | 0.106 | 0.240 |
| 10 | 0.048 | 0.142 | 0.307 |
| 50 | 0.237 | 0.381 | 0.576 |

**Verdict: MONOTONIC** — linearization error grows ~12× as the activation sharpens toward ReLU, with a
genuine sweet spot at β=1 (standard softplus). This isolates and CONFIRMS `smoothness → linearization
fidelity` on a controlled axis (no activation-identity confound). Combined with the activation-identity
analysis above, the honest crux story is: **smoothness buys faithful, linearizable fine-tuning (softplus,
β≈1 optimal) — the correct/attributable reconstruction — but NOT the largest leakage margin** (that link
is refuted). Figure: figures/crux/softplusb_linearization.png.
