# Next Experiment Plan — prioritized, honest, single source of "what to do next"

**Written:** 2026-08-11 · **Supersedes the Part A/B/C status in** [experiment_plan.md](experiment_plan.md)
(~3 months stale — still marks Addition 3 / DI-Phase 0 / GB-Phase 1 as ☐ proposed, all now ✅).
**Grounded in:** [STATUS.md](../STATUS.md) (2026-08-11 review), [LESSONS_LEARNED.md](../LESSONS_LEARNED.md)
(2026-07-22/23), [simudy_decision_brief.md](simudy_decision_brief.md) (B1/B2 gates),
`results/rescored_metrics_2026-07-22.csv`, and a full on-disk audit of jobs
857271 / 956997 / 957044 / 887704 / 956994 / 863020. **Execution order approved 2026-08-11**
(`.claude/plans/ou-are-a-research-clever-boole.md`).

**The crux (Gal's steer):** Addition 2 (smooth activations) and Addition 3 (anchor α-sweep) are **one
coupled experiment**, not two. Use the anchor α-sweep *as part of* finding the best activation, run it
in full for the winner, and back it with an **analytical argument for why the winner is provably
"best"** — tied to the *level of linearization* (NTK survival + anchor lin-error) and to the
*gradient-bridge attack*. A bare SSIM ranking is not the goal; a mechanism is. This is the top priority.

**Center of gravity:** *next-meeting-first* — the ≈0-GPU wins that unblock the crux come first, then the
crux itself, then the follow-ups and theory.

**Direction status (⚠ blocked-on-Gal):** the prioritization assumes the **SimuDy reframe** — SimuDy
(ICLR'25) owns full-FT direct inversion, so we re-center on **LoRA-adapter-only leakage +
identifiability/anchor-α theory**, SimuDy as the cited baseline. The reply proposing this was *drafted
but never confirmed sent*; Gal has not weighed in. **Until he does it is a working assumption** — the
full-FT/DI hedge stays; confirmation is under "Blocked / needs Gal."

---

## The two conventions every item obeys (non-negotiable)

1. **Never raw SSIM alone.** Always pair with `ssim_norm` + mean-baseline + `clipped_fraction` **and**
   the same-class **control margin** (`ssim − ctrl_ssim`) and/or **retrieval** top-1 vs 1/N. On
   MNIST-N=2 background dominates both the number and the baseline; control margin + retrieval are the
   only trustworthy leakage bars (they cancel the shared background/clipping).
2. **Check `weight_change` + `delta_w_effective_rank` first.** Near-zero weight change (or effective
   rank ≪ r) ⇒ the fine-tune barely happened ⇒ the number is meaningless. *Every* activation config in
   job 857271 is `ntk_passed:False` — see Step 2a's risk.

Legend: **[QW]** quick analysis win (little/no GPU) · **[EXP]** experiment · **[THEORY]** · **[DOC]**.
Compute in L40S GPU-hours (MLP LoRA config ≈ 5 min; GB decoder arm ≈ 1–2 h).

---

## Current state in one screen (verified on disk)

| Track | State | Headline (verified) |
|---|---|---|
| Addition 3 — anchor α-sweep | ✅ solid (full-FT) — **GELU only** | α*≈0.75 replicates: s44-T10 0.965, s42-T5 0.934, s42-T10 0.939, N=10 0.497. **LoRA α-dependence NOT robust.** Never run for any non-GELU activation. |
| Gradient Bridge (GB-Phase 1) | ✅ milestone — GELU only | 0.951 cosine on **realistic batch gradient** (m=8); 0.685 single; 0.794 two-sided. ⚠ **best-epoch, not converged** (m8 final 0.930; r8 final 0.56). |
| LoRA leakage (gate B1) | weak **yes** | Retrieval ~2.0–2.3× chance, pooled z=4.3, **p≈8.5e-6** (N=4..32×3 seeds); control margins +0.13–0.18. ⚠ **prose-only, no saved table.** |
| Direct inversion (DI) | ✅ but does not scale | SSIM 0.57 (N=4) → 0.27 (N=10) → 0.14–0.18 (N=20). Superposition wall. SimuDy published the full-FT version. |
| Metric audit | ✅ | Raw SSIM: 65/76 recons sit below `ds_mean`; the fix is the conventions above. |
| **Addition 2 (Gal's TOP ask)** | **⚠ OPEN** | Job 857271 died at 96 h RUNLIMIT; **21 configs on disk, never analyzed**; kinked controls + softplus-β knob + T-variation never ran; all `ntk_passed:False`. |
| Addition 1 (harder data) | ⚠ partial | MNIST breadth done (N-sweep). Fashion-MNIST/CIFAR/faces **not started**. |
| B2 gate (linearized vs unroll) | ❌ never run | Decision brief calls **B1+B2 "the whole bet."** Only a cheap MNIST proxy exists (DI-T1 0.57 ≈ anchor-NTK 0.50). |

**One-line takeaway:** "best activation," "how linear is the fine-tune," and "how decodable is the
gradient" have each only ever been measured **at GELU**, and never connected. **Connecting them is the
contribution.**

---

## The analytical spine (why the crux can *prove* "best", not just rank)

The mechanistic chain the experiment is built to demonstrate:

> **smoothness (C^∞ vs kinked) → Φ(θ;x) is a smoother function of θ → the first-order NTK/anchor
> linearization is more accurate and stays accurate over more fine-tuning steps T → the reconstruction
> (which matches ∇Φ(θ_anchor;x) to the observed Δw) is more faithful → higher retrieval / control
> margin. The *same* smoothness makes the LoRA→gradient map more decodable → the bridge recovers a
> higher-cosine gradient.**

**Proof criterion for "best" — all four co-moving, not one SSIM number:**
1. **NTK survival** — largest T with feature-stability above threshold (longest linear regime).
2. **Leakage** — peak retrieval / control-margin at *matched `weight_change`*.
3. **Anchor window** — lowest function-space lin-error at every α and the widest usable α window before
   the α≈0.9 identifiability collapse. **The anchor α *is* the "level of linearization" knob.**
4. **Attribution** — the gain is *explained by* falling lin-error (SSIM peaks at/before the lin-error
   minimum = legit linearization win, Addition 3's two-curve test), **not** anchor x_i-leakage (SSIM
   climbing past the lin-error minimum = red flag).

"Best" is proven when one activation wins 1–3, passes 4, **and** (Step 3) is the most bridge-decodable —
smoothness, linearization, reconstruction, and gradient-recoverability all move together. The
co-movement is the result; the single winning activation is the headline. If any axis diverges, report
it — the mechanism is falsifiable, which is the point.

---

## STEP 1 — enabling quick wins (≈0 GPU, do first)

### QW1 · Rescore job 857271 → first-pass activation ranking + matched-point diagnosis **[QW]**
- **Goal:** convert the 21 orphaned tensors into the first activation signal and decide whether the crux
  needs a GPU re-run (Step 2a) or can lean on existing data.
- **Method:** rescore with the full metric suite **plus `weight_change`**; group by activation; plot
  recon-quality (`ssim_norm`, control margin, retrieval) **vs `weight_change`, not LR**.
- **Reuse:** `experiments/recompute_metrics.py` (+ `retrieval_metric.py`), zero GPU. **Two ~5-line
  patches first:** (i) add a `weight_change` column (pull `d['lora_diagnostics']['weight_change']`,
  already saved); (ii) parse `finetune_activation` from the filename (not in the saved `config`).
- **Compute:** ~0 (CPU minutes). **Success:** a `ssim_norm`/margin/retrieval-vs-smoothness trend at
  comparable `weight_change`, or a clear "no usable matched point" verdict.
- **⚠ Risk (real):** low-LR arms have `weight_change`≈0.03–0.06 (barely trained), high-LR arms 1.0–3.75
  (far outside NTK), all `ntk_passed:False`. If no two activations overlap in a sane band, the honest
  output is "grid missed the band" → **Step 2a** is required (not optional).

### QW2 · Complete + persist the LoRA-vs-full retrieval story **[QW]**
- **Goal:** finish STATUS's #1 open item and turn the p≈8e-6 result (prose-only) into a durable artifact.
- **Method:** run `retrieval_metric.py --classifier` over the **956997 full-model tensors** + the 12
  LoRA files; emit a CSV + bar figure (retrieval top-1 vs 1/N, LoRA vs full, per N).
- **Reuse:** `experiments/retrieval_metric.py` (`--classifier`, `--glob`); **add `--out`/`savefig`** (it
  currently only `print`s — why no table exists).
- **Compute:** ~0. **Success:** saved LoRA-vs-full table+figure at N=4..32; statement of whether LoRA
  stays > chance at N=32. **Risk:** full-model margins at N=32 are tiny (+0.02–0.03) — a weak retrieval
  there is still a result (leakage fades with N).

### QW3 · Rescore the newest prose-only tensors **[QW]**
- **Goal:** N=10 anchor (957044) + DI large-N (887704) numbers live only in prose/logs — put them under
  the standard suite with control margins.
- **Reuse:** patched `recompute_metrics.py` (anchor `.pth` already store `x_ctrl`/`ds_mean`).
- **Compute:** ~0. **Success:** control-margin + `ssim_norm` columns for every newest tensor.

### QW4 · Make retrieval + control-margin the standard bar in code **[QW/DOC]**
- **Goal:** the conventions are stated but not enforced (`beats_baseline`/margin came from throwaway
  scripts not in the repo). Bake them in.
- **Method:** add `ctrl_margin = ssim − ctrl_ssim` + a retrieval column to `recompute_metrics.py`;
  regenerate one canonical `results/rescored_metrics_<date>.csv` across all live tracks.
- **Compute:** ~0. **Success:** one canonical CSV every doc/figure references (single source of truth).

### QW5 · Reconcile the stale docs **[DOC]**
- Fix: `experiment_plan.md` unticked-but-done checkboxes; LESSONS_LEARNED L102 "anchor *creates* LoRA
  leakage" (STATUS retracts as seed-42-specific); `simudy_decision_brief.md` B1 verdict (now weak *yes*
  via retrieval/control); STATUS `## Pending Tasks` (May-era jobs listed "running"). Compute 0.

---

## STEP 2 — THE CRUX: coupled activation → anchor → linearization study **[EXP, top priority]**

### 2a · Shortlist the best activation across the smoothness spectrum, at matched weight_change
- **Goal:** per-activation **NTK survival (feature-stability-vs-T)**, **leakage (retrieval/control-margin
  -vs-T)**, and **function-space lin-error** — the raw material for proof criteria 1–3.
- **Method:** add a **`--target_weight_change` LR-search** (bisect LR until `weight_change`≈0.1–0.3 with
  `ntk_passed:True`) — this replaces the failed fixed-LR grid. Sweep, ordered by smoothness:
  `sigmoid, tanh` (bounded C^∞) · `gelu, silu, softplus, mish, gelu_tanh` (C^∞) ·
  **`softplus_b{0.5,1,2,5,10,50}`** (continuous sharpness knob — the clean smooth↔kinked axis) ·
  `elu, celu, selu` (C¹/intermediate) · `hardswish` (non-smooth control) · `leaky_relu, relu` (kinked).
- **Reuse:** `run_experiment_b.py:119` (`make_activation`; `tanh`/`hardswish`/`softplus_b<β>` already
  exist — add `sigmoid`/`celu`/`selu`, ~2 lines each in `configs.py:89` + a class in
  `test_activations.py:27`); `ntk_verification.py:52` + `compute_function_space_lin_error`. Model on
  `scripts/run_gal_additions_sweep.sh` (Stage-0 guard, `--skip_if_exists`, `python -u`); **split into
  ≤24-config sub-jobs**, smooth-first, so a preemption still delivers.
- **Compute:** ~2–4 GPU-h. **Success:** feature-stability(T) + retrieval(T) + lin-error, ordered by
  smoothness, with the softplus-β continuous curve.
- **Risk:** bounded activations (sigmoid/tanh) also change the *extraction* model (`create_model` shares
  the activation, run_experiment_b.py:327) — may destabilize extraction; report separately if so. If
  even the LR search can't reach `ntk_passed:True` at usable `weight_change` for the smooth ones, *that
  is the finding* ("smooth activations resist the NTK regime at this scale").

### 2b · Anchor α-sweep two-curve for the winner — the headline AND part of the proof
- **Goal:** for the top activation (+ ≥1 comparator: a runner-up and a kinked control), the full α-sweep
  {0,0.25,0.5,0.75,0.9}: **lin-error(α) vs SSIM/retrieval(α)** on one x-axis. Winner should show the
  lowest lin-error at every α and a peak satisfying the attribution test (criterion 4). **This is where
  "best activation" and "level of linearization" finally meet** — Addition 3 has only ever run at GELU.
- **Reuse:** `run_anchor_sweep.py` (already accepts `--finetune_activation`, computes both curves +
  per-α grids); model on `scripts/run_anchor_multiseed_wexac.sh`. Partial α-sweep OK if compute tight.
- **Compute:** ~1–2 GPU-h. **Success:** the coupled two-curve for the winner + comparators; a stated
  usable-α window per activation. **Risk:** if the winner's anchor two-curve *fails* the attribution
  test (SSIM climbs past the lin-error min), the "win" is anchor leakage, not linearization — report it.

### 2c · The analytical write-up
- **Goal:** the deliverable Gal actually asked for — *why* the winner is best, tied to linearization
  level. State the proof criterion, show the four metrics co-moving with smoothness and with lin-error,
  name the winner with the mechanism (not just the number). Feeds Step 6 theory. **Compute:** 0.

### 2d · N × lr ablation for GELU + the winner (softplus) **[EXP — RUNNING, job 483935]**
- **Goal:** a clean, standalone ablation mapping the **(N, lr) → weight_change → reconstruction/leakage
  surface** for the two key activations, so we can state exactly how sample count and step size trade
  off against recovery for gelu (deployment-standard) and softplus (Step-1 winner) — not buried inside
  the spectrum sweep.
- **Method:** {gelu, softplus} × N∈{2,4,8,16} (n_per_class 1,2,4,8) × lr∈{0.005,0.01,0.02,0.05,0.1},
  T=1, r=8, seed 42 = 40 configs. `weight_change` saved per run → read the surface at matched
  weight_change post-hoc (rescore with `recompute_metrics.py`/`retrieval_metric.py`). Deliverables: two
  heatmaps (ssim_norm & control-margin over N×lr) per activation + a weight_change contour.
- **Reuse:** `run_experiment_b.py` (no new code); `scripts/run_N_lr_ablation_wexac.sh`.
- **Compute:** ~3.3 GPU-h. **Success:** a clear (N, lr) operating-point map — where each activation
  leaks most, and whether softplus's advantage is uniform across the grid or concentrated at low N/lr.
- **Risk:** at large N×lr, weight_change blows past the NTK band and clipping rises — that's part of the
  map, reported (not hidden); pair every cell with weight_change + clip fraction.

---

## STEP 3 — the bridge connection **[EXP]**

- **Goal:** is the best-for-reconstruction activation also the most gradient-*decodable*? If smoothness
  raises the decoded-gradient cosine the same way it raises NTK survival, one property unifies **both**
  thesis attacks (direct NTK inversion + the bridge) — a compact, strong analytical claim.
- **Method:** run the bridge decoder per activation (winner vs ≥1 kinked control) on the hidden layer;
  correlate decoded-gradient cosine with the activation's Step-2 NTK-survival / leakage ranking.
- **Reuse:** `gradient_bridge/generate_pairs.py` (`--activation` already exists), `train_decoder.py`;
  model on `scripts/run_gb_improve_wexac.sh`. **Report converged, not best-epoch** (0.951 → 0.930
  final; fix training stability first — early-stop on val / checkpoint-best).
- **Compute:** ~2–4 GPU-h. **Success:** a per-activation decoded-cosine ranking; a stated verdict on
  whether it matches the reconstruction ranking. **Risk:** rankings may diverge — that too is a result
  (smoothness helps linearization but not decoding, or vice versa), and it's honest.

---

## STEP 4 — Addition 1, harder data (SAME MLP cookbook) **[EXP — RUNNING]**

- **Goal:** move off MNIST-N=2 (where `ds_mean ≈ each image`) to harder data where leakage numbers
  speak for themselves — **on the same 784-MLP + same NTK/anchor cookbook** (θ₀ stays MNIST-pretrained
  → a realistic transfer/PEFT attack). Does the softplus>>…>elu ranking + the smoothness→linearization
  story **transfer** to harder image structure?
- **Built + running (no arch change — both drop into the 784-MLP):**
  - **Fashion-MNIST** (28×28×1 native) — job **482018** (spectrum, N=2 & N=10).
  - **Flowers102 as 28×28 grayscale** (real natural images, downsampled to the MNIST input) — job
    `run_activation_flowers_wexac.sh` (spectrum, N=2 & N=10). This is "the same cookbook for flowers"
    per the 2026-08-13 steer — NOT the ViT gradient-inversion track (that's a separate pipeline).
- **Reuse (done):** `experiments/data_utils.py` `_load_dataset` dispatcher + parity `_get_binary_label`
  (validated end-to-end; MNIST backward-compat intact); `--dataset {mnist,fashion,flowers}` in
  `run_experiment_b.py`. CIFAR/faces stay stretch (need `INPUT_DIM`/arch change, or the ViT track).
- **Compute:** ~1–2 GPU-h (FMNIST). **Success:** retrieval > chance + control margin > 0 where the
  mean-baseline is *not* ≈ the image. **Risk:** if leakage vanishes on harder data, that bounds the
  attack — a real, reportable result.

---

## STEP 5 — follow-ups (sequenced after the crux)

- **GB-Phase 2 — decoded gradient → image (biggest new code).** Build a small MNIST-MLP gradient
  inverter (the existing `invert_gradient` at `phase0_vit_inversion.py:284` is ViT/224px-hardwired and
  consumes nothing from `gb_decoder_*.pth`); reuse its cosine+TV+signed-Adam+restart machinery. Does the
  decoded gradient recover an image that beats control/retrieval? ~3–6 GPU-h + coding. *Risk:* 0.95 is
  necessary not sufficient for pixels — the honest thing this exists to find out.
- **Push the bridge.** Two-sided + multi-sample as default; rank×m×layer ablation; other layers;
  converged-not-best reporting. Reuse `gradient_bridge/`, `run_gb_improve_wexac.sh`. ~4–8 GPU-h.
- **Larger-N + a real `[0,1]` pixel box.** Keep `x + ds_mean ∈ [0,1]` (not the current soft `[-1,1]` on
  the centered var, which never sees `ds_mean`). Edit `ntk_extraction.py:122`, `direct_inversion.py:157`.
  ~2–4 GPU-h. *Fixes the metric, not DI's fundamental N-collapse.*
- **Multi-config anchor generalization.** α × seeds × T × **pinned hard digit-pairs**, gate on
  `weight_change`, avoid easy digits — settles whether the anchor helps LoRA leakage *in general*.
  **Needs a `--digits` flag** in `data_utils.get_finetuning_data` (pairs currently implicit in `--seed`)
  + `--skip_if_exists` in the anchor sweep. ~3–5 GPU-h.
- **B2 gate (linearized vs unroll).** Cheap MNIST proxy now (DI = unroll, anchor = linearized on one
  testbed; add peak-memory + wall-time). ResNet-18/CIFAR confirmation = stretch, needs Gal's sign-off.
  ~1–2 GPU-h (proxy) / tens (ResNet).

---

## STEP 6 — theory (parallel, 0 GPU) **[THEORY]**

Identifiability + anchor-α — the analytical backbone for Step 2c. When does θ_T/ΔW uniquely determine
{x_i}? Formalize the anchor tradeoff (lin-error ↓ vs x_i-contamination ↑); derive an analytic **α\***;
and — the new tie-in — **predict the observed activation ordering from the smoothness→linearization
mechanism.** Q-A (well-posedness of `R: ΔW → x̂`), Q-B (pretraining overlap: recover x_i or only the
residual). *Success:* one proposition with an empirically-matching prediction. Un-scoopable; feasible
even if every scale experiment stalls.

---

## What to show Gal next
1. **The crux (Steps 2a–2c):** "activation X is best, here's its anchor two-curve, and here's *why* — it
   survives the NTK regime longest / linearizes cleanest / leaks most, all co-moving." (His top ask,
   now mechanistic.)
2. **LoRA leakage quantified + reproducible** (Step 1) = gate B1 weak-yes.
3. **The bridge connection** (Step 3): smoothness helps *both* attacks — one mechanism, two axes.

## Blocked / needs a Gal decision
- **⚠ Direction confirmation** (blocks framing) — the SimuDy reframe was proposed but unconfirmed;
  confirm/send the reply or raise it at the meeting before committing E7b (ResNet-scale) and deep
  theory. Keep the full-FT/DI hedge until then. (Gmail/Calendar MCP not authorized here.)
- **CIFAR/faces + ResNet-18 B2** are real infra builds — get Gal's read on small-scale+theory vs
  ViT/SD-scale target first (the decision brief already argues the former).
- **GB-Phase 2 SDS/diffusion prior** — multi-day build; only if plain gradient inversion leaves noise.

---

## Top 3 to do first — and why
1. **QW1 — rescore 857271.** Unblocks the crux: gives the first-pass activation ranking and decides
   whether Step 2a's GPU re-run is needed. ~free. Highest value/effort in the plan.
2. **QW2 — complete + persist the retrieval story.** STATUS's own #1 open item; converts gate B1 (the
   thesis premise) from prose to a defensible artifact. Pure CPU.
3. **QW3+QW4+QW5 as one pass** — newest numbers under the standard bar, conventions enforced in code,
   stale docs reconciled — so nothing built on top (the crux, Steps 2–6) can regress to bare SSIM.

*Ordering rationale:* all three are ≈0 GPU, they harden the premise and unblock the crux, and they gate
the expensive work — so the (idle-since-2026-07-26) cluster only spins up once Step 2a's exact configs
are known necessary. The first GPU job to submit is **Step 2a** (matched-weight_change activation
spectrum), split into ≤24-config sub-jobs, `python -u`, Stage-0 guard, smooth-first.
