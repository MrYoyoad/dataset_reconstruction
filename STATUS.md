# Project Status

Last updated: **2026-08-23** (Phase J0 of the Jacobian-spectrum program implemented + submitted; prior update 2026-08-21 MineGrad re-plan)

---

## Jacobian-spectrum leakage program — Phase J1 COMPLETE: two decisive de-confounds (2026-08-23)

Job 983139 (T-sweep + seed-whitening / q_eff). **Two diagnostics turned the tentative J0 "N controls
identifiability" story into an honest, sharper result — and neither the deterministic eff_rank nor the
current q_eff is yet a clean privacy number.** Results: `results/jacobian_j1_*.pth`,
`figures/jacobian_spectrum/j1_*.png`.

**1. T-sweep (eff_rank vs T=5/20/50) — the N=4 "collapse" is largely UNDERFITTING, not structural.**
- N=2, k=8: eff_rank ≈ 15.9/16 flat across T (genuinely preserves all coords).
- N=4, k=4: eff_rank **climbs** 9.28 → 9.50 → 12.68 as T=5→20→50 (verdict: UNDERFITTING).
- ⟹ the J0 "N=4 → frac≈0.6" was mostly a T=5 training artifact. Deterministic eff_rank at fixed small
  T is NOT a structural identifiability number. (Vindicates the pre-registered caveat.)

**2. J1 whitening — the B0-init noise is ~ORTHOGONAL to the data-signal, so q_eff is not yet
measurable.** The key number is the new reliability diagnostic (yoado-29): **J-energy inside the
measured seed-noise subspace = 0.0–0.1% across ALL configs** (N∈{2,4}, k∈{4,8}, S∈{16,32,64}).
- The B0-init noise cloud is nearly isotropic (anisotropy 1.07–1.18) and ≈Gaussian (skew ≈0.1,
  excess-kurt ≈ −0.5), i.e. a well-behaved noise — it just lives in different adapter directions than
  the data-perturbation Jacobian.
- ⟹ the reported `q_eff` (e.g. 8/8, 16/16 at small ρ) is **dominated by the shrinkage floor ρμ, not
  measured noise** — exactly the ρ-artifact the diagnostic flags. `q_eff` is trustworthy only for the
  ~0% of J-energy the noise spans, so it is **not a valid privacy number in this setup.**
- **Leakage bracket:** known-init upper bound = raw eff_rank (N=2 full ~16/16; N=4 underfit ~9–20/32);
  unknown-init q_eff = not trustworthily measurable from B0 noise here.

**Why (mechanism — structural, from A₀=0).** LoRA inits A=0, B=random. At the first SGD step
`∂L/∂A = B₀ᵀ(∂L/∂W)` (data-dependent) but `∂L/∂B = (∂L/∂W)A₀ᵀ = 0`. So the DATA signal first enters the
A-block while the init NOISE lives in the B-block; they mix only as A≠0 grows, hence J (mostly A-block)
⊥ Σ_seed (mostly B-block).

**Mechanism test (job 983279): the orthogonality is ROBUST across T, not a small-T artifact.**
yoado-29 predicted the J-energy-in-noise-subspace fraction would grow with T (as A grows). It does NOT:
N=2 k8 = 0.1% at T=5/20/50; N=4 k8 = 0.0%/0.1%/0.1% at T=5/20/50 — flat through T=50, even though
`eff_rank` climbs with T (N=4 k8: 20.0→23.2→26.0, the underfitting signature). So the data-signal stays
~orthogonal to the init-noise subspace even at T=50. Prediction refuted (in a good way): the
separation is a durable structural property, which strengthens the conclusion below.

**POSITIVE finding (not just "whitening was inoperative"): random LoRA init provides ~zero privacy
protection.** If the init noise is orthogonal to the data signal, an attacker who does NOT know B₀ can
factor it out perfectly, so the unknown-init attacker ≈ the known-init attacker. ⟹ the OPERATIVE
leakage is the deterministic / known-init number (raw eff_rank), which is HIGH (full for N=2). "Randomize
the LoRA init" is therefore not a defense — a clean, quotable privacy statement.

**Next (promoted from footnote to necessary):** introduce a randomness source that lives in J's column
space — **minibatch SGD / data-order / augmentation noise** — so `Σ_seed` actually spans J and `q_eff`
becomes a real measurement. Then re-run the bracket. Also: use T large enough to converge (T-sweep
shows T=5 underfits N=4) and scale S≥4·Nk for headline configs.

---

## Jacobian-spectrum leakage program — Phase J0 COMPLETE (machinery validated; NOT yet a privacy result) (2026-08-23)

Job 982855 ran the full J0 sweep to completion. **The data-latent Jacobian machinery works and correctly
detects/handles rank deficiency.** AD validated exactly (toy FD 5.9e-10, jvp-vs-reverse 3.5e-18; real
MNIST single-module FD 3.9e-9). All tensors + per-config spectrum figures saved
(`results/jacobian_j0_*.pth`, `figures/jacobian_spectrum/j0_*.png`).

**IMPORTANT (do NOT over-read pre-whitening).** The deterministic `eff_rank(J)/Nk` below conflates
*magnitude* (how much a private direction moves the adapter) with *recoverability*, and is confounded by
(1) the LoRA-rank bottleneck (a rank-r module caps J's column space regardless of how many private
directions exist) and (2) T-underfitting (T=5 may just not have moved the adapter along some directions
yet — small σ = "not moved," not "cannot be recovered"). **The privacy-meaningful quantity is the
seed-whitened `q_eff` from J1** (dividing by seed noise via CRLB is what turns "small σ" into "provably
below the noise floor"). So the table below is a *machinery/sanity* readout, not leakage evidence.

**Pre-whitening spectrum readout (qr tangents, rank-8 single module, GELU, T=5, MNIST):**

| N | k | Nk | eff_rank | frac |
|---|---|----|---------|------|
| 2 | 4 | 8 | 7.66 | 0.96 |
| 2 | 8 | 16 | 15.83 | 0.99 |
| 2 | 16 | 32 | 31.82 | 0.99 |
| 4 | 4 | 16 | 9.45 | 0.59 |
| 4 | 8 | 32 | 19.62 | 0.61 |
| 4 | 16 | 64 | 39.14 | 0.61 |

- The N=2 (frac≈1.0) vs N=4 (frac≈0.6) contrast is **suggestive** (images competing through a shared
  rank-8 bottleneck) but **unconfirmed**: it could be magnitude/underfitting, not identifiability.
  Confirm/refute with J1 whitening + a T-sweep before writing it up.
- **Falsification test PASSES (machinery works):** `svd` tangents (injected geometric decay 0.5^j) drop
  `eff_rank` below the qr value at matched (N,k) — e.g. N=2,k=8: 15.83→7.75; N=4,k=16: 39.1→9.95 — so
  `σ_i(J)` tracks a *known* rank deficiency. This validates the spectrum measures injected structure.
- **Deterministic recovery has no principled cutoff → this is exactly why J1 is needed.** With
  `rcond=1e-10`, `recover_a` inverts near-null directions (σ→1e-6..1e-8 in heavy-decay svd) and rel_err
  explodes (60+ at ε=1). Without a noise floor, "recoverable" is ill-defined.

**Next steps (before any privacy claim):**
1. **J1 whitening** — S training seeds, `Σ_seed`, `snr_spectrum`/`q_eff` (scaffolded + Woodbury), CRLB
   law, `q_eff/q`-vs-recovery. **This is the first privacy-meaningful number.**
2. **Report σ-spectrum SHAPE** (gap vs gradual decay), not just the scalar eff_rank.
3. **T-sweep (5/20/50)** to disambiguate underfitting from structural rank: if eff_rank climbs toward
   Nk with T it was underfitting; if it plateaus <Nk it is structural.
4. J0 polish: `rel_err_row` against a fixed relative rcond (recoverable subspace).

---

## Jacobian-spectrum leakage program — Phase J0 built + submitted (2026-08-23)

Turned the identifiability note into buildable code. Plan rewritten PhD-readable (background primer →
J0/J1 concrete → J2–J6 goals): **notes/jacobian_leakage_experiment_plan.md** (v3).

- **New module `experiments/jacobian_spectrum.py`** implements Phase J0: the data-latent Jacobian
  `J = ∂vec(A_T,B_T)/∂a` where private data hides in image variations `x_i(a_i)=x_i^0+U_i a_i`.
  `J` computed by **forward-over-reverse JVP via double `autograd.grad`** (`exact_jacobian`,
  method `jvp_double`) — composes with the existing create_graph unroll. J1 whitening functions
  (`estimate_sigma_seed`, `snr_spectrum` via Woodbury, `q_eff`) scaffolded.
- **Single LoRA module** (`target_layers=(0,)`), GELU, float64, ds_mean frozen at a=0. Reuses
  `generate_target`/`get_finetuning_data`/`effective_rank`; does NOT edit `direct_inversion.py`.
- **Audit caught two things before submit:** (1) reusing `direct_inversion.A_rank_shape` (hardcoded
  MNIST dims 784/1000) would break the toy net → added local `_a_shape` reading in_features from
  `frozen`; (2) `generate_target`'s θ_T is ALL-layer, not the single-module target → use it only for
  `frozen/b0/B0[0]/ds_mean` and define `Y0:=forward_Y(0)`.
- **Gate = toy-AD finite-difference check** (`<1e-6` rel err, jvp-vs-reverse `<1e-8`) as Stage 0 of the
  bsub job (`sys.exit(1)` aborts on fail); then real MNIST smoke (`J` is [3568,8], FD `<1e-4`); then J0
  coordinate-recovery-vs-ε sweeps (qr + svd tangents, N∈{2,4}, k∈{4,8,16}).
- **Submitted job 966830** (`short-gpu`). All compute on WEXAC — no local runs (user rule).

---

## CRUX (activation × anchor × linearization) — smoothness→leakage REFUTED; softplus wins on LINEARIZATION (2026-08-23)

Analysis of existing matched-weight_change activation sweeps + anchor two-curves (notes/crux_activation_analysis.md,
figures/crux/*.png). Metric hygiene: ssim_norm / ctrl_margin_norm (clip-robust), activations compared at
MATCHED weight_change (wc is a confound).

- **Winner = softplus, but on LINEARIZATION grounds, NOT leakage magnitude.** Highest NTK survival at
  matched wc (feature_stability 0.953 MNIST), leakage flat across wc (+0.203, not a wc artifact), LOWEST
  function-space lin-error at every anchor α on both datasets, and anchor two-curve peaks at α=0 (linearizes
  so well at init the anchor buys nothing — cleanest signature).
- **The naive "smoother → more leakage" law is REFUTED.** Spearman(smoothness, leakage) ≈ +0.03. Matched-wc
  MNIST leakage is TOPPED by kinked/C¹ (selu +0.49, leaky_relu +0.48) while other C¹ (elu/celu +0.03) sit at
  the bottom; softplus +0.20, gelu/mish +0.08. flowers32 INVERTS it — smooth C∞ (gelu/silu/mish) leak LEAST
  (+0.02–0.04) and clip hardest (extraction failure); in realistic free-c, smooth acts have NEGATIVE margin
  (no leakage), only relu leaks.
- **The chain breaks at one joint:** smoothness→linearization-fidelity ✓ (softplus best, relu worst);
  linearization-fidelity→leakage-magnitude ✗. **Defensible claim: smoothest activation = most FAITHFUL,
  best-attributed reconstruction (softplus), not the largest leakage margin.**
- **Anchor attribution:** MNIST gelu/silu PASS (leakage peaks α=0.75 then collapses α=0.9); relu is the
  chain-breaker (worst lin-error yet highest margin, flat in α = extraction artifact); flowers32 free-c
  gelu/softplus negative margin at every α, only relu climbs to α=0.9 → attribution FAIL (hard refutation
  on flowers in the realistic regime).
- **Gaps → decisive test running (job 911475):** the softplus_b(β) sharpness knob was NEVER run — the clean
  controlled smooth↔kinked axis. Sweeping β×LR at matched wc directly tests smoothness→linearization without
  the confound of activation identity. Also open: feature_stability-vs-T (the "stays accurate over MORE T"
  clause is untested — all matched-wc data is T=1); flowers matched-wc band (needs --target_weight_change).

- **softplus_b(β) knob RESULT (job 911475) — CONFIRMS smoothness→linearization on the controlled axis.**
  LoRA function-space lin-error at matched weight_change (wc=0.10), β up = sharper→ReLU: β=1 **0.032** (best)
  < β=0.5 0.071 ≈ β=2 0.069 < β=5 0.106 < β=10 0.142 < β=50 **0.381** (≈ReLU, 12× worse). MONOTONIC in
  sharpness with a genuine sweet spot at β=1 (β=0.5 too flat is slightly worse). Isolates the mechanism
  without the activation-identity confound: **smoothness → faithful (linearizable) fine-tuning is CONFIRMED**;
  only the downstream 'linearization → leakage magnitude' link stays refuted. Figure:
  figures/crux/softplusb_linearization.png.

## MineGrad (AISTATS 2026) teardown + bridge re-plan (2026-08-21)

Read *MineGrad: Gradient Inversion Attacks on LoRA Fine-Tuning* (Sami, Sen, Güler; arXiv 2608.01521;
code `info-ucr/MineGrad`) end-to-end + its source. Archived to `papers/`. Full analysis:
**`notes/minegrad_analysis.md`**. Headlines:

- **It is a malicious-server analytical attack**, the LoRA successor to PEFTLeak (2506.04453). Our
  CLAUDE.md mislabels 2506.04453 as "honest CVPR2025" — it is malicious. **Fixed understanding.**
- **How it beats the rank bound (the question that motivated the read):** *not* by making a rank-`r`
  matrix leak >`r`. It uses `L≈2(S−1)` **independent coordinated LoRA modules** (V,O × encoders),
  each engineered (orthogonal position fingerprints + identity attention + coordinate-selector `A`
  init) to expose a **disjoint** set of ≤`r` tokens. Leakage `≈ L·r`. Fig 6: multi-encoder `r=2` =
  single-encoder `r=16`. **Takeaway for us: leakage scales with #independent modules, not `r`.**
- **All of its power is malice** (chosen parameterizations); **nothing transfers to our passive
  setting** except the *ideas*: multi-module tiling, input/`A`-side is where data lives, batch=average
  needs a codebook to de-mix (vision has none → our N≈10 superposition wall; a generative prior is the
  codebook replacement).
- **Vision:** ViT/CIFAR, LPIPS 0.20 (imperfect); **batch:** vocab-cosine de-mixing, no vision analog.
- **Theory upgrade (supersedes the two standalone ceilings):** the predictor is the **restricted
  composite Jacobian** `J = ∂vec(P_LoRA(∇_W L(g(z))))/∂z` and its `σ_min`/conditioning on the image
  manifold — subsumes `rank(M)` (feature) and `dρ≳Nk` (capacity), computable cheaply at the anchor,
  correlatable with SSIM. Retire `ρ_eff` and the standalone `dρ≳Nk` (its high-`k` prediction already
  failed on flowers).
- **Reframe the decoder target:** for the input layer `∇_{W₀}L = g_err·xᵀ`, the single-step
  `A`-gradient is `U Xᵀ` (`U=B₀ᵀG`) — the *same* `Ω=GXᵀ` factorization, LoRA-projected. The
  inversion-relevant statistic is **`X` (the row factor), not the full gradient**. The 0.997
  hidden-layer decode optimized the wrong quantity; single-sided input `x`-cosine may already beat
  0.637. **Re-score existing runs with `x`-cosine (Experiment B).**
- **Closest passive competitor is NOT MineGrad but Yao 2024** (*Risks When Sharing LoRA Fine-Tuned
  Diffusion Weights*, arXiv 2409.08482): learned passive map from final LoRA weights → private images,
  vision. **Read next.** Our surviving novelty: honest + discriminative + ordinary-init + the
  identifiability characterization (Yao is diffusion/black-box VAE + needs in-domain data).
- **5 next experiments** (all runnable in the analytic sim): A rank-vs-#views (matched `L·r`) ·
  B `x`-cos vs `g_err`-cos for A/B channels · C bridge-acc vs inversion-acc sensitivity map ·
  D shuffle cross-module correspondence (coordination vs prior) · E two-collinear-datasets
  (information vs hallucination). **A and D are decisive.**
- **Novelty caveat (honesty):** MineGrad + Yao already publish "LoRA→image for vision"; we cannot claim that
  per se. Claim the **honest, ordinary-LoRA, multi-module identifiability** result, reported under
  **free-coefficient** (not oracle) with `x`-cosine and LPIPS.

---

## Recovery (2026-03-19)

### What Was Lost
The WEXAC home directory lost its connection to the GitHub repo. Conversation history from Jan–Mar was lost. Claude Code custom skills were deleted.

**Still missing:**
- 3 custom Claude Code skills still need recreation: `/write`, `/lesson`, `/project-manager`
- `multi_seed_analysis.png` was regenerated from a 15-seed run (original was 200-seed; would need re-running the 200-seed sweep to reproduce)

**Recovered (2026-03-24):**
- All 5 missing figures regenerated from saved .pth tensors: `experiment_b_grid_r32.png`, `rank_sweep_sprint1.png`, `sprint1_summary.png`, `multi_seed_analysis.png`, `t_sweep_examples.pdf`
- 8 Claude Code commands recreated: `/review`, `/supervisor`, `/experiment`, `/debug`, `/figure`, `/paper`, `/research`, `/status`

**Recovered from GitHub (`myfork/main`):**
- All 18 papers in `papers/`
- All 7 notes files (GRADIENT_BRIDGE_PLAN.md, R2F_Guide, Inversion_Feasibility, Thesis_Direction)
- 4 figures (parameter_as_function_of_epoch variants)
- Full git history (15 commits)

**What was fixed (2026-03-19):**
- Rebased WEXAC state onto `myfork/main` history — all files restored
- Added Sprint 2 results (87 CSVs), WEXAC scripts, new experiment code
- Moved WEXAC job scripts to `scripts/`, logs to `scripts/wexac_logs/`
- Recreated `/research` and `/project-manager` Claude Code commands
- Updated CLAUDE.md, STATUS.md, LESSONS_LEARNED.md
- Pushed to `myfork` via SSH

---

## What's In Progress

### Feasibility reframed: rank → end-to-end Jacobian singular spectrum + a falsifiable experiment (2026-08-20)
Major conceptual sharpening of the identifiability note (supervisor-level input). The discipline:
**separate what is ESTABLISHED (the rank theorem = an information-MIXING mechanism, exact only under
frozen-known-G) from what is CONJECTURED (that an ordinary final LoRA preserves the private-image
directions).** The rank/`dρ ≥ Nk` inequality is downgraded to a *dimensional plausibility check*
(necessary, not a capacity law — `diag(1, 1e-10)` preserves rank but not information). The real object
is the **singular spectrum of the end-to-end data→adapter Jacobian** `J_full = ∂vec(A_T,B_T)/∂z`, and
the central falsifiable prediction is: **σ_min(J_full) collapses in the same regimes where
reconstruction collapses.** The headline experiment (replaces "estimate Flowers' intrinsic dim"):
**control k by construction (generator g:ℝ^k→images) → sweep N → measure σ(J_full) → run inversion**,
and check whether spectral collapse predicts reconstruction collapse. Three outcomes: World A (both
collapse together = identifiability transition, the dream); World B (J well-conditioned, attack fails =
decoder/optimizer is the bottleneck, adapter DID preserve info); World C (J collapses, decoder still
emits plausible images = **hallucination from prior**, the critical privacy distinction a capacity
number can't make). Also: staged Jacobians (J_grad / J_LoRA-step / J_full) localize the GELU-vs-ReLU
effect to gradient-stage vs LoRA-stage vs inversion-stage; the bridge is a *restricted inverse*
(non-invertible on arbitrary gradients, ~invertible on 𝓜_G) testable via ablation controls +
`rank(∂Y/∂z) ≈ Nk`; **k = rank J_g of the prior you actually restrict to, NOT a literature
intrinsic-dim estimate** (this resolves the "unvalidated d/k gain" audit flag — control k, don't assume
it). Local-vs-global and seed-vs-signal confounds flagged. Honest bottom line: the note does not prove
ordinary LoRA leaks — it gives a mechanism + the experiment that decides it. Revision LaTeX (drop into
Overleaf v4): `notes/identifiability_feasibility_revision.tex`.
**PENDING — full experiment plan (v2): `notes/jacobian_leakage_experiment_plan.md`.** Central quantity
upgraded (supervisor refinement) from raw σ_min to the **seed-whitened Fisher Jacobian**
J_SNR = Σ_seed^{-1/2} J and the **effective recoverable dimension** q_eff(ε)=#{i: ε·σ_i(J_SNR)>1};
central law = per-coordinate recovery error tracks the **Cramér–Rao floor** 1/(ε σ_i(J_SNR)), recon
collapses as q_eff/q→below 1 (data-processing inequality makes the bridge decode-vs-hallucinate test
rigorous). First experiment is now **tangent-COORDINATE recovery** (hide Nk private numbers in realistic
image variations, recover them), not whole images: J0 tiny deterministic testbed → J1 seeds+whiten+CRLB
(sweep ε past the linear-regime tautology) → J2 (N,r,L) phase diagram (fixed-epochs AND fixed-steps) →
J3 disjoint-adapter control (Δ_adapter) → J4 whitened v_min/v_max figure → J5 staged Jacobians + bridge
decomposition → J6 scale/whole-images. k = measured rank_eff(J_g) via top-k well-conditioned singular
dirs, never nominal latent width. Grounded in Belrose (2412.07003), SLQ/PyHessian, SimuDy code. Not yet
coded/submitted.

### Identifiability theorem written up — rank(M) < N ⟹ training data non-recoverable (2026-08-20)
Formal proof note: the first-layer weight signal factorizes as **Ω = G Xᵀ** (G = the gate matrix
M_{ki}=σ'(⟨w_k,x_i⟩), scaled by diagonal output-weight/loss matrices), and **rank(M) ≥ N is a
necessary condition** for recovering the N training inputs. If rank(M) = k < N the datasets consistent
with Ω form a **d(N−k)-dimensional affine family** (proof by kernel-dimension count via rank–nullity),
so recovery is information-theoretically impossible — with a hand-checkable 2-sample example (two
disjoint datasets giving identical weights). Scoped honestly: fixed-gate / NTK-linearized regime;
**necessary not sufficient** (sufficiency in the full bilinear problem = the open identifiability
question). This is the information-theoretic ceiling behind the activation/anchor/LoRA-rank study
(leakage ≤ min(rank(M), r, N)); the activation sets σ' (M's entries), the anchor reshapes M, LoRA
projects Ω to rank r. Deliverables: `notes/identifiability_rank_bound.pdf` (detailed, 7pp) +
`.tex` (Overleaf source) + `make_identifiability_pdf.py` (reproducible generator).

### Flowers-native reconstruction track — infra built + jobs submitted (2026-08-13)

New **native-dimension** flowers track (Step 4 "harder data" done properly, not the 28×28-grayscale
transfer hack). Trains a flowers-OWN base model θ₀ at RGB native dims and runs the full reconstruction
study on it. Framed to Gal's meeting: **Addition 1** (harder data) at native dims, carrying Additions
2/3 as inner axes; the **dimension ladder is the Q-A** (well-posedness) probe; **Phase D is Q-B**
(pretrain/finetune overlap = "additional *similar* images").

#### Clean Q-B re-run with a proper [0,1] pixel box — RESULT: novel leaks MORE, not less (2026-08-20, job 952081)
The first Q-B "seen>novel" gap was a **clipping artifact**. Added `--pixel_box` (`get_pixel_box_loss`
boxes the *image* `x+ds_mean` to `[0,1]`, not just centered `x in [-1,1]`; `build_base_name` tags it
`pbox`; default off -> MNIST byte-identical) and re-ran Q-B seen/novel in the validated free-c recipe
(sgd + relu_alpha 10000 + consistency 1.0 + n_restarts 5), 3 seeds each. The box killed the clip
(overflow `pre_clamp_max`->1.00006, ~1e-4 vs the old ~50%), and **the conclusion reversed**:

| arm | ssim_norm | ctrl margin | NCC dist (lower=better) | weight_change | feat_stab |
|-----|-----------|-------------|-----------|---------------|-----------|
| seen (overlap) | 0.477 | +0.265 | 5044 | 0.035 | 0.75 |
| **novel** (held-out) | **0.531** | **+0.421** | **1032** | **0.160** | 0.47 |

Chain: overlap -> theta_0 already fits the species -> tiny residual -> `weight_change` ~4-5x smaller
(~30x dropping the noisy seen-43 config) -> *less* instance-specific leakage (control margin +0.26 vs
+0.42, NCC 5044 vs 1032). Novel species force a large, instance-bearing dW that reconstructs the
specific image better. Raw SSIM still marginally favors seen (0.47 vs 0.42) only because novel touches
the pixel boundary (28% of pixels at the edge, overflow ~1e-4) -- every scale-robust metric favors
novel. **Takeaway for Gal**: overlap does NOT protect the instance; it weakens leakage of the specific
image (theta_0 absorbs the class-generic content). Feature-map-injectivity story, sign made explicit.
Tensors: `results/exp_b_T1_flowers32_r8_free_s4{2,3,4}_a10000_vw5_{seen,novel}_pbox.pth`.
**Next:** regenerate seen-vs-novel example grids from these tensors (the old `figures/pdf_examples/FREEC_QB_*`
show the stale clipped run).

#### Re-doing the two clip-contaminated experiments with --pixel_box (2026-08-21, jobs 38528/38529)
Audit (LESSONS_LEARNED) found only two experiments we rely on are clip-contaminated; both are being
re-run honestly. First threaded `--pixel_box` into `run_sequential_peeling` too (the N>=4 canonical
path used `--sequential_peel`, which had bypassed the box).
- **38528 N-sweep** (npc 1/2/4/8, 3 seeds): canonical recipe + `--pixel_box --verify_weight 5.0`.
  Expect the collapse to hold (it shows on ssim_norm+margin) but N>=4 absolute numbers to rise off the
  clip floor (npc=4 had 47% clip).
- **Optimizer (matched-wc)**: jobs 38529 + 70216 (adamw low-lr ladder, since adamw makes ~3x larger
  updates/lr than sgd — sgd@0.01 wc=0.036, adamw@0.0005 wc=0.029, a clean match).
The activation / rank / Q-A axes were clip-free (<0.05) and are NOT re-run.

**FINAL RESULTS (2026-08-21):**
- *N-collapse* (margin, 3-seed mean): N=2 **+0.260** -> N=4 **+0.175** -> N=8 +0.021 -> N=16 +0.030.
  Clean collapse: strong at N=2, ~halved at N=4, gone by N>=8. Box HELPED N=4 (+0.028->+0.175 vs the old
  default run). Caveat: the N>=8 peel path clips unboxed (~0.36 at seeds 43/44) and blanks boxed, so the
  N>=8 value carries +-0.05 -- but every treatment gives a small margin, so the collapse is robust.
- *Optimizer* (matched wc~0.03, clip-controlled): sgd/cosine ssim_norm **0.790** margin **+0.277** >
  sgd/l2 0.668/+0.195 > adamw/cosine 0.559/+0.176 ~ adamw/l2 0.540/+0.188. Two clean findings:
  (1) **cosine >> l2 but only for sgd** (adamw shows no cosine gain); (2) **sgd leaks more than adamw at
  matched wc** (adamw's per-parameter scaling makes dW harder to invert).

- **Dimension ladder:** two base models — `flowers32` (RGB 32×32, **D=3072**, exact Haim CIFAR recipe)
  and `flowers64` (RGB 64×64, **D=12288**, rich target). Task = species-index **parity** over 102
  species; base trained on **train+val pooled** (~2040 imgs, 500/class); fine-tune/reconstruct from the
  disjoint **test** split.
- **Infra (all additive, MNIST paths byte-identical, 26 new tests pass):** `configs.DATASET_SPECS`
  (per-dataset shape/dim/hidden/θ₀); `data_utils` flowers32/64 RGB loaders + Phase-D `source` filter;
  `ntk_extraction` shape-aware x̂-init; `run_experiment_b` dim/θ₀ threading + `--pretrained_path`/
  `--source`/`--holdout_species` + figure-gating fix + config now saves activation/optimizer/dataset;
  `plotting` RGB imshow + dataset-aware labels + per-dataset figure dir; `recompute_metrics` dataset
  column; `run_anchor_sweep` `--dataset`/`--lr`/`--skip_if_exists`; new
  `dataset_reconstruction/problems/flowers102_parity.py` (+CreateData/GetParams dispatch, `--flowers_hw`
  /`--flowers_gray`/`--flowers_holdout`).
- **Jobs (WEXAC, dedicated A100 — FP64 workload, ~15× faster than shared A40/L40S):** training
  527051 (flowers32 main + Phase-D holdout base) and 527054 (flowers64); 5 sweeps chained via LSF
  `done()`: 527255 activation×lr (flowers32), 527258 anchor two-curve, 527262 N/rank/optimizer,
  527264 Q-B overlap (Phase D), 527265 activation×lr (flowers64). Each sweep has Stage-0 (θ₀ load +
  shape + filename-uniqueness) and Stage-0.5 (short-config NTK sanity) guards.
- **Next:** confirm θ₀ reaches max-margin (train-error→0, loss<1e-20, p-val growing); rescore
  each sweep with the full metric suite; build the Q-A dimension-ladder curve + Q-B overlap-vs-novel
  contrast; send review grids (GT/recon/control, best+worst) per phase.

#### ⭐ DE-CONFOUNDED activation result (2026-08-17) — "smoothness → leakage" is REFUTED on native flowers
- **The weight_change confound is resolved.** Sweeping each activation's lr over a **10× weight_change range**:
  softplus (wc 0.057→0.571), selu (0.040→0.395), relu (0.018→0.178) keep **ssim FLAT** (~0.69/0.70/0.71) and
  ctrl_margin flat (~+0.28) — the clean **linearization-stability** signature. silu (wc 0.013→**0.132**),
  gelu, mish keep ssim **DEAD** (~0.05/0.09) at *any* weight_change. So the ranking is a **genuine activation
  property, not a training-motion artifact.**
- **Smoothness does NOT predict leakage** (contra MNIST/Additions-2/3 spine): softplus smooth+strong, silu/gelu
  smooth+DEAD, relu kinked+strong. **The real divide is self-gated/swish (silu, gelu, gelu_tanh, mish, hardswish
  → dead, margin ~0) vs ReLU-like/saturating (relu, leaky_relu, softplus, selu, tanh, sigmoid, elu, celu → strong,
  margin +0.25–0.30).** Mechanism hypothesis: the swish family's non-monotone bump near 0 (x·σ(x) dips negative)
  kills the reconstruction signal at these input scales. **This revises the "smoother → better linearization →
  better leakage" argument** — take to Gal as the headline flowers finding.
- Matched-wc (≈0.04) ranking, flowers32: selu +0.297, tanh +0.296, relu +0.281, softplus +0.278, elu/celu +0.273,
  leaky_relu +0.268, sigmoid +0.250 ≫ mish +0.011, gelu −0.007, silu −0.011. flowers64 (D=12288) same pattern
  (leaky_relu +0.253 top; swish family dead). **Q-A ladder holds at matched wc** — leakage degrades 3072→12288.
- All configs still `ntk_passed:False` (the strict wc<0.01 gate); feature_stability high. Data: 78 flowers32 +
  39 flowers64 result .pth. Figures: `figures/sprint1/flowers32/REVIEW_*`.

#### First flowers32 results (2026-08-16) — richer data gives CLEANER leakage; MNIST activation ranking does NOT transfer
- **All three θ₀ trained** (capped 150k epochs after finding the 1e-20 threshold unreachable — loss ~t^-1.77):
  `weights-flowers32.pth`, `weights-flowers32_holdout.pth` (Phase-D), `weights-flowers64.pth`. All reached
  max-margin (train-error 0 by ep5k, p-val ~11, test-error ~0.44 = chance = the expected memorization signature).
- **flowers32 activation sweep** (lr=0.01, N=2, T=1, r=8, oracle), ranked by control margin:
  **selu +0.284 (ssim .697) ≈ tanh +0.283 ≈ softplus +0.278 (.688) ≈ celu/elu +0.273 ≈ sigmoid +0.250**
  ≫ mish +0.011 ≫ gelu/silu/hardswish ~0.00/−0.01 (ssim ~.05).
- **Headline (plan prediction confirmed):** control margins are **~2× MNIST** (+0.25–0.28 vs +0.11–0.18) and
  **selu (.697) & softplus (.688) BEAT the mean-baseline (.646)** — on MNIST recons sat *below* baseline. Native
  RGB flowers (where `ds_mean ≠ each image`) is a **stronger, cleaner instance-leakage demonstration.** Softplus/
  selu/sigmoid clip 0.00 (vs 0.44–0.62 on MNIST). Grids: `figures/sprint1/flowers32/REVIEW_flowers32_{selu,softplus,gelu}_oracle.png`.
- **The MNIST "softplus ≫ all" ranking does NOT cleanly transfer** — bounded/saturating acts (selu/tanh/sigmoid/celu/elu)
  are comparable to softplus here.
- **⚠ CAVEAT (metric law):** *every* config is `ntk_passed:False` and **weight_change is unmatched** — the high-margin
  acts simply moved the net more (wc 0.05–0.11) than gelu/silu/hardswish (wc~0.027, barely trained = the same lr-confound
  as MNIST GELU). This ranking is **confounded, not a verdict**; the clean comparison is the matched-weight_change read
  from the lr-cal runs (other lr values in the sweep, on disk).
- **Addition-3 anchor two-curve (flowers32, T=10, r=8) — interior optimum REPLICATES.** gelu **full-FT** SSIM
  `0.629→0.746→0.977(α=0.5)→0.630→0.403` while lin-error falls monotonically `0.0048→0.0006`: SSIM peaks at the
  **midpoint α*≈0.5** (MNIST was ≈0.75), then falls past it even as linearization improves — the clean attribution
  signature (linearization win up to the peak; identifiability degradation past it). **softplus barely benefits**
  from the anchor (α=0 already near-best) → the anchor gain is **activation-dependent**. Figs:
  `figures/anchor_sweep/anchor_two_curve_{full,lora_r8}_flowers32_T10_r8_{gelu,softplus}_s42.png`;
  data `results/anchor_sweep_flowers32_T10_r8_{gelu,softplus}_s42.pth`. (silu arm still running.)
- **Q-B (Phase D) path bug fixed + rerunning** (job 727508): the sweep referenced the holdout θ₀ via a repo-root-relative
  `models/` path instead of `dataset_reconstruction/models/` → Stage-0 FileNotFoundError (18s abort). Fixed; 4 seen/novel
  results landing. flowers64 activation sweep (Q-A ladder rung) still running (10 results).
- **Q-B RESULT (Phase D, overlap vs novel; softplus r=8, holdout θ₀, avg 3 seeds) — familiarity AIDS leakage (flips the naive prediction).**
  seen (overlap): ssim 0.619, ctrl_margin **+0.389**, weight_change 0.0965. novel: ssim 0.412, ctrl_margin +0.339, weight_change 0.1070.
  ✅ overlap → **smaller weight_change** (the cᵢ-shrink mechanism: θ₀ already fits → smaller residual → smaller ΔW). ❌ but overlap →
  **HIGHER faithfulness, not lower** (ssim 0.619 vs 0.412) — contradicting the plan's "recovers only novelty" prediction. The richer
  feature map θ₀ has for *known* species makes the instance recover *better* from a *smaller* ΔW ⇒ Gal's feature-map-injectivity worry is
  NOT borne out; familiarity helps the attack. (Minor caveat: seen species 20–101 vs novel 0–19 are different flower sets.)
- **Q-A RESULT (dimension ladder, softplus lr=0.01 N=2 r=8) — leakage clearest at the MIDDLE rung.**
  D=784 (MNIST, prior ~+0.11) → D=3072 (flowers32 **+0.278**) → D=12288 (flowers64 **+0.227**); ssim 0.688→0.618, weight_change
  0.114→0.066. Non-monotonic: richer-than-MNIST helps (ds_mean ≠ image), but past ~3072 the higher pixel dim reduces well-posedness
  (Q-A prediction). **Caveat:** lr=0.01 unmatched weight_change → part of the 3072→12288 drop is the training-motion confound; the
  matched-wc ladder (lr-cal runs) is the clean version. flowers64 full activation sweep still running (~11h, heavy D=12288 extraction).

#### REALISTIC free-coefficient results (2026-08-18) — the activation ranking FLIPS; oracle was misleading
All of the above is **oracle** (cᵢ from the true x = cheating). Re-ran the whole program in the
**realistic free-coefficient attack** (Haim-style). **The free-c recipe was the crux** (see LESSONS):
default free-c is broken (sign-flip + ModifiedReLU + LBFGS); the working recipe is **SGD extraction +
`relu_alpha=10000` (≈ReLU) + `consistency_weight=1.0` + random restarts** (reproduces the Sprint-2
known-good ~0.59 on MNIST). For the activation study, extraction is **decoupled** to fixed ReLU-like
(`--extract_activation modified_relu --relu_alpha 10000`) — the realistic fixed-ReLU attacker.

- **Rank curve (N=2):** flat, ssim 0.60–0.65, ctrl-margin +0.10–0.15 across r=4→64.
- **N curve:** collapse at N≥4 (N=2 **0.604** → N=4 0.191 → N=8 0.190 → N=16 0.226) — the superposition
  wall is real in the realistic attack.
- **Q-B REPLICATES in free-c:** seen (overlap) **0.587** > novel 0.368 — familiarity aids leakage. ✓
- **Q-A ladder holds:** flowers32 0.604 → flowers64 (D=12288) 0.604 (ctrl-margin +0.10 → +0.08).
- **★ ACTIVATION RANKING FLIPS (overturns the oracle "softplus wins / swish dead"):** under the
  realistic fixed-ReLU attacker, **silu 0.711 > gelu/gelu_tanh 0.692 > mish 0.689 > elu 0.643 >
  softplus 0.609** — the swish family (what real ViTs use) is the **most** vulnerable, and silu leaks at
  *lower* weight-change (0.026 vs softplus 0.114). The oracle "softplus wins" was a **matched-extraction
  artifact** (softplus/gelu extraction is a bad free-c inverter). **The defensible headline is the
  realistic one: silu/gelu leak most to a ReLU attacker.**

#### Optimization improvements (SPEAR / ANA-GIA / priors) — what helped (2026-08-18)
Tested closed-form-c (ANA-GIA least-squares), SVD-init (SPEAR low-rank), diversity, TV prior, and
sequential peeling on the LoRA N-collapse. **Result: the coefficient *recipe* is the win, not the
structural additions.** SVD-init *hurts* on raw LoRA (0.38→0.22) — its SVD spans the **adapter's**
subspace, not the data (SPEAR/ICA need the **full gradient** → concrete motivation for the Gradient
Bridge; see `notes/related_work_spear.md`). TV is marginal on clean MLP gradients (its leverage is for
ViT). Peeling helps modestly at N≥8 (0.235 vs 0.156). New flags in `run_experiment_b`:
`--closed_form_coeff --svd_init --diversity_weight --tv_weight --sequential_peel --peel_refine`; SPEAR
note written. **N-separation on LoRA needs the bridge** — a thesis argument, now measured.

### GB-Phase 2 — the SVD inversion was the weak link; decode ↑ ≠ image ↑; model-based extraction is the fix (2026-08-18)

A 5-experiment learning chain on turning the bridge's decoded input-layer gradient into an image:
- **Priors don't help (determined factorization).** TV (job 383791) HURT monotonically (softplus best λ=0
  SSIM 0.15; gelu 0.11) — wrong prior (digits are sharp, TV smooths). L1-sparsity + non-negativity —
  the *right* prior for digits (job 389521) — ALSO lost to raw SVD on both activations. The bridge
  inversion x=row-factor of g_err·xᵀ is *determined*, so no prior has a null-space to resolve.
- **Higher rank hurts (job 392328).** Two-sided r∈{8,32,64}: decode/SSIM DROP monotonically (softplus
  0.888/0.136 → 0.632/0.037); the bigger decoder input under-fits / dilutes. r=128 OOM'd. Best = r=8.
- **Decode was under-trained, not capped (job 400603).** Two-sided r=8 at 4× data / 3× epochs: softplus
  decode **0.888 → 0.945** (the ViT-milestone cosine!), gelu 0.634 → 0.717. So more proxy data lifts the
  decode.
- **But decode ↑ ≠ image ↑ — the Q-A decoupling.** Pushing decode 0.89 → 0.945 barely moved SSIM
  (0.136 → **0.172**). Even a 0.945-cosine gradient, via raw SVD, gives a coarse blob — exactly Gal's
  deck slides 21-23 ("cos saturates, SSIM varies, the PRIOR is the lever").
- **The fix my own data shows: the SVD was the wrong inverter.** Exp 2's corruption test — the
  MODEL-BASED Experiment-B extraction fed a ΔW with the input layer at cosine 0.64 → **SSIM 0.52**, vs
  the SVD's 0.10 at the same cosine. The model-based inverter uses θ₀ + all-layer structure as an
  implicit prior (the ViT trick), closing the cosine→pixel gap the SVD can't.
- **CULMINATION CONFIRMED (job 445780): the bridge attack closes end-to-end.** Real per-layer decoders
  (trained on PUBLIC proxy only) -> decode each victim sample's per-layer LoRA update -> model-based
  `run_ntk_extraction` (NOT SVD). N=2, oracle coefficients + oracle per-layer sign (upper bound):

  | arm | softplus ssim / norm | gelu ssim / norm |
  |---|---|---|
  | TRUE ΔW (ceiling) | 0.692 / 0.825 | 0.995 / 0.997 |
  | **DECODED all-layers** (the attack) | **0.458 / 0.622** | **0.554 / 0.740** |
  | DECODED input-only | 0.482 / 0.629 | 0.557 / 0.742 |
  | TRUE input-only | 0.688 / 0.829 | 1.000 / 1.000 |
  | SVD (old inverter) | 0.17 | 0.10 |

  Headline: the model-based inverter recovers **recognizable digits (SSIM ~0.5, ssim_norm 0.62-0.74),
  3-5x the SVD** -- the deck's Q-A lesson made literal (the base-model structure, not the cosine, is the
  lever). Gap to the ceiling = the **input-layer AGGREGATE decode cosine (0.42 softplus / 0.37 gelu)**:
  per-sample the input decoder is 0.92/0.67, but two OPPOSITE-LABEL victim samples partially cancel in the
  sum (signal cancels, decode error does not), and the input layer is the one carrying x. Near-perfect
  hidden/output decoders (0.99) add nothing beyond input-only; `TRUE input-only`=1.00/0.69 confirms the
  input layer alone suffices. Code: `experiments/gradient_bridge/phase2_e2e.py`. Jobs 383791/389521/
  392328/400603 (diagnostic chain) + 424120/445780 (e2e). Optional refinement: converged input decoder
  (30k/200ep, per-sample 0.945) to lift the aggregate cos toward the ceiling; N=1 to remove the
  opposite-label cancellation entirely.

### GB-Phase 2 e2e — generalization: large N (MNIST) + flowers (2026-08-19)

Extended `phase2_e2e.py` to be **dataset-aware** (threads DATASET_SPECS geometry/base-model/proxy;
flowers32 D=3072 RGB) and **N-sweeping** (train decoders once, attack at each npc). Jobs 807059 (MNIST
N-sweep) + 949874 (flowers32).

**MNIST large-N (`ssim_norm`, DECODED all-layers / TRUE ΔW; baseline-adjusted, 0.5 = leak line):**

| N | softplus DECODED / TRUE | gelu DECODED / TRUE |
|---|---|---|
| 2  | 0.623 / 0.825 | 0.745 / 0.997 |
| 4  | 0.555 / 0.668 | 0.601 / 0.835 |
| 10 | 0.479 / 0.559 | 0.540 / 0.648 |

Findings:
- **The bridge is NOT the extra bottleneck for softplus** — DECODED ≈ TRUE at every N; softplus's *direct*
  attack also collapses toward baseline and the adapter tracks it down.
- **For gelu the bridge IS the extra bottleneck at large N** — the *direct* attack (TRUE) stays strong
  (0.997→0.835→0.648) but DECODED collapses faster because the input-layer AGGREGATE decode falls to ≈0
  (L0: 0.39→0.07→**0.00** at N=10). The adapter loses information the full ΔW still has.
- **"More than 2" verdict:** the bridge clearly leaks at N=2 (0.62–0.75), is marginal at N=4, and is
  essentially gone by N=10 (softplus 0.48 < 0.5). The **superposition wall** — shared with direct
  inversion (raw SSIM drops below the mean-baseline by N=4 for both). N, not the adapter, is the limit.

**Flowers32 (job 949874) — the bridge is DATA-HUNGRY; it mostly fails on flowers while direct inversion
succeeds.** Only ~7k public flowers images → 5,810 proxy pairs (vs MNIST 15k) for a 134M-param input
decoder at 3072 dims → the input-layer decode is STARVED (softplus L0 0.33 / gelu L0 0.66 proxy cosine).
Results (ssim_norm, DECODED all-layers / TRUE ΔW; baseline-adjusted, 0.5 = leak line):

| N | softplus DECODED / TRUE | gelu DECODED / TRUE |
|---|---|---|
| 2 | **0.565** / 0.753 | **0.260** / 0.999 |
| 4 | 0.322 / 0.458 | 0.295 / 0.925 |

- **Direct inversion leaks flowers cleanly** (softplus 0.75, gelu near-perfect 0.999) — full ΔW skips the
  decoder, so RGB/high-dim doesn't hurt it.
- **The bridge mostly does NOT**: softplus MARGINAL (0.57, just over 0.5, vs direct 0.75); gelu FAILS (0.26
  vs 0.999) — the largest direct-vs-bridge gap seen. gelu additionally hurt by its OUTPUT-layer decode
  collapsing to 0.016 on flowers (drags all-layers below input-only 0.378).
- **Honest headline:** the LoRA/bridge attack needs enough public proxy relative to input dimension.
  Flowers starves it; direct inversion has no such weakness. **Fix to try:** a larger RGB proxy (CIFAR-100,
  50k, same 32×32) as the flowers-geometry decoder's training set — R2F explicitly allows proxy≠private.
  Grids: `figures/gradient_bridge/phase2_e2e_flowers32_N{2,4}_{softplus,gelu}.png`.

**CORRECTION (2026-08-23, rescue test job 918308): flowers is DIMENSION-limited, NOT starvation-limited.**
Hypothesized flowers failed from proxy starvation (7k images); tested by retraining the flowers32 decoder
on a 3x-larger CIFAR-100 proxy (20k, same 3072 geometry). Result: input decode barely moved (softplus L0
0.33->0.37, gelu 0.66->0.65) and DECODED all-layers ssim_norm UNCHANGED (softplus 0.57, gelu 0.31->0.385,
still <=baseline 0.646). So more proxy does NOT rescue flowers -> the limiter is INPUT DIMENSION vs
measurement rank: two-sided rank-8 observes ~r/D of the input directly = 8/784=1% (MNIST) vs 8/3072=0.26%
(flowers), so the decoder must hallucinate far more of a 3072-dim input regardless of proxy quantity. The
Fashion(784,leaks) vs flowers(3072,fails) contrast is input-DIMENSION, not proxy-abundance. (Caveat:
CIFAR-100 is also a different distribution, conflating 'more data' with cross-distribution transfer; either
way it didn't rescue.) Supersedes the earlier 'starvation' framing for flowers.

**Fashion-MNIST (job 953487) — DECISIVE: flowers failed from STARVATION, not hardness.** Fashion is
harder than MNIST but reuses the 784-MLP (small decoder) with an abundant corresponding proxy (60k). The
decoders trained MNIST-quality (softplus L0 **0.934**, L1 0.9997, L2 0.9999 — NOT starved), and the
bridge LEAKS:

| N | softplus DECODED / TRUE (ssim_norm) | gelu DECODED / TRUE |
|---|---|---|
| 2 | **0.619** / 0.798 | **0.584** / 0.597 |
| 4 | 0.383 / 0.410 | 0.314 / 0.490 |
| 10 | 0.205 / 0.212 | 0.268 / 0.278 |

- softplus N=2 ssim_norm **0.619 ≈ MNIST's 0.623** — the bridge handles harder *content* fine when the
  proxy is abundant. Confirms the differentiator is **proxy abundance vs input dimension**, NOT data
  difficulty. (Aside: Fashion's input-layer AGGREGATE decode is low, L0 0.167, yet the model-based
  extraction still recovers 0.62 — the hidden/output layers 0.98/0.997 carry it.)
- N=4/10 collapse toward baseline again — the superposition wall, shared with MNIST/flowers/direct.

**Monster-network track (CIFAR-10, wide+deep 3072-2048x4-1, ~19M, 5 layers) — the BRIDGE DOES NOT SCALE
TO DEPTH (job 997876).** θ₀ trained to max-margin (100% acc, margin +24.8, loss 5e-13) after the init fix
(Kaiming — a global 1e-4/default init collapses the 5-layer forward pass: logit std 0.002, stuck at ln2
with flowing grads; jobs 953913/956281/978619). Bridge N=2 (CIFAR 50k proxy = abundant, so NOT starved):

| arm | softplus ssim/norm | gelu ssim/norm |
|---|---|---|
| TRUE ΔW (direct) | 0.774 / 0.776 | **1.000 / 1.000** |
| **DECODED all-layers (bridge)** | **0.299 / 0.316** | **0.247 / 0.271** |
| DECODED input-only | 0.562 / 0.558 | 0.533 / 0.544 |
| per-layer decoders | 0.86 0.92 0.95 0.92 0.96 | 0.38 0.69 0.70 0.72 0.89 |

Three things cleanly separated: (1) the **decoders scale** — all 5 hit 0.86-0.96 (softplus), not starved;
(2) **direct inversion scales** — TRUE ΔW 0.77 / **1.000**, the extraction handles the deep net perfectly
with the exact ΔW; (3) **but the end-to-end bridge FAILS** — DECODED all-layers 0.30-0.32, BELOW baseline
(0.615). The depth-specific signature: on shallow nets all-layers≈input-only, but on the monster
**all-layers (0.30) << input-only (0.56)** — the 4 decoded HIDDEN layers HURT. **Per-layer decode errors
compound when assembling the full ΔW across depth**, so deeper = more noise not more signal. This is a
BRIDGE-SPECIFIC depth limitation (direct inversion, using the exact ΔW, has no such problem). Code:
`experiments/train_monster_base.py` + `phase2_e2e.py --dataset cifar10`. Grids:
`figures/gradient_bridge/phase2_e2e_cifar10_N2_{softplus,gelu}.png`.

**4-AXIS GENERALIZATION SUMMARY (bridge = DECODED all-layers, ssim_norm at N=2; 0.5 = leak line):**
MNIST 0.62/0.75 ✓ · Fashion 0.62/0.58 ✓ · Flowers 0.57/0.26 ✗(starved) · Monster 0.32/0.27 ✗(depth).
The bridge leaks when it has abundant proxy AND a shallow net; it fails on scarce/high-dim data (flowers)
and on deep nets (monster). Direct inversion leaks in ALL four. N>2 collapses both (superposition wall).

**⚠ METRIC-HYGIENE CAVEAT (folded in from sibling session, 2026-08-21).** Raw SSIM on a CLIPPED recon is
an artifact: extraction only softly boxes the CENTERED x∈[-1,1], so hard recons let x+ds_mean leave [0,1]
and clip. Measured clipped_fraction on the bridge recons: **MNIST 32-47%, Fashion 22-30%, monster
all-layers 17-20%, flowers 0-7%, monster input-only/TRUE 0-5%.** So the RAW ssim columns above are
inflated for MNIST/Fashion/monster-all-layers. **The conclusions are stated on `ssim_norm`, which matches
mean/std to target before scoring and REMOVES the clip artifact (per metrics.py) — so the 4-axis story,
the ordering, and the depth/starvation findings all HOLD.** **RE-RUN COMPLETE (--pixel_box, jobs 173383-173392 + monster 38538).** Clean DECODED all-layers
ssim_norm at N=2: MNIST sp 0.607 / ge 0.566 · Fashion 0.609 / 0.584 · Flowers 0.566 / 0.305 · Monster
0.328 / 0.247 (TRUE ceilings 0.74-1.00). **All conclusions HOLD.** Only real shift: MNIST gelu
0.745->0.566 — pixel_box CHANGES the reconstruction (constrains it to valid pixels during optimization),
not just the metric, so the old 'gelu leaks best on MNIST' was partly a clip benefit; the honest read is
softplus~gelu comparable on MNIST. Everything else within noise. PDF + bar charts regenerated on clean
ssim_norm. Rule going forward: print clipped_fraction before trusting any raw-SSIM number; if >~0.05,
read ssim_norm/NCC/margin or re-run with --pixel_box.

### GB-Phase 2 (earlier) — two-sided measurement rescues the input-layer decode; the inverter needs the INPUT layer (2026-08-18)

### GB-Phase 2 (earlier) — two-sided measurement rescues the input-layer decode; the inverter needs the INPUT layer (2026-08-18)

Two follow-ups (jobs 366577 Exp 1, 367539 Exp 2) resolve the GB-Phase 2 negative into a nuanced positive
with a clean mechanism.
- **Exp 1 — two-sided measurement rescues the input-layer decode.** softplus input-layer (layer 0)
  decoder cosine **0.637 → 0.912** and image `img_cos` **0.49 → 0.90** under a two-sided (nonzero-A)
  single-sample measurement; higher rank (r=32/64) does NOT help (stays 0.64). Mechanism: x is the
  **row factor** of `∂L/∂W₀ = g_err·xᵀ`; single-sided observes only `col(B₀)` (misses the row space),
  two-sided observes `row(A₀)` — exactly where x lives. So the failure was a measurement-CHANNEL mistake,
  not fundamental. Caveat: `img_SSIM` still 0.151 — strong *coarse/directional* recovery but blurry
  (low-freq structure, not fine strokes); a visual confirms recognizable-but-blurry.
- **Exp 2 — the full inverter depends on the INPUT layer, not the hidden layer (overturns the hypothesis).**
  Corrupting the true single-step ΔW per layer to measured decoder cosines and running the Experiment-B
  extraction: for gelu, `hidden-only bad (1/0.64/1)` → ssim **0.982** (≈ the true-baseline 1.000), but
  `input-only bad (0.64/1/1)` → **0.521**. So corrupting the hidden layer barely hurts; corrupting the
  input layer breaks it. **The bridge's 0.997 hidden-layer decode is useless to the inverter — it needs
  the input layer.** (softplus: true 0.676 → input-bad 0.576, milder but same direction.)
- **Combined resolution:** the inverter needs the input layer → single-sided decodes it poorly (0.64,
  row-space missed) → GB-Phase 2 fails → **two-sided decodes it well (0.91) → partial image recovery**
  (`img_cos` 0.90, blurry). The bridge CAN recover images end-to-end with two-sided measurement, coarse
  fidelity only. Data: `figures/gradient_bridge/phase2_*.png`, job logs 366577/367539.

### GB-Phase 2 (first pass) — single-sided bridge does NOT recover images; the layer it decodes ≠ the layer with the image (2026-08-18)

Ran the end-to-end bridge attack (job 357050): decode the INPUT-layer gradient (layer 0, the only one
whose per-sample gradient contains x, since ∂L/∂W₀ = g_err·xᵀ), take x̂ = top right singular vector,
score vs the true image. Result: **image NOT recovered.**
- softplus decoder_cos **0.637**, img_cos 0.488, **img_SSIM 0.021** (garbage; MNIST baseline ~0.7);
  gelu 0.518/0.466/0.020; relu 0.515/0.478/0.021. More training raised the gradient cosine
  (0.55→0.64) but not the image.
- **Why (the key insight):** the bridge's 0.997 (Exp B) was on the HIDDEN layer, which does not contain
  the image. On the INPUT layer softplus decodes only to 0.637 — the gate-rank mechanism *in reverse*:
  the hidden layer's input is the *clustered* activations σ(W₀x) (predictable → 0.997), but the input
  layer's input is the *raw, diverse images* (unpredictable from the proxy prior → 0.64). **The layer
  the bridge can decode (hidden) ≠ the layer that holds the image (input); the image layer decodes too
  poorly to recover pixels.**
- **Thesis implication:** the gradient-bridge attack does **not** recover images end-to-end on this
  testbed — the "necessary but not sufficient" caveat is now *measured insufficient*. Direct inversion
  (Experiment B, full ΔW) remains the attack that works; the bridge's high hidden-layer cosine does not
  translate to pixels. Grids: `figures/gradient_bridge/phase2_{softplus,gelu,relu}.png` (vague blobs).
- Open route (not yet run): decode ALL layers → assemble the full ΔW → feed the Experiment-B inverter
  (rather than SVD one layer). But each layer decodes at only ~0.5–0.64, so a ~0.6-cosine full ΔW is
  likely still short of the image; the input-layer bottleneck is fundamental.

### Bridge connection — softplus is the MOST decodable: a duality between the two attacks (2026-08-18)

Ran the never-tested link between direct inversion and the gradient bridge (job 106085): trained the
R2F decoder `f_φ:(A,B)→∇_W L` on the hidden layer (L1, r=8) per activation. Decoder full-cosine
(single-sample gradient; projection ceiling ≈0.087 for all):
- **softplus 0.997 (final 0.996, STABLE) — the >0.9 milestone MET on a single-sample gradient** (11×
  above the ceiling; previously only the m=8 *batch* gradient hit 0.95). softplus_b50 0.773, relu 0.750,
  gelu 0.685 (final 0.560, unstable), silu 0.664.
- **DUALITY (the headline):** the same collinearity that makes softplus the *worst* for the
  direct-inversion attacker (`cos_M=0.997`, `eff_rank(M)≈1` → entangles samples → least leakage) makes
  it the *best* for the **bridge** attacker — its gradient is so predictable that the decoder recovers
  the full gradient nearly perfectly. **The two thesis attacks have opposite activation preferences**;
  gelu/silu (the deployment C∞ units) are the worst-decodable. Data: `results/gb_decoder_L1_r8_*.json`.
- Bears on the thesis: a softplus-based victim is highly vulnerable to the *bridge*, weakly to *direct
  inversion*; a relu victim the reverse. Real ViTs use GELU — the least-decodable here — so the bridge
  is hardest exactly on the deployment-realistic activation.

### Theory closure — the σ''/||∇Φ|| correction is NOT supported; flowers64 also not high-k (2026-08-18)

Two quantitative checks (job 106084, `results/theory_closure_test.csv`):
- **Corrected Lemma B REFUTED quantitatively.** Pearson(lin_err, σ'')=+0.109 vs
  Pearson(lin_err, σ''/‖∇Φ‖)=**+0.042** — the rev.3 "fix" is *worse*, not better, and neither predicts
  the relative lin-error within the smooth family (no clean law: softplus_b0.5 low-ratio/high-lin, gelu
  mid-ratio/low-lin, softplus_b50 high-ratio/high-lin). **Honest walk-back:** only the *coarse* kinked≫
  smooth split (the Dirac kink, robust at matched wchg) holds; there is **no simple fine-grained
  σ''-based law** for the relative lin-error. rev.3's σ''/‖∇Φ‖ claim is withdrawn (→ note rev.4).
- **High-k item CLOSED (negative).** eff_rank(X) at N=64: MNIST **46.3** > flowers64 **39.7** >
  flowers32 37.3. Even D=12288 flowers64 is *lower* effective-rank than MNIST — downsampled natural
  images collapse to a few global modes. The capacity ceiling cannot be isolated with downsampled
  flowers; it needs genuinely high-frequency (full-res) data.

### High-k capacity test on the flowers-native MLP — N-collapse replicates, but the specific high-k prediction FAILS (2026-08-17)

Ran the capacity test on the flowers-NATIVE MLP (D=3072, RGB 32×32, its own θ₀; job 830630, gelu).
- **Replicates (robustness):** the **N-collapse** (LoRA margin peaks ~N=8 then decays +0.030→+0.008 as
  N 8→64; retrieval → below chance at N=64) and the **rank-climbing** (margin rises with r: −0.001 at
  r=1 → +0.033 at r=16) both hold on a genuinely different, harder dataset with a native base model.
- **Fails the specific "high-k binds capacity earlier" prediction (honest negative):** the data
  intrinsic-dim proxy came out **backwards** — `eff_rank(X)` at N=64: **MNIST 46.3 > flowers32 37.3**.
  32×32-downsampled flowers are *lower* effective-rank than MNIST (they reduce to correlated
  "colored-blob-on-green"; 10 digit shapes spread wider), so kN does not tighten faster and the margin
  decay is **nearly identical to MNIST** (+0.033→+0.006). The capacity ceiling did not bind distinctly
  earlier. The only clear flowers difference is far lower reconstruction quality (ssim_norm 0.09 vs
  0.48) — the harder D=3072 ambient space, not the capacity term.
- **Takeaway:** the capacity ceiling is theoretically sound and the N-collapse is confirmed, but 32×32
  flowers are NOT high-k (proxy below MNIST), so this test cannot isolate the capacity effect. To see it
  bind earlier needs genuinely high-k data — the **D=12288 flowers64** model or full-resolution.

### Theory follow-ups — Lemma B (matched-wchg) + capacity/rank sweep (2026-08-13)

Two tests closing out [notes/linearization_leakage_theory.tex](notes/linearization_leakage_theory.tex).
Data: `results/linerr_matched_wchg.csv`, `results/gate_matrix_test.csv`, and the
`exp_b_T1_r*_..._npc*_vw5` capacity tensors. Jobs 671378 (Lemma B) + 671386 (capacity).

- **Lemma B at MATCHED weight_change (0.05, T=10) — half-confirmed, half-refuted (honest).** Function-
  space lin-error: **kinked ≫ smooth even at matched ‖δ‖** — relu 0.576 / leaky 0.562 vs smooth
  0.02–0.13 (softplus 0.070, gelu 0.026, sigmoid 0.020). So smoothness→linearization is **real, not a
  weight-change artifact** (this *corrects* the earlier guess that softplus's high lin-error was purely
  the ‖δ‖ confound — softplus > gelu **persists** at matched wchg). **BUT the fine ∝σ'' law is REFUTED
  within the smooth family:** gelu has the *highest* σ'' (0.78) yet the *lowest* lin-error (0.026);
  softplus low σ'' (0.25) but higher (0.070) — opposite of ∝σ''. Cause: the anchor metric is the
  *relative* Taylor residual (÷‖ΔΦ‖), so it tracks **σ''/‖∇Φ‖**, not σ''. softplus-β is U-shaped
  (b0.5 0.130 anomalous; b2 0.038 min; then b5 0.062 → b10 0.116 → b50 0.362 → relu). Lemma B needs
  the /‖∇Φ‖ correction.
- **Capacity / rank sweep (N=10, gelu vs relu, r∈{1..64}) — the LoRA-amplification mechanism confirmed.**
  **relu is rank-robust** (margin +0.051 / norm 0.57 already at r=1, flat across ranks) — its distinct
  binary gates survive even a rank-1 projection. **gelu is rank-climbing** (+0.014 / norm 0.44 at r=1,
  rising to a plateau by ~r16) — its collinear gates collapse under low rank and need more of it. The
  relu–gelu gap is largest at low rank. (The crisp "saturates at r≈rank(M)" was too clean; the
  rank-robust-vs-rank-climbing split is the real, same-mechanism signature.)
- **Capacity / large-N (gelu r=8, N∈{10,16,32,64}) — the superposition collapse.** Instance margin
  decays monotonically **+0.033 → +0.023 → +0.012 → +0.006** and retrieval falls to chance
  (0.10/0.12/0.03/0.02 vs 0.10/0.06/0.03/0.02) as N grows — leakage per-sample → 0 as kN eats the
  ρ(m+d) budget. (ssim_norm stays ~0.48 = shared background, not instance info.)
- **Net:** the two-ceiling picture is empirically supported — feature ceiling (gate rank, confirmed
  below), its rank-dependence (amplification), and the capacity/N collapse — with the honest Lemma-B
  refinement (relative lin-error is σ''/‖∇Φ‖).

### Gate-matrix test — the linearization-vs-leakage theory's feature ceiling CONFIRMED (2026-08-13)

Ran the falsifiable test from [notes/linearization_leakage_theory.tex](notes/linearization_leakage_theory.tex)
(§8): at θ₀, per activation, the gate matrix `M_ki=σ'(⟨w_k,x_i⟩)` + per-sample gradient features
`φ(x_i)`; measured effective rank, per-neuron gate variance/range, `mean|σ''|`, pairwise cosine.
Job 668832 (short-gpu, ~2 min). Data: `results/gate_matrix_test.csv`.

- **eff_rank(M), N=10 — kinked ≫ smooth, and it PREDICTS the measured leakage ordering:**
  relu 6.37 ≈ leaky_relu 6.33 ≫ **selu 3.39** > gelu 2.91 > mish 2.39 > silu 2.34 > softplus 1.73 >
  tanh 1.59 > sigmoid 1.19. The kinked units are ~4× higher rank than softplus, and the ordering
  matches the independently-measured LoRA leakage — including the earlier **selu surprise** (a C⁰ kink,
  rank 3.39, above every smooth C∞ unit → why it out-leaked them on Fashion/Flowers). The three with
  N=10 LoRA margins line up monotonically: relu (rank 6.37, +0.04) > gelu (2.91, +0.007) > softplus
  (1.73, ~0).
- **Softplus-β is a clean monotone dial of the whole frontier.** β 0.5→50: eff_rank(M) 1.40→5.30,
  gate_range 0.037→0.555, `mean|σ''|` 0.125→3.26, cos_M 0.999→0.869 — traversing smooth→ReLU
  (limit 6.37 / 0.585 / Dirac / 0.810). One-parameter confirmation that σ'' dials gate range dials
  rank(M) dials leakage.
- **The Dirac / total-variation point confirmed.** relu/leaky have autograd `mean|σ''|=0` (the kink's
  curvature is a Dirac autograd can't see) **yet the maximal gate range (0.585)** — their info is in the
  range, not pointwise σ''. cos_M: softplus **0.997** (collinear gates → rank≈1, entangled) vs relu
  0.81; cos_phi (full gradient features): softplus **0.85** (entangled → least leakage) vs relu/gelu
  ~0.39 (separable).
- **⚠ Lemma B caveat (honest).** `mean|σ''|` at θ₀ does NOT order like the measured linearization error,
  because lin-error ∝ σ''·‖δ‖² and the weight-change ‖δ‖ differs across activations (softplus trained
  ~2× more). The σ''→lin-error link holds only at matched weight_change — the same matched-wchg caveat
  as everywhere. The **feature-ceiling half is what is cleanly confirmed.**
- **Net:** softplus's `eff_rank(M)≈1` (collinear gates) is exactly why it entangles samples and leaks
  least on the adapter; kinked activations produce distinct (near-binary) gates → separable → leak. The
  theory's feature ceiling is validated; the capacity ceiling (rank vs data-dim) is the separate,
  untested-here bound.

### Step 1 first-pass results — activation rescore + LoRA-vs-full retrieval (2026-08-13)

Executing the approved coupled activation×anchor×linearization plan ([notes/next_experiment_plan.md](notes/next_experiment_plan.md)).
Step 1 (zero-GPU analysis) is done; the results already sharpen the direction.

- **Addition 2 first-pass — SOFTPLUS is the best activation, not GELU (rescore of job 857271).**
  Rescored the 21 on-disk activation tensors with the full metric suite + `weight_change` + control
  margin (`experiments/recompute_metrics.py`, now emits `weight_change`, `delta_w_effective_rank`,
  `ntk_passed`, `ctrl_margin`, and recovers `finetune_activation` from the filename). **At matched
  `weight_change`≈0.04, softplus wins on every metric**: ssim_norm 0.65, ssim11 0.49, **l2 4.8 (vs
  ~18 for gelu — a ~4× gap)**, control margin **+0.115**, and it clips least (44% vs ~58%; its raw
  recon stays in [−0.49, 1.32] not saturating to ±1). Ranking is unanimous across 7 metrics:
  **softplus ≫ silu > gelu ≈ gelu_tanh > mish > elu** (elu has a *negative* margin). `feature_stability`
  ranks identically (softplus **0.993** > silu 0.976 > gelu 0.965 > mish 0.947 > elu 0.808) — proof
  criteria 1 (NTK survival proxy) and 2 (leakage) **co-move**, exactly as the crux predicts.
  **Softplus is uniquely linearization-stable:** its reconstruction is invariant to **1.7e-4** across a
  10× `weight_change` range (0.038→0.379) then breaks at 1.14, whereas gelu's shifts ~0.26 over the
  same LRs. **⚠ Caveat (do not overclaim):** *all* 23 configs are `ntk_passed:False` with
  `delta_w_effective_rank = 1–2` (degenerate, sub-NTK), single seed (42), single N (2), single dataset
  (MNIST), oracle-coefficient. This is a strong **directional** signal, not a confirmed result — it is
  what Step 2a re-runs in-regime (target `weight_change`→`ntk_passed:True`, multi-seed, the softplus-β
  knob, and the never-run kinked controls relu/tanh/hardswish). Data: `results/rescored_activations_857271_2026-08-11.csv`.
- **QW2 — LoRA-vs-full retrieval, now a durable artifact (was prose-only).** `retrieval_metric.py`
  now writes a CSV + pooled significance test + figure. Over the N-sweep (N=4..32 × 3 seeds):
  **LoRA leaks ~2× chance (NCC pool obs 26 vs exp 12, z=4.30, p=8.5e-6 — reproduces the earlier prose
  exactly)**; **full-model leaks ~3× chance (z=8.3, p=5e-17)**. So gate **B1 = yes (weak)**: the adapter
  leaks instance-level info, roughly half as strongly as the full model. Pixel-space and the θ₀
  classifier-feature space stay near chance for LoRA (z≈0.9–1.2); only the background-robust NCC/SSIM
  rankings carry the LoRA signal. Data: `results/retrieval_lora_vs_full_2026-08-11.csv`,
  figure `figures/retrieval/retrieval_lora_vs_full_2026-08-11.png`.
- **QW3 — newest tensors under the standard bar.** Anchor N=10 (957044): full-FT control margin peaks
  at **α=0.75 (+0.132)**, confirming α*≈0.75 at N=10 under the control-margin bar (previously raw SSIM
  only); LoRA margins are tiny (+0.006–0.008) — adapter leakage is very weak at N=10. DI large-N
  (887704): `ssim_norm` confirms the N-collapse (N=4 0.60 → N=10 0.26 → N=20 0.24). **Gap found: DI
  `.pth` files save no `x_ctrl`**, so the +0.049/+0.058 DI control margins in the 2026-07-22 box are
  *not* re-derivable from disk (they were runtime-only); `direct_inversion.py` should save `x_ctrl`.

### Step 2 batch — MNIST confirms softplus; harder-data transfer is inconclusive (2026-08-13)

First results from the coupled-study batch (7 long-gpu jobs). Rescored via `recompute_metrics.py`
(`results/rescored_batch_2026-08-13.csv`), read through the control-margin bar.

- **N×lr ablation (job 483935, MNIST, gelu vs softplus) — SOFTPLUS WIN CONFIRMED, clean.**
  control margin, matched testbed:
  - softplus N=2 = **+0.115 at every lr** (0.005→0.1) while weight_change scales 0.019→0.379 — i.e.
    **perfectly lr-invariant** (the linearization-stability signature, now reproduced cleanly on a grid).
    gelu N=2 = **+0.02** (≈5× worse) and is *not* lr-invariant.
  - Leakage **decays with N** for both (softplus +0.115→+0.050→+0.023→+0.004 for N=2→4→8→16;
    superposition), so the softplus advantage is largest at small N.
- **Harder data via MNIST-θ₀ TRANSFER (Fashion job 482018, Flowers-28×28 job 484480) — DOES NOT cleanly
  transfer, but the test is confounded/degenerate, NOT a refutation.**
  - Ranking *flips*: kinked activations (selu, relu, leaky_relu) top the list; softplus lands mid-pack
    or **last** (Fashion N=10 softplus +0.047 = worst; Flowers N=10 +0.043 near-worst).
  - **BUT Fashion N=2 is degenerate**: 11/13 activations have **weight_change = 0.000** (the MNIST net
    gives a ~zero BCE gradient on Fashion at one step → saturated logits) → identical trivial output
    (+0.253 for all); softplus is one of only two activations that produced a *nonzero* update at all.
    Fashion/Flowers cells are also **not at matched weight_change** (0.006–0.135) and mostly
    `ntk_passed:False`. So the "kinked wins on harder data" read is confounded by the transfer setup,
    not a clean result.
  - **Interpretation:** softplus's advantage is a solid *clean-MNIST-testbed* result; whether it
    survives on genuinely harder data is **unresolved** — the downsampled-flowers + MNIST-θ₀ transfer
    proxy is too degenerate to tell. **This is exactly why the flowers-NATIVE θ₀ track (a model trained
    on flowers, run by the parallel session) is the right test** — real gradients + matched-weight_change.
- **Follow-up queued:** matched-weight_change harder-data re-run (higher LR to escape the wchg≈0
  degeneracy on the transfer setup) — `scripts/run_harder_matched_wchg_wexac.sh`.

### Step 2b anchor two-curves (per activation) — the "softplus wins" headline is NUANCED, not clean (2026-08-13)

The anchor α-sweep run for softplus/silu/gelu/relu (N=2) + softplus/gelu/relu (N=10) at T=10 — the
first time the anchor sweep has been run for any activation besides GELU. Read via **control margin**
(raw SSIM is background-dominated) and the **function-space lin-error** curve.

- **Smoothness → LINEARIZATION holds (robust, clean):** function-space lin-error at α=0 ranks
  **silu 0.006 < gelu 0.008 < softplus 0.028 ≪ relu 0.30** — the kinked baseline linearizes ~40×
  worse than the smooth ones. But **within the smooth family softplus is NOT the best linearizer**
  (silu/gelu are), so the Step-1 hypothesis "the recon winner also linearizes best" is **false**.
- **Smoothness → LEAKAGE does NOT hold — the key dissociation.** On the **LoRA path** (the thesis
  target), control margin at N=2: **relu is HIGHEST across all α (+0.123 → +0.233)**, softplus middling
  (+0.06 → +0.11), gelu/silu low at α=0 (+0.03) rising to +0.14 only with the anchor. At N=10 the LoRA
  margins are: **relu +0.035–0.049 > gelu +0.007 ≈ softplus ~0 (softplus barely leaks on LoRA at N=10)**.
  So the *kinked* relu leaks the most on the adapter path, despite linearizing the worst. Linearization
  quality and leakage are **different axes**.
- **⚠ The "softplus wins" from Step 1 was premature/incomplete:** it (a) EXCLUDED relu (the 857271
  CONTROL_SET never ran), (b) was T=1 only, (c) was at a single matched weight_change. With relu
  included and at T=10, softplus is NOT the LoRA-leakage winner.
- **CONFOUND RESOLVED (from the job logs) — relu genuinely out-leaks the smooth activations on LoRA.**
  Fine-tune weight_change (per activation, T=10, lr=0.01): **N=2 LoRA — softplus 0.183, relu 0.165,
  silu 0.165** → essentially MATCHED, yet relu's control margin (+0.12→+0.23) beats softplus (+0.06→+0.11).
  **N=10 LoRA — softplus 0.125 vs relu 0.072, gelu 0.069** → softplus trained ~2× MORE yet leaks LESS
  (softplus margin ~0 vs relu +0.035–0.049). So the weight_change confound runs *against* softplus, and
  relu still wins — the kinked-relu adapter-leakage advantage is **real, not a training-amount artifact.**
  A matched-weight_change anchor sweep is therefore **not needed** to settle this.
- **The anchor's value is ACTIVATION-DEPENDENT (a clean result that engages Gal's α idea):** the
  anchor (α≈0.75) *rescues* gelu/silu (full-FT margin +0.18 → +0.28, LoRA +0.03 → +0.14) and helps relu,
  but **softplus is anchor-independent** (flat in α — already recovers at α=0). Combined with the
  N×lr ablation (softplus lr-invariant), softplus's robust signature is **linearization-STABILITY**,
  not peak leakage. The α*≈0.75 full-FT peak replicates for gelu/silu (0.28 margin at 0.75, collapse at 0.9).
- **Honest net:** there is **no single winning activation**. Smoothness → better linearization (clean);
  smoothness → more leakage is **unsupported** (relu leaks most on LoRA). The durable findings are the
  dissociation itself + softplus's linearization-stability, pending the matched-weight_change anchor
  sweep to settle the relu-vs-smooth leakage question. Data: `results/rescored_batch_2026-08-13.csv` +
  the `anchor_sweep_T10_r8_*_s42*.pth` tensors.

### Status review — final job outcomes from the Jul 23–26 batch (2026-08-11)

All four late-July jobs are finished; **nothing is running on WEXAC** (idle since 2026-07-26).
Recorded from the logs (`scripts/wexac_logs/`):

- **Job 956997 (full-model N-sweep reference) — ✅ completed 2026-07-23.** The full-model reference
  for the retrieval metric at larger N now exists (`results/exp_b_T1_full_s{42,43,44}_a149_npc{2..16}.pth`).
  At npc=16 (N=32): full-model SSIM 0.22–0.26 vs control 0.20–0.23 — small raw margins, as expected
  in the meaningful-mean regime. **The retrieval analysis on these full-model runs has NOT been run
  yet** — that's the open item, and `retrieval_metric.py` now also supports a base-classifier
  (θ₀ penultimate-feature) ranking space via `--classifier` for it.
- **Job 957044 (anchor α-sweep at N=10, vw=5, seed 42, T=10) — ✅ completed 2026-07-23.** Full-FT:
  α=0 0.463 → α=0.75 **0.497** (peak) → α=0.9 0.440 — the **interior optimum at α≈0.75 persists at
  N=10**, much compressed. LoRA r8: rises monotonically 0.096 → 0.173 (α=0.75) → 0.176 (α=0.9) —
  **no α=0.9 collapse on the LoRA path at N=10**. Absolute SSIMs are far lower than at N=2
  (superposition, consistent with the DI N-scaling collapse). Control-margin / mean-baseline rescore
  of these tensors (`results/anchor_sweep_T10_r8_gelu_s42_N10_vw5.pth`) still pending.
- **Job 857271 (trustworthy activation×LR re-run) — ⚠ hit the 96 h RUNLIMIT 2026-07-26.** Partial
  but substantial: 68 configs produced SSIMs, through ADD2a `elu lr=0.3 T=1`. Resumable via
  `--skip_if_exists`; the remainder was never resubmitted. Results not yet analyzed.
- **Job 863020 (anchor multi-seed replication) — ⚠ hit the 96 h RUNLIMIT 2026-07-26.** The seed-43/44
  and T-sweep stages that matter completed before the limit and their findings are already recorded
  in Track 1 below (full-FT α*≈0.75 replicates; "anchor creates LoRA leakage" was seed-42-specific).

**Next steps (priority order):** (1) run `retrieval_metric.py --classifier` over the full-model
npc results from 956997 — this completes the retrieval story (LoRA-vs-full at matched N);
(2) rescore the N=10 anchor tensors with control margins; (3) decide whether the 857271 remainder
is still worth compute given the metric-audit conclusions.

### Retrieval metric — LoRA DOES leak instance-level info that SSIM missed (2026-07-23)

Follow-up to the metric audit. Since absolute SSIM on MNIST is background-dominated, added an
**instance-level retrieval metric** (`experiments/retrieval_metric.py`, 11 tests): among the N
training images, is reconstruction *i* most similar to target *i*? Background cancels (common to all
candidates); random baseline is 1/N, so it strengthens with N. Distances: pixel-L2, NCC, SSIM, and a
pluggable classifier-feature space.

**Result (LoRA r=8, T=1, N=4..32 via the N-sweep, NCC/SSIM ranking):** retrieval top-1 is
**consistently ~2.0–2.3× the random baseline** at every N (e.g. N=32: ~0.07 vs 0.031 chance).
Pooled across N and 3 seeds: **26 correct vs 12 expected by chance, z=4.3, one-sided p≈8.5e-6.**
Pixel-L2 ranking stays near chance; the background-robust NCC/SSIM rankings carry the signal.

**Why it matters:** absolute SSIM said single-step LoRA was "below the trivial baseline → nothing."
Retrieval shows it **does** leak instance-level information — statistically significant, consistent
across seeds, and surviving to N=32. This **flips the preliminary gate-B1 read**: the earlier
"LoRA doesn't beat trivial" was a *metric* artifact, not the absence of leakage. The leakage is
weak (~2× chance, not high-fidelity recovery) but real. Caveats: small per-cell counts (significance
comes from pooling); the full-model reference at larger N (N-sweep full runs had a `--no_baseline`
script bug, fixed and resubmitted as job **956997**) **completed 2026-07-23 — tensors saved, but the
retrieval analysis over them is still pending** (see the 2026-08-11 status review above).

### Metric audit — most single-step LoRA reconstructions sit at/below the trivial baseline (2026-07-22)

Chasing a metric anomaly (identical GELU config: SSIM 0.358 at 5 extraction epochs, 0.041 at
50,000) led to a metrics overhaul (`experiments/metrics.py`) adding: **`ssim11`** (window=11,
literature/SimuDy-comparable — ours was window=3), **`ssim_norm`** (mean/std-matched, scale
invariant), **`ssim_mean_baseline`** (SSIM of the trivial `ds_mean` predictor), and **clipping
diagnostics**. New offline re-scorer `experiments/recompute_metrics.py` re-scores saved `.pth`
tensors with no GPU (→ `results/metrics_recomputed.csv`). Tests: `experiments/tests/test_metrics.py`
(9, all pass) + `test_activations.py`.

**Headline (rescore of 76 historical reconstructions):**
- **65/76 score BELOW the `ds_mean` trivial baseline** on raw window-3 SSIM.
- **full-model:** 10/28 beat baseline (mean SSIM 0.728 vs baseline 0.758) — the clean full-model
  runs (0.96–0.99) genuinely and clearly beat it; those results stand.
- **LoRA:** **1/48 beat baseline** (mean 0.479 vs 0.753) — on raw SSIM, single-step LoRA
  reconstructions mostly do **not** demonstrably exceed "just output the dataset mean."

**Two confounds the new metrics expose (they pull opposite ways):**
1. **Clipping** — smooth-activation runs clip 40–60% of pixels (`x_recon` pinned at ±1, then
   `+ds_mean` exceeds [0,1]); raw SSIM *understates* them. `ssim_norm` recovers +0.12–0.16
   (gelu 0.041→0.198, silu 0.341→0.559, softplus 0.497→0.650).
2. **Inflated baseline** — with N=2 and MNIST's mostly-black background, `ds_mean` already scores
   ~0.76 against each digit, so raw SSIM *overstates* how impressive any single number is.

**Interpretation (with caveats).** Raw SSIM on MNIST-N=2 is a poor leakage discriminator: both the
absolute values and the baseline are background-dominated (the same effect we flagged for SimuDy,
now measured on our own data). This does **not** prove the reconstructions are worthless — a recon
can carry instance detail `ds_mean` lacks while both share the black background — but the historical
"LoRA r≈8 SSIM ~0.6–0.8" numbers must be reported **against the baseline**, not alone. The
full-vs-LoRA gap is real signal: **LoRA recovery (the thesis target / gate B1) does not yet clearly
beat trivial on this testbed.**

**Next:** report `ssim_norm` + baseline + clip everywhere; move to larger N / harder data so the
baseline isn't ≈ each image; add a background-robust identifiability/retrieval metric. Trustworthy
re-run was job **857271** (`scripts/run_gal_additions_sweep.sh`, resumable via `--skip_if_exists`) —
it hit the 96 h RUNLIMIT on 2026-07-26 with 68 configs done (see 2026-08-11 status review).

### Gal's missing missions built: Addition 3 + DI-Phase 0 + GB-Phase 1 (2026-07-22)

Three tracks Gal asked for that job 435843 did **not** cover — each needed new code — are now
built, locally validated, and running on WEXAC (`long-gpu`).

> **⚠ Mean-baseline re-scoring (2026-07-22) — read before quoting any SSIM below.** We re-scored the
> saved reconstructions against `ssim_mean_baseline` (what the trivial dataset-mean predictor scores)
> using the new `experiments/metrics.py`. Results in `results/rescored_metrics_2026-07-22.csv`. **Small N
> makes the mean a *hard* bar** (N=2 → baseline 0.763; N=4 → 0.674), and clip fractions are high
> (0.37–0.67 of pixels saturate out of [0,1], inflating raw SSIM). Verdicts:
> - **Direct inversion: real but WEAK leakage that DEGRADES with N (the superposition problem).**
>   At N=4 it fails the mean baseline (0.43–0.58 vs 0.674) but passes the control test (+0.17). At
>   **N=10 (job 887704) the reconstruction collapses to SSIM 0.27** and the mean-baseline gap widens
>   (0.27 vs 0.564), yet the control margin **survives, shrunken: +0.049 (T=1), +0.058 (T=10).** So DI
>   does recover *some* instance-specific info, but joint inversion of more images degrades it fast —
>   it does **not** scale. (Falsifies the earlier guess that larger N would clear the mean bar; larger N
>   made it worse.) Method is sound (endpoint matching converges, T=1 exact); the toy just doesn't
>   disentangle many images at once. **Confirmed (job 887704): SSIM falls monotonically with N —
>   N=4 ~0.55, N=10 ~0.27, N=20 ~0.15–0.18 — and tightening the pixel box (box=5) does NOT rescue it
>   (0.13/0.27, mixed), so the collapse is fundamental joint-inversion difficulty, not clipping.** DI is
>   a very-small-N method; the superposition problem is the wall.
> - **Anchor full-FT: REAL leakage for α ≤ 0.75** (beats 0.763; structure-only `ssim_norm` agrees,
>   0.83–0.96; α=0.75 = 0.94/0.97/0.96). The α*≈0.75 finding **holds up under the strict metric.**
> - **Anchor LoRA-only (r=8): does NOT beat the 0.763 baseline at any α** (best 0.64 at α=0.75). A clear
>   monotone trend, but **no distinguishable instance leakage** — consistent with the decision brief's
>   B1 "yellow flag." Bears on the thesis: LoRA-adapter-only recovery is not yet demonstrated here.
>
> Net (mean-baseline view): the strict-baseline headline is the **full-FT anchor α≈0.75 win**.
>
> **BUT the mean baseline is arguably the wrong bar.** The decision brief's B1 gate is "beat the
> **same-class control**" — the instance-leakage question ("did we recover *this* image or just
> something class-typical?"). Scoring recon-vs-true minus recon-vs-control (seed 42, T=10, both share
> the same clipping/scale so the *margin* is clean):
> - **Full-FT: instance leakage at every α** (+0.18 to +0.28; strongest at α=0.75, +0.28).
> - **LoRA-only: real instance leakage (B1 signal holds), but α-dependence is config-specific.** On
>   seed 42 the anchor appeared to *create* leakage (+0.03 at α=0 → +0.14 at α=0.75). **Replication
>   tempered this** (see Track 1 below): seed 44 shows LoRA leaking already at α=0 (+0.18) with the anchor
>   *hurting*. So adapter-only leakage is genuine (control margins +0.13–0.18 across configs → **B1
>   passes**), but "the anchor creates it" was not robust. Green light for the LoRA-leakage *direction*;
>   the α-story needs a multi-config study.
>
> Reconcile: two different questions. Mean baseline = "beat predicting the dataset mean" (hard at tiny
> N). Control baseline = "match the true image better than a different same-class image" (the actual
> leakage test; robust to clipping). **The control-margin is the right headline for LoRA; report both.**
> Larger-N runs (jobs launched) will confirm under the meaningful-mean regime. The tracks below report
> raw SSIM — always pair with these baselines.

**Track 2 — DI-Phase 0 (direct weight inversion) — ✅ demonstrated (job 500913).**
`θ_T = F(θ₀, x̂)` with autograd through an unrolled SGD `F`; MNIST MLP, N=4, LoRA r=8, GELU.
- **SSIM-vs-T:** T=1 **0.57**, T=2 0.57, T=5 0.53, T=10 **0.58**, T=20 0.43. Recovers digits at
  SSIM ~0.55, **stable through T=10**, degrading at T=20.
- The differentiable `F` is **bit-exact** at T=1: `F(x_true)` reproduces the true θ_T to 0.0 and the
  endpoint loss / input-gradient are exactly 0 at the truth (reachable optimum).
- **T=1 equivalence holds:** DI-T1 0.57 ≈ Experiment-B oracle NTK T=1 (~0.50 from job 435843), and
  DI is *slightly above* — expected, since DI matches endpoints exactly while NTK is linearized.
- Artifacts: `results/direct_inversion_N4_r8_gelu.pth`, `figures/direct_inversion/di_ssim_vs_T_*.png`
  + per-T grids. Code: [experiments/direct_inversion.py](experiments/direct_inversion.py),
  [scripts/run_di_phase0_wexac.sh](scripts/run_di_phase0_wexac.sh).

**Track 1 — Addition 3 (anchor α-sweep + two-curve validation) — ✅ demonstrated (job 532232, T=10, seed 42).**
`θ_anchor(α) = (1−α)θ₀ + αθ_T`, α ∈ {0,0.25,0.5,0.75,0.9}, on **both** full-FT and LoRA paths.
- **Headline — an interior optimum at α≈0.75, confirming Gal's tradeoff:**

  | path | α=0 | 0.25 | 0.5 | **0.75** | 0.9 |
  |------|-----|------|-----|----------|-----|
  | full-FT SSIM | 0.801 | 0.807 | 0.815 | **0.939** | 0.484 |
  | LoRA r8 SSIM | 0.063 | 0.334 | 0.404 | **0.643** | 0.560 |
  | full lin-err (fn-space) | 0.008 | 0.004 | 0.002 | 0.001 | 0.000 |
  | LoRA lin-err | 0.192 | 0.206 | 0.165 | 0.087 | 0.034 |

  SSIM rises with α as the linearization error falls (dramatic on LoRA: **0.06→0.64**), peaks at
  **α≈0.75**, then **collapses at α=0.9** (full-FT drops to 0.48, *below* the α=0 baseline). The
  collapse is the identifiability-degradation regime Gal predicted — past ~0.75 the anchor absorbs
  θ_T's training signal. SSIM peaks *before* the lin-error minimum (not after), so this is a legit
  linearization win, **not** the x_i-leakage red flag. Gal's midpoint helps; **α*≈0.75** is the sweet
  spot; cap below 0.9.
- **α=0 reproduces the current baseline bit-for-bit** (verified offline). Only the *linearization
  point* moves (features + recomputed coefficients at θ_anchor); the reconstruction still matches the
  full observed Δw. Headline metric: **function-space** Taylor residual; weight-space kept as companion.
- Deliverables: `figures/anchor_sweep/anchor_two_curve_{full,lora}_*.png` + per-α grids +
  `results/anchor_sweep_T10_r8_gelu_s42.pth`. Code: `--anchor_alpha` in
  [run_experiment_b.py](experiments/run_experiment_b.py),
  [experiments/run_anchor_sweep.py](experiments/run_anchor_sweep.py),
  `compute_function_space_lin_error` in [ntk_verification.py](experiments/ntk_verification.py).
- **Replication (job 863020) — full-FT robust; LoRA inconsistent.** (Seed 43 drew [1,0], the two
  easiest digits → Δw≈0 → lin-error 0.0000, byte-identical output at every α → degenerate/discarded; the
  recurring "check weight_change first" trap. Use non-trivial digits: 44–50 are all fine.)
  - **Full-FT: α*≈0.75 replicates.** seed 44 [8,3] peaks 0.965@0.75 (control margins +0.27→+0.43, strong
    leak at every α); seed 42 T=5 peaks 0.934@0.75 then collapses at 0.9. Across seeds/T the full-FT
    anchor gives strong instance leakage peaking ≈0.75 — **this is the solid, replicated headline.** (The
    α=0.9 collapse is itself seed-dependent: seed 42 collapses, seed 44 stays high at 0.925.)
  - **LoRA: leakage is present but the *anchor's effect is not consistent.*** Seed 42 T=10 = anchor
    *creates* leakage (margin +0.03→+0.14 as α↑). Seed 44 T=10 = LoRA already leaks at α=0 (SSIM 0.42,
    margin +0.18) and the anchor *hurts* (→+0.14, SSIM↓). Seed 42 T=5 = leak only at α=0.9 (+0.13). So
    LoRA-adapter instance leakage **does occur** (control margins +0.13–0.18 across configs → **B1 signal
    holds**), but the earlier "anchor creates LoRA leakage" claim was seed-42-specific, **not robust.**
    Honest LoRA takeaway: adapter-only leakage is real but config-dependent; needs a proper multi-config
    study (pinned hard digits × several seeds × T) before any α-dependence claim.

**Track 3 — GB-Phase 1 (Gradient Bridge decoder) — ✅ first result (job 532180), higher-rank arms running.**
`f_φ:(A,B)→∇_W L` on a public MNIST proxy, single-step LoRA measurements, cosine loss.
- **Honest first result (r=8, reported as full-cosine vs the col(B₀) projection ceiling):**
  - **Hidden layer (layers.1, the real milestone):** decoder full cosine **0.685** vs projection
    ceiling **0.086** — the decoder recovers the gradient **~8× above** what the measurement subspace
    affords, i.e. it genuinely *hallucinates the out-of-subspace component from the proxy prior* (the
    R2F claim, now shown for vision).
  - **Rank sweep (r=8/32/64) — a clean finding:** decoder full-cosine is **flat at 0.685** across all
    three ranks, while the measurement ceiling rises 0.086→0.178→0.252 (exactly √(r/1000)). So the
    decoder beats the ceiling at every rank, but **higher rank does not help** — the recovery is
    prior/decoder-limited, not measurement-limited. Rank is **not** the lever to reach 0.9; the next
    levers are decoder capacity, nonzero-A (two-sided) measurement, or multi-sample gradients.
  - **Improvement arms (job 956994) — the >0.9 milestone is REACHED.** bigger decoder (hidden 2048,
    depth 3) = **0.687** ≈ baseline (capacity is NOT the limit — the single-sample plateau is
    information-limited); two-sided measurement (nonzero-A → observes col(B₀)⊕row(A₀)) = **0.794**
    (ceiling rose 0.087→0.124, decoder exploits it); **multi-sample / realistic batch gradient** (m=8,
    target = rank-8 batch gradient) = **0.951 — clears >0.9** at an unchanged 0.086 ceiling (11× above
    it). Real fine-tuning uses *batch* gradients, so this is the more realistic AND far more decodable
    target than a single-sample rank-1 gradient. **Verdict: the R2F-style gradient bridge works at
    milestone level for vision (0.95) — the strongest GB result of the project.**
  - **Output layer (layers.2):** decoder full cosine **0.94** but projection ceiling **1.0** — the
    decoder is *below* trivial analytic inversion, so ">0.9" here is **weak evidence** (near-analytic,
    out=1). The dual-cosine report exposes this cleanly.
- Risk-fixes that made the numbers honest: **gradient-norm filtering** of degenerate near-zero-gradient
  pairs (output-layer ceiling went 0.865→1.000 after filtering), rank-1 factored storage (no 200 GB
  blowup), stable QR projection (gram pinv blows up when out < r).
- Risk-fixes baked in: rank-1 factored storage (no 200 GB blowup), fresh B₀ per pair, and
  **gradient-norm filtering** (confidently-classified proxy samples give ≈0 gradient → useless pairs).
- Code: [experiments/gradient_bridge/](experiments/gradient_bridge/) (generate_pairs / decoder /
  train_decoder), [scripts/run_gb_phase1_wexac.sh](scripts/run_gb_phase1_wexac.sh).

### SimuDy collision + Gal's Additions launched (2026-07-21)

**Novelty collision found (resolves the Part D literature search).** Gal sent
[SimuDy (Tian et al., ICLR 2025)](papers/Tian_2025_SimuDy_Simulating_Training_Dynamics_ICLR.pdf) —
*"already showed an idea that we discussed."* It publishes our direct-weight-inversion primitive:
unroll SGD through dummy data, match `θ_f−θ₀` by cosine sim + TV. **The full-FT direct-inversion
headline novelty is taken.**

- **What it does NOT do (= what remains ours):** no LoRA/PEFT anywhere; no identifiability/stability
  theory; brute-force full-unroll only (**22 GB / 15 h for 120 CIFAR-32² imgs on ResNet-18**; ViT
  result is **N=10** only). Their numbers: MLP/100 SSIM 0.337, ResNet/50 **0.198**, ResNet/120 ~0.12.
- **Decision: reframe, don't abandon.** Cite SimuDy as baseline + feasibility de-risker; re-center on
  (i) LoRA-adapter-only leakage, (ii) identifiability/anchor-α theory, (iii) memory-tractable
  (linearized) inversion. Gated on two cheap tests — **B1** (adapter-only recovery at all) and **B2**
  (linearized anchor can replace full unroll).
- **Full analysis:** [notes/simudy_decision_brief.md](notes/simudy_decision_brief.md) (1→N chain) ·
  paper teardown: [notes/related_work_simudy.md](notes/related_work_simudy.md)

**Gal's meeting Additions — job `435843` submitted to `long-gpu`**
(`scripts/run_gal_additions_sweep.sh`, priority-ordered so the top ask lands first):

| Ask | Status |
|---|---|
| Addition 2 — smooth activations (GELU top priority) | ⏳ running (Stage 1). **Required a code change** — `gelu`/`silu`/`softplus` did not exist in `ACTIVATION_CHOICES` |
| Addition 1 — more LoRA samples / breadth | ⏳ running (Stage 2): `n_per_class` 1–4 × seeds 42/43/44 |
| Loss ablation l2 vs cosine | ⏳ running (Stage 3) — extra relevant: SimuDy reports cosine ≫ Euclidean |
| Addition 3 — anchor α-sweep | ☐ **blocked on code** — no `--anchor_alpha` flag exists yet |
| Gradient Bridge GB-Phase 1 (decoder) | ☐ **blocked on code** — no decoder/bridge code exists at all |

### Next Direction (post-2026-05-14 meeting): Direct Weight Inversion — PLANNED

After the first supervision meeting with Gal Vardi, the **new primary research axis is direct weight
inversion** — recover the fine-tuning samples by inverting the deterministic map `θ_T = F(θ₀, {x_i})`
(minimize `‖θ_T − F(θ₀, {x̂_i})‖²`). It is **complementary to** the Gradient Bridge, not a replacement.
Three concrete meeting additions accompany it (varied-data LoRA breadth; smooth-activation / GELU sweep;
anchor α-sweep with a two-curve validation protocol). **Status: proposed / planned** — only an
Approach-G / S3.4 sketch exists; nothing run yet.

- **Actionable to-do (single source of truth):** [notes/experiment_plan.md](notes/experiment_plan.md) (PDF reading copy: [notes/experiment_plan.pdf](notes/experiment_plan.pdf))
- **Direction rationale, taxonomy, concerns:** [notes/unified_direction_analysis.md](notes/unified_direction_analysis.md) → "Direct Weight Inversion — New Primary Axis"
- **Full briefing:** [notes/thesis_update_briefing.md](notes/thesis_update_briefing.md)

> **Phase-naming note (avoids collision):** the completed "Phase 0" below is the **ViT-gate** track
> (full-gradient inversion). The new direct-inversion phases are labeled **DI-Phase 0…3** and the
> Gradient Bridge decoder phases **GB-Phase 0…2** in experiment_plan.md — three distinct tracks, not the
> same Phase 0.

### Sprint 2c: KKT & NTK Reconstruction Ablations — COMPLETE

Comprehensive ablation study across two tracks. 148+ configs completed.

**Track A: Experiment A (KKT) — CLOSED (negative result confirmed)**
- 15/48 configs completed before 48h timeout (job 583398). KKT loss stuck at 330-350 for ALL N values tested.
- Confirms Sprint 1 structural analysis: composed model W=W₀+BA satisfies KKT over all ~502 samples. No amount of N tuning overcomes the pre-training residual.
- **This definitively closes the KKT approach for composed models.**

**Track B: Experiment B (NTK) — Ablations**
- B1: Phase 3+4 (LR scheduling + warm-start) — **DONE** (results in sprint2b_phase3/4 CSVs)
- B2: Loss ratio ablation (verify_weight) — **DONE** (16 configs, results/sprint2c_track_b2_*.csv)
- B3a: Optimizer × activation for LoRA — **DONE** (results/sprint2c_track_b3a_*.csv). Winner: **SGD + LeakyReLU** (SSIM 0.830 for both r=8 and r=32)
- B3b: Scale best combo across T — **DONE** (SGD+LeakyReLU matches L-BFGS for T≤20, NaN at T=100)
- B4: N sweep (NTK) — **DONE** (results/sprint2c_track_b4_*.csv)
- B5-B8: Additional ablations — **DONE** (results in sprint2c_track_b5/b6/b7/b8 CSVs)

### Phase 0 (ViT-gate): ViT Gradient Inversion Gate — D2 COMPLETE, GATE CROSSED

> *Naming:* this is the **ViT-gate** "Phase 0" (full-gradient inversion, taxonomy row 1) — distinct from
> the new **DI-Phase 0** (direct-weight-inversion toy) and **GB-Phase 0** (Gradient Bridge scaffold) in
> [notes/experiment_plan.md](notes/experiment_plan.md).

Critical gate experiment: can gradient inversion reconstruct images from exact ViT-B/16 gradients?

**Status: GATE CROSSED.** D2 sweep (2026-04-28) found tv=1e-1 + lr=0.05 + 30K iters achieves **SSIM=0.548, PSNR=15.11, cos_sim=0.955** on Flowers102 — a 3.8× improvement over D1's best (0.144) and well past the 0.3 SSIM gate. 7/29 configs cleared the gate, all at tv=1e-1. The thesis can proceed to LoRA-only inversion, multi-seed validation, and the Gradient Bridge decoder.

**Run history:**
1. **2026-03-27**: SSIM=0.015. Failed due to 4 bugs. All fixed.
2. **2026-04-07**: Bug fixes applied. Vague structure visible. Non-standard SSIM metric (don't cite).
3. **2026-04-09**: Code audit fixed 6 issues (SSIM metric, backward, signAdam, TV norm, clamping, LoRA dims). Switched to Flowers102.
4. **2026-04-10**: signAdam bug found (was SignSGD). cos_sim=0.97 but SSIM=0.008 (noise). After fix: SSIM=0.022 full / 0.009 LoRA.
5. **2026-04-14**: D1 controlled comparison — 4 configs, same image/gradient.

#### D1: Controlled Optimizer × TV Comparison — COMPLETE (2026-04-14)

4 configs on the SAME Flowers102 image (seed=42), full-model gradient (86M params), 8 restarts × 10K iters each:

| Config | Optimizer | TV weight | SSIM | PSNR | cos_sim | Time |
|--------|-----------|-----------|------|------|---------|------|
| A | Adam | 1e-4 | 0.030 | 8.7 | 0.920 | 3.0h |
| B | signAdam | 1e-4 | 0.020 | 8.2 | 0.934 | 1.8h |
| C | Adam | 1e-2 | 0.090 | 9.8 | 0.887 | 1.7h |
| **D** | **signAdam** | **1e-2** | **0.144** | **10.9** | **0.933** | **3.0h** |

**Key findings:**
1. **Strong TV (1e-2) is essential.** Both strong-TV configs (C, D) beat both weak-TV configs (A, B) in SSIM. tv_weight=1e-4 is 100× too weak at 224×224.
2. **signAdam beats Adam at every TV level.** D > C by 60% (0.144 vs 0.090), B ≈ A at weak TV. signAdam maintains high cos_sim (0.93) even with strong TV drag.
3. **cos_sim alone is misleading.** Config B has highest cos_sim (0.934) but worst SSIM (0.020). Config D has similar cos_sim (0.933) but 7× better SSIM. The TV prior makes the difference.
4. **signAdam convergence is faster and tighter.** Cos_sim overlay shows signAdam restarts cluster tightly (0.920-0.934) while Adam restarts spread widely (0.465-0.920).

**Go/no-go outcome:** Config D SSIM=0.144 is just below the 0.15 gate threshold. **Proceeded to D2** (see below) — D2 crossed the gate decisively at SSIM=0.548.

**Instrumentation added (2026-04-14):**
- `best_cos_sim` + per-restart `loss_history` saved in .pth files
- Loss curve plots (3-panel: cos_sim/TV/total vs iteration, all restarts)
- D1 comparison figure + cos_sim overlay

- Code: `experiments/phase0_vit_inversion.py` + `experiments/phase0_d1_compare.py`
- WEXAC scripts: `scripts/run_phase0_d1_{A,B,C,D}.sh`
- Results: `results/phase0_full_r8_n1_s42_20260414_*.pth`, `results/phase0_d1_comparison_*.csv`
- Figures: `figures/phase0/phase0_d1_comparison.png`, `figures/phase0/phase0_d1_cossim_overlay.png`

#### D2: Targeted Sweep Around Winning Config — COMPLETE (2026-04-28)

40-config sweep (signAdam, full gradient, Flowers102, seed=42), 29 configs analyzed:

**Top 7 configs (all cleared 0.3 SSIM gate):**

| Rank | TV weight | LR    | Iters | SSIM   | PSNR  | cos_sim |
|------|-----------|-------|-------|--------|-------|---------|
| 1    | **1e-1**  | 0.05  | 30000 | **0.548** | **15.11** | **0.955** |
| 2    | 1e-1      | 0.10  | 10000 | 0.496  | 12.93 | 0.955   |
| 3    | 1e-1      | 0.01  | 30000 | 0.469  | 12.44 | 0.941   |
| 4    | 1e-1      | 0.10  | 30000 | 0.469  | 12.81 | 0.955   |
| 5    | 1e-1      | 0.50  | 10000 | 0.466  | 12.20 | 0.946   |
| 6    | 1e-1      | 0.50  | 30000 | 0.464  | 12.27 | 0.959   |
| 7    | 1e-1      | 0.05  | 10000 | 0.385  | 12.23 | 0.930   |

**Key findings:**
1. **tv_weight=1e-1 is the dominant winning factor.** All 7 gate-passing configs use tv=1e-1. The next TV level (2e-2) tops out at SSIM=0.27. The 10× jump from D1's tv=1e-2 (SSIM=0.144) to tv=1e-1 produced a 3.8× SSIM improvement.
2. **LR is secondary.** Across lr ∈ {0.01, 0.05, 0.1, 0.5} at tv=1e-1, SSIM stays in [0.46, 0.55]. lr=0.05 is best but the spread is small.
3. **30K iters helps but not dramatically.** lr=0.05 jumps from 0.385 (10K) to 0.548 (30K). lr=0.1 marginal: 0.496 → 0.469. Diminishing returns past 10K for most configs.
4. **High cos_sim (0.94-0.96) at all 7 winners**, while D1's noisy configs had similar cos_sim with garbage SSIM. Strong TV is what converts gradient match into visible structure.

**Go/no-go outcome:** **GATE CROSSED.** SSIM=0.548 is well past the 0.3 threshold. The flower's pink color, petal arrangement, and leaf structure are all clearly visible. **Proceed to multi-seed validation, LoRA-only inversion, and face-photo extension (face1/2/3 sweep already submitted).**

- Code: `experiments/phase0_d2_compare.py`, `phase0_vit_inversion.py` `--d2 --config_index N` mode
- WEXAC scripts: `scripts/run_phase0_d2_wexac.sh` (40 jobs), face sweep: `scripts/run_phase0_face_sweep.sh`
- Results: `results/phase0_d2_*.pth` (29 configs), `results/phase0_d2_comparison_<ts>.csv` (sweep summary)
- Aggregate figures (in `figures/phase0/d2_sweep/`):
  - `phase0_d2_heatmap.png` — SSIM grid, tv × lr, panels for 10K/30K iters (this is where the "tv=1e-1 dominates" story lives)
  - `phase0_d2_top_comparison_by_tv.png` — GT + best reconstruction at each TV level (analog of D1's 4-config side-by-side)
  - `phase0_d2_cossim_overlay_by_tv.png` — cos_sim & total-loss curves, one per TV level
  - The earlier top-5-by-SSIM variants were dropped: when one axis dominates, top-N collapses to a single regime — by-axis is the canonical view (see LESSONS_LEARNED.md "by-axis vs top-N visualization").
- Custom image support: `--image_path` flag in `phase0_vit_inversion.py`, tests in `experiments/tests/test_phase0_custom_image.py`
- Repo reorg: figures grouped under `figures/{phase0,sprint1,training_dynamics,free_c_all_seeds}/`. Per-iter snapshot dirs (~800 MB) excluded via `.gitignore`.

#### D3v2: Frequency + LPIPS Prior Ablation — COMPLETE, PRIORS DON'T HELP (2026-04-28)

7-config ablation on top of the D2 winner backbone (signAdam, tv=1e-1, lr=0.05, 30K iters, n_restarts=2 for speed) on Flowers102, seed=42. Variables: `freq_weight ∈ {0, 1e-3, 1e-2, 1e-1}` × `lpips_weight ∈ {0, 1e-3, 1e-2}` (subset of 7 cells).

| idx | config | SSIM | PSNR | cos_sim |
|---|---|---|---|---|
| 0 | freq=1e-3 | **0.558** | 11.9 | 0.975 |
| 4 | lpips=1e-2 | 0.548 | 12.7 | 0.974 |
| 1 | freq=1e-2 | 0.495 | 12.6 | 0.951 |
| 6 | freq=1e-2 + lpips=1e-2 | 0.478 | 12.6 | 0.944 |
| 2 | freq=1e-1 | 0.423 | 12.9 | 0.933 |
| 3 | lpips=1e-3 | 0.416 | 12.4 | 0.946 |
| 5 | freq=1e-2 + lpips=1e-3 | 0.411 | 12.4 | 0.949 |

**Conclusion:** TV at 1e-1 does all the prior work in pixel space. Stacking freq / LPIPS on top of it adds nothing measurable at best and actively over-regularizes at worst (the SSIM 0.41-0.43 cluster). Cos_sim stays high (~0.93-0.97) across the board — the difference is structural correctness, which neither prior fixes.

- Code: `scripts/run_phase0_d3_v2.sh`, results: `results/phase0_full_r8_n1_s42_20260428_12071*_d3v2_idx{0..6}_*.pth`
- Figures: `figures/phase0_report/fig_d3_ablation.{png,pdf}` and `figures/phase0_report/last2days/fig_d3_grid.png`

#### Face1 at the D3 Winner — REAL FACE CROSSES THE GATE (2026-04-28)

Same hyperparameters as the D3 winner (signAdam, tv=1e-1, lr=0.05, freq=1e-3, 30K iters × **8 restarts**, seed=42), applied to a real human portrait (`data/faces/face1.jpg`) via the new custom-image loader.

**Result: SSIM=0.522, PSNR=13.8 dB, cos_sim=0.974** — within seed/restart noise of the Flowers102 D2 winner. The reconstruction is a visibly identifiable person: skin tone, collar, eye placement all recovered. The 30K-iter trajectory is clean coarse-to-fine — recognizable face shape by iter 5K, fine detail between iter 10K and 30K, no late-stage collapse.

**Why this matters:** Flowers102 was the technical gate. Faces are the privacy payload. Same hyperparameters with zero re-tuning transferred from a flower image to a real face. The privacy attack now works on the relevant data modality.

- Tensor: `results/phase0_full_r8_n1_s42_20260428_134922_face_d3winner_freq1e-3.pth`
- Figures: `figures/phase0_report/last2days/fig_face1_recon.png`, `fig_face1_iters.png`
- Supervisor handoff: `notes/phase0_report.tex` (350-line LaTeX report covering D1→D4) and `notes/phase0_last2days.md` (chronological log)

#### Post-D3 Gaps & Next Levers

What's missing from Phase 0:
- **face2.jpg, face3.jpg at the new winner**: only face1 has been re-run. The face2/face3 numbers in old logs (`wexac_logs/phase0_face{2,3}_584*.out`, SSIM 0.21-0.24) were from the weak-TV March sweep — stale, no tensors saved.
- **N>1 anywhere**: every Phase 0 .pth on disk is `n1`. Multi-image inversion in Phase 0 is still backlog (Sprint 3 / superposition section in CLAUDE.md).
- **Multi-seed faces**: the SSIM=0.522 number is one seed. No mean±std yet.
- **LoRA-only at the winner**: every run is `--mode full` (86M-param gradient). The LoRA-only sweep at rank ∈ {8, 16, 32, 64} is still pending.

The cheap next levers (all submitted 2026-05-13, jobs running): D4 face-structure prior, D5 chroma-coupled TV, multi-seed face1.

#### D4: Face-Structure Prior — INFRA WIRED, SWEEP RUNNING (2026-04-29 → 2026-05-13)

**Motivation.** D3 confirmed TV does all the prior work; freq/LPIPS only re-impose smoothness and don't fix the structural failure mode (mouth in wrong place, multiple face fragments). Adding a *semantic* prior — a frozen face detector + landmark-layout penalty — is the next lever.

**What was added (this turn):**
- [experiments/face_prior.py](experiments/face_prior.py): `load_face_prior`, `compute_face_prior` (presence + 5-pt landmark layout + bbox symmetry), `face_detection_score`, `face_prior_ramp`. Backbone: kornia.contrib.FaceDetector (YuNet, ~600KB, already in env). Detection-confidence threshold lowered to 0.05 to keep gradient flow during warm-up.
- [experiments/phase0_vit_inversion.py](experiments/phase0_vit_inversion.py): new CLI flags `--cos_weight`, `--face_weight`, `--face_layout_weight`, `--face_sym_weight`, `--face_warmup_iters` (default 5000), `--face_ramp_iters` (default 2000), `--face_model`. With `face_weight=0` the legacy code path is byte-equivalent.
- [experiments/tests/test_face_prior.py](experiments/tests/test_face_prior.py): 9 pytest unit tests (all pass). Coverage: loader, real-face vs noise loss, differentiability, no-grad eval score, ramp helper, layout penalty zero on canonical face, layout penalty positive on eye-mouth swap, pipeline plumbing.
- [scripts/run_phase0_face_prior_sweep.sh](scripts/run_phase0_face_prior_sweep.sh): 9-arm bsub ablation grid on face1.jpg (A control, B face-only, C low TV+face, D high TV+face, E1/E4 face_weight strength sweep, F1/F3/F4 cos_weight sweep). 30k iters × 8 restarts × 9 jobs in parallel.
- [experiments/analyze_face_prior_sweep.py](experiments/analyze_face_prior_sweep.py): post-sweep analyzer. Produces 5 figures (per-arm grid with landmark overlays, face_weight strength curve, cos_weight curve, per-arm loss panels, winner landmark evolution) under [figures/phase0/face_prior/](figures/phase0/face_prior/) plus a metrics CSV.

**Bug fix in this turn**: kornia's YuNet postprocess does `(cls * iou.clamp(0,1)).sqrt()`, which produces `sqrt(0)` for anchors with non-positive raw iou. The backward of `sqrt(0)` is `0.5/0 = inf`; even though the threshold filter discards those anchors, IEEE `0 * inf = NaN` poisoned the entire input gradient. Patched by adding `+1e-12` inside the sqrt (monkeypatch in `_patch_postprocess_nan_safe`). See [LESSONS_LEARNED.md](LESSONS_LEARNED.md).

**Sweep launched (2026-05-13).** 9 jobs running on WEXAC: `phase0_face_prior_{A,B,C,D,E1,E4,F1,F3,F4}` (jobs 777007-777019 / 777085-777095). n_restarts dropped 8 → 4 after an earlier sweep hit the 48h wall; the new `partial_save_fn` checkpoint hook (see below) makes early kills safe. Companion runs share the same submission window: multi-seed face1 (5 seeds, jobs 777058-777063) and D5 chroma-TV (2 arms, jobs 777084/777086) — all submitted together so they can be analyzed jointly when they land. Headline numbers will be appended once the sweep completes.

**Infrastructure: per-restart checkpointing (2026-05-13).** `invert_gradient` now takes a `partial_save_fn(restart_idx, best_x, best_cos, loss_history)` callback fired after every completed restart. `run_phase0` wires it to the same `.pth` path the final save uses, so a job killed at restart 4/8 leaves a usable partial reconstruction on disk (marked `metrics['partial'] = True`). This lets us run more arms in parallel without paying the full 12h × 8-restart wall every time. Unit test in `experiments/tests/test_face_prior.py`. Commit `8a7cfa2`.

#### D5: Chroma-Coupled TV (LAB-space) — INFRA WIRED, JOBS RUNNING (2026-05-13)

**Motivation.** D3 / D4 address structural correctness (face layout). The remaining visual defect on face1 (SSIM=0.522) is *colored speckle* — high-frequency RGB noise the spatial-only TV cannot see because it doesn't constrain per-channel coherence. Natural images have smooth chroma; speckle pixels violate this. Penalizing TV in LAB space with a heavier weight on a/b (chroma) than L (luminance) is the cheapest possible fix.

**What was added (this turn):**
- [experiments/phase0_vit_inversion.py](experiments/phase0_vit_inversion.py): `tv_norm='lab'` option in `invert_gradient` and `run_phase0`, new CLI flags `--tv_norm lab` and `--tv_chroma_weight` (default 5.0). LAB values are rescaled (L/100, a/128, b/128) so `tv_weight=1e-1` stays on the same magnitude as the l2/RGB version — no other hyperparameters change.
- [scripts/run_phase0_face1_chroma_tv.sh](scripts/run_phase0_face1_chroma_tv.sh): 2-arm bsub sweep on face1.jpg at `chroma_weight ∈ {5, 20}`, otherwise identical to the D3-winner config (signAdam, tv=1e-1, lr=0.05, freq=1e-3, 30k iters × 8 restarts, seed=42).

**Status.** Jobs `phase0_face1_chroma_tv_cw5` (777084) and `phase0_face1_chroma_tv_cw20` (777086) submitted. ETA ~5h each. Companion runs: multi-seed face1 (5 seeds, jobs 777058–777063) and D4 face-prior sweep (9 arms, jobs 777007–777019) — all submitted same turn so they can be analyzed jointly once they land.

**Result (2026-05-13): FAILED — both arms degrade the baseline by ~60%.**

| Config | SSIM | vs D3 baseline (0.522) |
|---|---|---|
| chroma_weight=5 | **0.203** | −61% |
| chroma_weight=20 | **0.169** | −68% |

Diagnosis: the LAB-rescaling I applied (L/100, a/128, b/128 to keep `tv_weight=1e-1` on the same magnitude as RGB-l2) gave chroma channels effective weight 5× and 20× *stronger* than the original RGB TV, but the L-channel (which carries most pixel structure) became 100×/128× *weaker*. The reconstruction lost luminance detail trying to satisfy chroma smoothness. Two possible fixes if we ever revive this:
1. Drop the LAB rescaling and just tune `tv_weight` down by ~100× to compensate.
2. Keep RGB-TV for luminance and add a *separate* chroma-only penalty.

**Verdict**: skip both. Chroma TV is the wrong knob — TV-l2 at 1e-1 was already nearly optimal. Color speckle won't be fixed by reweighting TV; it needs an actual image-manifold prior (D6 latent-space / SDS).

#### N=3 same-person reconstruction — DONE, surprising result (2026-04-29)

First-ever Phase 0 N>1 run completed overnight (job 976038, finished 2026-04-29 09:54). face1.jpg + face2.jpg + face3.jpg (all the same person) jointly inverted from one captured fine-tuning gradient (all labels=0). Same D3-winner backbone otherwise.

**Aggregate vs N=1:**

| Run | SSIM | PSNR | cos_sim |
|---|---|---|---|
| face1 solo (D3 winner, N=1) | 0.522 | 13.8 | 0.974 |
| **N=3 same person** | **0.662** | **14.5** | **0.979** |

Per-image diagonals (each recon vs its own GT): 0.603 / 0.674 / 0.710. All three individually beat face1 solo.

**Partial superposition collapse, though.** Cross-matrix shows every reconstruction has its single strongest SSIM at face2 (the centroid of the GT set: mean GT-GT 0.601 vs face1's 0.555, face3's 0.567), not at its own corresponding GT. Recon-recon SSIM is 0.66–0.71 vs GT-GT 0.52–0.61 — the three outputs are MORE similar to each other than the inputs are.

**Why N=3 beats N=1.** Five mechanisms compose:
1. **Gradient SNR amplification** — averaging 3 same-person gradients reinforces shared-identity directions, cancels per-photo noise.
2. **More degrees of freedom for the same constraint** — 3 × 150K pixel variables to satisfy one 86M-dim cos_sim, with per-slot TV/freq priors keeping each output natural.
3. **Implicit identity-manifold prior** — three samples nearly span the same-person manifold; reachable solutions cluster near "consistent renderings of this identity".
4. **Same-class, same-identity defuses superposition** — CLAUDE.md's mixing-symmetry warning applies for cross-class / cross-identity N>1. Here label and identity coincide so the symmetry degrees of freedom align with the gradient direction (not orthogonal to it).
5. **The "win" is identity, not snapshots** — recons recover the person well but lose pose/clothing detail. For a privacy attack, identity recovery is the threat; this is arguably *worse* for privacy than pixel-perfect copies.

**Open question.** Whether the result generalizes to cross-identity N>1 (where mechanism #4 reverses). Until we test, we don't know whether identity-manifold or mixing-symmetry dominates.

Details: [notes/phase0_last2days.md](notes/phase0_last2days.md) "N=3 same-person — what actually happened" section.
Figures: [figures/phase0_report/last2days/fig_n3_grid.png](figures/phase0_report/last2days/fig_n3_grid.png), [fig_n3_crossmatrix.png](figures/phase0_report/last2days/fig_n3_crossmatrix.png).

#### D6: Latent-Space Reconstruction / SDS Prior — PLAN (2026-05-13)

**Motivation.** If multi-seed median fusion + chroma TV still leave visible speckle and structural error after D5, the diagnosis is that 224×224×3 pixel-space optimization is too high-dimensional to satisfy a "looks natural" constraint. The fix: replace pixel-space optimization with **latent-space optimization** through a frozen generative model. Off-manifold pixel patterns become mechanically impossible because the decoder only produces natural-image-distribution outputs.

**Two flavors (cheapest first):**

1. **Latent-space recon (S3.5 in Sprint 3 backlog).** Optimize `z ∈ R^4 × 28 × 28` in the latent space of the Stable Diffusion VAE; decode to pixel space via the frozen VAE decoder for the gradient-matching loss. Search space drops from 150K to ~3K dims. Decoder gradient flows through every iter — no autograd subtlety beyond what we already do.
   - Engineering: load SD VAE (~330MB, fp16), wrap `x_recon` as `decode(z)`. The cos_sim loss, TV, freq, and face priors all still apply but operate on `decode(z)` instead of `x_recon`. Maybe 2 days of code + 1 day of tuning.
   - Risk: VAE-decoded image manifold may not contain the *specific* face well enough — could decay to a generic-looking face. Mitigated by joint optimization of `z` and a small residual `δ` in pixel space, so the latent provides the prior and the residual recovers the identity.

2. **Score Distillation Sampling (SDS).** Add an SDS term using a frozen diffusion model (e.g., Stable Diffusion 1.5): `sds_loss = E_t,ε[w(t)·(ε_θ(x_t,t) − ε)·∂x_t/∂x]`. This is the strongest manifold prior available and composes with the face-structure prior already in the codebase. Replaces TV entirely for late iters.
   - Engineering: requires a working SD checkpoint on WEXAC, gradient checkpointing for memory, careful weight scheduling. 3-5 days conservatively. SDS gradients are noisy at small batch sizes — likely needs `n_restarts=16+`.
   - Risk: known to over-smooth and converge to generic mean-faces. Mitigated by warm-starting from the D4/D5 reconstruction and using SDS only in the last 10K iters as a polish step.

**Go/no-go gate.** Trigger D6 only if **all** of (a) multi-seed median, (b) chroma TV, and (c) face-structure prior together still leave SSIM<0.6 *and* qualitative artifacts that hurt identifiability. If we get to ~0.6 with the cheap interventions, D6 becomes a thesis stretch goal, not a critical path.

**Code locations (placeholders, not yet created):** `experiments/latent_recon.py` for #1, `experiments/sds_prior.py` for #2. Both plug into `phase0_vit_inversion.py` via the same flag pattern as `face_weight`.

#### D7: Cross-Identity N>1 — RUNNING (opened 2026-05-13)

**Why this matters.** D-N3 (same-person, same-label) gave SSIM=0.662 — a clear win over N=1 (0.522). The reason is that label and identity coincide so the gradient-mixing-symmetry from CLAUDE.md becomes harmless. **The critical untested case is cross-identity N>1**, where the symmetry returns: the gradient is a sum over distinct identities, and any linear recombination of per-sample gradients that hits the same sum is indistinguishable to the optimizer.

**Hypotheses (one will dominate):**
- H1 — Identity-manifold dominates: cross-identity N=2 produces two recognizable but slightly blended faces (similar to D-N3, with a stronger centroid bias because there's no shared identity to amplify).
- H2 — Mixing-symmetry dominates: full superposition collapse — both recons converge to a single "averaged face" of the two identities, qualitatively unidentifiable.
- H3 — Same-class still defuses: as long as labels match, even different identities reconstruct cleanly (the symmetry only bites at cross-class). Plausible because the label fixes the BCE gradient sign; identity drift would still cost cos_sim.

**Experiment design.**
- 3 arms, all using the D3 winner backbone (signAdam, tv=1e-1, lr=0.05, freq=1e-3, 30K iters × 8 restarts):
  - **Arm 1 — Cross-identity, same label (label=0 for both)**: face1.jpg + a Flowers102 image labeled 0. Tests H3 vs H2.
  - **Arm 2 — Cross-identity, opposite labels (face1 label=0, Flowers label=1)**: forces the model to split the gradient along the BCE sign. Strongest stress test for mixing symmetry. Tests H2.
  - **Arm 3 — Same scene, two photos** (e.g., two different Flowers102 photos, both label=0): same class but distinct subjects. Cleanest test of "does same-class N>1 actually defuse mixing" independent of the same-identity manifold (which only applies to faces).
- Each arm: 1 bsub job, ~12h wall budget (use `W 18:00` since solo runs blew through 12h).
- Output: `.pth` per arm with x_true (N=2, 3, 224, 224), x_recon (N=2, 3, 224, 224), plus all the per-image SSIM cross-matrices.

**What "winning" means here.**
- If H1: paper-positive (the attack works for arbitrary N as long as labels are same-class; centroid bias is a calibrated caveat).
- If H2: paper-negative-but-publishable (cross-identity collapse is a hard ceiling — motivates the diversity penalty / decomposition approaches from CLAUDE.md).
- If H3: paper-positive at a cost (works for any same-class N, breaks at cross-class; need cocktail-party / SPEAR-style decomposition for the cross-class case).

**Code shipped (2026-05-13).**
- [experiments/phase0_vit_inversion.py](experiments/phase0_vit_inversion.py): `--image_path` now accepts tokens of the form `dataset:index` (e.g. `flowers102:42`) alongside filesystem paths in the same comma-separated list. New `--image_labels` flag accepts a parallel `0,1`-style list; if omitted, defaults to all-zero. The `get_sample_images` loader was refactored to share a `_build_dataset(name)` helper between custom-image and dataset paths. 7-test smoke battery passed (single path, N=3 same-person regression, dataset:index, mixed path+dataset, labels override, length-mismatch raises, backward-compat dataset mode).
- [scripts/run_phase0_d7_cross_identity.sh](scripts/run_phase0_d7_cross_identity.sh): 3-arm bsub script, each N=2 at the D3 winner config.

**Status.** Jobs submitted 2026-05-13 00:55:
- Arm A (`d7_A`, face1+flowers102:42, both label=0): **779866** RUN
- Arm B (`d7_B`, face1+flowers102:42, labels 0/1): **779867** RUN
- Arm C (`d7_C`, two distinct Flowers102 photos, both label=0): **779868** RUN

ETA ~18h each. Track: `bjobs -J 'phase0_d7_*'`. After completion, generalize the N=3 cross-matrix renderer to 2×2 and update this section with headline numbers.

**Owner**: Yoad.

**Partial result (2026-05-13 ~10:00): Arm A complete, no collapse.**

Cross-identity N=2, both label=0 (face1.jpg + flowers102:42):

| recon[i] | vs GT[i] (own) | PSNR |
|---|---|---|
| recon[0] (face) | 0.435 | 11.2 |
| recon[1] (flower) | 0.428 | 11.4 |
| **aggregate** | **0.431** | 11.3 |

**Interpretation.** Per-image SSIMs are nearly equal (0.435 ≈ 0.428) — the optimizer split the captured gradient between two distinct identities without collapsing. Aggregate SSIM is below face1 solo (0.522) because there's no identity-manifold prior to exploit (the win mechanism #3 from the N=3 same-person section doesn't apply here — face and flower don't share an identity manifold). But the recon quality is comparable across both subjects, so this is **positive evidence for H1/H3**: cross-identity same-class N>1 does NOT trigger superposition collapse.

**Partial saves from arms B and C** (still RUN, checkpoints written ~7.5h in):
- Arm B (opposite labels 0,1): recon[0]=0.509 (face), recon[1]=0.438 (flower) — opposite labels appear to *help* slightly, not hurt; counter to the strongest version of the mixing-symmetry hypothesis.
- Arm C (two flowers, same class): recon[0]=0.334, recon[1]=0.300 — both low; will only be reliable after final save.

**Status of dependency jobs** (informs interpretation):
- 5-seed multi-seed face1: 3/5 still running, 2 (s7, s13) hit exit code 127 (heredoc shell bug) and were resubmitted as **798244** / **798356** with `W 16:00`.
- face2/face3 resubmits (777106 / 777107): RUN, partial saves written.
- D4 face_prior_A control replicates close to D3 baseline (0.499 vs 0.522 — within seed/restart noise).

---

### Sprint 3: Scaling Beyond MNIST — IN PROGRESS (via D1→D2→D3)

**Goal**: Establish gradient-based reconstruction on realistic (non-MNIST) data. Sprint 2 proved the NTK attack works on MNIST MLPs (SSIM=0.997 full, 0.557 LoRA free-c). Sprint 3 bridges to ViT-scale.

**Updated strategy after D1 results:** D1 showed the bottleneck is regularization, not architecture. signAdam + TV=1e-2 reached SSIM=0.144 on ViT-B/16 with no architectural changes. The path forward is hyperparameter tuning + stronger priors on the same ViT, not retreating to simpler architectures.

- ~~**S3.1**: Fix Phase 0 hyperparameters~~ → **Absorbed into D1 (done) + D2 (next)**
- **S3.2**: Shrink reconstruction space (optimize in 32×32 or frequency domain, upsample to 224×224) — still relevant if D2 plateaus
- ~~**S3.3**: Shrink architecture~~ — **Deprioritized** (ViT works, no need to retreat)
- **S3.4**: Differentiable unrolling (bypass NTK approximation) — future direction
- **S3.5**: Add stronger image priors (LPIPS, frequency, SDS) — **maps to D3**

Each sub-sprint has a clear **go/no-go gate** so we don't waste time on dead ends.

#### S3.1: Phase 0 Hyperparameter Sweep (1-2 days)

**Hypothesis**: Phase 0's poor SSIM is primarily due to untuned hyperparameters, not a fundamental ViT limitation. The current config (lr=0.1, tv_weight=1e-4, Adam, 10K iters, 8 restarts) was never swept.

**Design**:
- Independent variable: lr × tv_weight × n_iters × optimizer
- Grid:
  - `lr`: [0.01, 0.05, 0.1, 0.5, 1.0]
  - `tv_weight`: [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
  - `n_iters`: [10000, 30000]
  - `optimizer`: [Adam, SGD+momentum, L-BFGS]
- Fixed: rank=8, n_images=1, seed=42, mode=full, n_restarts=4 (reduced for speed)
- Total: 5 × 6 × 2 × 3 = 180 configs (but use early stopping + parallel restarts)
- **Practical reduction**: run a 2-stage sweep — first stage: coarse lr × tv_weight (30 configs, 4 restarts, 10K iters). Second stage: top-5 configs with 30K iters, 8 restarts.
- Metrics: SSIM, PSNR, final cosine similarity, wall time
- Output: `results/sprint3_s1_phase0_sweep_*.csv`, best config `.pth`

**Go/no-go gate**: If best SSIM > 0.3 (full model), proceed to S3.5 (priors). If SSIM < 0.15 for all configs, the bottleneck is architectural → proceed to S3.2/S3.3.

**Code**: Extend `phase0_vit_inversion.py` with `--sweep` mode.
**Script**: `scripts/run_sprint3_s1_phase0_sweep.sh`

#### S3.2: Low-Dimensional Reconstruction Space (1-2 days)

**Note**: Phase 0 already uses CIFAR-10 images resized to 224×224 (see `get_sample_images()` in `phase0_vit_inversion.py`). The data isn't the issue — the 224×224 *search space* is.

**Hypothesis**: Optimizing x_recon in 224×224 pixel space (150K dims) is wasteful when the source image is 32×32 CIFAR-10 (3K dims of actual information). Reconstructing in a lower-dimensional space and upsampling should dramatically improve convergence.

**Design**:
- **Variant A — 32×32 reconstruction**: Optimize x_recon in 32×32 space, bilinear-upsample to 224×224 before feeding to ViT. Reduces search space 49×.
- **Variant B — Fourier-truncated reconstruction**: Parameterize x_recon in frequency domain, zero out high-frequency components (>32×32 bandwidth). Smoother optimization landscape.
- **Variant C — Patch-aware reconstruction**: ViT-B/16 uses 14×14 = 196 patches of 16×16 pixels. Reconstruct per-patch means + low-rank structure (196 × ~50 dims = 9.8K parameters).
- Compare all variants against baseline (full 224×224 pixel space)
- Fixed: rank=8, seed=42, mode={full, lora}, best hyperparams from S3.1
- Seeds: 5 seeds for quick validation, 20 seeds if promising

**Go/no-go gate**: Any variant SSIM > 0.3 → search space was the bottleneck, proceed to S3.5 (priors). All variants SSIM < 0.15 → ViT gradient signal itself is too weak → proceed to S3.3.

**Code**: Extend `phase0_vit_inversion.py` with `--recon_space {pixel, lowres, fourier, patch}` flag.
**Script**: `scripts/run_sprint3_s2_lowdim.sh`

#### S3.3: Simpler Architectures on CIFAR-10 (2-4 days)

**Hypothesis**: ViT's 86M parameters + attention double-backward make gradient inversion intrinsically harder than CNNs/ResNets. Testing simpler architectures isolates whether the bottleneck is the model or the data.

**S3.3a: Small CNN on CIFAR-10** (1-2 days)
- Architecture: Conv(3→32, 3×3) → ReLU → MaxPool → Conv(32→64, 3×3) → ReLU → MaxPool → FC(64×8×8→128) → FC(128→1)
- ~200K parameters (comparable to LoRA param count)
- Train from scratch on CIFAR-10 binary (vehicles vs animals, matching Haim et al.)
- Fine-tune on 1-2 held-out images, T=1 SGD step
- Run NTK reconstruction (Experiment B style) from ΔW
- Run gradient inversion (Phase 0 style) from exact gradient
- Compare both methods on same model

**S3.3b: ResNet-18 on CIFAR-10** (1-2 days)
- Load pretrained ResNet-18 from torchvision (11M params)
- Apply LoRA (r=8, 16) to conv layers via peft
- Fine-tune on 1-2 held-out CIFAR-10 images
- Run gradient inversion from exact gradient
- Compare to ViT results — ResNet's skip connections stabilize gradient flow

**S3.3c: DeiT-Tiny on CIFAR-10** (1 day, parallel with S3.3b)
- Load DeiT-Tiny from timm (5.7M params, same ViT architecture but 15× smaller than ViT-B)
- Apply LoRA (r=8)
- Fine-tune + invert
- Tests whether ViT architecture itself is the issue, or just its scale

**Go/no-go gate**:
- CNN SSIM > 0.4 + ViT SSIM < 0.2 → architecture is the bottleneck; thesis focuses on CNN/ResNet LoRA reconstruction
- Both SSIM > 0.3 → method works across architectures; proceed to scale up
- Both SSIM < 0.15 → gradient inversion on color images is fundamentally harder; consider differentiable unrolling (S3.4)

**Code**: `experiments/phase0_cnn_cifar.py`, `experiments/phase0_resnet_cifar.py`, `experiments/phase0_deit_cifar.py`
**Script**: `scripts/run_sprint3_s3_arch_comparison.sh`

#### S3.4: Differentiable Unrolling — Approach G (3-5 days)

**Hypothesis**: The NTK approximation (ΔW ≈ -η Σ cᵢ ∇f(θ₀; xᵢ)) is a first-order linearization that breaks at T>1. Gradient inversion (Phase 0) requires brittle `create_graph=True` double-backward through attention. **Differentiable unrolling** avoids both problems: simulate the actual fine-tuning steps differentiably and match the resulting weights to the observed weights.

**Method**:
```
# Outer optimization over x_recon
for outer_iter in range(N_outer):
    # Inner loop: simulate T fine-tuning steps
    θ = θ_base.clone()
    for t in range(T):
        loss = L(θ; x_recon)
        grads = autograd.grad(loss, θ, create_graph=True)
        θ = θ - η * grads  # differentiable SGD step

    # Outer loss: match simulated weights to observed weights
    outer_loss = ||θ - θ_observed||²
    outer_loss.backward()  # backprop through all T inner steps
    optimizer_x.step()
```

**Design**:
- Phase 1: Validate on MNIST MLP (should reproduce Experiment B at T=1: SSIM≈0.997)
- Phase 2: Test T=1,2,5,10,20 on MNIST MLP, compare to NTK results
- Phase 3: Apply to CNN/ResNet on CIFAR-10 (if S3.3 identifies a working architecture)
- Phase 4: Apply to ViT on CIFAR-10 (if memory permits — T steps of ViT forward/backward)
- Memory management: gradient checkpointing for T>10

**Key advantages over NTK**:
- Exact for any T (no linearization error)
- No coefficient estimation needed (no cᵢ unknowns)
- Reduces to Experiment B at T=1 (validation check)
- Works with any architecture (no NTK assumptions)

**Key risks**:
- Memory: O(T) computation graphs (mitigated by gradient checkpointing)
- Must know exact lr and T (can sweep if unknown — realistic attacker may not know these)
- Non-convex outer optimization — needs restarts

**Go/no-go gate**: T=1 on MNIST reproduces SSIM>0.99 → validates implementation. T=10 beats NTK SSIM → publish as improvement. T=1 fails → implementation bug, debug before proceeding.

**Code**: `experiments/differentiable_unrolling.py`
**Script**: `scripts/run_sprint3_s4_unrolling.sh`

#### S3.5: Stronger Image Priors (2-4 days, after S3.1/S3.2 gate)

**Hypothesis**: Even with correct gradient signal, 224×224 RGB reconstruction requires priors beyond TV to converge. Natural images occupy a tiny manifold in pixel space.

**Prior hierarchy** (implement in order of complexity):
1. **Frequency-domain prior** (0.5 day): Penalize high-frequency Fourier components. Natural images have most energy in low frequencies. `freq_loss = ||FFT(x)[high_freq]||²`. Nearly free to implement.
2. **LPIPS perceptual loss** (1 day): Use frozen DINO or ResNet50 features. `lpips_loss = ||F(x_recon) - F(x_recon_smoothed)||²` (self-regularization) or compare to a "natural image" centroid. Requires `lpips` package.
3. **Batch normalization statistics prior** (0.5 day): If model has BN layers, match running mean/var of reconstruction to the BN statistics. Free signal from the model itself. (Only applicable to ResNet/CNN, not ViT.)
4. **Score Distillation Sampling (SDS)** (2-3 days): Frozen diffusion model (Stable Diffusion) guides reconstruction toward natural images. `sds_loss = E_t,ε[w(t)(ε_θ(x_t, t) - ε) ∂x_t/∂x]`. Most powerful but most complex. Requires diffusion model on same GPU.
5. **Latent-space reconstruction** (1-2 days): Instead of optimizing in pixel space, optimize in the latent space of a frozen VAE (from Stable Diffusion). Decode to pixel space for the gradient matching loss. Reduces search space from 150K to ~4K dims.

**Design**: Add priors as composable loss terms in `invert_gradient()`. Each prior has a weight hyperparameter. Sweep prior weights on the best Phase 0 config.

**Code**: `experiments/image_priors.py` (prior loss functions), integrated into `phase0_vit_inversion.py` via `--prior` flag.
**Script**: `scripts/run_sprint3_s5_priors.sh`

#### Sprint 3 Summary Table

| Sub-sprint | What | Architecture | Data | Time | Depends on |
|------------|------|-------------|------|------|------------|
| **S3.1** | Phase 0 hyperparam sweep | ViT-B/16 | CIFAR-10 224×224 | 1-2d | — |
| **S3.2** | Low-dim recon space | ViT-B/16 | CIFAR-10 (32→224) | 1-2d | — |
| **S3.3a** | CNN baseline | Conv2+FC | CIFAR-10 32×32 | 1-2d | — |
| **S3.3b** | ResNet-18 + LoRA | ResNet-18 | CIFAR-10 32×32 | 1-2d | — |
| **S3.3c** | Small ViT | DeiT-Tiny | CIFAR-10 32×32 | 1d | — |
| **S3.4** | Diff. unrolling | MNIST MLP first | MNIST → CIFAR | 3-5d | — |
| **S3.5** | Image priors | Best from above | Same | 2-4d | S3.1 or S3.2 |

**Parallelism**: S3.1, S3.2, S3.3a-c, S3.4 Phase 1 can all run independently. S3.5 depends on having a working baseline to improve.

**Priority ordering** (Gradient Bridge > NTK > Priors):
1. **Week 1, batch 1** (parallel, cheapest first):
   - S3.1: Phase 0 hyperparam sweep — quick diagnostic, clarifies if tuning alone helps (1 WEXAC job)
   - S3.4 Phase 1: Unrolling on MNIST — validates the approach on known-working data (local or 1 WEXAC job)
2. **Week 1, batch 2** (parallel, informed by batch 1 results):
   - S3.2: Low-dim reconstruction space — if S3.1 shows ViT signal exists, this amplifies it
   - S3.3a: CNN on CIFAR-10 — architecture isolation test, independent of ViT results
3. **Week 2** (depends on Week 1 gates):
   - S3.4 Phase 2: Unrolling T-sweep on MNIST — if Phase 1 validates
   - S3.3b or S3.3c: whichever architecture shows most promise
4. **Week 2-3**: S3.5 priors on best-performing config from above
5. **Week 3+**: S3.4 Phase 3-4 (unrolling on CIFAR-10 architecture)

**Critical path**: S3.4 (unrolling) feeds directly into the Gradient Bridge pipeline (Tier 1 of thesis roadmap). If unrolling works on MNIST and scales to CIFAR-10, it becomes the inversion engine for the full attack. S3.1-S3.3 are supporting experiments that de-risk the architecture choice.

### Sprint 2 Multi-Seed Validation — COMPLETE (2026-03-27)

50-seed free-c vs oracle comparison and 30-seed LeakyReLU validation completed overnight.

**Key findings:**
- **Seed=42 was an outlier**: SSIM=0.830 vs 50-seed mean=0.558±0.034. Report 50-seed stats as canonical.
- **Free-c beats oracle**: Mean SSIM 0.557 (free-c) vs 0.408 (oracle) across 50 seeds. Free-c wins 46/50. The consistency penalty provides implicit regularization that prevents sign-flip local minima.
- **LeakyReLU validated**: 30 seeds × {T=1, T=10} × {r=8, r=32}. Mean SSIM 0.558 (T=1), 0.572 (T=10). Control: 0.394-0.426.
- **r=16/32 improved**: SGD+LeakyReLU gives r=16 SSIM 0.624 (was 0.422), r=32 SSIM 0.680 (was 0.415).

### Sprint 2 Track 2: LoRA Free-Coefficient Extraction — COMPLETE

**Final results (best config per rank — SGD+LeakyReLU fixes r=16/32):**

| Rank | Oracle SSIM | Free-c SSIM | Coeff Error | Gap | Method |
|------|-------------|-------------|-------------|-----|--------|
| 4    | 0.615       | 0.509       | 0.192       | 0.11 | ReLU+L-BFGS |
| 8    | 0.692       | 0.617       | 0.177       | 0.08 | ReLU+L-BFGS |
| 16   | 0.769       | **0.624**   | —           | 0.15 | **SGD+LeakyReLU** (was 0.422) |
| 32   | 0.697       | **0.680**   | —           | 0.02 | **SGD+LeakyReLU** (was 0.415) |
| 64   | 0.714       | 0.635       | 0.019       | 0.08 | ReLU+L-BFGS |

### Sprint 2b: Multi-Step NTK Sweep — COMPLETE

Phases 0-2 completed (WEXAC jobs 669864, 674627). Phases 3-4 completed as Sprint 2c Track B1.

**Phase 0 (activation ablation):** LeakyReLU is dramatically more stable than ReLU at high T. Full model SSIM stays ~0.77-0.80 through T=100 vs ReLU collapsing/NaN'ing at T>=50.

**Phase 1 (SGD + free-c baseline, ReLU):** LoRA results terrible with ReLU — NaN everywhere at T>=50. Confirms activation choice is critical.

**Phase 2 (random restarts, LeakyReLU):** LoRA r=8 and r=32 nearly match full model (gap only 0.01-0.03) through T=100. Random restarts show low variance (~0.01).

**Phase 3 (LR scaling)** and **Phase 4 (warm-start):** Completed as Sprint 2c Track B1.

### Few-Shot Threat Model Analysis — Documented

The few-shot threat model is documented in CLAUDE.md (information density argument): when LoRA fine-tuning uses very few samples, the number of adapter parameters far exceeds the number of unknowns, making the system highly overdetermined and reconstruction theoretically feasible.

---

## What's Been Done

### Sprint 2a: Free-Coefficient Extraction — COMPLETE (2026-02-22)

Fixed the oracle-coefficient cheating. Implementation in `ntk_extraction.py` and `run_experiment_b.py`.

**Full model free-c result:** SGD + consistency α=1 achieves SSIM=0.997 (matches oracle). The attack works without cheating on full model.

### Sprint 2 Track 2: LoRA Activation + Optimizer Ablation — COMPLETE (2026-02-23)

**Activation ablation** (WEXAC job 669885): Swept alpha ∈ {10, 50, 150, 10000} × {L-BFGS, SGD} for LoRA r=8.
- alpha=10000 (≈ReLU) + L-BFGS = SSIM **0.744** (best)
- alpha=150 (ModifiedRelu default) + L-BFGS = 0.183 (terrible)
- ModifiedRelu actively harms LoRA extraction

**Separate optimizer for c** (2026-02-23): Decoupled L-BFGS for x from SGD/Adam for c. Added `coeff_optimizer_type` parameter with 'sgd' and 'adam' options.

**LoRA rank sweep with free-c** (WEXAC jobs 674631, 681126): r=4/8/16/32/64 with ReLU + L-BFGS + separate SGD for c. See results table in "In Progress" section above.

### Sprint 1: LoRA Reconstruction on MNIST FCN — COMPLETE (2026-02-22)

**Goal**: Produce preliminary results showing LoRA-trained weights leak training data.

**Experiment A — Convergence + Compose → KKT Reconstruct (Pre-Trained Init): FAILED**
- LoRA (r=8) reached loss=7.22e-8 after 1M epochs; full FT reached 1.29e-7. Neither hit the 1e-40 threshold.
- Loss decays as ~1/t from pre-trained init. Reaching 1e-40 would take ~10^39 epochs.
- KKT extraction NaN'd at epoch 7-8 (KKT loss started at ~460, should be ~0 for converged models).
- **Root cause (structural, not just convergence)**: The composed model W = W₀ + BA satisfies KKT with respect to all 502 samples the model was effectively trained on (500 pre-training + 2 fine-tuning). The extraction assumes only 2 samples, so the KKT loss of ~460 is essentially ||W₀||² — the unexplained pre-training residual from ~100-250 original support vectors. Even with perfect convergence, 2 images cannot explain weights that encode 502 images. (The 2 fine-tuning samples ARE support vectors for the N=2 case — the issue is the other 500 baked into W₀.)
- **This negative result motivates the Gradient Bridge**: compose-and-reconstruct fundamentally cannot separate fine-tuning signal from pre-training weights in the KKT framework.

**Experiment B — 1-Step NTK Reconstruction from Pre-Trained Weights: SUCCESS**

**IMPORTANT: All Sprint 1 Experiment B results use oracle coefficients** — cᵢ = (σ(f(θ₀; xᵢ)) - yᵢ)/N computed from the true private data x. In a real attack, the adversary doesn't have x and can't compute cᵢ. These results are an **upper bound** on attack quality. The next step is implementing free-coefficient extraction (see Sprint 2 plan below).

| Variant | SSIM | DSSIM | Notes |
|---------|------|-------|-------|
| Full model (T=1) | **0.9999** | 5.2e-5 | Near-perfect reconstruction (oracle c) |
| LoRA rank=8 | **0.797** | 0.102 | Recognizable digits, blurry (oracle c) |
| LoRA rank=16 | **0.802** | 0.099 | Slight improvement (oracle c) |
| LoRA rank=32 | **0.826** | 0.087 | Best LoRA result (oracle c) |
| Control (same class) | 0.582-0.693 | — | Proves instance-specific leakage |

- NTK diagnostics (T=1): weight_change=0.025, feature_stability=0.749, coefficient_drift=0.500
- **Key insight**: ΔW = θ₁ - θ₀ cancels the pre-trained component, isolating the fine-tuning signal
- **Oracle coefficient caveat**: Coefficients cᵢ are currently computed from true x and passed as fixed constants. This mirrors a "best-case attacker" scenario. The structural parallel to Haim et al.'s λᵢ (Lagrange multipliers) is exact — both are scalar unknowns that should be optimized alongside x. See LESSONS_LEARNED.md for full analysis.

**Multi-seed analysis (200 seeds, oracle coefficients):**
- 22/200 (11%) seeds produce strong signal (coeff_mag > 0.1)
- 13/200 (6.5%) produce medium signal (0.01-0.1)
- 165/200 (82.5%) produce weak/no signal (< 0.01)
- Perfect correlation: model wrong after centering ↔ strong signal
- Digits 4, 5, 8, 9 over-represented in attackable seeds
- **Figures**: `figures/sprint1/multi_seed_analysis.png`

- **Figures**: `figures/sprint1/experiment_b_grid_oracle.png`, `figures/sprint1/experiment_b_grid_free.png`, `figures/sprint1/experiment_b_grid_r32.png`, `figures/sprint1/rank_sweep_sprint1.png`, `figures/sprint1/sprint1_summary.png`
  - Note: Old figures (`experiment_b_free_coeff_grid.png`, `free_coeff_reconstruction_grid.png`) were stale — showed grey/blank reconstructions due to missing ds_mean correction. Deleted and replaced with correctly rendered versions (2026-04-28). `generate_experiment_b_figure()` is now mode-aware (auto-detects oracle vs free-coefficient).

### Base Reconstruction (Haim et al.) — Complete

The original paper's pipeline is fully working end-to-end:

- **2 trained models** (both D-1000-1000-1 MLPs, 1M epochs, BCE loss, SGD):
  - CIFAR-10 vehicles vs animals (250/class) → `dataset_reconstruction/models/weights-cifar10_vehicles_animals_d250_*.pth`
  - MNIST odd vs even (250/class) → `dataset_reconstruction/models/weights-mnist_odd_even_d250_*.pth`

- **4 reconstructions** (2 per model, via W&B sweeps):
  - CIFAR-10: `reconstructions/cifar10_vehicles_animals/{b9dfyspx,k60fvjdy}_x.pth`
  - MNIST: `reconstructions/mnist_odd_even/{kcf9bhbi,rbijxft7}_x.pth`

- **Analysis notebooks** with outputs:
  - `reconstruction_cifar10.ipynb` — CIFAR-10 reconstruction visualization & metrics
  - `reconstruction_mnist.ipynb` — MNIST reconstruction visualization & metrics

- **Datasets downloaded**: MNIST, CIFAR-10 (in `dataset_reconstruction/data/`)

- **Environment**: Apple Silicon / MPS backend via `environment_macos.yaml` (Python 3.8, PyTorch 2.4.1, Kornia 0.7.0, wandb)

### Thesis Planning — Complete

- Wrote comprehensive thesis prospectus covering 3 research directions (see `papers/Thesis Ideas_ LoRA, NTK, Reconstruction.pdf`)
- Formulated the Gradient Bridge attack (see `papers/Gradient Bridge_ PEFT Privacy Attack.pdf`)
- Created phased coding roadmap: Phase 0 → Phase 1 → Phase 2 (see `notes/GRADIENT_BRIDGE_PLAN.md`)
- Detailed R2F (Recover-to-Forget) reference analysis in `CLAUDE.md`
- Collected all key reference papers in `papers/`

### Project Organization & Infrastructure (2026-02-22)

Major setup day — went from a working base reconstruction to a fully organized thesis project:

**Repository structure:**
- Organized flat directory into structured layout: `papers/`, `figures/`, `results/`, `notes/`, `experiments/`
- Created `CLAUDE.md` with full project context, theoretical foundations, and R2F deep-dive
- Created `LESSONS_LEARNED.md` with base reconstruction insights
- Created this `STATUS.md`
- Set up `.gitignore` and initialized the Thesis-level git repo (separate from `dataset_reconstruction/`)
- Cleaned up `papers/`: removed 3 duplicate/corrupted files (84 MB of junk)

**Claude Code tooling:**
- Originally set up 10 custom skills: `/review`, `/supervisor`, `/experiment`, `/debug`, `/figure`, `/paper`, `/write`, `/lesson`, `/status`, `/project-manager`
- **Lost during data loss.** Recreated 2 commands on 2026-03-19: `/research`, `/project-manager`. Others need recreation if needed.

**Theoretical analysis documents (in `notes/`):**
- `R2F_Guide.tex/.pdf` — detailed walkthrough of the Gradient Decoder mechanism from R2F
- `Inversion_Feasibility_Analysis.tex/.pdf` — information-theoretic analysis of when reconstruction is possible
- `Thesis_Direction_Analysis.tex/.pdf` — comparison of all three thesis directions with risk assessment

### Sprint 1 Experiment Code (2026-02-22) — Complete

All infrastructure code written and debugged in `experiments/`:
- `lora_wrapper.py` — LoRALinear class, apply_lora, compose_state_dict
- `data_utils.py` — few-shot MNIST loading (train + test set), control images (in-dist + OOD)
- `train_lora.py` — LoRA + full fine-tuning training loops (full-batch SGD, BCE, float64)
- `ntk_steps.py` — multi-step gradient computation, NTK coefficient extraction
- `ntk_extraction.py` — NTK reconstruction loss with oracle and free-coefficient modes, N sweep
- `ntk_verification.py` — NTK diagnostics (weight change, feature stability, coefficient drift)
- `run_experiment_a.py` — convergence + compose experiment (pre-trained init)
- `run_experiment_b.py` — multi-step NTK experiment orchestrator
- `run_sweep.py` — sweep driver for both experiments (rank × N, rank × T)
- `metrics.py` — wrapper around existing evaluations.py (SSIM, DSSIM, NCC, L2)
- `plotting.py` — publication-quality figure generation (grids, heatmaps, diagnostics)
- `configs.py` — constants, sweep grids, device auto-detection
- 5 test files in `experiments/tests/`

**Key design decisions made during implementation:**
1. Experiment A consolidated to one script using pre-trained init (deleted duplicate `run_experiment_a_v2.py`)
2. All experiments use held-out MNIST test data for fine-tuning (not train set)
3. Device auto-detection: CUDA > MPS > CPU
4. Per-image SSIM scores on reconstruction grids (not just mean)

### Early Analysis Figures

Four plots in `figures/`:
- `parameters_as_function_of_epoch.png` — parameter dynamics over training
- `parameters_as_function_of_epoch_full_fine_tune_comparison.png` — LoRA vs full fine-tune comparison
- `parameters_as_function_of_epoch_with_sweet_spot.png` — optimal reconstruction window
- `experiment_b_grid.png` — NTK experiment preview grid

---

## Current Folder Structure (as of 2026-04-09)

```
/home/projects/galvardi/yoado/     ← WEXAC home = top-level git repo
├── .gitignore
├── CLAUDE.md
├── STATUS.md                      ← this file
├── LESSONS_LEARNED.md
├── STYLE_GUIDE.md
├── papers/                        ← reference PDFs
├── figures/                       ← 12 files (incl. Phase 0 results)
├── results/                       ← 105 files (.csv metrics + .pth tensors)
├── notes/
│   └── reconstruction_approaches.tex  ← catalog of approaches (March 2026)
├── scripts/                       ← 28 WEXAC job submission scripts
│   └── wexac_logs/                ← job stdout/stderr logs
├── experiments/                   ← 25 .py files (LoRA recon + Phase 0 ViT inversion)
│   └── tests/                     ← pytest test suite
└── dataset_reconstruction/        ← original Haim et al. codebase (separate .git)
```

---

## Thesis Roadmap (updated 2026-04-09)

Sprint 2 established the NTK attack on MNIST MLPs. The path forward has three tiers, ordered by thesis impact:

### Tier 1: Gradient Bridge (highest priority — core thesis contribution)
The LoRA → full gradient → image reconstruction pipeline:
1. **Sprint 3 (current)**: Scale gradient inversion to ViT/CNN on CIFAR-10 — establishes the inversion engine
2. **Sprint 4 (future)**: Train Gradient Decoder (R2F-style) — 50K (BA, ∇_W L) pairs from proxy data, per-layer MLP, cosine sim loss
3. **Sprint 5 (future)**: End-to-end attack — frozen decoder → inversion engine → reconstructed images on victim LoRA adapter

### Tier 2: NTK Reconstruction (supporting evidence)
Differentiable unrolling (S3.4) extends the NTK approach to exact multi-step matching, removing the linearization assumption. If it works on CIFAR-10, it's a publishable improvement over Sprint 2's NTK results and provides an alternative attack path.

### Tier 3: Diffusion Priors (stretch goal)
Hybrid gradient-matching + SDS loss for low-rank reconstruction. Blocked on having a working inversion engine (Tier 1). Target: face reconstruction from ViT LoRA adapters fine-tuned on CelebA.

---

## Known Issues & Housekeeping

- **Uncommitted changes** in `dataset_reconstruction/`: `wexac_connect.sh`, `wexac_disconnect.sh` modified — likely WEXAC config tweaks
- **`settings.default.py` deleted** from git tracking in `dataset_reconstruction/` — README expects it for fresh clone setup
- **Untracked large file**: `Miniforge3-MacOSX-arm64.sh` (51 MB installer) in `dataset_reconstruction/` — already .gitignored there
- ~~**Corrupted/duplicate PDFs** in `papers/`~~ — **FIXED** (2026-02-22): removed `2407.15845` and `Djdj .15845`, kept properly named `Oz_et_al_2024_Reconstruction_Transfer_Learning.pdf`
- **No `runs/` directory** yet — gets created at runtime by Main.py

---

## Pending Tasks

### Completed
- [x] **Run Experiment A on WEXAC** — FAILED (expected): KKT can't separate fine-tuning from pre-training
- [x] **Run Experiment B on WEXAC** — SUCCESS: SSIM=0.9999 (full) / 0.797 (LoRA r=8) — oracle coefficients
- [x] **Run rank sweep (Experiment B)** — ranks 8/16/32: SSIM improves with rank — oracle coefficients
- [x] **Multi-seed analysis (200 seeds)** — 11% of seeds produce strong signal; perfect correlation with model being wrong after centering
- [x] **Generate Sprint 1 figures** — `rank_sweep_sprint1.png`, `sprint1_summary.png`, `multi_seed_analysis.png`, experiment B grids
- [x] **Identify oracle-coefficient cheating** — current NTK extraction uses true x to compute cᵢ; documented in LESSONS_LEARNED.md
- [x] **Sprint 2b Phase 0: Activation ablation** — LeakyReLU wins (stable through T=100, ReLU NaN's at T>=50)
- [x] **Sprint 2b Phase 1: SGD + free-c baseline (ReLU)** — confirms ReLU instability at T>1
- [x] **Sprint 2b Phase 2: Random restarts (LeakyReLU)** — LoRA nearly matches full model through T=100
- [x] **Activation function ablation for LoRA extraction** — ReLU (alpha=10000) + L-BFGS best for LoRA
- [x] **Free-coefficient LoRA rank sweep** — works at r=8 and r=64, stubborn at r=16/r=32

### Sprint 2a: Free-Coefficient Extraction — DONE
- [x] **Implement free-coefficient NTK extraction** — `get_coeff_penalty()`, free-c mode, N sweep
- [x] **Ablate consistency weight α** — α=1 + SGD is optimal (SSIM=0.997, matches oracle)
- [x] **LoRA rank sweep with free-c** — r=4/8/16/32/64 with ReLU + L-BFGS (WEXAC jobs 674631, 681126)
- [x] **Activation ablation** — ReLU (alpha=10000) is critical for LoRA (WEXAC job 669885)
- [x] **Separate optimizer for c** — L-BFGS for x, SGD/Adam for c (mirrors Haim et al.'s λ handling)
- [x] **Fix r=16/32 convergence** — SGD+LeakyReLU: r=16 0.624, r=32 0.680 (was 0.42)
- [x] **Multi-seed comparison** — 50 seeds: free-c (0.557) beats oracle (0.408), 46/50 wins

### Sprint 2b: Multi-Step & Scaling
- [x] **Phase 0**: Activation ablation (3 activations × 5 T values)
- [x] **Phase 1**: SGD + free-c baseline (T × rank sweep)
- [x] **Phase 2**: Random restarts
- [x] **Phase 3**: LR scaling with LeakyReLU — done as Sprint 2c B1
- [x] **Phase 4**: Progressive warm-start — done as Sprint 2c B1
- [x] **Multi-seed validation** of LeakyReLU — 30 seeds: SSIM 0.558±0.034 (T=1), 0.572±0.088 (T=10)
- [x] **Per-image SSIM** — 10 seeds saved as .pth for visual inspection

### Sprint 2c: KKT & NTK Ablations
- [x] **Track A**: CLOSED — KKT loss 330-350 for all N values, confirms structural failure
- [x] **Track B1**: Phase 3+4 (LR scheduling + warm-start) — DONE
- [x] **Track B2**: Loss ratio ablation (verify_weight) — DONE (16 configs)
- [x] **Track B3a**: Optimizer × activation for LoRA — DONE (winner: SGD + LeakyReLU, SSIM 0.830)
- [x] **Track B3b**: Scale best combo across T — DONE (SGD+LeakyReLU ≡ L-BFGS for T≤20, NaN at T=100)
- [x] **Track B4**: N sweep (NTK) — DONE
- [x] **Track B5-B8**: Additional ablations — DONE

### Phase 0: ViT Gradient Inversion
- [x] ~~**Setup phase0 conda env**~~ — not needed, `rec` env has timm+peft
- [x] **Phase 0 (fixed)**: Resubmitted with bug fixes — SSIM=0.089 (full) / 0.264 (LoRA). Poor but real signal (was 0.015 before fixes).
- [x] **D1 controlled comparison** (2026-04-14): signAdam + tv=1e-2 → SSIM=0.144 (4 configs)
- [x] **D2 hyperparameter sweep** (2026-04-28): 40 configs, best tv=1e-1 + lr=0.05 + 30K → **SSIM=0.548** — gate crossed on Flowers102
- [x] **D3v2 freq/LPIPS prior ablation** (2026-04-28): 7 configs; priors don't help (best 0.558, within seed noise of TV-only D2)
- [x] **Face1 at D3 winner** (2026-04-28): real-face gate crossed — **SSIM=0.522, PSNR=13.8, cos_sim=0.974**
- [x] **Custom image loading**: `--image_path` flag + 7 unit tests
- [x] **Partial-save checkpoint hook** (2026-05-13): per-restart .pth save so 48h-wall kills leave usable data
- [x] **Supervisor handoff report**: `notes/phase0_report.tex` (350 lines, D1→D4) + `notes/phase0_last2days.md`
- [ ] **D4 face-structure prior sweep**: 9 arms running on WEXAC (jobs 777007-777019/777085-777095); analyzer ready
- [ ] **D5 chroma-coupled (LAB) TV**: 2 arms running on face1.jpg (jobs 777084/777086); chroma_weight ∈ {5, 20}
- [ ] **Multi-seed face1**: 5 seeds running (jobs 777058-777063) at D3 winner config — canonical SSIM mean±std
- [ ] **Re-run face2/face3 at D3 winner**: existing numbers (SSIM 0.21-0.24) are stale, from weak-TV March sweep
- [ ] **LoRA-only at D2/D3 winner**: rerun tv=1e-1, lr=0.05, 30K with --mode lora across rank 8/16/32/64
- [ ] **N>1 reconstruction (Phase 0)**: never run; folded into superposition work in CLAUDE.md
- [ ] **D6 (conditional)**: latent-space recon / SDS — only if D4+D5+multi-seed don't reach SSIM≥0.6
- [ ] **Phase 0b**: Noise tolerance sweep — deprioritized, folded into Sprint 3

### Sprint 3: Scaling Beyond MNIST
- [ ] **S3.1**: Phase 0 hyperparameter sweep (lr × tv_weight × optimizer × n_iters)
- [ ] **S3.2**: Low-dim reconstruction space (32×32 / Fourier / patch-aware)
- [ ] **S3.3a**: CNN baseline on CIFAR-10 (simplest architecture)
- [ ] **S3.3b**: ResNet-18 + LoRA on CIFAR-10 (skip connections)
- [ ] **S3.3c**: DeiT-Tiny on CIFAR-10 (small ViT, 5.7M params)
- [ ] **S3.4**: Differentiable unrolling — Phase 1: validate on MNIST (T=1 should match Exp B)
- [ ] **S3.4**: Differentiable unrolling — Phase 2: T=1,2,5,10,20 on MNIST vs NTK
- [ ] **S3.4**: Differentiable unrolling — Phase 3: apply to best CIFAR-10 architecture
- [ ] **S3.5**: Image priors (frequency, LPIPS, BN stats, SDS, latent-space)

### Research Backlog
- [ ] **Image priors for ViT inversion** — folded into S3.5; TV alone is insufficient at 224×224
- [ ] **N>1 superposition problem** — deprioritized until N=1 works reliably on CIFAR-10. Approaches: diversity penalty, ICA (Cocktail Party Attack), cross-gradient orthogonality
- [ ] **Read Gradient Inversion on PEFT (Sami et al., CVPR 2025)** — PEFT dimensionality reduction makes inversion *easier*; directly validates thesis. **HIGH PRIORITY** — read before starting S3.3
- [ ] **Read Cocktail Party Attack (ICML 2023)** — ICA-based gradient inversion, scales to N=1024 (needed for N>1)
- [ ] **Read SPEAR (NeurIPS 2024)** — exact batch recovery via SVD + ReLU sparsity

### Writing & Communication
- [ ] **Write LaTeX summary** — `notes/lora_reconstruction_writeup.tex` (after Sprint 3 results)
- [ ] **Email supervisor** with Sprint 2 + Phase 0 results and Sprint 3 plan
- [ ] **Verify figure quality** — publication-ready (axes, legends, DPI, colorblind-safe)

### Reading (Sprint 3 prep)
- [ ] **Read Inverting Gradients (Geiping et al.)** — the gradient inversion algorithm Phase 0 implements. **HIGH PRIORITY** — may reveal hyperparameter guidance we're missing
- [ ] Read R2F paper Section 3 in detail (decoder architecture) — needed for Sprint 4
