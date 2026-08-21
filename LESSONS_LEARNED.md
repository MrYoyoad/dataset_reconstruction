# Lessons Learned

Running log of insights, pitfalls, and things to remember as the thesis progresses.

---

## Raw SSIM on a clipped reconstruction is a metric artifact (2026-08-21, from sibling session)

- **What:** the extraction softly boxes only the CENTERED x∈[-1,1]; the DISPLAYED image x+ds_mean can
  leave [0,1] and get silently clamped before SSIM. On hard reconstructions this clips a LOT — measured
  on the bridge recons: MNIST 32-47%, Fashion 22-30%, monster all-layers 17-20% (flowers/easy cases <7%).
  Raw SSIM on such a clamped image is inflated/distorted; the sibling found it FLIPPED the sign of the
  Q-B seen-vs-novel result (not just the magnitude).
- **Robust metrics (unaffected):** `ssim_norm` (matches recon mean/std to the target before scoring),
  NCC, control margin, retrieval. Our bridge conclusions were on ssim_norm, so they held; but any raw-SSIM
  number was suspect.
- **Fix:** `--pixel_box` (run_ntk_extraction pixel_box=True + ds_mean) boxes x+ds_mean to [0,1] during
  extraction -> clipped_fraction ~0 -> raw SSIM trustworthy again. Default off (MNIST byte-identical).
- **Rule:** print/check `clipped_fraction` before trusting ANY free-c/raw-SSIM number; if >~0.05, use
  ssim_norm/NCC/margin or re-run with --pixel_box. Never rank reconstructions on raw SSIM alone.

## Scaling the reconstruction testbed to a deeper net: two lessons (2026-08-20)

### The tiny-init max-margin recipe does NOT transfer to a deep net (forward collapse)
- **Presented as:** a wide+deep MLP (3072-2048x4-1) stuck at loss=ln2, train-acc 0.508, margin 0 for
  40k epochs — with NONZERO gradients (so it looked like it was training but never moved). Root cause:
  the MNIST 2-layer recipe uses a tiny init (1e-4), and even PyTorch's DEFAULT Linear init undershoots
  the variance-preserving (Kaiming) scale by ~2.4x/layer; over 5 layers the forward signal COLLAPSES
  (logit std 0.002 = input-independent), so BCE sits at ln2. Fix: `kaiming_normal_(nonlinearity='relu')`
  restores logit std ~0.26 and the net trains to interpolation. **Apply:** when scaling a shallow-net
  recipe to depth, always use variance-preserving init; diagnose "stuck at ln2 with flowing grads" by
  checking the forward logit std, not the gradient norms.

### The gradient-bridge attack does not scale to network DEPTH (per-layer errors compound)
- **Finding:** on the 5-layer monster, every per-layer decoder trained well (0.86-0.96) AND direct
  inversion from the exact ΔW was near-perfect (gelu 1.000), but the end-to-end bridge (assemble all 5
  decoded layers -> extract) FELL BELOW baseline (0.30 vs 0.615). Signature: on shallow nets
  all-layers≈input-only; on the deep net all-layers (0.30) << input-only (0.56) — the decoded HIDDEN
  layers HURT. Assembling many imperfectly-decoded layers compounds error across depth. **Why it matters:**
  a bridge-specific limitation (direct inversion, using the exact ΔW, is immune) — the adapter-only attack
  weakens with depth even when each decoder is individually good.

## Math-quality PDFs on WEXAC: fpdf2 prose + matplotlib mathtext equations (2026-08-20)

### Context
Wrote a rigorous proof note (`notes/identifiability_rank_bound.pdf`, generator
`notes/make_identifiability_pdf.py`). LaTeX engines are all broken here (glibc), so the recipe is
fpdf2 for prose + matplotlib **mathtext** (`usetex=False`) for typeset equations rendered to PNGs and
embedded via `pdf.image()`.

### Pitfalls (found the hard way)
- **mathtext ≠ LaTeX.** It rejects `\big`, `\begin{pmatrix}`/`array`/`cases`, `\underbrace`, and
  `\ge`/`\le` (use `\geq`/`\leq`, `\left`/`\right`). Matrices must be drawn by hand (bracket lines +
  text) and h-composited with the text pieces via PIL.
- **fpdf2 `multi_cell` leaves the x-cursor at the RIGHT edge.** Two consecutive `multi_cell`s with no
  `ln()`/`set_x()` between them make the second start at the right margin and run off-page (this was the
  title→subtitle overflow). Always `set_x(left_margin)` before each stacked `multi_cell`.
- **Inline math in prose must be converted to Unicode**, else `w_k`/`c_i`/`∇_W L` render as literal
  underscores next to the crisp equations. A `mathify()` regex maps `_x`/`_{..}`/`^T`/`^d` to Unicode
  sub/superscripts. DejaVu covers subscript i,j,k,l,m,n,x + digits + superscript T,d,n,m — but **no
  subscript 'c' or capital N/W**, so reword `D_c`, `x_N`, `∇_W L` rather than emit a missing-glyph box.
  Verify coverage with fontTools `getBestCmap()` first, then re-scan the built PDF text for any raw
  `_`/`^`.
- **Never `set_y()` with a y captured before a possible page break.** A two-column table row that did
  `y0=get_y(); multi_cell(...); set_y(y0+h)` cascaded to one row per page once earlier content pushed
  the table near the bottom: the `multi_cell` auto-broke to a new page, then `set_y(y0+h)` forced the
  cursor back to the *old* page's bottom y on the new page → infinite one-row-per-page. Fix: page-break
  *before* drawing the row (`if get_y()+rowh>h-margin: add_page()`), capture `y0` after, and advance
  with `set_y(max(col_ends, y0+h))`.
- **Verify layout blind-spots with pymupdf** (`pip install pymupdf`): render pages to PNG and flag any
  text block whose right edge exceeds the margin, AND scan per-page text length for near-blank pages
  (a page-break cascade shows up as many ~5-char pages). The Read tool can't rasterize PDFs (poppler
  absent); pymupdf can.

### Apply
Reusable generator at `notes/make_identifiability_pdf.py` is portable (matplotlib font dir + `tempfile`
scratch + `__file__`-relative output). Same pattern for any math-heavy PDF here. See also the
`reference_pdf_generation_method` memory.

---

## Raw-SSIM on a clipped reconstruction is a metric artifact, not a leakage result (2026-08-20)

### The bug
- The Q-B seen-vs-novel gap (seen SSIM > novel) was partly a **clipping artifact**: the extraction
  only bounds the *centered* variable `x∈[-1,1]` via `get_ntk_verify_loss`, but the DISPLAYED image is
  `x+ds_mean`. Nothing stopped `x+ds_mean` from leaving `[0,1]`, so the novel arm clipped ~50% of its
  pixels on display → raw SSIM collapsed while `ssim_norm` (scale-invariant) barely moved (seen 0.580
  vs novel 0.495, a much smaller gap). Reporting raw SSIM alone made the gap look bigger than it is.
### The fix
- Added `get_pixel_box_loss(x, ds_mean)` + a `--pixel_box` flag (`ntk_extraction.py` /
  `run_experiment_b.py`): penalize the **image** `x+ds_mean` leaving `[0,1]` directly (weighted by
  `--verify_weight`), not just the centered `x`. `build_base_name` appends `pbox` so the clean run
  never collides with the old clipped Q-B `.pth`. Default off → all existing paths byte-identical.
### Apply
- **Never trust raw SSIM when `clipped_fraction` is non-trivial.** Pair it with `ssim_norm`/NCC, and if
  a natural-image reconstruction clips, constrain the *image* `[0,1]`, not the centered variable.
### Result: fixing the clip REVERSED the Q-B conclusion (job 952081)
- The old clipped run said "seen (overlap) leaks more than novel." With the proper `[0,1]` box the
  novel arm stopped clipping and the direction flipped: **novel leaks MORE** (ctrl margin +0.42 vs
  +0.26; NCC-dist 1032 vs 5044; ssim_norm 0.53 vs 0.48), because novel species produce a ~4-5x larger
  `weight_change` (0.160 vs 0.035) that carries the specific instance, while overlap leaves only a tiny
  class-generic residual. A metric artifact didn't just add noise -- it inverted the headline. Lesson:
  a clipping artifact can flip the SIGN of a comparison, not merely its magnitude; fix the box before
  drawing any seen-vs-novel / overlap conclusion.
### Audit: which OTHER experiments are affected (2026-08-20)
Clipping tracks reconstruction DIFFICULTY (poor/large-dW reconstructions leave [0,1]; good ones don't),
so only the hard-case experiments are contaminated. Checked `clipped_fraction` per config across the
flowers-native free-c logs:
- **SAFE (clip < 0.05, raw≈ssim_norm):** activation ranking (flowers32 & flowers64, max clip 0.047),
  the rank/leakage curve r=4..64 (clip ~0.002), and the Q-A dimension ladder 32 vs 64 (clip ~0.000).
  These headline conclusions stand as-is.
- **N-sweep (npc>=2, i.e. N>=4): PARTIALLY affected.** clip 0.10-0.47 (npc=4 clipped 47%). The N>=4
  COLLAPSE is real -- it shows on the scale-robust `ssim_norm` (0.68 at N=2 -> ~0.12) AND the control
  margin (+0.30 -> +0.01), which clipping can't fake -- but the ABSOLUTE numbers at N>=4 are depressed
  by the clip. A `--pixel_box` re-run would give an honest (still-declining) N curve; the direction
  will not reverse (unlike Q-B) because both scale-robust metrics already collapse.
- **Optimizer axis (adamw): doubly confounded, unreliable.** adamw configs have `weight_change`=0.60
  (far out of the NTK band) AND ~30% clipping AND a raw-vs-norm gap -- discard for any leakage claim.
- **Q-B: fixed** (job 952081, sign flipped).
Apply: before trusting ANY free-c leakage number, print `clipped_fraction`; if >~0.05 in the configs you
compare, either read only `ssim_norm`/NCC/control-margin or re-run with `--pixel_box`. High clip is a
symptom of a hard reconstruction (superposition at high N, large dW), so it clusters exactly where the
result is most fragile.
### Correction: the [0,1] box HELPS the joint path but BLANKS the sequential-peel path (2026-08-21)
Re-running the N-sweep with --pixel_box exposed two mistakes:
1. **Audit over-generalized the clip.** The 0.10-0.47 clip was the SOFTPLUS N-sweep (nrank_665601).
   The DEFAULT-recipe N-sweep (main_427349) was already clip-clean at N>=4 (npc=4 clip 0.009, npc=8
   0.001). Only the moderately-clipped joint config npc=2 (clip 0.104) actually needed a box. Lesson:
   attribute a clip level to the SPECIFIC recipe that produced it, don't generalize across activations.
2. **--pixel_box at verify_weight 5.0 over-constrains the low-signal sequential-peel path.** Peeling
   recovers one weak source at a time (true c~0.06 for N=8); the box penalty (sum over 3072 pixels,
   weight 5.0) dominates the tiny NTK signal and drives the peel coefficients to ~0 (c~0.005), blanking
   the reconstruction: margin +0.045/+0.085 (old, unboxed, already clip-clean) -> ~0 (boxed). Even
   though the *final* clean solution sits inside [0,1], the box blocks the optimization path to it.
   FIX: box the joint configs (npc 1,2, where clip is real and the box HELPS: npc=2 margin
   +0.028 -> +0.176), do NOT box the already-clean peel configs (npc 4,8). A penalty box is only safe
   when the data term is strong enough to compete with it; for a weak/underdetermined reconstruction a
   hard clamp (project x+ds_mean to [0,1] each step) would be the non-interfering alternative.

## GB-Phase 2 end-to-end: the inverter matters more than the decoder (2026-08-19)

### The gradient-bridge headline: SVD is the wrong inverter; the base model IS the prior
- **Insight:** turning a decoded LoRA gradient into an image is NOT a factorization problem, it is a
  gradient-INVERSION problem. The naive SVD (top singular vector of the decoded input-layer gradient)
  gives only a coarse blob (SSIM 0.10-0.17) even when the decoder cosine is 0.945 -- because SVD is a
  prior-free DETERMINED rank-1 factorization (no null-space for a prior). Feeding the SAME decoded ΔW
  into the model-based `run_ntk_extraction` (Experiment B) jumps to **SSIM ~0.5 / ssim_norm 0.62-0.74,
  3-5x the SVD** -- the known θ₀ + all-layer structure act as an implicit prior. This is the deck's Q-A
  lesson made literal: at a fixed high cosine, the MODEL/PRIOR is the lever, not the cosine.
- **Apply:** for any "decoded/approximate gradient -> image" step, invert THROUGH the known base model,
  never by bare SVD/pseudo-inverse. The decoder cosine is necessary but not sufficient; the inverter
  converts cosine into pixels.

### Pitfall: `generate_pairs` sets the default dtype to float64 at IMPORT -> decoder built in float64
- **Presented as:** `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and Double` on
  the first decoder training batch, before any result. Root cause: `generate_pairs.py` runs
  `torch.set_default_dtype(torch.float64)` at module import, so `GradientDecoder(...)` was constructed in
  float64 while `train()` casts its data to `.float()` (float32). Fix: set float32 default for the
  decoder-training phase (matches phase2_image; the loaded model stays float64 internally so banks are
  still float64), then switch to float64 only for the victim measurement + extraction (matches
  phase2_full). Toggle the default dtype PER PHASE.

### Pitfall: decoder outputs +∇W but the weight update is ΔW = -lr·∇W (sign flip)
- **Presented as:** aggregate decode cosines of -0.99 for the near-perfect hidden/output layers (looked
  like catastrophic decode; was actually a perfect decode with the wrong sign), and DECODED all-layers
  scoring BELOW DECODED input-only because the sign-flipped layers fought the inverter. Fix: align each
  decoded layer's aggregate sign to the true ΔW of that layer (an oracle SIGN only, consistent with the
  oracle-coefficient upper-bound framing). The per-sample decode DIRECTION is the real quantity; the
  global sign is a convention nuisance.

### N=2 opposite-label aggregation cancels the SIGNAL, not the decode error
- **Observation:** per-sample the input decoder is 0.92 (softplus) / 0.67 (gelu), but the AGGREGATE
  decode cosine over the two opposite-label victim samples is only 0.42 / 0.37. Summing two samples with
  opposite `g_err` signs shrinks ‖Σ true‖ (signal cancels) while the decode errors do not, so the
  aggregate cosine falls well below the per-sample cosine. This aggregate input-layer fidelity -- not the
  hidden/output decoders (near-perfect, 0.99, and useless beyond input-only) -- is what caps the
  end-to-end reconstruction below the true-ΔW ceiling. To close the gap: a stronger input decoder, or
  N=1 to remove the cancellation.

## Flowers-native track: dim-threading, FP64 GPU, RGB plotting (2026-08-13)

### Threading a new input_dim through ~10 create_model/load_pretrained call sites — use local shadowing, not 10 edits
- **Design decision:** `run_experiment_b.run_single_config` builds the model at ~10 sites (both FT
  paths, lin-error models, extraction models, nested `_make_model` closures). To run at D=3072/12288
  without editing every site, the module functions were split into `_build_network`/`_load_theta0`
  (accept `input_dim`/`hidden`) + public `create_model`/`load_pretrained` shims; then
  `run_single_config` defines **local closures named `create_model`/`load_pretrained`** that bind the
  dataset's dims and call the `_build_*` base. All existing call sites resolve to the locals via lexical
  scope — zero call-site edits, no recursion (the closures call the differently-named base). Public API
  stays intact for tests. Pattern to reuse when a config value must reach many call sites in one function.

### The pipeline is FP64 → request an A100, not the shared-queue default
- **What:** `--precision=double` + `configs.get_dtype` force float64 on CUDA. The `long-gpu` default
  `GPU_REQ` is `j_exclusive=no:gmem=6248` — a *shared* card, often a poor-FP64 A40/L40S (~0.6 TFLOPS
  FP64). A100 is ~9.7 TFLOPS FP64 → **~15× faster** for this workload.
- **Fix:** `#BSUB -gpu "num=1:j_exclusive=yes:gmem=16000:gmodel=NVIDIAA100_SXM4"`. Discovery:
  `bhosts -gpu` (idle cards = NJOBS 0), `lsload -gpu`, `bqueues -l long-gpu | grep GPU_REQ`. Full recipe
  in the `reference_wexac_good_gpu` memory.

### RGB broke only plotting; metrics were already channel-safe
- **What:** kornia SSIM, NCC, mean-baseline, ds_mean all reduce/flatten channel-agnostically, so RGB
  needed **no** metric changes. The only break was `imshow` on a (3,H,W) array. **Fix:** squeeze just
  the batch dim (`ds_mean[0]`, not `.squeeze()` which would drop a channel), then CHW→HWC transpose for
  3-channel and drop `cmap='gray'`.

### Free-coefficient reconstruction: the RECIPE, and read settings off the artifact before theorizing (2026-08-18)
- **What broke:** re-running the flowers sweeps in `--free_coefficients` (the realistic Haim attack)
  produced garbage — NTK loss stuck flat at 9.6, coefficients **sign-flipped** (`[-0.68, +1.00]`,
  c_err 0.71), ~0 SSIM, and ~11 h/config (never converges → runs to the epoch cap).
- **How it presented / my error:** I first *theorized* it was fundamental (outside the NTK regime, too
  high-dimensional). The user correctly pushed back — they remembered free-c *working* (0.686 on MNIST,
  the numbers shown to Gal). It was **all-default settings**, not a real limitation.
- **Root cause:** three defaults are wrong for free-c, all documented in STATUS's Sprint-2 section:
  (1) `consistency_weight=0` → the sign-flip local minimum (the consistency penalty
  `‖c − (σ(f(x))−y)/N‖²` is what prevents it); (2) `relu_alpha=149` (ModifiedReLU) — STATUS: "ModifiedReLU
  actively harms extraction" (0.183 vs 0.744); (3) `optimizer=lbfgs` — LBFGS overfits x to the current c
  and stalls; the working runs used **SGD**.
- **The fix (verified, reproduces Sprint-2 ~0.59):** `--optimizer sgd --relu_alpha 10000
  --consistency_weight 1.0 --n_restarts N`. On flowers32 this lands 0.60–0.65 (within ~0.04 of oracle).
- **Meta-lesson (the important one):** when reproducing a documented result, **`torch.load` a known-good
  saved `.pth` and read its `config` FIRST** — the working free-c files literally had `optimizer='sgd',
  relu_alpha=10000` in their config. That one look would have skipped hours of wrong theorizing. Don't
  theorize a mechanism when the ground-truth settings are sitting on disk.

### `--pretrained_path` must use the full `dataset_reconstruction/models/...` path (2026-08-16)
- **Bug:** the Phase-D Q-B job script passed `--pretrained_path models/weights-flowers32_holdout.pth`
  (relative to the repo-root cwd), but the models dir is `dataset_reconstruction/models/`. Stage 0's
  `torch.load` hit `FileNotFoundError` and the job aborted in 18s (`STAGE 0 FAILED`).
- **Why the other sweeps were fine:** they pass only `--dataset flowers32` and resolve the checkpoint
  through `DATASET_SPECS[...]['pretrained']` = `os.path.join(MODELS_DIR, ...)` (correct absolute path).
  Only the Q-B script hardcoded a relative `models/` path for the holdout θ₀.
- **Fix / rule:** any explicit `--pretrained_path` must be `dataset_reconstruction/models/<file>.pth`
  (or better, `configs.MODELS_DIR`). `models/` is NOT at the repo root.

### Figure-clobber (LESSONS 2026-07-21) now actually fixed
- `run_experiment_b.__main__` called `generate_experiment_b_figure(results)` **unconditionally**,
  rewriting `figures/sprint1/experiment_b_grid_oracle.png` on every run (incl. smoke tests). Removed the
  unconditional call; figures are written only under `--save_results`, and non-mnist datasets route to
  `figures/sprint1/<dataset>/`.

---

## Activation rescore (job 857271): softplus wins, but the whole sweep is sub-NTK (2026-08-13)

### The "matched weight_change" comparison was in a degenerate regime — a directional read, not a verdict
- **What:** rescoring the 21 activation tensors gave a clean, unanimous ranking (softplus ≫ silu >
  gelu ≈ gelu_tanh > mish > elu, across ssim/ssim11/ssim_norm/l2/ncc/clip/control-margin **and**
  feature_stability). But **every** config is `ntk_passed:False` with `delta_w_effective_rank = 1–2`
  for a rank-8 adapter — the fine-tune barely moved the net and the update is essentially rank-1.
- **Why it matters:** the ranking is a *first-pass direction*, not a confirmed result. The LR grid
  {0.01,0.03,0.1,0.3} straddled the usable band — low LR gives `weight_change`≈0.04 (barely trained),
  high LR gives 1–3.7 (far past NTK), and none lands in-regime. **Confirm any activation claim with a
  target-`weight_change` re-run that actually reaches `ntk_passed:True`, multi-seed, before quoting it
  to Gal.** (This is Step 2a.)
- **The positive signal that IS robust:** softplus's reconstruction is invariant to **1.7e-4** across a
  10× `weight_change` range (0.038→0.379) then breaks at 1.14, while gelu's shifts ~0.26 over the same
  LRs. That LR/weight-change invariance is a genuine **linearization-stability** property (at T=1,
  Δw direction is LR-independent; a cleanly-linearizing activation recovers the same x across the whole
  range where the linearization holds). Consistent with "smoother ⇒ more linearizable" — the crux hypothesis.

### Two small gotchas found while wiring the rescorer
- **`delta_w_effective_rank` is an `int`, and a float-only print formatter silently showed it as `-`.**
  `f"{v:.4f}" if isinstance(v, float) else "-"` drops ints. Handle `int` explicitly (or use `float(v)`).
  The value was in the CSV all along; only the console table hid it. An effective rank of 1–2 for a
  rank-8 adapter is a red flag on its own — do not let a formatting bug hide it.
- **Direct-inversion `.pth` files save no `x_ctrl`.** So the DI control margins (+0.049/+0.058) quoted
  in the 2026-07-22 metric box are runtime-only, **not** re-derivable from disk — `recompute_metrics`
  / any offline rescore can only get ssim_norm/mean-baseline for DI, not the control margin.
  `direct_inversion.py` should persist `x_ctrl` like `run_experiment_b.py` does, or those margins stay
  unreproducible.

---

## Building Addition 3 + DI-Phase 0 + GB-Phase 1 (2026-07-22)

### Anchor α-sweep: match the FULL Δw, not a residual (design decision)
- **What was almost wrong:** an early plan matched the *residual* displacement `(1−α)·Δw` in the
  reconstruction. That rescales the observable and introduces an ambiguous "effective step count."
- **Correct formulation:** the spec says the inversion linearizes around θ_anchor and uses
  `∇Φ(θ_anchor)`. So **only the linearization point moves** (gradient features + recomputed oracle
  coefficients at θ_anchor); the reconstruction still matches the full, observed `Δw`, with `lr` and
  `T` unchanged. This reduces to the current code **exactly** at α=0 (verified bitwise), which is the
  built-in sanity check. The `(1−α)` factor belongs only inside the *function-space* lin-error
  diagnostic (its Taylor direction `δ = θ_T − θ_anchor = (1−α)Δw`).
- **Two lin-errors, not one:** the existing `compute_linearization_error` is *weight-space*
  (`‖Δw − (−lrT·Σc∇f)‖/‖Δw‖`); the plan's deliverable is the *function-space* Taylor residual on Φ.
  They are different metrics — ship the function-space one as the headline, keep weight-space as a
  companion.

### Sanity-baseline pitfall: DI-T1 vs the RIGHT Experiment-B mode
- **Bug (spurious "large gap"):** the T=1 sanity compared DI-T1 (SSIM 0.57) to Experiment-B
  **free-coefficient** LoRA at only 2000 extraction epochs (SSIM 0.067) and flagged a 0.47 gap.
- **Root cause:** free-coefficient LBFGS needs ~50k epochs to converge; at 2k it badly under-reports.
  The gap was an artifact of the *baseline being undertrained*, not a bug in `F`.
- **Fix:** compare against Experiment-B **oracle** (fixed coefficients) — fast, well-defined, ~0.50.
  DI-T1 (0.57) ≳ oracle (0.50) is the *expected* result: DI matches endpoints exactly while NTK is
  linearized, so DI should sit at or slightly above the oracle. `F` itself is bit-exact at T=1.

### Gradient bridge: degenerate near-zero-gradient pairs poison the bank
- **Symptom:** projected-cosine ceiling for the output layer came out 0.865 instead of 1.0.
- **Root cause:** proxy samples the base model already classifies confidently have `|g_err|≈0` →
  per-sample gradient ≈ 0 → cosine undefined (~0), dragging the mean down. Such pairs also carry no
  LoRA signal for the decoder.
- **Fix:** filter pairs by per-sample gradient norm (`grad_tol`), oversampling the proxy pool to still
  reach `n_pairs`; log the survival rate (no silent capping). After filtering, layer-2 ceiling = 1.000.

### The col(B₀) subspace ceiling is the real GB milestone risk
- A single-step LoRA adapter at A=0 observes `∇_W L` only through `col(B₀)` (since `∇_A L = scaling·B₀ᵀ∇_W L`).
  So the full-gradient cosine is capped near `√(r/out)` unless the decoder hallucinates the rest.
- **Output layer (out=1)** is trivially invertible (col(B₀)=R¹, cosine 1.0) → *weak evidence only*.
- **Hidden layer (out=1000, r=8)** ceiling ≈ 0.089 — the honest bar. **Always report full-cosine vs
  projected-cosine** so a ">0.9" claim can't hide behind the near-analytic layer. Use a stable QR
  projection (`Q Qᵀ grad`), not a gram pseudo-inverse — `B₀ᵀB₀` is rank-deficient when out < r.

### Local (login-node) CPU is unusable for real compute — validate on GPU
- Multi-step **full** fine-tuning on the shared login node ran ~**10 s/step** (T=10 → 99.5 s) — ~100×
  slower than an L40S — pure thread-thrash on a loaded node. Confirms the "GPU only" rule.
- **Do validate logic locally** with fabricated/tiny inputs (anchor bitwise-equality, `F` bit-exactness,
  pair self-consistency all ran in 1–15 s) — just never real sweeps.
- **LSF buffers `python -m` stdout** (output appears only at job end); use `python -u` for jobs you
  need to monitor live. A **stuck** job shows ~5 MB host RAM / 1 thread / 1 PID in `bjobs -l` — a
  running PyTorch+CUDA process uses hundreds of MB, so that reading means "not computing."
- **Intermittent WEXAC startup hang — check every job ~15 min after it starts RUN.** Three jobs this
  session (452468, 877297, 886406) landed on a bad node and **hung before Python even started**:
  `bjobs -l` shows **~5–8 MB MEM, 1 thread, ~0.17 CPU-sec**, and the `.out` is **empty even with
  `python -u`**. One held a GPU idle for its full 8 h `-W` wall before being killed. Likely cause: a
  `conda activate` / `source conda.sh` stalling on a node with bad NFS to the shared env. **Detection:**
  empty `-u` stdout + single-digit-MB MEM after ~15 min = hung (a live job streams and uses hundreds of
  MB). **Fix:** `bkill` + resubmit to get a fresh node (most reruns work — it's node-specific). Don't
  wait on it; and after killing a hung job, verify the *result files* are absent before assuming the run
  produced nothing.
- **`long-gpu` RUNLIMIT is 96 h and big staged sweeps WILL hit it** — jobs 857271 and 863020 both died
  at TERM_RUNLIMIT (2026-07-26) mid-sweep. `--skip_if_exists` made them resumable, but nobody resubmitted,
  so the tail stages silently never ran for 2+ weeks. Two fixes: (a) order stages so the highest-value
  work lands first (this saved 863020 — its key stages finished), and (b) after ANY long job ends, check
  the `.out` for `TERM_RUNLIMIT` vs `Successfully completed` before treating the sweep as done.

### Two empirical findings worth remembering
- **Anchor has an interior optimum with a hard cliff.** SSIM(α) rises to **α≈0.75** (LoRA 0.06→0.64,
  full-FT 0.80→0.94) then **collapses at α=0.9** (full-FT 0.48, below the α=0 baseline). The rise
  tracks the falling linearization error; the collapse is the identifiability-degradation regime
  (anchor absorbs θ_T's training signal). **Cap α ≤ ~0.75.** SSIM peaks *before* the lin-error minimum,
  so the gain is a linearization win, not x_i leakage. (Single seed, T=10 — harden before the thesis.)
- **More LoRA rank does not improve the gradient bridge.** Decoder full-cosine is **flat at 0.685**
  across r=8/32/64 while the measurement ceiling rises √(r/1000). So most of the 0.685 comes from the
  **proxy prior / decoder**, not the measurement — the bridge is prior-limited, not bandwidth-limited.
  Don't reach for rank to hit 0.9; try decoder capacity, nonzero-A (two-sided) measurement, or
  multi-sample gradients instead. And **always report full-cosine vs the projection ceiling** — 0.685
  reads as "meh" until you see the measurement only afforded 0.086.

### An SSIM is not leakage until it beats the dataset-mean baseline
- We reported direct-inversion SSIM ~0.55 and anchor-LoRA up to 0.64 as "recovery." Re-scoring against
  `ssim_mean_baseline` (the trivial dataset-mean predictor) killed both claims: **DI (N=4) never beats
  0.674; anchor-LoRA (N=2) never beats 0.763.** Only anchor **full-FT** clears its baseline (and
  `ssim_norm` confirms it's structure, not brightness).
- **Small N makes the mean a brutal bar.** With N=2 the dataset mean is (img₁+img₂)/2 — already very
  similar to each image — so the baseline is ~0.76 and almost nothing beats it. Any leakage claim at
  small N must clear the mean *and* preferably use larger N so the bar is meaningful.
- **Watch the clip fraction.** Extraction only *softly* constrains x to [-1,1], so 0.37–0.67 of
  `x_recon+ds_mean` was saturating out of [0,1] here — that shared clamped background inflates raw SSIM.
  Report `clipped_fraction`; a high value means the absolute SSIM is not trustworthy.
- Rule: **report SSIM alongside (a) the mean-baseline, (b) `ssim_norm`, and (c) the clip fraction** —
  and, per the older note, weight_change / effective_rank. A bare SSIM proves nothing.
- **BUT the mean baseline is not the right bar for *instance* leakage — the same-class control is.**
  "Beat the dataset mean" and "match the true image better than a different same-class image" are
  different questions; the second (the decision brief's B1 gate) is what "leakage" actually means, and
  its **margin (recon-vs-true − recon-vs-control) cancels the shared clipping/scale**, so it's far more
  robust at small N. This flipped a conclusion: LoRA-only *fails* the mean baseline at every α but
  *passes* the control test at α≥0.75 (+0.14) — the anchor **creates** adapter-only instance leakage.
  Lesson: when a mean-baseline result looks negative at small N, re-check against the control before
  concluding "no leakage" — and prefer the control margin as the headline for instance-recovery claims.
  - **⚠ RETRACTED (2026-08-13): "the anchor *creates* LoRA leakage" was seed-42-specific, not robust.**
    Replication (job 863020, STATUS.md Track 1) showed seed 44 leaks already at α=0 (+0.18) with the
    anchor *hurting*, and the N=10 rescore (QW3) gives tiny LoRA margins (+0.006–0.008) with no α trend.
    Adapter-only leakage is real (control margins +0.13–0.18 across configs → B1 passes), but its
    α-dependence is config-dependent — needs the multi-config anchor study before any "anchor creates
    leakage" claim. The *lesson above* (prefer the control margin) still stands; only the anchor-α claim
    is withdrawn.

---

## Infrastructure & Data Loss (2026-03-19)

### The Git Repo That Never Was
The git repo WAS initialized and pushed to `myfork/main` on GitHub — but the WEXAC working directory lost its connection to the remote. The `.git` was either deleted during a reprovisioning or never properly set up on WEXAC. The WEXAC copy had newer experiment code/results (Sprint 2+) that were never pushed, while GitHub had all the papers, notes, and figures.

**What was actually lost:**
- All Claude Code conversation history from Jan 15 – Mar 18
- 10 custom Claude Code skills (8 still need recreation)
- Some generated figures not on GitHub (`multi_seed_analysis.png`, `sprint1_summary.png`, etc.)

**What was recovered from GitHub:**
- All 18 papers, 7 notes files, 4 figures, full experiment infrastructure

**Lessons:**
1. **Always verify git operations actually succeeded.** `git init` + `git add` + `git commit` — check `git status` after each.
2. **Push to GitHub after every significant commit.** The remote saved everything this time.
3. **WEXAC home dirs can lose local state.** Untracked files are unprotected files.
4. **Claude Code conversation history is ephemeral.** Important findings must go into STATUS.md / LESSONS_LEARNED.md.
5. **Don't have nested .git repos.** `dataset_reconstruction/` having its own git caused confusion. The top-level repo now tracks it as regular files.

---

## Base Reconstruction (Haim et al.)

### Setup & Environment
- Apple Silicon (MPS backend) works but watch for dtype mismatches — MPS doesn't support all float64 ops.
- The `settings.py` file with relative paths (`./data/`, `./runs/`, `./models/`) keeps things portable.
- **Primary compute is WEXAC cluster**, not the MacBook. GPU: NVIDIA L40S (46 GB VRAM), CUDA 12.6. Connect via `wexac_connect.sh` (requires Weizmann VPN). Conda env on cluster: `/home/projects/galvardi/yoado/.conda/envs/rec`.
- The L40S easily handles all planned experiments: ViT LoRA fine-tuning (~2-3 GB), gradient decoder training (~50k pairs), gradient inversion (~4-8 GB), and even Stable Diffusion for SDS priors (~8-12 GB). **Compute is not a bottleneck.**

### Training
- Models need to train to near-stationarity (very long — 1M epochs) for the KKT conditions to hold. Don't cut training short.
- `ModifiedReLU` is critical — standard ReLU gives much worse reconstruction because the smooth gradients matter during extraction.
- BCE loss (not cross-entropy) is required for the implicit bias / max-margin convergence theory to apply.

### Reconstruction
- KKT loss optimization is sensitive to initialization — random restarts help.
- Lambda (Lagrange multiplier) optimization needs a separate, typically smaller learning rate.
- The number of reconstructed samples should match the actual training set size for best results.

---

## LoRA / Gradient Bridge

### Key Realizations
- The Gradient Bridge is theoretically sound and all building blocks exist independently (R2F for decoding, Inverting Gradients for inversion). But **"building blocks exist" ≠ "easy to do"** — the real research risks are empirical, not computational:
  1. **Decoder accuracy for pixel-level reconstruction**: R2F proved the decoder works for unlearning (tolerant of noisy gradients). Nobody has shown it works for pixel-level image reconstruction, which is far more sensitive to gradient noise. Even 0.9 cosine similarity may not be enough.
  2. **Multi-step accumulation**: The decoder is trained on single-step LoRA updates. Real adapters train for thousands of steps. How to handle accumulated updates is an open question.
  3. **Error compounding**: LoRA approximation × decoder approximation × inversion approximation — each stage is "pretty good" but errors multiply through the pipeline.
- These are answerable by running experiments, and we have the compute (L40S) to run them fast.
- The correct strategy is to de-risk in order: (1) Sprint 1 compose-and-reconstruct, (2) Phase 0 "cheating" with perfect gradients to find the ceiling, (3) only then attempt the decoder. If Phase 0 fails, the decoder won't save it.

### What Worked
- **Experiment B (NTK, 1-step) works because it targets ΔW not W.** By reconstructing from the weight *change* (θ_T - θ₀), the pre-trained component cancels out. Full model SSIM=0.9999, LoRA rank 8 SSIM=0.797, rank 16 SSIM=0.802, rank 32 SSIM=0.826. This proves the gradient from a single fine-tuning step leaks private data.

### What Didn't Work
- **Experiment A (compose + KKT) completely fails with pre-trained init — and the reason is structural, not just slow convergence.** The composed model W = W₀ + BA is just a set of weights. The KKT reconstruction asks: "what training data would produce these weights as the max-margin solution?" The answer is: **all 502 samples** the model was effectively trained on (500 pre-training + 2 fine-tuning). The KKT stationarity condition is W ∝ Σᵢ₌₁⁵⁰² λᵢ yᵢ ∇_W Φ(W; xᵢ), but the extraction sets `extraction_data_amount = 2` — asking 2 images to explain weights that encode 502 images of information. The pre-training residual W₀ contains contributions from ~100-250 original support vectors. The KKT loss of ~460 is essentially ||W₀||² — the huge unexplained pre-training component. **Even with perfect convergence, the extraction would still fail** because the composed weights satisfy KKT with respect to all 502 samples, and 2 images can't explain them. (Note: the 2 fine-tuning samples ARE on the margin for the N=2 case — with 2M+ params and 2 points, both are necessarily support vectors. The issue is the other 500 samples baked into W₀.) **This is the key negative result that motivates the Gradient Bridge.**
- **Reconstruction is sensitive to seed.** With seed=42, full model NTK reconstruction achieves SSIM=0.9999. With seed=32, SSIM=0.378 despite perfect NTK conditions (feature_stability=1.0000). The extraction optimizer gets stuck in local minima depending on which digits are selected and how x is initialized. Random restarts are essential.

### Pitfalls to Avoid
- **W₀ must be pre-trained, not random.** The original Experiment A design used random init as W₀. This doesn't match the thesis's attack model (pre-trained model → LoRA fine-tune on private data → attacker reconstructs). Fixed by updating `run_experiment_a.py` to load the pre-trained MNIST model as W₀. The same lesson applied to Experiment B (random init destroys NTK feature stability). In both cases: **always start from pre-trained weights** — that's the realistic scenario.
- **Held-out data, always.** Fine-tuning data must come from the MNIST test set, not the train set. The pre-trained model already converged on its training data (gradients ≈ 0), so fine-tuning on overlapping data is meaningless.
- **Max-margin theory assumes small init.** The pre-trained model has layer weights with std 0.004–0.12, much larger than the 0.0001 init the theory requires. The reconstruction may fail in A-v2 for this reason — which would actually strengthen the argument for the Gradient Bridge approach (Phase 1-2).
- **Consolidate, don't duplicate.** Initially created `run_experiment_a_v2.py` as a separate script for the pre-trained init scenario. But then `run_experiment_a.py` was also updated to use pre-trained init, creating an identical duplicate. Deleted A-v2; keep one canonical script per experiment.

---

## NTK Regime & High-Rank LoRA

### Key Realizations
- **Start from pre-trained weights, not random init.** The original Experiment B design used random init with `init_scale=0.0001`, which put ALL pre-activations within ±0.002 of the ReLU kink. After 1 SGD step, ~half the neurons flipped on↔off, destroying NTK feature stability (cosine similarity 0.39 instead of >0.99). The NTK approximation was completely invalid, and all reconstructions were noise (SSIM ~0.35).
  - **Root cause**: `init_scale=0.0001` was inherited from the convergence/max-margin pipeline (Experiment A / Haim et al.), where tiny init helps the implicit bias theory. But for NTK, you need pre-activations AWAY from the ReLU kink. The pre-trained model (`weights-mnist_odd_even_d250`) has layer 0 std=0.004 (40× larger), giving pre-activation std=0.25 — well away from zero.
  - **Fix**: Load pre-trained weights as θ₀ and fine-tune on **held-out data** (MNIST test set, not the 250/class training data). With pre-trained weights, feature stability = 1.000 at T=1.
  - **The correct attack scenario**: Pre-trained model + LoRA fine-tuning on private data → attacker reconstructs private data from ΔW = BA. This matches real-world LoRA usage.
- **Fine-tuning data must not overlap with pre-training data.** The pre-trained model was trained on first 250 even + first 250 odd digits (sequential, no shuffle from MNIST train set). If you fine-tune on samples already in the training set, the gradient is essentially zero (coefficients ≈ 1e-18) because the model already converged on them. Use MNIST test set or clearly held-out indices.
- **The pre-training data selection is deterministic**: `shuffle=False, start=0, end=50000` in `mnist_odd_even.py`, so the exact 250/class samples are reproducible.

### What Worked
- **Pre-trained weights + held-out data → SSIM=0.996 (full) / 0.793 (LoRA r=8) at T=1.** A single gradient step from a pre-trained model leaks private fine-tuning data with near-perfect fidelity (full model) and recognizable digit structure (LoRA r=8). Control images (different instances of same digit) score 0.568, proving instance-specific leakage. Feature stability = 0.75 — not ideal (>0.99 target), but good enough.
- **NTK loss decreases steadily**: 0.25 → 9.4e-4 in 5K epochs (compare to stuck at 1.85e-5 with random init). Real gradient signal (coefficients ±0.5) makes the optimization landscape tractable.

### What Didn't Work
- Random init with init_scale=0.0001 for NTK experiments → all pre-activations at ReLU kink → destroyed NTK feature stability → noisy reconstructions

---

## ViT Gradient Inversion (Phase 0) — 2026-04-09

### Key Realizations
- **ViT gradient inversion is a fundamentally harder optimization problem than MNIST MLP reconstruction.** The gap between MNIST MLP (SSIM=0.997 at T=1) and ViT-B/16 is not just about fixing bugs — it reflects the jump from 784-dim grayscale to 150K-dim RGB, from 2M-param MLP to 86M-param transformer, and from well-conditioned piecewise-linear activations to attention + GELU.
- **LoRA-only inversion effectively uses ~147K dims, not 294K.** Peft initializes B=0, A=randn. Since ∂L/∂A = B^T · (∂L/∂y) and B=0, the gradient w.r.t. A is identically zero at initialization. Both true and predicted A gradients are zero (model B is never updated during inversion). The matched zeros cancel in cosine similarity. Only B matrix gradients carry signal.
- **LoRA-only outperformed full-model in early tests.** This is counterintuitive but makes sense: with fewer gradient dimensions (~147K vs 86M), the cosine similarity optimization landscape is smoother. The full-model gradient has more information but the optimizer can't exploit it in 10K iterations. Echoes Sami et al. (CVPR 2025).
- **Hyperparameters were never tuned.** The Phase 0 config (lr=0.1, tv_weight=1e-4, Adam, 10K iters) was a one-shot default. Sprint 2 required 148+ configs to find optimal settings.
- **Image priors matter more as dimensionality grows.** MNIST's 784-dim space is small enough that box constraints (x ∈ [-1,1]) suffice. At 150K dims (224×224 RGB), the optimizer needs TV, perceptual, frequency, or generative priors to stay on the natural image manifold.

### What Worked
- Bug fixes (create_graph=True, global cosine sim, full-model gradient) raised SSIM from 0.015 to measurable levels. The bugs were real and the fixes were necessary.
- LoRA-only mode is a viable (and surprisingly effective) simplification for gradient inversion.
- Reconstructions show correct color palette and vague shape of the boat — real signal exists, just not enough to be useful yet.

### What Didn't Work
- 86M-parameter full-model gradient inversion with default hyperparameters.
- TV-only prior at 224×224 — too weak to constrain the search space.
- Single seed/image — no multi-seed statistics to distinguish bad luck from bad method.

### Pitfalls to Avoid
- **Don't conclude "ViT inversion doesn't work" from one untuned run.** Sprint 2 showed how much hyperparameter tuning matters (seed=42 outlier at SSIM=0.830 vs 50-seed mean 0.558).
- **Don't skip the dimensionality ladder.** Going from 784-dim MNIST to 150K-dim ImageNet in one jump is asking for trouble. CIFAR-10 (3K dims) is the natural stepping stone.
- **Attention double-backward is fragile.** Must use SDPA math-only backend (no flash attention). Memory-intensive. Consider whether differentiable unrolling (Approach G) can bypass this entirely.
- **Use standard metrics.** Early Phase 0 used a non-standard global-mean SSIM instead of windowed SSIM (Kornia/Wang et al. 2004). All reported SSIM values from the first runs were not comparable to Sprint 2 or the literature. Fixed 2026-04-09: now uses `kornia.metrics.ssim(window_size=3)`.
- **`backward(inputs=[x_recon])` matters for performance.** The default `total_loss.backward()` computes gradients for all 86M model parameters even though only x_recon is optimized. Using `backward(inputs=[x_recon])` avoids this waste.
- **Geiping et al. use signed Adam, not standard Adam.** The sign of the gradient update was a key finding in their paper. Standard Adam may plateau earlier on cosine similarity maximization. Now available via `--optimizer signAdam`.
- **SignSGD ≠ signed Adam.** First implementation of signAdam was just `sign(raw_gradient) × lr` (= SignSGD). This ignores Adam's momentum/variance, flipping direction wildly per pixel per step → high-frequency noise. Got cos_sim=0.97 but SSIM=0.008 (noise image). Correct signed Adam: compute full Adam update (momentum + variance + bias correction), *then* take sign. Always verify optimizer implementations against the reference code, not a paper description.
- **Never upscale low-res images for ViT inversion.** CIFAR-10 (32×32) upscaled to 224×224 creates blocky, unnatural images. The gradient encodes the upscaling artifact, not natural image structure. TV regularization fights the block artifacts. Always use datasets with native high-res images (Flowers102 ~500px, Food101 ~512px, ImageNet) and center-crop to 224. Default changed from `cifar10` to `flowers102`.

### Optimizer Deep Dive (2026-04-14)

Three variants of "signed Adam" exist — they are NOT interchangeable:

1. **SignSGD** (buggy, was our first impl): `update = sign(raw_grad) × lr`. No momentum/variance. Every pixel gets ±lr each step → uniform HF noise. Maximizes cos_sim (0.97) because noise is directionally "correct" but magnitude-uniform → SSIM=0.008.

2. **Sign-then-Adam** (Geiping et al., current code): `x.grad.sign_()` before `optimizer.step()`. Adam receives ±1 inputs and applies momentum + variance. Problem: after enough iters, Adam's variance tracker `v_t ≈ 1` for all params (all inputs are ±1), so adaptive scaling degenerates. Effectively becomes momentum SGD with uniform step size. May explain the "good patches + HF noise" observation.

3. **Adam-then-sign** (described in literature, never implemented): compute full Adam update, *then* take sign. Preserves Adam's directional intelligence while enforcing uniform step magnitude. Not validated by any published code.

**Key insight:** The "good patches + HF noise" reconstruction is the signature of an optimizer that found the right basin (correct colors, spatial structure) but oscillates within it due to too-aggressive uniform updates.

**Critical instrumentation gap found and fixed:** `invert_gradient()` returned only `best_x` — no cos_sim, no loss curves. Cos_sim was printed to console but WEXAC logs were lost. Added: return `best_cos_sim` + full per-restart `loss_history`, save to .pth, generate loss curve plots.

### D1 Results — Hypothesis Tested (2026-04-14)

The hypothesis that signAdam hurts was **wrong**. D1 controlled comparison (4 configs, same image, same gradient):

| Config | Optimizer | TV weight | SSIM | cos_sim |
|--------|-----------|-----------|------|---------|
| A | Adam | 1e-4 | 0.030 | 0.920 |
| B | signAdam | 1e-4 | 0.020 | 0.934 |
| C | Adam | 1e-2 | 0.090 | 0.887 |
| **D** | **signAdam** | **1e-2** | **0.144** | **0.933** |

**What actually mattered — TV weight, not optimizer choice:**
- Weak TV (1e-4) produces noise regardless of optimizer. SSIM=0.02-0.03.
- Strong TV (1e-2) produces visible structure. SSIM=0.09-0.14.
- The 100× TV increase was the dominant factor (4.5× SSIM improvement for Adam, 7× for signAdam).

**signAdam wins at every TV level**, but the margin is small with weak TV (0.02 vs 0.03) and large with strong TV (0.144 vs 0.090). Explanation: strong TV constrains the search space enough that signAdam's aggressive direction-finding becomes an advantage rather than producing noise. With weak TV, both optimizers find gradient-matching noise images.

**Convergence pattern:** signAdam restarts are remarkably consistent (all 8 in 0.920-0.934 cos_sim) while Adam restarts spread widely (0.465-0.920). signAdam is more robust to initialization.

**Lesson:** Don't blame the optimizer when the regularizer is 100× too weak. The previous conclusion that "signAdam creates HF noise" was actually "tv_weight=1e-4 is insufficient at 224×224 resolution." Always test regularization strength before changing the optimizer.

### D2 Sweep — TV Weight Is the Dominant Lever (2026-04-28)

**Context:** D1 (2026-04-14) showed signAdam + tv=1e-2 reached SSIM=0.144, just below the 0.15 gate. D2 swept tv_weight × lr × n_iters around the D1 winner: 5 × 4 × 2 = 40 configs.

**Result: gate crossed.** Best D2 config (tv=1e-1, lr=0.05, 30K iters) achieves **SSIM=0.548, PSNR=15.11, cos_sim=0.955** — 3.8× over D1's best. 7/29 analyzed configs cleared the 0.3 SSIM gate, **all at tv=1e-1**.

**TV-weight ranking (best SSIM at any lr/iters per TV level):**

| TV weight | Best SSIM | Notes |
|-----------|-----------|-------|
| 1e-1      | 0.548     | All 7 gate-passing configs are here |
| 2e-2      | 0.267     | Plateaus far below gate |
| 1e-2      | 0.207     | Matches D1's tv=1e-2 finding (~0.14-0.20) |
| 5e-3      | 0.109     | Effectively no signal |

**Lessons:**
1. **TV at 224×224 needs to be much stronger than papers suggest.** Geiping et al. used tv≈1e-4 to 1e-2 for ImageNet. Our system (Flowers102 image, full ViT-B/16 gradient, signAdam) needs tv=1e-1 — 10× stronger than D1's winner and 1000× stronger than the original Phase 0 default. The D1 conclusion ("strong TV is essential") was directionally right but understated the magnitude.
2. **Cos_sim is loosely coupled to SSIM at the high end.** All 7 D2 winners had cos_sim 0.94–0.96; the worst configs (5e-3 TV) also reach cos_sim 0.92+. Cos_sim is a necessary-not-sufficient metric — once it saturates near 0.95, only the pixel-space prior (TV) determines whether the answer is a noisy match or a recognizable image.
3. **lr is a secondary lever, iters has diminishing returns.** Across lr ∈ {0.01, 0.05, 0.1, 0.5} at tv=1e-1, SSIM stays in [0.46, 0.55]. Going from 10K → 30K iters gives only ~0.05 SSIM lift on average.
4. **Always sweep at least one order of magnitude past the previous winner.** If we'd tested only tv ∈ {5e-3, 1e-2, 2e-2} (a tighter sweep around D1), we would have concluded tv=1e-2 is optimal. The 1e-1 finding required deliberately overshooting.

**Action:** Set tv=1e-1 + lr=0.05 + 30K iters + signAdam as the new Phase 0 baseline. Rerun LoRA-only mode and multi-seed at this config before adding any new priors (D3).

---

### Visualization: Always Add ds_mean Back Before Plotting Reconstructions (2026-04-28)

**Context:** Free-coefficient experiment figures (`experiment_b_free_coeff_grid.png`, `free_coeff_reconstruction_grid.png`) showed grey/blank reconstructions despite SSIM=0.59 confirming real signal.

**Root cause:** Reconstructions are optimized in mean-subtracted space: `x_centered = x_ft - ds_mean`, so `x_recon_lora` lives in range [-0.2, 0.2]. Ground truth `x_train` lives in pixel space [0, 1]. Plotting both on the same [0, 1] colormap without adding `ds_mean` back makes reconstructions appear flat grey.

**Fix:**
1. The plotting code in `plotting.py` (`plot_reconstruction_grid`, `generate_experiment_b_figure`) already correctly marks reconstructions as `is_centered=True` and adds `ds_mean` back at display time. The broken figures were generated by an older, now-deleted code path.
2. Made `generate_experiment_b_figure` mode-aware: auto-detects free-coefficient vs oracle mode from `results['config']['mode']` or `coeff_error` presence; adjusts title, subtitle, and output filename (`experiment_b_grid_free.png` vs `experiment_b_grid_oracle.png`).

**Lesson:** When working with mean-centered data, always track which tensors are in which space. Mark them explicitly (e.g., `is_centered` flag) and handle the conversion at the display boundary. Never assume pixel range [0, 1] — always check `.min()` and `.max()` before plotting. And when figures look wrong, check the value ranges before suspecting the algorithm.

### Visualization: By-Axis Beats Top-N When One Axis Dominates the Sweep (2026-04-28)

**Context:** D2's first aggregate figures replicated D1's two-figure layout — `phase0_d2_top5_comparison.png` (GT + 5 best reconstructions) and `phase0_d2_cossim_overlay.png` (cos_sim curves for the same 5). They were not informative: all 7 gate-passing configs share `tv_weight=1e-1`, so the top-5 panels collapsed to five near-identical reconstructions of the same regime, and the cos_sim curves bunched at 0.94–0.96.

**Fix:** Replaced both with **by-axis** variants (`phase0_d2_top_comparison_by_tv.png` + `phase0_d2_cossim_overlay_by_tv.png`): one panel/curve per TV level (best config in that level), TVs ordered low→high. The reconstruction quality progression (noise → recognizable flower) is now on screen and matches D1's "show one panel per qualitatively-different setting" framing.

**Lesson:** When one sweep axis dominates SSIM (or whatever the headline metric is), top-N collapses onto a single regime and tells the reader nothing about the rest of the grid. Use **by-axis-of-interest** instead: pick the dominant lever, render one panel per level (best at that level), order monotonically. The heatmap still owns the full-grid story; the by-axis figures own the "what does each regime look like" story.

**Rule of thumb:** if your top-N panels would all share the same value on the dominant axis, you've built a degenerate version of the by-axis figure — drop the top-N.

**Title/layout pitfall:** matplotlib's `tight_layout()` packs subplots tightly and won't add horizontal padding for wide titles. When per-panel titles include several pieces of metadata (`#k tv=… lr=… SSIM=…`), neighboring titles collide. Fix is explicit `fig.subplots_adjust(wspace=0.18, top=0.78)` and shorter title text (drop redundant separators, use one line of metadata + one line of metrics, slightly larger fontsize).

---

## Discrete Sequence Reconstruction (LLMs)

*(Fill in as you go)*

### Key Realizations
-

### What Worked
-

### What Didn't Work
-

---

## Diffusion Priors / SDS

*(Fill in as you go)*

### Key Realizations
-

### What Worked
-

### What Didn't Work
-

---

## Document & Presentation Generation

### Audit Process
- **Fix numbers first** → grep all files for stale values (not just the ones you think are affected). `grep -rn "old_value" **/*.tex **/*.py` catches stragglers.
- **Fix examples second** → cross-reference every example against source data.
- **Add new content third** → new slides, sections, definitions.
- **Sync between formats** → PPTX and Beamer (or any parallel formats) must match.
- **Polish last** → speaker notes, transitions, naming consistency.
- **Always do a final audit** → the first sweep ALWAYS misses some stale references.
- **Commit after each phase** — makes rollback possible if a later phase breaks something.

### Common Bugs
- **Stale numbers in speaker notes**: Notes are invisible in PDF/PPTX so easy to forget. Always grep notes too.
- **Same example, different numbers on different slides**: Always cross-reference.
- **Illustrative vs actual data**: Pedagogical slides with simplified examples contradict real data slides.
- **Table overflow**: Always verify `table_y + (rows+1) * row_height < next_element_y`.

### OOXML Animation (python-pptx)
- `python-pptx` has **NO animation API** — must write raw OOXML XML via `lxml.etree`.
- **3-level par nesting is non-negotiable**: `par(delay="indefinite") > par(delay="0") > par(clickEffect)`. Skip a level and PowerPoint silently ignores all animations.
- **`para_build` must default to `False`** — when `True`, multi-paragraph text shapes get hidden on entry.
- **Card animations must include ALL child shapes** in the same `anim_groups` entry.
- **Never regex-manipulate OOXML namespaces** — stripping `xmlns:p14` once caused PowerPoint to blank an entire slide.

### Plot-Presentation Integration
- matplotlib defaults are fine for standalone analytical plots, but plots embedded in dark-background slides need dark-theme variants.
- Always test embedded plots at **50% zoom** — simulates projector distance readability.

### Generator Architecture (>500 lines)
- Refactor into a package: slim orchestrator + config.py + helpers.py + per-section slide modules.
- Each slide is a function. Auto-number via a counter. Centralize image paths in config.

### Narrative Framing (Academic)
- Frame negative results as "successfully identified the bottleneck" rather than "the approach failed."
- Consolidate fine-grained failure modes into 3-5 categories that map to remediation strategies.
- One concept per slide. If you need "Part 1" and "Part 2" labels, split into two slides.

### Markdown → PDF on WEXAC: use fpdf2, not a LaTeX toolchain (2026-06-23)
- **What broke:** to turn `notes/experiment_plan.md` into a reading PDF, the obvious routes all failed —
  no `pandoc`, `pdflatex`, `xelatex`, `wkhtmltopdf`, or `weasyprint` installed, and the local `tectonic`
  binary aborts with `GLIBC_2.35 / GLIBCXX_3.4.30 not found` (system libstdc++/glibc too old for it). No
  `pdftoppm`/ghostscript/pymupdf either, so PDFs can't even be rasterized to eyeball here.
- **What works:** Python **`fpdf2` 2.8.4** + the system **DejaVu** fonts
  (`/usr/share/fonts/dejavu-sans-fonts/`, `dejavu-sans-mono-fonts/`). DejaVu covers all the Greek/math
  glyphs the notes use (θ α ∇ Σ ‖·‖² → x̂ □ ✓ ✗); verify coverage with a fontTools cmap check since
  missing glyphs render blank with no error.
- **Gotchas:** (a) `multi_cell`'s default cursor leaves x at the right edge → the next call throws "Not
  enough horizontal space"; wrap it to default `new_x=LMARGIN, new_y=NEXT` (the `pdf.table()` API passes
  these explicitly, so a `setdefault` wrapper leaves tables alone). (b) Use `pdf.table()` for tables — it
  wraps cells, avoiding the fpdf O9 silent-truncation trap. (c) Reset draw/fill/text state after any
  colored box (O10). Recipe saved to memory; formal thesis LaTeX still goes through Overleaf.

---

## General Research Process
- **"The math says it works" vs. "you can make it work"** are different claims. Optimistic theoretical analyses (like the Gradient Bridge feasibility argument) are correct about the information being there, but gloss over engineering gaps and empirical unknowns. Calibrate accordingly: the idea is sound, but the execution is the hard part — which is exactly what makes it a thesis.
- **De-risk before building**: always run the cheapest experiment that could falsify your approach before investing weeks in the full pipeline.
- **ALL experiments run on WEXAC GPU, not MPS.** The MacBook's MPS backend is for light local dev/debugging only. Real training, reconstruction, and any serious compute must run on the WEXAC cluster (NVIDIA L40S). MPS is too slow, has dtype limitations, and results won't match CUDA. Always use `wexac_connect.sh shell` to get a GPU node before running experiments.
- **WEXAC `rec` env has PyTorch 2.4.1+cu121** (with timm 0.9.12, peft 0.7.1, torchvision 0.19.1). Use `weights_only=False` in `torch.load()` to suppress FutureWarnings. The old claim of "PyTorch 1.11" was stale documentation.

---

## [INSIGHT] L-BFGS vs SGD for NTK Extraction (2026-02-22)

**Context:** Implementing and comparing optimizers for the NTK reconstruction loss in Experiment B. The supervisor suggested L-BFGS as a better optimizer for this small-scale least-squares problem (~1,570 unknowns).

**Lesson:** L-BFGS converges extremely fast but gets trapped in a shallow local minimum for the full-model case. SGD with momentum is slower but finds a much better solution. For LoRA extraction, both hit the same loss plateau — the bottleneck is the irreducible rank mismatch, not the optimizer.

**Details:**
- **Full model**: L-BFGS drops loss from 0.25 → 0.005 in the *first step* (20 func evals), then stalls at SSIM ≈ 0.82. SGD with momentum takes 50K epochs to reach loss 9.4e-4, but achieves SSIM ≈ 0.996. The NTK landscape is non-convex (ReLU kinks), and SGD's momentum helps escape shallow basins that trap L-BFGS.
- **LoRA r=8**: Both optimizers converge to loss plateau ~3.3 with SSIM ~0.80–0.84. The plateau is an *irreducible residual* from rank mismatch: the predicted gradient is full-rank but the target ΔW lives in the rank-r column space of B₀.
- **LoRA subspace projection**: Projecting both target and prediction into col(B₀) via P = B₀(B₀ᵀB₀)⁻¹B₀ᵀ *hurts* reconstruction (SSIM drops from 0.78 → 0.49). The null-space gradient components, while they can't match the target, provide useful optimization signal that guides x toward the right solution.
- Relevant files: [experiments/ntk_extraction.py](experiments/ntk_extraction.py) (L-BFGS closure, projection function), [experiments/run_experiment_b.py](experiments/run_experiment_b.py) (LoRA extraction without projection).

**Action:** Use SGD for publication-quality full-model results (SSIM > 0.99). Use L-BFGS for quick preliminary LoRA rank sweeps (same quality in 1/50th the epochs). Do NOT project into LoRA subspace — keep the unprojected loss.

**Update (Sprint 2c B3b, 2026-03-26):** SGD + LeakyReLU matches L-BFGS identically for T ≤ 20 (SSIM within ±0.003), but NaN's at T=100 where L-BFGS still works (0.775-0.809). For multi-step extraction beyond T=20, L-BFGS remains essential. For the realistic few-shot regime (T ≤ 20), SGD is a viable and simpler alternative.

---

## [INSIGHT] NTK Coefficients Are "Cheating" — Must Be Free Parameters (2026-02-22)

**Context:** The NTK reconstruction loss is ||ΔW + η Σ cᵢ ∇f(θ₀; xᵢ)||² where cᵢ = (σ(f(θ₀; xᵢ)) - yᵢ)/N. The current code ([experiments/ntk_steps.py:28-33](experiments/ntk_steps.py#L28-L33)) computes cᵢ from the **true private data x** and passes them as fixed constants to the extraction ([experiments/ntk_extraction.py:8](experiments/ntk_extraction.py#L8)). This is cheating — in a real attack, the adversary has θ₀ and ΔW but not x, so they cannot compute cᵢ.

**Lesson:** The coefficients cᵢ must be treated as **free optimization variables** alongside x, exactly as Haim et al. treat the Lagrange multipliers λᵢ in KKT reconstruction. The structural parallel is exact:

| | Haim et al. (KKT) | NTK (current, cheating) | NTK (correct) |
|---|---|---|---|
| **Equation** | W ∝ Σ λᵢ yᵢ ∇Φ(W; xᵢ) | ΔW = -η Σ cᵢ ∇f(θ₀; xᵢ) | same |
| **Optimize over** | x AND λ | x only (c fixed from true x) | x AND c |
| **Scalar unknowns** | λᵢ ≥ 0.05 (penalized) | — | cᵢ ∈ [-1, 1] (penalized) |

**Why this matters beyond honesty:**
1. **Multi-step (T > 1)**: The current code uses `coefficients_at_init`, which is only exact for T=1. With free c, the optimizer finds effective average coefficients that explain the cumulative ΔW without requiring the frozen-feature NTK assumption.
2. **Smooths the optimization**: Decouples the x→c→ΔW chain, avoiding the chicken-and-egg problem where you need good x to compute good c and vice versa.
3. **Adds only N scalars** (N=2 in our case) to the optimization — negligible cost.

**Haim et al. λ regularization (reference for our c penalty):**
- Separate optimizer: `SGD([λ], lr=1e-4)` — 100× smaller lr than x
- Lower bound: `5 * (-λ + 0.05).relu().pow(2).sum()` — keeps λ ≥ min_lambda
- Init: `torch.rand(N, 1)` — uniform [0, 1]
- See [dataset_reconstruction/extraction.py:43-51,79-85](dataset_reconstruction/extraction.py#L43-L85)

**Risk:** Non-uniqueness — large c with wrong x could explain ΔW as well as true c with true x. Mitigate with a **self-consistency penalty**: `α * |cᵢ - (σ(f(θ₀; xᵢ)) - yᵢ)/N|²` which encourages the free c to agree with what the model would actually produce on the current x estimate. This is a soft constraint — start with α=0 (pure free c), ablate over α.

**Action:** Implement free-coefficient NTK extraction as the new default. Keep the "known coefficients" mode as a diagnostic/oracle baseline for comparison.

---

## [RESULT] Free-Coefficient Extraction Works — Consistency Penalty Is Essential (2026-02-22)

**Context:** Implemented free-coefficient mode in [experiments/ntk_extraction.py](experiments/ntk_extraction.py) and ran the α ablation on seed=42, T=1, full model.

**Results (full model, seed=42, T=1):**

| Mode | Optimizer | Epochs | SSIM | Coeff Error | Notes |
|------|-----------|--------|------|-------------|-------|
| Oracle (cheating) | L-BFGS | 500 | 0.817 | 0 (fixed) | Upper bound with L-BFGS |
| Free c, α=0 | L-BFGS | 500 | 0.282 | 1.066 | **Signs flipped** — non-unique solution |
| Free c, α=1 | L-BFGS | 500 | 0.777 | 0.005 | Correct signs, near-oracle c |
| Free c, α=10 | L-BFGS | 500 | 0.638 | 0.005 | Over-penalized, hurts NTK fit |
| **Free c, α=1** | **SGD** | **5000** | **0.997** | **0.0004** | **Matches oracle — attack works honestly** |
| Free c, α=1 LoRA r=8 | L-BFGS | 500 | 0.539 | 0.244 | One coeff stuck near 0 |

**Key lessons:**
1. **α=0 (pure free c) fails**: L-BFGS finds a sign-flipped solution where c₁ and c₂ swap signs. The NTK loss is equally satisfied because flipping c and replacing x with a "mirror" image produces the same ΔW. This is the non-uniqueness risk we predicted.
2. **α=1 (moderate consistency) is the sweet spot**: The self-consistency penalty `|c - (σ(f(θ₀;x))-y)/N|²` breaks the sign ambiguity by coupling c to the model's actual predictions on the current x. With α=1, coefficients converge to within 0.005 of oracle values.
3. **α=10 over-constrains**: The consistency penalty dominates the NTK loss, preventing x from moving freely. SSIM drops to 0.638 despite correct c.
4. **SGD beats L-BFGS again**: Free-c + SGD + α=1 achieves SSIM=0.997, essentially matching oracle SGD. L-BFGS gets trapped in shallow minima (consistent with earlier L-BFGS vs SGD findings).
5. **LoRA needs SGD**: With L-BFGS and LoRA r=8, one coefficient gets stuck near 0 while the other converges. SGD's separate optimizer with smaller lr should fix this.

**The punchline: the "cheating" didn't matter.** For the full-model case with the right optimizer (SGD) and regularization (α=1), free-coefficient extraction matches oracle quality. The attack is honest AND effective.

**Relevant files:**
- [experiments/ntk_extraction.py](experiments/ntk_extraction.py) — `get_coeff_penalty()`, `run_ntk_extraction()` with `free_coefficients=True`
- [experiments/run_experiment_b.py](experiments/run_experiment_b.py) — `--free_coefficients`, `--consistency_weight`, `--n_sweep` flags
- [experiments/configs.py](experiments/configs.py) — `COEFF_LR`, `COEFF_BOX_WEIGHT`, `COEFF_CONSISTENCY_WEIGHT`

**Action:** Use `--free_coefficients --consistency_weight 1.0 --optimizer sgd` as the default for all future runs. Run LoRA rank sweep with this config on WEXAC.

---

## [RESULT] Activation Function Is Critical for LoRA Extraction (2026-02-22)

**Context:** Sprint 1 (L-BFGS, implicit ReLU) got LoRA r=8 SSIM=0.797, but Sprint 2a (SGD, ModifiedRelu alpha=150) got SSIM=0.183. Ran an activation ablation to disentangle optimizer vs activation effects.

**Results (LoRA r=8, oracle coefficients, seed=42, T=1):**

| Alpha | L-BFGS (SSIM) | SGD (SSIM) |
|-------|---------------|------------|
| 10 (very smooth) | 0.044 | 0.177 |
| 50 | 0.126 | 0.149 |
| 150 (default) | 0.184 | 0.183 |
| **10000 (≈ ReLU)** | **0.744** | **0.467** |

**Free-coefficient (SGD, consistency α=1):**

| Alpha | SSIM | Coeff Error |
|-------|------|-------------|
| 10 | 0.414 | 0.212 |
| 10000 | 0.476 | 0.234 |

**Key lessons:**
1. **ModifiedRelu actively hurts LoRA extraction.** At alpha=150, both L-BFGS and SGD get SSIM ~0.18. At alpha=10000 (≈ plain ReLU), L-BFGS jumps to 0.744. The sigmoid-modulated gradients in ModifiedRelu create smooth but incorrect gradients when the LoRA subspace projection interacts with the activation's non-linearity.
2. **For LoRA, L-BFGS + ReLU is the best combo.** L-BFGS (0.744) >> SGD (0.467) at alpha=10000 — opposite of the full-model result where SGD wins. The LoRA extraction landscape is smoother (lower effective dimension due to rank constraint), favoring L-BFGS.
3. **ModifiedRelu was tuned for Haim et al.'s KKT extraction**, which optimizes W ∝ Σ λᵢ yᵢ ∇Φ(W; xᵢ) — the model is evaluated at the *extraction point*, so smooth gradients through the model help. NTK extraction evaluates at frozen θ₀ — the model is just a fixed feature extractor, and smooth gradients don't help (and actually hurt by introducing approximation error).
4. **Previous lesson partially wrong:** The earlier "L-BFGS vs SGD" lesson said "L-BFGS gets trapped in shallow minima." That was true for **ModifiedRelu** but NOT for **ReLU**. The optimizer-activation interaction matters more than either alone.

**Action:** For LoRA extraction, always use alpha=10000 (≈ ReLU) + L-BFGS. For full model, use SGD (which works with any alpha). Add `--relu_alpha` to all LoRA experiment commands.

**Relevant files:**
- [run_activation_ablation_wexac.sh](run_activation_ablation_wexac.sh) — WEXAC batch script for the ablation
- WEXAC job 669885 — results on A10 GPU

---

## [RESULT] Free-Coefficient LoRA Extraction: Partial Success (2026-02-23)

**Context:** Ran LoRA rank sweep with the winning config (alpha=10000/ReLU + L-BFGS for x + separate SGD for c). Tested coeff_lr tuning and epoch count.

**Results (all: alpha=10000, L-BFGS, T=1, seed=42):**

| Rank | Oracle SSIM | Free-c SSIM (lr=1e-2, 5Kep) | Coeff Error | Gap |
|------|-------------|------------------------------|-------------|-----|
| 4    | 0.615       | 0.509                        | 0.192       | 0.11 |
| 8    | 0.692       | **0.617**                    | 0.177       | 0.08 |
| 16   | 0.769       | 0.422                        | 0.282       | 0.35 |
| 32   | 0.697       | 0.415                        | 0.310       | 0.28 |
| 64   | 0.714       | **0.635**                    | **0.019**   | 0.08 |

**coeff_lr ablation (r=8):** lr=1e-3→0.457, lr=1e-2→**0.617**, lr=1e-1→0.536 (overshoots)

**Key lessons:**
1. **Free-c works well at r=8 and r=64** — within 0.08 SSIM of oracle. The coefficient optimization converges when the LoRA subspace captures enough of the gradient information.
2. **r=16 and r=32 are stubbornly bad** — coeff_error stays ~0.28-0.31 despite more epochs and higher lr. The optimization landscape has local minima at these ranks that trap the coefficient SGD.
3. **coeff_lr=0.01 is the sweet spot** — 10x default. Too low (1e-3) = underfitting, too high (0.1) = overshooting.
4. **Separate SGD for c is correct** (vs joint L-BFGS) — confirmed by the improvement from job 674631→681126.
5. **The residual gap** for well-converging ranks (r=8, r=64) is small enough that the free-c attack is viable — an attacker doesn't need oracle access.

**Next steps:** Try Adam for c (adaptive lr), random restarts, or higher consistency_weight for stubborn r=16/32.

---

## [INSIGHT] Always Save Visual Examples from Every Experiment Run (2026-02-22)

**Context:** After running the T-sweep on WEXAC, we had a CSV of SSIM numbers but no saved reconstruction images. To generate a PDF of visual examples, the entire extraction had to be re-run locally for each (T, rank) configuration — wasting hours of compute that was already done.

**Lesson:** Every experiment run should automatically save representative reconstruction images (both good and bad examples) alongside the numeric metrics. Numbers alone don't tell the full story — a "SSIM=0.48" could look like random noise or like a blurry-but-recognizable digit. Visual inspection is essential for understanding what's working and what's failing.

**Action:**
- Every sweep or single-config run should save a `.pth` file containing the actual image tensors (`x_train`, `x_recon_full`, `x_recon_lora`, `x_ctrl`, `ds_mean`) — not just scalar metrics.
- For sweeps, save a per-config results dict (e.g., `results/experiment_b_sweep_<timestamp>/T{T}_r{rank}.pth`) so any configuration can be visualized later without re-running.
- Also generate a quick PNG/PDF grid of examples (best and worst by SSIM) as part of the sweep output, so you never have to re-derive figures from scratch.

---

## [INSIGHT] SGD Required for Fine-Tuning, Not for Extraction (2026-02-23)

**Context:** Clarifying the role of the optimizer in the fine-tuning vs. extraction phases. The theoretical framework (implicit bias of GD on BCE → KKT/max-margin convergence) constrains the fine-tuning optimizer, but NOT the extraction optimizer.

**Key distinction:**
1. **Fine-tuning MUST use SGD.** The implicit bias of gradient descent on BCE loss → convergence to the max-margin solution → KKT stationarity conditions → weights encode support vectors. This is the theoretical foundation of the entire reconstruction attack. Adam, RMSProp, or any adaptive optimizer breaks this implicit bias guarantee.
2. **Extraction can use ANY optimizer.** The extraction phase solves an inverse problem: given ΔW, find x such that the NTK loss is minimized. This is just optimization — use whatever converges best. Adam is a strong candidate because its adaptive per-parameter learning rate handles the mixed-scale landscape well (pixel values, coefficients, and regularizers all have different scales).

**Why this matters:** We initially used SGD for extraction because the theoretical framework seemed to require it everywhere. But the theory only constrains the *forward* process (training). The *inverse* process (reconstruction) is an engineering problem where we're free to use the best tool. This opens up Adam, L-BFGS, or even learned optimizers for extraction.

**Relevant files:**
- [experiments/ntk_extraction.py](experiments/ntk_extraction.py) — extraction optimizer (L-BFGS, SGD, or Adam)
- [experiments/train_lora.py](experiments/train_lora.py) — fine-tuning optimizer (must be SGD)
- [experiments/run_experiment_b.py](experiments/run_experiment_b.py) — `--optimizer` and `--coeff_optimizer` flags

---

## [INSIGHT] Few-Shot Fine-Tuning Is the Attack Sweet Spot (2026-02-23)

**Context:** Connecting the NTK reconstruction results to real-world few-shot fine-tuning of large online models (LoRA adapters published on HuggingFace, CivitAI, etc.).

**Key insight:** Few-shot fine-tuning (N=5-50 samples, T=1-100 gradient steps) is the regime where LoRA reconstruction attacks are most potent:

1. **Overdetermined system**: A LoRA adapter for ViT-B/16 has ~300K-1M parameters per adapted layer. Fine-tuning on N=5 images of 224×224×3 ≈ 150K pixels each gives ~1M constraints for ~750K unknowns. The adapter contains enough information to reconstruct the data.

2. **Few gradient steps**: Users typically fine-tune for 1-10 epochs. With N=5 and 5 epochs, that's ~25 gradient steps. Phase 2 results show reconstruction holds through T=100 with LeakyReLU — real few-shot LoRA fine-tuning lives comfortably in this regime.

3. **All samples are support vectors**: With N << parameters, every training sample sits on the decision boundary. This is exactly the condition Haim et al.'s theory requires.

4. **Realistic threat model**: θ₀ is public (foundation model), BA is published (adapter on HuggingFace/CivitAI), and N is small. The attacker has everything needed.

**Concrete threat scenarios:**
- Face LoRA (CivitAI): Stable Diffusion + 5-20 selfies → adapter shared publicly
- Medical LoRA: ViT/BiomedCLIP + patient scans → shared with collaborators
- Legal/financial: LLaMA + confidential docs → internal model registry

**What our results say:**
- T=1: SSIM ≈ 1.0 (full) / 0.83 (LoRA) — even one gradient step leaks data
- T=100: SSIM ≈ 0.78-0.80 with LeakyReLU — realistic training is still vulnerable
- Free-coefficient extraction works — attacker doesn't need oracle access
- 11% of seeds attackable — not every run is vulnerable, but attacker can't predict which

**Gap this thesis fills:**
- Haim et al.: needs 1M+ epoch convergence (unrealistic for modern fine-tuning)
- Gradient inversion: needs actual gradient (not available from published adapter)
- This thesis: reconstructs from adapter weights via NTK/Gradient Bridge — the few-shot regime is where the attack is most potent and the threat model is most realistic

**Caveat:** All results so far are MNIST + 2-layer MLP. Scaling to ViTs on real images (Sprint 3) is the key open question.

---

## [BUG] Phase 0 Used Bilinear-Upscaled CIFAR-10 as ViT Input — Methodological Error (2026-04-09)

**Context:** Phase 0 (ViT gradient inversion) used a CIFAR-10 image (32×32) bilinearly upscaled to 224×224 as input to ViT-B/16. The ground truth image looked blurry and blocky — each original pixel became a ~7×7 smeared blob.

**Why this is wrong (not just ugly):**
1. **Wasted information**: ViT-B/16 creates (224/16)²=196 patches, but only (32/16)²=4 patches worth of real information exists. 192 patches encode interpolation artifacts, not image content.
2. **Artificially harder inversion**: The reconstruction target is 150K pixel values that contain only 3K real degrees of freedom (32×32×3). The optimizer wastes capacity matching interpolation artifacts.
3. **Misleading gradients**: Patch embeddings learn features from blurry blobs, not real image structure. The gradient signal is spread across 196 patches but concentrated in ~4.
4. **No serious paper does this**: Gradient inversion papers using ViT use ImageNet at native 224×224 (Geiping et al., GradInversion, Sami et al. CVPR 2025). Papers using CIFAR-10 use small-patch ViTs (patch_size=4, img_size=32).

**Fix (two legitimate options):**
1. **Small-patch ViT for CIFAR-10**: Use `vit_small_patch4_32` or configure ViT with `img_size=32, patch_size=4` → 64 patches at native resolution. Good for quick validation.
2. **Native 224×224 dataset with ViT-B/16**: Use ImageNet (or a 100-class subset) or CelebA at 224×224. This matches the real-world LoRA threat model (foundation model + adapter) and aligns with what PEFT gradient inversion papers use.

**Impact on Phase 0 results:** The SSIM=0.089 (full) / 0.264 (LoRA-only) numbers are partially explained by this error. The inversion was fighting a 49×-inflated search space filled with interpolation artifacts. Re-running with the correct setup (either option) may yield substantially better results.

**Lesson:** Always verify that the input pipeline matches how the model was trained and how the evaluation literature uses the same model. ViT-B/16 was trained on 224×224 ImageNet — use that, or use a ViT variant designed for your resolution.

---

## [BUG] Phase 0 Gradient Inversion: Three Critical Implementation Errors (2026-04-07)

**Context:** Phase 0 (ViT-B/16 gradient inversion gate experiment) returned SSIM=0.015 — total failure. Cosine similarity stuck at 0.04 throughout 3000 iterations. Root cause analysis found THREE bugs, the worst of which made the entire optimization a no-op.

**Bug 1 (ROOT CAUSE): Non-differentiable cosine similarity.** The code used `loss.backward(retain_graph=True)` to populate `param.grad`, then computed cosine similarity from these `.grad` tensors. But `.grad` attributes are **detached leaf tensors** — they have no computation graph connecting them back to `x_recon`. The subsequent `total_loss.backward()` produced zero gradients for `x_recon` from the cosine similarity term. The optimizer was only minimizing TV regularization (making a smooth random image). **Fix:** Use `torch.autograd.grad(loss, params, create_graph=True)` to get predicted gradients that remain in the computation graph.

**Bug 2: Per-tensor cosine similarity averaging.** Computed cosine similarity per parameter tensor (24 tensors), then averaged. Small tensors with random alignment got equal weight as large tensors. Geiping et al. compute ONE global cosine similarity on the entire flattened gradient vector. **Fix:** `torch.cat` all gradient tensors, compute single cosine similarity.

**Bug 3: LoRA-only gradients.** `capture_gradient()` iterated `model.named_parameters()` but peft freezes base model params → only 294K LoRA parameters had gradients. The inversion was trying to reconstruct 150K pixels from 294K low-rank gradient values. **Fix:** Temporarily enable `requires_grad_(True)` on all params to capture the full 86M-parameter gradient.

**Bug 4: SDPA double-backward not supported.** After fixing bugs 1-3, `create_graph=True` triggered `RuntimeError: derivative for aten::_scaled_dot_product_efficient_attention_backward is not implemented`. PyTorch 2.x's efficient/flash attention kernels don't support double-backward. **Fix:** Wrap the inversion loop in `torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)` to force the math-only SDPA backend.

**Bug 5: requires_grad mismatch.** After `capture_gradient` restores `requires_grad=False` on base model params, the inversion's `torch.autograd.grad(loss, params)` fails because those params don't require grad. **Fix:** Re-enable `requires_grad_(True)` on all matched params at the start of `invert_gradient`, restore after.

**Lesson:** When implementing gradient inversion, **always verify the gradient flows end-to-end** from target to optimized variable. A quick test: `total_loss.backward(); print(x_recon.grad.norm())` — if it's zero or doesn't exist, the optimization is broken. Also: never average cosine similarities across parameters — always flatten first. And when using `create_graph=True` with transformers, disable efficient/flash attention backends.

---

## [RESULT] Track A (KKT + N-Sweep) Definitively Closed (2026-04-07)

**Context:** Sprint 2c Track A tested whether using the correct N (up to N=502 total samples) would fix Sprint 1's Experiment A failure. Ran 15/48 configs before 48h timeout.

**Results:** KKT loss stuck at 330-350 for ALL N values tested (N=1 through N=100 per class). No trend — the loss didn't decrease as N approached the true support vector count.

**Why this was expected:** The composed model W = W₀ + BA satisfies KKT with respect to all ~502 samples (500 pre-training + 2 fine-tuning). The KKT loss of ~330 is essentially ||W₀||² — the unexplained pre-training residual. Even with N=502, the extraction would need to simultaneously reconstruct 500 pre-training images alongside the 2 fine-tuning targets — a fundamentally different (and much harder) problem than reconstructing 2 images from a model trained on 2 images.

---

## [INSIGHT] N>1 Reconstructions Are Superpositions — Decomposition Strategies (2026-04-07)

**Context:** When reconstructing N=2 images from NTK extraction, each reconstructed image visually looks like a ghostly superposition (blend) of BOTH training images. The NTK loss is a linear combination in gradient space: ΔW = -η Σᵢ cᵢ J(xᵢ), and nothing prevents the optimizer from distributing information across image slots. SSIM is ~0.5-0.6 when it should be higher — the information is there, just mixed.

**Root cause:** The NTK loss `‖ΔW + η Σ cᵢ ∇f(θ₀; xᵢ)‖²` has a **permutation and mixing symmetry** — any linear recombination of the per-sample contributions that sums to the same total gradient gives the same loss. The optimizer finds a blended local minimum rather than the clean separation.

**Key insight — linearity enables analytical separation:** Because the NTK regime linearizes the model, the weight gradient matrix for each FC layer has a special structure: each ROW is a different linear mixture of the N source images, with mixing coefficients from the loss gradients. With layer width 1000, this gives 1000 independent observations of the N-way mixture. This is exactly the setup for ICA (Independent Component Analysis).

**Approaches for general N (prioritized):**

1. **Cross-gradient orthogonality penalty (N=2-10):** Add `cos_sim(∇f(θ₀; x₁), ∇f(θ₀; x₂))` to the loss. Forces images to produce orthogonal gradients, directly attacking the superposition mechanism. Most theoretically principled for small N.

2. **Label-based grouping (any N, binary classification):** Coefficients cᵢ have opposite signs for the two classes (cᵢ>0 for class 0, cᵢ<0 for class 1). Separate the positive and negative gradient contributions first, then decompose within each class. Halves the effective problem size for free.

3. **ICA on weight gradient matrix — "Cocktail Party Attack" (N=10-1000):** Each row of the FC layer gradient is a linear mixture of the N source images. Apply FastICA with n_components=N to the weight gradient matrix. Scales to N ≤ layer width. Reference: Cocktail Party Attack (Kariyappa et al., ICML 2023).

4. **Sequential peeling with joint refinement (N=5-20):** Reconstruct images one at a time from the residual (matching pursuit), then jointly optimize all N using the greedy solutions as warm start. Each sub-problem is N=1 where the pipeline is strong.

5. **Overcomplete slots + clustering (any N):** Optimize for N'=2N image slots, then cluster similar results by SSIM. Redundancy helps coverage; extra slots absorb garbage solutions.

6. **Post-hoc NMF/ICA (N=2, quick experiment):** Apply `sklearn.decomposition.NMF(n_components=2)` to the two blended reconstructions. NMF is ideal for MNIST (non-negative, sparse pixels). Zero-code-change experiment.

**Phase transition:** Theoretical limit is N < network width (1000 for our MLP). SPEAR (NeurIPS 2024) achieves exact recovery up to N=25 on FC+ReLU networks using SVD + activation sparsity. The Cocktail Party Attack scales to N=1024 with ICA on the FC gradient matrix. Practical optimization-based limit is N~50-100.

**Critical existing code:** `get_diversity_penalty()` in `ntk_extraction.py` (lines 439-461) is already implemented but NOT wired into the extraction loop. Connecting it with a tunable weight is the lowest-hanging fruit.

**Key references:**
- Cocktail Party Attack (Kariyappa et al., ICML 2023) — ICA on FC gradient rows, scales to N=1024
- SPEAR (NeurIPS 2024) — exact batch recovery via SVD + ReLU sparsity filtering, N≤25
- ARES (2025) — sparse recovery in DCT basis, N≤384
- GradInversion (Yin et al., CVPR 2021) — group consistency + label recovery, N≤48
- Gradient Inversion on PEFT (Sami et al., CVPR 2025) — PEFT dimensionality reduction *focuses* gradient info, making inversion easier; N≤128 on CIFAR-100
- ReCIT (2025) — reconstruct private data from PEFT gradients
- Deep Adversarial Decomposition (Zou et al., CVPR 2020) — learned superimposed image separation
- Cold Diffusion for Superimposed Image Decomposition (IEEE 2025)
- MAGIA (2025) — alternating subset gradient matching for federated learning

**Action:**
1. Quick win: wire `get_diversity_penalty` + cosine repulsion into extraction loop for N≥2
2. Quick experiment: post-hoc NMF on existing N=2 blended results
3. Medium-term: implement ICA on weight gradient matrix (Cocktail Party style)
4. For thesis: characterize the N vs. SSIM curve to find the practical phase transition

**This negative result is thesis-valuable:** It definitively closes the compose-and-reconstruct pathway and strengthens the argument for the Gradient Bridge / NTK approach, which works by targeting ΔW (canceling the pre-training component) rather than the composed W.

---

## [RESULT] Multi-Seed Validation: Free-c Beats Oracle, Seed=42 Was Outlier (2026-04-07)

**Context:** Ran 50-seed free-c vs oracle comparison (SGD+LeakyReLU, T=1, LoRA r=8) and 30-seed LeakyReLU validation across T and rank.

**Key findings:**
1. **Seed=42 was an outlier.** SSIM=0.830 on seed=42 vs 50-seed mean=0.558±0.034. Seed=42 happens to produce fine-tuning samples where the model is confidently wrong after centering, giving large coefficient magnitude. Most seeds produce moderate signal.
2. **Free-c beats oracle (46/50 seeds).** Mean SSIM: free-c 0.557 vs oracle 0.408. The consistency penalty |c − (σ(f(θ₀;x))−y)/N|² acts as implicit regularization: it prevents the sign-flip local minima that plague oracle mode (where fixed coefficients can mislead the pixel optimizer). Free-c can adjust c jointly with x, finding better overall solutions.
3. **LeakyReLU validated across seeds.** 30 seeds × {T=1, T=10} × {r=8, r=32}: SSIM 0.558±0.034 (T=1), 0.572±0.088 (T=10). Control: 0.394-0.426. Consistent gap (0.13-0.15) proves real leakage.
4. **r=16/32 fixed.** SGD+LeakyReLU gives r=16 SSIM 0.624 (was 0.422), r=32 SSIM 0.680 (was 0.415). The fix was switching from L-BFGS+ReLU to SGD+LeakyReLU.

**Action:** Use 50-seed statistics as canonical numbers in the thesis, not seed=42. The attack works but is moderate (SSIM ~0.55-0.58), not dramatic (0.83). Frame as: "reconstruction quality sufficient to identify sensitive content but not pixel-perfect" — which is actually more realistic for a privacy threat analysis.

---

## [PITFALL] Uniform-Spaced Snapshots Make Long-Run Progress Figures Unreadable (2026-04-28)

**What presented:** `figures/phase0/phase0_full_r8_n1_progress.png` rendered as a ~12000 px wide strip — 32 columns × 3 rows of tiny crammed thumbnails — once Phase 0 runs grew from 10K to 30K iters. Earlier 10K-iter runs gave readable 11-col figures.

**Root cause:** `save_progress_grid` and the training-loop snapshot save both used a hardcoded `snapshot_interval=1000`, so column count scaled linearly with `n_iters`. Going 10K → 30K tripled column count while figsize stayed at `2.5 * n_cols` inches.

**Fix:**
1. `save_progress_grid` now picks ~10 *log-spaced* iters (iter 0, ~100, ~300, ~1K, ~3K, …, final), snaps each target to the nearest available frame, and (with `cleanup=True`) deletes unused frames after rendering.
2. The training loop in `invert_gradient` now precomputes a log-spaced `snap_iter_set` and only saves frames at those iters (`snapshot_log_spaced=True` default). The d2 sweep keeps uniform behavior via `snapshot_log_spaced=False` because its 10-step uniform grid is already the right density.
3. One-off `tmp/rerender_and_cleanup.py` pruned 1642 frames (2355 → 713) across the existing snapshot dirs.

**Lesson:** Uniform sampling wastes most columns on the late-stage near-identical frames. Log-spacing matches the actual dynamics of gradient inversion (most progress in the first ~3K iters). Whenever a figure-generation function depends on run length, default to log spacing or cap the column count — never let n_iters silently set the figure width.

---

## [BUG] kornia FaceDetector backward returns NaN gradient via sqrt(0) (2026-04-29)

**What presented:** First differentiability test for the new face-structure prior failed with all-NaN gradient on the input image. The forward pass worked fine (the loss was a finite ~0.07 on `face1.jpg`); only `out['total'].backward()` produced `tensor(nan)` for every pixel.

**Root cause:** kornia 0.6.8's `FaceDetector.postprocess` (in `kornia/contrib/face_detection.py`) computes
```python
scores = (cls_scores * iou_scores.clamp(0.0, 1.0)).sqrt()
```
The YuNet `iou` head emits real-valued logits, including negatives (observed range −0.07 → 2.34). `iou.clamp(0, 1)` maps the negatives to exactly 0, so `cls * iou_clamped == 0` for those anchors, and `sqrt(0)` is taken. The autograd backward of `sqrt` is `0.5 / sqrt(z)`, which at `z = 0` evaluates to `inf`. Even though the threshold filter `inds = scores > confidence_threshold` later discards those anchors (so the upstream gradient at those positions is 0), IEEE-754 says `0 * inf = NaN`. That NaN propagates through the chain rule into every input pixel's gradient.

**Fix:** Monkeypatch `FaceDetector.postprocess` at load time with a NaN-safe variant that adds a tiny epsilon inside the sqrt:
```python
scores = (cls_scores * iou_scores.clamp(0.0, 1.0) + 1e-12).sqrt()
```
Now `sqrt(1e-12) ≈ 1e-6`, the gradient is `0.5 / 1e-6 = 5e5` — large but finite — and `0 * 5e5 = 0` cleanly. See [`_patch_postprocess_nan_safe`](experiments/face_prior.py) in `experiments/face_prior.py`. Tests pass after the patch.

**Diagnosis trick that found it:** Backward through each loss component separately (`presence`, `layout`, `symmetry`) showed the symmetry term — which doesn't traverse the detector — gave clean gradients, while presence and layout (which both touch the detector output) gave NaN. That isolated the bug to the YuNet path. Then `iou.min()` printed `-0.074`, which combined with the `clamp(0, 1)` and `.sqrt()` made the failure mode obvious.

**Lesson:** When a frozen pretrained model produces NaN gradients despite finite forward values, suspect `sqrt(z.clamp(min=0))` or `log(z.clamp(min=ε))` patterns where the clamp boundary is *exactly* the singular point. Adding `+ε` *inside* the sqrt/log (rather than relying on clamp) is the standard fix because it shifts the singularity away from the working domain. This is a generalizable autograd hazard, not specific to face detection.

---

## [DESIGN] Semantic priors need a warm-up before they engage (2026-04-29)

**Context:** When wiring the face-structure prior into Phase 0 ViT inversion, the natural impulse is to add `face_loss` to `total_loss` from iteration 0. This fails: at iter 0 the reconstruction is pure Gaussian noise — no face detector will fire. With no detection, the layout loss is undefined (no landmarks), the symmetry term defaults to a global-image symmetry (uninformative), and the presence loss falls back to a constant (no gradient). The face prior contributes nothing useful and risks destabilizing the early dynamics.

**Decision:** Default `--face_warmup_iters=5000` (no face term until iter 5000) and `--face_ramp_iters=2000` (linear ramp to full strength over the next 2000 iters). The TV term carries the first ~5K iters and produces enough coarse face-shaped structure that the detector reliably fires by iter 5K. This same pattern applies to any pretrained-model-based prior (LPIPS in classification space, ArcFace identity, etc.): the prior must be active over the *natural input distribution* of its source model, and noise is not in that distribution.

**Lesson:** When adding a frozen-model prior, the warm-up schedule isn't a tuning detail — it's load-bearing. Without it the optimization either stalls (no gradient signal) or blows up (NaN-bordering values from a model evaluated wildly out-of-distribution). Always start the prior off and ramp it in.

---

## [RESULT] D2 / D3 winner config transfers from flowers to a real face (2026-04-28)

**What happened.** The Phase 0 D3v2 ablation tested 7 freq+LPIPS prior configs on top of the D2 winner backbone (signAdam, tv=1e-1, lr=0.05, 30K iters) on a Flowers102 image. Best D3 result was SSIM=0.558 at freq=1e-3, within seed/restart noise of the prior-free D2 winner (SSIM=0.548). Then the same hyperparameters were applied to a real human portrait (`data/faces/face1.jpg`) with zero re-tuning: SSIM=0.522, PSNR=13.8 dB, cos_sim=0.974 — recognizable person, correct skin tone, collar, eye placement.

**Why it matters.** Flowers102 was the technical gate (texture-heavy, single foreground object). Faces are the privacy payload (the modality the thesis actually attacks). The fact that the same hyperparameters generalize means the per-image hyperparameter tuning concern (which would have been a thesis-credibility problem) is overblown — at least within the natural-image regime — and we can use one canonical config for downstream experiments instead of re-sweeping per image.

**Caveat.** Single seed only on the face number. Multi-seed validation is in flight (5 seeds, jobs 777058-777063). The transfer claim becomes stronger after we have mean±std.

**Side finding from D3v2.** Freq and LPIPS priors stacked on top of strong TV add nothing measurable. Strong freq (1e-1) and the combined freq+lpips configs actively *degrade* SSIM to 0.41-0.43 while cos_sim stays high (0.93-0.95) — classic over-regularization (loss matches, pixels wrong). TV at 1e-1 already does all the pixel-space prior work; the only remaining lever from extra priors is *semantic* (D4 face-structure prior, D6 latent / SDS), not additional smoothness.

---

## [DESIGN] Long WEXAC jobs need per-restart checkpoints, not just end-of-run saves (2026-05-13)

**Context.** Phase 0 inversion runs n_restarts independent optimization passes (default 8), each ~30K iters and ~1.5h on an L40S. A `--n_restarts 8 --n_iters 30000` job takes ~12h. WEXAC's `long-gpu` queue has a 48h wall, but earlier-finished jobs already at restart 4/8 had been getting killed before any `.pth` was written because the save happened only at the end of `invert_gradient`. We were losing hours of compute every time the queue rolled over.

**Fix.** `invert_gradient` now takes a `partial_save_fn(restart_idx, best_x, best_cos, loss_history)` callback. `run_phase0` wires it to the same `.pth` path the final save uses, with `metrics['partial']=True` and `restarts_completed` set. So a killed job at restart 4 leaves a valid 4-restart reconstruction on disk; the analyzer can consume it normally; a re-run can warm-start from it. Restarts can now be bumped (or dropped) without re-architecting the run script.

**Side effect.** Once partial saves were safe, the face-prior sweep dropped `n_restarts` from 8 → 4 and let 9 arms run in parallel instead of 5. Wall-clock for the full sweep dropped from ~26h to ~12h, with the option to re-run the winner at n_restarts=8 cheaply afterwards.

**Lesson.** Any optimization loop with N independent passes and a long per-pass cost should expose a per-pass-completion hook from day one. Treat "end of `for` loop is the only save point" as a code smell whenever the loop body costs more than ~30 min. Test in `experiments/tests/test_face_prior.py::test_partial_save_fn_called_each_restart`.

---

## [DESIGN] Chroma-coupled TV in LAB space targets the speckle TV-RGB can't see (2026-05-13)

**Why.** After D3, the visible failure mode on face1 (SSIM=0.522) is *colored speckle* — clusters of high-frequency RGB noise where the *spatial* gradient in each channel is small enough to slip under RGB-TV but the *cross-channel* coherence is wrong. RGB-TV penalizes `‖∂_x I‖²` per channel independently; it has no notion of "natural images have smooth chroma even when luminance varies". Speckle pixels are exactly where chroma varies fast while luminance stays roughly constant.

**Design.** Replace `tv_norm='l2'` with `tv_norm='lab'`. Convert `x_recon` to LAB (via `kornia.color.rgb_to_lab`), rescale channels to ~[0,1] (`L/100, a/128, b/128`) so `tv_weight=1e-1` still has the same dimensional magnitude, then take the per-channel squared-difference TV with a heavier coefficient on a and b than on L. Default `tv_chroma_weight=5.0`, sweeping {5, 20} as a safety check against over-flattening.

**What this is not.** It's not a replacement for D4 (face-structure prior). It addresses the *texture* failure (speckle), not the *layout* failure (eyes in the wrong place). Both can compose: chroma-TV on top of face-structure prior is the planned D4+D5 stack if both prove out individually.

**Lesson.** When a regularizer leaves a specific structured artifact (here, colored speckle), think about which property of natural images the regularizer fails to constrain. RGB-TV doesn't see chroma incoherence. LAB-TV does, almost for free. This is the cheapest possible perceptual improvement, much cheaper than LPIPS / SDS / latent-recon, and should be tried first when a low-frequency-only prior is leaving visible high-frequency color noise.

---

## [BUG] Smoke Tests Silently Overwrite Canonical Result Figures (2026-07-21)

**What happened.** While validating a new activation (`--finetune_activation gelu`), I ran
`experiments.run_experiment_b` with `--extraction_epochs 3` and **without** `--save_results`.
That 3-epoch garbage run still overwrote `figures/sprint1/experiment_b_grid_oracle.png` — a real
Sprint 1 result figure. It was caught only because `git status` showed the file as modified with a
timestamp minutes old; the corrupted figure was one `git add -A` away from being committed.

**How it presented.** A figure appearing as ` M` in `git status` that no one consciously edited.
Content looked plausible (same layout), so a casual diff review would not have flagged it.

**Root cause.** `generate_experiment_b_figure(results, save_dir=None)` in `experiments/plotting.py`
defaults `save_dir = FIGURES_DIR/'sprint1'`. `--save_results` gates the **tensor** (`.pth`) saving
(`run_experiment_b.py`, `if args.save_results:`) but **not** figure generation. So *any* invocation
that reaches the plotting call rewrites the canonical Sprint-1 figure, regardless of flags, epochs,
or how meaningless the run is.

**Fix applied.** `git checkout -- figures/sprint1/experiment_b_grid_oracle.png` to restore it.

**Lessons.**
1. **Never smoke-test an experiment entry point in the repo working tree** without checking what it
   writes. Smoke tests should use a throwaway `--save_dir`/output path, or run from `/tmp`.
2. **`git status` before every commit is a data-integrity check, not just hygiene.** Check *file
   mtimes* on anything unexpectedly modified — `ls -l --time-style=+"%F %H:%M"` tells you instantly
   whether it was you, minutes ago.
3. **Output paths that default to a canonical results directory are a trap.** A default of
   `figures/sprint1/` means every debug run is one call away from corrupting a published figure.
   Worth making the figure write conditional on `--save_results` too, or defaulting to a scratch dir.
4. Related to the standing data-freshness rule: a stale/corrupt *figure* is harder to detect than a
   stale *number*, because nothing greps for it.

**Fix applied (2026-07-21).** `generate_experiment_b_figure()` now takes a `base_name` argument and
`run_experiment_b.py` passes the per-config `base_name`, so figures no longer collide.

---

## [BUG] Swept Dimensions Missing From Output Filenames Destroyed 80% of a Sweep (2026-07-21)

**What happened.** Submitted a 43-config sweep (job 435843) over `finetune_activation`,
`n_per_class` and `loss_type`. The output filename was built as
`exp_b_T{n_steps}_r{rank}[_free]_s{seed}_a{relu_alpha}` — **none of the three swept dimensions
appear in it.** So all 5 activations at a given T wrote to the *same* `.pth`; Stage 2's
`n_per_class` values collided by seed; Stage 3's `l2`/`cosine` overwrote each other.
**43 runs would have collapsed to ~8 surviving files**, plus a single figure rewritten 43 times.
Caught ~10 minutes in by reading the job log; job killed and resubmitted.

**How it presented.** Nothing failed. Every run "succeeded" and printed
`Saved tensors to results/exp_b_T1_r8_s42_a149.pth` — the *same path* every time. Silent data loss
with a success message is the worst failure mode: you only notice at analysis time, hours later.

**Root cause.** `run_experiment_b.py:652-661` predates the activation / n_per_class / loss_type
flags. Each new CLI knob was added to the *parser* without being added to the *filename builder*.

**Fix applied.** `base_name` now appends `finetune_activation`, `npc{n_per_class}`, `loss_type` and
`lr{lr}` whenever they are non-default (non-default only, so existing result filenames stay stable).
The sweep script also grew a **Stage 0 guard** that runs two activations and asserts two distinct
`.pth` files exist, `exit 1` otherwise — so a collision aborts in minutes instead of wasting a night.

**Lessons.**
1. **Every swept dimension must appear in the output filename.** When adding a CLI flag that changes
   what a run *computes*, update the filename builder in the same commit — otherwise the sweep that
   uses it silently self-overwrites.
2. **Add a cheap self-check at the top of long sweeps.** A guard stage that proves outputs are
   distinct costs ~2 minutes and pays for itself the first time it fires. Generally: assert your
   *plumbing* before spending hours on your *science*.
3. **Read the first config's log before walking away from an overnight job.** Both this bug and the
   GELU confound below were visible in the first ~10 minutes of output.
4. Corollary to the "always save visual examples" rule: saving them to a **colliding path** is the
   same as not saving them.

---

## [INSIGHT] A "Negative Result" That Is Really an Un-tuned Learning Rate (2026-07-21)

**What happened.** First GELU run (Addition 2, LoRA r=8, T=1, oracle) returned **SSIM 0.0414 vs a
control of 0.0203** — essentially chance, against a LeakyReLU/ModifiedRelu baseline of ~0.797. Read
naively this says "GELU is catastrophically worse", which would directly contradict the prediction
that reconstruction quality *improves* with activation smoothness.

**Why it is probably not a real negative.** The diagnostics tell a different story:
`weight_change=0.039` (tiny), `delta_w_effective_rank=2`, `ntk_passed=False`. The fine-tuned network
barely moved from `θ₀`. Learning rate was `TRAIN_LR=0.01`, tuned for ReLU. GELU's smaller effective
gradient magnitude at that LR means the fine-tune essentially did nothing — so the attack is
inverting an update that carries almost no information. **We measured "nothing happened", not
"GELU leaks less".**

**Lessons.**
1. **When comparing activations (or any architectural change), match the effective update magnitude,
   not the nominal hyper-parameters.** Comparing at a fixed LR conflates "this activation is worse
   for reconstruction" with "this activation trained less at this LR".
2. `weight_change` / `delta_w_effective_rank` are the tell. A near-zero weight change means the
   reconstruction number is meaningless regardless of how it looks — **check them before reporting
   any SSIM.** Effective rank 2 for a rank-8 adapter is a red flag on its own.
3. **A surprising negative on a headline claim deserves a confound check before it is reported.**
   Handing a supervisor "GELU fails" when the truth is "our LR was wrong" is an expensive mistake.
   The sweep now calibrates LR ∈ {0.01, 0.03, 0.1, 0.3} per activation before drawing conclusions.

---

## [RESULT] Most Single-Step LoRA Reconstructions Sit At/Below the Trivial `ds_mean` Baseline (2026-07-22)

**What happened.** Adding a trivial-predictor baseline to the metrics (`ssim_mean_baseline` =
SSIM of the dataset mean vs each target) and re-scoring all 76 saved reconstructions
(`experiments/recompute_metrics.py`) showed **65/76 score below that baseline** on raw window-3
SSIM. By recon type: **full-model 10/28** beat it (the clean 0.96–0.99 runs clearly do), but
**LoRA 1/48** (mean 0.479 vs baseline 0.753). We had been reading LoRA numbers of 0.6–0.8 as
"good" with no baseline to compare against.

**Two confounds, pulling opposite directions, both now measured:**
- **Clipping understates:** `x_recon` saturates the soft [-1,1] box, so `x_recon + ds_mean` leaves
  [0,1] and the metric's clamp silently discards 40–60% of pixels. `ssim_norm` (mean/std matched)
  recovers +0.12–0.16 (gelu 0.041→0.198).
- **Baseline overstates:** with N=2 and MNIST's mostly-black background, `ds_mean` already scores
  ~0.76 against each digit. Most of SSIM is background agreement, not digit recovery.

**Root cause of the illusion.** SSIM on near-binary, mostly-black MNIST is dominated by the
constant background both images share (exactly the "background carries no signal" effect we flagged
for SimuDy). Absolute SSIM on this testbed is not a leakage discriminator.

**Lessons.**
1. **Always report a trivial-predictor baseline.** "0.68" means nothing until you know the dataset
   mean scores 0.76. Any metric without a floor invites reading noise as success.
2. **A metric that both understates (clipping) and overstates (baseline) is unusable raw.** Report
   `ssim` + `ssim_norm` + `ssim_mean_baseline` + `clipped_fraction` together, never `ssim` alone.
3. **The testbed itself is the problem, not just the metric.** N=2 makes `ds_mean ≈ each image`;
   MNIST makes background dominate. Larger N and harder data are needed for SSIM to mean anything —
   or a background-robust identifiability/retrieval metric.
4. **This sharpens gate B1** (can we recover from the adapter at all?): on raw SSIM, single-step
   LoRA does *not* yet clearly beat trivial. That is the honest current state, not a solved case.

---

## [INSIGHT] `ast.parse` Does Not Catch "return Outside Function" — Use `py_compile` (2026-07-22)

**What happened.** Added a `--skip_if_exists` early-exit using `return`, but the `main` body runs
under `if __name__ == '__main__':`, not a `def`, so `return` there is illegal. My syntax check
(`ast.parse`) reported **OK**, and the bug only surfaced when a pytest run failed to import the
module (`SyntaxError: 'return' outside function`). A GPU job submitted with that file would have
crashed on its first Python call (the compile error blocks import entirely, including the smoke
stage) — wasting the queue slot.

**Root cause.** `ast.parse` checks *grammar* only. "return outside function", "break outside loop",
etc. are *semantic* checks performed later by the symbol-table/compile pass, which `ast.parse`
never runs. `return` is grammatically valid anywhere.

**Lesson.** Validate scripts with **`python -m py_compile <file>`** (or `compile(src, f, 'exec')`),
not `ast.parse` — `py_compile` runs the full compile and catches these. Cheap, and it does not
import heavy deps (`import torch` costs ~270 s on the WEXAC login node, so a real import-based check
is expensive; `py_compile` sidesteps that while still catching compile-time errors). Even so, the
*test suite* is what actually caught this — CPU-only import tests earn their keep.

---

## [RESULT] Retrieval Metric Reveals LoRA Leakage That SSIM Declared Absent (2026-07-23)

**What happened.** After the metric audit concluded single-step LoRA "sits below the trivial SSIM
baseline," an instance-level **retrieval metric** (`experiments/retrieval_metric.py`) told a
different story. Retrieval asks: among the N training images, is reconstruction *i* most similar to
target *i*? Scoring the N-sweep LoRA runs (r=8, T=1, N=4..32), NCC/SSIM-space top-1 retrieval is
**~2.0–2.3x the 1/N random baseline at every N**; pooled across N and 3 seeds, **26 correct vs 12
expected by chance (z=4.3, one-sided p≈8.5e-6).**

**Insight.** Two metrics gave opposite verdicts on the *same* reconstructions:
- Absolute SSIM (background-dominated on MNIST) → "below trivial baseline, no leakage."
- Relative retrieval (background cancels across candidates) → "significant instance-level leakage."

The retrieval verdict is the trustworthy one for a *privacy* claim: leakage is about recovering
*which specific* training image, which is exactly the relative question. Pixel-L2 retrieval stayed
near chance; only the background-robust NCC/SSIM rankings surfaced the signal.

**Lessons.**
1. **Match the metric to the claim.** "Did we leak private data?" is an identifiability question
   (retrieval), not a pixel-fidelity one (SSIM). We nearly filed a false negative by using the wrong
   ruler.
2. **A weak-but-significant signal needs pooling, not per-run eyeballing.** Any single N=32 run
   (2/32 correct) looks like noise; 12 runs consistently ~2x chance is a 4-sigma effect. Report the
   pooled test, not the best run.
3. **Retrieval strengthens with N** (baseline 1/N), the opposite of absolute SSIM — another reason
   larger-N testbeds are the right move.

---

## [BUG] Full-Model Runs Need `run_baseline=True` — `--no_baseline` Silently Broke the N-Sweep (2026-07-23)

**What happened.** The N-sweep helper passed `--no_baseline` to *every* run. For LoRA that is
correct (skip the full reference, reconstruct the adapter). For the **full model**, the
reconstruction *is* the baseline run, so `rank=None` + `--no_baseline` hits
`run_single_config`'s guard: `ValueError: Nothing to run: rank is None and run_baseline is False`.
Result: all 15 full-model configs errored while all 12 LoRA configs succeeded — so the sweep looked
~half-productive and I only noticed because the full-model `.pth` files were missing.

**How it presented.** Missing output files, not a failed job — the job "completed" (exit 0 overall)
because each config is a separate process; the full-model ones just raised and moved on.

**Lesson.** When a sweep has two modes (full vs LoRA) that need *different* flags, don't route both
through one helper with hardcoded flags. Split the helpers (`run_full` without `--no_baseline`,
`run_lora` with it). And: **verify a sweep produced the files you expect per config**, not just that
the job exited cleanly — `ls results/…full…npc*` would have flagged this immediately.

