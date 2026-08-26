# The LoRA leakage story — consolidated (identifiability × rank × reconstruction)

**One-line thesis result:** *LoRA fine-tuning encodes a large, recoverable fraction of its private training
data's latent geometry in the adapter — and multi-class training does **not** amplify this; if anything
cross-entropy **mildly self-protects** via its own training noise, but only in two low-resource regimes
(small attacker budget **and** low LoRA rank), and that protection **vanishes** as the attacker gains budget
or the adapter approaches full fine-tuning. (Converting that geometry to pixels via the current adapter-only
decoder does not yet beat the trivial mean-image baseline — §3.)*

Single source of truth for every number: this file + STATUS.md. Figures cited are all committed.

---

## 0. The setup (what "leakage" means here)

We treat LoRA fine-tuning as a differentiable map from the private data to the adapter weights, and measure
the **data-latent Jacobian** `J = ∂vec(A_T,B_T)/∂a`, computed EXACTLY by forward-over-reverse
double-backprop through the unrolled fine-tuning (validated, never computed, by finite differences). The
private coordinates `a` nudge each image along `k` directions. Two derived leakage quantities:

- **r_J** = hard rank of J = the number of private directions the adapter geometrically encodes.
- **q_eff(ε) = q_eff|col(J)** = the number of those directions an attacker with SNR budget `ε` can recover
  **against the masking of random-init noise** — a **conservative LOWER bound** ("≥ X directions").

Both are reparametrization-honest; leakage is stated as r_J / q_eff, **never** eff_rank (which reads the
spectrum's *shape*, not its leakage, and moves backwards). GELU, float64, WEXAC only.

---

## 1. Identifiability — leakage is HIGH for both bases; multi-class leaks FEWER (not amplified)

**Figure:** `figures/combined/leakage_identifiability_plus_reconstruction.png` (left panel).

At convergence, on the noise-whitened count `q_eff|col(J)`:
- **Both** binary (BCE) and multi-class (10-class CE) leak a **large** fraction of all private directions —
  at the canonical converged cell (N=20, T=1000, S=1280, of 160 directions): binary **54/119/151/156** and
  10-class **6/80/135/144** at ε=0.3/1/3/10 (results/jacobian_j1_roundB_mnist_nc{2,10}_T1000_S1280.pth).
- **Multi-class recovers FEWER at every ε** — the opposite of the intuition that more classes = more leakage.
  The old "multi-class ≈2× amplifies leakage (45→97)" claim is **DEAD** (it was a low-lr underfit +
  S-undersampling artifact).

**Mechanism (established at the anchor):** cross-entropy injects **more** of the random-init seed-noise
**into** the signal subspace col(J) — `iso_ratio = tr(Σ_J)/(μ·r_J)` is higher for 10-class (0.68 vs 0.49
binary at r=8) → a higher noise floor exactly where the signal lives → fewer directions clear the bar.

**Confirmed, not an artifact** (fully-converged N=10 clean lock, job 484948): both bases memorize (10-class
max_bce 7.3e-4), both FD-clean, and BOTH the mechanism (iso 0.68>0.49) AND the reversal (q_eff 36<59 @ε=1)
hold. Two honesty checks fold in: (a) the reversal is strong at **low ε** (noise-limited) and **vanishes at
high ε** (signal-limited) — exactly where the noise-floor mechanism predicts, hard to fake; (b) **held-acc
asymmetry** — memorizing HARMS binary held-acc (−0.10/−0.11) but NOT 10-class (~0.00), so 10-class leaks
LESS *and* at ~no utility cost.

---

## 2. Rank — the reversal is a genuinely LOW-RANK effect that vanishes at full fine-tuning

**Figures:** `figures/rank_sweep/rank_sweep_headline.png` (+ `_eps.png`, `_spectrum_r8.png`). Job 581629.

Sweeping LoRA rank r ∈ {2,4,8,16,32} at the locked converged recipe (MNIST, N=10, k=8, T=1000, lr=0.5,
S=320; all r_J=80, all FD-clean incl r=32 dimY=57088; anchor r=8 reproduces the lock 59/36):

| r | regime | binary q_eff@ε1 | 10-class q_eff@ε1 | **gap** | converged |
|---|---|---|---|---|---|
| 8 | r<N | 59 | 36 | **23** | both ✓ |
| 16 | r≥N | 60 | 47 | **13** | both ✓ |
| 32 | r≫N | 58 | **58** | **~0** | both ✓ |

- **The reversal gap closes monotonically 23 → 13 → 0.** Binary is flat (~58); 10-class climbs to meet it.
  By Jang (2024), r≥N ≈ full fine-tuning — so at r=32 the effect is **gone**: the reversal is a genuinely
  **low-rank** phenomenon, not a general CE property. This incidentally gives the **full-fine-tuning-regime**
  answer too (r=16,32 ≈ full-FT): the multi-class self-protection does not survive there.
- **The mechanism DECOUPLES at r≥N (honesty caveat):** the iso (noise-coupling) story explains the reversal
  only at r=8. By r=16 the iso-gap **flips** (10-class 0.39 < binary 0.81) — which anti-predicts the reversal
  — yet q_eff still reverses (47<60). So the OUTCOME (gap 23→13→0) is solid; the WHY is settled only at low
  rank. Figures state this explicitly and do **not** claim iso "drives" the r≥N reversal.
- **Low-rank measurability bracket:** r=2/4 10-class are **underfit** (max_bce > 1e-3) at the matched recipe
  → their q_eff is convergence-confounded and quarantined to the convergence-gate panel, never a leakage
  axis. r=1 is the only true capacity floor (never memorizes). So the clean like-for-like comparison is
  r=8/16/32.

---

## 3. Reconstruction — the geometry is real, but end-to-end pixel decoding does NOT yet beat the trivial baseline

**Figure:** `figures/combined/leakage_identifiability_plus_reconstruction.png` (right panel).
**Data:** `results/gb_e2e_{mnist,fashion}_N{2,4,10}_{gelu,softplus}.pth` (gradient-bridge Phase-2 e2e, Aug 2026).

Honest status (corrected against the metric gate — audit yoado-a2, verified here): the project's own metric
convention (experiments/metrics.py) is that **`ssim_mean_baseline` = what the trivial dataset-mean predictor
scores; a reconstruction at or below it carries NO instance-specific information.** Checked against that gate:

- The **DECODED (adapter-only) reconstruction does NOT beat the mean-image baseline on any MNIST cell.**
  Raw ssim (like-for-like with the raw baseline) 0.34–0.46 vs baseline 0.56–0.76 — e.g. mnist N=2 softplus
  decoded raw 0.43 vs baseline 0.76. The `ssim_norm ≈ 0.57–0.61` figure previously cited is the mean/std-
  MATCHED score (which removes the luminance/contrast penalty and inflates the number); it is **not** the
  like-for-like comparison to the raw baseline, and on the raw basis the decoder clears nothing on MNIST.
- Decoded beats baseline in only **3 of 12 cells** — all low-N fashion (fashion N10-gelu 0.255 vs 0.207,
  N4-gelu 0.291 vs 0.283, N4-softplus 0.323 vs 0.283) — and even there the absolute ssim is very low
  (barely-structured), so "recognizable" overstates it.
- The **TRUE ΔW (oracle upper bound)** does better (ssim_norm up to ~0.83) and confirms the *information* is
  present in the full gradient — but that is the known-recipe upper bound, not the adapter-only attack.

**So the honest claim:** the leakage is real and high as a *geometry / direction count* (§§1–2, verified),
and the *information* to reconstruct is present at the oracle (TRUE-ΔW) level — but the **end-to-end
gradient-bridge decoder from the adapter alone does not yet clear the trivial mean-image baseline** (except
marginally on a few low-N fashion cells). Pixel-level reconstruction is a **current limitation, not a settled
result.** (This corrects the earlier "recognizable digits, ssim 0.57–0.61, leakage REAL in pixels" framing in
STATUS.md line 32 and the combined-figure caption, which leaned on the baseline-uncorrected `ssim_norm`.)

---

## 4. The unified picture — two knobs, one story

The multi-class "reversal" (10-class leaks FEWER, not amplified) is **real and confirmed**, but it is
confined to low values of **two distinct knobs** — the attacker's budget ε and the training map's rank r —
and increasing **either** attenuates it to nothing:

| knob | reversal STRONG | ATTENUATING | GONE |
|---|---|---|---|
| attacker budget ε | low ε (noise-limited) | ε≈1 | high ε (ε≥3, signal-limited) |
| LoRA rank r | r<N (gap 23 at r=8) | r≈N (gap 13 at r=16, still present) | r≫N (gap ~0 at r=32 ≈ full-FT) |

(Note the rank knob is a *gradient*, not a cliff: the reversal is still present at r=16 — which is already
r≥N — and only vanishes at r=32/r≫N. The two knobs are genuinely distinct: ε is the attacker, r is the
training map — not one shared mechanism.)

So the safe reading for a defender is **not** "use many classes to leak less." Multi-class self-protection is
a fragile, low-budget/low-rank artifact of CE's training noise; a stronger attacker or a higher-rank adapter
(toward full fine-tuning) removes it, and in **all** regimes the leakage *geometry* stays **high** for both
bases (whether it converts to pixels is the separate, currently-unmet bar of §3).

---

## 5. Honest headline for Gal

*LoRA fine-tuning encodes a large, recoverable fraction of its private data's latent geometry — high for
both binary and multi-class. Multi-class does NOT amplify leakage — cross-entropy mildly self-protects, but
only at low attacker budget AND low rank, and that protection vanishes as ε grows or the adapter approaches
full fine-tuning (r≫N). The protective mechanism (CE couples more init-noise into the signal subspace) is
established at low rank and decouples at high rank — the outcome is robust, the "why" is a low-rank story.
The information is present to reconstruct at the oracle (TRUE-ΔW) level, but the current adapter-only decoder
does not yet beat the trivial mean-image baseline — pixel reconstruction is an open limitation, not a result.*

**Caveats carried honestly:** q_eff is a conservative LOWER bound. The N=20 reversal *magnitude* is bounded
by a characterized meta-gradient-chaos limit of the exact-Jacobian method (direction confirmed at
fully-converged N=10). Pixel reconstruction from the adapter alone is not yet demonstrated (§3). r=1 is the
capacity floor (never memorizes) per the low-rank convergence probe (job 635386, r∈{1,2,4,8}); the leakage
sweep itself is r∈{2,4,8,16,32}. Fashion 10-class is **numerically unstable** at the matched recipe
(FD/rigor/Σ_seed nondeterministic NaN via GPU-atomic nondeterminism) → bounded out; MNIST carries the
headline. Fashion binary is clean (q_eff@ε1 30/35 at r=8/16).

---

## Provenance
- Identifiability: jobs 399884/484948; data results/jacobian_j1_roundB_mnist_nc{2,10}_T1000_S1280.pth
  (canonical N=20 cell) + the N=10 lock 484948.
- Rank sweep: job 581629; data results/jacobian_j1_ranksweep_*.pth; plan notes/lora_rank_sweep_plan.md;
  figure plan notes/rank_sweep_plots_plan.md (designed yoado-e9, audited yoado-30).
- Reconstruction: gradient-bridge Phase-2 e2e; data results/gb_e2e_{mnist,fashion}_N{2,4,10}_*.pth (Aug 2026);
  metric gate experiments/metrics.py (ssim_mean_baseline). Decoded-vs-baseline table computed this session.
- Low-rank capacity floor: job 635386 (r∈{1,2,4,8}).
- Numbers verified vs logs (yoado-35 + yoado-30). Figures: figures/combined/ + figures/rank_sweep/.
