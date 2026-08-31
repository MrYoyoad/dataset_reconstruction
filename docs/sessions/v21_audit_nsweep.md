# Audit — "reconstruction quality collapses with N" (metric / experiment-design lens)

Auditor: yoado-23, for yoado-6c. Read-only. Data: `results/recon_showcase_sweep.csv` + per-image tensors in
`results/exp_b_T5_*_free_s42_*_npc{2,3,5}_*.pth` (N=4/6/10). Per-image SSIM recomputed locally (numpy, min-max
per-image norm) — **absolute values differ from the CSV's SSIM convention; only the relative/identity pattern is used.**

## Bottom line — which parts of the claim survive

| Claim clause | Verdict | How to word it |
|---|---|---|
| "recognizable examples are an N=2 phenomenon" | **SURVIVES (strong)** | keep — per-image confirms no recognizable recovery at N≥4 |
| "at N≥4 neither LoRA nor full-FT beats the mean-image baseline" | **literally true, but MISLEADING as leakage** | reword: it is a *fidelity* statement; the baseline moves with N and the leakage-relevant control margin stays positive |
| "this is the mixing symmetry, not learning-rate under-tuning" | **PARTIAL — over-claims isolation** | reword: attack-limited (World B); mixing is the *evidenced leading* mechanism; lr ruled out at N=4 only; not isolated from other N-scaling confounds at N>4 |

**Recommended wording:** *"Free-coefficient reconstruction fidelity degrades with N — recognizable recovery is an N=2
phenomenon, and by N=4 the mean per-image reconstruction no longer beats the (N-dependent) mean-image baseline, with no
single image recovered recognizably at N=10. This is attack-limited (World B: the identifiability ruler shows the
information is present), and the per-image reconstructions carry a superposition signature (they collapse onto ~2
dominant images). We ruled out learning-rate under-tuning at N=4; we have not isolated the mixing symmetry from other
factors that scale with N (extraction iterations, restarts, free-coefficient count) at N>4, nor sampled beyond one
seed/draw. The instance-specific control margin stays weakly positive at every N (+0.11 at N=10), so 'below the
mean-image baseline' bounds fidelity, not per-image leakage."*

## Ranked findings

**1. [ATTRIBUTION — reword] "mixing symmetry, not lr under-tuning" over-states isolation.**
The lr control (job 528750) is **N=4 only**: full and r8 each have 4 lrs at N=4, and there the control margin is flat
(full 0.26–0.32 across lr∈{2e-4..6e-3}; r8 0.288–0.296) — so lr under-tuning IS convincingly excluded *at N=4*. But
N=6 and N=10 have a **single lr each** (0.0006 / 5.4e-4). Everything else that scales with N is uncontrolled: the number
of free coefficients being jointly fit, extraction iterations, restarts, per-image pixel-box. So "not lr under-tuning"
is established at N=4 and *extrapolated*. Note also lr and the readout fight each other: at N=4 full, the best raw ssim
(0.605 @ lr=2e-3) has the **lowest** margin (0.261) — tuning lr up improves fidelity but lowers instance-specificity.
*However* — the per-image structure (finding 3) is genuine evidence FOR mixing over generic optimization failure, so
the fix is to soften "not lr under-tuning → World-B, mixing is the evidenced mechanism, not isolated at N>4", not to drop it.

**2. [BASELINE — reword] the mean-image baseline moves with N, so "fails the baseline" is not one test across N.**
Baseline ssim(mean,true) = 0.763 → 0.674 → 0.606 → 0.564 for N=2/4/6/10 (the dataset mean is less like any one image
as N grows). Recon fidelity falls *faster* than the baseline, so recon drops below it at N≥4 — but this conflates
"low absolute fidelity" with "no leakage." The two readouts disagree in direction: by ssim−baseline the story is
N=2 positive → N≥4 negative; by the **control margin** (ssim(recon,true) − ssim(recon,control), a within-image contrast
that does NOT move with the dataset mean) it is +0.6–0.79 → +0.29 → +0.08–0.16 → **+0.11, positive at every N**.
Right invariant readouts: (a) the control margin (already stored: `margin_norm`); (b) a per-image identity/permutation
floor — Hungarian-match recon_i to true_j and report accuracy vs the 1/N chance floor; (c) a per-N sign-flip/shuffle
null on the margin. The moving mean-image baseline should be reported but not used as the leakage verdict.

**3. [PER-IMAGE — survives, and sharpens the mechanism] no single image is recovered at N=10; recons collapse onto ~2 attractors.**
Opened the N=4/6/10 full tensors. Per-image self-SSIM: at N=10 the *best* images reach only ~0.05–0.09 (raw), the rest
≈0 or negative — so the mean is NOT hiding a clean single recovery (the alarming "1 of 10 comes back" case is ruled
**out**). Identity-matching (does recon_i resemble true_i more than any true_j?) is only **2/4, 2/6, 2/10** correct —
i.e. degrading toward the 1/N chance floor — and the mis-matched recons overwhelmingly best-match the *same two*
images (the most mean-like ones). That collapse-onto-a-few-attractors is a direct superposition signature and supports
the mixing attribution over "recons are just noise." (Per-image deep-dive was on full; r8 shows the same aggregate
margin +0.107 at N=10, not separately per-imaged — minor gap.)

**4. [SAMPLE SIZE — gap] one seed, one image draw per N; the curve is non-monotonic.**
All cells are s42, one draw per N. Raw ssim is **non-monotonic** — N=6 (0.52–0.54) > N=4 (0.36–0.45) for both full and
r8 — which is itself a symptom of the moving-baseline confound (the margin is correctly lower at N=6 than N=4) AND a hint
of draw-sensitivity. No error bars. Before "collapses with N" is stated as a curve, it needs ≥3 seeds/draws per N; right
now it is a single trajectory through a noisy 2-D (fidelity, N) space.

**5. [minor] metric provenance.** The CSV `ssim` (e.g. N=4 full 0.451) is much higher than my raw per-image mean (~0.03),
so the CSV column is a normalized/kornia SSIM, not raw — fine for the paper as long as the *same* convention is used for
recon and baseline (it is: `ssim` vs `ssim_mean_baseline`). Flagging so the per-image numbers here are read as relative,
not as pipeline values.

## What the claim should become
The negative result is real and worth keeping — but as a **fidelity + attack-limited (World B)** statement, not an
information-limited one, and not with the mean-image baseline as the leakage bar. The strongest honest version leads
with the per-image collapse (no recognizable recovery, superposition onto ~2 images), scopes the mechanism to "evidenced
mixing, lr excluded at N=4, not isolated at N>4", keeps the control margin as the invariant that shows residual per-image
signal persists weakly, and flags the single-seed limitation.
