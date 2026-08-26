# LoRA rank-sweep — visualization plan (plots + examples) — REV 2 (post yoado-30 audit)

**Purpose:** figures for *"the multi-class q_eff reversal is a genuinely LOW-RANK effect that attenuates with
rank and vanishes at full fine-tuning (r≥N)."* Designed by yoado-35/e9 (numbers verified vs log); audited by
yoado-30 (independently re-verified numbers + read STYLE_GUIDE); execute as a WEXAC bsub plotting job.

## Numbers — VERIFIED (twice: yoado-35 + yoado-30) against `scripts/wexac_logs/mc_rank_sweep_581629.out`
MNIST, gelu, N=10, k=8 (Nk=80), T=1000, lr=0.5, S=320, qr, seed42. **All r_J=80 (full); all FD-clean incl
r=32 (dimY=57088); anchor r=8 reproduces locked run 484948 (59/36, iso 0.49/0.68).** q_eff = q_eff|col(J), a
conservative LOWER bound ("≥ X directions"). q_eff at **ε=1** (reversal-clearest):

| r | regime | dimY | bin q_eff@ε1 | 10cls q_eff@ε1 | gap | bin iso | 10cls iso | 10cls converged? |
|---|---|---|---|---|---|---|---|---|
| 2 | r<N | 3568 | 57 | 7 | — | 0.197 | 0.923 | **NO** (max_bce 6.1e-3) |
| 4 | r<N | 7136 | 60 | 35 | — | 0.220 | 0.585 | **NO** (max_bce 1.9e-3) |
| 8 | r<N | 14272 | 59 | 36 | **23** | 0.491 | 0.683 | yes |
| 16 | r≥N | 28544 | 60 | 47 | **13** | 0.808 | 0.389 | yes |
| 32 | r≫N | 57088 | 58 | 58 | **~0** | 0.720 | 0.424 | yes |

Anchor r=8 ε-grid: bin 4/33/59/72/79, 10cls 0/4/36/71/72. {S,2S} stable at r=8 (36→34) & r=32 (58→60).
Held-acc harm asymmetry persists across r: binary Δ≈−0.09..−0.11, 10-class Δ≈−0.01..−0.02.

**Story beats:** (1) reversal ATTENUATES then VANISHES with r (gap 23→13→0). (2) MECHANISM is only
established at LOW rank — see the honesty note below. (3) reversal is LOW-ε (noise-limited), washes out at
high ε.

**⚠ MECHANISM CAVEAT (yoado-30, load-bearing for honesty):** the iso (noise-coupling) story explains the
reversal ONLY at r=8. At **r=16 the iso-gap FLIPS** — 10-class now has LESS noise in col(J) (0.39 < 0.81),
which ANTI-predicts the reversal, yet q_eff STILL reverses (47<60). So the r≥N reversal persists AGAINST the
iso prediction → the mechanism **DECOUPLES / is unexplained at r≥N**. The OUTCOME (q_eff gap 23→13→0) is
solid; the WHY is settled only at low rank. Figures must NOT claim iso "drives" the reversal at r≥N.

---

## 1. Plots (data read from .pth, NOT hardcoded)

### FIGURE 1 (ESSENTIAL) — headline. ε=1. `figures/rank_sweep/rank_sweep_headline.png`
2×2 (or 1×4 per STYLE_GUIDE sizing); panels:
- **A — q_eff vs r, CONVERGED r ONLY {8,16,32}, both bases.** (yoado-30 fix: the underfit r=2,4 are OFF this
  panel entirely — see §2, the gestalt of the huge r=2 gap must never touch the leakage axis.) x=rank log2,
  y=q_eff/80. Binary flat ~58–60; 10-class climbs 36→47→58 to meet it. Vertical band at **r=N=10** (between 8
  and 16) labeled "LoRA≈full-FT (Jang 2024)". THE money panel.
- **B — reversal gap (bin−10cls, ε=1) vs r, converged.** One line 23→13→0, 0 at r=32 annotated "full-FT:
  reversal gone." The punchline. (Droppable-first if the figure is dense — it's the vertical distance in A.)
- **C — iso_ratio vs r, both bases.** Binary rises ~monotone (0.20→0.81); 10-class non-monotone. Mark the
  iso-gap FLIP (between r=8 and r=16). **Annotation MUST read: "noise-coupling explains the reversal at r=8;
  by r=16 the iso-gap has FLIPPED yet q_eff still reverses — mechanism DECOUPLES (iso no longer explains it
  at r≥N)."** NOT "iso drives it."
- **D — convergence gate: max_bce vs r, both bases, with the 1e-3 line.** (yoado-30's fix for the honesty
  risk.) This is where r=2,4 10-class LIVE — showing they're above threshold (excluded), on a
  CONVERGENCE axis, so the misleading leakage-gap never appears on a leakage panel. Small companion panel.

### FIGURE 2 (RECOMMENDED — defeats cherry-pick) — `figures/rank_sweep/rank_sweep_eps.png`
Small multiples per CONVERGED rank {8,16,32}: q_eff vs ε (log-x), both bases. Shows the reversal lives at low
ε, washes out by ε≥3, shrinks with r → proves ε=1 in Fig 1 is representative, not cherry-picked.

### FIGURE 3 (SUPPLEMENTARY, optional) — `figures/rank_sweep/rank_sweep_spectrum_r8.png`
σ_i(J_SNR) with the ε=1 threshold line, r=8, both bases — the geometry behind the count (10-class has more
directions buried below the noise floor). Label **"geometry behind the count — NOT a reconstruction."**
Different visual grammar → separate fig, not in the headline (yoado-30). Good "what is q_eff" backup for Gal.

### Do NOT make: standalone gap-vs-r or standalone iso as separate files (they're Panels B/C).

---

## 2. Caveats — honest handling (hard constraint; ghosting alone is INSUFFICIENT — yoado-30)

**Underfit 10-class r=2,4 — REMOVE from the leakage panel; show on the convergence-gate panel (D) instead.**
The eye integrates the huge r=2 gap (57 vs 7) as gestalt BEFORE any annotation is read → "reversal biggest at
low rank" misread survives grey ghosting. So: **Panel A = converged r only**; r=2,4 appear only on Panel D
(max_bce-vs-r, above the 1e-3 line = visibly excluded). The misleading leakage-gap then never appears on a
leakage axis. **Fallback if forced to one panel:** STYLE_GUIDE out-of-spec zone = **red shading alpha=0.08**
over the r<8 region (reads instantly as "invalid zone," far stronger than grey markers) + annotation.

**Fashion — table-row + caption, NOT a plot; and the caption must be RANK-ACCURATE (yoado-30 provenance fix).**
Do NOT blanket-label "numerically unbounded." Accurate two-rank status:
- Fashion binary: r=8 q_eff@ε1≈30, r=16≈35 — fine to table.
- Fashion 10-class r=8: **FD-chaotic → bounded out** (fails the FD gate).
- Fashion 10-class r=16: **FD-CLEAN and CONVERGED (max_bce 2.1e-4)** but Σ_seed/q_eff|col(J) **did not emit**
  (separate failure, NOT FD-instability).
So: "Fashion 10-class q_eff unavailable — r=8 FD-chaotic, r=16 FD-clean+converged but q_eff did not emit."
No fashion crossing plot, no implied direction for the missing point. (I verified MNIST; the fashion status
is per yoado-30's log-read — whoever finalizes should confirm the fashion .pth if it gets cited prominently.)

---

## 3. Examples — NO pixel grids (wrong track)
Identifiability/geometry track ≠ pixel reconstruction; digit grids belong to the separate
`figures/combined/leakage_identifiability_plus_reconstruction.png`. Reprojection+SSIM would be base-dominated
and misleading. The table + curves ARE the result. The σ-spectrum (Fig 3) is the only honest "geometry
example," and it's supplementary, not headline.

## 4. Data source & single-source-of-truth
- Read `results/jacobian_j1_ranksweep_mnist_nc{2,10}_r{2,4,8,16,32}.pth` (metrics+tensors); do NOT hardcode.
- **Anchor self-check assertion:** loaded r=8 must == bin 59 / 10cls 36, iso 0.491/0.683 — abort otherwise.
- Provenance stamp: MNIST/N10/k8/T1000/lr0.5/S320/qr/seed42, job 581629, "verified vs log (yoado-35+30)."

## 5. STYLE_GUIDE specs (from yoado-30's read of STYLE_GUIDE/plots.md — authoritative)
- **DPI 250**; figure size ~**(16,10)** for the multi-panel headline; **seaborn base + guardrail font sizes**.
- **Palette (consistent every panel):** binary = blue **#1f77b4**; 10-class = orange **#ff7f0e**.
- **Out-of-spec / underfit region = red shading alpha=0.08** (the project convention — used on Panel D and the
  one-panel fallback).
- **Insight/annotation boxes = light-yellow rounded**; **value labels bold 11pt**.
- log2 rank x-axis; r=N=10 crossing marked identically in every rank panel; y as q_eff/80; ε=1 labeled
  "reversal-clearest ε" for transparency. float64 source; PNGs → `figures/rank_sweep/`.

## Gates / constraints
- **bsub-only, never local** (executor's script already excludes bad nodes: `hname!='lgn28' && !='hgn46'`).
- No stale numbers (read .pth; anchor self-check). Underfit r=2,4 OFF the leakage panel (convergence panel
  only). Fashion exclusion rank-accurate. ε=1 backed by the ε-figure. q_eff = "≥ X directions". Leakage =
  r_J/q_eff, never eff_rank. No pixel-reconstruction examples.
- **Iso is NOT claimed as the r≥N driver** — Panel C says "mechanism decouples/unexplained at r≥N."

## Changelog (rev 2, yoado-30 audit)
Honesty fix: Panel A converged-only + new convergence-gate Panel D (was: ghost r=2,4 in place). Substance:
Panel C annotation = "mechanism decouples, iso doesn't explain r≥N" (was: risked implying iso drives it).
Fashion caption: rank-accurate two-status (was: blanket "unbounded"). Added STYLE_GUIDE specifics (DPI/size/
palette/red-alpha/boxes). σ-spectrum confirmed supplementary. 3-panel kept, gap-vs-r droppable-first.
