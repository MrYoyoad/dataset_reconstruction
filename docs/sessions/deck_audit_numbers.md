# Numbers audit — supervisor_meeting_2026_08_31.pptx (28 slides)

Audit date: 2026-08-30. Read-only pass. Every numeric claim on a slide (visible text + speaker
notes) was traced to its source of truth (STATUS.md, notes/thesis_note_v2.md, LESSONS_LEARNED.md,
results/*.json, rescored CSVs, wexac_logs, figure generator). Deck built by
scripts/deck/build_deck_2026_08_31.py from scripts/deck/deck/slides_*.py; figures by
scripts/deck/make_deck_figures.py.

## SUMMARY — WRONG / STALE / MISATTRIBUTED

Verdict: the deck is exceptionally clean. **Every** known retraction / correction listed in the
audit brief was already handled correctly on the slides. Only ONE genuine numeric discrepancy was
found, and it is minor.

| # | slide | number as shown | correct value | verdict | source |
|---|-------|-----------------|---------------|---------|--------|
| 1 | 16 (notes) & 28 (E-row) | arm E duplication **R² 0.85** (quoted jointly for β r8=0.234 / r32=0.241) | r8 R² = **0.757** (≈0.76) | STALE / imprecise (low severity) | `results/arm_e_duplication/arm_e_summary.json` scaling.8.r2_sensitivity = 0.7572; STATUS.md:2543 also says "T=1000 → 0.234 (R².76)". The 0.85 comes only from STATUS.md:2511-2512, which is internally inconsistent with 2543 and with the JSON. r32 MNIST R² has no JSON to verify (only STATUS asserts 0.85). Fix: quote r8 R²≈0.76 (or state per-rank: r8 0.76 / r32 0.85), not a blanket 0.85. |

No WRONG-magnitude, no MISATTRIBUTED, no live-retracted-claim items on any slide.

### All flagged known-retractions were handled correctly (no fix needed)
- **"‖ΔW‖/‖W₀‖ = 0.23"** — appears ONLY in speaker notes of slides 18 & 28, each explicitly labelled
  "NOTES ONLY, never on a slide" and "neither goes on a slide". Correctly identified as the
  weight-space linearization error of a full-FT config (job 480485), not a relative-update norm. The
  per-module 0.226 (summary.json lazy_diagnostic) is kept off-slide. No visible-text "0.23". OK.
- **Valley width "1.07"** — never appears anywhere in the deck. Slides 21/27/28 use geomean 1.02 /
  median 0.86 / "narrower 4/6, target-dependent" and slide 21 notes state "NEVER quote the arithmetic
  mean (Jensen-biased)". Matches fullft_valley/valley_headline_dstar.json (job 695782). OK.
- **"ssim_norm 0.6x" reconstruction** — every ssim_norm figure carries the "mean/std-matched, inflates
  the absolute" caveat; slide 28 notes: "Never put 0/40 or ssim_norm 0.61 on a slide"; slide 27
  retraction list includes "ssim_norm reconstruction → matched score vs raw baseline". OK.
- **"2× amplification" and "sharpens with N"** — both listed as retracted on slide 27
  (→ S=64 undersampling; → winner's-curse denominator) and in slides 11/14/16 notes. Not live. OK.
- **Jang "r ≳ N"** — slide 10 notes and slide 27 retraction list both state r ≳ **√N**
  (arXiv:2402.11867), and √(K·N) is explicitly labelled "OUR extrapolation, not Jang's". OK.
- **E6 composition "+0.00 / indeterminate"** — replaced everywhere by +0.989 CI[+0.973,+1.005] G=30
  (job 838868; 811847 flagged fold-buggy), described as content/digit-subset level NOT instance-level.
  Verified against scripts/wexac_logs/atlas_analyze_838868.out (ARI +1.000, (B,A)~seed +0.546,
  ΔW~seed −0.028, acc-diff +0.989 CI[+0.973,+1.005] G=30). OK.
- **Full-FT g₀ transfer "+0.83"** — slide 28 E4 row job = **695782** (NOT 272309). Slide 28 notes
  even flag that thesis_note_v2.md's E4 row wrongly lists 272309 ("272309 is the H gate"). Verified:
  results/fullft_valley/F_summary.json g0_piggyback.rho = 0.8286, n=6, job 695782. OK.
  (NOTE: the source doc notes/thesis_note_v2.md:150 still carries the stale "272309" — a NOTE bug the
  deck already corrected; worth fixing in the note but not a deck defect.)
- **g₀ cohort values** — +0.857 at n=12 (job 260171) and +0.777 at n=24 (job 272504,
  CI[0.53,0.91], graded indeterminate) used consistently on slides 15/18/23/28; g₀-vs-λ 0.857 vs
  0.538 (n=12) and 0.51 (n=24). No "+0.78 at n=12" cohort slip anywhere. Verified against
  results/margin_at_scale/summary.json (headline.rho_sens_g0=0.7765, ci95=[0.529,0.907], n=24;
  mechanism_table.rho_sens_lam=0.5104; tercile_rhos=[0.881,0.5,-0.119]). OK.
- **Figure-internal numbers** — estimator_honest "+44% / +6.3%" hard-coded at
  make_deck_figures.py:231, matches notes/whitened_sensitivity_metric.md:172-173. Anchor-lead "25×"
  = relu lora_lin_fs 0.398→0.016 (=24.9×) from anchor_sweep_T10_r8_relu_s42.pth. Knobs β read from
  data (arm_e scaling; MNIST r8=0.234, Fashion r8=0.288≈0.29). All OK.

## OK count

**~70 distinct numeric claims checked; all OK except item #1 above (arm E R²).** Full verification
detail below.

## STEP-2 verification detail (representative, by slide)

| slide | claim | source of truth | verdict |
|---|---|---|---|
| 2/7 | DI ssim_norm 0.57@N=4 → 0.27@N=10 → ~0.15@N=20 (jobs 500913/887704) | thesis_note_v2 E7; STATUS | OK (ssim_norm caveat present) |
| 3 | kinked ≈5× smooth; cluster means 0.47 vs 0.09; per-rung Spearman −0.48/−0.25/−0.27/−0.59 | STATUS.md:83-86,121; job 392821 | OK |
| 3/28 | Spearman(fs, ctrl-margin) ≈ 0 (−0.06) | thesis_note_v2 E3; rescored_tsweep (job 390026) | OK |
| 5/27 | relu lin-err 0.398→0.016 (25×); erratic non-monotone | anchor_sweep_T10_r8_relu_s42.pth lora_lin_fs=[0.398,0.027,0.07,0.032,0.016] | OK |
| 5/27 | softplus lin-err 0.087→0.010, lowest at every α | anchor_sweep softplus pth | OK |
| 8 | ViT faces per-image SSIM 0.38/0.26/0.52; SSIM up to ~0.99 | STATUS / thesis_note_v2 E7 | OK |
| 10/28 | E2 rank-sweep q_eff gap 23→13→0 (binary 59/60/58, 10-class 36/47/58) at r8/16/32; r_J=80 | job 581629; figure self-check asserts b[8].q[1.0]=59, m[8].q[1.0]=36 | OK |
| 11 | 36 vs 59 of 80 at r8; iso 0.683 vs 0.491 | STATUS.md:174; rank sweep | OK |
| 13/25 | gaussianity skew −2.3 / exkurt 36 at N=4 | arm_b_summary.json k2k: skew −2.327, exkurt 36.02 | OK |
| 14 | null-diag −0.001@N4 / +0.003@N16, p 0.58/0.42, q_eff=0; K=200 | results/arm_b_dilution/null_diag.json | OK |
| 14/25 | 2-way +44% vs 3-way +6.3% K-drift | whitened_sensitivity_metric.md:172-173 | OK |
| 15/28 | H gate ρ(mem,sens)=+0.881, p=1.5e-4, n=12; excl m=1 +0.85; ρ(mem,g₀)=+0.798; per-m 1/1/1/0.5 | h_spotcheck.json (0.8811, p=1.4999e-4, excl_m1 0.85, mem_g0 0.7983) | OK |
| 16/28 | arm E β=0.234(r8)/0.241(r32); β(T) 0.313→0.256→0.234; Fashion β 0.288(r8)/0.359(r32) R²0.99 | arm_e_summary.json (r8 β 0.2343), arm_e_summary_fashion.json (0.2885/0.3587, R² 0.995/0.993); STATUS.md:2543 β(T) | OK (β values); see item #1 for R² 0.85 |
| 16/28 | arm D rarity gain 1.21/0.96/1.16 → mean 1.11 | arm_d_context/arm_d_summary.json (m1/m8 ratios 1.206/0.957/1.164, mean 1.109) | OK |
| 17/28 | arm C balanced ratio 3.28 → 0.34 role-swap; m=1 raw 7.1 | arm_c_summary.json m8 ratio 3.28, m1 7.144; arm_c_summary_minc0.json m8 0.34 | OK |
| 18/28 | g₀ ρ +0.857(n12)/+0.777(n24) CI[0.53,0.91]; tercile +0.88/+0.50/−0.12; λ 0.51(n24); partial|atyp 0.78 | margin_at_scale/summary.json headline+mechanism_table+typicality_control | OK |
| 18/21 | full-FT g₀ transfer +0.83 (n=6, job 695782) | F_summary.json g0_piggyback.rho=0.8286 | OK (correctly 695782, not 272309) |
| 19/28 | S1 identity 0 (p=1.000); near-dup 0.03–0.07; cross-digit 8–24 | similarity_ladder_summary.json (identity 0.0/p1.0; noise 0.031/0.071; r_cross 8.06/23.86) | OK |
| 21/28 | ViT sens 1.13/1.24/1.52, p=0.002, fit 2.3e-4, 9,216 params, blocks 0-2 qkv | vit_lora_..._r4_N16.pth metrics (1.126/1.244/1.516, fit 2.29e-4) | OK |
| 21/28 | cross-regime rank ρ=+0.94; footprint ~5× (median); valley geomean 1.02/median 0.86/4-of-6 | F_summary.json P5b.rho=0.9429; valley_headline_dstar.json | OK |
| 22/28 | atlas ARI +1.00; raw(B,A) seed +0.55; ΔW scrubs −0.03; acc-diff +0.989 CI[0.973,1.005] G=30 | atlas_analyze_838868.out | OK |
| 27 | E7 bridge 0.951 cos; q_eff 156/160 | STATUS.md:1765 (0.951); STATUS.md:173 (156 at ε=10) | OK |
| 27 | retraction list (2× amp, sharpens-with-N, ssim_norm, init-masks, g₀ tercile, crux sign, Jang r√N, atlas +0.00) | LESSONS_LEARNED.md retractions | OK — all correctly listed |
| 28 | full provenance table E2–E7 + arm/gate jobs | thesis_note_v2.md:144-153 + results/*.json | OK (E4 job column corrected to 695782 vs the note's stale 272309) |
