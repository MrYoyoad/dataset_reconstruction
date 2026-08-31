# Content & honesty audit — supervisor_meeting_2026_08_31_v21.pptx (37 physical slides, paged "/35")

Auditor: content/honesty sibling, 2026-08-31, read-only on deck/spec/generator. Input: full text + notes dump of all 37 slides.
Slide numbers below are **dump/physical numbers**; the page number printed on the slide is given as (pN). Physical 8 and 9 are the two
DI picture-only slides and also print "7 / 35", so p7 = physical 7–9; from physical 10 on, page = physical − 2.

## Executive summary
- **Counts:** WRONG 3 · CONTRADICTION 4 · OVERSTATED 4 · UNTRACEABLE 2 · SCOPE-MISSING 2 · BANNED-STRING 1 (notes-only) · OK-but-note 12. Everything else traced (see "Checked and clean").
- **Worst 1 — slide 10 (p8), oracle mislabelled as free-c:** the gallery ceiling (SSIM ~0.99, `results/gb_e2e_*_N2_gelu.pth`) was produced with **oracle coefficients** (`phase2_e2e.py:159 compute_known_coefficients`; STATUS.md:1450 "oracle coefficients + oracle per-layer sign (upper bound)"), but the slide says "recover the xᵢ (and the free cᵢ)" and the notes call it "the free-coefficient attack".
- **Worst 2 — ViT faces (slides 10/30/36 notes):** deck quotes per-image SSIM 0.38/0.26/0.52 (copied from the figure captions) while STATUS.md:2299–2308 (job 976038, the only logged N=3 joint run) records 0.603/0.674/0.710; the figure also titles the columns "Person 1/2/3" while the deck says "three portraits of ONE person".
- **Worst 3 — slide 11 (p9) SimuDy + slide 7 (p7) DI:** "our reframe as agreed in the follow-up mail" contradicts slide 30's own decision item ("agreed on your side?") and `notes/next_experiment_plan.md:20-24` ("drafted but never confirmed sent; Gal has not weighed in"); DI's "3 of 4 recovered" and its per-slot correlations (0.68/0.59/0.28/+0.67/−0.16) exist in no result file or STATUS entry (only `docs/presentation-remarks-log.md:42`, "resolved from cluster data").
- **N-sweep (new context):** v21 is clean — no slide implies free-c reconstruction quality beyond N=2 (the free-c showcase is not in v21 at all); the N≥4 material is the DI/endpoint and TRUE-ΔW ceiling settings, both labelled as such on the visible slide (p7 lead "endpoint matching", p8 lead "the ceiling"). The only gap is that p7 carries no visible "known-recipe upper bound / this attacker" line (notes only).

---

## Findings — worst first

### 1 · WRONG (oracle label missing) — slide 10 (p8) "More data: the full-gradient ceiling is recognizable"
- **claim (visible):** "the weight change as a coefficient-weighted sum of per-image feature gradients at the anchor θa … recover the xᵢ **(and the free cᵢ)** that reproduce it." **Notes:** "the free-coefficient attack recovers x_i and c_i jointly (Haim-style; oracle c_i is an upper bound only)"; "SSIM up to ~0.99 on MNIST / CIFAR / Flowers". Slide 2 notes repeat "SSIM up to ~0.99"; slide 30 notes "Full-gradient reconstruction works (SSIM up to ~0.99 …)"; slide 36 row E7 "SSIM ~0.99".
- **source:** `scripts/deck/make_deck_figures.py:525-549 fig_gallery` reads key `"TRUE ΔW (ceiling)"` from `results/gb_e2e_{mnist,cifar10,flowers32}_N2_gelu.pth`; those files are written by `experiments/gradient_bridge/phase2_e2e.py:159` `coeffs = compute_known_coefficients(m0, x_cen, y_ft)` (c_i from the TRUE images; `experiments/ntk_steps.py:47-53`). STATUS.md:1448-1453: "N=2, **oracle coefficients + oracle per-layer sign (upper bound)** … TRUE ΔW (ceiling) … gelu 0.995 / 0.997". `experiments/plot_reconstruction_baseline.py:55` labels the same arm "TRUE ΔW (oracle upper bound)". `notes/meeting_prep_2026-08-31.md` F-0 also omits the oracle-c fact.
- **verdict:** WRONG (the slide asserts free coefficients for an oracle-coefficient result; "oracle must be labelled").
- **fix:** visible → "recover the xᵢ that reproduce it, **with oracle cᵢ (coefficients taken from the true images — an upper bound, not the realistic free-coefficient attack)**". Lead → "everything here inverts the true weight change with oracle coefficients — the ceiling — not the released adapter". Notes/slide 36 E7 → "SSIM ~0.99 (**oracle-c** ceiling, N=2); the realistic free-c N=2 ceiling is 0.92 raw / 0.97 norm (MNIST T=5 leaky-ReLU, STATUS 2026-08-31)".

### 2 · CONTRADICTION — ViT faces per-image SSIM and identity (slide 10 notes, slide 30 notes, slide 36 E7 row)
- **claim:** "three faces recovered jointly from one captured fine-tuning gradient (figures/phase0/n3_three_faces.png; per-image SSIM **0.38 / 0.26 / 0.52**)"; "three portraits of **ONE person**".
- **source:** `figures/phase0/n3_three_faces.png` captions read "SSIM = 0.38 / 0.26 / 0.52" and title the columns "**Person 1 / Person 2 / Person 3**". STATUS.md:2299-2308 (job 976038, "face1.jpg + face2.jpg + face3.jpg (all the same person) jointly inverted"): "Per-image diagonals (each recon vs its own GT): **0.603 / 0.674 / 0.710**", aggregate 0.662. `notes/meeting_qa_cheatsheet.md:276` also says 0.60/0.67/0.71. No results file carries 0.38/0.26/0.52 (only `notes/thesis_note_v2.md:118` and `notes/mac_handoff_brief.md:106`, both transcribed from the figure). QA cheatsheet line 77 calls them "Real OOD portraits"; the deck notes instruct "say … own/consenting portraits".
- **verdict:** CONTRADICTION (figure vs STATUS on SSIM convention; figure vs deck on one-vs-three persons). The identity mismatch is the more damaging one if the slide is shown.
- **fix:** determine which SSIM the figure computed (a different window/normalisation than STATUS's 0.60/0.67/0.71) and quote one convention with its name; either retitle the figure columns "portrait 1/2/3 (one person)" or state on the slide "three portraits of one consenting subject". Until resolved, drop the three numbers from notes and say "structure returned, identity partially collapsed onto the centroid (STATUS 976038)".

### 3 · OVERSTATED + CONTRADICTION — slide 11 (p9) SimuDy characterisation
- **claim (visible):** (a) "full fine-tuning, full weight access, **known recipe (lr, T, batch)**"; (b) "no attack-independent measure: **success = the decoder worked**; failure is uninterpretable"; (c) "our reframe **as agreed in the follow-up mail**"; (d) "their MLP/100: SSIM ~0.34; ResNet/50: ~0.20; ~22 GB, ~15 h **per run**".
- **source:** `papers/Tian_2025_SimuDy_fulltext.txt`: Algorithm inputs include η, T, |B| (known recipe) **but** Fig. 2 shows recipe grid-search from the final loss; `notes/related_work_simudy.md`: "recipe (η, |B|, T) known, **or grid-searched** from early-loss (robust to wrong batch-size guess)". SimuDy has **no decoder** (it optimises dummy data through the unrolled run) and does report a linearity metric M_lin (`simudy_decision_brief.md` step 6). SSIM 0.3374 / 0.1982 ✓; Table 3: 22272 MB / 15.39 h is the **120-image** CIFAR cell (100 images: 19016 MB / 13.70 h). Reframe status: slide 30 (p28) asks "SimuDy reframe — agreed on your side?"; `notes/next_experiment_plan.md:20-24,266` "drafted but never confirmed sent; Gal has not weighed in"; `notes/thesis_scientific_summary.md:119,128` "SENT ~2026-08-21 … settled on the supervisor side" (repo-internal contradiction, flagged in handover-latest.md as unresolved).
- **verdict:** (a) OK-but-note (add "or grid-searched"); (b) OVERSTATED/inaccurate ("decoder"); (c) CONTRADICTION with slide 30 and with the plan; (d) OK-but-note (cell-specific).
- **fix:** (a) "known — or grid-searched — recipe (lr, T, batch)"; (b) "no attack-independent measure: success = the optimiser produced recognisable images; a failure cannot be attributed (our Worlds A/B/C question)"; (c) "our reframe, **proposed** in the follow-up mail — your view is decision 1 on slide 28"; (d) "~22 GB / ~15 h for 120 CIFAR images on ResNet-18 (their Table 3)". Also worth adding for fairness: "ViT shown at N=10 only". Otherwise the "what it does not do" column is fair to the paper and matches `related_work_simudy.md` §"Why we survive" 1–4.

### 4 · OVERSTATED + UNTRACEABLE — slide 7 (p7, and its picture slides 8–9) "Direct inversion works at small N"
- **claim:** title "works"; lead "recovers **3 of 4** at N = 4"; notes "per-slot, the recovered slots correlate 0.68 / 0.59 with the two 5s, slot 3 is a weak 0 (0.28), and slot 4 correlates +0.67 with the 0 and −0.16 with the 8".
- **source:** the per-slot correlations appear nowhere in STATUS.md, results/, or notes/; only `docs/presentation-remarks-log.md:42` ("Resolved from cluster data … corr 0.67 with the 0, −0.16 with the 8 → S7 lead says '3 of 4 recovered'"). The DI record (STATUS.md:1940-1949, 1978-1984) says: "At N=4 it **fails the mean baseline (0.43–0.58 vs 0.674)** but passes the control test (+0.17)"; ssim_norm 0.57 @N=4 ✓ (`results/direct_inversion_N4_r8_gelu.pth`); N=10 0.27 ✓; N=20 quoted "~0.15" but STATUS disagrees with itself (1773: ssim_norm 0.24 from job 887704; 1949: "0.15–0.18").
- **verdict:** OVERSTATED (title) + UNTRACEABLE (per-slot numbers; no `x_ctrl` saved, STATUS.md:1774-1776).
- **fix:** title → "Direct inversion recovers instance signal at small N"; lead → "endpoint matching recovers recognisable digits in 3 of 4 slots at N = 4 (below the mean-image baseline, above the same-class control), and degrades sharply by N = 10"; log the per-slot correlation script + numbers to `results/direct_inversion/` before quoting; notes N=20 → "0.15–0.24 (two STATUS reads)". Add a visible scope chip: "known recipe, full endpoint = best-case attacker (upper bound)".

### 5 · WRONG (notes) — slide 22 (p20) similarity ladder, the "1.39" rung
- **claim (notes):** "one same-digit rung even exceeds the cross-digit anchor"; "NON-monotone mid-ladder: the row-2 **same-digit exemplar** rung reads 1.39 vs cross-digit 1.00". (Visible text is fine: "same-digit exemplars occupy a noisy intermediate regime".)
- **source:** `results/similarity_ladder/similarity_ladder_summary.json`, target 1: `p3_rot15` sens 33.23 vs `r_cross` 23.86 → 1.39 is the **15° rotation** rung; the same-digit exemplar rungs on target 1 read r_nn 5.78 (0.24), r_med 17.71 (0.74), r_far 8.47 (0.36) — none exceed the cross-digit anchor. Target 0: rot15 2.83 (0.35), blur 4.29 (0.53).
- **verdict:** WRONG (mis-attributed rung; the non-monotonicity is real but it is a geometric-perturbation rung).
- **fix:** "NON-monotone mid-ladder: on target 2 the 15° rotation rung reads 1.39× the cross-digit swap; same-digit exemplars sit at 0.24–0.74". Same correction wherever "s=1.39 same-digit swap" is repeated (remarks-log item 10).

### 6 · CONTRADICTION (docs vs deck) — the A₀=0 span-leakage slide is absent from v21
- **claim:** STATUS.md "Deck: slide 8 added — A₀=0 first-layer LoRA publishes its input span (2026-08-31)" and `docs/sessions/handover-latest.md` "Deck now 29 slides: slide 8 = A₀=0 span-leakage result".
- **source:** the v21 dump contains no "span", "A₀", "row-span", LP or membership-selector text on any of the 37 slides (grep of the dump). The span slide lives only in the generator build (`figures/supervisor_meeting_2026_08_31_v1.pptx` lineage), not in the hand-finished v20/v21 spec.
- **verdict:** CONTRADICTION (STATUS/handover describe a deck that is not v21). No scope-line issue arises in v21 because no span number appears.
- **fix:** either re-add the slide to the spec (carrying the verbatim scope line from `notes/lora_span_leakage_note.md:3-7`: first-layer LoRA, A₀=0, SGD-family, N ≤ r, no gallery, reads only the released A) or amend STATUS/handover to say the span slide is in the generator build only.

### 7 · SCOPE-MISSING — absolute q_eff counts without the standing bias caveat: slide 14 (p12) and slide 15 (p13)
- **claim (visible):** p12 "51 / 117 / 150 / 156 at ε = 0.3 / 1 / 3 / 10"; p13 "23 → 13 → 0 directions at r = 8 / 16 / 32"; notes p12 "36 vs 59 of 80 at r=8".
- **source:** values ✓ (STATUS.md:52-53, 520-525, 739-745; `notes/thesis_note_v2.md:173`). But `notes/whitened_sensitivity_metric.md:115-123`: "running the bias-corrected / permutation estimator may **LOWER the published ABSOLUTE q_eff** (the 59/36 anchor, the roundB counts) … **before anyone quotes an absolute q_eff as 'N recoverable directions,' re-run the anchor**"; STATUS.md:52-53 repeats it as a standing caveat. The differential 23→13→0 is stated there to be robust.
- **verdict:** SCOPE-MISSING (p12 says "coordinate-dependent" and "quote the curve", but not "absolute counts pending the bias-corrected re-run").
- **fix:** p12 caption → "… quote the curve; absolute counts are provisional until the anchor is re-run through the bias-corrected estimator — the *differences* (next slide) are what is robust". p13 lead is a difference → fine once p12 carries the line.

### 8 · OVERSTATED (notes) — "World A is proven by the ruler": slide 14 notes, slide 30 notes
- **claim:** "Worlds: A is **proven** by the ruler (attack-independent)"; slide 30 notes "World A is proven by the ruler". (Visible p28 says "would establish A" — fine.)
- **source:** `notes/thesis_note_v2.md` §4: "World A is **measured** by the ruler … A is scoped and local … never 'no attacker can recover'"; triage B2 rescoped A to "a local first-order wall".
- **verdict:** OVERSTATED (observe-don't-conclude; also a banned-family word).
- **fix:** "A is *measured/placed* by the ruler — a scoped, local statement".

### 9 · CONTRADICTION (visible vs own notes) — slide 33 (p31) "+44% (retracted)"
- **claim (visible):** "Drift falls from +44% **(retracted)** to +6.3%." **Notes (same slide):** "a pure estimator bug, since fixed (do **not** present it as a scientific claim that was withdrawn)"; slide 35 notes: "(2) [removed from the slide per Yoad: the 2-way cross-fit winner's curse was an estimator bug …]".
- **source:** LESSONS_LEARNED.md:1681-1697 (3-way fix; +44% vs +6.3% ✓); `notes/whitened_sensitivity_metric.md:165-176` ✓.
- **verdict:** CONTRADICTION (wording policy).
- **fix:** "Drift falls from +44% (the 2-way estimator) to +6.3%."

### 10 · OVERSTATED — slide 37 (p35) "why these knobs" (hand-added, no notes)
- **claim:** "lr comes from a per-regime grid search … **frozen in calibration.json before any arm runs**"; "r8 vs r32 in duplication (**β is rank-invariant**)".
- **source:** the only `calibration.json` is `results/fullft_valley/calibration.json`, written by `experiments/dataset_sensitivity/fullft_valley.py:1256` for the full-FT valley wave (2026-08-29), i.e. **after** arms B–E ran (2026-08-27/28); the arms' lr=0.5 comes from "memprobe + yoado-35" (STATUS.md:738). Band [1e-4, 1e-3) ✓ (`fullft_valley.py:88`). β: MNIST 0.234 (r8) vs 0.241 (r32) ✓ (`results/arm_e_duplication/arm_e_summary.json`, STATUS.md:2793), but Fashion 0.288 vs 0.359 — STATUS.md:2896 "β(r32)>β(r8) slightly … capacity modulates duplication a bit on harder data".
- **verdict:** OVERSTATED on both.
- **fix:** "lr comes from a per-regime grid search into a fixed convergence band (max per-sample BCE ∈ [1e-4, 1e-3)), re-verified per arm; frozen in calibration.json for the full-FT wave"; "β is rank-invariant on MNIST (0.234 vs 0.241); Fashion shows mild rank dependence (0.29 vs 0.36)".

### 11 · WRONG (notes, minor) — slide 4 (p4) "fs > 0.99 … only for sigmoid/softplus"
- **claim (notes):** "fs > 0.99 (the 'linear regime' line) only briefly, and only for sigmoid/softplus."
- **source:** `results/rescored_tsweep_2026-08-29.csv` (recomputed): sigmoid 0.983/0.999/0.993/0.978/0.969/0.964 at T=1/2/5/10/20/50; **softplus peaks at 0.973** (T=2), never above 0.99. Also the notes' pooled Spearman(fs, ctrl_margin_norm) "−0.06" recomputes to **−0.034** (n=426); Spearman(fs, ssim_norm) +0.082 ✓.
- **verdict:** WRONG (minor; conclusion unaffected).
- **fix:** "fs > 0.99 only briefly and only for sigmoid; softplus plateaus ~0.88–0.97"; "≈ 0 (−0.03 to −0.06 depending on the pooling)".

### 12 · UNTRACEABLE (minor) — slide 5 (p5) notes "LoRA SSIM 0.06 -> 0.64"
- **claim (notes):** "on the first-pass SSIM read this was the 'interior optimum at alpha~0.75' (LoRA SSIM 0.06 -> 0.64)".
- **source:** not found in STATUS.md (the 2026-07 anchor headline block was searched by α/ssim keywords), crux_activation_analysis.md, or the anchor .pth notes. Everything else on p5 traces (see clean list).
- **verdict:** UNTRACEABLE (low stakes; first-pass, superseded).
- **fix:** drop the parenthetical or cite the exact table.

### 13 · SCOPE-MISSING (visible) — attacker framing on the DI slide, slide 7 (p7)
- **claim:** p7 shows recovered digits with no visible line on attacker knowledge or bound direction; notes have "known recipe (lr, T, full batch) = best-case upper bound; single seed per N".
- **source:** the deck's own policy (remarks log item 9: "scope line on every leakage-number slide"); slide 3 and slide 10 carry one, p7 does not.
- **verdict:** SCOPE-MISSING.
- **fix:** add the chip "known recipe + full endpoint: best-case attacker (upper bound), single seed per N, toy MLP".

### 14 · BANNED-STRING (notes only) — several slides
- slide 20 (p18) notes: "per-module **||Delta W||/||W_0||** ~ 0.23 … NOTES ONLY, never on a slide" (self-flagged); slide 36 notes: "Never put **0/40** or ssim_norm 0.61 on a slide", "**0.226** on the single LoRA target module — neither goes on a slide" (self-referential); slide 19 notes "Arm D … **confirms** the context effect is ~1.1x"; slide 14/30 notes "**proven** by the ruler" (item 8).
- **source:** `scripts/deck/SLIDE_CONTRACT.md` bans these on slides; no visible text contains "0/40", "‖ΔW‖/‖W₀‖", "0.226", "1.07", "ssim_norm 0.6x", "confirmed", or "settled" (the only "settled" is "settled on your side", allowed). ≤2-numbers rule: p12 shows 4 numbers in the curve (51/117/150/156) plus 160/640/1280 — a deliberate exception per its caption; p15 shows six AUC/error numbers (standard-method table); p33/p34/p35 are appendix.
- **verdict:** BANNED-STRING in notes only — OK as speaker prompts, but "confirms"/"proven" should go.
- **fix:** slide 19 notes "Arm D … reads the context effect at ~1.1×"; item 8 for "proven".

---

## OK-but-note (traced, minor wording or provenance drift)

| slide | claim | source | note |
|---|---|---|---|
| 2 (p2) TOC | "slides 3–4 / 5 / 6 / 7 / 8 / 9 / 10–26" | page numbers on slides | All TOC targets match the printed page numbers (SimuDy = p9 ✓, instrument = p10–p26 ✓). But physical 7, 8, 9 all print "7 / 35" — the page counter is not unique; if the picture slides are meant to be separate pages, renumber to /37. |
| 2 notes | "STATUS.md lines 49-105 (crux), 1667-1690 (DI, anchor)"; 12 notes "appendix 29" for the gates; 15 notes "Worlds A/B/C, next slide"; 33 notes "arm-B bars … live on slide 16 panel B", "(next slide)" for the arm table; 20/22/28 notes "slide R1/R4/R7" | STATUS.md has been prepended since (crux ladder now ~405-420, DI ~1978, anchor ~1990); knobs are p35 (p29 is the rank theorem); worlds are p28; p16 is the plan table (no panel B); R-numbering is the 2026-08-30 build | Stale cross-references in notes only; no number is wrong. |
| 3 (p3) notes | "kinked mean ~0.47 vs smooth mean ~0.09" | `results/rescored_freec_ladder_2026-08-29.csv` rung 0.1: relu 0.646, leaky 0.546, selu 0.506, hardswish 0.151 → mean 0.462; smooth 0.068–0.122 → 0.093 | ✓ only if hardswish is counted kinked; without it 0.566 (6×). Say "~5–6×". |
| 7 notes | "→ ~0.15 at N=20" | STATUS.md:1773 (ssim_norm 0.24, job 887704) vs :1949 ("0.15–0.18") | STATUS disagrees with itself; quote a range. |
| 11 (p9) | "~22 GB, ~15 h per run" | SimuDy Table 3: 22272 MB / 15.39 h at 120 images | cell-specific, see item 3d. |
| 17 (p15) notes | "skew −2.3 / excess kurtosis 36 at N=4" | `results/arm_b_dilution/arm_b_summary.json` **k2k (K=200)**: −2.33 / 36.0; at K=50 N=4: −0.08 / 13.5 | add "K=200". The AUC/error table (0.76/31%, 0.92/16%, 0.998/2%) recomputes exactly from Φ(d/√2), Φ(−d/2). |
| 18 (p16) | "four predictions failed" | plan §III table (`dataset_sensitivity_program_plan.md:172-186`) | B, E, C, D failed; g₀ indeterminate; S1/ViT/H held → ✓. Arm E "R² 0.76" ✓ JSON (plan's 0.85 is stale). |
| 16/32/33 | "resolved sensitivity at K … never lower bounds" | plan rule 3 (`…program_plan.md:118-121`) says "Magnitude = a LOWER BOUND at a STATED K" | deck is deliberately *more* conservative than the plan (per the 2026-08-30 audit); fine, but the plan should be updated to match. |
| 25 (p23) notes | "absolute footprint ~5x larger under full FT (target MEDIAN; per-target ~3-6x)" | STATUS.md:3069 "~5x (e.g. t1 …)"; :3093 "~5.8x LARGER (median per-target)"; thesis_note E5 "~5×, per-target 3–6×" | 5.8× median in STATUS; "~5×" is a rounding-down. |
| 25 (p23) | "Fashion-MNIST: reliable at every N from 4 to 32" | `results/arm_b_dilution/arm_b_summary_fashion.json` N=4 sens 1.25 p=0.002 (STATUS.md:2903 lists N=8/16/32 only) | ✓ from the JSON; the notes' "verify the N=4 cell" is resolved — delete the reminder. |
| 35 (p33) | retraction "Jang bound r ~ N → it is r ~ √N" | `notes/note_v2_review_triage.md` A1: Jang's own condition is r(r+1)/2 > K·N | write "r(r+1)/2 > K·N (Jang's own; √N at K=1)". Slide 15 notes already say this correctly. |
| 13 (p11) | "all 80 recorded, at every LoRA rank" | STATUS.md:511 r_J = 80 at every r | ✓ for r_J; the r=2/4 10-class cells are quarantined for q_eff (not memorised) — irrelevant to r_J, but this hand-added slide has no notes to say so. |
| 21 (p19) | "+0.83 on full fine-tuning (n = 6)" | STATUS.md:3071 ρ = +0.829 | ✓. |
| 36 (p34) | "arm B … flat; N=32 decline open" | `arm_b_summary.json` 22.4/24.3/23.8/12.8 | ✓ (deck's "22/24/24/13"). |

---

## Checked and clean (traced to source, value matches, framing OK)

- **Slide 1 / 2 notes:** two-kinds-of-number stance ✓ (`note_v2_review_triage.md` B3); "never say even the weakest attacker" respected on every slide (the only "weakest" is p28 notes "render the weakest / strongest singular directions", unrelated). DI 0.57 @N=4 (job 500913, STATUS.md:1980) ✓; 0.27 @N=10 ✓; gelu required (no double-backward for ModifiedReLU) ✓; ladder "~5× (job 392821)" ✓; anchor "job 532232" ✓; L-BFGS note internal.
- **Slide 3 (p3):** N=2, T=1, r=8, 13 activations, rungs {0.005, 0.03, 0.1, 0.3} ✓ (CSV weight_change ≈ 0.005/0.03/0.1/0.3); per-rung Spearman −0.48/−0.25/−0.27/−0.59 ✓ (STATUS.md:413, `crux_activation_analysis.md:6`); "do not quote a single canonical Spearman" ✓ (contract); oracle diamonds labelled "upper bound" ✓; lower-bound strapline ✓; two-cluster / selu / hardswish caveats ✓ (`thesis_note_v2.md:84`); "STATUS marks the T=1 ladder first-pass" ✓; flowers flip precedent ✓ (STATUS.md:406-407).
- **Slide 4 (p4):** job 390026, 65/65 ✓; T ∈ {1,2,5,10,20,50} ✓; kinked lowest at every T (relu 0.705→0.506, sigmoid 0.983→0.964) ✓ (CSV); gelu drifts to 0.59 ✓ ("smooth outlier"); Spearman(fs, ssim_norm) +0.08 (n=426) ✓ (`crux_activation_analysis.md:300-302`).
- **Slide 5 (p5):** α grid ✓ (`experiments/configs.py` per notes); relu lin-err 0.398 → 0.016 (25×) ✓, margins 0.382/0.385/0.621/0.535/0.569 ✓; softplus 0.087 → 0.010, 0.320 at α=0 ✓; gelu peak 0.502 @0.75, silu 0.292 ✓; full-space ssim_norm 0.96→0.56 ✓; flowers32 free-c gelu/softplus negative throughout, relu +0.141…+0.227 monotone ✓ — all `notes/crux_activation_analysis.md:161-207`. Attribution PASS/FAIL table on p33 identical ✓. "one seed; N=2" caveat present ✓.
- **Slide 6 (p6) and 31 (p29):** eff_rank(M) at N=10: relu 6.37 ≈ leaky 6.33 ≫ selu 3.39 > gelu 2.91 > mish 2.39 > silu 2.34 > softplus 1.73 > tanh 1.59 > sigmoid 1.19 ✓; softplus-β 0.5→50: eff_rank 1.40→5.30, mean|σ″| 0.125→3.26 ✓ (`results/gate_matrix_test.csv`); scalar-head scope G = M ⊙ (VᵀC) ✓ (triage B1); "local first-order wall, not non-injectivity (x→x³)" ✓ (B2); "SUGGESTED by the toy factorization — not proven" ✓.
- **Slide 10 (p8)** apart from item 1: "N=2 per dataset; ceiling only; faces from a captured full gradient, not an adapter" ✓; P_LoRA stated as one-step tangent ✓ (B4); "do not read this slide as the adapter attack succeeding" ✓; E7 = open milestone ✓; "never infer World B across cells" ✓ (`thesis_note_v2.md` E7).
- **Slide 12 (p10):** measurement-map framing ✓ (`identifiability_feasibility_revision.tex:41-72` per notes); rank theorem read as necessary floor, not verdict ✓.
- **Slide 14 (p12)** apart from item 7: S=320, N=10, k=8 ✓ (STATUS.md:510); iso 0.683 vs 0.491 ✓; 36 vs 59 ✓; N·k=160 cell S=640→1280 stable ✓ (STATUS.md:739); optimal-detector / "ε·σ=1 is a chosen threshold" ✓; Schur "conservative" sense ✓; FD 3.9e-8 @lr 0.6 → 1.0 @0.7 ✓ (STATUS.md:712); "2× multi-class" retraction = S=64 undersampling ✓ (STATUS.md:457, 45→97); "recovery reading not yet licensed (E1)" ✓; idea-provenance line ✓.
- **Slide 15 (p13):** 59/60/58 vs 36→47→58, gap 23→13→0 ✓ (STATUS.md:518-525); r=2/4 quarantined, job 635386 T≈4000–8000 past the FD wall ✓ (:537); r=32 dimY=57088, 3.2e-8 ✓ (:512); iso flip at r=16 (0.389 vs 0.808) ✓ (:528-529); Jang attribution and K convention ✓ (triage A1, D1); 2026 preprint caveat ✓ (A3); "Fashion 10-class bounded out" ✓ (:540).
- **Slide 16 (p14):** K=50 headline / K=20 scouting ✓ (plan rule 4); ΔW=BA gauge ✓; sign-flip null, p floor 1/501 ✓; "resolved sensitivity at K" ✓.
- **Slide 17 (p15) and 32 (p30):** four equivalences + S1/S2 failure modes ✓ (`whitened_sensitivity_metric.md:49-70` per notes); CRB wording "scalar shift t, F_z = JᵀΣ⁻¹J" ✓; "not an f-DP guarantee" ✓; "bounds adapter shift, not pixels" ✓; AUC/error numbers ✓ (recomputed).
- **Slide 18 (p16):** null-diag −0.001/+0.003, p 0.58/0.42 ✓ (`null_diag.json`); arm B flat, N=32 decline ✓; E β 0.234 (R² 0.76) / 0.241 (r32), β(T) 0.313→0.256→0.234 ✓ (JSON, STATUS.md:2793, 2824); C 3.28→0.34 ✓; D 1.11 ✓ (`arm_d_summary.json` 1.21/0.96/1.16); g₀ +0.857 / +0.777 (p 1e-4) ✓; S1 0.03–0.07 vs 8–24 ✓; ViT 3 targets p=0.002 ✓; H +0.881 ✓; "Fashion arm B and arm F not in table" ✓.
- **Slide 19 (p17):** m=1 raw ratio 7.1 ✓; class-1 sens 18.9/19.3/10.3/11.6(~11)/5.6 ✓; class-0 2.5–5.8 flat ✓ (`arm_c_summary{,_minc0}.json`); "label polarity untested" ✓ on the visible slide; 2.95 vs 4.70 margins, 1.50 vs 0.61 g₀ ✓ (STATUS.md:2885).
- **Slides 20–21 (p18–19):** g₀ definition ✓; +0.777, p=1e-4, CI [0.53, 0.91], half-width 0.189, terciles +0.88/+0.50/−0.12, λ +0.51 (n=24) / +0.538 (n=12), atypicality 0.05, partial 0.78 ✓ (`margin_at_scale/summary.json`, STATUS.md:2893, 3017-3021); precision-gate reading ✓ (triage C2); "no paired test", "KKT convergence not verified" ✓; USPS counterexample n=2 ✓ (STATUS.md:2908-2911); "lazy ~0.23 notes only" ✓.
- **Slide 22 (p20)** apart from item 5: identity rung 0 / p=1.000 ✓; d_pixel +0.807 > |Δg₀| +0.657 > DINO +0.399 ✓ (`pooled_spearman`; note n=20/20/18, deck says "n=18"); L0 0.0223 > L1 0.0133 > L2 0.0033 ✓ (STATUS.md:3072); "do not phrase as concept-vs-instance" ✓ (a correction over STATUS.md:2930's "records the CONCEPT" wording — good).
- **Slide 23 (p21):** d* defined as threshold crossing along a path, "not a radius" ✓ (triage C7); geomean 1.02, narrower 4/6, n=6 ✓ (`valley_headline_dstar.json`); d* vs d² disambiguation ✓; open items honest ✓.
- **Slide 24 (p22):** ρ=+0.881, p=1.5e-4, n=12; excl. m=1 +0.85 p=0.003; per-m 1.0/1.0/1.0/0.5; ρ(mem,g₀)=+0.798 ✓ (`h_spotcheck.json`); "provisional 'leakage'", full gate still required ✓.
- **Slides 25–27 (p23–25):** ViT-tiny rank-4 qkv blocks 0–2, 9,216 params, N=16, K=50, sens 1.13/1.24/1.52, fit 2.3e-4; MVP p=0.03 at N=6/K=10 ✓ (STATUS.md:2863-2894); removal ρ=+0.94 (0.943) ✓; d* geomean 1.02 / median 0.86 ✓; "no stable resolution advantage", magnitude ≠ resolution ✓; P_LoRA as tangent, "seed enters through A" ✓; "scope: gate theory exact only in the toy first layer" ✓ on the visible slide.
- **Slide 28 (p26):** 169 of 180, 11 dropped ✓; ARI +1.00 / activation −0.01 / lr −0.006 / seed −0.028; (B,A) by seed +0.546 ✓; +0.989 CI [0.973, 1.005], G=30 ✓ (`results/atlas/atlas_analyze_838868.txt`); "lead with ARI, +0.989 under re-audit" ✓; content-not-instance scope ✓; Learning-on-LoRAs venue ✓ (triage A4); "population (stronger-than-weakest) attacker" ✓.
- **Slide 30 (p28):** three worlds ✓ (`thesis_note_v2.md` §4); bridge 0.930 converged (0.951 best-epoch) ✓ (C4); q_eff curve in normalised local coordinates ✓ (B5, C3); F5 honest null at n=8 ✓ (STATUS.md:3135-3150); decisions list ✓; pushback table ✓ (matches note §5 except the "+0.989 under re-audit" softening, which is *more* cautious).
- **Slide 33 (p31)** apart from item 9: 3-way fold description ✓; +44% vs +6.3%, gate ≤ 15%, 60 synthetic datasets ✓; K=200 null reads ✓; arm B 22/24/24/13 at K=100, 8→22→46 ✓ (`arm_b_summary.json` k2k 46.1); identity rung ✓.
- **Slide 34 (p32):** construction ✓ (`jacobian_spectrum.py` per notes); "eff_rank reads backwards" ✓ (LESSONS_LEARNED.md:196-201); "'97 directions' used S=64 ≪ r_J=160" ✓.
- **Slide 35 (p33) retractions list:** 2× amplification ✓; ssim_norm vs raw baseline ✓ (STATUS.md:735-736); "init masks half" seed-42 ✓ (LESSONS_LEARNED.md:300-308, 759-765); g₀ tercile backwards ✓ (STATUS.md:3017); crux Spearman sign ✓; atlas fold bug ✓ (ab9eb99). The 2-way estimator bug correctly kept off the visible list (but see item 9).
- **Slide 36 (p34):** every row matches `notes/thesis_note_v2.md:170-178` + JSONs; the E4 job correction (695782 for full-FT, not 272309) ✓; "within-metric drops only" ✓ (C5).
- **New-context check (N-sweep):** no slide claims free-c reconstruction quality beyond N=2. The only N>2 reconstructions shown are DI (p7, endpoint/known-recipe) and the TRUE-ΔW ceiling (p8, N=2 anyway). `results/recon_showcase_sweep.csv` confirms the context: at N=4/6/10 every MNIST cell (LoRA and full) sits below the N-dependent mean-image baseline (e.g. N=4 full 0.605 vs 0.674; r8 0.358 vs 0.674; N=10 r8 0.289 vs 0.564) while `margin_norm` stays positive (0.08–0.32); STATUS.md:9-19 now carries the audited wording (`docs/sessions/v21_audit_nsweep.md`).
- **Hand-added slides 13, 20, 26 (p11, p18, p24):** definitional; no numbers beyond those verified above; no notes in v21 (commit f6f5905 adds them to the spec → v22).

---

## Addendum — v23 re-scope (2026-08-31, after yoado-6c's status check)

**Applicability.** v23 = user's hand-finished v20 + six speaker notes; the regenerated `deck_spec.json` (17:13) was
re-dumped and its VISIBLE text diffed against the text audited above: **identical** (only picture-shape markers differ).
Every visible-text finding above therefore applies to v23 unchanged. Below: the six NEW notes (slides 11/13/21/23/26/27),
checked number-by-number. Worst first.

| slide (notes) | claim (quote) | source | verdict | fix |
|---|---|---|---|---|
| 26 | "at this repo's init (A0 = 0, B0 random) the first step exposes the data only through the row space of a random B0" | `notes/thesis_note_v2.md:32`: P_LoRA(H)=BBᵀH+HAᵀA; with **B=0** the observation is H·AᵀA (row space of random A). With **A₀=0** the first step is B₀B₀ᵀH — the data is mixed through the **column space of B₀**, while the row/data side is exactly span{xᵢ}, seed-free (`notes/lora_span_leakage_note.md` §1: row(ΔW)=span{xᵢ}, D(re-seeded)=0.0000). | **WRONG** (mechanism inverted — says the opposite of the span note it cites as provenance) | "at this repo's init (A₀=0, B₀ random) the first step is B₀B₀ᵀH: the seed enters only through B₀'s column space; the row (data) side of ΔW is exactly span{xᵢ}, seed-independent. At the HF default (B₀=0) the roles swap: H·A₀ᵀA₀, and the observation lives in the row space of a random A₀." |
| 23 | "parametric near-duplicates read 0.03-0.07; a different digit reads 8-24" | `results/similarity_ladder/similarity_ladder_summary.json` (n=2 targets): only the **pixel-noise** rung reads 0.031/0.071. Brightness 0.39/0.83, rot5 0.77/6.03, rot15 **2.83/33.2**, blur 4.29/8.52 — rot15 on target 1 (33.2) EXCEEDS the cross-digit anchor (23.9). Cross-digit 8.06/23.86 ✓. | **OVERSTATED** (the 0.03–0.07 band is one rung, not "parametric near-duplicates") | "the pixel-noise rung reads 0.03–0.07 and brightness ≤0.8; rotation/blur climb into the cross-digit range (8–24), and a 15° rotation of one target reads above its cross-digit anchor — so 'concept not instance' holds for tiny perturbations, not for geometric ones. n=2 targets (job 268959)." |
| 27 | "absolute footprint is about 5x larger under full fine-tuning (per-target median; range 3-6x)" | `results/fullft_valley/F_summary.json` concat sensitivity full/LoRA per target: 7.55, 5.25, 6.33, 3.53, 8.34, **1.16** → median **5.8**. The "3–6×" range appears only in `notes/thesis_note_v2.md:104`, `notes/mac_handoff_brief.md:104`, `notes/meeting_prep_2026-08-31.md:96` — none of which shows a computation; the file gives 1.2–8.3×. | **UNTRACEABLE range** (the ~5× median is fine) | "about 5× (per-target median 5.8 on the concat sensitivity; per-target 1.2–8.3×, one target ≈1×)" — or drop the range. Also fix the same "3–6×" in the three notes files. |
| 11 | "if Gal asks, say it was sent ~2026-08-21 and invite him to confirm" | `notes/next_experiment_plan.md:23-24`: reframe "drafted but never confirmed sent; Gal has not weighed in… working assumption"; `:266` "proposed but unconfirmed". No file records a send; all 2026-08-21 STATUS entries are MineGrad/pixel-box, not the mail. | **UNTRACEABLE** (instructs the speaker to assert an unverified fact) | "say: 'I drafted a reply around Aug 21 — I'm not certain it reached you; here is the reframe in one line' — do not say it was sent." (Same issue as v21 slide 11 visible text, "as agreed in the follow-up mail".) |
| 13 | "seeds S must satisfy S >= 4 * N * k for the noise floor to be estimable" | No file states this rule. The anchor run used S=320 = 4·10·8 (`STATUS.md:520`, `notes/jacobian_leakage_experiment_plan.md:567`), and J1 found the full Σ_seed unmeasurable (eff_rank≈S−1), which is why `q_eff\|col(J)` is used. | **OK-but-note** (a design choice presented as a requirement) | "we used S=320=4·N·k; even so the full seed covariance is not estimable (eff. rank ≈ S−1), which is why the count is taken on col(J)." |
| 23 | "Per-image d* over 6 targets" next to the 0 / 0.03–0.07 / 8–24 numbers | Ladder numbers are n=**2** targets (job 268959); d* n=6 is the separate valley job 695782 (`valley_headline_dstar.json`). | OK-but-note (two runs blended in one sentence) | "the ladder numbers are n=2 targets; d* (n=6) comes from the valley run." |
| 21 | ρ=+0.78, CI [0.53,0.91], p=1e-4, n=24; λ-proxy +0.51; n=12 +0.857 vs +0.538; transfer +0.83 n=6; half-width 0.189; terciles +0.88/+0.50/−0.12; INDETERMINATE; jobs 260171/272504/695782 | `results/margin_at_scale/summary.json` headline/mechanism_table/verdict; `STATUS.md:2891-2893, 2952-2961, 3017-3024`; `F_summary.json` g0_piggyback 0.8286. | **OK** — every number traces exactly | — |
| 27 | Spearman ρ=+0.94 n=6; d* geomean 1.02 / median 0.86 / 4 of 6 / 2 flip; "never quote the arithmetic mean" | `F_summary.json` P5b 0.9429 n=6; `valley_headline_dstar.json` note (D/A 0.77–0.88 on 4/6, t4 1.33 / t10 1.75 flip). | **OK** | — |
| 11 | "SimuDy (Tian et al., ICLR 2025)", "same primitive, different question", "their scale is larger" | `notes/related_work_simudy.md:8-9` ✓; framing matches `simudy_decision_brief.md`. | OK (see v21 item 3 for the "known recipe" nuance — SimuDy grid-searches it) | — |
| 13 | N=10, k=8, 80 directions; double-backprop J checked against finite differences; q_eff on col(J) is a lower bound | `STATUS.md:520`; J0 FD residual 3.9e-9 (job 982855); `notes/whitened_sensitivity_metric.md`. | **OK** | — |
| 26 | P_LoRA(H)=BBᵀH+HAᵀA, self-adjoint PSD, not a projection; one-step statement; HF default B₀=0 | `notes/thesis_note_v2.md:32,167`; `lora_span_leakage_note.md` §3. | OK (apart from the inverted sentence above) | — |

**Net for v23:** 1 WRONG (slide 26 note, mechanism inverted), 1 OVERSTATED (slide 23 near-dup band), 2 UNTRACEABLE
(slide 27 "3–6×" range; slide 11 "sent ~08-21"), 2 OK-but-note; slides 21, 13 (numbers), 27 (ρ, d*) and 11 (citation)
are clean. The visible-text findings in the main report (oracle-c label on the ceiling gallery p8; ViT SSIM
0.38/0.26/0.52 vs STATUS 0.60/0.67/0.71; DI "works / 3 of 4" untraceable; p7/8/9 all printed "7 / 35"; slide 22 notes
1.39 rung; slide 33 "+44% (retracted)"; slide 37 calibration.json / β rank-invariance) all still stand for v23.
