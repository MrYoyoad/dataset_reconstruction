# Reconstruction showcase — VISUAL PASS (2026-08-31)

Reviewed: `figures/recon_showcase/freec_mnist_T5_lora_vs_full.png`,
`freec_flowers32_T{5,10,20}_lora_vs_full.png`, and `figures/recon_showcase/README.md` (zip cover note).
Lens: does each grid read correctly at a glance, any overclaim/under-scoped label, the N=2 superposition
question, layout defects. Auditor: yoado-ef.

## Summary

**The showcase reads well and is honest — no blocker.** Reconstructions are genuinely recognizable; every
figure states "free coefficients (realistic attack)", defines margin/clip/baseline, and prints the
per-figure baseline verdict (including the flowers failures — honest). The README is thorough and
weakest-attacker scoped. **One real title-vs-content mismatch (MED)** and **three clarity items (LOW)**.

## Findings

### MED — flowers titles say "LoRA vs full fine-tune" but the flowers grids have NO full-FT row
The flowers grids show `LoRA r=32 · LoRA r=8 · control` — **no "full fine-tune" row**, because the flowers
sweep has no `full` cell (make_recon_showcase.py:209 skips a slot not in `best`), yet the title
(make_recon_showcase.py:233) is the fixed string "— LoRA vs full fine-tune" regardless. So the title
promises a full-FT comparison the flowers figures don't deliver (MNIST does, correctly). The README's
overall "LoRA vs full fine-tune" framing over-promises for flowers the same way.
→ **Fix:** make the title conditional — when no `full` row is present, drop "vs full fine-tune" (e.g.
"Free-coefficient LoRA reconstruction, flowers32 …"); or add flowers full-FT cells to the sweep. Add one
README line: "flowers is LoRA-only (no full-FT cell in the flowers sweep)."

### LOW — control-row number placement is ambiguous (question a)
On every grid the control row shows the **control image** as the tile, with a number below it (e.g. flowers
`0.24 / 0.20`) that is actually **the reconstruction scored against that control** — not the control's own
score. This breaks the pattern of the other rows (where the tile IS the reconstruction and the number is
recon-vs-target), so a number sitting under the control image reads as "this image scored 0.24." The label
"(recon scored against it)" and the README line carry the meaning, but the number placement fights them.
→ **Fix:** label the control-row number explicitly, e.g. print it only in the right-hand row-label as
"recon vs this control: 0.24 / 0.20", or prefix the under-tile number with "recon↔".

### LOW — the MNIST LoRA rows show clear N=2 superposition; worth one note (question c)
The MNIST `LoRA r=16 / r=8` "5" tiles visibly **blend both private images** (horizontal ghost streak, the
"0" bleeding into the "5") — the expected N=2 mixing symmetry. The per-tile SSIM (0.81, 0.71) shows the
degradation but not *why*; a general viewer could read it as "LoRA just fails" rather than "the two images
superpose." **Recommendation: add a light note** (on the MNIST grid or the README) —
"LoRA rows blend the N=2 images (the mixing symmetry), which the raw SSIM registers as degradation." This
is the honest scientific framing and pre-empts a misread; the SSIM alone is defensible but weaker.

### LOW — cosmetic
- Control-row right-label prints an empty margin field as a double middot: "ssim 0.65 · norm 0.25 · · clip
  41%" → drop the empty segment for control rows.
- The three flowers grids (T=5/10/20) are near-identical (T barely changes the flowers recon: r=32 ≈ 0.61
  throughout) — not a defect, but the deliverable carries three near-duplicate figures; consider one line
  noting "T has little effect on the flowers recon" or keeping a single flowers-T panel.
- Titles lowercase the dataset ("mnist N=2", "flowers32") → "MNIST" / "Flowers-102" reads more finished.

## What's correct (verified visually)
- Free-c / realistic attack stated on every title + footer; margin, clip, baseline all defined in-figure.
- Per-figure baseline verdict printed — MNIST "all rows beat the dataset-mean baseline"; flowers "baseline
  gate FAILED for: …" (all flowers cells fail the brutal N=2 baseline, shown not hidden — matches the
  method-audit call).
- Clip fraction now on each row label (the method-audit fix landed).
- No overclaim / banned strings on any figure or in the README; weakest-attacker scoping present
  ("bound this attacker … not the reconstruction limit").
- Reconstructions are legible at render size; no clipped or overlapping text.

**Priority: fix the MED (flowers title) before zipping; the LOW items are polish.** r16/32 + T=10 rows
still landing — this is the method/label pass on the four current grids; a delta check follows.

---

## DELTA CHECK (2026-08-31, post-sweep — 62 cells)

**⚠ Note on method:** the image-read hook was DOWN during this pass, so I could NOT do the visual
glance-check (reads-at-a-glance / not-confusing). I verified the NUMBERS and label LOGIC from
results/recon_showcase_sweep.csv + the script; the visual eyeball is PENDING (recommend the requester
do a quick look, or I redo it when the hook recovers).

**(b) T=10 MNIST "baseline gate FAILED for LoRA r=8" vs +0.62 margin — ✓ CORRECT & honest.**
CSV: baseline (ssim_mean_baseline) = 0.7630 for MNIST. The selected r8 T10 cell = leaky_relu lr0.00027,
ssim 0.7387 (< 0.7630 → fails by 0.024), margin_norm +0.6196. So the footer "FAILED for LoRA r=8" is
correct, and the +0.62 is the CONTROL margin (recon much closer to the private image than to a same-class
control) — a different bar than the N=2 mean-image gate. Both true; the footer honestly prints the
mean-gate failure. Reads correctly numerically. (Same fail-mean-but-strong-control-margin pattern as
flowers.)

**(a) T=20 MNIST mixed-activation grid — ⚠ ONE DISCREPANCY TO RESOLVE.**
The activation-label mechanism is sound in code (make_recon_showcase.py:253-260: rows whose winning cell's
activation differs from the title's main activation get a "\n({act} net)" tag). BUT the SELECTION for the
r32 slot looks off vs gate-first: under `_score = (ssim>baseline, margin)`, the CSV has a **gate-PASSING**
r32 cell — `exp_b_T20_r32_..._relu_lr0.000445`, ssim **0.8529 > 0.7630 (gate=1)**, margin +0.4702 — which
should OUTRANK the failing cell the grid reportedly shows (relu lr0.000135, ssim 0.5839 < 0.7630, gate=0,
margin +0.5234). Gate-first must pick the (1, …) over the (0, …) cell. So the T20 r32 slot should show the
0.85 PASSING cell, not the 0.58 failing one. **Possible causes:** the 0.85 cell was dropped at load
(missing recon tensor / load error — note its ncc = 0.62 is anomalously low vs all others 15–1189, so it
may be degenerate), OR a selection gap for that slot. **Action:** confirm why the passing r32 cell isn't
shown — showing a gate-failing r32 when the CSV lists a gate-passing one needs either the passing cell
(and a one-line reason it's excluded) or a fix. Can't adjudicate without viewing (hook down).

**(c) Regression — code-side ✓, visual PENDING.** All prior fixes are present in the script: recon↔ prefix
+ "recon vs this control: ssim · norm" (l236-238), the "LoRA rows may blend the N images (mixing
symmetry)" note (l240), conditional title with activation + N ("{act} net, N={N}, T={T}", l251), and the
per-row "({act} net)" mismatch tag (l260). Whether they render cleanly is the pending visual item.

**RESOLVED (a)/r32:** no selection gap. The requester's quoted numbers were STALE (04:xx scoring, before
relu_lr0.000445 landed 10:39). The 10:53 render already selects the gate-PASSING cell for (mnist,20,r32):
exp_b_T20_r32_..._relu_lr0.000445, ssim 0.853 / norm 0.868, margin +0.47, beats baseline — labelled
"(ReLU net)" because the title's majority activation is leaky-ReLU. The v3 zip was built after that render,
so it carries the correct grid. (My CSV-based gate-first reasoning was right — the passing cell is the one
selected.) STILL PENDING: a visual eyeball of freec_mnist_T20 + T10 once the image-read hook recovers.
