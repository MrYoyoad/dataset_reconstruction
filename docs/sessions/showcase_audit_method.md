# Method audit — free-coefficient LoRA-vs-full reconstruction showcase

**Scope:** read-only method audit of `scripts/deck/make_recon_showcase.py --tsweep` and the figures
`figures/recon_showcase/freec_{mnist,flowers32}_T{5,10,20}_lora_vs_full.png`, before the figures are
zipped for the user. Data: `results/exp_b_T*_free_s42_*.pth` (jobs 323866/323867/336206/341742).

## Summary

The showcase is **methodologically sound and honest**. The per-tile SSIM recompute reproduces the
stored metric-dict means **exactly** (4/4 dp on all 3 spot-checked files, both recon and control
rows) — centering conventions match `run_experiment_b.py` with no double-centering. The selection
rule implements the project's documented two-tier hygiene correctly: **baseline gate on RAW ssim
FIRST, then rank by control margin on ssim_norm** (`tsweep._score`, lines 348–352), matching
`LESSONS_LEARNED.md` §"An SSIM is not leakage until it beats the dataset-mean baseline". **No oracle
contamination**: all 19 files the glob matches carry `config['mode'] == 'FREE-COEFFICIENT'`. **No
banned strings** appear in the script or the rendered figures; every figure states "free
coefficients (realistic attack)" and prints the per-tile baseline verdict (including explicit
"baseline gate FAILED for: …" on the flowers panels). Two items to fix are **minor/imprecise, not
wrong**: (a) the code checks `config['free_coefficients']` nowhere — the oracle guard is
filename-only, and the config key the audit brief names does not exist (the real field is
`config['mode']`); (b) the clip fraction is not surfaced on the figure, though LESSONS mandates
reporting it alongside baseline + ssim_norm (here it is benign — high fraction, ~0.01 magnitude).
Recommendation on flowers: **keep the gate-failing cells in, exactly as done** (failure printed on
the tile), because the control margin is the correct instance-leakage bar at N=2 and the mean
baseline is disclosed as failed.

---

## Check 1 — Per-tile labels reproduce stored means  ✓ correct

**What the code does.** `per_tile()` (make_recon_showcase.py:60–89) recomputes per-image raw ssim /
ssim_norm via `experiments.metrics.compute_ssim` / `compute_ssim_normalized`. Centering:
- recon rows: `rec = d["x_recon_{key}"]` (already mean-centered), `tgt = d["x_train"] - dm`, and `dm`
  passed as `ds_mean` (line 82–83).
- control row: `rec = d["x_recon_{full|lora}"]`, `tgt = d["x_ctrl"] - dm`, `dm` passed (line 78–80).

This exactly mirrors what `run_experiment_b.py` stored:
`compute_all_metrics(x_recon_*, x_centered=x_ft - ds_mean, ds_mean)` (lines 586, 712) and
`compute_all_metrics(recon_for_ctrl, x_ctrl - ds_mean, ds_mean)` (line 739). Inside
`_prepare_pair` (metrics.py:47–72) `ds_mean` is added back to **both** operands, so the net input to
kornia SSIM is `(recon + ds_mean)` vs `x_train` — no double-centering, no mismatch. The showcase's
`to_img` add-back is display-only and independent of scoring.

**Spot-check (CPU recompute, login node, rec env — deterministic, LOADS stored tensors only, no
extraction/fine-tuning):**

| file | row | recompute ssim / norm | stored ssim / norm | match |
|------|-----|-----------------------|--------------------|-------|
| exp_b_T5_full_free_s42_a149_leaky_relu_lr0.002 | recon (full) | 0.9082 / 0.9178 | 0.9082 / 0.9178 | ✓ |
| " | control | 0.6528 / 0.2511 | 0.6528 / 0.2511 | ✓ |
| exp_b_T5_r8_free_s42_a149_leaky_relu_lr0.0054 | recon (lora) | 0.7451 / 0.8350 | 0.7451 / 0.8350 | ✓ |
| " | control | 0.5722 / 0.3520 | 0.5722 / 0.3520 | ✓ |
| exp_b_T10_flowers32_r8_free_s42_a10000_lr0.001 | recon (lora) | 0.6421 / 0.6254 | 0.6421 / 0.6254 | ✓ |
| " | control | 0.4462 / 0.4198 | 0.4462 / 0.4198 | ✓ |

All 6 recon/control means reproduce to 4 dp. **Verdict: ✓ correct.**

**One nuance (not a bug).** For the control row `per_tile` is called with `first_key` (the key of the
first rendered row: `full` if a full row exists, else the first LoRA rank present). The stored
`control_metrics` were computed with `recon_for_ctrl = x_recon_full if present else x_recon_lora`
(run_experiment_b.py:738). These agree **per-file** (each `.pth` holds a single recon type), and the
flowers files hold only LoRA — so `first_key` and `recon_for_ctrl` coincide in every case here. The
recompute above confirms the control means match, so the convention is consistent as used.

---

## Check 2 — Selection rule  ✓ correct

**What the code does.** `tsweep()` picks the best cell per `(dataset, T, rank)` via
`_score` (make_recon_showcase.py:348–352):
```
s, b = row.get("ssim") or -1, row.get("ssim_mean_baseline") or 9
return (1 if s > b else 0, row["margin_norm"])
```
`best[slot]` keeps the cell with the largest `_score` tuple. Tuple comparison ⇒ **baseline gate first**
(raw ssim vs the dataset-mean baseline; a gate-passing cell always outranks a gate-failing one),
**then** the leakage margin `margin_norm = ssim_norm(recon) − ssim_norm(control)` (line 128–130).

This matches project hygiene precisely:
- `experiments/metrics.py:112–125` — `compute_mean_baseline_ssim` is "the floor. A reconstruction
  scoring at or below it has learned nothing instance-specific." Confirmed the stored
  `ssim_mean_baseline` is computed against the **train** target (compute_all_metrics, metrics.py:160
  with `x_target = x_centered`), so the gate compares recon-vs-train against mean-vs-train — the right
  bar.
- `experiments/recompute_metrics.py:87` — `ctrl_margin_norm = ssim_norm − ctrl_ssim_norm`, exactly
  the margin used here.
- `LESSONS_LEARNED.md` §702–722: "report SSIM alongside the mean-baseline… prefer the control margin
  as the headline for instance-recovery claims"; §370 "Never rank reconstructions on raw SSIM alone."

**Evidence the gate does real work (MNIST r8, T5).** CSV cells: lr0.00178 has the **highest** margin
(0.6312) but raw ssim 0.7055 < baseline 0.7630 → gate FAILS; lr0.00054 has margin 0.6058, ssim 0.7897
> 0.7630 → gate PASSES. The figure shows lr0.00054 (ssim 0.79 · norm 0.86 · margin +0.61), i.e. the
gate correctly overrode the higher-margin-but-failing cell. **Verdict: ✓ correct.**

**⚠ Documented sub-case (flowers, all cells fail the gate).** When every cell fails (flowers N=2, see
Check 3), `_score` reduces to pure max-margin. Max-margin among gate-failing cells can pick a cell
whose recon ssim_norm is actually *lower* because its control happened to score lower (margin is a
difference). Example — flowers32 T10 r8: the selected lr0.003 has recon ssim_norm 0.575 / ctrl 0.335
→ margin 0.240, whereas lr0.001 has a **higher** recon ssim_norm 0.625 / ctrl 0.420 → margin 0.206.
So the displayed tile is the max-margin cell, not the max-absolute-recon cell. This is **defensible**
(margin is the sanctioned instance-leakage metric and cancels shared clip/scale, LESSONS §716–718)
but worth being aware of: on all-failing panels the shown recon is not necessarily the visually
closest one. No fix required; optionally note "max control-margin cell" in the caption.

---

## Check 3 — Flowers N=2 honesty  ✓ honest (recommend: keep, as done)

**Facts.** flowers32 dataset-mean baseline = 0.6464 (CSV, all flowers rows). At N=2 the mean image ≈
each image, so the raw-ssim gate is brutal — every flowers cell lands ssim ≈ 0.59–0.65, i.e. at or
just under 0.6464, and **all flowers cells fail the gate**. The selected cells nonetheless carry a
clear control margin (+0.19 … +0.24). The figures print this honestly: the note reads
"baseline gate FAILED for: LoRA r = 32, LoRA r = 8" (T10) / "…LoRA r = 8" (T20), and every tile shows
its raw ssim so the reader sees it is ≈0.6.

**Argument to exclude gate-failing flowers cells.** LESSONS §702 is blunt: "An SSIM is not leakage
until it beats the dataset-mean baseline." A purist reading says a panel where nothing beats the
floor shows no *instance* leakage and could mislead a skim-reader into "LoRA reconstructs flowers."

**Argument to keep them (with the failure shown).** LESSONS §715–722 refines the same rule: "the mean
baseline is **not the right bar for instance leakage — the same-class control is**… its margin cancels
the shared clipping/scale, so it's far more robust at small N." The whole point of the mnist-established
two-tier scheme is that the *control margin* is the headline and the mean gate is a disclosed
secondary bar that N=2 makes unfairly hard. Excluding the panel would hide a legitimately positive
control-margin result on the harder (natural-image) dataset; showing it **with the failure printed**
is the honest presentation the hygiene docs actually prescribe.

**Recommendation: KEEP the flowers panels exactly as rendered.** The margin (+0.19…+0.24 over a
same-class control) is the correct instance-leakage signal at N=2, and the mean-baseline failure is
disclosed on the figure, not buried. Two optional strengthenings (not blockers): (i) in the flowers
caption, state in words that raw ssim ≈ mean baseline at N=2 so the margin — not the absolute ssim —
is the claim; (ii) avoid any surrounding deck text that phrases these as "reconstructed the flower"
— the honest claim is "beats a same-class control by +0.2 ssim_norm; does not beat the N=2 mean."

---

## Check 4 — Oracle contamination  ✓ clean (but guard is filename-only)

**Glob + filter.** `scan()` (make_recon_showcase.py:295–298) unions
`glob("exp_b_T*_free_s42_*.pth")` and `glob("exp_b_T*_*_free_s42_*.pth")`, then filters by
`PAT` (line 288): `exp_b_T(\d+)(?:_(flowers32|fashion))?_(r\d+|full)_free_s42_a(?:149|10000)…`, with
`min_T=2` (T=1 reference files excluded). The regex only admits `flowers32`/`fashion`/mnist(default),
so `flowers64` files are excluded, and the literal `_free_s42_` is required.

**Config mode of every matched file.** Loaded `config` from all 19 matched `.pth` (read-only):

- **19 / 19** have `config['mode'] == 'FREE-COEFFICIENT'`.
- `config['free_coefficients']` is **`None` on all of them** — that key is **never written**;
  `save_dict['config']` (run_experiment_b.py:1000–1017) stores `'mode'`, not `'free_coefficients'`.
  So the audit brief's literal test ("free_coefficients=True in config") **cannot be satisfied by any
  file** and is the wrong field; `config['mode']` is the correct indicator and confirms free-coeff.
- No `*_oracle*` file exists under `exp_b_T*` (glob returned none), and no oracle-mode file is matched.

**Matched files (all mode=FREE-COEFFICIENT):** flowers32 T5/T10 r8+r32 (3 lr each), T20 r8 (3 lr),
mnist T5 full (leaky_relu) + T5 r8 (3 lr). **19 files.**

**Verdict: ✓ clean — no oracle contamination.** ⚠ **Imprecision to note:** the oracle guard is
purely the filename token `_free_`; the script never asserts `config['mode']`. If an oracle run were
ever mis-named with `_free_` in the filename it would slip through. A one-line hardening (skip any
loaded file whose `config['mode'] != 'FREE-COEFFICIENT'`) would make the guard robust. **Not a
current problem** — all present files are genuinely free-coefficient.

**Also note (data freshness):** `results/recon_showcase_sweep.csv` has 18 rows but the glob now
matches **19** files — `exp_b_T20_flowers32_r8_free_s42_a10000_lr0.0015.pth` landed after the CSV was
written. The figures (regenerated 03:21–03:23) include it; the CSV is slightly stale. Re-running
`--tsweep` regenerates both consistently.

---

## Check 5 — Overclaim / wording  ✓ correct (one minor omission)

**Banned strings** — grepped the script and viewed the rendered PNGs (mnist T5, flowers32 T10,
flowers32 T20):
- `"0/40"`, `"‖ΔW‖/‖W₀‖"`, `"0.226"`, `"1.07"`, `"ssim_norm 0.6x"`, `"confirmed"`/`"settled"` as a
  conclusion — **none present** in `make_recon_showcase.py` or on any figure.

**Wording present and correct on every figure:**
- "free coefficients (realistic attack)" is in the note (make_recon_showcase.py:742) and renders on
  each panel; titles say "Free-coefficient reconstruction". Oracle is named nowhere. ✓
- Per-tile metric is explained: "under each tile: raw ssim / ssim_norm of the reconstruction vs that
  tile's image" and "row label = mean; margin = ssim_norm(recon) − ssim_norm(control)". ✓
- Baseline verdict is stated per panel: "all rows beat the dataset-mean baseline" (mnist T5) or
  "baseline gate FAILED for: …" (flowers T10/T20). ✓
- The control row is labeled "same-class control image (recon scored against it)" and its margin is
  stripped from the label (line 736), so the control is not mislabeled as a reconstruction. ✓

**⚠ Minor omission (imprecise, not an overclaim).** `LESSONS_LEARNED.md` §710–714 rules: "report SSIM
alongside (a) the mean-baseline, (b) `ssim_norm`, and (c) the **clip fraction**." The figure reports
(a) and (b) but **not (c)**. From the CSV/stored dicts, mnist recons have `clipped_fraction` ≈
0.40–0.45 (flowers ≈ 0.0). That fraction *looks* alarming, but the stored `pre_clamp_min/max` for the
mnist full cell are −0.0096 / 1.005 — the clipping is high-**fraction** but ~0.01 in **magnitude**, so
it is benign and the raw ssim shown (0.79–0.91) is trustworthy here. Still, to satisfy the project's
own rule and pre-empt a reviewer question, either (i) add `clipped_fraction` to the per-panel note, or
(ii) add a one-line caveat that mnist raw ssim is computed after a negligible-magnitude clamp.
Selection is unaffected — it ranks on the clip-robust ssim_norm margin. **Verdict: ✓ correct wording;
⚠ add the clip-fraction disclosure to fully meet hygiene.**

---

## Fix list (all optional; none block the zip)

1. **[minor] Clip fraction on the figure** (Check 5) — surface `clipped_fraction` (or a one-line
   caveat) so mnist raw ssim carries its clip context, per LESSONS §710–714.
2. **[minor] Harden the oracle guard** (Check 4) — skip any loaded file whose
   `config['mode'] != 'FREE-COEFFICIENT'` instead of trusting the `_free_` filename token.
3. **[cosmetic] Flowers caption** (Check 3) — state that at N=2 raw ssim ≈ mean baseline, so the
   control margin (not absolute ssim) is the claim.
4. **[cosmetic] All-failing panels** (Check 2) — optionally label the shown cell "max control-margin"
   so it's clear it is not necessarily the max-absolute-recon cell.
5. **[housekeeping] Regenerate the CSV** (Check 4) — `recon_showcase_sweep.csv` (18 rows) is one file
   behind the glob (19 files); re-run `--tsweep`.
