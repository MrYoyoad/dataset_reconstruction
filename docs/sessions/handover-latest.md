# Handover — 2026-09-03 17:15

## State
Branch `step1-activation-rescore-retrieval`. My lane this session was the **document**, not the compute:
`notes/exact_channel_rev10.tex` (~21 pp) is a self-contained, theorem-first Rev 10 delta section for the
Plan of Record, rebuilt from a prose draft into Definition → Lemma → hypotheses → Theorem → proof form and
audited by three independent sessions. It builds locally to `notes/exact_channel_rev10.pdf` via
`bash scripts/rev10_figs/build_pdf.sh` (static musl tectonic; the PDF is gitignored, the .tex is not).
The exact-inversion **experiments are owned by another session** (executor, currently `yoado-f4`; the
lineage rotated twice today: yoado-6c → yoado-b7 → yoado-f4). Ten of its jobs are still running.

## Done this session
- Rewrote the mathematical core as theorems with proofs; every hypothesis (A1)–(A6) stated as a formula
  where it is used. Added Corollary (general adapted layer: `k < m+r−N+1` without softmax, `k ≤ mr/N` for
  `N ≥ r`) and the Q-parametrisation (seed unknowns `N(N+1)/2` instead of `rN`), both derived-not-run.
- **Theorem I (capacity)** `k < m+r−N`: strict form DERIVED from the softmax simplex constraint
  (`1ᵀB_t = 0`), measured sharp at N=4,8,14, at five head widths, three ranks on MNIST, and on a TRAINED
  784→1000→1000→10 MLP with unseen test digits.
- **Theorem II (imprint)** `B_T = Σ_i C_i`, `C_i = −ηs Σ_t D_t[:,i](A_t h_i)ᵀ`: what the release records per
  example is bounded by that example's own accumulated error. Figure `figures/rev10/fig_imprint.png`.
- Wrote up, with scoping: encoder-quality ladder; where (A4) fails; out-of-distribution data; the
  attacker-side rank trigger and its prevalence; chart-dependence and chart-richness; new-class (CIFAR).
- **Four retractions of my own claims**, all caught by peer audit and all logged in LESSONS_LEARNED
  (items 10–15): per-example P_T columns (basis/order-dependent); the feature-Gram "coupling"; the
  "richer chart costs conditioning" trade-off; the averaged encoder rung (2.0e-15 — a value no measurement
  lies within three orders of).
- Restructured §measured into four labelled groups and rewrote §"What to say to Gal" as a two-theorem
  pitch (the last supervisor meeting failed on "correlations, no tool").

## Next step(s)
1. **The matched control for the EMNIST new-class cell (job 658575) has not landed.** Two `batch=new` rows
   are in: on the 98.2% MNIST model an 11th class (EMNIST 'a') gives `rank_B_T = 8` with all imprints within
   ~3.7× and imprint-Gram σ_N/σ_1 ≈ 0.092 (present and aligned). The pre-registered prediction is that the
   quality ladder FLATTENS for a new class where ordinary data loses rank — i.e. "how good the model is" and
   "has it seen this category" are independent axes of exposure. **Do not write that up until the
   same-job control (MNIST digits on the extended head, same chart and k) is in**, because at the strong
   encoder ordinary digits give rank 8 on-chart/distinct and rank 6 raw/distinct, so the comparison must be
   matched within the job.
2. The attacker-realisable arm (job 650890, `most_leaking.py`) is the first cell that starts from random
   public-scale coordinates rather than near the truth. Headline must be the **selection** — does
   `argmin_residual` pick the right label and image — not the success rate. `k ∈ {24,25,26}` tests the
   one-image line `k < m+r−1 = 25` in the only arm an attacker could occupy.
3. Job 652786 is the negative control that matters: act on a trigger that fires for the WRONG reason
   (random encoder, eight 1s, collinear features). If the argmin pick is wrong there, the .tex footnote on
   the trigger becomes a measured caveat instead of a caution.
4. Still open in §opens: global uniqueness; the fibre's extent past the line (one direction of forty walked,
   one cell still rising); the accuracy at which (A4) begins to fail; multi-layer LoRA (measured, but both
   arms 20 orders off the floor with the ORACLE arm nearly as bad, so no identifiability statement);
   Adam's predicted one-unit-higher line; the schedule family under weights-only release.

## Open threads / gotchas
- **Running jobs**: 614344 (strong/mid sweep), 624463 (chart budget rerun at 3000 iters), 624465 (β-VAE),
  624573 (labels/random-encoder), 634238 + 644064 (subset & OOD inversions), 650890 (most-leaking),
  652786 (negative control), 656205 (CIFAR new class), 658575 (MNIST/EMNIST new class).
- **Three peer sessions in play**: executor `yoado-f4`; theory auditor `yoado-77`; claims auditor `yoado-36`.
  They have both offered further passes. Names rotate — use `ListAgents` and re-identify rather than
  assuming a handle.
- **The tooling failure that bit three times today**: a Python patch script that applies several edits in
  memory and writes once at the end silently DISCARDS everything if a later assert fails — and it once
  produced a commit whose message claimed edits the file did not contain. Write after each edit, and verify
  target strings in the file rather than trusting the script's own "ok" lines.
- **Never read `jac_sigma_min` or `jac_cond`** — those are at the point the solver stopped. The theorems are
  about the Jacobian at the truth: use `jac_sigma_min_truth`, and compute cond as
  `jac_sigma_max_truth / jac_sigma_min_truth`. This substitution produced three separate wrong readings.
- **Never read the jsonl `verdict` field** into a document: it thresholds on image error alone and calls a
  9e-7-residual run "recovered".
- The `labels` field in `step51`/`step58` is the STRING "distinct"/"repeated", not a list — classifying by
  `len(set(...))` silently mis-splits every group.
- All MNIST/trained cells start from truth + 10% noise (`--init near`) and are identifiability tests, not
  attacks; the .tex says so. The only exception is 650890.

## Pointers
- Deliverable: `notes/exact_channel_rev10.tex` (+ wrapper `notes/exact_channel_rev10_main.tex`); build with
  `bash scripts/rev10_figs/build_pdf.sh`; figures `figures/rev10/` from `scripts/rev10_figs/fig_*.py`
  (CPU-only, read committed jsonl). Rev 9 source is Mac-only, so the .tex carries a MERGE NOTE mapping every
  Rev 9 number it uses — the user merges on Overleaf.
- Executor's authoritative write-up: `experiments/exact_inversion/RESULTS.md` (retracted claims are stamped
  in place, not deleted). Durable record: `STATUS.md` top section; pitfalls `LESSONS_LEARNED.md` items 10–15.
- Memory: `project_exact_inversion_capacity_law` carries the corrected framing — the law bounds the CHART
  COORDINATES, never say "wrong image" or "recognisable", and reconstruction factorises as
  capacity × chart quality.
