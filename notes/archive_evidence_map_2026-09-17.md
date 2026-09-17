# Evidence map — every empirical claim in the 2026-09-17 archive, traced

The repository's own evidence records stay where they are: `results/CLAIMS_LEDGER.md` is the claim → evidence
index with audit status, `results/00_map.md` maps code and releases, and each track's
`experiments/<track>/RESULT(S).md` is its authority with rows in `results/<track>/`. **This file adds the one
thing those cannot give: where each number in the archive's documents and figures actually came from**, and
whether it survives contact with the rows.

Verdicts used below:

- **TRACED** — the number is reproduced by a row or a function in this repository, and the job id is named.
- **ARCHIVE-ONLY** — the claim's generator exists only in the archive; it has never run here.
- **SUPERSEDED** — traced, but a later run in this repo reports a different number for the same cell.
- **UNTRACED** — no script, row or log anywhere, in the archive or the repo.

Job ids below were read from `results/**/*.jsonl` rows, not from adjacent prose.

---

## 1. The figure pack shown on 2026-09-15

| panel | claim it makes | traces to | verdict |
|---|---|---|---|
| `03_reconstruction_results`, letters | "8/8 PCA targets"; four rows original / PCA training target / NTK output / certificate output | `figures/ntk_vs_cert/mnist_letter_a+letter_t_k32_N8_r64.png`, generator `experiments/cifar/ntk_vs_certificate.py`, cell `mnist_mixed`; the raw row is a re-raster of `figures/exact_inversion/letters_recovery_k32_760909.png`, job **760909** | **TRACED** — with the scope note that the eight-example run is the **mixed a+t batch**, not eight A's; only the four A columns are shown |
| same, NTK row | the NTK row is a real free-coefficient linearised fit, errors ≈0.48–0.62 | same figure, row "NTK linearised (free coefficients)" | **TRACED** |
| `03_reconstruction_results`, motorcycles | "8/8 PCA targets", original/reconstruction pairs | job **335732**, `results/cifar_newclass/nc_structured_335732.jsonl`: `arch mlp, k 32, r 64, m 11, n 1000, N 8, T 400, starts 200, landed 135, images_found 8, landings_per_image [6,33,12,7,17,33,14,13]`; residual at truths 1.6e-14 against public median 4.2e-01 | **TRACED** |
| `motorcycle_results_template` | "8/8 training inputs recovered", "135/200 successful starts" | same row | **TRACED** |
| `03_reconstruction_results`, Fashion bags | "4/8 PCA targets + 2 approximations shown" — **not a wrong number: the same cell read at a smaller budget** (see §2a) | job **473802**, `results/cifar_newclass/sharp2_473802.jsonl`: `landed 9/150`, `images_found 4`, `landings_per_image [0,1,2,0,0,2,4,0]` (landed columns 2, 3, 6, 7 — exactly the panel's), residual max 1.3e-13 against public median 3.86e-01, CNN 92.67% test / 99.83% train | **SUPERSEDED** — the repo's 400-start re-run of the same cell reports **23/400 landed, 6 of 8** (job **556643**, `figures/cifar_newclass/cnn_fashion_bag_k64_onchart.png`) |
| `07_mnist_coverage` | three chart strips: public PCA, warped, "private-built reference" | `figures/exact_inversion/chart_dependence_k17.png`, generator `experiments/exact_inversion/chart_dependence.py`; per-row errors 5.1e-01 / 5.1e-01 / **5.7e-14** | **TRACED**, with a labelling hazard: the archive's strip omits that the third chart is **oracle-built from the private digits and not an attack**; the repo figure says so on its face. Its three source crops are byte-identical to one another, so each digit is paired with a copy of itself |
| `06_multilayer` | `k=32`, `d=3072`, `m=11`, `r=64`, `q=8`, "56 equations per image" | `cert_line = rank_C = 56`, `m = 11` in every `r=64` row of `results/cifar_newclass/*.jsonl` | **TRACED** |
| `02_certificate_subspaces`, `04_ntk_partial_information`, `05_image_family`, `08_local_families`, `01_one_layer`, `figure_pack` pages 1–3 and 6–8 | the certificate invariant, the one-step NTK gauge, the toy score, local families | schematics with no data; the algebra is asserted by `scripts/figpack_2026_09_15/validate.py` (synthetic, seed 713: `‖BP‖<1e-10`, `‖P(A−A0)‖<1e-10`, `rank B = q`, `rank(PA) = r−q`) | **ILLUSTRATIVE**; the same invariants are checked in-repo by `experiments/exact_inversion/certificate.py` and `experiments/multilayer_cert/theory_checks.py` (`sanity.CH_rel ≈ 1e-14…1e-16` in every row) |
| figure-pack prose | "normalized glyph correlations are approximately 0.95–0.97" for the A-column match | a **visual** correspondence check, described in `notes/figure_provenance_2026-09-15.md:110`; no script, row or log anywhere | **UNTRACED**, and see §2a — the *claim* survives, the *number* does not |

## 2a. Two notes on how these verdicts must be read

**SUPERSEDED does not mean wrong.** The Fashion-MNIST "4 of 8" and the repo's "6 of 8" are the *same cell read at
two budgets* — 150 starts and 400 starts. A landed count without its start budget is not a quantity at all, so
neither number is a correction of the other. Quote the denominator every time, and put the budget in the ledger's
`NOT shown` column rather than in a footnote. (Raised by the GM lane, 2026-09-17.)

**UNTRACED here is subtler than "no source".** The glyph-correlation number *does* describe a check somebody
performed — the provenance note says the correspondence was checked by comparing displayed PCA glyphs after
accounting for polarity, scaling and crop, and that each of the first four source glyphs has its strongest match
in the same-numbered column. What is missing is that the check was **visual**, and a visual estimate written to two
decimal places reads as a measurement. So the fix is not only to drop the number: say the correspondence was
checked by eye. **That claim can survive; the number cannot.**

**A grep hazard on that number, recorded so the next person does not trip on it.** There are two unrelated
occurrences of 0.95–0.97 in this repository:

| where | what it is | status |
|---|---|---|
| `notes/figure_provenance_2026-09-15.md:110` | "normalized glyph correlations are approximately 0.95–0.97" — the visual figure-level check above | **UNTRACED**, do not repeat as a measurement |
| `notes/phase0_report.tex:275` | `cos_sim` near the optimum, "hyperparameter-tuned points achieve only ~0.95–0.97, not ~1.0" — a different quantity in a different study | **legitimately sourced**, leave alone |

Anyone purging the untraced number by grep will damage the phase0 report. (Collision found and verified at source
by the GM lane; both lines re-checked here before recording.)

## 2. Claims in the archive's briefs and notes

| claim | traces to | verdict |
|---|---|---|
| One-step identity `A_1 = A_0`, `B_1 = −η D_0 (A_0H)^T`, and the merged form carrying `A_0^T A_0` | `notes/ntk_vs_certificate_comparison.md`; measured model floors 2.89e-16 (LoRA-aware) vs 9.64e-01 (merged) at T=1 and 8.05e-16 vs 7.15e-01 at T=400, job **308859** | **TRACED** |
| Certificate 0/60 vs replay 19/60 on one release and chart, zero aliases | job **331384**, `results/exact_inversion/affine_two_routes_331384.jsonl`; 18 at worst-image ≤2.2e-15, 16 at ≤2.0e-15 | **TRACED** |
| The capacity line `k < m + r − N`, sharp to one unit of `k` | jobs **467914**, **469120**, **481079**, and on real MNIST **568095** / **574169** | **TRACED** |
| Wrong-release control recovers nothing | `wrong_release: true` rows: 0/200 (293350, 293438, 293516, 335732, 335738), 0/150 (542264), 0/300 (444942) | **TRACED** |
| Apples chart ladder (191/400 at 7/8; 2% → 169/400 at 4/8; 5/10/20% → 0/400; public PCA 0/400) | `figures/cifar_charts/table.md` oracle block, jobs **279934–279958** | **TRACED**, and fenced: the oracle cells are **not attacker-available** and must not be pooled |
| Conv-backbone table (keyboard 52/200 6-of-8, skyscraper 35/200 6-of-8, mushroom 23/200 5-of-8, Flowers-102 66/200 8-of-8, controls 0/200) | job **293438** and siblings; `notes/results_bundle_2026-09-06_index.md` Part 2 | **TRACED** |
| Record strength spans eleven orders with no effect on recovery in exact arithmetic; graded only under reduced-precision training | jobs **255095**, **255098**, **279342**, reproduction gate against **760909** / **764976** / **706721** | **TRACED** |
| Deep certificate under drift: members 1e-15 → 1e-5, non-members ~1e-1, width 16, N=4, truncated tests | `meeting_handoff/audit/trajspan_fig.json` + `build_audit_figures.py`; the generator has **never run here**, and `experiments/multilayer_cert/` is a different harness | **ARCHIVE-ONLY** |
| "Heavy drift leaves useful discrimination against nearby perturbations" | the same external figure | **ARCHIVE-ONLY and withdrawn by the archive itself** (the audit's correction 4: separation survived only against *fresh* inputs) |
| Two-boundaries plot with recovery/alias points at (4,30), (4,34)… | points taken from `lora_plan_rev11_2.tex`; the law itself is measured in `step11_capacity_467914.jsonl` and `step13_capacity_law_469120.jsonl` | figure **ARCHIVE-ONLY**, law **TRACED** |
| "A linear PCA chart beats a learned decoder chart" | withdrawn in the archive's own `READ_ME_FIRST` (unmatched comparator, one seed against three); repo agrees, matched controls **395771/395773/395774** | **WITHDRAWN** — do not reinstate |
| "Keyboards 7–8/8" | the supplied CNN grid reports **6/8**; repo rows `nc_cnn_293438` 52/200 6-of-8, `nc_mlp_293350` 200/200 7-of-8, `nc_mlp_ot_293516` 179/200 8-of-8 | **SUPERSEDED by cell** — the three are different cells and must not be merged into one number |
| "Counts refer to PCA projections, not raw photographs" | repo is stronger: attack SSIM 0.58 equals the chart ceiling 0.58 and sits **below** the 0.60 same-class control, so identification rests on image error (1e-14 against 0.25), never on a similarity score | **TRACED** |
| The text plan's Gaussian-dictionary example (`r=2`, `rank H = 4`, `rank B_T = 2`, `‖C‖_F = 5.33e-16`, `‖A_T − A_0‖_F = 0.1317`, 3,060 supports scanned, true support {1,4,9,13} recovered, best wrong-support residual 0.006853) | the plan's own §10; a deterministic NumPy computation described but not shipped as a file | **ARCHIVE-ONLY** — reproducible in minutes; worth re-running as the first line of any text pilot |

## 3. What the archive documents contain no evidence for

Recorded because their absence is itself a finding.

- **The September documents carry no job ids at all.** Provenance in them is by figure or cell name; job ids
  appear only in the August lineage and as a *requirement* in the replay brief's cluster protocol. Every number in
  §1 and §2 above had to be matched back to rows by cell parameters, not by citation.
- **The three proof notes contain no experiments.** The chart note says so twice ("No numerical experiments or
  numerical research results are claimed"); the multilayer note's "numerical margins" are analytic constants; the
  text plan performed exactly one computation. So none of the three can be cited as evidence for anything — only
  as proofs under stated hypotheses.
- **Three hand-computed expansions in the multilayer note** (coefficients `−3/8`, `1/16`, `−1/3`) are marked
  PROVED and have never been checked numerically here; the repo's T3 check validates a different instance.
- **0 of 37 scripts in the 14 September bundle exist in this repo by content**, now on a repo-wide hash scan of
  **43,832 `.py` files** (including the vendored `dataset_reconstruction/` tree), not just the obvious
  directories. Only `lora_exact_inversion.py` matches **by name** — 278 lines there against 583 here, and the
  hashes differ, so the repo file is the executed descendant. The consequence is the one that matters: the
  bundle's entire numerical apparatus (`trajspan.py`, `stackrank.py`, `rn_*.py`, `p3sim*.py`, `cap_check.py`,
  `pop_features.py` and the ten audit scripts) is archive-only, so **none of the †provisional numbers in
  `framework_rev10/11/12` or its audit has a reproducing script here**. Also confirmed by hash: `rn_phase.png` and
  `fig_certificate_phase.png` are the same file — one figure, not two results.

## 3a. The bundle: a citation rule, not a caveat

The finding in §3 — zero content matches for any of the 14 September bundle's 37 scripts across a repo-wide hash
of 43,832 Python files — is exhaustive, and it is **a statement about numbers, not about claims.** Letting it
travel undifferentiated would be wrong in both directions at once: it would either call proved statements
unsupported, or let a measured statement license unreproduced numbers. (Same distinction as two independent
derivations agreeing: evidence about the derivation is not a second proof of the result.)

**Several of the bundle's statements have been independently derived and measured here, by code that does not
descend from it.** For those, **cite our own rows, never the bundle** — *subject to the status column*, which is
not decoration:

> **A job id is a pointer to rows, and rows have a certification state.** Citing our own measurement beats citing
> the bundle **only where the measurement has passed audit**. Otherwise it trades an undisclosed *provenance* risk
> for an undisclosed *certification* risk, and the second is harder for a reader to catch, because a job id looks
> authoritative in a way a document citation does not. (Raised by the narrative lane, 2026-09-17, against the first
> version of this very table — which got two rows wrong in exactly that way.)

| statement | our own evidence | status | how it may be cited |
|---|---|---|---|
| The certificate annihilates every recorded representation — no recipe, no labels, no seed | job **701679**, ledger row A3 (`results/CLAIMS_LEDGER.md:68`) | **SETTLED**, two passes, register derived + read-rows | freely, with A3's own scope (`q = N`; SGD-class; `B_0 = 0`; non-shared module) |
| Quotient-sensing form `Π A_T = c_T Π A_0` with `c_T = 1.000000000000` derivable rather than fitted | job **675031** gate; proof at `notes/w1_certificate_proofs.tex:115` | **proved + measured, but it has NO ledger row**, so it is uncertified by the ledger's own process | cite the proof first and the gate as confirmation; do not present the job id as an audited measurement |
| The capacity line `k < m + r − N`, sharp to one unit of `k` | jobs **467914, 481079, 593146**, ledger row C4 (`:85`) | **SETTLED** — *for the reduced channel* | **never bare.** C4's `NOT shown` is explicit: for the **full factor pair** the line is a *candidate, not a result*. The figure `capacity_law.png` is job 469120 |
| The three hand-computed coefficients of the multilayer note | job **353169** (`results/archive_checks/hand_coefficients_353169.jsonl`) | run 2026-09-17, **one reader**, no second pass | cite with the margins (3353× / 35014× / **2.0×**), never as three equal passes |
| ~~Certificate exact at depth over the training span, and the rank law~~ | — | **WITHDRAWN FROM THIS TABLE, 2026-09-17** | see below |

**Why that last row was pulled, since it was wrong in two distinct ways and it is the one that would have reached
Gal.** (Both faults verified at source here before removal.)

1. **The rank law as I cited it was refuted the same day.** `theory/T2` stated `rank C = r − N'`; the de-rigged
   check `T2_rank_C_matches_AS_WRITTEN_r_minus_Nprime` records **`passed: false` at job 351745**, while
   `T2_rank_C_matches_corrected_min_law` (`max(0, min(r−N', n_l−N'))`) passes. So the law may be cited **only in
   its corrected form and only to 351745** — the job that tests it as written is the job that fails it.
2. **"Exact at depth" is uncertified and it is Gal's own main question.** Ledger M1 carries register
   `read-prose (GM)` and status **UNAUDITED**, and the question index records **zero passes for M1–M3** while M4/M5
   have PASS #1. Job 688036's own layer-3 row reads `N' = 15, rank_B = 4, rank_C = 16, expect 5, rho 0.078` — the
   contamination regime, where the hypothesis fails. Promoting this to cite-our-job-id status was premature by a
   full audit pass, on precisely the claim it would be worst to get ahead of.

*One correction back to the lane that caught this:* job **692603 does carry rank-law rows** —
`expect_rank_C` appears in 200 of its 216 — so the withdrawn row was wrong for the certification reason, not for
that one.

**And the not-a-finding is now a finding, with a refinement that makes it larger than the account I was given.**
Registers: `read-rows` for the counts, `read-function` for the gate.

- **The split is real and is not an artefact of my reading of `n_l`.** Of 125 B3-holding rows, **116 agree with the
  corrected rank law and 9 disagree**. The 9 carry adapter norm `beta` from **1.28e+21 to 3.84e+302** (three are
  `nan`); the 116 top out at **572.9**. Eighteen orders of magnitude with **nothing in between** — so any threshold
  in that range gives the same split, which is what "untuned" means when it is measured rather than asserted.
  All 9 are flagged `diverged: False`.
- **The nine are four exploded configs, not nine scattered rows**: `(seed 0, lr 1.0, T 8)`, `(1, 1.0, 8)`,
  `(1, 3.0, 4)`, `(2, 3.0, 4)`. All 16 rows of those four configs read `diverged: False`.
- **The explanation offered — that the detector cannot fire at layer 0, because layer 0's inputs are the data so
  its drift is identically zero — is correct and covers exactly one of the nine.** That row has `delta` exactly
  `0.000e+00` with `beta` 1.28e21. **The other eight sit at layers 1–3, where drift IS measured and is
  astronomical**: `delta` = 4.8e+43, 9.5e+51, 1.2e+20, 3.4e+86, 4.5e+103, 8.0e+29, 5.2e+129, and one row at
  **`inf`** — all still flagged `diverged: False`.
- **So the hole is in the gate, not only in layer 0's geometry.** `experiments/multilayer_cert/survival.py:33-43`
  marks a config diverged iff some entry of the representations or factors is non-finite **or** some
  representation entry exceeds `1e100`, and then returns stub rows (`layer`, `diverged` and the config only — no
  `delta`, no `rho_full`, no `B3_holds`). It fired on four *other* configs, producing the 16 stub rows already
  known. **It did not fire on these four, one of which reports infinite drift** — which the guard as written
  should have prevented, since an infinite Frobenius norm needs entries far above `1e100`.

**Two things then settled by reading the code rather than re-running it** (register `read-function`):

- **The proposed cause — that the gate inspects a different collection than the drift — is REFUTED.**
  `survival.py:49-51` computes `D = [rt[l] - H0 for rt in reps]` and `delta = max(‖d‖/‖H0‖)` over **the same
  `reps` object** the gate iterates at `:35-38`. There is no subset mismatch, and tuning the threshold would not
  have been hiding this particular bug because this particular bug is not there.
- **The real structural gap: `reps` never contains the final state.** `common.py:61-64` appends the
  representations at the *top* of each step, so `reps` holds `t = 0 … T−1` — the inputs **before** each of the `T`
  updates — while the returned `A, B` are the parameters **after** update `T`. So the gate and the drift both look
  only at pre-final steps, and **`beta`, computed from the returned `A[l], B[l]` at `:57`, is the only quantity in
  the row that sees the final update at all.** That is why `beta` separates the two populations cleanly while the
  drift-based flag misses them, and it is exactly the shape of the one layer-0 row: `delta` identically `0.000e+00`
  with `beta` 1.28e+21 — an explosion living entirely in the parameters.

**The absolute-versus-relative mismatch, and how far arithmetic alone settles it.** The guard is **entrywise and
absolute** (`h.abs().max() > 1e100`); the reported drift is **relative** (`‖d‖/‖H0‖`). If the gate did not fire
then every entry is `≤ 1e100`, so for a `30×3` block `‖d‖_F ≤ √90 · 2e100 = 1.9e101`, and therefore
**`‖H0‖ ≤ 1.9e101 / delta`** — an *upper* bound on the base norm, which is the direction that decides each row:

| reported `delta` | requires `‖H0‖ ≤` | verdict |
|---|---|---|
| 1.2e+20, 8.0e+29, 4.8e+43, 9.5e+51, 3.4e+86 | 1.6e+81 … 5.6e+14 | **consistent — no bug.** A base norm of order 10 already satisfies these, so relative drift can reach ~1e100 with every entry legitimately under the threshold |
| 4.5e+103 | **4.2e-03** | possible, but needs a base norm below ~1e-3 |
| 5.2e+129 | **3.6e-29** | **implausible** for any base representation of a GELU MLP at these dimensions |
| **inf** | **0** exactly | **impossible** unless `‖H0‖` underflowed to zero, in which case the row is a division artefact and never was a drift measurement |

**So five of the eight rows are the gate behaving exactly as written**, and the defect there is *calibration*, not
logic: the threshold is orders too permissive relative to the quantity the row reports. The fix is to **gate on the
relative drift the row itself reports**, not on entry magnitude. **Two rows still demand an explanation** — the
`5.2e+129` and the `inf` — and both turn on the same unrecorded field.

**What the instrumentation must capture**, then, is narrower than "max entry per layer and step": it is **`‖H0‖`**,
checked for *underflow* specifically and not merely for smallness, on those two rows. And the representations
**after** the final update, which the harness never computes. I have not run it and have not touched the harness.

**The one assumption that could overturn the two verdicts above is measured by that same field at no extra cost.**
`‖H0‖ ~ 10` is inferred from the architecture here, not measured. If the base representations are in fact tiny at
some layer, both the `4.5e+103` and `5.2e+129` verdicts move. `‖H0‖` per layer is already the field being
requested, so this needs **no separate job** — and nothing here should be run locally to get it.

*(Arithmetic by the narrative lane, recomputed here. Their table states the bound as `‖H0‖ ≥`; it is `≤`. The five
easy rows are satisfied under either reading, which is why the slip does not affect them — it bites only at the two
extremes, where it turns "constrains, still possible" into "requires a base norm of 3.6e-29".)*

**What this changes.** `116` is a clean count **once the adapter-norm guard is applied**, and the guard is now
measured rather than asserted. But the divergence detector admits configs with infinite drift at depth, so
`diverged: False` in this file does not mean "did not diverge" and must not be used as a filter on its own.

**Reserve the dagger for what remains the bundle's alone**: its basin fractions, exact-inversion residuals and
phase-diagram percentages, and the deep-certificate / stacked-Jacobian numbers whose only generators are
`trajspan.py`, `stackrank.py`, `rn_*.py` and the audit scripts. Those have no reproducing code here and stay
†provisional until they do.

**The boundary, which is part of the rule and not a note beside it.** This rule says *cite our rows for the
statements we measured*. **It says nothing about whether the bundle's other numbers are right.** Unreproduced here
is a **provenance status, not a verdict**: §3 marks those numbers ARCHIVE-ONLY, which means no reproducing code
exists in this repository — **not refuted, not doubted, not impeached**. Anyone who reads this rule as evidence
against the bundle has misread it. (Stated inside the block deliberately: a boundary kept in a neighbouring
paragraph is separated from the rule the first time someone quotes the rule alone, which is exactly how the
misreading would happen.)

**Why this is better than a caveat.** A claim sourced to our own measurement is strictly stronger than the same
claim sourced to an unreproduced document, and it carries a job id. So the rule costs nothing and closes the
exposure instead of disclosing it: for anything we have measured, the provenance question simply does not arise.
The consequence for anything supervisor-facing is that a sentence currently leaning on the bundle gets
**upgraded** when it is re-sourced, not hedged — a caveat costs credibility, a substitution gains it.
*(Raised by the approver lane, 2026-09-17; routed to the narrative lane because it changes how the bundle may be
cited in Gal-facing material.)*

## 3b. The mirror image: what has NO repo counterpart at all

§3a lists what to re-source. This lists the opposite and more dangerous set: archive statements that have
**neither a row to cite nor an independent derivation here** — nothing in this repository bears on them. They are
the ones most likely to be quoted straight from the document, because the document is the only source.

| archive statement | why nothing here bears on it | what would give it a counterpart |
|---|---|---|
| **The short-time trained-family theorems**: certificate error `O(τ)` for a trained two-layer softplus MLP inside `τ ≤ min{ρ/(2g), γ/(4K)}`, and the ReLU width version (`multilayer_lora_theory_source.tex` Prop 4.2, Thm 5.3, Thm 5.4) | The repo has **no theorem converting trajectory bounds into a statement about an actual trained net**, and its measured regime (drift to 923%, `T ≤ 8`, lr to 1) lies far outside any such window — the repo's exactness result does not need one | A cell inside the window: small `τ`, measured `γ` and `K`, checking `e(τ)` against the bound |
| **The quadratic residual family** achieving `O(τ²)` by the range condition `col(∇_B L(θ_0)) ⊆ U` (Thm 8.2, Cor 8.3) | The repo's T3 proves `O(ε)` is sharp *generically* and has no analogue of a designed family where the first-order forcing vanishes. Ruled C1: same mechanism as T3.2, different vocabulary — but the *family* is untested here | The C1 merge, plus one FP64 cell of the aligned identity-residual construction |
| **The polynomial / quadratic span-separating families** (chart note Thm 2.7, Thm 2.20; multilayer Thm 6.1) — the only settings where global uniqueness of the image is proved | Proved only where the adapted layer *itself receives* monomial features. Every repo chart is PCA, conv-AE, β-VAE or oracle; none is an expansion family | A chart whose features are the monomials, on a layer that receives them |
| **The distributional laws**: wrong-affine-chart `Beta((p−k)/2,(b−p)/2)`, the Wishart conditioning rate, the fixed-candidate null law, the affine bias–variance MSE (chart note Thm 2.11, 2.14, 2.18, 2.19) | Never measured here in any form. They are laws of the *kernel experiment* and, by the note's own Correction A.4, do not survive conditioning on a valid release | A synthetic sampler over seeds at fixed `H`, comparing empirical residual distributions to the stated laws |
| **The cross-layer min–max rank theorem** over allowable covector spaces, and the ideal→trained transfer `σ_min ≥ μ − Δ_J` (Thm 10.2, Prop 10.3) | The repo's own stacking law (T5.2) was **refuted as stated** on 2026-09-17; the archive's strictly more general form is untested here, and Prop 10.3 is the conditional route that would upgrade T5.4 | A stacked-Jacobian measurement with the per-layer `E_l` computed, not assumed |
| **Everything in the text plan**: the endpoint certificate in text, finite-codebook separation, the sparse known-initialisation route, the distinct-target excitation family (`Text_Certificate_Chart_Research_Plan.md` Props 1–3) | **No text experiment has ever run in this repository.** Its own §10 Gaussian-dictionary example is the only computation behind any of it | Stage 1 of its programme — the theorem instrumentation cell |
| **The shared-concept theorems and the architecture discussion** (multilayer Sec 12, Sec 13.2–13.4: homogeneity obstruction, width, CNN) | Nothing here measures shared-concept charts at the theorem's level, and the CNN paragraph is discursive in the source itself | The E4a shared-concept arm exists but measures fidelity, not these rank statements |
| **The deep-certificate-under-drift numbers** (members 1e-15 → 1e-5, non-members ~1e-1, width 16, N=4) and the stacked-rank numbers, from `trajspan.py` / `stackrank.py` | The generators are archive-only; the repo's multilayer harness is a **different** experiment, so these specific numbers have no counterpart even though the *claim* of exactness at depth does | Re-run under our harness, or drop the numbers and cite 688036/692603 for the claim |

**How to use it.** These are not marked doubtful — most are proofs, and a proof does not need a row. The point is
narrower: if a supervisor-facing sentence rests on one of them, it rests on the document **alone**, so it must be
stated as a proved statement under its hypotheses and never as something this project has observed. Two of the
eight are cheap to convert (the quadratic family and the distributional laws), one is already the agreed next
step (text), and one is a merge that was ruled tonight.

*(Compiled 2026-09-17 from the reconciliation tables of the two theory digests, cross-checked against `theory/`,
`experiments/multilayer_cert/` and the chart/text notes. **One reader** — it has not been independently checked,
and a missing counterpart is the kind of claim that is falsified by one grep, so treat an entry as refutable.)*

## 4. How to use this file

Before quoting a number from any archived PDF or figure: find it here. If it is **TRACED**, cite the job id and
the row, not the PDF. If it is **SUPERSEDED**, quote the repo number with its start count. If it is
**ARCHIVE-ONLY**, say so in the sentence that uses it, or do not use it. If it is **UNTRACED**, it does not go
out. And before citing the 14 September bundle for anything at all, read §3a: where we have measured the statement
ourselves, cite our rows and their job id rather than the bundle. New claims go to `results/CLAIMS_LEDGER.md` and the track's `RESULT(S).md`, which remain the claims record.
