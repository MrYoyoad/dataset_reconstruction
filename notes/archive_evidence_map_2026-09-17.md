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
- **0 of 37 scripts in the 14 September bundle exist in this repo by content.** Only
  `lora_exact_inversion.py` matches by name (278 lines there against 583 here — the repo file is the executed
  descendant).

## 4. How to use this file

Before quoting a number from any archived PDF or figure: find it here. If it is **TRACED**, cite the job id and
the row, not the PDF. If it is **SUPERSEDED**, quote the repo number with its start count. If it is
**ARCHIVE-ONLY**, say so in the sentence that uses it, or do not use it. If it is **UNTRACED**, it does not go
out. New claims go to `results/CLAIMS_LEDGER.md` and the track's `RESULT(S).md`, which remain the claims record.
