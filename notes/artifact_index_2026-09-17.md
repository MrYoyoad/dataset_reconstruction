# Artifact index — `Thesis_files_2026-09-17.zip`, and the coverage record of its review

The archive is **92 top-level files plus seven nested bundles (526 files unpacked, ~99 MB)**. The zip itself is
**not committed** (`.gitignore` excludes `*.zip`); it sits at the repository root as
`Thesis_files_2026-09-17.zip`, and a read-only working copy of the unpacked tree was used at
`/tmp/yoado_thesis_archive_2026-09-17/` during the review. This file is the map from that archive into the
repository: what was imported and where, what was already here, and what was deliberately left outside git.

Related records: `notes/research_overview_2026-09-17.md` (current position) · `notes/technical_record_2026-09-17.md`
(definitions, claims, assumptions) · `notes/archive_evidence_map_2026-09-17.md` (claims → jobs/code) ·
`notes/corrections_from_archive_2026-09-17.md` (corrections and supersession) ·
`notes/meeting_2026-09-15_decisions_and_backlog.md` (decisions and backlog) · `results/00_map.md` (the repo's own
orientation map, still the best entry point to code and releases) · `results/CLAIMS_LEDGER.md` (audited claims).

---

## 1. What was imported, and where it went

| repo path | from the archive | why it was kept |
|---|---|---|
| `notes/gal_2026-09/The_LoRA_certificate.{tex,pdf}` | `01_Proofs_and_theory/` (2026-09-10) | The Gal-facing certificate note: A1–A5, Claims 1–3, the Theorem, the NTK/factor-fit remarks. The `.tex` is the editable source. |
| `notes/gal_2026-09/multilayer_lora_theory{_source.tex,.pdf}` | `01_Proofs_and_theory/` (2026-09-07) | 56 pp of multilayer proofs: exact-over-the-training-span, the truncated bound, the short-time trained-family theorems, the cross-layer rank theorem. |
| `notes/gal_2026-09/chart_inversion_theory.{tex,pdf}` | `01_Proofs_and_theory/{chart_inversion_theory.tex, charts.pdf}` (2026-09-08) | 65 pp on charts: local vs global identifiability, the `k < p` theorem, twelve counterexamples, Appendix A's corrections to our own premises. The PDF was renamed from `charts.pdf` to match its source. |
| `notes/gal_2026-09/Text_Certificate_Chart_Research_Plan.md` | `02_Summaries…/` (2026-09-15) | The text-extension plan: Props 1–3, the excitation family, the rank-saturation obstruction, the staged programme. |
| `notes/gal_2026-09/LoRA_meeting_audit_2026-09-14.md` | `02_Summaries…/LoRA_meeting_audit.md` | The pre-meeting audit: the corrections it makes to our own circulating claims, the literature pass, figure roles. |
| `notes/gal_2026-09/B_factor_representer_idea.md` | `02_Summaries…/` (2026-09-08) | The joint factor-fit idea kept deliberately separate from the proof note, with its own limits. |
| `notes/gal_2026-09/gal_ntk_cluster_brief.md` | `02_Summaries…/` (2026-09-10) | The four-objective one-step diagnostic and the sanity counterexample showing the free-coefficient symmetry is not a replay symmetry. |
| `figures/gal_2026-09/*.png` (10) + `figure_pack_2026-09-15.pdf` + `figures_for_gal_2026-09-09.pdf` + `vector/*.pdf` (8) | `03_Figures/` and `figure_pack_editable/vector_figures/` | The figures actually shown on 2026-09-15, in raster and vector form, plus the raster set several panels were cut from. |
| `scripts/figpack_2026_09_15/` (9 `.tex`, `style.tex`, `build.py`, `fill_results_template.py`, `validate.py`, `assets/` ×11, `compositors/` ×6) | `figure_pack_editable/source/`, `LoRA_visual_figures/source/`, `meeting_handoff/audit/` | The editable sources for every figure, the results-page filler, the synthetic algebra check, and the six compositor scripts that record how each published panel was cut (crop boxes, strides, polarity). |
| `notes/figure_provenance_2026-09-15.md` | `LoRA_visual_figures/SOURCES.md` + `ground_truth_provenance.md` + `meeting_handoff/audit/figure_roles.md` | One deduplicated provenance record, with the three corrections found when panels were traced to rows (see §4). |
| `notes/results_bundle_2026-09-06_index.md` | `results_2026-09-06/…/INDEX.md` + `READ_ME_FIRST_deck_v6_update.md` | The only copy of the experiment index and the slide rules, including four claims withdrawn the same night. |
| ~~`experiments/cifar/cifar_certificate_supplied_replica.py`~~ | `results_2026-09-06/…/scripts/cifar_certificate.py` | **NOT imported — it was already here, and I got this wrong first time.** The supplied replica is byte-identical (`52ae16f5…`) to **`cifar_certificate.py` at the repository root**, which is tracked. I imported a second copy and described it as "the only copy of the code behind *the replica does not recover its images*"; both halves were false. The duplicate has been removed. **Cite the root file.** |

**Build caveat recorded at import:** `scripts/figpack_2026_09_15/build.py` needs pdflatex and
`fill_results_template.py` shells out to `kpsewhich`; neither exists on WEXAC (CLAUDE.md, "Markdown → PDF on
WEXAC"). The pack rebuilds on Overleaf or a local TeX; the prebuilt vector PDFs and PNGs are committed alongside
so nothing has to be rebuilt to be read. `compositors/adopt_figure.py` and `build_audit_figures.py` carry
hard-coded `/workspace/scratch/...` paths and need repathing before they run.

## 2. Already in the repository — not re-imported

- **149 of the 526 unpacked files are byte-identical to files already here.** `results_2026-09-06` accounts for
  114 (rows under `results/cifar_newclass/` and `results/record_strength/`, figures under `figures/cifar_charts/`,
  `figures/cifar_study/`, `figures/ntk_vs_cert/`, `figures/record_strength/`, `experiments/cifar/charts/*/result.json`,
  9 of 12 scripts, and `EQUIVALENCE_linearised_vs_certificate.md` == `notes/ntk_vs_certificate_comparison.md`).
  `thesis_review_2026-08-29` accounts for 34.
- `01_Proofs_and_theory/identifiability_rank_bound(5).pdf` is byte-identical to `notes/identifiability_rank_bound.pdf`.
  **But see §4.3:** that file is the *oldest* content of the five revisions, not the newest.
- `Thesis_meeting_summary_2026-09-15.md` is byte-identical to `notes/meeting_summary_2026-09-15.md`, which a
  sibling lane committed on 2026-09-17 (8b42fc3). The archive copy was **not** re-imported.
- `results_2026-09-06/…/scripts/cifar_certificate.py` is byte-identical to **`cifar_certificate.py` at the
  repository root**, which is tracked. **This one was missed on the first pass and is worth the confession:** the
  duplicate scan behind §2 walked `notes figures results scripts docs theory experiments papers` and **never looked
  at loose files in the repository root**, so a tracked file sitting there was invisible to it and I imported a
  second copy. Found only because a filesystem-wide search launched at the very start of the session finished hours
  later and listed the root. **Any "is this already here?" scan must include the root directory** — which is
  precisely where an un-filed script ends up.

## 3. Deliberately not imported, with the reason

| left out | reason |
|---|---|
| The zip itself, and all seven nested bundles as bundles | `.gitignore` excludes `*.zip`; the useful contents are imported file-by-file above. The zip stays at the repo root as the authoritative original; **it is the only copy of anything not listed here.** |
| 24 superseded plan/brief/walkthrough PDFs (`lora_plan{,_rev4,_rev5,_rev6,_rev9,_rev11,_rev11(1),_rev11(2)}`, `LoRA_meeting_{board_guide,board_guide_v2,walkthrough_v8_2,walkthrough_FINAL,FINAL(1),FINAL(2)}`, `lora certificate replay brief` ×2, `lora thesis master record` ×2, `LoRA_gate_meeting_short_v3`, `meeting_told_slowly`, `lora_brief_v2`, `LoRA_reconstruction_brief`, `LoRA_meeting_personal_companion`, `Gal.pdf`, `lora_thesis_research_plan`, `thesis_experiment_guide_revised_2026-08-29` ×2) | Their surviving content is the certificate note, the audit and the meeting record, all imported. The development history is recorded in `notes/corrections_from_archive_2026-09-17.md` §2 and in the chronology below rather than as 27 MB of near-duplicate PDFs. |
| `identifiability_rank_bound{,(1),(2),(3),(4)}.pdf` | Five revisions of a retired note; `(4)` is the latest content. Its status table is summarised in §4.3; the repo holds the oldest build and an unapplied patch (`notes/identifiability_feasibility_revision.tex`). |
| `04_Presentations/*.pptx` (4 decks, 14.9 MB) | `.gitignore` excludes `*.pptx` by standing convention. v17 is an ancestor of the repo's v20/v21 spec (`figures/deck_v20_spec/`), which is the reproducible source. |
| `07_Research_report_exports/*.json` (4, 2.4 MB) | AI-generated research reports, advisory not primary evidence. Their substance was already in the repo before they were written; the one idea that never transferred (a staged risk ladder P1–P7) is noted in §4.4. |
| `06_Project_supporting_images/*` (13) | Eleven are phone screenshots of figures or LaTeX pages that exist here in better form; one is a file icon. The one with independent content — a rendered "write on the board" notation table — is transcribed in `notes/technical_record_2026-09-17.md` §1. |
| The 35 older `cifar_newclass` PNGs, `tables/cifar_charts_table.md`, `cifar/RESULT.md` and 2 scripts inside `results_2026-09-06` | Superseded by the repo's replotted figures and longer write-ups; keeping both would put two captions on one cell. |
| `letters_raw_source_panel.png`, `letters_original.png`, `fig_chart_dependence.png` and its crops, `figures_for_gal(1).pdf` | Lossy re-rasters or byte-duplicates of figures the repo already generates from source. |
| External scripts in `lora_thesis_bundle_14sep2026/03_scripts/` (37 files) | **None exists in the repo by content**; they are the provenance of externally-reported deep-certificate and stacked-Jacobian numbers that have never been run here. Left in the zip; if that provenance is ever needed, import under a directory fenced as *external, never run here, provisional*. |

## 4. What the review found that changes something

1. **The Fashion-MNIST "4 of 8" is superseded.** It is the 150-start cell (job 473802,
   `results/cifar_newclass/sharp2_473802.jsonl`: `landed 9/150`, `images_found 4`, landed columns 2, 3, 6, 7 —
   exactly the panel's). The repo's current figure for the same cell is a 400-start re-run: **23/400 landed,
   6 of 8 images** (job 556643). Quote 6/8 with its start count.
2. **The MNIST "private-built reference" strip pairs each digit with a byte-identical copy of itself** (three
   filenames, one sha256). That is consistent with an oracle chart's 5.7e-14 error, but the panel shows a reader no
   cue that the chart is **not attacker-available**. The repo's later figure says so on its face
   (`figures/exact_inversion/chart_dependence_k17.png`); use that one.
3. **"Glyph correlations ≈ 0.95–0.97"** appears in the provenance prose with no script, row or log behind it
   anywhere. **UNTRACED** — do not repeat it.
4. **The identifiability-note numbering is not chronological.** By PDF creation metadata the content order is
   `(5)` → base → `(1)` → `(2)`=`(3)` → `(4)`, so **`(4)` is the latest and `(5)` the oldest** — and `(5)` is the
   one the repo holds. The repo's shipped `notes/identifiability_rank_bound.pdf` therefore still contains claims
   that `(1)` explicitly retracted (the capacity law `N_max ≈ ρ_eff·d/k`; "SDS restricts the search to a
   k-manifold"; "known base weights make the gates known"; "a decoder raises observational rank"; "the final
   adapter is stacked checkpoints"), and `(2)` additionally declares **"released adapter = compressed ΔW" false**.
   `notes/identifiability_feasibility_revision.tex` is the unapplied patch. Anyone citing that PDF must read this
   paragraph first.
5. **A sentence in `CLAUDE.md` inherits a retracted claim.** The Gradient-Bridge direction is described there as
   "LoRA adapters … are structured, compressed recordings of cumulative training gradients", which is the sentence
   `(2)`/`(4)` mark false and `lora_plan_rev11` retires. Annotated in place rather than deleted, because the
   section is a record of a historical direction.
6. **The repo's multilayer track and the archive's multilayer note are independent derivations.** `theory/T1…T6`
   cite neither the note nor its lemmas; the two agree on every shared statement, which is a consistency signal
   worth stating, not a joint proof. Conflicts are listed in `notes/corrections_from_archive_2026-09-17.md` §3.

## 5. Coverage record — what was actually read

"Read" means the text was read in this session (by me or by a sub-agent whose report is the basis for the entry);
"inspected" for images means the image was viewed. Nothing below is claimed as read on the strength of a filename.

| material | count | coverage |
|---|---|---|
| `01_Proofs_and_theory` — `The_LoRA_certificate.tex` | 1 | **read in full** (main session) |
| `01_Proofs_and_theory` — `chart_inversion_theory.tex` (2393 lines) + `charts.pdf` (65 pp) | 2 | **read in full** (sub-agent digest, 294-line report; every theorem, counterexample and correction catalogued with line numbers) |
| `01_Proofs_and_theory` — `multilayer_lora_theory_source.tex` (1743 lines) + PDF (56 pp) | 2 | **read in full** (sub-agent digest, 574-line report, reconciled against `theory/T1…T6` and the multilayer jsonl rows) |
| `01_Proofs_and_theory` — the five `identifiability_rank_bound` revisions, `lora_leakage_note_v3`, `LoRA_Training_Data_Observability_Math_Note`, `LoRA_bridge_reconstruction_report_math_typeset` | 8 | **read** (sub-agent; text extracted and diffed revision to revision) |
| `02_Summaries…` — `Thesis_meeting_summary_2026-09-15.md`, `Text_Certificate_Chart_Research_Plan.md`, `LoRA_meeting_audit.md`, `gal_ntk_cluster_brief.md`, `B_factor_representer_idea.md` | 5 | **read in full** (main session) |
| `02_Summaries…` — the 28 PDFs (plans rev3–rev11.1, board guides, walkthroughs, briefs, replay brief, master record, `Gal.pdf`, `meeting_told_slowly`, companion, experiment guide) + `mac_handoff_brief.md` | 29 | **read** (sub-agent; per-document cards, a nine-phase chronology, and concrete text diffs of the four near-duplicate pairs) |
| `03_Figures` — 10 PNGs | 10 | **inspected** (viewed as images) and traced to sources |
| `03_Figures` — `figure_pack.pdf`, `figures_for_gal.pdf` (image-only; rendered to inspect), `LoRA_figures_only.pdf`, `motorcycle_results_template.pdf` | 4 | **inspected** |
| `05_Original_bundles/figure_pack_editable`, `LoRA_visual_figures` | 73 | **read** (`SOURCES.md`, `ground_truth_provenance.md`, `README.md`, every `.tex`, `build.py`, `validate.py`, `fill_results_template.py`); assets inspected |
| `05_Original_bundles/LoRA_meeting_audit_and_figures/meeting_handoff` | 95 | audit (25) and proofs (4) **read**; figures (60) **inspected**, with the compositor scripts read to recover crop provenance |
| `05_Original_bundles/results_2026-09-06` | 155 | `INDEX.md`, `READ_ME_FIRST`, `EQUIVALENCE`, `cifar/RESULT.md`, `record_strength/RESULT.md` **read in full**; the 50 rows and 3 tables **read as data** (schema + sha-compared against repo rows); 85 figures **inspected or sha-matched** |
| `05_Original_bundles/lora_thesis_bundle_14sep2026` | 63 | `README.md` **read**; `framework_rev10` vs `rev11` **text-diffed**; 7 figures **inspected**; 37 scripts **inventoried and sha-compared only — not read line by line** |
| `05_Original_bundles/thesis_review_2026-08-29` | 44 | `00_START_HERE.md` and the overview **read**; the rest **inventoried and sha-compared** (34 of 44 identical to repo files) |
| `06_Project_supporting_images` | 13 | **inspected** (all 13 viewed) |
| `07_Research_report_exports` | 4 | **read** (JSON parsed, report bodies extracted and summarised; they are AI-generated reports, treated as advisory) |
| `04_Presentations` (4 pptx) | 4 | **slide titles extracted for v17 only**; the other three **inventoried, not opened**. v17 is an ancestor of the repo's v20/v21 spec (35 → 37 slides). |

**Deduplicated, not deleted.** Six byte-identical pairs exist among the 92 top-level files
(`identifiability_rank_bound(2)`=`(3)`, `LoRA_meeting_board_guide`=`(1)`, `walkthrough_FINAL(1)`=`(2)`,
`lora_plan_rev11`=`(2)`, `thesis_experiment_guide_revised_2026-08-29`=`_1`, `figures_for_gal`=`(1)`), and about
forty more across the nested bundles. Two pairs are byte-different but **text-identical** re-exports (`lora
certificate replay brief` and `lora thesis master record` against their `(1)` copies). Three pairs are genuine
revisions and were diffed: `walkthrough_FINAL` vs `FINAL(1)` (a systematic de-numbering — the minute quoting two
predicted failures with their numbers becomes a structural caveat), `rev11` vs `rev11(1)` (Rev 11.1's twenty
downgrades, including sufficiency → a conditional dichotomy and "identifiability" → "separation at tested
points"), and `board_guide` vs `_v2`.

**Could not be read.** Nothing was unreadable. Two mechanical obstacles were worked around and are recorded so the
next session does not rediscover them: `pdftotext` is not installed (all PDF text came from `pypdf`), and
`figures_for_gal.pdf` is image-only, so its pages had to be rendered before they could be inspected. One process
hazard: files unpacked into a `/tmp` scratch directory kept their 2026-09-07 archive timestamps and were removed
by an age-based cleaner mid-review; re-extracting with reset timestamps fixed it.

## 6. The development chronology, in one table

Established from content and explicit corrections, not from dates or filenames.

| phase | what changed | where it is recorded now |
|---|---|---|
| Aug 2026 — the retired lineage | gate matrix, the whitened-Jacobian "leakage ruler" (`r_J`, `q_eff`, `d²`), the gradient bridge as the spine | `notes/thesis_scientific_summary.md`; retired on 2026-08-31 after the "correlations, no tool" meeting |
| 2026-08-20 → 08-30 | the identifiability/observability note and its four retractions | §4.3 above; `notes/identifiability_rank_bound.pdf` (oldest build) + the unapplied patch |
| 2026-08-31 → 09-03 | the certificate lineage begins; quotient sensing; Rev 11 retires the adapted-gradient/KKT and spectral routes and installs replay + the capacity line `k < m + r − N` | `notes/exact_channel_rev10.tex`, `results/CLAIMS_LEDGER.md` |
| 2026-09-03 (Rev 11.1) | twenty downgrades: sufficiency becomes a conditional dichotomy with uncertified witnesses; "identifiability" becomes "separation at tested points" | `notes/corrections_from_archive_2026-09-17.md` §2 |
| 2026-09-06/07 | the CIFAR study, record strength, and the route equivalence; four claims withdrawn in one night | `notes/results_bundle_2026-09-06_index.md`, `notes/ntk_vs_certificate_comparison.md` |
| 2026-09-07 → 09-10 | the three proof notes reach their current form (multilayer 09-07, charts 09-08, certificate 09-10) | `notes/gal_2026-09/` |
| 2026-09-14 | the pre-meeting audit rewrites the story chart-first and corrects seven substantive items | `notes/gal_2026-09/LoRA_meeting_audit_2026-09-14.md` |
| 2026-09-15 | the meeting: direction agreed, four next steps, the scoped NTK conclusion | `notes/meeting_summary_2026-09-15.md` + `notes/meeting_2026-09-15_decisions_and_backlog.md` |
