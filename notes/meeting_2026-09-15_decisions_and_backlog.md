# Supervision meeting 2026-09-15 — decisions and the experiment backlog

**Source of record:** `notes/meeting_summary_2026-09-15.md` — the verbatim summary written 2026-09-15 from Yoad's
own account of the meeting with Gal Vardi and research colleagues, committed by a sibling lane in 8b42fc3 and
byte-identical to the archive copy (sha256 1efb0119…). That file is the primary record; this page is its
companion.
This page separates (A) what was *reported* about the meeting, (B) what was *agreed*, (C) what was *discussed as a
possible direction*, and (D) our own deductions and the prioritised backlog that follows from them. Nothing in (D)
was decided at the meeting.

Material shown at the meeting: the figure pack `figures/gal_2026-09/figure_pack_2026-09-15.pdf` (eight slides
`01_one_layer` … `08_local_families`, same directory as PNGs) and the certificate note
`notes/gal_2026-09/The_LoRA_certificate.pdf`. The multilayer and chart notes
(`notes/gal_2026-09/multilayer_lora_theory.pdf`, `notes/gal_2026-09/chart_inversion_theory.pdf`) were in the bundle but the summary does not say they
were discussed page by page.

---

## A. Reported outcome (from the summary; Yoad's account)

- **Overall:** the meeting went very well. The participants were impressed by the figures, by the certificate
  equation `C H = 0` with `C = P_{row(B_T)^⊥} A_T`, and by the chart idea. The first two (broadly, first three or
  four) figures already carried substantial material and most of the discussion was about understanding them; they
  asked how the equation is derived and what its geometric intuition is, and called it a good equation. They knew
  LoRA; this reconstruction approach was unfamiliar to them.
- **NTK reconstruction, scoped conclusion:** they first asked about the difficulties with existing LoRA
  reconstruction and NTK-based approaches. By the end of the discussion they accepted the argument that *the NTK
  reconstruction approach discussed* would not work for *the LoRA setting considered*. Recorded as the meeting
  conclusion, with its technical scope: it concerns the specific merged-coordinate / free-coefficient one-step
  formulation and the head-adapter setting (see the "One-step NTK and the free-coefficient fit" slide,
  `figures/gal_2026-09/04_ntk_partial_information.png`, and Remarks in `notes/gal_2026-09/The_LoRA_certificate.tex`); it is **not** an impossibility
  theorem for every NTK-based method. The repo's own algebra on this point is
  `notes/ntk_vs_certificate_comparison.md` (the LoRA-aware linearised fit and the certificate share a zero set; the
  merged-weight fit is mis-specified at every T; measured gaps are solver gaps).
- **Charts:** low-dimensional PCA matched their intuition for the search restriction (already implemented);
  they understood and liked nonlinear charts and raising the search dimension, and were interested in combining
  chart constraints with different reconstruction equations. They explicitly distinguished this from their own
  earlier, unsuccessful deep-image-prior attempt. The meeting establishes that distinction only, not
  literature-wide novelty (the 2026-09-14 audit, `notes/gal_2026-09/LoRA_meeting_audit_2026-09-14.md`, lists the
  closest prior work: Oz et al. 2024, GIAS, GIFD, Bora et al., Yao 2024, FineXtract, DAGER, VGIA).
- **Extension question from Gal:** does the certificate remain useful when a layer's inputs change during
  training (earlier layers adapted)? Discussion included measuring feature/subspace change with the frozen base
  model and the final adapter, and looking for additional equations involving perturbed inputs.
- **Charts beyond the certificate:** whether PCA, possibly adapted iteratively during the search, could also
  reduce the search dimension for KKT reconstruction and other objectives; memory limits of KKT reconstruction on
  larger/more demanding images were raised.
- **Properties they found appealing:** per-image search (no joint reconstruction of the whole dataset); a possible
  link between reconstructability and how strongly an example contributes to training ("how wrong the model gets
  it"), which differs from the margin-example view in KKT reconstruction; the demonstrated reconstruction despite
  few adapter parameters.
- **Scope statement:** they said successful extensions to text or to new kinds of images could be
  *groundbreaking* and lead to a very good paper. This is their assessment of the potential *if* those extensions
  succeed. Public-adapter reconstruction (e.g. adapters trained with SGD) was an ambition discussed, not a result.

## B. Agreed next steps (reported as decisions)

1. Test the multilayer extension and determine how changing layer inputs affect the certificate and reconstruction.
2. Try different charts, including extensions beyond a single fixed PCA parameterisation.
3. Improve reconstruction in concrete experimental settings and explore useful combinations of equations and charts.
4. Make an initial attempt in text.

Emphasis: practical — develop the method, establish where it works; theory explains the equations and names the
conditions the experiments should test.

## C. Discussed, not agreed (possible future directions)

- Charts for KKT and other reconstruction objectives (iteratively adapted PCA); memory limits with larger images.
- The training-residual / "how wrong the model gets it" link to reconstructability. **Not established** at the
  meeting; a large loss alone does not guarantee recoverability (summary, "Technical precision").
- Public-adapter reconstruction under suitable training conditions.
- Better and more complex images; additional training settings.

## D. Our reading (deductions, not decisions)

- The scoped NTK conclusion is consistent with, and narrower than, what the repo already records: the two routes
  are an equivalence where every image is recorded (`notes/ntk_vs_certificate_comparison.md`, ruling family in
  `notes/math_rulings_2026-09-06.md`), so the meeting statement must not be quoted as "NTK cannot identify".
- Gal's changing-input question is exactly the split the repo's multilayer track already measured: the **full**
  certificate is exact at any drift over the training span but loses rank (`rank C = r − N'`, lifetime
  `T < min(r, n_l)/N`), while the **truncated** certificate carries an `O(ε_⊥)` error (theory/T2, T3;
  `experiments/multilayer_cert/RESULTS.md`, job 688036/692603). The archive's multilayer note adds short-time
  trained-family theorems (`notes/gal_2026-09/multilayer_lora_theory_source.tex`, Thm 5.3, 8.2). What has **not**
  been run is a reconstruction with an adapter *inside* the backbone on a trained network (the "conv cell" caveat in
  the 2026-09-06 results bundle: every image cell so far has the adapter on the head or on a first layer).
- "How wrong the model gets it": the repo has a measured negative on the ordering claim (record strength is a
  threshold in exact arithmetic, graded only under reduced-precision training; recording measures do not order
  recovery — `STATUS.md` 2026-09-06 sections, results bundle `READ_ME_FIRST`). Any experiment here must be
  designed as a ratio/threshold test, not a correlation.
- The summary's own "Technical precision" caveat is important for the multilayer experiments: the attacker sees only
  endpoint features (`Φ⁰(x)`, released adapter), never the intermediate feature trajectory; controlled experiments
  can record the trajectory and test whether endpoint diagnostics predict certificate error.

---

## E. Prioritised experiment backlog (proposed; each item states question, baseline, measurements, criteria)

Priority order is ours. Items 1–3 map to agreed steps 1–3; item 4 to agreed step 4. Nothing here has been launched
as part of the 2026-09-17 integration. Compute: WEXAC only, FP64 for anything that feeds a certificate residual.
Every run saves `x_train`, reconstructions, chart projections and residuals as `.pth` and a best/worst grid.

### 1. Multilayer / changing-input reconstruction on a trained backbone (agreed step 1) — HIGHEST
- **Question.** With the adapter on an *inner* layer of a trained network (inputs to the adapted layer move
  during training), does certificate-guided chart search still return the private images, and which diagnostic
  (full vs truncated certificate, `N'` estimate from the `B_T` spectrum, `ε_⊥`) predicts success from the release?
- **Baseline.** The same network with the adapter on the head (the current working cell: CIFAR new-class MLP/CNN,
  `experiments/cifar/cifar_newclass.py`, 8/8 recoveries, wrong-release control 0/200).
- **Design.** Adapt layer ℓ ∈ {head, last hidden, first hidden}; T ∈ {1, 4, 16, 64}; record the true feature
  trajectory `H_{ℓ,t}` (experimenter-side), `N'`, `rank B_T`, spectral gap, `ρ_full`, `ρ_trunc`, then run the
  certificate search from ≥100 random starts on the fixed public PCA chart (k = 32).
- **Measurements.** landings/starts, images found (of 8), precision of the lowest-residual starts, residual at
  truth vs at landed points; regress success on `ρ_trunc` and on `N'/min(r,n_ℓ)`.
- **Success / failure.** Success: ≥6/8 found at some T > 1 with precision ≥ 0.9 and the wrong-release control at
  0. Failure is informative if it separates cleanly: `rank C_full = 0` (span filled — information failure) vs
  residual-not-zero (solver failure) vs residual-zero-wrong-image (alias). Do not merge the three.
- **Code to reuse.** `experiments/multilayer_cert/{common,survival}.py` (drift/rank bookkeeping),
  `experiments/cifar/cifar_newclass.py` (search + controls); the M4 item already listed in
  `experiments/multilayer_cert/RESULTS.md`.

### 2. Chart comparison at matched information (agreed step 2)
- **Question.** Which chart family raises coverage of raw (non-projected) private images while keeping the
  identifiability count `k < r − q` and search reliability: fixed PCA, class-conditioned PCA, local PCA patches
  (chart_inversion_theory §7/§11), a decoder-backed chart `G(z) = D(a + Uz)` with a frozen public decoder
  (VAE/diffusion codec), a feature-autoencoder chart with image readout?
- **Baseline.** Fixed public PCA at k = 32 on the head-adapter CIFAR cell; the E4a comparison already measured
  (pixels 0.24–0.32 projection error, CLIP 0.26, DINO 0.41 at k = 32; STATUS 2026-09-17).
- **Measurements.** Per chart: projection error of the truth (coverage), stacked Jacobian rank of `C·DΦ⁰·DG` at
  the truth, condition number, landings/starts, images found, and image error of landed points against the *raw*
  image (not only its chart projection).
- **Success / failure.** A chart counts as a gain only if raw-image error falls at equal or better precision; a
  chart that raises `k` past `r − q` and loses precision is a negative with a named cause (count), not a tuning
  failure. Report the `k = p` saturated case separately (chart theory Thm 2.11(3): zero residual a.s. even for a
  wrong chart).

### 3. Equation combinations and reconstruction improvement (agreed step 3)
- **Question.** Does chaining certificate candidates with the joint factor fit (representer independence clause,
  `notes/gal_2026-09/B_factor_representer_idea.md`; `notes/ntk_vs_certificate_comparison.md` §3) or with exact
  replay on the certificate fibre raise coverage (the certificate's measured bottleneck: 53 of 89 landings on one
  image) without losing precision?
- **Baseline.** Certificate-only search on the same release/chart/starts/budget.
- **Measurements.** Distinct images found per 100 starts; residual at truth for each objective; alias count under
  optimal permutation; wall-clock and peak memory per objective (the memory question raised at the meeting).
- **Success / failure.** Coverage up with precision held. A joint objective that fails to reach its own model
  floor at the truth is a solver failure and is reported as such.

### 4. Text pilot (agreed step 4)
- **Question.** In the smallest exact setting (frozen contextual features, LM-head or first-block value adapter,
  vanilla SGD, `B_0 = 0`, no dropout, FP64), does the certificate separate a finite candidate family of short
  private records (Text plan Prop 2, "finite codebook separation") with a measurable margin?
- **Baseline.** Public LM prior alone; certificate alone; shuffled-secret and wrong-adapter controls.
- **Design.** Stages 1–2 of `notes/gal_2026-09/Text_Certificate_Chart_Research_Plan.md` §11 (theorem
  instrumentation with r ∈ {4, 8, 16, 32}; 1–4 records from 10³–10⁴ candidates). Record token support, ordered
  match and whole-record match separately; supervise EOS so the last private token enters a contributing prefix.
- **Success / failure.** Exhaustive scan isolates the true candidate with the false-candidate margin above the
  perturbation error; the `q ≥ r` region must show `C = 0` (the main text obstruction) and be reported as such.
  Adam and dropout are separate arms, expected to break the invariant (Text plan §9 counterexample).

### 5. E1B refit against the full released factor (carried from the approver's handover, not from the meeting)
- Per `notes/APPROVER_HANDOVER.md` and A13/A14 in `notes/plan_audit_2026-09-07.md`: refit the replay arms against
  `A_T` (1536 equations) instead of the product (192), re-measure nullity in the configuration run, and report an
  identifiability negative if it stays positive. Jobs 350929–350931 (LM reruns) were in flight on 2026-09-17.

### 6. Training-residual vs reconstructability (discussed, not agreed)
- Only as a threshold/ratio test on the existing record-strength cells (jobs 255095/255098/279342), never as a
  correlation over a handful of points; the existing negative (threshold, not ordering) stands until a matched
  design says otherwise.

---

## F. Cross-references

- Meeting summary (verbatim): `notes/meeting_summary_2026-09-15.md`
- Pre-meeting audit of the briefing: `notes/gal_2026-09/LoRA_meeting_audit_2026-09-14.md`
- Technical record of definitions, claims and assumptions: `notes/technical_record_2026-09-17.md`
- Audited claim state, claim by claim: `results/CLAIMS_LEDGER.md` (GM lane); orientation map `results/00_map.md`
- The method behind agreed step 1's "perturbed inputs" remark: `notes/perturbed_inputs_what_they_supply_2026-09-17.md`
  and `notes/fibre_capacity_formula_2026-09-17.md` (sibling lanes, 2026-09-17)
- Evidence record (claims → jobs/code): `notes/archive_evidence_map_2026-09-17.md`
- Artifact index for the 2026-09-17 archive: `notes/artifact_index_2026-09-17.md`
- Feedback on slides is logged in `docs/presentation-remarks-log.md` (entry 2026-09-15).
