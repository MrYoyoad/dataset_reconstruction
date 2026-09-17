# Corrections and superseded ideas — what the 2026-09-17 archive changes, and what it confirms

Companion to the repository's own correction record. **The rulings of record stay where they are:**
`notes/math_rulings_2026-09-06.md` (R1–R20), `notes/plan_audit_2026-09-07.md` (A1–A17+),
`results/CLAIMS_LEDGER.md` §F (withdrawn claims, kept in place with their reason),
`theory/AUDIT_2026-09-17.md` (independent audit of T1–T6), `LESSONS_LEARNED.md` (how numbers went wrong).
This file records only what the **archive** adds: corrections that live inside the archived documents, the
supersession chain between archive revisions, and the places where an archive document and a repository file
disagree. Nothing here overrides a ruling. The four conflicts §3 raises were **ruled by the approver lane the
same evening** (A24, commit e0a2bb1); they are recorded with their rulings and their owning lanes, and the edits
they call for belong to those lanes, not to this one.

---

## 1. Corrections the archive documents make to themselves

### 1.1 `chart_inversion_theory.tex` / `charts.pdf`, Appendix A (8 September 2026)

The chart note's Appendix A is explicitly titled "Technical corrections to the supplied premises" — the premises
being our own earlier certificate formulation, which the note takes as an input it did not re-derive (line 2368).

| # | Correction | Consequence for us |
|---|---|---|
| A.1 | The score `R(h) = ‖Ch‖²/‖A_Th‖²` is defined only on `{h : A_T h ≠ 0}`; the zero-set theorems are about `Cψ`, not about `R`. | A candidate with `A_T h = 0` must be skipped, not scored. Already the behaviour of the certificate note §2. |
| A.2 | The affine minimum is an **infimum** and can be attained only at infinity (Counterexample 5.2: `R = 1/(1+w²)`). | "λ_min = 0" does not certify a finite chart point. |
| A.3 | The stated optimizer class permits a scalar update that annihilates the initialisation imprint, so the rank/kernel identities need either `γ ≠ 0` or to be assumed directly. Counterexample 5.1: two SGD steps with scalar weight decay give `rank B_2 = 1` **and** `C = 0`, on an open neighbourhood of initialisations. | The assumption "vanilla SGD, no weight decay" is load-bearing and its failure is not a small perturbation. Also: raw-scale conditioning laws need the extra scalar-imprint assumption `C = γ P A_0`, which the kernel identity does **not** imply. |
| A.4 | The exact distributional laws (beta, Wishart, MSE) are laws of the *kernel experiment* and do **not** survive conditioning on "the release was valid"; only almost-sure statements do. | Any threshold computed from those laws on a real release is heuristic, not a guarantee. |
| A.5 | **"The proposed capacity line involving image count is not used here and is not converted into a theorem by substituting `q`."** | The strongest single correction in the archive for our prose: the chart theorems give `k < r − q` with `q = rank H`, and the repo's `k < r − N` form carries the extra hypothesis `q = N`. Limitation 6.6 repeats it: "None of the chart theorems … establishes the proposed capacity line." |
| Remark 3.9 | The earlier "200 equations" claim is re-read as a bookkeeping bound: the independent first-order constraints are the **rank of the actual chart derivative**, never a count. "`q_ℓ` … is not interchangeable with `M`." | Matches the approver's standing rule that a count is never identifiability. |

**Not adopted, and why.** The note's Thm 2.13 displays its uniform small-residual bound as a product where the
proof requires a sum; as printed the bound is too small. Recorded here, not corrected in the source (we do not edit
the theory documents — `CLAUDE.md` ground rule 6 for the exact-inversion track, applied here by analogy).

### 1.2 `multilayer_lora_theory_source.tex` (7 September 2026)

- The edition states that it "rewrites and extends the earlier report *What survives representation drift? Exact
  and perturbative multilayer LoRA certificates*", making the ordinary MLP the architectural anchor and the
  residual family a special case. The earlier report is not in the archive.
- Its own FALSE results are stated as such: a universal `O(τ²)` claim for ordinary smooth MLPs is **false**
  (Ex 3.5, Ex 5.5); an unrestricted pathwise horizon-free assertion is false (accumulation `Tε` and depth
  amplification `a^j`); "zero B-initialisation does not make the functional adapter quadratic" (App A.2).
- It records a PROVED cap the repo had not written down: at a softmax logit layer `1^T D_t = 0` forces
  `rank B_T ≤ m − 1`.

### 1.3 `LoRA_meeting_audit.md` (14 September 2026) — now `notes/gal_2026-09/LoRA_meeting_audit_2026-09-14.md`

The pre-meeting audit of the briefing and companion documents. Its corrections are about how evidence was
**classified**, and several are directly about claims that had been circulating in our own prose:

- "The LoRA seed is unknown even at one step" — **false** for simultaneous SGD with `B_0 = 0`, because `A_1 = A_0`,
  so the released factors reveal it at `T = 1`. At later times the certificate avoids needing it.
- "`k < p` plus tangent separation guarantees exactly the private images globally" — **missing the global span
  separation clause**. Tangent separation is local; wrong images already inside the private span survive every
  seed. The audit supplies an explicit counterexample: `ψ(t) = (1+t, t(t−1), t²(t−1))` has a true point at `t = 0`
  with `k = 1 < p = 2` and a *second* passing point at `t = 1`, because `ψ(1) = 2e₁` lies in the private span.
- "A rank-`r` representer determines only the training images' span" — must be qualified: it is a statement about
  compatibility with a recorded sketched span under rank conditions, **not** a statement about all NTK methods or
  about the full release.
- "Nonlinearity between family and adapter eliminates blends" — nonlinearity can remove affine mixtures, but the
  family must separately satisfy the separation conditions.
- "Every on-family private input is exactly recovered" — coverage, identifiability and solver success are separate;
  the record is all eight projected letters and motorcycles but **4 of 8** Fashion target matches with two further
  recognisable approximations.
- "Joint fits stalled because they optimised `N(k+m)` unknowns" — **not** the diagnosis: the walkthrough already
  eliminated coefficients by least squares.
- "No one has published LoRA-factor reconstruction" — not established as a broad claim. The audit's primary-source
  pass names Yao 2024 (released diffusion-LoRA weights → private identities), FineXtract (ICML 2025), DSiRe,
  UTR, FedSpy-LLM, Hu et al. 2026, PEFTLeak (CVPR 2025, malicious model), PRISM, plus Oz et al. 2024, GIAS, GIFD
  and Bora et al. as the closest chart/generator prior work. **Priority remains a question, not a finding.**
- Figure-level corrections that we inherit: the three-chart MNIST panel is **replay** evidence, not another
  certificate success; the letters panel must keep four rows (original / PCA training target / NTK output /
  certificate output), because the PCA row is what separates chart approximation from inversion error; "found"
  means the projection was recovered.

### 1.4 `Text_Certificate_Chart_Research_Plan.md` (15 September 2026)

Self-limiting statements worth carrying: only one computation was performed (a Gaussian-dictionary example); no
transformer reconstruction experiment has been run; a title claiming adapter-gradient text reconstruction (UTR) is
about nonlinear bottleneck adapters and is **not** a LoRA-factor endpoint attack; "do not reuse the smooth
image-chart theorem on hard strings"; Adam and dropout are theorem boundaries, with an explicit two-dimensional
Adam counterexample where `rank B_1 = q` yet `C h ≠ 0`.

---

## 2. Supersession chains inside the archive

Established by content and by explicit corrections, not by filename or modification date. (The archive's own
README warns that it retains separately-named older drafts deliberately, and that "FINAL" in a name means nothing.)

| chain | current member | superseded members | basis |
|---|---|---|---|
| Certificate note | `The_LoRA_certificate.{tex,pdf}` (2026-09-10) | the certificate sections of the earlier briefs and walkthroughs | Latest statement of A1–A5 + Claims 1–3; the only one that states the excitation caveat and the `ker C` decomposition together |
| Multilayer | `multilayer_lora_theory{.pdf,_source.tex}` (2026-09-07) | *What survives representation drift?* (not in archive) | Stated in the edition note, line 1737 |
| Charts | `chart_inversion_theory.tex` / `charts.pdf` (2026-09-08) | the chart paragraphs of the earlier plans | Appendix A supersedes the earlier premises it corrects |
| Meeting materials | `LoRA_meeting_audit.md` (09-14) + `LoRA_reconstruction_brief.pdf` + `LoRA_meeting_personal_companion.pdf` (09-14) | `LoRA_meeting_walkthrough_*`, `LoRA_meeting_board_guide*`, `lora_brief_v2` | The audit states it is a structural rewrite that *replaces* the earlier long section sequence |
| Plans | `lora_plan_rev11*` (09-03/09-07) | rev9, rev6, rev5, rev4, `lora_plan.pdf`, `lora_thesis_research_plan.pdf` | Sequential revisions of one document |
| Identifiability note | `identifiability_rank_bound(5).pdf` (08-30) — **byte-identical to the repo's `notes/identifiability_rank_bound.pdf`** | `(4)`, `(2)`=`(3)`, `(1)`, base | sha256 match against the repo file; the repo already holds the current member |
| Meeting record | `notes/meeting_summary_2026-09-15.md` (in repo, committed 8b42fc3) | the archive copy `Thesis_meeting_summary_2026-09-15.md` | byte-identical (sha256 `1efb0119…`); the archive copy was not re-imported |

Byte-identical duplicates inside the archive (same sha256, different paths) are listed in
`notes/artifact_index_2026-09-17.md`; they were deduplicated, not deleted.

---

## 3. Where an archive document and the repository disagree — RULED 2026-09-17 (approver, commit e0a2bb1)

Raised here as conflicts with both sources cited, and ruled the same evening by the approver lane as A24. **The
edits belong to the lanes that own those files, not to this one**, so the rulings are recorded rather than
applied. No working code was changed to agree with a document.

1. **`theory/T4` C5 wording vs the archive's Thm 8.2 / App A.4** — **RULED: reword, and it is a unification.**
   C5 records that a cancellation making the truncation error quadratic was "searched, not found"; the archive
   exhibits a structural one (the initial-gradient range condition `col(∇_{B_l}L(θ_0)) ⊆ U`, first-order forward
   drift staying inside the span). A claim about a search is refuted by exhibiting one, so the wording must go.
   The ruling's substantive point goes further than the correction: **that range condition is the same mechanism
   as T3.2's "in-span drift is free"** — two files describing one phenomenon in different vocabularies, which
   should be merged rather than patched. *Owner: the multilayer/theory lane.*

2. **`experiments/multilayer_cert/RESULTS.md` §3 attribution** — **RULED: FAIL, with two faults rather than one.**
   The attribution is wrong (36 of 64 such rows sit at hidden layers 1–2, only 28 at the last layer), *and* the
   last-layer rows are additionally capped at `m − 1` by a proved softmax bound the file does not mention — which
   is **the simplex cap this project already holds in its own ledger** (`N′ ≤ m − 1`). So the file attributes to
   one cause what demonstrably has two, and the second was already ours and unreferenced. Correct at source.
   **The re-aggregation has had one reader**, so the multilayer lane confirms before the edit.
   *Owner: the multilayer lane.*

3. **`theory/T5`'s stale R2 row** — **RULED: remove it; three independent routes now agree.** A counterexample
   built with independent Gaussian seeds that still fails; the measured `[9,18,20,20]` identical under shared and
   independent seeds; and the archive's Thm 10.2 requiring only a **density** on the allowable covectors rather
   than independence. Independence was never the hypothesis. The refutation already in T5 is the correct reading.
   *Owner: the theory lane.*

4. **Three hand-computed coefficients marked PROVED and never evaluated** (Ex 3.5 `−3/8`, Ex 5.5 `η/16`, §8.6
   `−1/3`) — **RULED: run the checks.** "A hand-computed constant carrying a PROVED status with no independent
   evaluation is the same shape as a numerical check whose harness implements the corrected law — a status resting
   on the author's own arithmetic." **Written and committed here as
   `experiments/archive_checks/check_hand_coefficients.py`** (FP64, CPU, seconds; submitter
   `scripts/run_archive_coeff_check_wexac.sh`). It re-derives each coefficient from the dynamics the note
   specifies rather than re-evaluating the note's own closed forms, checks the note's exact intermediate matrices
   first, and withholds the coefficient verdict when those disagree — `construction mismatch` and
   `coefficient wrong` are different outcomes. Tolerances are pre-stated in the module docstring.
   **It has not been run**: this lane launches nothing, and the standing rule is that nothing runs locally.
   *Owner: whoever next submits — one bsub.*

5. **Emphasis, not content.** The archive's title page answers Gal's changing-input question with "yes, locally and
   for sufficiently short training", while `theory/README.md` says the perturbative framing "is not the right one".
   Both documents contain both results. The merged statement: *exact with rank `r − N'` for any `T`; truncated with
   `O(ε_⊥)` error, provably `O(τ)` in the MLP family inside a short-total-step-size window.*

**And one finding the ruling asked to be kept in these words.** `theory/T1…T6` cite neither the archive note nor
its lemmas and agree on every shared statement. That is a **consistency signal, not a joint proof** — evidence
about the derivations, not a second proof of the result.

---

## 4. Ideas the archive shows were tried and set aside

Kept so that a future session does not re-propose them as new.

- **The merged-weight linearised fit** (`ΔW ≈ Σ r_i φ(x_i)^T`): mis-specified at every `T`, measured model floor
  0.96 at `T = 1` against 2.9e-16 for the LoRA-aware form. Do not use it as the linearised arm of any comparison.
- **"Which route recovers more"** as a research question: withdrawn — the zero sets coincide where every image is
  recorded, so the comparison measures solvers.
- **The free-coefficient change-of-basis symmetry as a proof of non-identifiability**: it is not automatically a
  symmetry of training replay (toy check: free-fit error 2.65e-16 vs replay error 0.2998).
- **A defender-facing form of the counting rule**: withdrawn, because the rule is one-sided (sound when it says
  closed, silent when it says open) and the remedy belongs to the attacker, who picks the chart.
- **"A linear chart beats a learned one"** and **"more training makes the chart worse"**: the first withdrawn
  outright (unmatched comparator), the second narrowed to "the largest budget is worse than both smaller ones".
- **Shared `A_0` across layers as a defence**: refuted by measurement (identical stacked ranks).
- **The 32-dimensional seed-free fibre** and **"the reduced parametrisation makes the truth isolated"**: both
  withdrawn by the approver after job 697344 measured nullity 232 in both arms.
