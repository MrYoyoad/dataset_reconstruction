# Corrections and superseded ideas — what the 2026-09-17 archive changes, and what it confirms

Companion to the repository's own correction record. **The rulings of record stay where they are:**
`notes/math_rulings_2026-09-06.md` (R1–R20), `notes/plan_audit_2026-09-07.md` (A1–A17+),
`results/CLAIMS_LEDGER.md` §F (withdrawn claims, kept in place with their reason),
`theory/AUDIT_2026-09-17.md` (independent audit of T1–T6), `LESSONS_LEARNED.md` (how numbers went wrong).
This file records only what the **archive** adds: corrections that live inside the archived documents, the
supersession chain between archive revisions, and the places where an archive document and a repository file
disagree. Nothing here overrides a ruling; where a conflict is open it is marked **OPEN** and routed to the
approver lane rather than settled.

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

## 3. Where an archive document and the repository disagree — OPEN, routed to the approver

These are recorded as conflicts with both sources cited. None is settled here, and no working code was changed to
agree with a document.

1. **`theory/T4` C5 wording vs the archive's Thm 8.2 / App A.4.** T4 records that a cancellation making the
   truncation error quadratic was "searched, not found". The archive exhibits a *structural* mechanism for exactly
   that (initial-gradient range condition `col(∇_{B_l}L(θ_0)) ⊆ U`, first-order forward drift staying in the span)
   in one designed family. The theorems do not conflict — T3.2's "in-span drift is free" is the same mechanism —
   but C5's wording is too strong as a general statement.
2. **`experiments/multilayer_cert/RESULTS.md` §3 attribution.** It attributes the `rank B_T < N'` rows to the last
   layer. A re-aggregation of `results/multilayer_cert/survival_692603.jsonl` puts 36 of 64 such rows at hidden
   layers, with the last-layer rows additionally capped by the softmax bound `rank B_T ≤ m − 1` (archive, PROVED).
   Needs a row-level recheck before either statement is quoted.
3. **`theory/T5`'s assumption table** still lists R2 ("violated by a shared seed") although the defence was
   withdrawn. The archive's Thm 10.2 needs only a **density** on the layers' allowable covectors, not independence,
   which is what the measured shared-seed result shows.
4. **Three hand-computed expansions in the archive** (Ex 3.5, Ex 5.5, §8.6 — coefficients `−3/8`, `1/16`, `−1/3`)
   are marked PROVED and have never been checked numerically here; the repo's T3 check validates a different
   instance. A short FP64 check of each is cheap and would close the gap.
5. **Emphasis, not content.** The archive's title page answers Gal's changing-input question with "yes, locally and
   for sufficiently short training", while `theory/README.md` says the perturbative framing "is not the right one".
   Both documents contain both results. The merged statement: *exact with rank `r − N'` for any `T`; truncated with
   `O(ε_⊥)` error, provably `O(τ)` in the MLP family inside a short-total-step-size window.*

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
