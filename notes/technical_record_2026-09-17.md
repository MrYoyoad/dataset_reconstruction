# Technical record — definitions, claims, assumptions, open problems (as of 2026-09-17)

Built by reconciling the 2026-09-17 archive (`Thesis_files_2026-09-17.zip`, 92 files + 7 nested bundles) with the
repository. Each entry says what kind of statement it is: **PROVED** (with the document that carries the proof and
its exact hypotheses), **MEASURED** (with the job id), **CONJECTURE / PROPOSED**, or **WITHDRAWN**. Claim-to-evidence
links live in `notes/archive_evidence_map_2026-09-17.md`; the corrections history is in
`notes/corrections_from_archive_2026-09-17.md`; the meeting record is `notes/meeting_2026-09-15_decisions_and_backlog.md`.

**Standing rule (from `notes/APPROVER_HANDOVER.md`):** a numerical check agreeing with a statement never promotes
its proof status, a count is never identifiability, and verdicts never merge "residual not zero" (optimisation
failure) with "residual zero, wrong image" (alias). Nothing below was re-derived by this session; the proof status
recorded is the status the cited document claims, and the archive theory documents have not been independently
re-derived by a sibling session (`theory/README.md` rule).

---

## 1. Notation, and one trap

| symbol | meaning | note |
|---|---|---|
| `A_t ∈ R^{r×n}`, `B_t ∈ R^{m×r}` | LoRA factors of one adapted layer `W_0 + B_tA_t`; released at step `T` | `B_0 = 0`, `A_0` random |
| `h_i = φ(x_i) ∈ R^n`, `H = [h_1…h_N]` | private inputs *to the adapted layer* | `φ` = frozen backbone |
| `q = rank H` | **feature rank** | `q ≤ N`; `q = N` only if the private features are independent |
| `N` | number of private examples | **not** interchangeable with `q` |
| `S = col(A_0H)`, `P = P_{S^⊥}` | excited subspace and its complement | `P` is not observable |
| `C = P_{row(B_T)^⊥} A_T` | **the certificate** | computable from the release alone |
| `p = r − q` | certificate rank | the number of equations per candidate |
| `k` | chart dimension (free coordinates per candidate image) | |
| `G`, `ψ = φ∘G` | chart (image side) and induced feature chart | |
| `N'_l = dim Σ_{t<T} col(H_{l,t})` | **training span** dimension at layer `l` | `N' ≥ N`; not observable |
| `ε_l^⊥` | orthogonal representation drift at layer `l` | the axis that matters; in-span drift is free |

**The trap, and it has bitten the project's own prose.** `q` (feature rank) and `N` (example count) are different
objects. The chart note refuses to substitute one for the other ("`q_ℓ` … is not interchangeable with `M`",
`chart_inversion_theory.tex` Remark 3.9 line 922; Correction A.5 line 2391 declines to convert the capacity line
into a theorem by substituting `q`). The multilayer note says the same ("a rank, not a number of examples",
`multilayer_lora_theory_source.tex` line 141) and proves the case where they coincide for a softplus MLP
(Thm 5.1, lines 501–546: `n ≥ N + k` ⇒ `rank[H K_i] = N + k` a.s., so `q = N`; the takeaway at line 546 is "here,
and only after this proof, we may replace feature rank `q` by example count `N`"). Repo statements written with
`r − N` are therefore conditional on independent private features.

---

## 2. The single-layer certificate — PROVED, with its exact assumption stack

**Source of record:** `notes/gal_2026-09/The_LoRA_certificate.{tex,pdf}` (10 September 2026), Claims 1–3 and the
Theorem in §5. The repo's own proof is `notes/w1_certificate_proofs.tex`.

**Statement.** Under A1–A5 below, and for almost every initialisation, `C = P_{row(B_T)^⊥}A_T = P A_0`, hence
`C H = 0` and `rank C = r − q`. The test holds at any finite SGD step satisfying excitation, with no convergence
assumption, and gives exactly `r − q` independent equations per candidate.

**Assumptions, verbatim in substance (§3 of the note):**

- **A1 Initialisation.** `B_0 = 0`; entries of `A_0` drawn independently from distributions with densities,
  independently of `H`. Gives `rank A_0 = r` and `rank(A_0H) = q` almost surely. Strict `q < r` is what leaves a
  nonzero complement.
- **A2 Vanilla SGD.** Both factors updated with minibatches and scalar learning rates, differentiable loss of the
  layer outputs; no momentum, weight decay or parameter regularisation. Extensions covered: momentum with zero
  initial buffers; scalar weight decay *provided the preserved seed scale is nonzero at release*. **Adam and AdamW
  are not covered** (per-entry scaling does not preserve the subspaces).
- **A3 Fixed inputs.** `H` is the same at every step — a head adapter, or the first adapted layer behind a frozen
  deterministic backbone with fixed preprocessing.
- **A4 Few-shot dimensions.** `q < r ≤ n` and `q ≤ m`; at a softmax logit layer with a loss depending only on
  probabilities, `q ≤ m − 1`.
- **A5 Excitation at release.** `rank B_T = q`, i.e. the rows of `B_T` span the excited subspace. Without it the
  invariant still gives `row(B_T) ⊆ S`, so `s := rank B_T ≤ q` and **the release always gives a lower bound on `q`**.

**Two corrections that must travel with the statement.**

1. **Excitation is computable, not verifiable, by the attacker.** `rank B_T ≤ q` holds unconditionally, so the
   attacker gets a lower bound on `q`; equality is the hypothesis and testing it needs `H`. (Repo commit
   43f980e, from the W1 lane on itself; `notes/APPROVER_HANDOVER.md`.) A visible singular gap protects the
   *construction* of the certificate, it does not prove the hypothesis
   (`multilayer_lora_theory_source.tex` line 303; `chart_inversion_theory.tex` Remark 3.9 line 922).
2. **A2 is load-bearing and its boundary is sharp.** With scalar weight decay the certificate can collapse to
   `C = 0` while `rank B_T = q` still holds, on an open set of initialisations
   (`chart_inversion_theory.tex` Counterexample 5.1, line 1022). Adam breaks it at one step by direction mismatch
   (`notes/gal_2026-09/Text_Certificate_Chart_Research_Plan.md` §9 counterexample: `r = 2`, one feature,
   `rank B_1 = q = 1`, yet `C h ≠ 0`). A merged or refactorised release gives a zero certificate
   (`multilayer_lora_theory_source.tex` line 219) — which is why any defence evaluation needs an **absolute** rank
   floor, never a relative one.

**What the certificate leaves undetermined — PROVED.** `ker C = col(H) ⊕ ker A_0`, of dimension `q + n − r`
(certificate note, Remarks). So `ker C = col(H)` only when `r = n`, and **every element of the private span,
including blends that are not training images, is an exact zero.** This is the blend degeneracy, and it is shared
by the linearised route (§4 below).

---

## 3. Charts — what is proved about identifiability

**Source of record:** `notes/gal_2026-09/chart_inversion_theory.{tex,pdf}` (8 September 2026, 65 pp, 47 proved
theorem-class items, 12 counterexamples). The note states plainly: "No numerical experiments or numerical research
results are claimed" (line 79), and it treats the certificate identities as *supplied inputs it did not re-derive*
(line 2368).

Roles, as the note separates them (answering a question the meeting raised):

- **certificate = the evidence** — a fixed linear operator on features, computable from the release without `A_0`
  or the recipe, whose kernel contains the whole private span, so it can never by itself distinguish span points;
- **chart = the prior restriction, not another measurement** (line 1195) — it supplies coverage and decides which
  `k` coordinates are searched;
- **decoder = the frozen public expansion network inside the chart**, `G_j(z) = D_x(a_j + U_j z)` (Def 7.1). "A
  large frozen decoder does not add unknown coordinates; any weights, noise, or conditioning optimised during
  inversion do" (line 66).

| # | Statement | Status | Exact hypotheses |
|---|---|---|---|
| C1 | Local: `σ_min(J) > 0` with `J = DF·DG` ⇒ isolated, bi-Lipschitz zero; residual `η` ⇒ latent error `≤ 2η/α` | PROVED (Thm 1.2) | `F, G ∈ C¹`; local only, "no global optimization claim follows" |
| C2 | With the truth off-chart by `δ` and residual error `η`, every minimiser obeys `‖G(ẑ) − x_*‖ ≤ δ + 4L_G(L_Fδ + η)/α` | PROVED (Thm 1.3) | Lipschitz constants, derivative variation `≤ α/2` on the ball |
| C3 | Certificate on a chart: a.s. `rank(C·Dψ(w_*)) = min(p, k − d)`, `d = dim(T ∩ ℋ)` ⇒ local uniqueness iff `k ≤ p` and the tangent misses the private span | PROVED (Thm 2.4) | Chart fixed independently of `A_0`, immersed at the true point |
| C4 | **Global on one fixed chart:** if `k < p` then a.s. the certified set equals the chart's intersection with the private span (`ψ(Ω) ∩ ker C = ψ(Ω) ∩ ℋ`) | PROVED (Thm 2.5, parametric transversality + Sard) | Fixed `C^∞` chart independent of `A_0`; **strict** `k < p`; at `k = p` aliases are isolated but can be arbitrarily many and arbitrarily far (CE 5.3, 5.4, 5.7) |
| C5 | The same guarantee survives choosing the chart *index* from a fixed countable public atlas after seeing the release | PROVED (Thm 10.3) | Atlas fixed independently of `A_0`; a chart *constructed* after seeing `C` can manufacture zeros (line 1243) |
| C6 | Global uniqueness of the image | PROVED **only** for the synthetic polynomial (Thm 2.7) and quadratic (Thm 2.20) feature families | The adapted layer must actually receive those monomial features (Remark 2.8, line 456) |
| C7 | A wrong affine chart at distance `d_aff > 0` is rejected: `ρ_w²/d_aff² ~ Beta((p−k)/2, (b−p)/2)`; at `p = k` the residual is zero a.s. **even for a chart missing the span entirely** | PROVED (Thm 2.11) | Gaussian `A_0`; a law of the kernel experiment, *not* conditional on a valid release (Remark 2.3, Correction A.4) |
| C8 | Multilayer stacking: a.s. `rank J = min_I (Σ_{ℓ∉I} p_ℓ + dim Σ_{ℓ∈I} W_ℓ)`; `Σ p_ℓ ≥ k` is only the `I = ∅` member | PROVED (Thm 3.3) | Independent densities per layer on fixed sensing spaces — a model the note says fails under drift (Lim 6.7) |
| C9 | Fibre barrier: if two candidates give the same observation, any rule has worst-case error `≥ ‖x_1 − x_2‖/2` | PROVED (Lim 6.1) | — |

**The identifiability condition, stated once and correctly.** Local: `k ≤ p = r − q`. Global alias-freedom on a
fixed chart: **`k < r − q`**, *plus* the separate geometric requirement that the chart meet the private span only at
the intended points. The count alone is never identifiability — this is the approver's standing rule and the chart
note's Remark 2.6 ("does not say that the intersection with `ℋ` consists of the training points. That is a separate
property of the chart").

**Superseded by the chart note's own Appendix A:** the earlier "capacity line involving image count" is not used
there and is not converted into a theorem by substituting `q` (Correction A.5, line 2391); the score
`R = ‖Ch‖²/‖A_Th‖²` needs the domain `A_Th ≠ 0` (A.1); the affine minimum is an infimum that may be attained only
at infinity (A.2, CE 5.2).

---

## 4. Certificate vs the linearised / NTK route — PROVED equivalence, and the scoped meeting conclusion

**Source:** `notes/ntk_vs_certificate_comparison.md` (2026-09-06; byte-identical to the archive's
`EQUIVALENCE_linearised_vs_certificate.md` up to the repo's later edits) and the Remarks in
`notes/gal_2026-09/The_LoRA_certificate.tex`.

- **PROVED.** The LoRA-aware free-coefficient fit `B_T = Σ_i u_i (A_T h_i)^T` is exact **iff** every candidate is a
  certificate zero *and* the candidate readings are linearly independent — given `rank B_T = N`. The certificate is
  the per-candidate form; the representer is the joint form. Where `N' < N` are recorded the "only if" direction
  breaks specifically.
- **PROVED.** The *merged-weight* form `ΔW ≈ Σ_i r_i φ(x_i)^T` is mis-specified at every `T`, because a LoRA step
  moves the adapter by the gradient composed with the adapter. Measured model floors: `2.89e-16` (LoRA-aware) vs
  `9.64e-01` (merged) at `T = 1`; `8.05e-16` vs `7.15e-01` at `T = 400` (job 308859).
- **PROVED.** The free-coefficient fit has a change-of-basis symmetry `U(A_TX)^T = (US)(A_TXS^{-⊤})^T`, which is
  **not** automatically a symmetry of training replay: transformed columns must be realisable images and their
  actual loss derivatives must reproduce the release (certificate note, Lemma; toy check in
  `notes/gal_2026-09/gal_ntk_cluster_brief.md`: free-fit relative error 2.65e-16 against replay 0.2998).
- **MEASURED (solver, not information).** On one release/chart/starts/budget: linearised residual 1.05e-02 against
  its own floor 2.89e-16 (ratio 3.6e13), 0 of 8 images; certificate residual 4.39e-14, 4 of 8. Residual far above
  its floor with wrong images is an **optimisation failure**, not an alias.
- **The 2026-09-15 meeting conclusion, with its scope.** The participants accepted that the NTK reconstruction
  approach *discussed* would not work for the LoRA setting *considered*. Recorded as the reported meeting
  conclusion. It is consistent with the above and narrower than it sounds: the failing object is the merged
  coordinate / raw-gradient-mixture formulation (and, at a head, a finite kernel with no universal-kernel
  guarantee), not every NTK-based method. Any sentence of the form "cannot identify by either route" is **false** —
  a measured cell falsified it (`notes/APPROVER_HANDOVER.md`).
- **WITHDRAWN.** The earlier framing of the two routes as competitors, and any claim that one recovers more.

---

## 5. Multilayer and changing inputs

Two documents, complementary, no contradiction on any proved statement.

**Repo track (`theory/T1..T6`, `experiments/multilayer_cert/`) — the full certificate.**

- **PROVED (T2, Prop. A).** The closure induction never needed frozen inputs, only inputs confined to a *fixed
  subspace*. It runs verbatim at any depth with the training span replacing the private span, and the base
  representation `H_l^0` is annihilated exactly because it is the `t = 0` member of that span. MEASURED: relative
  residual `6.5e-15` at 923% representation drift; `2.7e-13` max over all 110 rows where `rank B_T = N'`
  (jobs 688036, 692603).
- **PROVED + MEASURED.** What depth costs is **rank**, not accuracy: `rank C_full = (min(r, n_l) − N')_+`, held
  110/110; the span inflates at the maximal rate `N' = N·T` in most deep rows, giving a **lifetime**: the
  certificate is empty once `T ≥ min(r, n_l)/N`. Death is discontinuous (measured ladder 4, 2, 0, 0).
- **PROVED FALSE (T3).** The truncated (rank `r − N`) certificate's error is **not** `O(ε²)`: log-log slope
  1.0004, closed-form coefficient matched to 0.45%. In-span drift is free (`4.7e-14` at 100% in-span drift), so
  the axis is the **orthogonal** drift.
- **MEASURED.** Where `rank B_T < N'` the certificate is *contaminated*, not weakened: median residual 2.3e-4, max
  0.27, with no small parameter. The sting: `rank B_T = N` is not attacker-side proof of exactness at depth,
  because the attacker cannot observe `N'`.
- **MEASURED.** Depth adds information additively: stacked chart-Jacobian rank 9, 18, 20, 20 for 1–4 layers.
  **WITHDRAWN (T5):** tying `A_0` across layers does *not* collapse additivity, so a shared seed is not a defence.

**Archive note (`notes/gal_2026-09/multilayer_lora_theory_source.tex`, 7 September 2026, 56 pp) — the truncated
certificate and trained families.** All its numbered positive results are claimed PROVED with full proofs; it
contains no experiments.

- Thm 2.1 = the single-layer theorem with scalar decay allowed, hypothesis "all update inputs lie in `U`" (already
  the subspace form); Cor 3.1 = the repo's Prop. A (exact over the training span), with the caveat that
  "arbitrarily small new directions can enlarge this span" but **no growth law** — the `N' = N·T` regularity is the
  repo's measurement, and equality is unproved.
- Thm 3.4: `‖C̃H^0‖ ≤ a‖H^0‖ + (z/b)‖P A_T H^0‖` and `‖C̃ − C°‖ ≤ a + (z/b)‖A_T‖`, valid under joint training with
  arbitrary backpropagated errors; generally first order in the seed-visible drift.
- Prop 4.2 / Thm 5.3 / Thm 5.4: for a two-layer softplus (and, with a width bound, ReLU) MLP the drift is
  *derived*, not assumed: within a **short-total-step-size window** `τ ≤ min{ρ/(2g), γ/(4K)}` the certificate error
  is `O(τ)` and local identification is stable. The window's length "may depend on the data, labels, chart, and
  initialization" (line 627) — this is the honest limitation, and the repo's measured regime (drift up to 923%) lies
  far outside any such window, which is why the repo's exactness result does not need it.
- Thm 8.2: one structured family (polynomial stem + identity-initialised residual blocks) does achieve `O(τ²)`,
  by the range condition `col(∇_{B_l}L(θ_0)) ⊆ U` — first-order forward drift stays inside the span.
- Thm 10.2: the generic stacked rank is a min–max over subsets of the layers' *allowable covector* spaces, under a
  **density** hypothesis — independence across layers is sufficient, not necessary. Prop 10.3: trained certificates
  inherit the ideal margin only through `σ_min ≥ μ − Δ_J`, with the warning that a new small singular value can be a
  perturbation artefact.
- PROVED cap worth carrying: at a softmax logit layer `1^T D_t = 0` forces `rank B_T ≤ m − 1` (line 216).

**Conflicts between the two, recorded and unresolved here** (flagged for the approver, not settled by this session):

1. Repo `theory/T4` C5 says a cancellation making the truncation error quadratic was "searched, not found"; the
   archive's Thm 8.2 and App A.4 exhibit a *structural* such mechanism (the same in-span mechanism as T3.2). The
   theorems do not conflict; C5's wording is too strong.
2. `experiments/multilayer_cert/RESULTS.md` §3 attributes the `rank B_T < N'` rows to the last layer; a
   re-aggregation of `results/multilayer_cert/survival_692603.jsonl` puts 36 of 64 such rows at hidden layers, and
   the last-layer rows are additionally capped by the softmax bound above. Needs a row-level recheck before either
   number is quoted.
3. `theory/T5`'s assumption table still lists R2 ("violated by a shared seed") although the defence was withdrawn;
   the archive's density-only hypothesis predicts the measured shared-seed result.
4. Three hand-computed expansions in the archive (Ex 3.5, Ex 5.5, §8.6 — coefficients `−3/8`, `1/16`, `−1/3`) are
   marked PROVED and have **never** been checked numerically in this repo; the repo's T3 check validates a
   different instance.

---

## 6. Replay (exact training simulation) vs the certificate

- **The certificate route** needs only the release, the public model and a chart; it is separable per image.
  **Replay** simulates the known recipe and is strictly stronger in information, but needs the recipe.
- **MEASURED, and the sharpest single fact in the track.** On one release and one chart, the recipe-free
  certificate landed 0 of 60 while replay recovered all eight images from 19 of 60 starts, with zero aliases
  (2026-09-06 results bundle). So identifiability is a property of the release, the chart **and the route**.
- **MEASURED (jobs 350928/350940, arms 670990/670993/675031).** Free-feature replay: with a *known* seed the
  truth is locally isolated (nullity 0); with a *free* seed a 232-dimensional family reproduces the release exactly
  (nullity 232), and all 232 directions move `H` while none moves `H` with the seed fixed. Constraining `H` to a
  `k = 12` chart takes the seed-free nullity to 0 — **but the chart measured is the release's own generating chart,
  an oracle chart, not attacker-available.** So this establishes the mechanism (a chart containing the private
  representations restores identifiability), not that a public chart achieves it.
- **WITHDRAWN (approver, A13/A14).** The earlier derivation of a 32-dimensional seed-free family, and the claim
  that the reduced parametrisation removes it: measured nullity is 232 in **both** arms, so the reduction is a
  conditioning gain, not an identifiability gain.
- **DEFECT, open (A14).** The E1B harness fits the product `A@H` (192 equations) when the release contains `A_T` in
  full (1536). Using the full factor drops the nullity 1704 → 232 unreduced and 681 → 232 reduced. Because nullity
  is positive in every measured cell, the truth is **not** locally isolated in either arm, and recovery of `H` from
  that release is an *information* problem, not a solver one. The refit is the first job in that lane.
- **PROPOSED (not run).** Chaining the two: generate candidates with the separable certificate, then use the
  representer's independence clause to select a spanning subset, or replay on the certificate fibre
  (`notes/gal_2026-09/B_factor_representer_idea.md`; chart note Proposal 4.3, which notes it adds computation, not
  information, to an exact replay problem that already implies the certificate).

---

## 7. Text extension — what is proved and what is an obstruction

**Source:** `notes/gal_2026-09/Text_Certificate_Chart_Research_Plan.md` (15 September 2026). Its only computation is
a small Gaussian-dictionary example (§10); **no transformer reconstruction experiment has been run.**

- **PROVED (Prop 1).** The frozen-input endpoint certificate transfers verbatim to text: `C = P A_0`, `C H = 0`,
  with `H` collecting the frozen features that can contribute at the chosen adapter and `p = dim(A_0 col H)`. The
  proof needs no linear downstream model, no constant gradient, no single step, and no knowledge of the
  initialisation; a transformer downstream of the adapter is allowed.
- **PROVED (Prop 2, finite codebook separation).** For a finite candidate set `F` fixed independently of `A_0`, with
  `q < r ≤ d` and Gaussian `A_0`: almost surely `{h ∈ F : C h = 0} = F ∩ U`. So scanning a finite codebook recovers
  the true features exactly **provided no extra codeword lies in the private span** — a substantive condition that
  randomness does not remove. Quantitative version: `r‖Ch‖²/dist(h,U)² ~ χ²_c` with `c = r − q`, giving a union
  bound over `M` false candidates.
- **PROVED (Prop 3, sparse support identifiability).** With `A_0` **known**, `(A_T − A_0)^⊤ = E X` for the public
  vocabulary dictionary `E`; if `X` has at most `s` nonzero rows and every `2s` columns of `E` are independent, `X`
  is the unique such solution. Requires an extra non-cancellation assumption to recover *every* contributing token.
  This route can work **after the nullspace certificate is empty**.
- **PROVED (a concrete excitation family).** For a frozen transformer with `K` independent contextual features and
  `K` *distinct* target vocabulary ids, `K < V`, `K < r`: the restricted gradient matrix is `P_S − I_K` with
  column sums below one, so it is invertible and `rank B_1 = K`; small-step continuity extends this to `rank B_T ≥ K`
  for fixed `T`. Not a uniform long-training or finite-precision guarantee.
- **THE MAIN OBSTRUCTION, stated as such.** For contextual token features one document can supply many independent
  columns; if `q` reaches `r` and `B_T` has rank `r`, then `C ≡ 0`. **A small semantic chart does not restore the
  lost nullspace.** Also: a first-block token codebook gives support, not order or multiplicity; a target token that
  never enters a contributing input feature is invisible to `CH`; a binary head can have a severe output-gradient
  rank restriction (`m ≥ p` is required).
- **PROPOSED.** Discrete charts (finite slot families, template + nuisance decomposition), causal prefix pruning
  (valid at a frozen causal feature map, not for a bidirectional encoder), LM-ordered search with the warning that
  a beam cutoff loses completeness. **Do not reuse the smooth image-chart theorem on hard strings**: a continuous
  map from a connected latent region to a discrete set is constant, so a soft-embedding Jacobian proves something
  about the proxy, not exact token recovery.

---

## 8. Open problems (consolidated)

1. **Does a public chart ever contain the truth?** Measured public-chart projection error is 0.24–0.32 against a
   landing gate of 0.0124 (oracle ladder), i.e. the public charts tested do *not* contain the private images; every
   image success so far is a recovery of the **chart projection** of the private image. This is the single largest
   gap between the mechanism and a privacy claim.
2. **The chart must meet the private span only at the private points.** Unproved for any natural-image family;
   proved only for the synthetic polynomial/quadratic families. The blend degeneracy is otherwise exact.
3. **Changing inputs at depth**: the full certificate's rank lifetime `T < min(r, n_l)/N` rests on the measured
   `N' = N·T` law, which is unproved and is flagged as the claim most likely to change on a trained backbone. No
   reconstruction with an adapter *inside* a convolutional stack has been run.
4. **`N'`, `q` and excitation are not attacker-observable.** Only lower bounds are.
5. **Replay identifiability with a free seed** is negative as measured (nullity 232); whether the full-factor refit
   changes that is the open question in that lane.
6. **Text**: whether a realistic setting keeps `q < r` at a contributing input, and whether the known-initialisation
   sparse route survives learned embeddings and finite-precision storage.
7. **Adam, dropout, weight decay, merged releases** are theorem boundaries, not small perturbations.
8. **Novelty** relative to Oz et al. 2024, Yao 2024, GIAS/GIFD, FineXtract, DAGER/TIGER/SOMP/UTR is not
   established; the 2026-09-14 audit lists the closest work and says so explicitly.
