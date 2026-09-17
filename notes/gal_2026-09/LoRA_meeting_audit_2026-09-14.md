# Audit of the LoRA briefing and personal companion

14 September 2026. The current deliverables are a three-page briefing and a twelve-page personal companion. The source and mathematical audit is retained below; earlier version-specific descriptions are historical.

The core argument is sound under its stated assumptions. The main weaknesses were how evidence was classified, how figure success was described, and two qualifications that were broader than their sources. The meeting story now starts from image controls: parameterize plausible candidates, solve training-derived equations within that family, show the working LoRA case, and ask when the family removes ambiguity or makes search practical. Novelty is a question to establish first.


## Preferred figure adopted

The user-supplied lora_brief_v2.pdf provides a clearer small-input expansion, image grid, full-network/LoRA branches, and Haim/Oz search-location comparison. Its vector diagram replaces the box diagram on page 1 of both PDFs. Exact-equation wording is conditional, and the caption distinguishes the head-only experiments from the multilayer extension. Only the figure was imported: the source's stronger global-identification and observability claims were not adopted. The first-page text was adjusted to fit; every subsequent page is pixel-identical to the preceding delivered version.

## Current rewrite: explain the strategy before the details

The latest change is a structural rewrite, not an appended audit. Both documents now begin with the same idea: use a small input to generate the whole image, then test that candidate using equations from the released model. The public image family is the meeting's main question; LoRA is the worked case with exact finite-step equations and reconstruction evidence.

The new figure explicitly shows k controls -> public decoder -> d pixel values -> full network (frozen encoder and LoRA head) -> predicted label, with k much smaller than d. Its note distinguishes original fine-tuning from reconstruction: the attack varies the image controls while keeping the released model fixed, and is guided by compatibility equations rather than the label alone.

The brief has three pages: idea and letters; equation intuition and CIFAR/Fashion; stronger families, MNIST and team questions. The companion has four story pages followed by Notes A-H: certificate, geometry, score quality, NTK, deep tests, experiment design, optional endpoint matching, and prior work/extensions. It is about half the previous prose length.

Actual originals and the NTK row are retained. Fashion's four target matches and two useful approximations appear together. The original truth/PCA/NTK comparison was useful; the mistake was the earlier revision's substitution of a PCA representation for the raw original. No source image payload was modified. Apples and experimental curves remain outside the PDFs.

The main questions are which stronger public family to use, when it excludes wrong solutions, how to search it, what the team already tried, and which second reconstruction equation would test generality. SimuDy remains a comparator or beneficiary, with less training information, better quality or lower cost as possible gains. No new inversion experiment or matched improvement is claimed.

Visual review covered every final PDF page. The architecture diagram is also supplied as a standalone PNG, vector PDF and editable TikZ source. The packet README gives the current page map; the technical entries below preserve the substantive audit history.

## Audit of the latest pasted feedback

The useful core is retained: distinguish the public image family from free embedding reconstruction; lead the novelty claim with the factor equations and their identification conditions; make coverage the main forward question; and explain the one-step NTK issue structurally. The confident wording of the feedback should not be pasted without the following corrections.

| Feedback claim | Audited treatment |
|---|---|
| Oz did embedding reconstruction, then decoding. | Correct for the published pipeline. Its reconstruction equations involve the trained nonlinear head, however; they are not merely the free bilinear LoRA span equation. The comparison concerns where image compatibility is enforced. |
| The LoRA seed is unknown even at one step. | False for simultaneous SGD with zero B: A₁=A₀, so original released factors reveal it at T=1. At later times the certificate avoids needing it. |
| The first-trained-layer span trick works on LoRA captions but breaks on images. | The factor-coordinate issue applies to either modality. FineXtract's ordinary-layer span derivation is useful prior art; its Appendix A.3 acknowledges approximate departures for Adam and adapters. Its main image extraction method is guided sampling plus clustering. |
| k<p plus tangent separation guarantees exactly the private images globally. | Missing global span separation. Tangent separation gives local control; the global theorem removes extra zeros outside the private span. Wrong images whose features are already inside that span survive every seed. |
| A rank-r representer determines only the training images' span. | Qualify the object: a free-coefficient equation determines compatibility with a recorded sketched span under the rank conditions. Realizable image candidates, their independence and their actual coefficients matter. This is not a statement about all NTK methods or the full release. |
| Nonlinearity between family and adapter eliminates blends. | Nonlinearity can remove affine mixtures, but the actual family must satisfy the separation conditions. It is not sufficient by itself. |
| Every on-family private input is exactly recovered. | Coverage, identifiability and solver success are separate. The record reports all eight projected letters and motorcycles, but only four of eight Fashion target matches, with additional recognizable approximations. |
| Joint fits stalled because they optimized N(k+m) unknowns. | The original walkthrough already eliminates coefficients by least squares. Do not use that variable-count story as the diagnosis of those runs. The displayed letter panel also lacks a T label. |
| Only show any NTK comparison after the four-arm T=1 audit. | Keep the supplied comparison with its provenance and limited claim. It does not establish which mechanism caused the gap. The proposed coordinate audit is separate from the equations-by-image-family ablation. Neither full four-arm study is claimed complete. |
| GIAS/GIFD only regularize a determined, ill-conditioned problem. | Too broad. GIAS explicitly motivates generators as a way to reduce underdetermination. Its Property 1 assumes an original unique minimizer; that assumption cannot characterize the whole method. |
| Three of four assumptions are realistic, leaving only Adam. | State the actual setting. Fixed feature spans and original factor access matter. The displayed 11-output head with r=64 is algebraically valid, but its factors exceed the directly trained head's parameter count. Standard-model deployment remains a separate test. |
| All zero-initialized branches inherit the certificate. | Only a gradient-span inclusion is immediate. Excitation, accessible candidate features, a sufficiently narrow trajectory span and the optimizer still need proof. Note H now states these conditions. |
| Same-data averages of CⱼᵀCⱼ converge to the complement projector. | With independent isotropic Gaussian seeds, fixed shared features, full recording almost surely and the exact no-decay invariant, the expectation is (r−q)σ² times that projector. Arbitrary seeds or selected/truncated releases do not inherit the identity automatically. |
| No one has published LoRA-factor reconstruction. | Not established as a broad claim. Released-LoRA privacy and extraction work exists, and 2026 PEFT-gradient work adds relevant neighbors. No reviewed source was found with this same conserved-subspace endpoint-factor certificate; priority remains a question. |
| A public repository README is stale. | The pasted feedback does not identify the repository URL, and the workspace contains no Git checkout. No public README was verified or changed. The handoff README now reflects the audited story. |

### A concrete counterexample to the overstrong chart claim

Let the only private feature be e₁ in R³, and let C retain coordinates two and three. Then p=2. Consider the fixed, injective, smooth one-parameter feature family

ψ(t) = (1+t, t(t−1), t²(t−1)).

The true point t=0 has tangent (1,−1,0), which is disjoint from span(e₁), and k=1<p=2. Nevertheless, the wrong point t=1 also passes, because ψ(1)=2e₁ lies in the private span. The random kernel is not the problem: when r=n, ker C is exactly that private span. This is why the global span-separation condition must stay in both documents.

### What the additional primary-source check found

- [FineXtract, ICML 2025](https://proceedings.mlr.press/v267/wu25as.html): the published paper's main method is guided image generation and clustering. Appendix A derives an ordinary linear-layer caption relation, then uses approximate PCA-based processing for departures; A.5 reports DreamBooth caption experiments. The public PDF was read, including A.1–A.5.
- [DSiRe, 2024](https://arxiv.org/abs/2406.19395): estimates dataset size from LoRA norms and spectra.
- [UTR, January 2026](https://arxiv.org/html/2601.17533v1): its §3.1 scope is nonlinear bottleneck adapters, with other PEFT architectures excluded; §3.2 observes shared training gradients. Calling this a LoRA-factor endpoint attack is incorrect.
- [FedSpy-LLM, April 2026](https://arxiv.org/html/2604.06297v1): §V-C evaluates SLoRA and FedAdapter; its observation is gradients. Its broad gradient-span equality also needs excitation/rank conditions before borrowing it as an exact theorem.
- [Hu et al., January 2026](https://arxiv.org/html/2601.21719v1): analyzes Wishart projection privacy, including membership leakage and LoRA-FA. Equation 1 already displays the right-hand AᵀA sketch. That sketch identity itself is not the new contribution.
- [PEFT gradient inversion, CVPR 2025](https://arxiv.org/abs/2506.04453): uses maliciously designed models/adapters and shared adapter gradients. It is a different threat model.
- [PRISM, 2026](https://arxiv.org/html/2606.00944v1): related gauge-aware differential-privacy work, not evidence of the same endpoint reconstruction certificate.

These are additions found in the new pass, not a claim that they were all published after our previous conversation. The current-date search also used combinations of LoRA, released weights, training-data reconstruction, factors, certificate and invariant. Search coverage does not justify a categorical absence claim.

### The story to narrate

“I started from your reconstruction work. LoRA changes the parameter relations, and I derived an exact finite-step test from the released factors. Passing that test alone does not prove image identity. I therefore restrict the candidates to a public image family inside the objective; PCA already gives these results. I want to understand which stronger families help, what can be proved about the surviving images, and whether you have tried this placement of the prior.”

Start with the letters, retaining original / PCA training target / NTK / certificate. Show motorcycles and Fashion as the broader positive evidence. The source's Fashion approximations remain worthwhile outputs; they are not relabelled as exact target recovery. MNIST has its separate coverage role. Apples and experimental diagnostic plots remain out of the meeting PDFs.

The next study has three linked questions: can a stronger public family cover raw targets; can the equations distinguish those targets from plausible alternatives; can adaptive search find and select them economically? SimuDy remains a comparator or potential beneficiary, with weaker-information, better-recovery and lower-cost outcomes to test. It is not the central method to reproduce.

### What changed in the files

That earlier audit corrected the structural argument, NTK claims and literature scope. The current rewrite replaces its long section sequence with the three-page brief and the four-page story plus eight reference notes described above. The original images and the distinction between one-step linearization, uniqueness and solver behavior are preserved.

No new inversion experiments were run. The update consists of source verification, analytic checks, prose and mathematical corrections, and rendered-document quality assurance. The public-repository claim remains unverified.

## Chart-first audit: the proposed story

The opening is now a small set of controls that generates an image. PCA is the worked image parameterization; LoRA supplies the exact finite-step residual. The documents distinguish the two changes and treat the broader contribution as a question to settle with the team.

| Proposed statement | Audit and replacement |
|---|---|
| “Your work did pixels; mine does latents.” | Too broad. The team's transfer-learning work reconstructs embeddings and uses image priors. Ask whether they also constrained candidates through a generator inside the parameter-reconstruction objective. |
| “PCA decodes the latent space of model features.” | Not the displayed experiments. Public PCA decodes image coordinates; the base model then computes features. A feature-space family with a separate image decoder is another design and needs image-feature consistency. |
| “The letters show that PCA beats pixel search.” | Both displayed methods use PCA. They support the tested certificate formulation within that family. A full equations-by-parameterization comparison remains to run. |
| “NTK failed because small-time linearization is wrong.” | False at one simultaneous head step. Parameter linearization is exact; the merged update contains a right-hand seed sketch. The finite head kernel lacks the automatic universal-kernel guarantee. Failed searches are not a general impossibility proof. |
| “Exact equations make reconstruction realistic and exact.” | Validity holds under fixed-feature/SGD/excitation/release assumptions. Fidelity, identification, coverage and search remain separate. Current image figures train on PCA projections. |
| “Adaptive local charts are new.” | Generator adaptation and progressive feature-domain search already exist. A distinct patch-selection rule needs a new guarantee or a fair measured advantage. |
| “The generalization is to replace pixels by g(z).” | That composition is standard. New work must verify the residual, control nuisance variables, and establish identification or search gains for the new observation model. |
| “A low score verifies a private image.” | Only if the family's wrong-image score floor exceeds the acceptance threshold, with equation error accounted for. This is not yet proved for rich natural-image families. |

### Closest prior work

- [Oz et al. (2024), §3](https://arxiv.org/html/2407.15845v1): embedding reconstruction followed by DIP or a conditioned decoder. Ask about joint variants.
- [Jeon et al. (2021), GIAS §4.1](https://arxiv.org/abs/2110.14962): latent search followed by generator-weight adaptation. Property 1 assumes a unique minimizer of the original inversion cost; it does not show that a family removes pre-existing ambiguity.
- [Fang et al. (2023), GIFD](https://arxiv.org/abs/2308.04699): progressive intermediate-feature optimization with constrained regions; a close richer-search comparator.
- [Bora et al. (2017)](https://proceedings.mlr.press/v70/bora17a.html): generative compressed sensing already studies recovery from few measurements. LoRA's annihilator depends on the private data and vanishes on their span; the independent-measurement theorem is not automatic here.
- [Yao (2024)](https://arxiv.org/html/2409.08482v1): private-identity reconstruction from released diffusion-LoRA weights is prior work.
- [Peng et al. (2025)](https://arxiv.org/html/2509.20177v1), [DAGER (2024)](https://arxiv.org/abs/2405.15586), and [VGIA (2026)](https://arxiv.org/html/2604.15063v1): respectively generator-tangent geometry, low-rank gradient tests for text, and verified isolation under crafted gradient queries.

These are close overlaps found in a targeted primary-source audit, not an exhaustive priority certificate. The remaining candidate contribution is specific: the finite-step LoRA invariant, its interaction with image-family geometry, and an attack-specific guarantee or measured search gain.

### The mathematical question

For a smooth residual with p independent equations on d pixel coordinates, a regular zero with p<d has a local family of alternatives. An immersed k-coordinate image family can isolate the target when the composed derivative has rank k. This is the ordinary local-inverse mechanism; useful application requires proving the conditions for the actual residual and family. The LoRA rank formula makes its private-span obstruction explicit.

If a computable score differs uniformly from an ideal score by at most δ, acceptance at τ is sound when the ideal wrong-image floor exceeds τ+δ. This is an elementary implication. Bounding the floor and δ, preserving true-image acceptance and coverage, and giving a search method are the substantive tasks. For batch residuals, retain the full batch and nuisance variables; the per-image LoRA factorization does not transfer automatically.

### The comparison needed to support the story

Keep release, observations, labels, selection rule and budget fixed. Compare joint fit versus certificate, each with free pixels versus the same public family. Then compare full generator controls, fixed directions, adaptive patches and appropriate existing generator-search methods. Fit known targets to the family only as an evaluation diagnostic. Include wrong releases and same-class alternatives; use caption-only / adapter-only / both when conditioning, with subject holdout.

No new training experiments were run. The positive letters, CIFAR and Fashion evidence remains. MNIST is a representation diagnostic from a separate replay study, not evidence that all ablation cells have been measured.

## Contribution and positioning revision

The main research program is now **finite-step LoRA constraints plus identifiable, searchable image charts**. The three gains to test are comparable reconstruction with less training information, better reconstruction at matched resources, or reduced cost for an existing attack. Simulation is a comparator or possible recipient of the new components. The seed and capacity derivations are supporting Appendices A–B, not the forward plan.

The original letters panel correctly labelled its first row as the public PCA projection. Keeping that reference was useful: it separated chart approximation from inversion error. The redesign had lost this distinction by removing the PCA row when adding raw originals. Both PDFs now show **original / PCA training target / NTK output / certificate output**, with four verified A columns. No experiment or original T image was invented.

The prior-work statements were checked against [SimuDy, Algorithm 1, §4.3 and discussion/Table 3](https://proceedings.iclr.cc/paper_files/paper/2025/file/945c4b5d6daf5294befbc5b7c275c300-Paper-Conference.pdf), [Haim et al.](https://arxiv.org/abs/2206.07758), [Oz et al., §§3 and 5](https://arxiv.org/html/2407.15845v1), and [Loo et al., Theorem 1](https://arxiv.org/html/2302.01428v2). The documents distinguish an information advantage from a measured quality or speed advantage. No head-LoRA figure is presented as a matched SimuDy comparison.

The wrong-image floor is now in the brief, and the companion explicitly proves the implication: if an allowed candidate has score at most τ and γ_X(ε)>τ, it is within ε of a training image. The score must be defined, ε>0 and τ≥0. This does not establish coverage, successful search or a computable natural-image gap; those remain separate questions. The polynomial example supplies a worked restricted case.

The full mathematical record below is retained. Formula labels, section references, normalization conditions, figure purposes and source scope were rechecked after moving sections. The companion's main discussion ends at §18; Appendices A–B follow.

## Substantive corrections

1. **The three-chart MNIST panel is replay evidence.** Its source places it in the exact-channel capacity study. With eight examples and adapter width 16, its 17 chart coordinates exceed the certificate's eight directions. Both PDFs now identify the method as exact replay. It illustrates chart-dependent fidelity; it is not another certificate-only success or an established comparison on one shared release.

2. **The original image is different from an on-chart training input.** Every figure retains the actual supplied originals. The letters use four aligned rows; the other main panels use original/output pairs. The captions say when training used a PCA projection. “Found” then means the projection was recovered; it does not mean the original photograph's missing detail was recovered. Fashion-MNIST distinguishes target matches from recognizable approximate reconstructions; both are positive visual evidence. The strict count remains 4/8. Keyboards are archived. Ground-truth matching is an evaluation procedure, not an attacker selection rule.

3. **Selecting a public patch does not invalidate the fixed-chart theorem.** A countable atlas fixed independently of the seed inherits the almost-sure result. The release may select among those patches. Recomputing their directions from the release needs another argument. Moving through many small patches also does not make the entire reachable image family low-dimensional.

4. **Heavy drift did not leave useful discrimination against nearby perturbations.** The full recorded span fills the adapter and makes the full projector zero. The truncated test retained some separation from fresh inputs, but not meaningful separation from the tested near-target perturbations. Both documents now distinguish these outcomes.

5. **NTK is a valid comparison, not a failed principle.** At one simultaneous head step, A remains at its seed and the parameter linearization is exact. The merged update contains the right-hand sketch A₀ᵀA₀; the head kernel uses A₀φ(x). Parameter linearity does not make these features linear in the image, and a sum of features is not itself an ambiguity proof. Loo et al.'s guarantee uses a universal limiting kernel, with additional hypotheses; it does not transfer automatically to this finite head kernel. See [Loo et al., Theorem 1 and Appendices F–G](https://arxiv.org/html/2302.01428v2).

6. **The reported NTK-style baseline was stronger than a naive joint fit.** The original walkthrough, pp. 13–14, reports matched release/chart/starts/budget, eliminates coefficients by least squares, and includes successful single-image cases. These facts are preserved. The displayed letter panel does not specify its step count. The source supports a reported solver comparison, not an independently reproduced benchmark or an impossibility theorem. The joint rank condition also excludes duplicate candidates, an advantage separate per-image tests do not automatically have.

7. **The positive evidence leads the presentation.** Letters, natural CIFAR motorcycles and Fashion-MNIST bags carry the working case. The brief shows all four Fashion target matches plus two recognizable approximate outputs; the companion introduces the matches and later discusses the approximations. An output can preserve useful shape and structure without meeting the projected-target recovery criterion. Apples and the redundant keyboard grid are removed from the meeting PDFs, with their source assets and distinct claims preserved in the archive. MNIST retains the separate chart-representation role. See `audit/figure_roles.md` for the private rationale behind every panel.

## What the math audit supports

| Claim | Audit conclusion and operative conditions |
|---|---|
| Preserved subspace and certificate | The two gradient products preserve BₜP = 0 and PAₜ = PA₀ under simultaneous scalar SGD, fixed inputs, B₀ = 0, and no regularization/decay. With independent initialization having a density, q < r ≤ n, and rank Bₜ = q, the observable C annihilates the training features and has rank r − q. Full row rank of Aₜ is unnecessary. |
| Kernel decomposition | ker C = col H ⊕ ker A₀ follows from seed injectivity on the private feature span. The certificate accepts that entire span, so validity alone is not individual membership verification. |
| Local chart rank | At a fixed immersed feature chart independent of the seed, rank(CDψ) = min{r − q, k − dim(T ∩ col H)} almost surely. A full derivative gives local stability. A deficient derivative at one point alone does not prove a continuum of exact solutions. |
| Global fixed-chart cases | The k < p / k = p / k > p conclusions apply away from the private feature span, for the stated fixed smooth chart. The chart must separately intersect the private span only at the desired points. Rich natural-image identification is still open. |
| Polynomial example | The row (ab, −a−b, 1) applied to (1,t,t²) gives (t−a)(t−b). The small rank-one example may have k = p = 1; it illustrates span separation rather than the strict k < p generic theorem. The layer must actually receive these features. |
| Wrong-image score floor and local bound | The separation target and nesting inequality are correct. The polynomial toy has an explicit positive bound on the specified compact domain. The local 2/α estimate requires a full derivative and a defined denominator; approximate zeros add their reference residual. Sensitivity must be compared at consistent image scales. |
| Deep certificate | The common span must include training-time and base-model features, with excitation of its seeded image. The seeded dimension is distinct from the input-span dimension unless injectivity is justified. The trajectory can depend on the seed, so fixed-span independence cannot be assumed. |
| Perturbation and stacking | The leakage/signal bound a + (z/b)‖Aₜ‖ and the stacked-rank identity are correct under their stated conditions. A spectral gap alone does not prove alignment. The odd tanh example's sign alias prevents a global-identification claim despite full local rank. |
| Concrete two-layer theorem | The supplied softplus theorem proves excitation, certificate accuracy and local stability under Gaussian initialization, sufficient widths, a fixed immersed chart, squared loss, and sufficiently short joint full-batch training. ReLU additionally needs the stated width and activation conditions. This is not a guarantee for the pretrained softmax image experiments. |
| Seed reduction and replay | X = A₀U has rN entries. The coefficient trajectory sees it through an N×N Gram matrix; matching original factors still needs its orientation. Completing the orthogonal seed block from Aₜ makes a zero of the reduced residual a match of both full factors. Reduction makes the search smaller, not necessarily unique or easy. |
| Replay capacity | The conditional regular local bound is k ≤ (m−1)+(r−N), equivalently k < m+r−N, for the fixed-feature softmax model. A8 ensures the remaining ambiguity changes data rather than only the seed. Numerical rank checks support sharpness in tested cells; they are not exact-rank proofs. The count is not a universal information-theoretic ceiling for every prior or training rule. |
| Gauge and learned priors | BA, AᵀA and BBᵀ are invariant under a common orthogonal factor rotation. Different seeds need not be such rotations. The distance experiment motivates an invariant baseline; it does not establish that every raw-factor model must fail. A prior narrows the family without adding another observation. |

## Figure provenance and page guide

| Images | Source and comparison | Companion / brief |
|---|---|---|
| Figure 1: four A examples | Four aligned rows: original, PCA training target, NTK output, certificate output. Raw A's come from board guide p. 3, matched through the PCA glyphs to figures_for_gal p. 4. Original dataset indices are unavailable; polarity is preserved. | p. 6 / p. 1 |
| Figure 2: CIFAR-100 motorcycles | All eight supplied originals beside certificate outputs. All eight public-PCA training inputs are reported recovered. Positive evidence beyond the letter example. | p. 6 / p. 2 |
| Figure 3: Fashion-MNIST | All four reported target matches (source columns 2, 3, 6, 7). The brief also shows two recognizable approximations (columns 4, 5), labelled separately; the full-run target count is 4/8. | p. 6 / p. 2 |
| Figure 4: MNIST | Six supplied originals beside chart-specific replay outputs. Separate chart-reference rows in the companion distinguish representation from search error. Not established as one shared release. | p. 7 / p. 3 |
| Figure 5: approximate Fashion outputs | Source columns 4 and 5 from the same run, beside their originals. Recognizable structure is credited; improving fidelity and reaching more projected targets remain the questions. | p. 17 / included in Figure 3 |
| Apples, archived | Valid fixed-raw-release chart comparison. Removed for presentation clarity; MNIST does not inherit its same-release control. | Not displayed |
| Keyboards, archived | Supplied CNN grid reports 6/8 projected targets, distinct from historical 7–8/8 class summaries. Redundant with the retained positive examples for this meeting. | Not displayed |

No generated or online look-alike image was substituted for experimental ground truth. Diagnostic plots remain outside the meeting PDFs. The archive retains original assets and provenance.

## Reading roles and questions for the team

The brief contains the claims and images needed for narration. The companion supplies the derivations, the model/class construction, conditions, and deeper alternatives. Section 18 connects the three possible advances to concrete comparisons and questions. Supporting seed/capacity derivations follow in Appendices A–B.

The most useful chart question is: “Your paper uses DIP or a conditioned decoder to turn estimated features into images. Did you also optimize those image parameters directly against reconstruction constraints? Which parameters did you allow, how did you initialize them, and what limited the result?” This acknowledges what their paper already does. See [Oz et al., Sections 3.2 and 5](https://arxiv.org/html/2407.15845v1).

For the certificate: “Which realistic setup preserves the invariant? Can we justify that B records every training direction, and how should we choose its numerical cutoff?” For equation quality: “Does this extra constraint retain the true images and reject wrong candidates the first constraint accepted?” For selection: “How would we tell a repeatedly found wrong mode from a training image?”

The controls remain caption only / adapter only / both, holding out whole subjects. Public, target-conditioned and shared-concept families remain distinct proposals. GAN controls, a diffusion decoder and DIP are concrete families to test, not established recovery guarantees.

## What remains unverified

The supplied reports do not expose the original image-experiment training code, complete class-head initialization/freezing choices, or all experiment logs. The added-class explanation therefore remains a conceptual implementation, not a claim to have inspected that code. The bundle explicitly marks its numerical results provisional pending reproduction in the project repository. Previously reproduced synthetic checks were inspected; no new training experiment was run for this audit.

Historical class cells and the F1_blend claim remain in the source record, but the missing original plots were not invented or treated as newly validated evidence. The rich-generator separation theorem, efficient global search, useful long-training deep certificates, and robust attacker initialization remain open.

The revised PDFs were rendered and inspected; the four-row letter reference, original/output pairings, target-match/approximation labels, contribution framing, figure roles, titles, references, page breaks and the distinction between original images and projected training targets were checked. The short brief remains four pages and the companion twenty-two.
