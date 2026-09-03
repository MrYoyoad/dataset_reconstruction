# Exact inversion of the LoRA training map — a tool, and the law it obeys

*Yoad Oxman · positioning note for supervision · 2026-09-03 · numbers †provisional (synthetic FP64 testbed + one real-MNIST confirmation); authoritative source `experiments/exact_inversion/RESULTS.md`.*

**One line.** The released LoRA factors are a *deterministic function of the private data under the known public recipe*, so we invert that function directly — simulate the recipe on candidate data, backprop through the unrolled training loop, and solve for the images. This is a method with an exact solve and a **derived-and-measured capacity law**, not a correlation.

## 1 — The tool
Fine-tuning is a differentiable map `(A_T, B_T) = F(recipe; {x_i}, A₀)`. The attacker holds `(A_T, B_T)`, the base model, and the recipe; **not** `A₀`, the activations, or the trajectory. Because `B₀=0`, the release depends on `A₀` only through the `rN` numbers `X = A₀U`, so the unknowns are just the data `{x_i}` and `X`. We reconstruct by minimising `‖F(recipe; {x̂_i}, X̂) − (A_T,B_T)‖²` with Levenberg–Marquardt (autograd Jacobian through the SGD unroll).

- **The simulator *is* the training map.** Fed the true data and true `X`, it reproduces the actual release to `5.3×10⁻¹⁶ – 1.8×10⁻¹⁵` (both factors, 54 rows — the FP64 floor). Every downstream claim rests on this one number.
- **It inverts.** From a 10 % start it recovers every image to `~10⁻¹⁵` with the release reproduced to `~10⁻³⁰`, across the tested grid (49/49 cells).
- **The recipe need not be assumed — it is recoverable and self-checking.** A wrong recipe cannot reach the residual floor (a 1 % learning-rate error leaves residual `9×10⁻⁷` vs `5×10⁻³¹`), and the per-step recipe scalars are probeable from the attacker's *own* data. The residual is the attacker's own instrument.

## 2 — The law (this is the theorem-shaped result to lead with)
`B_T = P_T Xᵀ` is `m×r` but has rank `N`, so it carries `N(m+r−N)` independent numbers — a **per-image budget of `m + r − N`**:

> **A private image is exactly recoverable iff its degrees of freedom `k < m + r − N`.**

1. **Derived (strict).** Under softmax cross-entropy the error columns sum to zero → `1ᵀB_T = 0`, putting `B_T` in `1^⊥⊗ℝ^r`; the rank-`N` manifold there has a deficit of exactly `N`, giving `k < m+r−N` (strict). Verified two ways: `‖1ᵀB_T‖/‖B_T‖ ≈ 10⁻¹⁵` on every SGD release, and `rank(B-block) = 216 = N((m−1)+r−N)` exactly.
2. **Measured (sharp).** Confirmed at `N = 4, 8, 12, 14` with predicted thresholds `k* = 32/28/24/22`; each bracketed by its last success and first failure, and **the brackets are disjoint** (`N=12` collapses at `k=26` while `N=4` is healthy at `k=30`) — a fixed-`k` explanation is ruled out. **Sharp to one unit of `k` at `N=8` and `N=14`**, and on **real MNIST at all three ranks** (`r=8/16/32`, last full-rank `k=9/17/33`, first collapsed `10/18/34`). Across that unit `σ_min(J)` at the truth falls **11–12 orders**.
3. **On a *trained* model — the question that matters most here.** The law is not an artefact of random weights: it holds on a **trained** MNIST MLP with LoRA on the head and **unseen test-split digits** (jobs 607896/610020) — full column rank at `k=17`, collapse at `k=18`, exactly `m+r−N`. A trained encoder does **not move the line**; it only makes it harder to *reach* — `σ_min(J)` at the truth is `7×/100×/420×/1100×` smaller than a random encoder at `k=6/10/14/17`, and below-line cells that stall at 80 LM iterations recover at 300. So training is a **conditioning cost, not an identifiability defence** — the *visual* fidelity there is limited by the public chart, not the attack (§3).
4. **It quantifies the channel.** The algebraic *certificate* channel stops at `k < r−N` (†bundle, not reproduced here); simulation reaches `k < m+r−N` — a **factor `(m+r−N)/(r−N) = 3.5×`** more per image at `N=8`, bought by the released head width `m`. **Falsifiable prediction:** widen `m` → reach grows *linearly* at fixed rank.

## 3 — What the law bounds, honestly
- **It bounds the *chart*, not the image — this is the scoping that makes it honest.** The theorems pin the `k` coordinates `w`; the attack returns `ψ(ŵ)`. At the same `k=17`, one unit below the line, three different charts each recover their *own* representable image to `10⁻¹⁴` at the same `σ_min` — but measured against the real digit, PCA and a warped PCA sit at `0.51` while a chart built to contain the digits sits at `5.7×10⁻¹⁴`. **Thirteen orders apart, same law, same collapse point.** So the law bounds the *dimension of the search*, and the search space is the attacker's to choose. That is what gives the generative-prior direction a precise job: **a prior is a chart-builder.**
- **It is a boundary of *exact identifiability*, not of leakage — in both directions.** Below the line the image returned is only as good as the chart. Past it the release is reproduced at the floor by a *different* point, at `0.17–1.7 %` from the truth on the synthetic testbed (whose chart contains the images by construction — the most favourable case) and `2.3–6.8 %` on real digits, where **5 of 8 past-line cells have no image inside tolerance**. Degradation tracks **`N`, the defender's variable**, not distance past the line. Every one of these is a *lower* bound on the fibre's extent — the searches start near the truth and the two continuations were stopped by budget along a single direction of 40 and of 14. **Whether the alternatives stay recognisable was not assessed**, and the trend does not favour it. The defensible statement is a *quantitative* capacity law, not "the adapter hides the data" and not "the alternative is the image."
- **Adam does not defend by non-identifiability.** It destroys the algebraic certificate (`C≡0`), yet the release stays locally identifiable at the truth at every scale tested (full column rank). It buys two *moderate* obstacles at scale — worse solution conditioning (`~400×` at the real work point) and a much smaller basin — and, ironically, breaks `1ᵀB_T=0` and hands the attacker back **one unit** of exact-inversion capacity.

## 4 — The open problem *is* the fundable direction: the initializer
The binding constraint is **the basin, not identifiability**. Measured from both sides:
- Perturbed-truth starts recover up to an **86 %** start error, with one budget-limited failure at 81 % bracketed by successes on both sides (basin is wide along truth-directions; edge not located).
- Attacker-reachable, **release-only** starts reach the residual floor in **none of 20**. One run of the twenty landed inside the 1 % image tolerance but at residual `9×10⁻⁷` with its restart budget exhausted — a search failure that stopped near the truth, not a recovery. The other nineteen ended at residuals `5×10⁻³` to `0.4`.

So the release *determines* the data (identifiability), but a naive attacker cannot *reach* it (basin). **That is exactly the job for a learned / population prior: supply the initializer, not a gradient bridge** — a crisp, well-posed target with a measured gap to close, and the natural home for the "foundation-model era" framing of the thesis.

## 5 — Four weeks to the deadline
1. **The initializer (the crux), which is now also a chart-builder.** A prior trained on public data to emit a release-conditioned start *and* a chart that can represent the private images; success metric = release-only recovery rate at the floor, currently 0 of 20.
2. **Falsify the law's prediction.** Sweep `m` at fixed `(r, N)`; read `σ_min(J)` at the truth for the linear widening.
3. **Adam basin** — is the small basin solver-fixable (trust-region / Gauss–Newton) or intrinsic?
4. **Real-data step** — frozen DINO/CLIP features + an adapter head (scoped as a separate task).

*Scope: synthetic FP64 testbed (one real-MNIST confirmation of the law); this-attacker (known/recoverable recipe, no `A₀`); observe-don't-conclude; the certificate `r−N` row is a †bundle number, not an in-repo measurement. Provenance: jobs 467914, 469120, 479587/479684 (law), 459111 (basin), 408560-63 (initializers), 466915/467622 (Adam), 568095 (real MNIST), 574169 (chart dependence), 607896/610020 (trained model).*
