# Exact inversion of the LoRA training map — a tool, and the law it obeys

*Yoad Oxman · positioning note for supervision · 2026-09-03 · numbers †provisional (synthetic FP64 testbed + one real-MNIST confirmation); authoritative source `experiments/exact_inversion/RESULTS.md`.*

**One line.** The released LoRA factors are a *deterministic function of the private data under the known public recipe*, so we invert that function directly — simulate the recipe on candidate data, backprop through the unrolled training loop, and solve for the images. This is a method with an exact solve and a **derived-and-measured capacity law**, not a correlation.

## 1 — The tool
Fine-tuning is a differentiable map `(A_T, B_T) = F(recipe; {x_i}, A₀)`. The attacker holds `(A_T, B_T)`, the base model, and the recipe; **not** `A₀`, the activations, or the trajectory. Because `B₀=0`, the release depends on `A₀` only through the `rN` numbers `X = A₀U`, so the unknowns are just the data `{x_i}` and `X`. We reconstruct by minimising `‖F(recipe; {x̂_i}, X̂) − (A_T,B_T)‖²` with Levenberg–Marquardt (autograd Jacobian through the SGD unroll).

- **The simulator *is* the training map.** Fed the true data and true `X`, it reproduces the actual release to `fwd_check ≈ 8×10⁻¹⁶` (FP64 floor). Every downstream claim rests on this one number.
- **It inverts.** From a 10 % start it recovers every image to `~10⁻¹⁵` with the release reproduced to `~10⁻³⁰`, across the tested grid (49/49 cells).
- **The recipe need not be assumed — it is recoverable and self-checking.** A wrong recipe cannot reach the residual floor (a 1 % learning-rate error leaves residual `9×10⁻⁷` vs `5×10⁻³¹`), and the per-step recipe scalars are probeable from the attacker's *own* data. The residual is the attacker's own instrument.

## 2 — The law (this is the theorem-shaped result to lead with)
`B_T = P_T Xᵀ` is `m×r` but has rank `N`, so it carries `N(m+r−N)` independent numbers — a **per-image budget of `m + r − N`**:

> **A private image is exactly recoverable iff its degrees of freedom `k < m + r − N`.**

1. **Derived (strict).** Under softmax cross-entropy the error columns sum to zero → `1ᵀB_T = 0`, putting `B_T` in `1^⊥⊗ℝ^r`; the rank-`N` manifold there has a deficit of exactly `N`, giving `k < m+r−N` (strict). Verified two ways: `‖1ᵀB_T‖/‖B_T‖ ≈ 10⁻¹⁵` on every SGD release, and `rank(B-block) = 216 = N((m−1)+r−N)` exactly.
2. **Measured (sharp).** Confirmed at `N = 4, 8, 12` with predicted thresholds `k* = 32/28/24`; each bracketed by its last success and first failure, and **the three brackets are disjoint** (`N=12` collapses at `k=26` while `N=4` is healthy at `k=30`) — a fixed-`k` explanation is ruled out. At the line `σ_min(J)` collapses **13–14 orders in one step**. Holds on **real MNIST at three ranks**.
3. **It quantifies the channel.** The algebraic *certificate* channel stops at `k < r−N` (†bundle, not reproduced here); simulation reaches `k < m+r−N` — a **factor `(m+r−N)/(r−N) = 3.5×`** more per image at `N=8`, bought by the released head width `m`. **Falsifiable prediction:** widen `m` → reach grows *linearly* at fixed rank.

## 3 — What the law bounds, honestly
- **It is a boundary of *exact identifiability*, not of leakage.** Past the line the release is reproduced at the floor by a *different* point — but that point is still `0.2–3.5 %` from the true image, i.e. a visually identical reconstruction. The defensible statement is a *quantitative* capacity law, not "the adapter hides the data."
- **Adam does not defend by non-identifiability.** It destroys the algebraic certificate (`C≡0`), yet the release stays locally identifiable at the truth at every scale tested (full column rank). It buys two *moderate* obstacles at scale — worse solution conditioning (`~400×` at the real work point) and a much smaller basin — and, ironically, breaks `1ᵀB_T=0` and hands the attacker back **one unit** of exact-inversion capacity.

## 4 — The open problem *is* the fundable direction: the initializer
The binding constraint is **the basin, not identifiability**. Measured from both sides:
- Perturbed-truth starts recover from a **65 %** start error (basin is wide along truth-directions; edge not yet located).
- Attacker-reachable, **release-only** starts recover **1 in 20** from `~80 %` off (random 0/5, certificate-anchor 0/5).

So the release *determines* the data (identifiability), but a naive attacker cannot *reach* it (basin). **That is exactly the job for a learned / population prior: supply the initializer, not a gradient bridge** — a crisp, well-posed target with a measured gap to close, and the natural home for the "foundation-model era" framing of the thesis.

## 5 — Four weeks to the deadline
1. **The initializer (the crux).** A prior trained on public data to emit a release-conditioned start; success metric = release-only recovery rate, currently 1/20.
2. **Falsify the law's prediction.** Sweep `m` at fixed `(r, N)`; read `σ_min(J)` at the truth for the linear widening.
3. **Adam basin** — is the small basin solver-fixable (trust-region / Gauss–Newton) or intrinsic?
4. **Real-data step** — frozen DINO/CLIP features + an adapter head (scoped as a separate task).

*Scope: synthetic FP64 testbed (one real-MNIST confirmation of the law); this-attacker (known/recoverable recipe, no `A₀`); observe-don't-conclude; the certificate `r−N` row is a †bundle number, not an in-repo measurement. Provenance: jobs 467914, 469120, 479587/479684 (law), 459111 (basin), 408560-63 (initializers), 466915/467622 (Adam), 568095 (real MNIST).*
