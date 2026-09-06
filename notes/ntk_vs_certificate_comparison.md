# The NTK-regime route and the certificate route, side by side (2026-09-06)

The thesis spent its first phase on a **linearized (NTK-regime) reconstruction**: assume the base weights are public,
observe the weight change, and fit pixels to it through a first-order model of training. This note puts that route's
measured results next to the **certificate** route that replaced it, so the change of direction can be judged on
numbers rather than on preference. Every figure below is quoted from a recorded run with its job id, and each is
labelled with what the attacker had to know to obtain it.

## What each route assumes

| route | what it needs beyond the released weights | what it optimises |
|---|---|---|
| **NTK / Experiment B** | the public base weights | pixels, against `‖ΔW + η Σ ĉᵢ ∇f(θ₀; x̂ᵢ)‖²`, a first-order surrogate evaluated at the frozen base point |
| **Anchor sweep** | base and final weights | the same loss, with the linearization point moved to `(1−α)θ₀ + αθ_T` |
| **Direct weight inversion** | the **whole recipe**: learning rate, step count, labels, batch composition, adapter init | pixels, through an unrolled simulation of training |
| **Gradient bridge** | an abundant public proxy dataset in-distribution | a decoder from adapter to full gradient, trained on the proxy |
| **Certificate** (current) | the public base model and a public chart. **No seed, no recipe, no labels, no batch size** | the exact linear condition `C hᵢ = 0`, where `C = P_{row(B_T)^⊥} A_T` is computed from the released factors alone |

The difference is not a matter of degree. The first four fit a residual and score the fit against ground truth the
attacker does not have; the last checks an algebraic identity the attacker can evaluate.

## The NTK route's measured results

**Its best numbers are oracle numbers.** With the per-sample coefficients computed from the true images — an upper
bound, not an attack — full fine-tuning reaches SSIM 0.9999 at N=2, T=1, and LoRA reaches 0.797 to 0.826 across
ranks 8 to 32 (Sprint 1). Even there the signal is fragile: over 200 seeds, only 22 (11%) gave a strong recovery.

**In the realistic free-coefficient mode the numbers are modest and fall off fast.** On MNIST at N=2, T=5 the best
cell is SSIM 0.922 against a same-class control of 0.643 and a mean-image baseline of 0.763; the LoRA cells sit at
0.790 to 0.866. On 32×32 flowers the best is 0.681 against a baseline of 0.646.

**The batch size is the wall.** Holding everything else fixed and raising N, mean per-image SSIM goes 0.922, 0.605,
0.536, 0.252 at N = 2, 4, 6, 10, against baselines of 0.763, 0.674, 0.606, 0.564. The route beats the trivial
baseline only at N=2, and by N=10 no single image is recognisable, with identity matching at the 1-in-N chance
floor. Direct weight inversion shows the same collapse (0.57, 0.27, 0.15 at N = 4, 10, 20) despite knowing the
entire recipe, and the gradient bridge behaves the same way.

**The linearization is never valid where the signal lives.** The route's own validity check demands a weight change
below 0.01, while the regime that carries recoverable signal is a weight change of 0.1 to 0.3. Those two bands do
not overlap, which is recorded in the project status as a finding rather than a tuning problem.

**The anchor idea helps only the path that was already working.** Moving the linearization point lifts full
fine-tuning to SSIM 0.939 at α = 0.75 (with oracle coefficients, jobs 532232 and 863020), but on the adapter alone
it never beats the mean-image baseline at any α, and under free coefficients on flowers it buys almost nothing
(0.020 to 0.084). The earlier claim that the anchor creates LoRA leakage was seed-specific and has been withdrawn.

## The certificate route, on the same kind of data

Measured today on CIFAR (this bundle), with only the released factors, the public model and a public chart:

| cell | result |
|---|---|
| head LoRA, public PCA chart, 8 private apples | 171 of 400 random starts land, 8 of 8 images, image error 1e-14 |
| hidden layer, same chart | 253 of 400, 8 of 8 |
| keyboards, skyscrapers, mushrooms, Flowers-102 photographs, plain MLP | 157 to 200 of 200 starts, 8 of 8 in three of four classes |
| the same on a backbone over-trained to 100% train accuracy | 99 to 179 of 200, 8 of 8 in two of four |
| wrong-release control | 0 of 400 |

Three differences matter more than the headline counts.

1. **It does not degrade with batch size the way the NTK route does.** What limits it is the number of examples the
   adapter failed to record, a separately measured quantity, not N itself.
2. **The attacker can tell which of their own starts succeeded.** In every landing cell all twenty lowest-residual
   starts are true landings, so success is detectable from the residual, with no ground truth. The NTK route has no
   such criterion measured anywhere in the repo, and its fidelity is always scored against the held-out truth.
3. **It is exact rather than fitted.** Certificate residuals at the private inputs are 1e-14 in float64 and 1e-5 in
   float32, against 0.25 for any other image.

## The honest ledger

The NTK route measures fidelity it cannot certify, and its best numbers need knowledge an attacker does not have:
oracle coefficients, or the full training recipe, or a rich in-distribution proxy. Its two walls, batch size and the
invalidity of the linearization in the useful regime, are measured and severe. That is why the direction changed.

The certificate route is not finished either, and two gaps should be stated whenever it is presented. Its recovery
is exact only for images the adapter actually recorded, and on raw, unprojected images it does not land at all
unless the chart is accurate to about 2%; a public chart returns the chart's projection of the private image, which
on CIFAR at k = 32 is too blurry for SSIM to distinguish from another image of the same class. And while the
attacker's residual ranking is a genuine self-check, the fraction of recoveries an attacker can verify end to end
has not yet been measured on these cells. The fair summary is that the earlier route could not certify what it
found, while the current route certifies an exact condition but has not yet closed the distance from found to
verifiably found.
