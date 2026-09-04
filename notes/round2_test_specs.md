# Round-2 test specs — the four directions reopened 2026-09-05
**Written by yoado-cd (GM) at Yoad's instruction: put these through the same audit machinery as round 1 —
design pre-audit (c9), scoring locked before rows (b9), genuineness controls (7e), execution (41).**

Context: I had asserted pixel reconstruction was closed. It is not. **The threshold I quoted
(`m ≳ s·log(n/s)`: ~250 for MNIST, ~54,000 for a 224² photo) is the SPARSITY threshold.** With a *generative*
prior the requirement scales with the generator's LATENT DIMENSION, not the image's sparsity — a published
result — so the counting argument bounds one particular prior and I let it stand for all of them. Three more
directions were similarly asserted closed and are not: certificate-as-start in the band, label-derived starts,
and the adaptive chart.

---

## TEST 5 — Generative-prior recovery: does the requirement scale with LATENT DIMENSION, not sparsity?

**Crux.** The measurement requirement for recovery under a generative prior should scale with `k_g` (the
generator's latent dimension), not with `s` (the image's sparsity). If so, the budget `r − N′` needs only to
exceed `k_g`, not ~250.

**Design.** A generator `G` fitted on **public data only**, latent dimension `k_g`. Sweep `k_g ∈ {8,16,32,64}`
at fixed `r`, so `k_g` crosses the budget `r − N′`. Two data arms, and they must be reported separately:
- **(a) on-manifold** — private images drawn from `G`'s range. **This is an idealisation and an upper bound**,
  and it must be labelled as such: the chart contains the target by construction.
- **(b) off-manifold** — real private images not produced by `G`. **This is the real test.**

**Pre-registered prediction.** The recovery crossing sits at **`k_g ≈ r − N′`**, NOT at the sparsity threshold
(~250 for MNIST). If the crossing lands at the sparsity threshold instead, the generative claim is wrong and
my correction was itself wrong.

**Controls.** (i) **Generator-only**: search `G`'s latent space with the certificate conditions REMOVED — if that
already returns something close to the private image, the generator is doing the work (this is round 1's
constraints-only control, transposed). (ii) **Scrambled release** at matched spectrum. (iii) **Oracle detector,
before the attack**: score `G`'s nearest output and `G`'s mean against the private target; if either is already
close, the cell is void. (iv) **Publish `G`'s representation error for the private images before the attack** —
if `G` cannot represent them at all, the cell is void rather than a failure.

---

## TEST 6 — Certificate→replay handoff, IN BAND

**Crux.** Untested where it matters. All existing handoff numbers are from **below** the certificate line
(`in_band: false`), where the certificate already isolates the images and there is nothing to chain.

**Design.** The band is `r − N′ ≤ k < (m−1) + r − N′`. For the standing cell that is `k ∈ {62, 64, 66, 68}`.
Corrected distance metric (to the image actually landed on), residual-decile split, `in_band` required on every row.

**Pre-registered prediction (mine, standing):** the handoff degrades sharply on crossing the line, and the decile
structure weakens from a clean cut to a graded signal. **If the cut SURVIVES in band, that is the most
attacker-relevant result available**, since it would mean an attacker can identify good starts precisely where the
certificate alone does not suffice.

**Controls.** Norm-matched chart baseline; landing image error logged BEFORE replay; wrong-manifold arm.

---

## TEST 7 — Label-derived starts (already with 41)

**Crux.** A released adapter is **labelled**. The attacker knows the concept before starting. Turn the label into
starting points and hand them to replay.

**Design.** Read the concept from the card; **collect PUBLIC PHOTOGRAPHS of that concept, disjoint from the private
set** (41's improvement on my generator version — more realistic, and free of the question whether a generator's
output distribution matches the victim's). Use them as replay starts. Exact residual verifies, so a wrong start
cannot pass.

**Baseline — and this is the load-bearing constraint (41's catch).** The standing 0-of-20 comes from the
**synthetic** bed, where an image is a latent vector and a label prior is *vacuous by construction*. **Build the
random-start baseline in the real-data setting** and report both from the same world. Nothing is quoted against
the synthetic number.

**Controls.** Oracle detector: score the collected candidates against the private target **before** replay runs.
Report the number of candidates needed — that is the attacker's real cost.

---

## TEST 8 — Adaptive / bootstrap chart

**Crux.** The only route the chain proposition does **not** bound, because it changes the search space rather than
the start. Untested.

**Design.** Recover with chart `C₀` → refit `C₁` on the recovered images **plus public data** → recover again →
measure the improvement. Iterate while it improves.

**Controls.** (i) **Refit on a RANDOM image** instead of on the recovery — if that improves recovery too, the gain
is from refitting rather than from what was recovered. This is the decisive control. (ii) Constraints-only.
(iii) **Private data must never touch the chart fitting at any stage** — burden of proof on the test, as with the
decoder.

**Void condition.** If `C₀`'s recovery is at chance there is nothing to bootstrap from — declare a minimum `C₀`
quality in advance and void below it rather than reporting a null.

**Guard that makes this safe.** The exact residual verifies: a bootstrapped chart that biases toward a wrong image
produces a candidate that does not reach the floor. So self-confirmation is detectable, not merely feared.

---

## Cross-cutting (all four, from round 1 — these are already standing rules)
Start-model column on every row · score against ground truth, never residual alone · control on the same row as
treatment · **negative rate mandatory, paired negative where constructible, population-only negative must state what
it does not control for** · **achievability floor** (from-truth solve at matched budget) reported beside every solve
with the ratio · realised pool size beside requested · **report the curve alongside the chosen point** · any learned
component scored against the private target before it feeds a solve · verdicts literal, `residual-not-zero` and
`zero-at-wrong-image` never merged.
