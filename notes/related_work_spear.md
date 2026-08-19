# Related work: SPEAR — and how it plugs into our bridge (note for Gal)

**Paper:** SPEAR: Exact Gradient Inversion of Batches in Federated Learning (Dimitrov et al., **NeurIPS 2024**),
SRI Lab ETH Zürich. [OpenReview](https://openreview.net/forum?id=lPDxPVS6ix) ·
[PDF](https://files.sri.inf.ethz.ch/spear/spear.pdf).

## What SPEAR does
First algorithm to reconstruct a **batch** (b > 1) **exactly** from a single honest-but-curious
gradient, for **fully-connected ReLU** networks, up to **b ≲ 25** ImageNet-scale inputs. Two structural
facts drive it:
1. **Low rank.** The weight gradient ∂L/∂W has rank ≤ b (batch < hidden/input dim), so SVD gives
   ∂L/∂W = L·R (L∈ℝ^{m×b}, R∈ℝ^{b×n}). **The inputs are a linear mixture of the right-singular-vectors:
   Xᵀ = Q⁻¹R** (Thm 3.1–3.2) — the images live in the row-space of R.
2. **ReLU sparsity.** The first-ReLU-layer input gradients ∂L/∂Z are sparse. SPEAR finds the unmixing
   matrix Q column-by-column: each column is the null-space vector of a submatrix of L such that L·q is
   sparse (Algorithm 1, sparsely-used dictionary learning), then Xᵀ = Q⁻¹R. Cost grows exponentially in b.

## Does it kill our approach? No — but it scopes our novelty.
It solves a **different inverse problem** and beats us on **one sub-problem**:

| axis | SPEAR | Our thesis |
|---|---|---|
| observable | one **single-step** mini-batch gradient | a **LoRA adapter (A,B)** / **multi-step** θ_T−θ₀ |
| activation | **requires ReLU** (sparsity is the whole trick) | studies **smooth** activations (softplus/gelu) |
| output | an **algorithm** (exact recovery) | **identifiability / anchor theory** + PEFT threat model |
| N>2 | **exact** to ~25 | free-c **collapses at N≥4** (superposition) |

So: **do not** headline "reconstruct image batches from an MLP gradient" — SPEAR owns that (single-step,
ReLU). Our defensible core is the **adapter-only observable + the Gradient Bridge + multi-step
accumulation + smooth-activation dependence + the identifiability/anchor theory** — none of which SPEAR
touches (it needs a raw gradient and ReLU).

## How it *helps* us (plug it into the bridge)
SPEAR is not a competitor to route around — it's the **downstream inversion engine** our pipeline can
call. Framing: *our* contribution is the **LoRA→gradient bridge + theory**; once a gradient is recovered
from the adapter, hand it to the best available inverter (**SPEAR** for exact N>2). That turns "SPEAR
scoops me" into "SPEAR is a component I plug in" — and directly fixes our N-collapse.

## What we adopted from it now (the practical takeaway)
SPEAR's low-rank fact ("images live in the row-space of the SVD of the first-layer ΔW") is usable in our
**optimization-based** reconstruction *without* the ReLU-sparsity machinery:
- **`--svd_init`**: seed the N reconstructions from the **top-N right-singular-vectors of the first-layer
  ΔW** (`svd_subspace_init`, `ntk_extraction.py`) — the correct low-rank subspace, vs random noise, which
  is what let the joint optimization collapse to superpositions.
- **`--diversity_weight`**: a repulsion penalty between reconstructions (SPEAR's disaggregation done
  softly) — wired the previously-dead `get_diversity_penalty`.
- **`--closed_form_coeff`**: analytic least-squares coefficient recovery (ANA-GIA / R-GAP family) — c is
  a *linear* least-squares given x, so we solve it in closed form instead of SGD (no sign-flip, no
  coefficient collapse). Companion to the SVD init.

These target the N>2 wall specifically. If they don't close it, the honest next step is to port SPEAR's
exact recovery (or its sparse-dictionary core) as the N-backend.

## Empirical result (2026-08-18): structural N-separation needs the bridge — a thesis argument
We tested all of the above on the flowers-native LoRA (r=8, T=1, free-c). Headline: **the coefficient
recipe — not the SPEAR-style additions — is what moves N-leakage.**

| N=4, flowers32 free-c | SSIM |
|---|---|
| baseline recipe (sgd + a10000 + consistency=1 + restarts) | **0.381** |
| + SVD-init (SPEAR low-rank) | 0.217 (**hurts**) |
| + SVD + diversity | 0.217 (diversity inert) |
| + TV 0.1 | ≈baseline (N=2: 0.704→0.707; TV over-smooths at 0.5) |

N=8: SVD+diversity 0.156, sequential-peeling **0.235** (peeling wins at high N but under-converges).
For context the *old* N=4 number was ~0.19, so the **recipe alone roughly doubled it (0.19→0.38)**.

**Why the structural methods fail on raw LoRA — and why it's a *result*, not a null:** SVD/ICA/SPEAR
all assume the observable is the **full gradient** ∂L/∂W = Σ cᵢ gᵢ xᵢᵀ, whose right-singular-vectors
*are* the input directions. But a LoRA adapter is **BA** (rank r): its SVD spans the **adapter's**
subspace (the row-space of the frozen down-projection A), **not the data**. So low-rank/sparsity
separation cannot work on the raw adapter — it needs the full gradient first. **This is a concrete,
measured motivation for the Gradient Bridge**: recover the full gradient from the adapter, *then*
SVD/ICA/SPEAR separation (and SVD-init) become valid. Only **x-space priors** (TV, diversity, peeling)
touch raw LoRA directly, and empirically they are weak here (clean MLP gradient ⇒ little for TV to fix).

**What we keep:** the recipe (durable win) for all free-c sweeps; `--sequential_peel` as an option for
N≥8; SVD/ICA/SPEAR filed as the **post-bridge** N-backend. TV is banked for the ViT/natural-image track
where its leverage is real.

**Recommendation for the proposal:** cite SPEAR as the SOTA *single-step, ReLU* batch inverter; scope our
novelty to the adapter / multi-step / smooth-activation / identifiability axes; and position SPEAR as a
pluggable N-backend behind the Gradient Bridge. Sits alongside the SimuDy note — both are "the raw
reconstruction is solved elsewhere; our contribution is the PEFT observable + theory."
