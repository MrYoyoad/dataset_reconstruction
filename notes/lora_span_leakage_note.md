# A released LoRA adapter can leak its exact training images — a mechanism, its attack, and its boundaries

**Scope (on every number below):** first-layer LoRA (the layer whose input is the raw image), **A₀=0 init**,
**SGD-family optimizer** (Adam/AdamW break it; weight-decay ≤1e-2 is fine), **N ≤ r** (private-set size ≤ LoRA
rank), **no gallery required**, this-attacker (passive, reads only the released adapter). Setting: MNIST-MLP,
binary {0,1}, r=8. Observe-framed; this is a clean *case study* of what an adapter can leak in principle, with
every boundary measured — not an unconditional claim about production LoRA.

## 1. The mechanism — a row-span theorem
For a first-layer LoRA with A₀=0 / B₀ random, every SGD step gives ∂L/∂A = Bᵀ∂L/∂W₁ and ∂L/∂W₁ = Σᵢ δᵢ xᵢᵀ, so
**every row of A_t is a linear combination of the training inputs xᵢ, at every step, exactly.** Hence

> **row(ΔW) = span{x₁ … x_N}, exactly, when N ≤ r — and it is seed-independent** (B₀ only mixes the coefficients;
> it never leaves that span).

Verified to machine precision: the residual of a training image onto the row space is **6×10⁻¹⁵**, and
D(same-set, re-seeded) in the row space is **0.0000** — the data side of the update carries *no* init noise.
This is why the attack needs no knowledge of the victim's initialization: unlike the rest of the thesis
program (where seed noise is the detection floor), here the floor is **zero by construction** on the data side.

## 2. The attack
- **Closed-world** (images drawn from a known gallery): rank the gallery by residual onto row(ΔW), take the N
  smallest → **exact identification of the private set, 100% (10/10), holding to a 10,000-image gallery**
  (member residual 3×10⁻¹⁴ vs non-member 0.75).
- **Open-world** (no gallery): the private images are the **sparsest vertices** of the box-constrained polytope
  P = {c : 0 ≤ Vc + m ≤ 1} in the span (an MNIST digit sits on ~600 exact-zero-pixel constraints → a vertex;
  mixtures have union-of-supports = fewer zeros). Recover by a linear program (min over random directions +
  the min-intensity direction; take the N sparsest non-collinear vertices). **Near-exact pixel reconstruction:
  SSIM 1.00 at N≤4, 0.96 at N=8**, confirmed on an actual adapter's ΔW (0.959). FastICA — a generic
  independence prior — reaches only 0.4–0.55 because it ignores the box and sparsity; the box also fixes the
  scale ICA cannot. *(The attack reads only the released A factor; B is irrelevant to the row space.)*

Ceilings on every plot: closed-world selector = 1.0 (exact), full-gradient reconstruction; floor: mean-image
baseline (~0.44). Per-image scores are Hungarian-matched.

## 3. Boundaries — all measured
- **Optimizer (realism boundary #1).** The exactness needs the A-update to be a *linear* map of the gradient.
  SGD / SGD+momentum: exact (SSIM 1.00, residual 10⁻¹⁵). **Adam / AdamW: the exact span dies** (SSIM → 0.60 /
  0.50; residual → 0.75) — the elementwise m/√v update is not linear. Weight decay ≤1e-2 preserves it (10⁻¹⁴);
  only pathological wd=0.05 collapses A's rank (its singular values fall to [0.66, 2.6e-3, 1.3e-4]), which
  makes the row space numerically undefined — not a failure of the theorem.
- **Init convention = a privacy design lever (realism boundary #2).** *Which factor you zero-initialize decides
  which side of ΔW exactly carries the private data.* A₀=0 → row(ΔW)=span{xᵢ} (inputs, recoverable as pixels).
  **B₀=0 (the HF PEFT default)** → the exact structure sits on the column/δ side (col(ΔW) ⊆ span{δᵢ}), a
  seed-independent fingerprint of the private set but in *activation* space, masked on the input side by the
  random A₀ (member residual ≈0.85 flat across T). So HF-default LoRA does **not** leak through *this* channel —
  the δ side is **open**, not safe.
- **N > r cliff.** Past the rank, row(ΔW) is an r-dim *projection* of the N-dim span, and recovery declines
  monotonically: **N=8 → 0.95, N=9 → 0.86, N=12 → 0.77, N=16 → 0.71** (row-rank pinned at r=8). The images that
  fail are exactly the **nested/overlapping-support pairs** (support-Jaccard 0.60 for failures vs 0.54 for
  successes) — the genuine superposition regime (ICA / SPEAR territory), tying this boundary to the
  rank / q_eff story.
- **Precision.** Real adapters ship quantized. Quantizing the released (A,B) and rerunning: float32 1.00,
  **bfloat16 0.91** (the common release precision — still clearly recognizable), int8 0.73 (still ≫ baseline).
  The noise-free span is a modelling convenience, not a load-bearing assumption.

## 4. Relation to prior work
The greedy-gradient-matching-from-a-fixed-pool core is published (SELECT, Zaman et al. 2506.15553, for text;
GradMatch for coresets), and a retrain/consistency certificate exists (VGIA); the novelty here is the
*composition*: the LoRA adapter as the measurement, the **exact** row-span structure (not approximate matching),
the box+sparsity LP that turns the span into pixels with no learning, the closed→open-world arc, and the
init/optimizer as **privacy design levers** — a passive attack, distinct from the malicious-server LoRA
inversions (MineGrad / PEFTLeak).

## 5. Takeaways and open frontiers
**Takeaway (defensive):** a released first-layer adapter trained with A₀=0 + SGD leaks its exact training images
for N≤r, seed-independently, even in bf16 — and *which factor you zero-init and which optimizer you use are
privacy choices.* **Open frontiers:** (1) standard-init (B₀=0) recovery from the δ/column side; (2) N>r
superposition (ICA/SPEAR de-mixing of the r-dim projection); (3) deeper/attention layers (there the layer input
is a hidden activation, so the LP recovers activations, and the pixel step needs the earlier layers' inverse);
(4) scale beyond MNIST.

**Figures:** `figures/harder_id/membership_selector.png` (closed-world exact, N-sweep), `.../lp_unmix.png`
(open-world LP vs ICA vs baseline vs closed-world ceiling), `.../open_world_unmix.png` (ICA-only detail).
**Code:** `experiments/dataset_sensitivity/{membership_selector, lp_unmix, robustness_checks, robustness_fixes,
precision_sweep, realism_gate}.py`. Full record + provenance: STATUS.md; mechanism lesson: LESSONS_LEARNED.md
"row-span theorem".
