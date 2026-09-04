# The assumption-relaxation program — and the pitch it enables
**Written 2026-09-04 (yoado-cd, from the GM/strategy thread with Yoad). Status: three checkable conjectures,
one settled fact, and the framing they support. Nothing here is measured unless it says so.**

Motivation: the strongest objection to the exact-inversion framework is not the chart and not precision — it is the
**assumption stack**. As written, the theory assumes: one adapted layer, SGD, full batch, the same private batch at
every step, `B₀ = 0`, output layer. Real LoRA is AdamW, minibatched, augmented, on many attention layers. A reviewer
(or Gal) asks this first. Below, each assumption with what is actually required.

---

## 1. Learning rate and batch size — SETTLED, not an assumption (measured)
`η`, the adapter scale `s` and the batch size `N` enter the recurrences **only as one product**, because the loss is a
mean. Consequences, all measured (job 484255 and the recipe arms):
- The learning rate is **fitted, not assumed**: carried as a free unknown from a start wrong by 2×, recovered to
  ~1e-15 relative with images still at machine precision. `η` and `T` are separately identifiable despite entering the
  continuum limit as a product.
- Seven deliberately wrong recipes (step count off by one, rate off by 1%, rate/step-count off by a factor, wrong
  optimiser) end at 6e-8 … 4.9 against 5e-31 for the correct one — so a wrong recipe is *detectable*, and a correct
  one is *verifiable*, both without private data.
- The batch size is never recovered and never needed (absorbed into the effective rate). **Exception:** decoupled
  weight decay contributes a differently-scaled second term — a defender who publishes a nonzero decay publishes the
  batch size with it.
**Pitch line:** "an attacker who does not know the recipe can find it; one who thinks they know it can verify it."

## 2. Minibatching — CONJECTURE (derivation below), highest-leverage check available
The assumption table says "the same private batch enters at every step", which rules out minibatched SGD. **The
induction appears not to need it.** Let `D_t ∈ R^{m×N}` be the error matrix with **zero columns for images not in the
step-t batch** (masking). Then with the full fixed `H`:
```
∇_B = s D_t Hᵀ A_tᵀ = s D_t (A_t H)ᵀ = s D_t (I + G M_tᵀ)(A₀H)ᵀ      ← still (·)(A₀H)ᵀ
∇_A = s B_tᵀ D_t Hᵀ = s (A₀H) P_tᵀ D_t Hᵀ = A₀H (s P_tᵀ D_t) Hᵀ      ← still A₀H(·)Hᵀ
```
so `B_t = P_t(A₀H)ᵀ` and `A_t = A₀(I + H M_t Hᵀ)` reproduce under **any per-step masking or reweighting** of `D_t`.
Shuffling and batch order fall out the same way. The imprint law survives with the accumulation running only over the
steps in which an image appeared.
**What actually breaks the closure is narrower than the table claims:** the *features* changing (augmentation, a
trainable block upstream) or updates not linear in the gradient (Adam). Minibatching is neither.
**To verify:** run a minibatched release through the existing simulator with masked `D_t`; `fwd_check` must stay at
machine precision. Cheap. If it holds, one of the three worst-looking assumptions disappears.

## 3. Multi-layer — CONJECTURE: the certificate localises to the FIRST adapted layer
Attack the earliest adapted layer. Its inputs come from frozen machinery, so "fixed inputs" holds exactly, and the
closure's *shape* survives even though the error signal reaching it is backpropagated rather than a softmax residual
(the derivation needs only a rank-structured update with fixed inputs and `B₀ = 0`). Two changes, in opposite
directions:
- **Lost:** the `e^{−margin}` interpretation and the cap `N′ ≤ m−1` — both come from the softmax's zero-sum columns,
  which a hidden error signal does not have. **Losing the cap helps the attacker**: more images can be recorded than
  the output layer permits.
- **Cost:** what is recovered is that layer's *inputs* (hidden activations), so an activation→image inversion step is
  needed — see §5.
**Correct statement to use:** not "multi-layer is outside the theory", but "the certificate localises to the first
adapted layer; the margin law is what is genuinely output-layer-bound."

## 4. A learned decoder over thousands of adapters — right idea, and the SLOT is the point
Generating thousands of `(adapter, private data)` pairs is cheap at small scale (each is a short fine-tune). Two slots:
- **As the answer** (adapter → images directly, à la Yao 2024): weak — recovery is indistinguishable from
  hallucination.
- **As the INITIALISER or the CHART-BUILDER: strong, and uniquely safe in this framework.** The certificate and the
  replay residual **verify the output**: a hallucinated candidate fails the exact test, a correct one reaches the
  floor. *The prior proposes; the algebra disposes.* This is the standard criticism of every prior-based
  reconstruction attack, and this framework is the one place it is answerable.
This also puts the original **gradient-bridge instinct back in play** — not as the attack (Gal disliked that), but as
the initialiser, which the record already names as replay's binding constraint.

## 5. Latent/embedding space instead of pixels — converges with everything above
Run `ρ` and the certificate over **embeddings**, not pixels: the certificate is natively a condition on `h = φ(x)`.
Treat embedding→image as a separate, priored step. This is exactly Oz et al. 2024's decomposition (see §6), it is
better conditioned, and it isolates the prior where it can be audited. Also the necessary endpoint for §3.

## 6. Where this sits against the supervisor's own recent work
- **Oz, Yehudai, Vardi, Antebi, Irani, Haim 2024 — reconstruction from transfer learning** (`papers/Oz_et_al_2024_…`).
  Same shape as ours: frozen foundation encoder (DINO-ViT/CLIP) + trained head; they reconstruct **in embedding
  space**. Two hooks: (i) their equations come from the *full head* (~`m·n` numbers) — a LoRA release exposes only
  `N((m−1)+r−N)`, so our counting **quantifies how much low rank compresses leakage relative to their setting**;
  (ii) their abstract's own contribution includes *"a novel clustering-based method to identify good reconstructions
  from thousands of candidates"* — **a heuristic for a problem our certificate solves exactly** (`Ch = 0` at machine
  precision, no training-set knowledge). We arrive with the missing verifier, not with a competitor.
- **Smorodinsky–Vardi–Safran 2025 — provable privacy attacks.** Rests on KKT via homogeneity + convergence;
  reconstruction only in `d = 1`; MIA probabilistic under near-orthogonality. Ours is a **deterministic** membership
  certificate at finite `T` with no homogeneity — stronger on that axis, in a setting that paper cannot enter.
- **Gronich–Vardi 2026 (Adam/Muon implicit bias).** Adam is what breaks the closure; that paper characterises where
  Adam-trained nets land. Hold in reserve as the route to the Adam case — do **not** lead with it (he called ℓ∞
  impractical).

## 7. Chart-free routes (for "what if the chart never works")
(a) **Embedding-space recovery + separate inversion** (§5, his own group's machinery).
(b) **Convex constraints instead of dimension reduction — already working:** the `A₀ = 0` row-span result recovers
    *pixels* with no chart and no learning (private images = sparsest vertices of a box polytope, by LP; near-exact
    for `N ≤ r`). Restricted (first layer, that init, SGD) but an existence proof that chart-free pixel recovery is
    possible here. See `notes/lora_span_leakage_note.md`.
(c) **More equations rather than fewer unknowns:** multi-layer adapters (each adapted layer is another measurement of
    the same data), multiple released checkpoints of one private set, higher rank.
(d) **Peel one image at a time** — the per-image budget is largest at `N′_kept = 1`.

## 8. Honest status of the whole stack after this
| assumption | status |
|---|---|
| recipe (η, T, batch size) known | **not needed** — fitted and verifiable (measured) |
| same batch every step | **probably not needed** — masking conjecture, §2, cheap to check |
| single adapted layer | **localises** to the first adapted layer, §3 — conjecture |
| output layer (margin law, `N′ ≤ m−1`) | genuinely output-layer-bound; cap loosens deeper |
| `B₀ = 0` | required (it is the HF PEFT default) |
| SGD-family | **required** — Adam breaks the closure; replay still identifies, no certificate |
| no augmentation | **required** — augmentation moves the features and breaks the induction |
| images on the attacker's chart | required for an exact zero; the open problem (§5, §7) |

**Value ranked by robustness (not by excitement):** imprint law > capacity bound > certificate > chart-based
reconstruction demo. The excitement runs in the opposite order; know which you are selling.

---

## §9 — theorem-side verification (yoado-81, write-up lane). Algebra checked independently.

### (1) Minibatching — **CONFIRMED, and it relaxes further than the note claims**

The induction goes through. With a per-step mask, `D_t → D_t^m := D_t·diag(mask_t)`:

    ∇_B L = s D^m Hᵀ A_tᵀ = s D^m Hᵀ(I + H M_tᵀ Hᵀ)A_0ᵀ = s D^m (I + G M_tᵀ)(A_0H)ᵀ
    ⇒ B_{t+1} = [P_t − ηs D^m (I + G M_tᵀ)](A_0H)ᵀ                    ✓ same shape
    ∇_A L = s B_tᵀ D^m Hᵀ = s (A_0H) P_tᵀ D^m Hᵀ
    ⇒ A_{t+1} = A_0(I + H[M_t − ηs P_tᵀ D^m]Hᵀ)                        ✓ same shape

so `P_{t+1}=P_t−ηs D^m(I+GM_tᵀ)` and `M_{t+1}=M_t−ηs P_tᵀ D^m`, identical but for the mask. Note
`D_t` itself is still the *full* residual `D(W_0H+sP_tQ(I+M_tG))` — masking is applied only where
the gradient uses it, so the recurrence stays closed.

**It relaxes further than masking.** The same algebra holds for `D·diag(w_t)` with arbitrary
per-example weights `w_t ≥ 0` (masking is the 0/1 case), and for a per-step scalar `η_t` (the
recurrences simply carry `η_t s`). So the honest hypothesis is:

> **(A1′)** a *fixed pool* of examples with fixed features `H`; per-step selection, per-example
> reweighting, and a per-step learning rate are all permitted.

That covers minibatching, sample weighting, gradient accumulation, and any schedule — a large
realism gain over "the same batch every step". **The simplex proposition survives**: masking zeroes
whole columns of `D`, so `1ᵀD^m = 0` still holds and `1ᵀB_t = 0` with it. The `−1` in the capacity
count is unaffected. Note the batch-size normalisation folds into `η_t` and is therefore covered by
R5: the trajectory still sees only one scalar per step.

### (2) Multi-layer — **RIGHT CONCLUSIONS, WRONG HYPOTHESIS**; state it as "first adapted layer"

The shape survives with a backpropagated `Δ_t` in place of `D_t`, since the derivation used only
the bilinear gradient form and `B_0=0`. But **self-containedness needs everything above frozen**:
`D_t` was a function of `P_t,M_t,Q,G` alone, whereas `Δ_t` depends on the layers above, so if those
are *also* adapted the system couples into a joint recurrence over every layer's `(P,M)`. Still
finite-dimensional and still batch-sized — worth saying — but not the clean closure. This is
consistent with the measured multi-layer cells sitting 20 orders off the floor with the oracle arm
nearly as bad.

So the hypothesis to write is **not** "one adapted layer" but **"the first adapted layer, with the
network below it frozen"** — which is what makes its inputs fixed. That is the real scope gain: the
certificate applies to the first adapted layer of *any* stack, whatever is above it.

The consequences in the note are correct, and one is stronger than stated:

- **The imprint law's structure transfers, its interpretation does not.** `C_i = −η s Σ_t Δ_t[:,i](A_th_i)ᵀ`
  is still a sum of one rank-one term per example with no cross term — so additivity, the rank
  reading, and `B_T=Σ_iC_i` all survive. What does *not* transfer is `‖C_i‖ ∼ e^{−margin}`, which
  came from `D = softmax − E`. **"What leaks is what the model had to learn" is an output-layer
  statement.**
- **The `N′ ≤ m−1` cap is output-layer-bound and vanishes deeper — more strongly than "loosens".**
  It came from `1ᵀD = 0`; a backpropagated `Δ` carries no such constraint, so the cap becomes
  `N′ ≤ min(m_ℓ, r, N)` with `m_ℓ` that layer's output width and **no `−1`**. At a hidden layer of
  width ~10³ the cap is `min(r,N)` — i.e. not binding at all. The head-width protection is a
  property of adapting the *head*, and an adapter on a hidden layer does not have it.

### (3) Recipe fitted — **split it; only half is a proposition**

Two different statements are being merged. The first is derived and belongs as a proposition: `η`,
`s` and `N` enter every recurrence only through one scalar (R5), so the release cannot determine
them separately and an attacker needs only the product. The second — that the product is
*identifiable and fitted to 1e-15*, and that seven wrong recipes are rejected — is **measured**, and
calling it a proposition would claim a uniqueness result nobody has proved. State the product
structure as a proposition; keep the fittability in the measured section, cross-referenced.

### §6 comparison — needs the same normalisation on both sides

`m·n` against `N((m−1)+r−N)` compares a *total* parameter count with a *manifold dimension*. The
paper's count is per-example — `(m−1)+r−N` coordinates — so the honest comparison is per-example on
both sides, and the full-head figure has to be derived in the same units before the ratio means
anything. Worth doing; not worth quoting until it is.
