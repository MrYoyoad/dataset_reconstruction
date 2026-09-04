# The assumption-relaxation program — and the pitch it enables
**Written 2026-09-04 (yoado-cd, from the GM/strategy thread with Yoad). Status: three checkable conjectures,
one settled fact, and the framing they support. Nothing here is measured unless it says so.**

Motivation: the strongest objection to the exact-inversion framework is not the chart and not precision — it is the
**assumption stack**. As written, the theory assumes: one adapted layer, SGD, full batch, the same private batch at
every step, `B₀ = 0`, output layer. Real LoRA is AdamW, minibatched, augmented, on many attention layers. A reviewer
(or Gal) asks this first. Below, each assumption with what is actually required.

---

## 0. Why the recipe is not the barrier it looks like (threat-model argument, Yoad 2026-09-04)
Three independent reasons, worth stating in the paper's threat model rather than defending case by case:
1. **It is usually published.** A released LoRA ships `adapter_config.json` (rank, alpha, target modules, dropout),
   and model cards routinely state optimiser, schedule, learning rate and epochs.
2. **Defaults dominate.** A large share of real fine-tunes use the framework defaults or a popular script verbatim.
3. **The space is small AND the attacker has an exact verifier.** Optimiser ∈ a handful; schedule ∈ {constant,
   linear, cosine, …}; lr in a log range; epochs, batch size, seed. That is a small discrete search — and the
   release residual *decides* each guess (7 wrong recipes at 6e-8…4.9 against 5e-31 correct; lr recovered
   continuously from a 2×-wrong start). A search with a decisive oracle is a different problem from an unknown.
**Consequence for framing:** recipe knowledge should be presented as *recoverable*, not assumed — and the
adversary who matters (someone attacking data that is worth real compute) will sweep it.

## 1. Learning rate and batch size — SETTLED, not an assumption
*(81: the PRODUCT STRUCTURE is derived → a proposition. That the product is identifiable and fitted to 1e-15 is
MEASURED. Do not call the second a proposition — that claims a uniqueness result nobody has proved.)*
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
Shuffling and batch order fall out the same way. **81 verified this independently and it relaxes FURTHER:** the same
algebra holds for *arbitrary per-example weights* (not just 0/1 masks) and for a *per-step learning rate*, so the
hypothesis becomes **a fixed pool of examples with fixed features, with per-step selection, reweighting and rate all
permitted** — covering minibatching, sample weighting, gradient accumulation and any schedule. The simplex property
survives (masking zeroes whole columns, so `1ᵀD = 0` still holds) and the `−1` in the capacity count is safe. The imprint law survives with the accumulation running only over the
steps in which an image appeared.
**What actually breaks the closure is narrower than the table claims:** the *features* changing (augmentation, a
trainable block upstream), updates not linear in the gradient (Adam), or **cross-example coupling inside a step**
(training-mode batchnorm — the masking argument needs the per-step gradient to be a SUM of per-example terms; fine for
layernorm and for frozen/eval-mode batchnorm, so a non-issue for transformer LoRA, but it belongs on the list — c9).
Minibatching is none of these.

> **SWAPPED, NOT REMOVED (b9's pre-registration, c9's framing fix — do not oversell this).** The closure survives, but
> masking is *per-step*, so a simulator now needs the **schedule** — which images in which step — a `T×N` unknown
> (~3,200 in the standard cell) against an identifiable budget of order 100. By the project's own counting the
> schedule is **not identifiable**. Honest claim: *"minibatching does not break the closure; it moves the requirement
> from the data to the schedule, and schedule-recoverability is open."* And the asymmetry that matters:
> **free for the CERTIFICATE** (it never simulates — it needs only `row(B_T)` and `A_T`) and **expensive for REPLAY**.
> That is why the certificate is the route that survives realistic training.
>
> **BUT — the schedule is not free bits, it is a SEED (Yoad, 2026-09-04).** Frameworks generate the shuffle
> deterministically from a seed (PyTorch `DataLoader` + generator, seed ⊕ epoch), so the `T×N` schedule is a
> *function of one small integer* and a known framework algorithm — not `T×N` independent unknowns. The attacker
> sweeps seeds with the release residual as an exact verifier (wrong recipes sit 20+ orders above the correct one).
> Seeds are overwhelmingly small conventional values (42 first). **So the honest statement is: the schedule is a
> low-entropy discrete unknown with a decisive oracle test, not a `3,200`-dimensional continuous one.** This
> materially weakens the "schedule swap" objection and should be checked (sweep seeds on a minibatched release and
> confirm only the true seed reaches the floor).
**To verify:** run a minibatched release through the existing simulator with masked `D_t`; `fwd_check` must stay at
machine precision. Cheap. If it holds, the assumption is **SWAPPED, NOT REMOVED** — see the box below.
**Sharper second test (41):** with masking, an image's imprint accumulates only over the steps it appeared in, so two
images sampled equally often should be recorded comparably while a rarely-sampled one drops toward the floor. That is
a stronger test of the same claim than `fwd_check` alone and costs nothing extra.

## 3. Multi-layer — the certificate localises to the FIRST adapted layer (hypothesis corrected by 81)
**State it as "the first adapted layer, with the network BELOW it frozen" — not "one adapted layer".** The certificate
needs only the *shape* `A_T − A₀ ∈ row(B_T)`, which survives whatever sits above; so **the certificate applies to the
first adapted layer of any stack.** The clean *closure* (cheap self-contained simulation, i.e. the replay route) does
NOT: with upper adapters the backpropagated error is no longer a function of the small variables alone, giving a
coupled joint recurrence — consistent with the measured multi-layer cells sitting ~20 orders off the floor.

**The cap does not loosen deeper — it VANISHES (81).** It came from the softmax zero-sum, which a backpropagated error
does not have, so `N′ ≤ min(m_ℓ, r, N)` with no `−1`; at a hidden layer of width ~10³ it is not binding at all.
**Head-width protection is a property of adapting the HEAD; an adapter on a hidden layer does not have it.** This is
the sharpest defender-side consequence in this note and deserves its own line in the paper — **scoped to recording in
ACTIVATION space, not to image recovery** (c9).

**What is CONJECTURE here, tagged (7e):** *closure does not survive multi-layer* — settled by 81's derivation and
retrodicted by Step 17 (the one multi-layer attempt was the REPLAY route and it failed, seeds known or not).
*Certificate localisation* — **UNTESTED**; needs its own multi-layer certificate cell. *The cap vanishing at a hidden
layer* — a conjecture resting on that conjecture, and the sentence that most strengthens the attack, so it must never
be written as a capability the attack has. Required tag wherever it appears: "(untested; requires certificate
localisation, which is itself untested)".

**Cost if localisation holds:** what is recovered is that layer's *inputs* (hidden activations), so an
activation→image inversion step is needed — see §5.

**Prediction to register BEFORE the run (41):** the same zero column sum that caps the count is also what keeps the
certificate's arithmetic well behaved — measured precedent: on a half-precision release where the rank exceeded the
cap, the count became unreadable at any tolerance. So the deeper-layer case may loosen the cap *and simultaneously
make the recorded count unreadable*. Predict both, don't discover the second. **Falsifier (b9):** the deep-layer test
is vacuous unless `N ≥ m`, with the magnitude (cap `m−1` → layer width) predicted in advance.

**Honest line to use, verbatim:** "the one multi-layer attempt was the replay route, and it failed as the corrected
theory predicts; certificate localisation is untested."

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
  `N((m−1)+r−N)`, so our counting **quantifies how much low rank compresses leakage relative to their setting** — *caveat (81): that
  compares a total parameter count against a manifold dimension; normalise both per-example before quoting a ratio*;
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
| same batch every step | **SWAPPED, not removed** — closure survives masking (derived, twice-verified, unrun); but replay then needs the *schedule* (`T×N` unknowns, not identifiable). Free for the certificate, open for replay |
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


## Pre-registration for both conjectures (yoado-b9, adopted; executor 2026-09-04)

Both are **derivations, unmeasured**, and carry that label in any ledger entry.

### C1 minibatching — state it as PROVED, and add the arm that decides what it costs
The induction never inspects `D_t`'s contents, only its shape, so `B_t = P_t(A₀H)ᵀ` and `A_t = A₀(I + HM_tHᵀ)`
reproduce for **any** masked or reweighted `D_t`; shuffling and batch order fall out identically. So the run is
**verification, not test**. The arm it is missing is the one that matters: masking is *according to the step-t
batch*, so the simulator must know **which images were in which step** — a new recipe unknown of size `T×N`
(3,200 in the standard cell) against an identifiable recipe budget of `N((m−1)+r−N) − Nk` = **120** there. A free
per-step schedule is therefore **not identifiable by this project's own counting**. Two arms, and (a) never stands
alone:
- **(a) schedule known** → `fwd_check` at machine precision (tests the closure).
- **(b) schedule wrong** — same release, a different valid schedule of the same batch size → residual far above the
  floor, as every wrong recipe is (6e-8 for one step in four hundred).
Honest line after a pass on both: *minibatching does not break the closure; it moves the requirement from the data
to the schedule, and whether the schedule is recoverable is the open question* — which the existing recipe-probe
machinery may answer, and that would be the real result. Without (b) a pass reads as an assumption **removed** when
it was **swapped**.

### C2 deep-layer localisation — the stated falsifier is vacuous, and the effect is larger than claimed
- **The cell must have `N ≥ m`.** `rank B_T ≤ min(m, r, N′)` always, so at m = 10, N = 8 the rank cannot reach 9
  whatever the layer does and "rank did not exceed m−1" would mean nothing. Require the output cap to *bind* first
  (N = 10–12), else the test is vacuous.
- **Pre-register the magnitude, not the direction.** At the output the cap is the class count minus one; at a hidden
  layer of width `W` the zero-sum is gone and the cap is `W` — one to two orders in any real model, not "one more
  image". Predict `rank B_T` reaching `min(W, r, N′)` and the replay line moving from `k ≤ (m−1)+r−N′` to
  `k ≤ W + r − N′`. Direction-only would let any increase count as confirmation.
- **Gates first:** `fwd_check` at machine precision on the hidden-layer release, and the certificate vanishing at the
  recorded truths there. If `Ch` does not vanish at a hidden layer, localisation has failed and no cap claim is
  scoreable.
- **What a pass does not license:** what is recovered at a hidden layer is that layer's *inputs* — activations, not
  images — so a pass is a wider recording cap **in activation space** and says nothing about pixels until the
  activation→image step exists. "More images recorded" and "more images reconstructed" are the two claims a reader
  will merge.
- **Both signs up front:** a larger `N′` moves the certificate line `k < r − N′` the *unhelpful* way, so the loosened
  cap widens the replay band and narrows the certificate-alone region at the same time. A mixed result is the
  prediction, not a failure.
- *Executor's addition:* the zero column sum is also what keeps the certificate's arithmetic readable — on a
  half-precision release its loss made the rank exceed the cap and become unreadable **at any tolerance**. So the
  deep-layer cell may loosen the cap and make the count harder to read at once; predict that before running.
- *Executor's addition to C1:* with masking an imprint accumulates only over the steps an image appeared in, so two
  equally-sampled images should be recorded comparably while a rarely-sampled one falls toward the floor. A sharper
  test of the same claim than `fwd_check` alone, at no extra cost.


### The minibatching result is an ASYMMETRY between the two routes, and that is how it should be stated

Relaxing the data assumption **buys a schedule assumption**, and the two routes pay it differently:

- **Free for the certificate.** `C = P⊥_{row(B_T)} A_T` needs only the released factors. It never simulates, so it
  never needs to know which examples were in which step. Per-step masking changes *which* images are recorded and
  how strongly (an imprint accumulates only over the steps its image appeared in) but not the algebra: `Ch_i = 0`
  still holds for every recorded image.
- **Expensive for replay.** The simulator must reproduce the trajectory, so it needs the schedule: `T × N` unknowns
  (3,200 in the standard cell) against a per-image identifiable budget of order 100. A free per-step schedule is
  **not identifiable by this project's own counting**, so replay must either know the schedule or fit it — and
  fitting it is a much larger recipe-probe problem than fitting a learning rate.

**So the honest statement is not "minibatching is handled" but: minibatching is free for the recipe-free route and
expensive for the replay route — which makes the certificate the route that survives realistic training.** That is
a point in the certificate's favour and belongs in the pitch that way, alongside the standing caveat that the
certificate pins fewer coordinates per image.
