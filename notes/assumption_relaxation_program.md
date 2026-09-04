# The assumption-relaxation program — and the pitch it enables
**Written 2026-09-04 (yoado-cd, from the GM/strategy thread with Yoad). Status: three checkable conjectures,
one settled fact, and the framing they support. Nothing here is measured unless it says so.**

Motivation: the strongest objection to the exact-inversion framework is not the chart and not precision — it is the
**assumption stack**. As written, the theory assumes: one adapted layer, SGD, full batch, the same private batch at
every step, `B₀ = 0`, output layer. Real LoRA is AdamW, minibatched, augmented, on many attention layers. A reviewer
(or Gal) asks this first. Below, each assumption with what is actually required.

---

## 0. The recipe question — CORRECTED (81's audit, commit 0e09360; my first version over-claimed)

**The sentence that needs none of the disputed legs, and is the strongest one available:**
> The certificate is **recipe-free by construction** and is the only route that runs from random starts. So recipe
> knowledge is a hypothesis of the **wide** channel (replay), not of the **narrow** one. The recipe question is moot
> for the route that matters.

**Three legs of my original argument, as corrected:**

1. **What a release actually ships.** `adapter_config.json` reliably carries the **architecture** — `r`, `alpha`
   (hence the scale `s`), target modules, dropout. That is a real, citable gain because those are hypotheses of the
   theory and they arrive with the file. It does **NOT** carry optimiser, learning rate or epochs; model cards state
   those only when authors choose to. **Do not claim both — it weakens the half that is solid.**
2. **"Defaults dominate" is an unsurveyed prior.** State it as a declared threat-model assumption, or survey it
   (a scan of public LoRA adapters' configs would make it empirical and is a cheap self-contained study). A referee
   will otherwise ask for the survey.
3. **The oracle leg is CIRCULAR as measured — this is the important one.** R1's seven recipe rejections were run
   from `--init-noise 0.10` (verified at `recipe_robustness.py:90`, not from the summary), i.e. in the regime where
   the inversion already succeeds. From attacker-buildable starts no replay cell reaches the floor at all, and there
   **a wrong recipe and a right recipe with a bad start produce the same observation** — a high residual. So the
   attacker who needs the oracle cannot evaluate it. **The recipe oracle is DOWNSTREAM of the start problem, not
   independent of it**; it becomes available only if the start problem is solved (e.g. by the chain, if the in-band
   handoff holds, or by a learned initialiser).

**Corrected general principle (the seed argument, which as I first wrote it proved too much).** An unknown a
framework derives from a seed is a **discrete search**, not a continuous one — but its cost is
`|plausible seeds| × (cost of one oracle call)`, **and it is only usable where the oracle is evaluable at all**.
A 32-bit seed is 4×10⁹ candidates and a 64-bit seed 1.8×10¹⁹, each needing a full inversion, so this is cheap only
when the plausible set is small (defaults, conventional values), which collapses it back into leg 2. **Never cost a
seeded quantity by the dimension of what it expands into — but do attach cardinality, per-call cost, and oracle
availability.**
**Corollary for 41's queued check: run it on the DEFAULT seed specifically.** "A default-seeded shuffle is
recoverable" is a real result; "a shuffle is recoverable given the seed" is not.

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

---

## §0 addendum — theorem/threat-model check on "the recipe is recoverable" (yoado-81)

The conclusion is right for the paper, but two of the three legs do not carry weight and the third
is circular as measured. The honest version is stronger than the one offered, because it does not
need any of them.

**(1) Split what is actually shipped.** `adapter_config.json` reliably ships the *architecture* —
`r`, `alpha`, `target_modules`, dropout — and that is a real, citable gain: those are hypotheses
(A2)/(A5) and the `s` in every recurrence. It does **not** ship the optimiser, learning rate, or
epoch count. Model cards state those when authors choose to, which is not a property of the format.
So: architecture shipped, recipe sometimes disclosed. Claiming both weakens the part that is solid.

**(2) "Defaults dominate" is a prior about an unmeasured population.** Plausible, and probably
true, but we have measured nothing about it. State it as an assumption of the threat model, not as
a fact, or a referee will ask for the survey we did not do.

**(3) The oracle is ours and measured — but the measurement presupposes what it is meant to
supply.** R1's seven rejections were run from `--init-noise 0.10` (confirmed in
`recipe_robustness.py:90`), i.e. from truth + 10%. So the oracle is decisive *in the regime where
the inversion already succeeds*. From attacker-buildable starts no replay cell reaches the floor
(0/20), and there **"wrong recipe" and "right recipe, bad start" produce the same observation** — a
high residual. The verifier cannot be evaluated by the attacker who needs it. As stated, the
argument is circular.

**The resolution, which is the sentence the paper should use.** The recipe question is *moot for
the route that matters*. The certificate is recipe-free by construction — no `η`, no `T`, no
schedule, no labels — and it is the only route that runs from random starts. So the threat model's
answer is not "the recipe is recoverable", it is:

> The recipe-free route needs no recipe; the replay route needs one, and its verifier is only
> usable by an attacker who already has a start good enough to invert with. Recipe knowledge is
> therefore a hypothesis of the *wide* channel and not of the *narrow* one.

That is defensible without any claim about model cards or defaults.

**On the seed argument — right in form, and it proves too much as stated.** A framework's shuffle
is indeed derived from one integer rather than `T×N` free bits. But the cardinality is the whole
question: a 32-bit seed is `4×10⁹` candidates and a 64-bit seed `1.8×10¹⁹`, and the oracle costs a
full inversion each. That is not a small discrete search; it is infeasible *unless the seed is a
default*, at which point the argument reduces entirely to leg (2). So the general principle needs
both quantities attached:

> A quantity a framework derives deterministically from a seed or short config is a **discrete**
> unknown rather than a continuous one — but it is a *tractable* search only when
> (cardinality × cost of one oracle evaluation) is affordable. State both, or the principle
> licenses enumerating a 64-bit seed.

The useful corollary for the paper is the reverse reading: this is why the shuffle-seed check 41
has queued is worth running on the *default* seed specifically. Confirming that a default-seeded
shuffle is recoverable is a real result; confirming that some seed is recoverable given the seed is
not.

## 9. Multi-layer, continued: one certificate PER adapted layer (Yoad's question, 2026-09-04)

**The idea.** Each adapted layer `ℓ` has its own released pair, hence its own certificate
`C_ℓ = P_{row(B_T^ℓ)^⊥} A_T^ℓ`, testing that layer's *inputs* `h^ℓ`. For the **first** adapted layer `h¹ = φ(x)`
with `φ` frozen, so the derivation holds exactly (§3). For a **deeper** layer `h^ℓ` moves during training, because
the adapters below it move — so the exact derivation fails.

**But it may hold approximately, and that is testable.** The imprint sum runs over steps in which `h^ℓ` took slightly
different values; if the lower adapters move little (small `lr`, few steps — the ordinary LoRA regime; the measured
`A_T` shift in the headline cell is 9.3%), then `h^ℓ` is nearly fixed and `C_ℓ h_i^ℓ ≈ 0` to the order of that drift.
**If it holds, every adapted layer supplies its own `r − N′_ℓ` equations on its own inputs**, which is the concrete
form of "more equations rather than fewer unknowns" (§7c) — and the deeper caps do not bind (§3).

**Falsifier and design.** Two-or-three-layer adapted MLP; measure `‖C_ℓ h_i^ℓ‖` at the recorded inputs for each `ℓ`
against the lower adapters' drift; the prediction is that the residual scales with the drift and stays far below the
non-member level (0.1–1) while the drift is small. If the residual is already at the non-member level at ordinary
`lr`, the idea is dead and only the first adapted layer is usable. **Cheap: no unrolled solve, only projections.**

**Why it matters for cost.** Replay on a multi-layer adapter must unroll the whole network (the layers couple), so it
inherits SimuDy's memory wall. The certificate never simulates, so it has no such wall — **multi-layer is where the
certificate's advantage over replay is largest**, not smallest.

## 10. The vacuity trap — and why it is the theory's own boundary, not a new failure mode (job 165750, 2026-09-04)

**What happened.** At a hidden layer with `rank B_T = r` exactly (16 of 16), the certificate residual read machine
precision for *every* image and was flat as the drift varied threefold — which looked like §9 confirmed beyond
expectation. It is vacuous instead: when the row space fills the whole space, `P_{row(B_T)^⊥} = 0`, so `C = 0` and
every input gives zero, members and non-members alike. **A vacuous certificate reads exactly like a perfect one.**

**It is not a new failure mode — it is the certificate line at its degenerate endpoint.** The certificate imposes
`r − N′` conditions, so the budget `k < r − N′` is already the non-vacuity condition: at `N′ = r` the budget is zero
and there is no test. The guard is therefore not ad hoc — **the vacuity flag is exactly `r − N′ ≤ 0`**, and the
right thing to report per row is the *margin* `r − N′` alongside the residual.

**The structural consequence, and it is the interesting part.** At the OUTPUT layer the softmax cap `N′ ≤ m − 1`
keeps `N′` below `r` whenever `m − 1 < r` — so the head-width cap was silently *guaranteeing* a non-trivial
certificate. Remove it at a hidden layer (§3) and `N′` can climb to `min(width, r, N) = r`, killing the certificate.
So the same fact cuts both ways:

> **A hidden layer records MORE (no head-width cap) and certifies LESS (the row space can fill the space).**
> The usable regime for a deep certificate is `r` comfortably above `N′` — the opposite of the regime that maximises
> recording.

41 half-anticipated this before the run ("the same zero column sum that caps the count is also what keeps the
certificate's arithmetic well behaved"); this is that prediction confirmed in a sharper form.

**Standing guard, all layers, all cells:** every certificate row must carry a held-out **non-member control** pushed
to that layer's inputs, the certificate's own norm and rank, the member/non-member separation in orders, and an
explicit vacuity flag. A separation of zero orders is the signature.

## 11. How many equations does one layer actually give? — the honest budget (Yoad's push-back, 2026-09-04)

**One layer is not enough and never was.** Per image the certificate gives `r − N′` numbers and replay gives
`(m−1) + r − N′`. At `r = 16, N′ = 8, m = 10` that is **8** and **17** numbers per image. An image is not eight
numbers. So a single-layer certificate can only ever return the chart's rendering of the image in ~8 coordinates —
a class prototype — and no amount of chart engineering changes the count. **Any story in which this becomes an
image attack must get more equations, not a better chart.**

**Where more equations can come from, in order of size.**
1. **Adapted layers.** A real LoRA fine-tune adapts `q,v` in every block: 2 × (blocks) modules — 64 for a 32-block
   model. If a per-layer certificate holds (§9), the budget is `Σ_ℓ (r_ℓ − N′_ℓ)`, i.e. **hundreds to thousands of
   numbers per image** rather than eight. That is the difference between a prototype and an image. Each layer's
   conditions constrain that layer's inputs `h^ℓ`, and all `h^ℓ` are functions of the candidate `x` through the
   released model, so they compose into constraints on `x`.
2. **Rank.** `r = 64` instead of 16 quadruples the per-layer term.
3. **Multiple releases** of the same private set (checkpoints, seeds).

**THE TENSION, and it is the crux.** The certificate is non-vacuous only while `N′ < r` at that layer (§10:
`rank B_T = min(m_ℓ, r, N′)`; at `N′ ≥ r` the projector is zero and the test is empty). At the OUTPUT layer the
softmax cap `N′ ≤ m−1` protects this whenever `m−1 < r`. At HIDDEN layers there is no such cap (§3, measured:
output rank 9 vs hidden rank 16 in one run), so `N′` climbs to `r` and the certificate dies **exactly at the layers
whose numbers we need**. So:

> **The only route to a sufficient equation budget is many layers, and many layers is the regime where the
> certificate is most likely to be vacuous.** Resolving that is the load-bearing experiment of the whole programme,
> not a side check.

**Scope this forces, stated plainly:** the certificate route requires **fewer recorded images than the adapter
rank, per layer**. Fine for personalisation (5–50 private images, `r` 16–64); dead for a fine-tune on hundreds of
images the model gets wrong. That belongs in the threat model, not in a footnote.

**Priority consequence:** §9 (per-layer certificates, with §10's non-member control and the `r − N′` margin as the
x-axis) is promoted above the chain and above the chart work. If it fails, the certificate is a membership
instrument and a prototype-level reconstructor, and image-level reconstruction has to come from replay — which is
gated on the start problem.

---

## §11 answer — the stacked certificate (yoado-81). A clean form, and it is not the sum.

### The setup, in the notation of the .tex

Adapted layers `ℓ = 1…L`, layer `ℓ` with released `(A^ℓ_T, B^ℓ_T)`, certificate
`C^ℓ = Π_{row(B^ℓ_T)^⊥} A^ℓ_T`, and condition `C^ℓ h^ℓ_i = 0` — that is `r_ℓ − N'_ℓ` scalar
conditions on that layer's *inputs*. Since the release is known, `h^ℓ = F_ℓ(x)` is a known
deterministic map, so every condition pulls back to the chart: `C^ℓ (F_ℓ∘ψ)(w) = 0`.

### (1) Independence — the pullbacks are structurally coupled, and the sum is not attained

**The key fact is that every deeper layer factors through the first.** `F_ℓ = G_ℓ ∘ F_1`, so
writing `J_1 = D(F_1∘ψ)(w)` and `Φ_ℓ = DG_ℓ`, the stacked Jacobian of all the conditions is

    J_stack  =  [ C^1 ; C^2 Φ_2 ; … ; C^L Φ_L ] · J_1

Every block shares the right factor `J_1`. So the conditions are *not* Σ_ℓ independent constraints
in general position — they are Σ_ℓ constraints read through one common `k`-dimensional tangent
space, and they overlap to the extent the `Φ_ℓ` fail to separate its directions.

That gives the exact local statement. Let `V = col(J_1)` (dimension `k` if the chart is
immersive). The candidate is *locally pinned by the stacked certificate* iff

    ⋂_ℓ { v ∈ V : Φ_ℓ v ∈ ker C^ℓ }  =  {0}

i.e. iff no chart direction survives, at every depth, into that depth's certificate kernel.

**Why the naive sum overcounts, and by how much.** Deep layers can only constrain directions the
network below has not already discarded. Write `V_ℓ = Φ_ℓ(V)` and `k_ℓ = dim V_ℓ ≤ k`, which is
non-increasing in `ℓ`: it is the part of the chart's tangent space still visible at depth `ℓ`.
Layer `ℓ` can then contribute at most `min(r_ℓ − N'_ℓ, k_ℓ)` conditions, not `r_ℓ − N'_ℓ`. So

    **Necessary condition (stacked line).**  Local identifiability from the stacked
    certificate requires   k < rank(J_stack),   and
    rank(J_stack) ≤ Σ_ℓ min( r_ℓ − N'_ℓ , k_ℓ ) ≤ Σ_ℓ ( r_ℓ − N'_ℓ ).

**Corrected from an earlier draft of this section, which stated it as a generic "iff".** That was
wrong, and wrong by my own argument two paragraphs below: genericity is not available for the
`Φ_ℓ`, which are fixed by the trained network, so there is no general position to appeal to. The
sum-of-minima is a legitimate *tightening of the upper bound* — `rank(C^ℓ Φ_ℓ J_1) ≤ min(r_ℓ−N'_ℓ,
k_ℓ)` blockwise — but it is not the rank, and the rank is the thing to measure. State the
inequality; let the run report the rank.

It reduces to `k < r − N'` at `L = 1` (`k_1 = k`). The naive `Σ_ℓ (r_ℓ − N'_ℓ)` is an upper bound
attained only where every layer sees the whole chart tangent space — i.e. where the encoder
discards nothing, which is the one thing an encoder is for.

**This is falsifiable by the run already queued, and it says what to log.** `k_ℓ` is measurable
directly: it is `rank(Φ_ℓ J_1)`, the rank of the pullback Jacobian at depth `ℓ`. The prediction is
that the *marginal* contribution of layer `ℓ` to `rank(J_stack)` equals `min(r_ℓ − N'_ℓ, k_ℓ)` and
therefore **decays with depth**, so measuring `rank(J_stack)` alone will not distinguish the
conjecture from the naive sum — the run must log `rank(J_stack)` after adding each layer *and*
`k_ℓ` per layer. If marginal contributions stay at `r_ℓ − N'_ℓ` while `k_ℓ` falls, the conjecture
is wrong and the sum is right, which would be the better outcome for the attack.

**On generic independence.** There is one thing to say and it is a caution: genericity arguments
are available for the *chart* (`ψ` in general position) but not for the `Φ_ℓ`, which are fixed by
the trained network. A network trained to be invariant to a nuisance direction makes `Φ_ℓ` kill
that direction at every depth beyond the invariance, and no amount of adapted rank downstream
recovers it. So the failure of independence here is not a measure-zero accident — **it is exactly
the encoder's learned invariances**, which is the same ceiling already stated for the single-layer
case, now with a mechanism. That is worth saying in the paper regardless of how the run comes out.

### (2) Exactness at depth — approximate conditions buy resolution, not identifiability

Strictly, identifiability is a statement about exact zeros, and a deep condition is exact only to
the lower adapters' drift. So the deep conditions do **not** enlarge the set of `k` for which the
truth is an *isolated exact* zero. What they do is bound how far a candidate can stray while still
satisfying them to the achievable tolerance: with separation `ε_ℓ` at depth `ℓ` and sensitivity
`σ_ℓ` of that condition along `V`, the stacked conditions pin the candidate to a ball of radius set
by `max_ℓ (ε_ℓ / σ_ℓ)` — the *worst* condition used, not the best.

**Which is the right currency here anyway.** An image attack does not need machine precision; it
needs the image. Measured separations are ~7 orders at the frozen layer and ~2.5 at depth, so the
deep conditions can be expected to pin to roughly `10^{-2.5}` relative rather than to `10^{-15}`.
Whether that suffices is an empirical question about the chart, not a theoretical one — and it is
the same question as the knee: use a condition to the tolerance it actually holds to, and stop.

**Practical rule that follows:** weight each layer's block by its measured separation rather than
including deep blocks unweighted. An unweighted stack lets the least exact condition dominate the
residual and reintroduces exactly the over-descent failure already characterised at §9.

### (2) The pixel parametrisation — and the theory does have something to say

Against chart coordinates the rank saturates at `k` by construction, so it can only ever report
"determined within the chart I chose". Against **pixels** it is chart-free and it is the right
question: *how many independent constraints does a released multi-layer adapter place on the raw
image?* Write `K(x) = [ C^ℓ · DF_ℓ(x) ]_ℓ`, a `(Σ_ℓ(r_ℓ−N'_ℓ)) × n_pix` matrix; the answer is
`rank K`.

Three bounds, in increasing sharpness:

    rank K  ≤  Σ_ℓ (r_ℓ − N'_ℓ)                          (row count)
    rank K  ≤  Σ_ℓ min( r_ℓ − N'_ℓ , rank DF_ℓ(x) )      (blockwise)
    rank K  ≤  rank DF_1(x)                              (the shared factor)

**The third is the one with teeth, and it is a defender-side statement independent of any
attacker's prior.** Every adapted layer's map factors through the first, `DF_ℓ = Φ_ℓ DF_1`, so
however many layers are adapted and whatever their ranks, the total number of independent
constraints on the raw image cannot exceed the rank of the map from pixels to the **first** adapted
layer's input. Concretely, that rank is bounded by the narrowest Jacobian rank along the path from
pixels up to that layer — created by pooling, striding, downsampling, or any narrow projection.

    **Bound.** The recipe-free channel constrains the raw image in at most
    `min( Σ_ℓ min(r_ℓ − N'_ℓ, rank DF_ℓ), rank DF_1 )` independent directions,
    and `rank DF_1` is a property of the frozen stem alone — not of the adapters,
    not of their ranks, and not of how many layers were adapted.

**Where the bound actually bites is an architecture question, and the answer is not "adapt late
and you are safe".** `rank DF_1` is capped by the narrowest Jacobian rank along the path from
pixels to the first adapted layer, so it depends entirely on whether the frozen stem has a
dimensional bottleneck at all. Computed from architecture arithmetic (`224×224×3 = 150528` pixels):

| stem below the first adapted layer | dimension at that point | ratio to pixels |
|---|---|---|
| ResNet-18, adapters on the head | 512 (global-pooled) | 0.003 |
| ResNet-50, adapters on the head | 2048 (global-pooled) | 0.014 |
| ViT-B/32, adapters in any block | 49 × 768 = 37632 | 0.25 |
| **ViT-B/16, adapters in any block** | **196 × 768 = 150528** | **1.00** |
| ViT-L/16, adapters in any block | 196 × 1024 = 200704 | 1.33 |
| adapters on embeddings / block 1 | — | ≈1 |

**The ViT-B/16 row is the one to notice, and it is not a coincidence.** Its patch embedding maps a
`16×16×3 = 768`-dimensional patch to a `768`-dimensional token: a *square* map, dimension-preserving
by construction. ViT-L/16 embeds into 1024 and therefore *expands*. So in the most common vision
transformer configuration there is **no bottleneck anywhere in the stack**, and the cap does not
bite until the head — while a pooled CNN caps at the pooled width, two to three orders below the
pixel count. The honest headline is therefore not "adapting late protects you" but:

> **The cap is the narrowest point of the frozen stem below the first adapted layer. Pooled
> convolutional stems have one and it is severe; standard ViT stems do not have one at all.**

That is a design statement, it names which architectures are actually protected, and it does not
overclaim for the family most people are adapting.

Two consequences worth stating in the paper:

- **A bottleneck below the adapters caps pixel-space leakage through this channel**, and adding
  adapted layers above it cannot raise the cap. That is an architectural defence with no accuracy
  cost of its own, and it is the first defender-side lever here that does not depend on the model's
  confidence or on precision.
- **Conversely, adapting early — embeddings, or the first block — removes the cap**, because
  `DF_1` is then near-full-rank in pixels. "Which layers you adapt" becomes a privacy decision and
  not only a utility one.

**Scope, and it matters:** this bounds the *certificate* channel. The replay route reads more of
the release than `row(B_T)`, so `rank K` is not a bound on all leakage — it is the chart-free
ceiling on what the recipe-free route can pin. Stating it as a bound on leakage per se would be
the same overreach as the "iff" above.

## 12. The certificate has FOUR jobs, and only one of them needs a large equation budget (Yoad's correction, 2026-09-04)

§11 argued "one layer gives ~8 numbers per image, so a single-layer certificate is not an image attack". True — but
it is an objection to only ONE of the certificate's uses, and I let it read as a general limitation. Corrected:

| job | what it needs | status |
|---|---|---|
| **1. Membership inference** | almost nothing — one exact test per candidate | **works today**, exact, recipe-free, deterministic (1e-16…1e-8 vs 0.1–1). A complete result on its own; deterministic MIA is rare in a literature that is almost entirely statistical |
| **2. Start generator for replay** | **proximity, not determination** | the equation count is IRRELEVANT here — a start does not have to be unique, only inside replay's basin. §11's counting does not bear on this at all. This is the chain, and the in-band handoff is its test |
| **3. Instance identification inside a KNOWN category** | far fewer numbers than an image | **MEASURED (MNIST):** a `k`-coordinate chart identifies its own source among 10k held-out candidates 52% at `k=6` and **94% at `k=32`** — so tens of coordinates suffice for instance-ID, against 784 for the exact image. **EXTRAPOLATION, not measured (c9's scope catch):** the "these are photos of my dog — which dog?" framing transfers that MNIST mechanism to a face/animal domain with a different intrinsic dimension and a different candidate pool. Keep the number as MNIST-measured; label the dog/face scenario **the mechanism generalised**, never a measured result. As first written it let an MNIST curve carry a dog-identification claim |
| **4. Standalone pixel reconstruction** | a large budget → many layers | this is the ONLY job §11's counting constrains |

**Consequence for framing:** the target is not pixel-perfect reconstruction. It is **instance identification within a
known category**, which is both the realistic harm and far cheaper in conditions. Lead with 1 and 3; treat 4 as the
stretch and 2 as the open engineering question.

### 12b. Depth caveat (Yoad): the signal reaching a very early adapted layer may be numerically tiny
In a deep network the backpropagated error reaching the first adapted layer can be vanishingly small. **Magnitude
alone is not the problem** — the certificate reads a *direction* (`row(B_T)`), and is scale-free: a release of norm
7.6e-18 was inverted exactly in FP64 (§ledger). The problem is when the signal is small enough that **rounding
dominates its direction**, which is precisely the measured half-precision-training failure (row space rotates
2–25%, certificate 0 of 8). So this is not a new failure mode: it is the known one, reached by depth instead of by
format. Mitigations that already exist in practice: residual connections and normalisation (which exist to prevent
exactly this), and fp32/bf16 exponent range. **Testable cheaply:** measure `‖B_T^(ℓ=1)‖` and the certificate's
member/non-member separation as network depth grows, at fixed format. Predict: separation degrades with depth only
once the first layer's imprint approaches the format's rounding scale.

### 12c. Would SimuDy's method work on a LoRA update rather than full fine-tuning?
**Conceptually yes, trivially — nothing in their method needs full fine-tuning.** Unroll the LoRA training, match
the released `(A_T, B_T)` instead of the full weight delta, same cosine objective. **That is our replay route.**
Two consequences: (i) empirically it should work in the small-`N` personalisation regime and degrade as `N` grows,
as theirs does (SSIM 0.12 at 120 images) — with less information than full weights, probably worse at matched `N`;
(ii) **competitively, this is the risk**: the unrolling primitive is published and extends to LoRA in an afternoon,
so our differentiator cannot be the unrolling. It has to be the **certificate** (recipe-free, exact, no start
needed), the **conditions** (capacity lines, the caps, the imprint law), and the **impossibility** statements —
none of which SimuDy has or could get from its own method.

## 13. Replay vs SimuDy, and the iteration idea — what prop:chain does and does not forbid (Yoad, 2026-09-04)

**Replay IS SimuDy's primitive, honestly.** Both unroll training as a differentiable map and match the endpoint by a
descent on candidate data; both need the recipe (they grid-search, we fit); both need endpoints only, not the
trajectory. So replay *presented alone* is SimuDy-on-LoRA, and we must not pitch the unrolling as ours. What is ours,
and only ours:
- **The closure** makes the match cheap at the first adapted layer (small `N×N` recurrences, not the full net). That
  advantage EVAPORATES for multi-layer, where the layers couple and we inherit their memory wall — so the closure is
  a single-layer speedup, not a general one.
- **The seed reduction** (`X = A₀U`, `rN` numbers) and **the certificate** — no SimuDy analogue.
- **The theory** — capacity lines, caps, imprint law, impossibility. SimuDy has none and its method cannot produce them.
- **The observed object**: adapter-only vs their full weights. Ours is the weaker (harder) observation, so at matched
  `N` replay should do *worse* than SimuDy, not the same — less information in an adapter than in full weights.

**Why SimuDy degrades with N (honest reading).** For FULL fine-tuning, information is not their wall — **but my first argument for that was a double standard and
I then over-withdrew it (c9).** Wrong argument: "86M params ≫ N·pixels", a raw count, when this framework insists
throughout that only the RANK of independent conditions counts (`B_T` is rank `N′`; its `mr` entries are not `mr`
conditions). Right argument, same standard applied to them: **a full-weight update has high EFFECTIVE RANK per
image**, so it over-determines each one — information-rich, and therefore limited by memory and by the mixing
symmetry (optimisation), which is what their own paper says. The conclusion stands; only the argument needed
fixing. Downstream this also un-confuses the "small-N is fundamental for us" line: the honest form is **our LoRA
budget hits an INFORMATION wall as `N` grows, sooner than their MEMORY wall bites** (they keep the whole graph; the mixing
symmetry gives the optimiser more ways to trade images off as N grows; SSIM 0.12 at N=120). For US the adapter has few
parameters, so the INFORMATION wall arrives sooner — which is exactly what §11's per-image budget measures. So a better
optimiser would extend SimuDy somewhat, but the small-N limit is partly fundamental for us in a way it is not for them.

**The iteration idea — CORRECTED after audit (81, theory side; c9's algebra check pending). My first version had a
real error in variant 2 and cited the wrong theorem in variant 3.**

1. **Homotopy / continuation — and it is NOT an alternating projection.** I first framed this as alternating
   projection between `{ρ=0}` and `Z_C`. That is wrong *in kind*: alternating projection needs two available
   projectors whose intersection is the target, but here one set contains the other, so the intersection is just
   `{ρ=0}` — **and the projector onto `{ρ=0}` is the unsolved problem itself**, while the projector we do have lands
   anywhere in a strictly larger manifold. No traction. The correct object is the RELAXATION: minimise
   `ρ² + λ‖Cφ‖²`, `λ → 0`. At large `λ` it is certificate-dominated and lands in `Z_C`; at `λ = 0` it is pure replay,
   so its limit set is exactly `S_ρ`. That is a continuation from an easy problem to the hard one — **a basin device
   by construction**, hence squarely inside `cor:chainbasin`: it cannot move the line, and any gain is in where the
   descent starts and how it is steered.

2. **PEELING BY SUBTRACTION IS WRONG — replaced by subset re-simulation.** I wrote "recover one image, subtract its
   imprint `C_1` from `B_T`, re-form the certificate". Two independent objections, one algebraic and one
   theorem-level, and they agree:
   - `C_i = −lr Σ_t D_t[:,i](A_t h_i)ᵀ` with `D_t[:,i]` a function of the logits, hence of `B_t, A_t`, hence of the
     WHOLE batch — so `C_1` is not computable from image 1 alone. Recovering `w_1` does not give you the trajectory
     either, and **simulating image 1 ALONE gives the wrong `C_1`** (a solo trajectory is not the batch trajectory),
     so there is no shortcut. Circular. (c9, independent derivation.)
   - Worse (81): **`B_T − C_1` is not a release at all.** No run produces it; the remaining `C_i` would themselves
     have been different had image 1 been absent. So `thm:quot`'s hypotheses do not hold for the peeled object, the
     containment proof fails at its first step, and `prop:chain` is not merely inapplicable — for a peeled release
     `{ρ=0}` may be empty and the statement is vacuous. **Subtraction is not the inverse of inclusion here**, and it
     fails for exactly the reason the imprint law is a decomposition of ONE trajectory rather than a sum of separate
     ones.
   - **The correct mechanism, already in the repo and the paper: SUBSET RE-SIMULATION at step `η·N′/N` (R5).** That
     object *is* a genuine release of a genuine smaller run, so every theorem applies with `N → N′`. Cost: the
     omitted imprints act as a floor, so the subset is determined to about `√floor / σ_min` — measured image error
     2.5e-2, not machine precision (706597). (R6) gives the selection rule and its boundary.
   - Also corrected: the budget gain from removing one image is **exactly one dimension** of `k` for the certificate
     (`r − N′ → r − N′ + 1`). The larger gain lives in replay's subset line `k < (m−1) + r − N′_kept` and comes from
     *choosing* a smaller subset, which is a choice, not an iteration. I had blurred the two.
   - **THE CLAIM SURVIVES ONCE REFRAMED (c9).** The operative loop is
     **isolate → replay → re-simulate → repeat**: the certificate *isolates* the dominant image (spectrum
     truncation); replay *recovers* it at the single-image subset budget `k < (m−1) + r − 1`; subset re-simulation at
     `η·N′/N` *removes* it. So "this is how the per-image budget becomes a multi-image attack" is **correct**, at a
     fidelity of about 2.5e-2 per image rather than machine precision. Only the subtraction wording was wrong.

3. **Bootstrap chart — right conclusion, WRONG THEOREM (81).** `prop:chain` is stated for a *fixed* chart: `S_ρ` is
   defined relative to `ψ`, so changing the chart between iterations changes `S_ρ` and the proposition simply does not
   speak to bootstrapping. What bounds it is **`thm:cap`, which is chart-INDEPENDENT** — it counts coordinates against
   the release's dimension and never asks which chart supplied them. So a bootstrapped chart is bounded by the same
   `k` as any other: **fidelity at fixed `k`, never budget.** Drop the formulation "prop:chain constrains start and
   operator, not the chart" — the chart's *choice* is unconstrained by it, but the chart's *dimension* is constrained,
   by a different theorem. Risk and guard unchanged: self-confirmation is the risk, the exact residual is the guard.
   Precision: the certificate condition is exactly linear in `h`, but the feasible set is that affine subspace
   **intersected with the manifold of realisable features**, which is not linear — "exact and linear in feature space"
   describes the constraint, not the search.

**Net:** iteration cannot beat replay's identifiability ceiling (variants 1–2) EXCEPT by improving the chart
(variant 3), which is the same open problem as everywhere else — the chart — now reached from a different direction.
The honest statement is that the certificate's value is membership + instance-ID + start-generation (§12), and image
reconstruction routes all funnel back to the chart, whether reached by a prior, by bootstrapping, or by feature-space
search.

## 14. The multi-layer route is WEAKER THAN HOPED, not dead — and the mechanism is the better result
*(Heading corrected 2026-09-04 after Yoad challenged "killed". "Dead" over-reads the rows: see the box below.)*

> **What actually died and what did not.** DIED: *"adapt every layer and harvest all of them."* In the 15-layer
> all-adapted cell, 12 of 15 layers were vacuous because every layer above the first had a DRIFTING input — §15's
> condition 3 failing everywhere at once. DID NOT DIE: (a) **the first adapted layer always has a frozen input, in
> every configuration**, so it is always harvestable; (b) **deep layers in a stack do yield valid certificates when
> their input is frozen** — the solo deep-conv arms held at the true image to ~11 digits at conv layers 3 and 4;
> (c) **drift is a function of training length, not a binary** — at a quarter of the training length the additivity
> was exact. So the 12-of-15 collapse is ONE configuration at FULL training length, not a theorem.
> **Magnitude, honestly:** the three live layers gave 112 supplied → 80 independent → **69 usable**, against `r − N`
> from the first layer alone. That is a modest gain (~20–25% if `N = 8`), not the multiplicative gain hoped for and
> not nothing. *(Exact single-layer comparator requested from 41; do not quote the percentage until it lands.)*
> **Correct statement: multi-layer harvesting is governed by the §15 criterion, and in a fully-adapted stack at full
> training length it degrades to a modest gain.** The untested cell that would give the real ceiling is a stack with
> FROZEN GAPS between adapted layers, or shorter training.

**Both pre-registered checks went against the optimistic branch. The 20%-of-the-image / chart-free extrapolation is
WITHDRAWN and must not appear in any document.**

**14a. The layer curve flattens, hard.** On a 15-layer MLP with every layer adapted at rank 64, **twelve of fifteen
layers supply nothing at all** — their recorded count equals the rank, so the certificate is the zero matrix (§10's
vacuity, at scale). Only layer 1, layer 2 and the head carry anything: 80 independent pixel conditions out of 112
supplied, of which **only 69 survive the release's own noise floor**, at condition number 9e9. First time in the
project the *usable* count has fallen below the *formal* count.

**14b. The mechanism — and this is the keeper. `N′` does not count images; it counts recorded DIRECTIONS.**
At a layer whose input moves during training, each image contributes a *distinct* direction per step:
`row(B_T^ℓ) = span{A₀ h_i^ℓ(t)}` over images `i` AND steps `t`. So the effective count grows with training length,
`N′_ℓ ≲ N · (distinct input positions over training)`, and fills the rank. Consequences:
- **The first adapted layer is immune, and not because it is first — because its input is FROZEN.** Its input is the
  image, which never moves, so its count equals the number of images at every training length tried, and its
  certificate holds to 14 digits throughout. Any layer with no adapted layer below it inherits this.
- **Training longer destroys the deep-layer certificate.** At a quarter of the training length the additivity of §11
  is back and exact. **CORRECTION (81): §11's INEQUALITY is fine as written**, because `N′_ℓ` is defined
  operationally as `rank B_T^ℓ`, so `rank(J_stack) ≤ Σ_ℓ min(r_ℓ − N′_ℓ, k_ℓ)` survives untouched. What needed
  scoping was the *gloss* — reading `N′` as a headcount of images — not the bound. I over-corrected by calling §11
  itself a low-drift statement.
- **Predicted threshold — REDESIGNED (81), and the new form can return "no lever", which mine could not.** Do NOT
  predict `N · d(T) ≥ r`: if the trajectory is smooth and confined to a low-dimensional manifold, `d(T)` SATURATES,
  the certificate rank stops falling, and **"train longer" is not a lever at all**. Instead sweep **`rank B_T^ℓ`
  against `T` directly** at fixed `r, N`. Climbs to `r` ⇒ there is a crossing and the defender has an action;
  plateaus below `r` ⇒ the drift lives in a subspace and **the lever does not exist**. Same measurement, one step
  closer to the claim, and it is falsifiable in both directions.

**14c. Convolutions — my first statement was OVER-GENERAL AND IT INVERTS (81's correction; the unqualified version
would have told practitioners the opposite of the truth).** The saturation holds only where the patch vectors
outnumber the patch dimension: **`N × positions ≥ in_channels · k²`** — true in EARLY conv layers (many positions,
few channels), FALSE in DEEP ones (many channels, few positions). ResNet-50, `3×3`, `N = 8`:

| layer | patch dim | vectors | verdict |
|---|---|---|---|
| layer1 (256 ch, 56²) | 2304 | 25088 | full → certificate **vacuous** |
| layer2 (512 ch, 28²) | 4608 | 6272 | full → **vacuous** |
| layer3 (1024 ch, 14²) | 9216 | 1568 | partial → **certificate SURVIVES** |
| layer4 (2048 ch, 7²) | 18432 | 392 | partial → **SURVIVES** |

So "conv paths carry nothing" is an **early-layer** statement, the crossover **moves with `N`**, and it is a
*computable condition* rather than an architectural fact. **Deep convolutions, where channels dominate positions,
keep a certificate.** Stated unqualified beside the cap table this would have told a practitioner that adapting deep
convolutions is safe — the opposite of what it says. (What was measured directly on our small conv net is the
early-layer regime; the high-rank apparent margins there were an output-width artifact and the certificate did not
hold at the true image, off by 6–70%.)

**14d. What replaces it, and it is a cleaner question.** The first adapted layer is drift-immune, its conditions land
on **raw pixels with no chart at all**, so the count should scale with the **adapter rank, not with depth**.
Pre-registered as `r − N` and matching exactly so far: `r=16 → 8`, `32 → 24`, `64 → 56`, `128 → 120`. Conditioning
degrades slowly as `r` grows; ranks to 900 are running, where the count would meet the 784-pixel count.
**Deployment scope, which is the honest bottom line:** at the ranks people actually deploy (8–64) this is
**8–56 conditions on the raw image** — nowhere near determining it. So at realistic ranks the recipe-free channel is
a **membership and instance-identification instrument, not a reconstruction one**, which is exactly §12's ordering.
Consistency check: this does not violate 81's cap, since at the first adapted layer the input *is* the image, so
`rank DF_1 = 784` and `r − N` may approach it.

**14e. Bug, logged: a relative rank tolerance has no absolute floor.** Asking for the rank of a matrix at a tolerance
relative to its own largest singular value calls a **numerically zero matrix full rank** — which made every conv layer
look like it had a healthy margin. Caught by the residual column beside it. **Second time this project has been bitten
by a relative test with no absolute floor** (cf. the 1e-3 threshold that "manufactured a zero by construction").
Standing rule: every rank/threshold test carries an absolute floor as well as a relative one.

## 15. THE USABILITY CRITERION — three checkable conditions covering every cell measured (2026-09-04, late)

**Withdrawn first:** §14d's "constraint count is set by the adapter rank and runs to the ceiling" is a **tautology**
and is retracted. In those cells the adapted layer took the raw image, so the condition is linear in `x`, its Jacobian
IS the certificate, and its rank is `r − N` by construction — the table could not have been anything else. Tell that
was walked past: the "encoder cost" column read exactly **zero on every row**, which is the signature of nothing
sitting between the condition and the input to charge for. **Standing rule: before reporting a Jacobian rank, ask what
map it is the Jacobian of and whether the answer is forced.** What survives, labelled small: every condition clears
the release's noise floor at every rank; the conditioning behaviour as `r` grows; no degeneracy anywhere. And the
deployment point is now in the text — **a rank-`n` adapter on an `n`-input layer is not low-rank adaptation, it is
that layer fine-tuned**, so the impressive end of that table is outside the regime the method is about.

**The synthesis, and it covers every cell measured today.** A certificate is **usable** iff all three hold:

1. **`r > rank(B_T)`** — otherwise the projector is zero and the test is empty (§10; the non-vacuity condition is the
   budget `k < r − N′` at its endpoint).
2. **The recorded count is limited by the DATA, not by the layer's OUTPUT WIDTH.** Where the output width truncates
   the released factor's rank, `row(B_T)` is a lower-dimensional projection that **stops containing the recorded
   directions**, so the condition fails at the true image — measured six digits worse than its neighbouring cell.
   *(This is new and sharp, and it generalises the head-width cap `N′ ≤ m−1`: that cap was this condition at the
   output layer. It also explains the early-conv vacuity — there the width/position ratio saturates the layer before
   any adapter is trained.)*
3. **The adapted layer's input is FROZEN** — no adapted layer below it. Otherwise the input drifts, the count grows
   with training length as `row(B_T)` spans `(image, step)` pairs rather than images, and the certificate dies (§14b).

**Confirmation from the deep-conv run (pre-registered, held):** with everything upstream frozen, conv layer 1 is
vacuous at every rank (condition 2 fails — positions saturate it), while conv layers 3 and 4 are non-vacuous and their
certificates hold at the true image to ~11 digits. So deep convolutions ARE attackable and the earlier "conv paths
carry nothing" was an early-layer statement, now explained by condition 2 rather than asserted.

**Why this is the useful form: all three are checkable with NO private data.** `rank(B_T)` and `r` come from the
release; whether the count is width- or data-limited is `rank(B_T)` against the layer's output width; which layers have
frozen inputs comes from the adapter config, which ships with the release. **So an attacker can determine before
attempting anything whether the recipe-free channel is open — and a defender can check the same three conditions to
know whether their release exposes one.** That symmetry is the paper's most practically usable statement, and it is
the thing to put in front of the supervisor: not a result about one cell, but a criterion that decides every cell.

## 16. The flattening verdict was on a STARVED cell — and a fourth quantity is the real ceiling (2026-09-04, latest)

**16a. Exact comparator (41), so the 23% is measured not inferred.** 15-layer stack, `r = 64`, `N = 8`: layer 1 alone
**56** usable; layers 1+2 **57**; all three live layers **69** usable (80 formal, 112 supplied). So layer 2 + head added
**13 usable conditions, +23%**. Modest, not multiplicative, not zero.

**16b. THE 12-of-15 COLLAPSE WAS RANK STARVATION, NOT DEPTH.** On the same stack at **`r = 256` all fifteen layers are
usable, at every training length swept from 10 to 3,200 steps.** The drift plateau is **90–110 recorded directions**,
and `r = 64` sits below it. **So the extended-layer curve has never been measured in a configuration where the layers
were live**, and both "the curve flattens" and my "multi-layer is dead" were verdicts on a starved cell. Real curve
running at `r = 256`.

**16c. THE "TRAIN LONGER" DEFENCE IS DEAD — measured, and exactly as 81 warned it might be.** 81's objection to my
`N·d(T) ≥ r` form was that drift could SATURATE, in which case the lever does not exist. **It saturates:** the plateau
is 90–110 directions, flat from 10 steps to 3,200. So once `r` exceeds the plateau, training longer never closes the
channel. **Withdraw the "train longer" lever entirely.** The defender's levers are now: the architectural bottleneck
(§ cap table), the head-width/output-width truncation (§15 condition 2), and keeping `r` *below* the drift plateau —
which is a utility choice, not a free one.

**16d. Frozen gaps between adapted layers would NOT help (41).** Drift at a layer comes from *any* adapted layer below
it, not only the adjacent one. My suggested cell is void; the two things that keep layers live are `r` above the drift
plateau (demonstrated) and shorter training (already known).

**16e. THE FOURTH QUANTITY, and it is the real ceiling: how much of the image the FROZEN ENCODER still transmits at
the truth.** Any layer's pixel-space contribution is capped by `rank(Dφ_ℓ)`, the Jacobian rank of the frozen path from
pixels to that layer — **and it collapses with depth**: full at 1–2 frozen layers, **220 at 4, 96 at 7**, out of 784.
Past that ceiling **extra adapter rank buys literally nothing** — layer 8 stops at 96 conditions whether the adapter is
rank 256 or 900. And **conditioning collapses faster than rank does**: `σ_min` falls 1e-2 → 7e-4 → 1e-10 as the frozen
encoder goes 1 → 2 → 4 layers, and from 4 frozen layers on the *usable* count sits below the *formal* count.

**Consequences.**
- **Early adapted layers dominate; deep ones are capped low regardless of rank.** Multi-layer helps, but the sum is
  dominated by the shallow end, so expect the real curve to rise steeply then flatten — for this reason, not for the
  starvation reason we first measured.
- This is **81's `rank K ≤ rank DF_1` bound, measured per-layer**, and it should be unified with the architecture cap
  table: the table gives the bound for the first adapted layer; this gives the whole profile with depth.
- **Defender reading:** adapting deep layers leaks less in pixel space, independently of the rank chosen — a design
  lever that costs no accuracy and does not depend on confidence, precision, or training length.

**Honest shape of the whole result:** the three-condition usability criterion (§15) decides *whether* a layer's
certificate works at all; this fourth quantity decides *how much of the image* it can pin once it does. The first is
about the adapter; the second is about the frozen network, and is not about the certificate at all.

## 17. The deepest statement to come out of this: leakage is bounded by what the model did NOT learn to ignore

**(81, from the unified multi-layer section, commit a02ebea.)** Architecture gives an *upper* bound on pixel-space
leakage; **training gives the real one, and it is far lower.** By widths, the MLP stem permits 784 throughout — yet
the measured transmitted rank is 220 at four frozen layers and 96 at seven. **The gap between 784 and 96 is not
architecture. It is what training does to the map:**

> **A classifier earns its accuracy by discarding variation, and the discarded directions are precisely the ones no
> downstream certificate can constrain.**

Three reasons this is the most valuable sentence of the session:
1. **It closes a loop.** The encoder's learned invariances appear in §11 as the *reason* deep conditions are not
   independent; §16e is the same phenomenon measured as a rank profile with depth. One mechanism, two symptoms.
2. **It validates the pre-registration that produced it.** The two-column design (dimensional bound vs measured
   Jacobian rank) was requested precisely because a divergence would mean "a stem bottlenecks harder than its widths
   suggest". **Every deep row diverges, by a factor of eight.**
3. **It is a privacy-generalisation link with a mechanism**, i.e. exactly the kind of question a theory supervisor
   engages with: *the better a model generalises, the less it can leak in pixel space through this channel* — because
   generalisation IS the discarding of the directions the channel would have to carry. Falsifiable, and testable by
   sweeping model accuracy against transmitted rank at fixed architecture (the four-checkpoint ladder already exists:
   random / 78% / 95% / 98%).

**Methodological rule, generalised (81):** a prediction stated as a **form** (`N·d(T) ≥ r`) can only be confirmed or
left ambiguous; the same question asked as a **measurement** (sweep the rank against `T`) can also return *no effect*.
**Any prediction whose falsification requires an absence must be posed as the measurement, not the form.**

**THE HONEST LEDGER OF DEFENCES — weaker than "four levers", and this is the version that survives a referee:**
- **Architectural bottleneck** — works, but **ViT-B/16 does not have one** (patch embedding is square, ratio 1.00).
- **Rank below the drift plateau** (~90–110 directions) — works, but **costs utility**; it is not a free choice.
- **Train longer** — **WITHDRAWN**, drift saturates (§16c).
- **Adapt deeper** — a free design choice at no accuracy cost, **but bounded by the trained stem's contraction**, and
  the bound is what §17 explains rather than an independent lever.

## 18. REVERSAL: multi-layer harvesting saturates the whole image — and the pre-registered negative did NOT fire

**Measured (r = 256, all 15 layers live, N = 8).** Independent conditions on raw pixels as adapted layers are added:
**248 → 446 → 637 → 784.** Seven hundred eighty-four of 784 is the entire image, every condition above the release's
own noise floor. At 4× training length it takes six layers instead of four. Beyond that, nothing: by eight layers
1,500 conditions are supplied against a rank stuck at the pixel count.

**So the pre-registered negative branch — "if the count is tens against 784, the release does not contain the image and
the prior supplies the remainder" — did NOT fire.** At sufficient rank and 3–4 adapted layers the release *locally
determines the raw image, with no chart and no prior*. That is an identifiability statement, and it is the one the
whole chart argument was waiting on.

**Why it saturates, and it is not about the certificate.** The frozen forward map's own rank, measured at the true
images: **784, 784, 692, 219, 187, 138, 104, 85, 72, 60, 50, 30, 23, 19** going up the stack. From the fifth layer on
the encoder transmits at most 219 of 784 directions, decaying to 19 at the head. **Only the first three or four
adapted layers can contribute at pixel level at all** — the deeper ones are not weak certificates, they are
certificates on a subspace the earlier layers have already pinned. This is §17 quantified as a profile.

**Caveats that must travel with it (41's, all correct).**
- Layer 1's own 248 is the **identity I caught in §15** — its input is the image, so `rank = r − N` by construction.
- The additivity of layers 2–4 is what matrices in general position do; **the measured content is the absence of
  degeneracy plus the encoder depth profile, not the arithmetic.**
- It is a **Jacobian rank at the truth**: local identifiability. It says nothing about *finding* the image, and
  nothing about distant alternative solutions.
- **Conditioning is the price and it is steep:** `σ_min` falls about five orders from ~1e-1 at one layer to ~4e-6 at
  four; the six-layer cell at long training sits at condition number 4e6. Above the floor, but not a system anyone
  inverts casually.

**THE QUESTION THIS RAISES, AND IT DECIDES PRACTICAL RELEVANCE (mine).** Everything above assumes adaptation *starts
at the pixel-input layer*. Real LoRA does not: adapters go on attention/MLP blocks, so **the first adapted layer's
input is already a feature.** By the profile above, if adaptation starts at depth 4 the whole stack is capped at
**219**, not 784 — and at depth 7, at 104. **So the 784 result is a property of adapting from the very first layer**,
and the realistic configuration must be measured separately. Combined with §16's rank scope (`r = 256` is ~1/3 of full
rank against a deployed 8–64), the two open practical numbers are: *at what rank does 3–4-layer saturation occur*, and
*what is the cap when adaptation starts at depth `d > 1`*.

## 19. CLOSED: the two numbers, and the honest headline

**19a. Depth of first adaptation is a TIGHT bound** (4 adapted layers, `r = 256`, everything below frozen):

| adaptation starts at | pixel conditions | fraction |
|---|---|---|
| pixel input | 747 / 784 | 95% |
| depth 2 | 717 | 91% |
| depth 4 | 445 | 57% |
| depth 7 | 138 | **17.6%** |

**The prediction is exact where the encoder binds.** At depth 7 the frozen path transmits exactly 138 directions and
the four adapted layers deliver exactly 138 — the 3rd and 4th adapted layers add **literally nothing**, their own
encoders (96, 85) being nested inside the first's. At depth 4 the bound is 692 but only 445 is reached: the same
effect from the other side, the later layers' encoders (220, 187, 138) too small to fill what the first left open.
**So the transmitted rank at the STARTING depth is a tight upper bound, attained exactly when small enough to bind.**

**19b. The deployment gap, as a number** (layers 1–4 adapted, rank swept): **7.1%** of the image at `r = 64`, **29.2%**
at 128, **62.0%** at 192, **95.3%** at 256. Deployed adapters run `r = 8–64`. So **at deployed rank the release pins
about seven percent of the image** — and that figure already assumes adaptation at the pixel input, which deployed
adapters also do not do. `r = 64` additionally sits below the 90–110 drift plateau, so layers 2–4 are starved and the
7% is **the first layer alone**. Reaching the image needs about a third of full rank, which is not low-rank adaptation.

**19c. The trade-off, and it is the sentence to keep.** Conditioning runs against both axes. Across starting depths
`σ_min` = 3e-6, 4e-9, 2e-10, 2e-10, and **from depth 2 onward the usable count is already below the formal count**.
Across ranks it runs the other way: condition number **1.7 at `r = 64`** where almost nothing is pinned, **2.2e5 at
`r = 256`** where the image is.

> **There is no setting in this sweep where the release both determines the image and is comfortable to invert.**

**19d. THE HONEST HEADLINE.**
> The release **locally determines the raw image, with no chart and no prior** — but only when adaptation reaches near
> the input *and* the rank is a large fraction of the input dimension. **Both conditions fail in deployment, and both
> are things a defender can check about their own configuration.**

**19e. What this settles, including a question open all evening.** "Will a better chart work?" is now answered with a
measurement rather than an opinion: at deployed rank and depth the release pins ~7% of the image, so **a prior would
have to supply the other 93%** — which is exactly the pre-registered negative ("the release does not contain the
image and the prior supplies the remainder"), **firing at the DEPLOYMENT level while failing at the theory level.**
That is the precise statement: the channel carries the image in principle, and does not in practice at the
configurations people ship. Hence §12's ordering stands and is now *measured* rather than argued —
**membership and instance-identification are the deliverable; pixel reconstruction is not, at deployed settings.**

## 20. Can it be optimised? — what theory LOCKS vs what is SLACK (Yoad, 2026-09-04)

**LOCKED by theory — no attacker effort changes these:**
- **`rank Dφ_d`, the frozen path's transmitted rank at the FIRST adapted depth.** Everything factors through it
  (81's `rank K ≤ rank DF_1`), measured tight at depth 7 (transmits 138, delivers 138). Set by the victim's
  architecture, their training, and where they put adapters.
- **`r − N′_ℓ` per layer**, and **which images are recorded at all** (imprint law). Both properties of the release.

**SLACK — and (1) is the big one.**

1. **CONSTRAINT STRUCTURE BEYOND RANK — rank is the wrong measure of recoverability for an object with structure.**
   248 linear conditions on a 784-dim image leaves 536 free *directions*, but an image is not a free vector: it lies in
   `[0,1]^784`, is sparse in a basis, and lies on a realisable-image manifold. Convex/structural constraints cut the
   feasible set in ways a rank count cannot see. **This project already demonstrates it in a restricted setting:** the
   `A₀ = 0` row-span result recovers near-exact PIXELS (SSIM 1.00 at `N ≤ 4`) by intersecting a subspace with the box
   and a sparsity preference, via an LP — no chart, no learning
   (`notes/lora_span_leakage_note.md`). **So "the release pins 7% of the directions" is a LOWER bound on what is
   determined, not an upper one.** The theoretical question this poses, and it is a good one:
   > *Given `r − N′` linear conditions plus box, sparsity and manifold constraints, when is the image determined?*
   That is a compressed-sensing question with an existing literature (restricted isometry / phase transitions), and it
   is the natural way to turn a rank count into a recoverability statement.
2. **Drop the useless conditions.** Deep adapted layers add rows but no rank (their encoders nest inside the shallower
   ones — measured: 96 and 85 inside 138). Keeping them enlarges the system and hurts conditioning for nothing.
   **Selecting the non-nested layers is free and should improve `σ_min` directly.**
3. **Conditioning is a property of the PARAMETRISATION, not of the information.** `σ_min` 4e-6 at four layers is a
   statement about the coordinates chosen, not about what the release knows. Preconditioning, whitened coordinates and
   reformulation are all available and none of them changes the rank.
4. **The start problem** — still the binding practical constraint, and entirely separate from any of the above.
5. **Multiple releases on DIFFERENT base models.** Within one network the transmitted subspaces are nested, so extra
   releases of the same model buy nothing. Across *different* base models fine-tuned on the same private set the
   encoders differ and the subspaces need not nest — **the union can exceed any single `rank Dφ_d`.** Realistic
   (people fine-tune one dataset on several bases) and untested.
6. **Reduce `N′` by subset selection**, which raises `r − N′` directly — the mechanism already validated by the
   subset re-simulation route.

**Summary:** the ceiling on *directions* is locked by the frozen network; the slack is in *how much of an image a given
set of directions determines*, which is a structured-recovery question, not a rank question — and it is the one avenue
that could change the deployment verdict without changing the release.

## 21. HARD NEGATIVE: weight sharing kills the recipe-free channel on transformers

**Measured on frozen pretrained weights (a base ViT and DINO-small), at real photographs, BEFORE any adapter is
trained.** A shared linear inside a block is applied at **every token**, so one image contributes **one recorded
direction per token**. The measured span is **exactly `min(tokens × batch, input dim)`** — exactly, in every module of
both models. **One image supplies 197 independent directions into a 768-dimensional input.**

**Consequence.** The certificate needs `r > span` (§15 condition 1). At the deployed range `r = 8–64` the margin is
**zero for every batch size, including one**. At `r = 256` (4× the largest deployed rank) it survives **only for a
batch of exactly one** and dies at two. Surviving at eight images would need `r > 1500` — twice the model's own width.
DINO is worse: **no module survives above a single image.**

**So: the recipe-free channel is dead on transformer attention/MLP blocks at any deployed configuration.**

**Scope, stated tightly (41's wording, and this is the sentence that will be quoted).** It kills the **recipe-free**
channel — the narrow one needing only the release and a public model. It says **nothing about replay**, which has a
different budget and is limited by the start problem instead. And it is a statement about **weight sharing**, not about
transformers being safe: the same arithmetic condemned early convolutional layers and exonerated deep ones.

**41's pre-registration was wrong in an instructive way (their own catch).** They registered a redundancy branch
because trained transformers have famously redundant token activations. **At the level of linear span they do not** —
the activations are in general position. *Redundancy in the sense of heads being prunable is not redundancy in the
sense of vectors spanning a small subspace.*

**THE RULE, which has now decided three separate results and is arguably the project's most useful output:**
> **Count the vectors a layer actually records — batch size × positions — against its input dimension.**
> Needs no training, no release and no recipe: **only the architecture and the batch size.** A defender computes it
> for their own configuration in a minute. It condemned early convs, exonerated deep convs, and now condemns
> transformer blocks.

**POSSIBLE SURVIVOR, to check (mine): the classification HEAD is the one non-weight-shared module in a transformer.**
It consumes a single token per image (CLS), so its recorded count is `N`, not `197N` — which is exactly the regime all
of this project's MLP experiments live in, and where the head-width cap `N′ ≤ m−1` applies. If that holds, the honest
surviving surface is **head adaptation on a transformer** (plus dense MLPs and deep convs), while **LoRA on attention —
the dominant deployment pattern — is out of reach for this channel.**
