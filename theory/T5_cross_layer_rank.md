# T5 — Does depth add information?

## Statement

Let `x = G(z)` be a `k`-dimensional chart, `Phi_l^0` the base network's feature map into layer `l`, and

    F(z) = [ Ctil_1 Phi_1^0(G(z)) ; ... ; Ctil_L Phi_L^0(G(z)) ],
    M_l  := J_{Phi_l^0}(x*) J_G(z*) in R^{n_l x k},     q_l := rank( Ctil_l M_l ).

**T5.1 (nesting ceiling — PROVED).** Because `Phi_l^0 = psi_l . Phi_{l-1}^0`, the chain rule gives
`M_l = J_{psi_l} M_{l-1}`, hence `ker M_1 subset ker M_2 subset ...` and

    rank J_F(z*)  <=  rank M_1  =:  k_1  <=  min(k, n_1).

No amount of depth exceeds the first adapted layer's chart sensitivity. Stated for the realism question: the
ceiling on stacked information is the first adapted layer's chart sensitivity `k_1`, and `k_1 <= k`, so it sits
**at** the requirement rather than below it — depth remains the mechanism that reaches that ceiling when one
layer's `r - N` falls short.

**T5.2 (generic additivity at zero drift — FALSE AS STATED; see Status and the corrected law below).** At zero drift `Ctil_l = P_{col(X_l)^perp} A_{l,0}` with
`A_{l,0}` independent Gaussians across `l`. Then almost surely

    rank J_F(z*)  =  min( k_1 ,  sum_{l=1}^{L} q_l ),        q_l = min( r_l - N , rank M_l ).

So depth **is** additive, up to two ceilings: the per-layer budget `r_l - N`, and the global nesting ceiling `k_1`.

> **This equality is FALSE as stated (audit F11, 2026-09-17).** It assumes the per-layer row spaces
> `row(Ctil_l M_l)` spread independently inside `row(M_1)`; they cannot, because `row(M_l)` nests (T5.1). When
> `q_l = rank M_l` a layer contributes `row(M_l)` exactly, so two deep layers whose row spaces have collapsed to
> the same subspace are counted twice. The **corrected law** (candidate, confirmed 12/12 on the counterexample
> family, not yet proved):
>
>     rank J_F = min_j ( d_j + sum_{l<j} q_l ),   d_j := rank M_j,  j = 1..L+1,  d_{L+1} := 0.
>
> T5.2 is only the `j=1` and `j=L+1` terms; it omits every intermediate nesting constraint. R2 (independence of
> `A_{l,0}`) is **not** where it fails — the failure is geometric and survives independent seeds.

**T5.3 (what this says about the "~200 equations" objection — CONDITIONAL; see Status).** The row count `r_l - N` is an *upper bound* on the
useful contribution; the effective one is `q_l = min(r_l - N, rank M_l)`. Two regimes:
- `r - N >= k_1`: **one layer already saturates**; depth adds nothing and the objection is moot in the other
  direction — the constraint was never rank-limited.
- `r - N < k_1`: depth is the mechanism that closes the gap, and `L >= k_1/(r-N)` layers suffice generically.
This is the precise sense in which "many layers" means "more information", and it is falsifiable by M3.

**T5.4 (with drift — CONJECTURE).** At nonzero drift the `Ctil_l` are correlated through the shared trajectory
(each depends on all upstream adapters), so independence fails and T5.2's a.s. argument does not apply. I
conjecture additivity survives with the same formula because the correlation enters at `O(eps^perp)` while rank is
a discrete, generically stable quantity — but I have no proof, and rank is exactly the kind of quantity that a
small correlated perturbation can degrade to *numerical* rank deficiency (small singular values) without changing
exact rank. **The honest object is therefore the effective rank at a stated tolerance, not the rank.**

## Assumptions

| # | assumption | used where |
|---|---|---|
| R1 | `Phi_l^0` is the *base* feature map, common to all layers | T5.1's nesting — it is what makes the deeper Jacobians factor |
| R2 | `A_{l,0}` independent across layers | T5.2. Violated by a shared seed, by tied adapters, or by a single adapter reused across blocks |
| R3 | `rank(Ctil_l) = r_l - N` | T5.2; fails under T4-C1 rank death |
| R4 | zero drift | T5.2 only; see T5.4 |

## Proof

**T5.1.** `Phi_l^0 = psi_l . Phi_{l-1}^0` holds by definition of a feed-forward base network, so
`M_l = J_{psi_l}(.) M_{l-1}` and `ker M_{l-1} subset ker M_l`. The `l`-th block of `J_F` is `Ctil_l M_l`, whose
kernel contains `ker M_l` and hence `ker M_1`. Therefore `ker J_F = intersect_l ker(Ctil_l M_l) contains ker M_1`,
so `rank J_F <= k - dim ker M_1 = rank M_1`. []

**T5.2.** Work inside `W := row(M_1) subset R^k`, `dim W = k_1`; by T5.1 every block's row space lies in `W`. Fix
`l`. `Ctil_l` is `P_{col(X_l)^perp}A_{l,0}`, and conditionally on `col(X_l)` the matrix `P_{col(X_l)^perp}A_{l,0}`
is a Gaussian matrix supported on an `(r_l-N)`-dimensional subspace, independent of everything at other layers
(R2). Then `row(Ctil_l M_l) = M_l^T (col-space of Ctil_l^T)`, which is a uniformly distributed `q_l`-dimensional
subspace of `row(M_l) subset W` (the image of a Haar-random subspace under a fixed injective-on-`row(M_l)` map),
with `q_l = min(r_l - N, rank M_l)` a.s. For independent, absolutely continuously distributed subspaces
`V_1,...,V_L` of a `k_1`-dimensional space, `dim(V_1 + ... + V_L) = min(k_1, sum dim V_l)` a.s. — proof by
induction on `L`: given `V_{<L}` of dimension `d`, a uniformly random `q_L`-dimensional subspace meets it in
dimension `max(0, d + q_L - k_1)` a.s., since non-generic intersection is a positive-codimension condition on the
Grassmannian and therefore null. Summing gives the formula. []

**T5.3** is arithmetic on T5.2. **T5.4** is stated as a conjecture; see below.

## Exactly where each assumption enters

- **R1** is the whole content of T5.1. Note it is a statement about the *base* network: the certificates are
  evaluated at `Phi_l^0`, not at the trained features, which is exactly what makes them stackable at all. If one
  instead stacked the *trained* feature maps, the nesting would still hold but each `Phi_l` would depend on the
  private data and the object would no longer be attacker-computable.
- **R2** is the crux of additivity in the *proof*: it says the leakage channels at different layers are
  independent because their LoRA initialisations are. I predicted that a release tying the initialisations would
  therefore collapse `sum_l q_l` toward `max_l q_l`, and offered it as a **defence**. **MEASURED 2026-09-07 (job
  688036): it does not.** Shared-seed and independent both give `rank J_F = [9, 18, 20, 20]`. The certificates
  still differ through the layer-specific feature span `U_l` and Jacobian `M_l`, which is enough to keep the row
  spaces in general position. **The defence claim is withdrawn**; what remains is the weaker conjecture that tying
  the *entire adapter* (initialisation, feature span and all) would collapse it, which is not a realistic release
  and is untested. R2 remains a sufficient hypothesis for the proof; its converse is false.
- **R3**: under T4-C1 the deep `q_l` are zero and the sum truncates to the shallow layers.

## Counterexample search

1. **Nested/contractive features (T4-C4).** `rank M_l` decreasing in `l` makes `q_l` decay, so the sum saturates
   below `k_1` in practice even though the formula allows more. Prediction: `q_l` measured layerwise should *fall*
   with depth in a contractive net and be flat in a normalised one.
2. **A linear chart containing the private affine hull.** If `G` is affine and `col(J_G) supset` the span of the
   private features' preimages, several layers can share the same informative directions; but by T5.2 they still
   add generically, because the *random* `Ctil_l` pick independent subspaces of the shared `row(M_l)`. So this is
   NOT a failure mechanism at zero drift — recorded because it was expected to be one and is not.
3. **Deterministic correlation between layers.** Shared seed (R2) — a real failure, and a defence.
4. **Rank vs effective rank.** The only serious threat to T5.2 in practice: `sum_l q_l` counted at exact rank can
   be much larger than the rank at a numerically meaningful tolerance. M3 must report the singular spectrum of
   `J_F`, not a rank integer.

## Status

- **T5.1 PROVED** (nesting ceiling), confirmed independently.
- **T5.2 FALSE as stated.** The `min(k_1, sum q_l)` equality fails because `row(M_l)` nests (T5.1), so the per-layer
  row spaces cannot spread independently. Measured: 0/12 seeds match T5.2, 12/12 match the corrected law
  `rank J_F = min_j(d_j + sum_{l<j} q_l)` (job 350996; audit F11, 2026-09-17). The corrected law is a **candidate**,
  confirmed on that counterexample family, **not yet proved**. R2 is not the cause — the failure is geometric and
  survives independent seeds.
- **T5.3 conditional.** `L >= k_1/(r-N)` is derived from T5.2 and holds only when `rank M_l = k_1` at **every**
  adapted layer (rank-preserving frozen path); otherwise the ceiling is `min_j(d_j + sum_{l<j} q_l)`, an **upper
  bound** on what depth delivers, and layers past a collapsed `d_j` add nothing. The 8-adapted-layers figure for
  `k_1 = 128` at `r=24, N=8` is that upper bound. Precondition: measure `rank M_l` per adapted layer first.
- **T5.4 CONJECTURE** (nonzero drift), unchanged, now sitting on top of a corrected T5.2. Under drift the honest
  object is effective rank at a stated tolerance (pre-register a ladder), not exact rank — and the elbow's
  stability across the ladder is the result, since an elbow that moves with the cut is a property of the cut.

## Numerical sanity check

`experiments/multilayer_cert/theory_checks.py::check_T5`. **RUN 2026-09-07, job 688036:**

| layers stacked | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| measured `rank J_F` | 9 | 18 | 20 | 20 |
| predicted `min(k_1, sum q_l)` | 9 | 18 | 20 | 20 |

with `k_1 = 20 = k`, `q_l = 9 = r - N` at every layer. This is a random FP64 MLP (widths 30 >= k), so it is
**rank-preserving** (`d_j = k_1` at every layer), and there the corrected law provably collapses to
`min(k_1, sum q_l)` — the two laws **coincide**. So this cell confirms **T5.1** and confirms the additivity **in
its valid (rank-preserving) regime**, but it discriminates **neither** law from the other (read-rows verification,
job 688036 not the survival sweep, `notes/m4_additivity_verification_2026-09-17.md`). The shared-seed arm gives the
identical `[9, 18, 20, 20]` and so **refutes** the R2 defence prediction above. The discrimination lives only in a
contracting net (`d_j < k_1`), where the counterexample gives 0/12 for T5.2 and 12/12 for the corrected law (F11).
