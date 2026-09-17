# T1 — Deep representation perturbation

## Statement

**T1.1 (linear network, exact — no remainder).** Let `sigma = id`. Then for every `t` and every `l`,

    Delta_{l+1,t} = sum_{j<=l} ( prod_{i=j+1}^{l} T_{i,t} ) s B_{j,t} A_{j,t} H_j^0 ,
    T_{i,t} := W_i^0 + s B_{i,t} A_{i,t}

with **no** remainder term. Separating orders in the adapter size, the single-layer recursion

    Delta_{l+1,t} = W_l^0 Delta_{l,t}  +  s B_{l,t} A_{l,t} H_l^0        (first order)
                  +  s B_{l,t} A_{l,t} Delta_{l,t}                        (second order, the ONLY one)

shows the second-order content of a linear LoRA network is exactly one term: *an adapter acting on an
already-perturbed representation*.

**T1.2 (smooth nonlinear).** Let `sigma` act entrywise with `|sigma'| <= L1`, `|sigma''| <= L2`. Write
`d_l := ||Delta_{l,t}||_F`, `h_l := ||H_l^0||_F`, `w_l := ||W_l^0||_op`, `beta_l := s ||B_{l,t}A_{l,t}||_op`,
`kappa_l := L1 w_l`. Then

    d_{l+1} <= kappa_l d_l + L1 s beta_l h_l  +  [ L1 s beta_l d_l + (L2/2) ( w_l d_l + s beta_l (h_l + d_l) )^2 ]

and, dropping the bracket (second order),

    ||H_{l,t} - H_l^0||_F  <=  K_l^{T1} * sum_{j<l} s beta_{j,t} + O(eps^2),
    K_l^{T1} := max_{j<l} ( L1 h_j prod_{i=j+1}^{l-1} kappa_i ).

**Depth law.** `K_l^{T1}` contains a product of `l-j` layer gains `kappa_i = L1 ||W_i^0||_op`. If the base network
has `kappa > 1` the admissible adapter size for a fixed drift budget shrinks **geometrically in depth**,
`beta <~ kappa^{-l}`. The perturbative regime is a statement about the *base network's* gain, not about LoRA.

## Assumptions

| # | assumption | why it is here |
|---|---|---|
| A1 | `B_{l,0} = 0` at every layer | makes `H_{l,0} = H_l^0` exactly, i.e. the base network is the `t=0` network |
| A2 | `sigma` acts entrywise, `|sigma'|<=L1`, `|sigma''|<=L2` | Taylor with uniform remainder. GELU (this repo) satisfies it; **ReLU does not** (see below) |
| A3 | adapters are the only trainable objects; `W_l^0` frozen | otherwise `T_{i,t}` is not `W^0 + sBA` |
| A4 | the same `N` examples at every step (no minibatching) | only so that `Delta` is a fixed-shape `n_l x N` matrix; the bound is per-column and survives minibatching |

## Proof

**T1.1.** `H_{l+1,t} = T_{l,t} H_{l,t}` and `H_{l+1}^0 = W_l^0 H_l^0`. Subtract:

    Delta_{l+1,t} = T_{l,t} H_{l,t} - W_l^0 H_l^0
                  = T_{l,t}(H_l^0 + Delta_{l,t}) - W_l^0 H_l^0
                  = (T_{l,t} - W_l^0) H_l^0 + T_{l,t} Delta_{l,t}
                  = s B_{l,t}A_{l,t} H_l^0 + T_{l,t} Delta_{l,t}.

This is an exact affine recursion in `Delta` with `Delta_{1,t} = 0` (A1 plus the standing assumption that the
input to the first adapted layer is frozen). Unrolling it gives the displayed sum, exactly. Splitting
`T_{l,t}Delta = W_l^0 Delta + sB A Delta` and noting `Delta = O(beta)` by induction gives the order separation. []

**T1.2.** `Delta_{l+1,t} = sigma(Z_{l,t}) - sigma(Z_l^0)` with
`delta Z_{l,t} = W_l^0 Delta_{l,t} + sB_{l,t}A_{l,t}(H_l^0 + Delta_{l,t})`. Entrywise Taylor with Lagrange
remainder gives `|Delta_{l+1}| <= L1|delta Z| + (L2/2)|delta Z|^2` entrywise, hence in Frobenius norm
`d_{l+1} <= L1 ||delta Z||_F + (L2/2) ||delta Z||_F^2` (using `|| |M|^2 ||_F <= ||M||_F^2` entrywise, valid since
`|| . ||_F` is absolute and monotone). Then `||delta Z||_F <= w_l d_l + s beta_l (h_l + d_l)` by submultiplicativity
of `||.||_op` against `||.||_F`. Substituting and discarding the products of two small quantities gives the linear
recursion `d_{l+1} <= kappa_l d_l + L1 s beta_l h_l`, whose solution with `d_1 = 0` is the discrete convolution
`d_l <= sum_{j<l} (prod_{i=j+1}^{l-1} kappa_i) L1 s beta_j h_j`, which is bounded by the displayed `K_l^{T1}` times
`sum_j s beta_j`. []

## Exactly where each assumption enters

- **A1** is used twice and is load-bearing both times: it sets `Delta_{1,t} = 0` (the initial condition of the
  recursion — with `B_0 != 0` the base network is *not* the `t=0` network and every statement below shifts), and it
  is the base case of the closure induction in T2.
- **A2** is used only in T1.2, at the Taylor step. For **ReLU** `sigma''` is a delta and the remainder bound fails
  *at the points where an activation flips sign*. The repair: on the event `E` = "no unit changes activation
  pattern between the base and the trained network on the private inputs", ReLU is locally linear and T1.1 applies
  exactly, with `sigma'(Z^0)` a fixed 0/1 mask. `P(E)` is itself a measurable, testable quantity (fraction of
  flipped units), and it is the honest place where a ReLU statement needs an extra hypothesis. This repo's nets are
  GELU, so A2 holds as stated.
- **A3** is used in writing `T_{i,t} - W_i^0 = sB_iA_i`.
- **A4** is cosmetic (shape bookkeeping) and can be dropped.

## Counterexample search

- **Does the bound's exponential depth factor actually bite, or is it an artefact of submultiplicativity?**
  It bites. Take `sigma = id`, `W_i^0 = c I` with `c > 1`, a single adapter at layer 1 with `sB_1A_1 = g v u^T`.
  Then `Delta_{l+1} = c^{l-1} g v (u^T H_1^0)`, so `d_{l+1}/h_{l+1} = (c^{l-1} g ||v|| ||u^T H^0||)/(c^l h_1)` —
  the *relative* drift is constant here because the base representation grows at the same rate. The exponential
  factor is therefore real in absolute terms but **cancels in the relative drift for a homogeneous base net**. This
  is a genuine sharpening: `eps_l` should be measured relatively, and for a norm-preserving base network
  (`kappa ~ 1`, which is what normalisation buys) drift does *not* explode with depth. Recorded as a prediction
  for M1: **relative drift grows sub-exponentially in well-normalised networks.**
- **Is the second-order term ever dominant?** Yes, whenever `beta_l >~ w_l`, i.e. the adapter is comparable to the
  frozen weight. Then the expansion is meaningless and only T2's exact statement survives. This is the honest
  boundary of the perturbative language, and it is one of the axes of M2.

## Status

**PROVED** (T1.1 exact; T1.2 under A2). The depth law is a bound, and the counterexample above shows the
*relative* version can be far tighter than the bound in normalised networks — do not quote the exponential as a
prediction without measuring `kappa_i`.

## Numerical sanity check

`experiments/multilayer_cert/theory_checks.py::check_T1` — builds a random 6-layer GELU LoRA net in FP64, trains
`T` steps, and verifies (i) the exact unrolled formula of T1.1 for `sigma = id` to machine precision, (ii) that the
measured `d_l` obeys the T1.2 bound, and (iii) the log-log slope of `d_l` against `sum_j beta_j` is 1.
