# T3 — Is the truncation error `O(eps^2)`? No.

## Statement

**T3.1 (no cancellation; the `O(eps^2)` conjecture is FALSE).** There is a two-step, rank-1, `N=1`,
two-layer-linear instance in which the truncated certificate error is exactly first order in the drift, with a
closed-form coefficient:

    ||Ctil H^0||  =  |c| * ||P_{x^perp} A_0 delta|| * |(d_0+d_1)^T d_1| / ||d_0+d_1||^2   +  O(eps^2),
    c := 1 + eta^2 s^2 (d_0^T d_1)(||h||^2 + delta^T h)          [the scalar in A_2 h = c x]

where `x := A_0 h` (`h` the single private feature at the adapted layer), `delta := Delta_{l,1}` the drift at step
1, and `d_0, d_1` the backpropagated error vectors at steps 0 and 1.  **`c` is NOT negligible**: the first draft
of this file dropped it as "1 + O(eta^2) ~ 1", which is false whenever `eta^2 (d_0^T d_1) ||h||^2` is not small --
at `eta = 0.7`, `||h||^2 ~ 30` it is `~ 40`, and the numerical check missed the closed form by a factor of 49
until `|c|` was restored (measured 2026-09-07, job 683234 -> 686846).  The *order* in `eps` is unaffected. The coefficient is a nonzero rational
function of `(d_0,d_1)` and vanishes only on the measure-zero set `(d_0+d_1)^T d_1 = 0`. **Therefore `O(eps)` is
sharp and no structural cancellation exists.**

**T3.2 (the true small parameter — the cancellation that DOES exist).** The error depends on `delta` only through
`P_{x^perp}A_0 delta`, and `P_{x^perp}A_0 delta = 0` whenever `delta in col(H^0)` (with `N=1`, `delta parallel h`).
Hence the first-order coefficient of the *in-span* drift is exactly zero, and the controlling quantity is the
**orthogonal drift**

    eps^perp := max_t || P_{col(H_l^0)^perp} Delta_{l,t} ||_F / ||H_l^0||_F     ( <= eps_l ).

This is a genuine first-order cancellation — just not the one the brief hoped for. It is a statement about *which
component* of the drift is harmless, not about the *order* in that component.

**T3.3 (blow-up of the constant).** The coefficient `|(d_0+d_1)^Td_1| / ||d_0+d_1||^2` is **unbounded**: it
diverges as `d_1 -> -d_0`, i.e. when the accumulated error signal nearly cancels. In that limit
`sigma_1(B_T) -> 0`, `gap -> 0`, and T2.2's `Gamma/gap` diverges consistently. So a nearly-trivial release is not
merely uninformative — its truncated certificate is *badly* wrong, and `gap` is the attacker-visible warning.

## Assumptions

Two adapted layers; `N=1`; the layer of interest has `B_0 = 0`, `A_0` Gaussian; `T=2` SGD steps; `d_0, d_1`
linearly independent (else `N' = 1` and everything is exact); `delta` not parallel to `h` (else T3.2).

## Proof (complete, by explicit computation)

With `B_0 = 0`, the `A`-update at `t=0` vanishes, so `A_1 = A_0 =: A`. Write `H_0 = h`, `H_1 = h + delta`,
`x := Ah`, `xi := A delta`, `kappa := eta^2 s^2`. Then

    B_1 = -eta s d_0 (A h)^T = -eta s d_0 x^T
    B_2 = B_1 - eta s d_1 (A H_1)^T = -eta s [ (d_0+d_1) x^T + d_1 xi^T ]
    A_2 = A - eta s B_1^T d_1 H_1^T = A + kappa (d_0^T d_1) x (h+delta)^T

**Step 1: the full certificate is exact.** `A_2 h = x [ 1 + kappa (d_0^Td_1)(||h||^2 + delta^T h) ] in span{x}`,
and `R := row(B_2) = span{x, xi}` (rank 2, since `d_0,d_1` independent), so `P_{R^perp}A_2h = 0`. This is
Proposition A with `N' = 2` — consistent, and it confirms the exactness is not an artefact of `T=1`.

**Step 2: the truncated certificate.** Decompose `xi = xi_par u + xi_perp` with `u := x/||x||`, `xi_perp perp u`,
`||xi_perp|| = ||P_{x^perp}A delta|| = O(eps)`. Then

    -B_2/(eta s) = a u^T + b xi_perp^T ,    a := (d_0+d_1)||x|| + d_1 xi_par ,   b := d_1 .

The Gram matrix is `M^TM = ||a||^2 uu^T + (a^Tb)(u xi_perp^T + xi_perp u^T) + ||b||^2 xi_perp xi_perp^T`.
First-order eigenvector perturbation of the top eigenpair (eigenvalue `||a||^2`, unperturbed eigenvector `u`,
coupling matrix element `(a^Tb)||xi_perp||`, energy denominator `||a||^2 - 0`) gives the top right-singular
direction

    Rhat = span{ u + theta * hat{xi}_perp } + O(eps^2),    theta = (a^T b) ||xi_perp|| / ||a||^2 .

By T2.1, `Ctil h = P_{R (-) Rhat} A_2 h`, and `A_2 h = c x` with `c = 1 + O(eta^2)`. Since
`R (-) Rhat = span{ hat{xi}_perp - theta u }`,

    ||Ctil h|| = |c| ||x|| |sin theta| = |c| ||x|| theta + O(eps^2)
               = ||xi_perp|| * (a^Tb)/||a||^2 * ||x|| + O(eps^2)
               = |c| ||P_{x^perp}A delta|| * (d_0+d_1)^T d_1 / ||d_0+d_1||^2 + O(eps^2),

using `a = (d_0+d_1)||x|| + O(eps)` and `b = d_1`, and keeping the scalar `c` from `A_2 h = c x`. []

## Exactly where each assumption enters

- `B_0 = 0` gives `A_1 = A_0`, which is what makes the whole computation closed-form.
- `d_0, d_1` independent is what makes `N' = 2 > N = 1`, i.e. what makes truncation *necessary*. With `d_1` a
  multiple of `d_0`, `row(B_2) = span{(d_0+lambda d_1)}`-side is still rank 1 in `m` but rank 2 in `r`; the honest
  statement is that `N'=2` needs `xi notin span{x}`, i.e. exactly the T3.2 condition.
- Gaussianity of `A_0` is used nowhere in the algebra; it only makes the generic conditions hold a.s.

## Counterexample search (against my own claim)

I searched for a mechanism that would force `a^Tb = 0` structurally, since that is the only way to get `O(eps^2)`:

- **Does the projection `P_{Rhat^perp}` annihilate the first-order drift term by construction?** No. The identity
  T2.1 shows the error is `P_{R (-) Rhat}A_TH^0`, and `A_TH^0` is (to leading order) `x`, which has an
  `O(eps)` component in `R (-) Rhat` precisely because `Rhat` is rotated off `u` by `O(eps)`. The projection
  removes the `O(1)` part and leaves the `O(eps)` part; it cannot remove what it created.
- **Is there a gauge/symmetry argument?** The LoRA symmetry `(A,B) -> (GA, BG^{-1})` acts on `R` and `Rhat`
  simultaneously and leaves `Ctil H^0` invariant, so it constrains nothing about the order in `eps`.
- **Does averaging over the Gaussian `A_0` kill the first-order term?** `E[||P_{x^perp}A_0 delta||] > 0` strictly
  (it is a norm of a nondegenerate Gaussian), so no.
- **Is the coefficient small in practice even if nonzero?** `(d_0+d_1)^Td_1/||d_0+d_1||^2` is `~1/2` when
  `d_0 ~ d_1` (slowly-varying error signal, the common case), so it is `O(1)`, not small. Prediction for M1: the
  measured `rho_l / eps_l^perp` should sit near `0.5` in the slow-drift regime, and blow up as `gap` shrinks.

## Status

**FALSE** — for the conjecture `||Ctil H^0|| = O(eps^2)`. **PROVED** — that `O(eps)` is attained (T3.1) and that
the in-span component of the drift contributes at order zero (T3.2, the repair). Any statement of the multilayer
theorem must therefore be written in `eps^perp` and must not promise a quadratic rate.

## Numerical sanity check

`experiments/multilayer_cert/theory_checks.py::check_T3`. **RUN 2026-09-07, job 688036, all PASS:**

| check | result | tolerance |
|---|---|---|
| full certificate exact (`C_full h`) | `1.5e-14` | `<1e-12` |
| closed-form coefficient (with `\|c\|`) | `0.45%` max rel. error over `eps in [1e-6,1e-2]` | `<2%` |
| **log-log slope in `eps`** | **`1.0004`** | `\|slope-1\|<0.02` |
| in-span drift free, at `\|\|delta\|\| = \|\|h\|\|` (100% drift) | `4.7e-14` | `<1e-12` |

The slope is the falsification: an `O(eps^2)` rate would read `2.00`. (c) sweeping `d_1 -> -d_0` is not yet run.
