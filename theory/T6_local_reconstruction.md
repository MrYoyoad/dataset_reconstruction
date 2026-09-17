# T6 — From certificate residual to latent reconstruction error

## Statement

**T6.1 (exact case — local identifiability).** If `F(z*) = 0`, `rank J_F(z*) = k`, and `J_F` is `L`-Lipschitz on
`B(z*, R)`, then `z*` is the unique zero of `F` in `B(z*, min(R, 2 sigma/L))`, `sigma := sigma_min(J_F(z*))`.

**T6.2 (approximate case — stability).** Let `rho := ||F(z*)||` (the certificate residual at the truth, nonzero
because the certificates are truncated, T2/T3), and let `zhat` minimise `||F||` over `B(z*, R')` with
`R' := min(R, sigma/L)`. Then

    ||zhat - z*||  <=  4 rho / sigma        provided   4 rho/sigma <= R'.

**T6.3 (the chain — what the whole track buys).** Composing T1 -> T2 -> T6.2:

    ||zhat - z*||  <~  4 * ||Ctil|| ||H^0|| * K_l * eps_l^perp / sigma_min(J_F(z*)),
    eps_l^perp  <~  K_l^{T1} * sum_{j<l} s beta_j  (orthogonal component only).

Latent error is (orthogonal drift) x (a conditioning constant) / (stacked-Jacobian conditioning). Every factor on
the right is measurable without ground truth **except** `eps^perp` — so this is a bound the attacker can *almost*
evaluate, and the missing factor is exactly what M1/M2 measure.

## Assumptions

| # | assumption | used where |
|---|---|---|
| S1 | `F` continuously differentiable, `J_F` `L`-Lipschitz on the ball | Taylor with quadratic remainder in both proofs |
| S2 | `rank J_F(z*) = k` (full latent rank) | `sigma > 0`; supplied by T5 |
| S3 | `zhat` is a minimiser *within the ball* | T6.2 controls the local basin only; it says nothing about global aliases |

## Proof

**T6.1.** Suppose `F(z') = 0`, `z' in B(z*, rad)`, `u := z' - z*`. Then
`0 = F(z') - F(z*) = J_F(z*)u + Rem`, `||Rem|| <= (L/2)||u||^2`, so
`sigma||u|| <= ||J_F(z*)u|| = ||Rem|| <= (L/2)||u||^2`, giving `||u|| >= 2 sigma/L` unless `u = 0`. []

**T6.2.** `||F(zhat)|| <= ||F(z*)|| = rho` since `z*` is in the feasible ball and `zhat` minimises. Then
`||F(zhat) - F(z*)|| <= 2 rho`. With `u := zhat - z*`,
`||F(zhat)-F(z*)|| >= ||J_F(z*)u|| - (L/2)||u||^2 >= sigma||u|| - (L/2)||u||^2`. On `||u|| <= sigma/L` the
right-hand side is `>= sigma||u||/2`. Combining, `sigma||u||/2 <= 2 rho`, i.e. `||u|| <= 4rho/sigma`. []

**T6.3** substitutes T2.2's bound on `rho = ||Ctil H^0||`-type residuals (summed over layers, which only changes
constants) into T6.2, and T1.2's bound on `eps^perp`. []

## Exactly where each assumption enters

- **S1** at the single Taylor step in each proof; `L` is the Jacobian's Lipschitz constant along the chart, i.e. a
  curvature property of `Phi^0 . G`, and is estimable by finite differences.
- **S2** is the entire content of T5 — without full stacked rank, `sigma = 0` and T6.2 is vacuous: the residual
  can be tiny with `zhat` arbitrarily far along the null direction. This is precisely the "alias" verdict of the
  exact-inversion track (residual zero, wrong image), and T6 makes the connection quantitative: **aliases are
  `sigma = 0`, basin failures are `rho` large.** The two verdicts the project already separates by convention are
  separated here by which factor of the same bound fails.
- **S3** is the honest limitation: T6 is local. It bounds the error of the *right* basin and says nothing about how
  many basins there are. Global statements need the phase-boundary counting of the Rev-10 framework
  (`k < r - N`), which T5 now generalises to `k_1 < sum_l q_l`.

## Counterexample search

- **Is the constant 4 tight?** No, it is the usual slack from `||F(zhat)|| <= rho` plus the halved quadratic. A
  sharper `2rho/sigma` holds if `zhat` is a stationary point of `||F||^2` rather than a minimiser. Not worth
  chasing.
- **Can `rho` be small and `zhat` still far?** Yes, when `sigma` is small but nonzero — the bound degrades
  smoothly and correctly. This is the "on the line" behaviour observed in the single-layer phase diagram
  (0.67-0.9 recovery exactly on `k = r - N`), which T6.2 now explains as `sigma -> 0`.

## Status

**PROVED** (T6.1, T6.2 are standard; recorded in full because the whole track's value is the chain T6.3, and each
link must be checkable). The novelty is not the inverse-function argument, it is that T2 supplies a *measurable*
`rho` and T5 a *measurable* `sigma` for the multilayer certificate.

## Numerical sanity check

`experiments/multilayer_cert/theory_checks.py::check_T6` — perturbs `z*`, measures `||F||` and `sigma_min(J_F)` by
autograd, and checks the predicted `4rho/sigma` envelope contains the observed `||zhat - z*||` over a sweep of
truncation levels (which is the cleanest way to vary `rho` at fixed geometry).
