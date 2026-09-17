# theory/ — Multilayer certificate track

Research track opened 2026-09-07. Each file is one proposed theorem in the mandated format:
**Statement | Assumptions | Proof attempt | Where each assumption is used | Counterexample search | Status |
Numerical sanity check.**

`Status` is one of **PROVED / CONJECTURE / FALSE**. A numerical check agreeing with a statement never promotes it
to PROVED — the status line records the state of the *proof*. Nothing here has been independently re-derived by a
sibling session yet, so nothing here may enter a Gal-facing PDF.

## Notation (fixed across the track)

Layer `l` of an `L`-layer network, adapted by LoRA:

    Z_{l,t} = W_{l,t} H_{l,t},   W_{l,t} = W_l^0 + s B_{l,t} A_{l,t},   H_{l+1,t} = sigma(Z_{l,t})

with `A_l in R^{r x n_l}`, `B_l in R^{m_l x r}`, `A_{l,0}` iid Gaussian, `B_{l,0} = 0`, `N` private examples as the
columns of `H_{l,t} in R^{n_l x N}`, and `D_{l,t} := dL/dZ_{l,t} in R^{m_l x N}` the backpropagated error. SGD:

    A_{l,t+1} = A_{l,t} - eta s B_{l,t}^T D_{l,t} H_{l,t}^T
    B_{l,t+1} = B_{l,t} - eta s D_{l,t} (A_{l,t} H_{l,t})^T

(simultaneous update, both from time-`t` values — matching `experiments/cifar/cifar_newclass.py`).

`H_l^0 := H_{l,0}` is the base-network representation: the network at `t=0` *is* the base network, because
`B_{l,0} = 0` at every layer. This identity is used constantly below.

| symbol | meaning |
|---|---|
| `Hcal_l := sum_{t<T} col(H_{l,t})` | the **training span** at layer `l`; `N'_l := dim Hcal_l >= N` |
| `Delta_{l,t} := H_{l,t} - H_l^0` | representation drift |
| `eps_l` | relative drift, `max_t ||Delta_{l,t}||_F / ||H_l^0||_F` |
| `eps_l^perp` | **orthogonal** drift, `max_t ||P_{col(H_l^0)^perp} Delta_{l,t}||_F / ||H_l^0||_F` — the quantity that actually controls the error (T3) |
| `R := row(B_{l,T})`, `q := rank B_{l,T}` | the leaked row space |
| `C_full := P_{R^perp} A_{l,T}` | the **full** certificate (rank `r - q`) |
| `Ctil := P_{Rhat^perp} A_{l,T}`, `Rhat` = top-`N` right-singular subspace of `B_{l,T}` | the **truncated** certificate (rank `r - N`) |
| `rho_l := \|Ctil_l H_l^0\| / (\|Ctil_l\| \|H_l^0\|)` | scale-free certificate residual |
| `beta_{l,t} := s \|B_{l,t} A_{l,t}\|_op` | adapter operator norm |

## The one-paragraph summary of what this track found

The perturbative framing in the track brief is **not** the right one, and the mathematics says something sharper.
The closure induction that makes `CH = 0` exact at the first adapted layer **does not need the inputs to be frozen** —
it needs them to stay in a *fixed subspace*, and it runs verbatim at any depth with the training span `Hcal_l`
replacing `col(H_l^0)`. So the full certificate is **exact at every layer, at any drift** (T2, Prop. A), and
`H_l^0` is annihilated exactly because the base representation is literally the `t=0` member of the span. What
depth costs is not accuracy but **rank**: `rank C_full = r - N'_l`, and the certificate dies discontinuously when
`N'_l` reaches `r`. The approximate, `O(eps)` object of the brief is what appears only once the attacker
*truncates* the row space back to `N` to buy rank back — and there the error is an exact identity,
`Ctil H_l^0 = P_{R (-) Rhat} A_T H_l^0`, first order in the **orthogonal** drift, with **no cancellation**
(T3, closed-form counterexample). In-span drift is free.

## Files

| file | question | proof status | measured (2026-09-07) |
|---|---|---|---|
| [T1](T1_deep_representation_perturbation.md) | how does drift grow with depth? | PROVED | unrolling exact to 3.4e-16; drift slope in `beta` = 1.000 |
| [T2](T2_certificate_error_bound.md) | does the certificate survive at depth? | PROVED (exact, over the training span) | `rho_full` **6.5e-15 at 923% drift**; rank law 110/110; death ladder 4,2,0 |
| [T3](T3_first_order_cancellation.md) | is the truncation error `O(eps^2)`? | **FALSE** — `O(eps)` is sharp | slope **1.0004**; coefficient to 0.45%; in-span drift free at 100% |
| [T4](T4_counterexamples.md) | when does the whole picture fail? | 4 modes, 3 explicit | contamination `O(1)`: median 2.3e-4, max 0.27 |
| [T5](T5_cross_layer_rank.md) | does depth add information? | PROVED (zero drift) / CONJECTURE (with drift) | `rank J_F` = 9,18,20,20 = prediction. **R2 defence claim REFUTED** |
| [T6](T6_local_reconstruction.md) | residual -> latent error | PROVED | bound holds on both rows where its minimiser hypothesis is met |

Numerical checks: `experiments/multilayer_cert/` (WEXAC, FP64), results
[experiments/multilayer_cert/RESULTS.md](../experiments/multilayer_cert/RESULTS.md), jobs 688036 (checks) and
692603 (sweep).

**One measured law the theory did not predict**, and which the practical picture now rests on: the training span
inflates at the maximal rate `N' = N*T`, so a deep certificate has a **lifetime** — `rank = min(r,n_l) - N*T`,
empty once `T >= min(r,n_l)/N`. The governing condition at depth is a small *training span* (few steps, few
examples, low-rank drift), not small drift.
