# E1B — free feature replay: forget the images, make H the variable, ask whether the dynamics invert

## Setup

The release §12 names: the affine two-routes cell (the job-331384 generator), `d = P = 64`, `N = 8`, `r = 24`,
`m = 20`, `k = 12`, `T = 400` SGD steps, `lr = 0.05`, seed 1, **fp64**, rebuilt deterministically. The unknown is
`H` in `R^{d×N}` directly — no chart — and the question is whether replaying the training dynamics determines it.

Three arms: **seed known** (`A_0` given), **seed free** (`A_0` unknown, initialised at the released `A_T`, which is
attacker-available), and **reduced seed free** (`A_0 = C/c + P_b M`, exploiting that the release pins the seed's
off-span component). 60 Gaussian + 60 ker-`C` starts per arm, matched across arms by construction.

## Job ids + configs + precision

| job | arm | solver | status |
|---|---|---|---|
| **670990** | seed known | Adam, 400 iters, lr 5e-2 | done |
| **670993** | seed free (unreduced baseline) | Adam | done |
| **675031** | reduced seed free, 513 unknowns | Adam | done |
| **688520** | rank-threshold verification | — | done |
| **693342** | arm-separation check | — | done |
| **350928 / 350940** | **fibre dimension (identifiability)** | — | done |
| 350929 / 350930 / 350931 | the three arms again under **Levenberg–Marquardt** | LM | running |

All fp64. Rows in `results/e1b/rows.jsonl`; tensors in `results/e1b/*.pth`.

## Raw numbers

**The two no-solve checks, both of which ran before anything was searched.**

*A1* — `rank H = 8 = N` and `q = r − rank C = 24 − 16 = 8 = N`, asserted before the solve, so the per-image metric
legitimately exists. Verified against an absolute rank floor (job 688520): the cuts sit in spectral gaps of
6.2e14 (`B_T`) and 2.1e14 (`C`), so no rank here is threshold-sensitive.

*A2, the scale pre-check* — the replay residual along `(αh*, A_0/α)`:

| α | 0.5 | 0.8 | 0.9 | 0.95 | **1.0** | 1.05 | 1.1 | 1.25 | 2.0 |
|---|---|---|---|---|---|---|---|---|---|
| residual | 2.8e-1 | 9.2e-2 | 4.4e-2 | 2.1e-2 | **7.5e-16** | 2.0e-2 | 3.9e-2 | 9.2e-2 | 2.8e-1 |

Not flat. A sharp machine-precision minimum at the true scale, rising **linearly** on both sides — two-sided slope
**0.414**, and halving the offset divides the residual by 2.05. Linear rather than quadratic means the zero is
non-degenerate in that direction: **the scale is first-order identifiable**, pinned to order 1e-15 against the fp64
floor, not merely identifiable in principle.

*The E3 gate* — `Pi A_T = c_T Pi A_0` holds to **3.409e-15** with **`c_T` = 1.000000000000**, which is derivable
rather than fitted (`Pi` annihilates `col(A_0 H)`, so the update term vanishes under plain GD with no decay).

**The three Adam arms: zero landings from 120 starts each.**

| arm | landings | objective min | (truth) | best max per-image error | median cosine to assigned truth | `‖ĥ‖/‖h‖` |
|---|---|---|---|---|---|---|
| seed known | **0 / 120** | 3.580e-02 | 8.750e-16 | 1.0155 | 0.484 G / 0.418 K | 0.967 / 0.981 |
| seed free | **0 / 120** | 3.533e-02 | 8.750e-16 | 1.0174 | 0.323 G / 0.314 K | 0.669 / 0.669 |
| reduced | **0 / 120** | 3.565e-02 | 8.750e-16 | 0.9863 | 0.385 G / 0.340 K | seed err 1.28 |

Every arm stalls **fourteen orders above** the residual at the truth, so nothing reached a zero at all. Per the
verdict rule this is `optimisation failure (residual not zero)` and **not** `alias` — the two must not be merged.

**The identifiability measurement (jobs 350928 / 350940).** The local dimension of the family of `(H, seed)`
reproducing the release, as the nullity of the residual Jacobian at the truth. This owes nothing to any solver.

| parametrisation | seed | objective | unknowns | rank | **nullity** |
|---|---|---|---|---|---|
| free `H` | known | full `A_T` + `B_T` | 512 | 512 | **0** |
| free `H` | free | full `A_T` + `B_T` | 2048 | 1816 | **232** |
| reduced seed | free | full `A_T` + `B_T` | 1025 | 793 | **232** |
| chart `k=12` | known | full `A_T` + `B_T` | 96 | 96 | **0** |
| chart `k=12` | free | full `A_T` + `B_T` | 1632 | 1632 | **0** |
| free `H` | free | the v1 objective | 2048 | 344 | **1704** |
| chart `k=12` | free | the v1 objective | 1632 | 288 | **1344** |

Of the 232-dimensional family, **all 232 directions move `H`**, and **0** move `H` with the seed held fixed.

## Claims

**C1 — The answer to E1B's decision line is PARTIAL, and the partition is exact.** *Dynamics invertible from a
free `H`:* **yes with a known seed** (nullity 0 — `H` is locally determined), **no with a free seed** (a
232-dimensional family reproduces the release exactly, so no solver can pick the truth out of it). The entire
ambiguity is a **seed-against-`H` trade**: every direction of the family moves `H`, and none moves `H` alone.

**C2 — The chart is what makes the seed-free problem well posed, and that is why job 331384 recovers.** Constrain
`H` to the `k=12` chart and the seed-free nullity falls from **232 to 0**. 331384's 19-of-60 recovery is therefore
not evidence that replay is strong on free features; it is evidence that the chart removes exactly the `H`
directions that trade against the seed. **This is the sharpest statement in the file**: the chart's role is not
merely to shrink a search space, it is to restore identifiability.

**C3 — The reduced parametrisation is a conditioning gain, NOT an identifiability gain.** Nullity 232 both
reduced and unreduced. The reduction removed 1023 unknowns and exactly 1023 of the Jacobian's rank, leaving the
fibre untouched, because every point of the fibre already satisfies `Pi A_0 = Pi A_T` — **the family lies inside
the reduced slice rather than transverse to it**. An intersection argument (`32 + 1025 − 2048 < 0`, therefore
isolated) assumes a genericity that the construction itself destroys, and equation counting also got the fibre's
size wrong: predicted 32, measured 232, because 200 of the 2016 equations are dependent.

**C4 — The v1 objective was wrong twice, and both are worth recording.** It matched `A_s @ Hc` against `A_T @ H`
rather than `A_s` against the released `A_T`. That is (a) only `r*N = 192` equations instead of `r*d = 1536`,
leaving a nullity of **1704**, and (b) **not attacker-available** — its target is built from the true `H`, the very
unknown being solved for. The corrected residual uses only released quantities and is the one the LM arms use.

**C5 — The registered ker-`C` prediction held.** On this linear chart, ker-`C` starts did **not** beat random
ones — median objective 5.34e-02 against 4.48e-02 in the seed-known arm, i.e. slightly worse, in every arm.

## What is NOT claimed

**The Adam arms do not show that recovery is impossible.** In the seed-known arm the nullity is 0, so recovery is
possible in principle and Adam's total failure there is a **solver** result, not an information result. The LM
arms (350929–350931) test exactly that, and their predictions are registered in `results/00_map.md` §4b-bis
before they land: P3 seed-known should land; P4 seed-free should show the **alias** signature (low residual, large
per-image error); P5 reduced should behave like seed-free and not like seed-known.

**The `c_hat` values from the reduced Adam arm test nothing.** They came out at 2.4–5.7 against a theorem that
says exactly 1. That is **not** the harness bug the pre-registration warned about, because no start converged —
these are stalled iterates, not solutions, and the prediction applies to solutions. P1 remains **untested**.

**A5, the direct manifold test, is VACUOUS on this release and is reported as vacuous rather than as a zero.**
`phi = identity` here, so `range(Phi_0) = R^d` and `min_x ‖Phi_0(x) − ĥ‖` is identically zero for every candidate
by construction. The real-backbone cell is a separate addition, not this one.

**Scope of any landing here** (carried in every row as a `scope` field): `Z_feature` and `Z_image` coincide on this
release, so a landing certifies that the dynamics invert from a free `H` and certifies **nothing** about free
feature replay landing on representations no image produces — this cell cannot exhibit that failure.

## Kill-criterion check

The decision line §12 asks for is answered (C1) rather than killed. No kill criterion was met: the reproduction
gates passed (residual at the truth 8.75e-16), A1's precondition held, and the A2 pre-check came back not flat, so
the derivation it guards is intact. Had A2 been flat, the instruction was to stop before spending the solve; it
was not.

## Audit log

- **A2 was run before any solve and reported regardless of outcome**, as instructed. It passed, and the *shape* of
  the pass (linear, not quadratic) turned out to carry more than the pass itself.
- **The two arms reported an identical median objective at their first checkpoint**, which would have been the
  signature of the seed-free arm being a relabelled copy. Checked directly (job 693342) rather than by re-reading
  the code: the seed block's relative gradient is 3.40e-01 against the `H` block's 1.33e-02 and the seed moves by a
  relative 1.15–1.25, so the arms are separated. The identical medians were a **format string** — `%.2e` shows
  three significant figures, so 4.4512e-02 and 4.4548e-02 both render as `4.45e-02`.
- **The approver's counting argument was checked rather than taken**, and both of its conclusions failed: the fibre
  is 232-dimensional, not 32, and the reduction does not isolate the truth. Measured, not re-derived.
- **Rank thresholds were verified against an absolute floor** after a sibling lane found that a relative-only
  threshold calls a numerically zero matrix full rank. Clean here, and the hazard is now recorded as a prohibition
  for defence evaluation, where a merged release gives `C = 0` identically.
