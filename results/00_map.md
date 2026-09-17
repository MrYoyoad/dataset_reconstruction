# Orientation map — what exists on disk, before the first job of the 7 Sept brief

Written by the executor session before any compute, per §12.0. Paths, job ids, and which release is which. **Where
the disk contradicts the brief, the disk wins and the discrepancy is logged here.**

## 1. Core code (located, not invented)

| what | path |
|---|---|
| replay / exact inversion (SGD and Adam modes, `train_release`, `simulate_sgd_reduced`, `invert_lm`) | `experiments/exact_inversion/lora_exact_inversion.py` |
| certificate `C = Pi A_T`, and `lm_cert` (the certificate-route solver) | `experiments/exact_inversion/certificate.py` |
| the 19/60 replay cell, both routes on one release | `experiments/exact_inversion/affine_chart_two_routes.py` |
| head-adapter cells on real backbones (CIFAR-10 MLP / CNN, added classes) | `experiments/cifar/cifar_newclass.py` |
| chart x layer x solver grid, isolation test, blend diagnostic | `experiments/cifar/cifar_charts.py` |
| linearised-representer vs certificate head-to-head, variable projection | `experiments/cifar/ntk_vs_certificate.py` |
| oracle-chart ladder (E8-adjacent, running) | `experiments/oracle_ladder/ladder_cell.py` |

## 2. The releases

| release | where | shape | fp | q vs N | use |
|---|---|---|---|---|---|
| **affine two-routes** — the 19/60 replay result | row `results/exact_inversion/affine_two_routes_331384.jsonl`; generator `affine_chart_two_routes.py` | k=12, N=8, r=24, m=20, d=P=64, T=400, lr=0.05, seed=1 | **fp64** | rank C = 16, r = 24 -> **q = 8 = N** | E1B-tiny (§12 names it) |
| head-adapter cells on real backbones | `results/cifar_newclass/*.pth` (35 cells) | N=8, r=64, k=32/48, T=400, d=1000 (MLP) or 256 (CNN) | fp64 | q = N = 8 (rank B_T = 8 in every row) | the cells where Phi0 is a real network |
| oracle ladder, raw privates | `results/oracle_ladder/` (28 cells, **complete**) | N=8, r=64, k=32, T=400 | fp64 | reported per cell | E8-adjacent |
| precision-varied letter releases | `results/exact_inversion/step80_760909/` | r=64, k=32, N=8 | fp64/fp32/bf16/fp16 | q = 8 | precision caveats (rule 9: not Gal-facing) |

`results/exact_inversion/` holds 115 `.pth` releases; the fp64 lab releases with known seeds are there
(`spectrum_sgd_*`, `step11_sgd_*`, ...).

## 3. DISCREPANCIES — the disk wins, logged per §12.0

**D1. The release §12 names for E1B makes three of its own day-one measurements vacuous.** In
`affine_chart_two_routes.py` the world is `phi = identity`: the adapted layer *is* the input layer. Consequences,
all structural rather than incidental:

- **The manifold test is vacuous.** A5 asks for `min_x ||Phi0(x) - h_hat||` beside the chart residual. With
  `Phi0 = id`, `range(Phi0) = R^d`, so *every* candidate is a real feature and the residual is identically zero.
  The zero-set question of §9 (`Z_image <-> Z_feature ∩ range(Phi0)^N`) cannot be asked on this release at all.
- **The image-space comparison is vacuous.** §12 asks for feature replay against image replay on matched starts.
  With `Phi0 = id` the two are the same computation, so there is no cost saving to measure and no basin difference.
- **`k = 128` exceeds `d = 64`.** The chart-distance measure at k in {32, 64, 128} degenerates: at k >= 64 a PCA
  chart of a 64-dimensional space is the whole space and the residual is zero by construction.

**Resolution, and it changes no instruction:** this release still answers the question E1B exists to answer — *can
replay recover H\* when H is free* — and it is the release §12 names, is fp64, and satisfies A1's `q = N`
precondition. It is run as specified. The three vacuous measurements are reported as vacuous with the reason,
**not** as zeros, and the manifold and cost questions are additionally run on a **real-backbone** release from
`results/cifar_newclass/` where `Phi0` is an actual frozen network (d = 1000 for the MLP, 256 for the CNN). That
addition is flagged as an addition, not as the named cell.

**D2. E5 is already done, and the brief tags it `new`.** Job 331384 matches its spec clause for clause: linear
chart, `k = 12` below both capacity lines and above `N-1`, certificate landings **0 of 60** with every returned
point an exact affine combination (coefficient sum 1.000000000 at every one of 60 starts, max deviation 8.9e-16), isolation rank 5 = `k - (N-1)`,
and replay on the same release and the same starts recovering all eight from **19 of 60** with **zero aliases**.
Two of those figures were quoted loosely elsewhere and are stated precisely here after a row check (this session,
against the 60 per-start rows): the coefficient sum is **1.000000000 at every one of the 60 starts, maximum
deviation 8.9e-16** — *not* "min equal to max", which is literally false (min 0.99999999999999911, max
1.00000000000000044) and is disprovable by opening the file. And the two bars must stay separate: **18** starts
return all eight with worst-image error at or below **2.2e-15** and replay residual below 1e-12; a **19th** clears
the 1e-2 landing bar at **4.4e-3**. The summary field `replay_best_err_min` = 6.6e-16 is the best SINGLE image in
the best start and is not a per-start worst case.
Already noted in the plan audit as A8. Retag `done`; do not re-run.

**D3. The "20 attacker-buildable starts" are referenced in `experiments/exact_inversion/constrained_replay.py`,
`multilayer_budget.py`, `train_precision.py` and `RESULTS.md`**, not in a single artefact file. E2 will have to
regenerate them from the constrained-replay path rather than load them; noted so it is not discovered mid-experiment.

## 4. Day-one status

| item | status |
|---|---|
| 00_map.md (this file) | done, before the first job |
| E1B-tiny | RUN. A2 pre-check **passed before any solve** (not flat; sharp minimum at alpha=1, first-order). Jobs **670990** seed_known, **670993** seed_free (unreduced baseline), **675031** reduced seed-free (E3, gate-first) |
| E4a | RUN as **674521** (DINO) and **674524** (CLIP). First attempt (670533/670536) **died at a shape bug before any measurement** — the blend matrix was built N-transposed; nothing was reported from it |
| E6 request | **drafted, NOT SENT** — `notes/e6_code_request_draft.md`. Was blocked (§11's spec was not on disk); the approver supplied §11/§12 verbatim. Two paragraphs to Haim and Irani for the stage-2 code paths (DIP+cosine for DINO/ViT; Karlo UnCLIP for CLIP), plus the rebuild-locally table and §13's questions. Sending is Gal's call and the user's. |
| oracle ladder (E8-adjacent) | **all 28 cells COMPLETE**, jobs 435271-435322; rows in `results/oracle_ladder/rows.jsonl` |

## 4b. Registered predictions — written BEFORE the rows land

Timestamped here so a deviation cannot be reinterpreted after the fact.

**P1 — E1B reduced arm (job 675031), the scalar c.** The arm is *reduced seed-free, 513 unknowns, of which 512 are
the theorem's and one is a deliberate diagnostic slack*. The theorem (W1): `Pi A_T = Pi A_0 + Pi A_0 H M_T H^T`,
and the second term vanishes because `Pi` annihilates `col(A_0 H)`, so under plain gradient descent with no weight
decay the scalar is **exactly 1**, derivably — not a fitted constant. The gate confirmed it from the *seed* side at
3.409e-15 with `A_0` known. The free `c` confirms it from the *search* side, from starts that never saw `A_0`.

**This release IS plain SGD with no weight decay, so the prediction is `c_hat = 1` to machine precision. A
departure from 1 on THIS release is a HARNESS BUG, not a discovery** — the parametrisation or the reconstruction
would be wrong, and it must be chased as a defect and never written up as evidence about optimisers. The
diagnostic reading of `c` (a departure signalling weight decay or a non-SGD optimiser) becomes available only on
releases whose recipe we do not control.

**Why `c` is left free rather than pinned at the theorem's value.** Pinning it would import "the release was
trained with plain gradient descent and no weight decay" into the *attacker's* parametrisation, trading the
certificate route's recipe-free property for one unknown out of 513 — no measurable gain for a quiet weakening of
the threat model. The pinned-`c` variant (`q*d` unknowns) is a **contingency, not a deliverable**: it is worth
running only if the free-`c` arm fails to land, where it would isolate whether that one unknown was the
obstruction. If the free-`c` arm lands, the pinned arm adds nothing the gate has not already given.

**P2 — E1B scale (jobs 670990 / 670993).** Registered in the plan audit before the solve and already measured by
the no-solve pre-check: the replay residual is not flat under `(alpha h*, A_0/alpha)`. Measured: sharp
machine-precision minimum at `alpha = 1` (7.5e-16), rising **linearly** on both sides, two-sided slope 0.414,
halving the offset dividing the residual by 2.05. Linear rather than quadratic means the zero is non-degenerate in
the scale direction, so scale is **first-order** identifiable. The prediction still open is `||h_hat||/||h||` near
1 in the seed-free arm.

## 4b-bis. E1B: the fibre is MEASURED, and P3/P4/P5 are registered before the LM arms land (jobs 350928, 350940)

The local dimension of the family of `(H, seed)` reproducing the release, as the nullity of the residual Jacobian
at the truth. This is an IDENTIFIABILITY measurement and owes nothing to any solver.

| parametrisation | seed | objective | unknowns | rank | **nullity** |
|---|---|---|---|---|---|
| free `H` | known | full `A_T`+`B_T` | 512 | 512 | **0** |
| free `H` | free | full `A_T`+`B_T` | 2048 | 1816 | **232** |
| reduced seed | free | full `A_T`+`B_T` | 1025 | 793 | **232** |
| chart `k=12` | known | full `A_T`+`B_T` | 96 | 96 | **0** |
| chart `k=12` | free | full `A_T`+`B_T` | 1632 | 1632 | **0** |
| free `H` | free | *v1's* `A_s@H` | 2048 | 344 | **1704** |
| chart `k=12` | free | *v1's* `A_s@H` | 1632 | 288 | **1344** |

Three things follow, none of which a counting argument gave correctly:

- **The reduction does not change identifiability.** 232 both ways. The reduced arm removed 1023 unknowns and
  exactly 1023 of the Jacobian's rank, leaving the fibre untouched — because every point of the fibre already
  satisfies `Pi A_0 = Pi A_T`, so the family lies INSIDE the reduced slice rather than transverse to it. The
  approver's intersection argument (32 + 1025 − 2048 < 0, therefore isolated) assumed a genericity that the
  construction itself destroys. The reduction is a conditioning gain, not an identifiability gain.
- **The whole ambiguity is a seed-against-`H` trade.** All 232 directions move `H`, but **0** of them move `H`
  with the seed held fixed. So with a known seed the free-`H` problem is well posed, and with a free seed it is not.
- **The CHART is what makes the seed-free problem well posed** — `k=12` takes the nullity from 232 to 0. That is
  why job 331384 recovers: not because replay is strong, but because the chart removes exactly the `H` directions
  that trade against the seed.

**P3 (seed_known, job 350929): nullity 0, so recovery is POSSIBLE and v1's total failure there was the SOLVER.
Predict landings > 0 under LM.** If LM also returns nothing from 40 starts, the obstruction is the basin, not the
information, and that is a different finding from v1's.

**P4 (seed_free, job 350930): nullity 232, so the truth is NOT identifiable. Predict the ALIAS signature** — LM
drives the residual near zero at points whose per-image error stays large. A landing here would falsify the
nullity measurement and must be chased as a contradiction, not reported.

**P5 (reduced, job 350931): nullity 232, identical to seed_free. Predict it behaves like seed_free and NOT like
seed_known.** The reduction was sold as a factor of three in search dimension; if it also changed outcomes, the
nullity table is wrong.

## 4c. Rank thresholds — verified, not assumed (job 688520)

The multilayer lane found that a purely relative threshold `s > rtol*s[0]` calls a numerically **zero** matrix
**full** rank (a zero matrix's own `s[0]` is at rounding level, so every singular value clears the bar), and that
this inverted the sign of a result there. Every rank in the E1B scripts uses that form. Checked on this release
against an **absolute** floor tied to `||A_T||` or `||H||` (`experiments/e1b/rank_threshold_check.py`):

| object | rank, relative | rank, absolute | agree | gap at the cut |
|---|---|---|---|---|
| `B_T` | 8 | 8 | yes | s[7]=2.53e-01 → s[8]=4.09e-16, ratio **6.2e14** |
| `C` | 16 | 16 | yes | s[15]=4.34e-01 → s[16]=2.04e-15, ratio **2.1e14** |
| `H` | 8 | 8 | yes | full rank, s[-1]=1.98 |
| `A_T` | 24 | 24 | yes | full rank, s[-1]=4.18e-01 |

So `qb = rank B_T = 8 = N` and `q = r - rank C = 24 - 16 = 8 = N` are the same number reached two independent ways,
each sitting in a fourteen-order spectral gap rather than near a threshold. **The hazard is real but inapplicable
here** — none of these matrices can vanish on this release (`B_T` is trained away from its zero init, `H` is data,
`A_T` is the released seed plus update). It would apply to any cell where the certificate can legitimately be
annihilated, and there the absolute floor is required. **Two such places are already on this project's roadmap, and
in the second the bug fails in the direction that matters:**

1. **The collapse regime.** `rank C = r - q` goes to zero as `q` approaches `r`, so `C` becomes numerically zero
   exactly at the capacity boundary — where this project's sharpest measurements sit. A relative threshold there
   reports a rank that is pure noise structure.
2. **Defence evaluation — where an absolute floor is MANDATORY.** The merged and balanced-factorisation defences
   give `C = 0` identically. A relative threshold applied to a merged release would report `C` as **full rank**,
   i.e. report as OPEN a channel the defence has CLOSED, and conclude the defence does not work. That is wrong in
   the direction that gets a paper attacked. **No defence evaluation in this project may use a relative-only rank
   threshold**, and merging is the named case.

## 5. Standing constraints carried from the brief

Rule 1: report **q and N side by side**, always; `q = r - rank C` is a span dimension and never an image count.
Rule 2: never present a raw equation count as identifiability. Rule 10 language, and A3's required continuation —
"not ruled out by count" is followed in the same sentence by *identifiability additionally requires the chart to
meet span(H) only at the training points (Lemma 15); the count does not test that*. Rule 9: precision results stay
out of Gal-facing material. Rule 13: self-trained releases only.
