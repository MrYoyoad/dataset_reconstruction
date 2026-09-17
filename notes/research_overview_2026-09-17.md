# Research overview — current position, 2026-09-17

One page on where the thesis stands after reconciling the 2026-09-17 archive with the repository. Written to
replace `notes/science_state_2026-09-04.md` as the front page (that file keeps the better *template* — spine,
measured/conjecture/withdrawn tags, ranked open problems — but predates the affine two-routes cell, E4a, the
multilayer track, the oracle ladder and E1B). `notes/thesis_scientific_summary.md` remains the only organised
record of the **pre-certificate** program (leakage geometry, `q_eff`/`r_J`, the activation crux, direct inversion,
the gradient bridge) and is superseded as the front line.

**Where the detail lives.** Audited claims: `results/CLAIMS_LEDGER.md`. Definitions, theorem hypotheses and open
problems: `notes/technical_record_2026-09-17.md`. Per-track evidence: `experiments/<track>/RESULT(S).md` with rows
in `results/<track>/`. Artifact and code paths: `results/00_map.md`. Meeting decisions and backlog:
`notes/meeting_summary_2026-09-15.md` + `notes/meeting_2026-09-15_decisions_and_backlog.md`. Corrections:
`notes/math_rulings_2026-09-06.md` (R1–R20), `notes/plan_audit_2026-09-07.md` (A1–A17+),
`notes/corrections_from_archive_2026-09-17.md`.

---

## The thesis in one paragraph

A LoRA adapter is released as its two factors. We show the released factors alone determine a linear operator —
**the certificate** `C = P_{row(B_T)^⊥} A_T` — that annihilates every representation the adapter recorded, with no
recipe, no labels, no seed and no batch size. Candidate images are then restricted to a low-dimensional **chart**
(a public image family, currently PCA), and the search minimises the scale-free certificate residual over chart
coordinates, one image at a time. On real backbones this hands back private images exactly from random starts,
with a wrong-release control at zero. What the method does **not** yet have is a chart that contains real
photographs: every image recovered so far is the *chart projection* of a private image, and the public charts
measured sit one to one-and-a-half orders too coarse to contain the raw ones.

## The mechanism, and what is proved about it

- **The certificate is exact, at any finite step, under a named assumption stack**: `B_0 = 0` and a generic
  initialisation; vanilla SGD (no Adam, no per-entry scaling); inputs to the adapted layer fixed *or confined to a
  fixed subspace*; `q < r ≤ n` with `q = rank H` the **feature rank**, not the image count; and **excitation**
  `rank B_T = q` at release. Proof: `notes/w1_certificate_proofs.tex` (repo) and
  `notes/gal_2026-09/The_LoRA_certificate.pdf` (the Gal-facing note). It gives exactly `r − q` equations per
  candidate and `ker C = col(H) ⊕ ker A_0`.
- **Excitation is computable, not verifiable, from the release.** `rank B_T ≤ q` always, so the attacker gets a
  lower bound on `q`; the hypothesis is the case of equality and testing it needs `H`.
- **The chart is a prior restriction, not another measurement.** Local identifiability needs `k ≤ r − q`; global
  alias-freedom on a fixed chart needs the strict `k < r − q` **and** that the chart meet the private span only at
  the private points. The count alone is never identifiability: every blend of the private features is an exact
  certificate zero. Proofs and twelve counterexamples: `notes/gal_2026-09/chart_inversion_theory.pdf`.
- **At depth the certificate survives exactly but loses rank.** The closure induction needs a fixed subspace, not
  frozen inputs, so it runs at any layer with the training span in place of the private span: measured residual
  `6.5e-15` at 923% drift, rank law `r − N'` held 110/110, and a **lifetime** — empty once `T ≥ min(r,n_l)/N`. The
  truncated certificate the attacker uses to buy rank back carries a first-order error in the **orthogonal** drift
  (slope 1.0004); in-span drift is free.
- **The certificate and the LoRA-aware linearised (NTK-style) fit have the same zero set** where every image is
  recorded. The certificate is the per-candidate form, the representer the joint form plus an independence clause.
  Measured gaps between them are solver and search-arity gaps, not information.

## What is measured, with the cells

| result | cell | number |
|---|---|---|
| New class recovered recipe-free from random starts | EMNIST 'a' as an 11th class, r=64, k=32, N=8, FP64 (job 760909) | 38.4% of 500 starts land, **8/8 images** |
| Colour images, adapter behind a nonlinearity | CIFAR head / hidden layer (jobs 252897–297325) | 171/400 and 253/400 starts, **8/8**; wrong-release control **0/400** |
| Pixel-layer failure, structural | same study | **0/8**; every returned point an exact blend |
| Route split on one release | affine two-routes (job 331384) | certificate **0/60**, replay **19/60** with zero aliases |
| Replay identifiability | E1B Jacobian nullity (jobs 350928/350940) | seed known **0**; seed free **232**; oracle chart k=12 **0** |
| Public chart fidelity | oracle ladder, 28 cells | landing survives chart error ≤1.24–1.86%; public PCA sits at 0.24–0.32 and lands **nothing** |

## The three things that would change the picture

1. **A chart that contains raw photographs.** This is the binding constraint on the whole image line, and it is
   now quantified as a bracket rather than a slogan: the shortfall is 17–54× at k=32 and never approaches 1 at any
   width tested (`results/CLAIMS_LEDGER.md` §0.1). A learned nonlinear chart is the agreed next attempt.
2. **Certificates at an adapted inner layer of a trained network, under moving inputs.** Gal's own question. The
   theory is in place on both sides (exact-over-the-span vs truncated-with-`O(ε_⊥)`-error); the reconstruction has
   never been run with the adapter inside a backbone.
3. **Text.** An exact finite-codebook separation theorem exists on paper, together with a stated obstruction
   (contextual features can fill the rank, and then `C ≡ 0`) and a second route for known initialisation. No
   transformer experiment has been run.

## Direction, after the 2026-09-15 meeting

The supervision frame changed: the figures, the certificate equation and the chart were well received, the
direction is agreed, and the emphasis is practical — multilayer, stronger charts, better reconstruction, an initial
text attempt. The participants said extensions to text or new image domains could be groundbreaking, conditional on
success. The NTK conclusion they accepted is scoped to the formulation and setting discussed and must not be
quoted as an impossibility result.

## Standing honesty rules that shape every claim here

Observe, don't conclude. A count is never identifiability. Verdicts never merge *residual not zero* (optimisation
failure) with *residual zero, wrong image* (alias). A number sourced only to our own prose is not a measurement.
Rank thresholds need an absolute floor, never a relative one. Leakage numbers bound the **weakest** attacker, so
they are a lower bound on leakage, never an upper bound on what an attacker could do.
