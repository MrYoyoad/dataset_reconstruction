# E4a — how many dimensions does a public chart of FEATURES need?

## Setup

Gal's realism objection, operationalised on the feature side. The premise under test, stated in the script's own
docstring: *"Feature space should need far fewer dimensions [than pixel space]; this measures how many."* Two frozen
foundation backbones, three chart regimes, three chart families, five widths. Measured quantity in every cell is the
**relative projection residual of the eight private feature vectors** onto the chart, and — separately — the residual
of **blends** of those same privates.

Hold-out is **by subject**: the universal pool excludes the private class entirely, and the target pool is public
images *of* the private class drawn from the train split, disjoint from the eight held-out test privates.

## Job ids + configs + precision

| job | backbone | d | status |
|---|---|---|---|
| **674521** | DINO ViT-B/16 (`vit_base_patch16_224.dino`) | 768 | done, 31 rows |
| **674524** | CLIP ViT-L/14 (`vit_large_patch14_clip_224.openai`) | 1024 | done, 31 rows |
| ~~670533 / 670536~~ | both | — | **DIED before any measurement**, shape bug in the blend matrix (built `64 x N` instead of `N x 64`). Nothing was reported from them. Fixed with assertions on the blend shape and on the column sums |

Private class `motorcycle`, N=8 held-out CIFAR-100 test images; universal pool 2000 train images excluding that
class; target pool 500 train images of that class. Charts: PCA, linear autoencoder, nonlinear autoencoder (300
epochs, 2000 samples). k ∈ {8,16,32,64,128}. Seed 1. **fp32** (embeddings; this is a chart-geometry measurement, not
an inversion — no fp64 claim is made or needed). Rows in `results/e4a/rows.jsonl`, tensors in `results/e4a/e4a_*.pth`.

## Raw numbers

Private residual / blend residual, PCA charts:

| regime | backbone | k=8 | k=16 | k=32 | k=64 | k=128 |
|---|---|---|---|---|---|---|
| universal | DINO | 0.7130 / 0.5545 | 0.6546 / 0.4958 | **0.5991** / 0.4429 | 0.5181 / 0.3745 | 0.4243 / 0.3082 |
| universal | CLIP | 0.4651 / 0.3455 | 0.4356 / 0.3152 | **0.4078** / 0.2932 | 0.3701 / 0.2624 | 0.3289 / 0.2301 |
| target | DINO | 0.5151 / 0.2598 | 0.4705 / 0.2401 | 0.4091 / 0.2014 | 0.3522 / 0.1712 | 0.2880 / 0.1441 |
| target | CLIP | 0.3213 / 0.1506 | 0.2973 / 0.1389 | 0.2634 / 0.1283 | 0.2290 / 0.1039 | **0.1963** / 0.0901 |

Shared-concept structure, both axes at the **same 95% variance threshold** (so the two counts are commensurable):

| backbone | k_shared (over 20 concepts) | k_nuisance (within the target class) |
|---|---|---|
| DINO | 15 | 205 |
| CLIP | 16 | 208 |

## Claims

**C1 — CORRECTED (see audit log): at MATCHED construction, CLIP features are comparable to pixels and DINO
features are worse; the blanket claim "features are worse than pixels" does NOT survive.** The first version of
this claim compared a pixel chart against a feature chart built differently. The oracle ladder's public PCA chart
is fitted on public train images **of the added (private) class** — a *target*-regime chart. Matched against E4a's
own target rows at k=32:

| chart at k=32, fitted on public images OF the private class | private projection residual |
|---|---|
| pixels (ladder public PCA, keyboard / motorcycle releases) | 0.2432 – 0.3176 |
| CLIP ViT-L/14 features | **0.2634** — *inside* the pixel range |
| DINO ViT-B/16 features | 0.4091 — worse than pixels |

What survives, and it is the part that answers the realism objection: **no public chart of features comes close to
being good enough.** The best cell measured anywhere is CLIP/target/k=128 at 0.1963, and conditioning on the
private category helps at every k without closing the gap. Moving to a frozen foundation-model embedding does not
dissolve the chart problem. What must NOT be said is that it *worsens* it — that reading came from comparing a
category-conditioned pixel chart against a random-public feature chart, and it dies under matched construction.

**C2 — Lemma 15's signature reproduces in feature space, at every cell of both backbones and both regimes.** Blends
of the privates sit roughly twice as close to the public chart as the privates themselves (DINO target k=32: 0.4091
against 0.2014; CLIP target k=128: 0.1963 against 0.0901). The affine-hull degeneracy is **not a pixel artefact** —
it is a property of what public charts represent well, and it holds in a frozen ViT's embedding space where it was
never claimed.

**C3 — The shared-concept parametrisation offers NO reduction here, because the measured structure is inverted
relative to the plan's illustration.** The plan proposes `k_shared + N*k_nuisance` as a cheaper unknown count,
illustrated with k_shared=32, k_nuisance=8, N=8 → 96 unknowns against 256, and states that whether reality has that
structure is what E4a measures. Measured: the concept axis is **tiny** (15–16 directions) and the within-concept
nuisance axis is **enormous** (205–208). At matched 95% fidelity the arithmetic is

    shared-concept:  k_shared + N*k_nuisance  =  15 + 8*205  =  1655
    per-image:                  N*k_nuisance  =       8*205  =  1640

so the shared part does not amortise anything — it **adds** 15 dimensions to a cost already dominated by per-image
nuisance. Pose, lighting, crop and background are where the dimension lives; identity is not. This removes a regime
from the plan rather than adding one.

> **A comparison NOT made, and why.** Against a per-image chart at k=128 the shared-concept count would look 60%
> worse (1655 against 1024), but k=128 is *below* the 95% threshold (DINO target residual 0.288 there), so that
> compares two parametrisations at different fidelities and is void. The fidelity-matched comparison above is the
> one that holds, and it is decisive without needing the inflated version.

**C4 — The linear-autoencoder assertion PASSED.** The linear AE tracks PCA to three decimals at every k in every
regime (DINO universal k=32: 0.6009 against 0.5991), as theory requires when the decoder is linear. This is the
arm's theory check and it is reported as passed rather than assumed.

## What is NOT claimed

**The nonlinear autoencoder rows are evidence about a 300-epoch fit on 2000 samples, not about nonlinear charts.**
It is worse than PCA at every cell, and in DINO universal it is **non-monotone in k** — 0.6843 at k=32 against
0.7064 at k=64. Capacity rose and the fit worsened, which can only be an optimisation failure. These rows must not
be cited as evidence about nonlinear charts in either direction.

**No ratio is quoted against the oracle ladder's landing gate.** It is tempting: the ladder's gate is a measured
projection error at or below 0.0124 (motorcycle/MLP) or 0.0045 (keyboard/CNN), which would make 0.1963 some 16 to 44
times too coarse. But that divides a **feature**-space residual on a foundation backbone by a gate measured in
**pixel** space on the CIFAR MLP and CNN releases — different space, different data, different release, different
search. The gate is a property of a release and its search, not a universal constant. **Whether the gate transfers
to a feature-space attack is unmeasured**, and until it is, the ratio is stated only conditionally if at all. C1's
like-for-like comparison carries the weight instead.

## Kill-criterion check

No kill criterion for E4a was supplied in what this session holds, so none is reported as met or unmet. The result
that *would* have triggered one — charts failing to fit at all, or the hold-out leaking — did not occur: residuals
decrease monotonically in k for both PCA arms in both regimes, and the universal pool excludes the private class by
construction.

## Audit log

- The first submission of both backbones **died before any measurement** on a transposed blend matrix. Reported as a
  failure, not silently resubmitted; assertions on the blend shape and column sums now make it non-silent.
- yoado-8b drew C3 out of the shared-concept numbers, which I had reported as an aside rather than as a result, and
  required the commensurability check on the two variance thresholds (both 95% — checked, they are commensurable).
- yoado-8b required the conditional on the gate ratio; C1 was rewritten to carry the weight like-for-like instead.
  The arithmetic in their version of C3 (1655 against 1024) compares at unmatched fidelity and is not used; the
  fidelity-matched form is above.
- **C1 was rewritten after a second audit.** Its first form said features are *worse* than pixels, comparing the
  ladder's pixel chart against E4a's *universal* rows. But the ladder's public PCA is fitted on train images of the
  added class (`load_cifar100_class(...)["train"]`, verified at source in `ladder_cell.py`), so it is a *target*
  chart and the matched counterpart is E4a's target rows — where CLIP sits inside the pixel range. The comparison
  was unmatched in the direction that flattered the conclusion. Raised by yoado-c8; verified here at the source
  rather than accepted.
- The pixel-space comparison figure was taken from this repo's own measured ladder rows (0.2432 / 0.3176), not from
  the 0.25 quoted in the script docstring, which this session refused as circular (it is sourced only to prose I
  wrote, with no job row behind it).
- **That refusal turned up a mislabelled quantity in the brief itself, and it is not ours to fix silently.** Traced
  by yoado-8b: the 0.25 and 0.01 pair originates at `experiments/cifar/RESULT.md`, where it is a **certificate
  residual** at a chart projection against a blend floor. §6.1 of the brief restates the same pair as *"private
  images sit off a 32-dimensional public PCA chart by a **projection residual** of 0.25 while blends of privates sit
  within 0.01"*. A certificate residual evaluated at a chart projection is not a chart's projection error, and the
  headline this session was asked to write would have divided one by the other. C1 uses measured projection errors
  with job ids instead. **The brief's §6.1 still carries the mislabelling and should be corrected at the source
  rather than worked around here** — flagged to the ledger (yoado-35) rather than quietly superseded.
