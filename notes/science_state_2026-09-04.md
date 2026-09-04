# Where the science stands — 2026-09-04

**Purpose:** basis for the supervisor meeting, Monday. **Written by the claims-audit lane (yoado-b9)**, which
checked the numbers below against `results/exact_inversion/*.jsonl` rather than against the write-ups. Sources:
`experiments/exact_inversion/RESULTS.md`, `STATUS.md`, `notes/assumption_relaxation_program.md`,
`notes/exact_channel_rev10.tex`.

**Reading rules.** Every claim carries a tag: **[M]** measured in this repo with a job id · **[D]** derived, not
yet measured · **[C]** conjecture · **[W]** withdrawn, recorded so it is not resurrected. No number appears
without the job that produced it. Where a recovery quality is quoted, the **on-chart caveat** applies unless
stated otherwise: the adapter was trained on the chart's projections, so those projections *are* the private
data and are what is recovered; against a raw image the residual error is the chart's, not the attack's.

---

## 1. The spine, in one page

> **Document order is not pitch order.** This section runs measured-breadth → single-cell demonstration, which
> is right for a state-of-science record. **In the meeting the certificate leads as the *tool* and the letters
> cell (e) is its demonstration**, because the supervisor's stated objection was "no tool", not "no result".
> Anyone building the talk from this file should invert §1's order.


**The setting.** A LoRA adapter `(A_T, B_T)` released after fine-tuning is a deterministic function of the
private data, the public recipe and the seed. Because `B₀ = 0`, the seed enters only through `X = A₀U`, `rN`
numbers; the rest of `A₀` is released verbatim. **[M]** (Theorem G; gauge invariance verified job 487882,
discrepancies 5.8e-16…2.1e-15.)

**(a) The imprint law — what gets recorded.** `B_T = Σᵢ Cᵢ`, and `‖Cᵢ‖` is proportional to the accumulated
softmax residual of image *i*: the release records *what the model had to learn*. **[M]** Kendall concordance
28/28 on the strong model; `rank B_T` = the number of images whose imprint clears the floor in all 40 batches
(jobs 631392, 650891). Recording is decided by **margin order within the batch**, not by an absolute
threshold: in all 9 rank-deficient batches of both jobs the below-floor images are *exactly* the highest-margin
ones, at drop counts 2 to 7 — joint probability ≈ 4e-13 under random assignment. Below-floor margins span
30–95 while above-floor reach 74, so there is no threshold; adding one hard example demotes the confident ones
rather than protecting them.

**(b) The certificate — a recipe-free channel.** `C = P_{row(B_T)^⊥} A_T` satisfies `C hᵢ ≈ 0` for recorded
images and not for the rest. **[M]** 1e-16…1e-8 against 0.1–1 for invisible images. It needs the release, the
public base model and a chart — **no recipe, no labels, no N, no start near the truth**. From-nothing recovery:
2000 random public-scale starts at k=6, **51.0% reach the floor and 51.0% land on a recorded image — the same
number**, so every floor-reacher is private and none is spurious; all seven recorded images found (job 706721).
*On-chart.*

> **One caveat, not a doubt.** Equal fractions alone would be consistent with the two sets overlapping only
> partially; what makes the identification *exact* at k=6 is the kernel count, which predicts the floor
> fraction to equal the recorded fraction there. That argument has been checked at k=6 and **not yet
> elsewhere in the chart range** — the k=16 check is recomputing (job 156607). This is the right reason to
> want the check and is independent of any measurement fault.

**(c) Two counting lines.** Reading the release through the certificate gives `r − N′` equations per image;
reading it through the full replay residual gives `(m−1) + r − N′`. **[M+D]** So

    certificate alone:  k < r − N'          replay (the exact channel):  k <= (m-1) + r - N'

with first collapse measured at `k = m + r − N′`; brackets are sharp to one unit of `k` at N=8 and N=14
(jobs 467914, 479587, 479684, 481079) and on real MNIST at r = 8/16/32 (job 568095), σ_min falling 11–12
orders across the unit.

**(d) The cap `N′ ≤ m−1`.** The softmax's zero-sum columns put `B_T ∈ 1⊥⊗ℝʳ`, so `rank B_T ≤ min(m−1, r, N′)`.
**[M]** Past it the certificate fails for *all* images at once, not gracefully: 20 optdigits on a 10-class head
give rank 9 and residuals 0.3–0.5 for every image, invariant across k ∈ {8…40}; the matched positive control on
a padded 26-logit head recovers 20 of 20 (job 725918).

**(e) The headline cell.** A release fine-tuned in **ordinary FP32** on a class the base model does not have
returns every private example to an attacker holding only the public model: EMNIST 'a' as an 11th class on the
98% MNIST MLP, r=64, k=32; 32.8% of starts land, **all eight found**, residuals 3e-7…1.3e-5 (jobs 760909,
764976, 771329). The adapter genuinely moved — `A_T` shifts 9.3%, feedback ratio 0.43, margins −2.65…−10.3 at
t=1 rising to +6…+14 by t=T. Chart error 0.235 → instance-level.

---

## 2. Measured / conjecture / withdrawn

| Claim | Tag | Evidence |
|---|---|---|
| Imprint ∝ accumulated softmax residual; rank = count above floor | **[M]** | 631392, 650891 (28/28; 40/40 batches) |
| Recording set by margin *order* within the batch | **[M]** | 9/9 rank-deficient batches, p≈4e-13 |
| Certificate vanishes at recorded truths, not elsewhere | **[M]** | 1e-16…1e-8 vs 0.1–1 |
| From-nothing recovery below the certificate line | **[M]** *on-chart* | 706721 (51.0%/51.0%, 7 of 7) |
| Capacity line `k < m + r − N′`, strict form from the simplex | **[M]** + **[D]** | 467914/479587/479684/481079; 568095 |
| Cap `N′ ≤ m−1`, catastrophic past it | **[M]** | 725918 and the 20-on-10 control |
| New-class exposure independent of model quality | **[M]** | 658575 (digits rank 6, cosine 0.04; letters rank 8, cosine 0.52) |
| Recipe is fitted and verifiable, not assumed | **[M]** | 484255 (η to 5e-16; 7 wrong recipes at 6e-8…4.9 vs 5e-31) |
| Certificate quality set by imprint **spread**, degrading with chart size | **[M]** | wide head: separation 1000×/117×/10× then overlap at k=32, 40 |
| Landings uniform across the good residual deciles below the line | **[M]**, but **confirmation of (b), not a new result** | job 159323, k=16 against a certificate line of 61, `in_band: false`: deciles 0–7 land within 1e-2 at rate 1.0 and beat the norm-matched control at rate 1.0; deciles 8–9 fall to 0.2 and 0.0. The kernel count already says every floor-reacher below the line is a private image, so the certificate is separating reachers from non-reachers rather than ranking landings — uniformity is what that predicts |
| **Operational corollary: below the line the attacker has a calibration-free selection rule** | **[M]** *below-line only* | the certificate residual **is** the floor-reaching test the kernel count licenses, and it is attacker-computable: a nine-order cut at the decile 7/8 boundary (median 7.66e-11 across deciles 0–7 against 0.123 at decile 8). The difference between an attacker having to guess which landings are good and reading it off — following from the mechanism rather than adding to it |
| **Minibatching preserves the closure** | **[D]** unrun | derivation in `assumption_relaxation_program.md` §2; the induction never inspects `D_t` |
| **Certificate localises to the first adapted layer** | **[C]** untested | §3 |
| **Cap loosens from class count to layer width at a hidden layer** | **[C]** on a **[C]** | §3; needs a cell with `N ≥ m` or it cannot fire |
| The chain **cannot move the line** | **[D-proved, prop:chain / cor:chainbasin]** | `S_ρ ⊆ Z_C^{×N}×ℝ^{rN}`, so `S_chain = S_ρ` exactly (containment route), with the dimension count as an independent second route to the same boundary `k ≤ (m−1)+r−N′` (commit bb323ac) |
| The chain **does** buy basin | **[C]** unproved, unmeasured | the upside the whole chain programme rests on; see §7 |
| "Past the line the alternative is a recognisable image" | **[W]** | recognisability never assessed; synthetic 0.17–1.7%, MNIST 2.3–6.8% with 5 of 8 cells having no image inside tolerance |
| "One recorded image at low rank" | **[W]** | see §6 |
| "Richer chart ⇒ worse conditioned" as a graded law | **[W]** | ordering held, mechanism did not; family-confounded, n=4 |
| Feature-Gram coupling; label-multiset causation; "24% basin, restarts as currency" | **[W]** | QR-basis artefact; confounded draws; pre-fix QR seam |

---

## 3. The assumption stack, as it now stands

The strongest objection to this framework is not the chart and not arithmetic — it is the assumption stack.
What each restriction *actually* requires:

| As written | What is really needed | Status |
|---|---|---|
| Same private batch every step | **A fixed feature set.** The induction uses only the *shape* of `∇_B = (·)(A₀H)ᵀ` and `∇_A = A₀H(·)Hᵀ`, never `D_t`'s contents, so masking or reweighting `D_t` per step changes nothing | **[D]** — but see the caveat below |
| Full batch | nothing; `η`, `s` and `N` enter only as a product | **[M]** 484255 |
| Known learning rate | nothing; it is fitted to 5e-16 from a 2×-wrong start | **[M]** 484255 |
| SGD | **updates linear in the gradient.** Momentum and scalar weight decay are fine; Adam's entrywise normalisation is what breaks it | **[M]** ‖1ᵀB_T‖/‖B_T‖ = 3e-16 under SGD vs 2.6–2.7 under Adam |
| One adapted layer | attack the **first** adapted layer; its inputs come from frozen machinery | **[C]** |
| Output layer | only the `e^{−margin}` reading and the cap `N′ ≤ m−1` are output-layer-bound | **[C]** |

**The caveat that must travel with the minibatch result.** The closure survives, but `D_t` is masked *according
to the step-t batch*, so the simulator needs the schedule — a T×N object (3,200 bits at T=400, N=8) against an
identifiable recipe budget of `N((m−1)+r−N) − Nk` = **120** in the standard cell. So minibatching moves the
requirement from the data to the schedule; whether the schedule is recoverable is open, and a pass on the
"schedule known" arm alone would establish an assumption **swap**, not a removal. Pre-registered with both arms.

*Arithmetic (one sentence, scoped):* training in fp16/bf16 costs the attacker a tuning step rather than
providing a defence — the error signal is partly rounded to zero, so the attacker must stop at the knee instead
of the floor; not pursued further, ledger rows kept as record.

---

## 4. Open problems, ranked

1. **The start and the chart.** Replay reaches the floor from perturbed-truth starts up to 86% (job 459111) but
   from attacker-buildable starts **none of 20** reaches the floor — the single run inside the 1e-2 tolerance
   sat at residual 9.06e-7 with its restart budget exhausted (jobs 408560-63). The certificate route needs no
   start but a chart, and job 574169 showed the chart decides everything: at identical `σ_min`, three charts at
   k=17 each recover their own representable image to 1e-14 while differing by **thirteen orders** against the
   real digit. **This is the one open problem that is also the pitch**: a learned prior's job is to build the
   chart or supply the start, and this framework is the only one where a prior can be *verified* — the
   certificate and the replay residual reject a hallucination exactly. *The prior proposes; the algebra disposes.*
2. **Adam.** Identifiable at the truth at every scale tested, ~400× worse conditioned at n=96 and with a much
   smaller basin (jobs 466915, 467622) — two moderate obstacles, not non-identifiability. It also breaks the
   simplex constraint, which hands back the `N` dimensions the softmax removed.
3. **The multi-layer certificate** — §3's conjectures, unrun.
4. **Global uniqueness.** Everything is local: `Dρ` full column rank isolates the truth in a neighbourhood. No
   alias has appeared below the line, which is evidence, not proof.

---

## 5. Honest robustness ranking — and it runs backwards to interest

    most robust  ->  imprint law  >  capacity bound  >  certificate  >  chart demonstrations  <- least
    most interesting ->  chart demonstrations  >  certificate  >  capacity bound  >  imprint law

The imprint law is measured on 40 batches with a mechanism and a 4e-13 rank test; the chart results are single
cells with a chosen chart, and they are the ones that produce a picture. **Say this ordering out loud in the
meeting** — it is the question a supervisor asks second, and having the answer ready is worth more than the
picture.

---

## 6. Today's negatives — real content, not housekeeping

1. **The recorded count is not a knob; the chart sets it.** For a fixed batch `N′` *falls as the chart grows*,
   because a richer chart makes projections easier to classify. Eighteen probed cells, three batches, chart
   sizes 6–16: counts run 7 down to 3, **never 1 or 2**. A cell constructed to the plan's own specification
   (lowest-margin image plus the highest-margin image of every other class) gave 7,7,6,5,4,4,4,4.
2. **So the one-image regime and the band are mutually exclusive at a deployable rank.** The thousandfold
   imprint spread the design needed opens only at chart size 18, above the band. This is the imprint law
   constraining which batches can *exist* in the band — a structural result, not a failure to find a batch.
3. **Three harness artefacts, each of which looked like a clean null** (LESSONS_LEARNED, 2026-09-04): a
   rank-deficient Jacobian reported as "no tangent directions"; an `lstsq` default driver assuming full rank,
   putting stations at image error 1e7; and a seed block started at zero instead of the span estimate. Rule now
   enforced in code: **a null from a constrained search is not reportable unless the same pipeline reproduces a
   known positive end to end.** **That control now exists and passes** (job 156607, `--positive-control`,
   r=64, k=16, three recorded images): 172 of 200 random starts land, best error 9.5e-16, with the certificate
   gate at the truth 9.3e-11 and `fwd_check` 1.1e-15. So the negatives above are bounded — a null from this
   harness is now reportable — rather than standing as open doubt about the pipeline.
4. **A threshold below the solver's own achievable floor is not a test** (LESSONS_LEARNED, same date). Found
   three times in one day, including a pre-registration whose success branch could not have fired.

---

## 7. What is pre-registered and unrun

**The chain's state in two sentences, which belong together.** What is *proved* is that the chain cannot move
the line: `S_chain = S_ρ` exactly (Prop. 13 `prop:chain`, Cor. 14 `cor:chainbasin`, commit bb323ac), so its
entire possible value is basin — reaching solutions, never reaching further ones. What is *unproved and
unmeasured* is that it buys any basin at all. The handoff gate is the measurement that will decide it, and
**no in-band reading of it exists yet.** The first attempt was scored against a single fixed recorded image
rather than the one each start landed on and is void; the corrected run (job 159323) is clean but sits at
**k=16 against a certificate line of 61 — `in_band: false`**, forty-five units below the band, where the
certificate already isolates the images unaided, so it measures the certificate working below its line rather
than the handoff the chain needs. The one *above*-line measurement that exists points the other way: at k=58,
one unit past the line, 14.3% of 5,000 starts reach an exact certificate zero but only **0.04%** land on a
recorded image, with the best point 0.84 away (job 753886). In-band cells (k = 62–68) are queued.
**So the chain's upside is bounded above by theory and, in band, unmeasured below.** The reusable general form is worth carrying: `C` is a function of the
release, so *any* derivation in which a chain widens the admissible `k` has counted the release twice.

- **The chain test** (certificate landings handed to replay). Three branches — floor at the truth / floor at a
  wrong image / residual above the floor — plus "no landings = not scored". Scored per landing, with the
  constrained arm required to beat **the scrambled manifold** (same dimension, wrong subspace), not merely the
  unconstrained one. Success = all `N′` images clear under a one-to-one assignment; partials reported
  separately with the prediction that recovered images are the higher-imprint ones.
- **The in-band handoff (k = 62–68), pre-registered now.** Above the certificate line spurious zeros exist, so
  the residual can no longer be a clean floor-reaching test and would have to do real *ranking* work.
  **Prediction: the decile structure degrades from a clean cut to a graded, calibration-dependent signal, and
  the selection rule loses its calibration-free property.** If instead the cut survives in band, that is a
  genuinely new finding rather than a restatement of the kernel count — and it would be the most
  attacker-relevant result in the chain work, because it would mean an attacker can identify good starts in
  exactly the regime where the certificate alone does not suffice.
- **Minibatch masking**, both arms (schedule known and schedule wrong).
- **Multi-layer localisation and the cap**, gated on a cell with `N ≥ m` and on the certificate vanishing at the
  hidden layer, with the magnitude predicted (cap from class count to layer width) rather than the direction.
