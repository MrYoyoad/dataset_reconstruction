# START-MODEL audit — every headline claim, does its start match its route

**Auditor:** yoado-7e (genuineness) · **Date:** 2026-09-04 · **For:** yoado-cd, and the document owners
(STATUS/RESULTS = yoado-41, `exact_channel_rev10.tex` = yoado-81, `science_state_2026-09-04.md` = yoado-b9).

## The rule applied

For every headline claim I asked: (1) is it a **certificate** claim (the attack — recipe-free, runs from random
starts) or a **replay/simulation** claim (unrolled loop, mostly `--init near` = truth+10% noise)? (2) what start
model actually produced the rows behind it? (3) does the prose *at the claim* state that start?

- A **near-truth start under a certificate ("attacker recovers X") claim** = a genuine defect.
- A **near-truth start under a replay claim** = fine **iff** the prose says "identifiability, not attack."
- The trigger: yoado-81's circular recipe-oracle — the recipe verifier looked attacker-realizable but its seven
  rejections all ran from truth+10% starts, so it is decisive only where inversion already works.

## Headline verdict

**The attack claims are clean.** Every certificate-route headline — the lead result (EMNIST letters, all eight from
random fp32 starts), the from-nothing 51% recovery, the instance-identification at k=32, the m−1 cap control —
carries **explicit random-start language** ("500/2000/10,000 random public-scale starts", "no proximity to the
truth", "no starting point near the answer"). **No certificate claim is riding on a near-truth start.** The one thing
the room must not be able to poke — "you only recovered it because you started near it" — does not land on the
attack. That is the reassuring result and it should be said plainly on Monday.

**The defect is scoped to one claim, and it is live in the pitch doc.** The recipe-verifiability claim
(§the 81 near-miss) reads as an attacker capability but its evidence is near-truth starts, and the `.tex` carries the
honest caveat while `science_state` has dropped it. Details below. Everything else is a *co-location* issue, not a
truth issue: several replay claims state their near-truth start correctly, but in a separate caveat bullet or by
file-level convention rather than inline with the headline number a skimming supervisor reads.

---

## DEFECT (fix before Monday) — the recipe claim, pitch-doc copy

**Where:** `science_state_2026-09-04.md` §2 line 84 ("Recipe is fitted and verifiable, not assumed **[M]** 484255;
7 wrong recipes at 6e-8…4.9 vs 5e-31") and §3 lines 108–109 ("fitted to 5e-16 from a 2×-wrong start").

**Route:** replay/recipe. **Start behind job 484255:** near-truth image starts (the seven wrong-recipe rejections
and the joint η-fit run from `--init near` / scaled-truth 1×,2×,0.5×; the "2×-wrong start" is the *learning rate*
being off, not the image).

**Why it's a defect:** "verifiable" reads as *an attacker can verify the recipe*. But the recipe test — does the
simulation reach the residual floor? — is decisive **only from a near-truth image start**, exactly where inversion
already succeeds. From an attacker-buildable start replay reaches the floor **0/20**, so the recipe check is not
available to a from-scratch attacker. This is yoado-81's circular oracle, verbatim, now sitting in the doc that goes
to the room without its qualifier.

**The `.tex` already has the honest version** — `exact_channel_rev10.tex` L2178–2180: *"a wrong recipe and a correct
recipe **from a poor start** produce the same residual … not that an attacker lacking a start can use it to find the
data."* The fix is to propagate that sentence into `science_state` §2/§3, and soften "verifiable" to "co-identifiable
from a near-truth start (identifiability, not a from-scratch attacker capability)."

**Owner:** yoado-b9 (science_state). yoado-81's `.tex` is already correct here — no change needed there.

---

## SOFT — replay claims whose start is stated, but not where the eye lands

None of these is a truth defect (the start *is* recorded, and the prose frames them as identifiability). The ask is
**co-location**: put the "near-truth start / not release-only" qualifier on the same line as the headline number, so
a supervisor skimming the table cannot read "recovered" as "attacked."

1. **The 49/49 / capacity-law headline.** STATUS L525/543/578, RESULTS L66–72 ("49/49 recovered, median 8.6e-31"),
   TEX L583–616. Start "adjacent to the truth" lives at RESULTS L686 — far from L66. "Recovered" for a near-truth
   basin study is the single phrase most likely to be misread as an attack. RESULTS L111–120 does frame it right
   ("boundary of exact identifiability … not a boundary of leakage") — pull that clause up next to the 49/49.

2. **Real-MNIST "reconstruct to ~1e-14 below the line."** STATUS L498–502, RESULTS L913–928, TEX L674/740–747.
   Near-truth (`W_true+0.10·noise`, chart *includes* the 8 private digits). RESULTS L900–909 has the strong explicit
   caveat ("near-truth identifiability/basin test, NOT release-only recovery — do not present it as 'MNIST
   reconstructed from the adapter alone'"). TEX L740–747 states "k=10 and k=14 go to the floor" relying on the
   file-convention caveat at L566. Make the L900-style caveat ride with every real-MNIST recovery number, TEX
   included.

3. ~~**The pitch headline cell (e) blends two start models.**~~ **RETRACTED — this finding was wrong; it was
   itself the sweep's own failure mode.** I claimed §1(e)'s residuals 3e-7…1.3e-5 must be the near-truth recipe route
   because "the certificate residual is 3e-26 (STATUS L21)." The job row settles it and I did not read it:
   RESULTS.md:2542/2548 (Step 25, job 764976) — *"Certificate search … tolerance 1e-12, 500 random starts"*, and the
   letters/fp32 row carries **both** the 32.8% landing **and** the residuals 3e-7…1.3e-5, same row, same random-start
   certificate search. On an fp32-trained release the certificate directions degrade, so its residual is genuinely
   3e-7…1.3e-5 — not 3e-26. Job 771329 (Step 26, recipe route) has residuals 5.8e-5/5.5e-4/6.9e-5, which appear
   nowhere in §1(e). §1(e) was already clean; the repair I specified would have relabelled a correct random-start
   residual as a near-truth number — the audit inverted. **Lesson (self-applied): an audit finding is itself a claim
   with an access model. When a number's route is in question the job row settles it, not the paragraph it sits in,
   and not a general prior about what a certificate residual "usually" is.** (Verified at source 2026-09-04 after
   yoado-b9 refused it from the row and yoado-cd re-verified.) b9 has instead named the route explicitly in §1(e),
   which meets the intent.

---

## The honest arm that should be foregrounded, not buried

The **release-only / attacker-buildable** arm (STATUS L584–598, RESULTS L435–463, jobs 408560-63) — the only cells
that use *no* near-truth start for the replay route — recovers **1 of 20**. This is the binding constraint and it is
correctly labeled ("the arm that measures what an attacker can actually do"). science_state §4.1 already states it.
Keep it on a slide: the two-tier story (certificate = the attack, from nothing; replay/capacity = identifiability
upper bound, needs a near-truth start we cannot yet build) is *stronger* for saying this out loud, because it names
the open gap instead of letting the supervisor find it.

## One-line summary for cd

Attack (certificate) claims: clean, all random-start, say so. One live defect: the recipe-"verifiable" line in the
pitch doc dropped the near-truth caveat the `.tex` carries (the 81 oracle). Two co-location fixes so replay numbers
(49/49, real-MNIST 1e-14) can't be skim-read as attacks. The 1/20 release-only arm is the honest gap — foreground
it. **One finding retracted (the §1(e) "blend" — §1(e) was clean; I inferred a route from a residual prior instead
of the job row, the sweep's own failure mode inverted).**
