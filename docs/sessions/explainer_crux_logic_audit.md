# Logic audit — `exact_channel_crux.html` (yoado-21's explainer), 2026-09-04, by yoado-cd

Scope: does the chain entail itself step to step; is anything asserted before what it rests on; is the two-route
framing right; does it contradict the theorem statements / today's closures. Prose not audited.

## Verdict
The mathematics is correct at every step I checked (closure recurrences, seed split, faithfulness construction,
both counts, certificate derivation, imprint law, subset rescaling). Five things need fixing, worst first.

## Findings (worst first)

1. **WRONG LEAD — §"second route", "The headline cell puts the two halves together: rank 64, chart 32, 66% of 500
   random starts land, all eight found (job 728592)".** That is the *demoted* confident-batch corner: release norm
   7.6e-18, `A_T = A_0` (the adapter never fed back — in the page's own terms `M_T = 0`, the "deformation" in the
   diagram did not happen). STATUS.md:10-33 and the .tex now lead with the fp32-trained letters cell, which the page
   itself calls "the thread's lead result" two sections later (§"what is in the release"). Fix: relabel 728592 as
   "the exact-arithmetic existence corner (release 7.6e-18, adapter unmoved; instructive because the certificate is
   scale-free)" and make the letters cell the headline in this section too. yoado-81 flagged exactly this regression
   risk ("the old headline is the more quotable").

2. **CLOSING SENTENCE CONTRADICTS THE LEAD — "In one sentence: … reachable today only from a start near the
   truth".** True of the replay route only. The certificate route reaches recorded images from *random* starts
   (51% at k=6; letters 8/8 from random starts), and that is the lead. Fix: "…pinned at up to m−1+r−N coordinates
   per example by replay (which today needs a start near the truth) and at up to r−N′ by the recipe-free
   certificate (which needs no start at all), only for the examples the model had to learn, and rendered only as
   well as the attacker's chart."

3. **STALE PRECISION (two places, both contradict today's four-lane closure).**
   (a) §"turning that into images": "Stop at the floor … which is where the truth itself sits" and "a zero exists
   and is attainable, because the truth is one" hold only when the release was produced in the arithmetic you
   simulate in. For a half-precision-TRAINED release the truth sits at an intermediate residual (the knee); running
   to the floor costs ~2× fidelity on the hard chart (fp16 4.2% → 7.5%). Scope the sentence.
   (b) §"the arithmetic floor": "replaying the recipe in matched arithmetic recovers … to about 3% (782682)" — that
   is the k=16 bf16 number; the consensus wording is: stop at the knee; fp16 4.2% / bf16 4.5% at k=32 vs the chart's
   own 23% cap; the cause is underflow not inaccuracy (bf16 is 4× less accurate yet barely suffers); fp32 has no
   knee. Also "Stored in half precision … underflows to exactly zero and the recipe-free route finds nothing" is a
   property of the 7.6e-18 cell, not of fp16 (STATUS: "the fp16 zero is this cell's norm underflowing, NOT a property
   of fp16"); bf16 storage narrows the channel to ≤1 image, doesn't close it. yoado-81 holds the consensus paragraph.
   (c) §"the line": "the alias signature and the only genuine non-identifiability in the study" — now also the
   bf16 alias (residual 240× below the truth's own; wrong images explain the release better). Say "the only
   counting non-identifiability".

4. **ASSERTED BEFORE ITS BASIS — "recorded" / N′.** The certificate section uses "for every recorded example" and
   the kernel count uses N′ before the imprint law (next section) defines recording. Either move "What is in the
   release" ahead of the certificate, or add one forward-reference sentence at first use: "recorded = imprint above
   the floor, which §7 derives; write N′ for their number". Same issue, smaller: §"counting" says "B_T has rank N by
   the closure" — closure gives rank ≤ N; equality is the every-example-recorded assumption the table lists later.

5. **TWO SMALL THEORY SLIPS.**
   (a) §"second route": "the second term's rows lie inside row(B_T)". The term `A_0 H M_T Hᵀ` is r×n; its ROWS live
   in ℝⁿ, `row(B_T)` in ℝʳ. It is the term's COLUMN space (= span{A_0 h_i}) that equals row(B_T); the left
   projector kills it. A theorist will catch this.
   (b) §"which fine-tunes": "a different mechanism rather than a stronger version" (new class vs domain shift)
   sits two lines after "the chain runs through the margin, not through provenance". Same mechanism; the difference
   is that a new class makes the low margin structural (cannot decay with model quality) rather than incidental.
   Say that; b9's audit reached the same reading.

6. **MISSING THEOREM (yoado-81's finding, adopted above my #3).** The certificate line is stated as `k < r − N′`
   with the cap `N′ ≤ m − 1` appearing only in the "Measured" paragraph ("a second cap was found on the way"), not
   in the statement. Without it the line is wrong, not incomplete: with more than m−1 recorded examples the
   certificate fails for ALL of them (20 digits on a 10-class head: residuals 1e-2–0.5 for every image in exact
   arithmetic; same 20 on a padded 26-logit head: floor). State it as part of the theorem, with the derivation
   (softmax columns sum to zero ⇒ rank B_T ≤ m−1 ⇒ with N′ > m−1 no individual A_0 h_i lies in row(B_T)).

**.tex consistency (yoado-81):** nothing on the page contradicts a theorem as derived; all three counts match the
.tex exactly. Findings 1–6 sent to yoado-21 with the consensus precision paragraph verbatim.

## Two-route framing — how I would state it (to Yoad or to Gal)
Route A, replay: needs the recipe (or fits it), pins the most (m−1+r−N per image), converges only from near the
truth → its crux is the INITIALISER, which is the job for a learned prior. Route B, certificate: needs only the
release and the public model, converges from random starts, reaching zero is proof, but pins less (r−N′) and only
below its own line → its crux is the CHART. The lead result is Route B. Route A's practical role today is (i)
recipe verification/recovery and (ii) the case Route B loses — half-precision-trained releases — where matched
arithmetic plus a stopping rule still recovers. The page has all of this but never says (ii) as the resolution of
the trade; one sentence would close it.

## Checked and clean
Closure recurrences (gradients, induction, P/M update, D_t); seed split (X, Y, closed-form release, row(B_T) =
col(X)); faithfulness construction A_0′ and the "complement carries no information" argument; count
N((m−1)+r−N) incl. the softmax unit; T=1 witness; certificate derivation and kernel count r−N′ (given fix 5a);
imprint law and e^{−margin}; η·s/N product ⇒ N unidentifiable, N′ readable; subset rescale η·N′/N; the outcome
table (floor+truth / floor+elsewhere / above floor); assumptions table. Every † number I spot-checked matches
RESULTS.md/STATUS.md except the two precision figures in finding 3.
