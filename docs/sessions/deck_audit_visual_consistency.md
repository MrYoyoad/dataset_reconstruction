# Deck visual-consistency audit — supervisor deck (28 slides)

- **Date:** 2026-08-30
- **Auditor:** yoado-ba (visual pass, per yoado-72 request)
- **Audited:** 28 post-honesty-audit render PNGs (`slide_01`–`slide_28`), read visually with the Read tool. Spire "Evaluation Warning" watermark ignored per instruction.
- **Scope:** read-only. No slide module, pptx, or figure was edited.

Findings are ranked worst-first. Each: **Slide · defect (dimension) · fix.** Nits grouped at the end.

---

## Should-fix (ranked)

**1 · Slide 13 — text runs off the right edge (dim 4 over-packed / readability).**
The blue callout line "⇒ a ceiling on detecting the change — for every atta…" is clipped by the right slide margin; the words "attacker, reconstruction included" are lost. This is the only *hard* clip in the deck and it lands on a load-bearing sentence.
*Fix:* shorten/wrap the blue line (e.g. two lines, or "…for every attacker — reconstruction included") so it sits inside the text column; the four bullets below already wrap correctly, match that width.

**2 · Slides 11 vs 23 — the A/B/C "three worlds" change colour between the two slides they appear on (dim 3 colour semantics).**
Slide 11: A "identifiability wall" = **red**, B "extraction-limited" = **blue**, C "prior hallucination" = **amber**. Slide 23: same three worlds, but A = **amber**, B = blue, C = **gray**. A and C swap/shift hue for the identical taxonomy. A theorist who anchored "red = the wall" on slide 11 meets "amber = the wall" on slide 23.
*Fix:* lock one palette for {A wall, B extraction, C prior} and use it on both 11 and 23 (recommend keeping slide 11's A=red / C=amber, since red-as-the-hard-wall reads well; recolour slide 23 to match).

**3 · Tag-chip grammar drifts once the deck leaves Part 1 (dim 1 visual-grammar consistency).**
Part-1 answer slides (3,4,5,6,7,8) all carry a **top-right blue "your ask: …" chip** — clean and consistent. In the later sections the chip motif reappears but changes both position and colour: slide 18 has a **blue "your ask: direct inversion → KKT?" chip placed in-body** (top of the right column, not top-right of the slide); slide 20 has a **green "rigor gate" chip**, also in-body. Slide 22 (same results section) has no chip at all. So the chip means "your ask" in Part 1 but is repurposed inconsistently later.
*Fix:* pick one rule. Either (a) restore all "your ask:" chips to the top-right corner (move slide 18's), and give "rigor gate"/"your ask" a single consistent style; or (b) drop the in-body chips on 18/20 and fold that label into the card heading, reserving the top-right chip strictly for Part-1 asks.

**4 · Slide 23 — the red "decisions" box is ~60% empty (dim 4 unbalanced).**
Four short bullets sit at the top; the tall red-outlined box then runs down the full slide height with a large empty red-framed void beneath. It reads as an unfinished column next to the dense three-card row on the left.
*Fix:* size the box to its content (shorter), or promote/space the four decisions to fill it, or add the one-line "discipline:" note into the box instead of under the cards.

**5 · Slide 14 — too many objects for one point (dim 2 first-glance).**
The point ("the two-way estimator cheated itself → retracted; the 3-way converges") competes with: two plots on *different* y-axes, two stacked equations, and a three-card column. At a glance the eye ping-pongs between the +44%/+6.3% bars and the per-N bars without knowing which is the headline. The lead sentence is good but the visual field buries it.
*Fix:* demote one element — e.g. move the two equations to the appendix (25/26 already carry the formal statement) and let the left "must converge in K" bar be the single hero, right bars + cards as support.

**6 · Red carries a "hero result" meaning early and a "bad/fail" meaning late (dim 3 colour semantics).**
On slides 3–6 **red = kinked activations**, which is the *interesting, high-leakage* finding (the whole "opposite of the prediction" story). But red = **✗ failed prediction** (slide 15), **FAIL** (slide 27), **caution / "moves backwards"** (slide 26), and **the attack-fails wall** (slide 11). A theorist's default read of a tall red bar is "bad"; on slide 3 the tall red bars are actually the *loud, leaky* ones you want them to notice. The legend and lead rescue it, but the cross-slide signal is mixed.
*Lower confidence* (series colours are somewhat unavoidable). *Fix if cheap:* give the "kinked" series a non-red accent (deep orange/plum) so red stays reserved for fail/caution deck-wide; otherwise leave, the legends disambiguate.

---

## Part-structure note (dim 5)

The no-divider choice mostly works because the **top-right "your ask:" chip is itself the Part-1 marker** — its disappearance at slide 9 ("Fine-tuning is a measurement system") is the de-facto seam into Part 2. That seam is *soft*: nothing else announces the shift, so a viewer not tracking the chip could miss that slide 8→9 turns from "answering your asks" to "the instrument." The roadmap (slide 2, "Parts 2–4") primes it, and the Appendix run (24–28) is explicitly labelled, so navigation is fine except at that one 8→9 turn.
*Optional fix:* one word in the slide-9 lead or a faint "Part 2 · the instrument" eyebrow would harden the only soft seam without adding a divider slide. Note this interacts with finding 3 — if you repurpose chips in the results section, you weaken the chip-as-Part-1-marker cue.

## Templated feel (dim 6)

No slide reads as filler. The recurring card motif is deliberate, not padding: two-card contrast (6, 17), three-card taxonomy (11, 14, 23). The only mild concern is the **three-card A/B/C row appearing on both 11 and 23** — intended as a callback, but combined with finding 2 (colour drift) it currently reads as "the same template, recoloured" rather than "the same idea, revisited." Fix finding 2 and the callback reads as intentional.

---

## Nits (low priority)

- **Slide 2** — the "where" column lists targets out of slide order (G1→7, G2→8, G3→3–4, G4→5, G5→6). A viewer using the table as a map jumps backward. Harmless (logical grouping) but a reader may notice.
- **Slide 7** — right-bottom quadrant (under the "endpoint matching" formula, beside the ten-image row) is a large white gap; the formula is top-anchored and nothing balances the lower right.
- **Slide 6** — the two lower equations (Ω = … and dM ∝ σ″dz) float in a wide empty band below the two cards; acceptable as deliberate equation placement but the lower third is airy next to the packed cards.
- **Scope line** ("every leakage number here is a lower bound…") appears on 3, 16, 17, 18, 22 but is absent from other slides that also show sensitivity magnitudes (20, 21). Either it's a per-slide caveat (fine) or it should be on every leakage-number slide — currently in-between.
- **Slide 21** — four mini-panels in a row is the densest results slide; each is titled so it's legible, but it asks the viewer to integrate four plots for one "~5× more signal" point. Borderline, not a defect.

---

## Visually exemplary — do not touch

- **Slide 8** (More data / ceiling): dense but perfectly balanced — MNIST/CIFAR/Flowers pairs + formula left, faces grid + caption right. Model layout.
- **Slide 10** (Only the spectrum measures leakage): clean three-box flow + formula left, one clear two-line plot right. Textbook one-point slide.
- **Slide 12** (The test: hide one image): the swap→fine-tune→Δμ diagram is self-explanatory in 3 seconds.
- **Slide 19** (adapter records the concept): the s/d filmstrip reads instantly; formula anchored bottom-left.
- **Slide 4** (Smooth activations stay linear longer): single plot + formula + caption, no clutter.

---

## Verdict

- **Should-fix: 6** (1 hard clip · 2 colour-semantic · 1 chip-grammar · 1 empty region · 1 density).
- **Nits: 5.**
- **Clean deck-wide:** header/lead/rule/footer grammar is consistent on every content slide (2–28); no clip except slide 13; typography is uniform.

**Overall:** visually coherent and clearly hand-designed, not templated. Two systematic issues do cross slides — the A/B/C colour drift (11↔23) and the tag-chip repurposing after Part 1 — and one hard clip (slide 13) needs fixing before it's shown. Everything else is polish. The no-divider structure holds; its single soft seam (8→9) rides entirely on the chip cue, so treat findings 3 and the Part-structure note together.
