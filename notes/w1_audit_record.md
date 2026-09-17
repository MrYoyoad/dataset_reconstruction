# W1 (`notes/w1_certificate_proofs.tex`) — audit record

**Written by the lane that authored W1, so this is a RECORD of what the audit found, not a
certification granted to itself.** It exists because the audit happened entirely in cross-session
messages between lanes that no longer exist, and the 2026-09-17 handover's central complaint is that
findings die with the sessions that hold them. Everything below is checkable from the commits named.

W1 is §1.3 item 1 of the 7 Sept brief: the proofs PDF for Gal. Four pages, theorem-first, spine
approved in A10. Built with `scripts/rev10_figs/build_pdf.sh`-style tectonic; source and PDF in `notes/`.

## What each seat checked, and what it found

| seat | register | verdict | what it changed |
|---|---|---|---|
| sentence level | read-function on the .tex | PASS | caught that the count section named the shared-concept chart regime but not the **target-conditioned** one, which is where the third link of the chain lives; caught that one remark's **heading** was a bare positive while its body led with the negative — headings travel without bodies |
| rows (measured number) | read-rows, job 331384 | **FAIL, then PASS** | see below |
| mathematics | read-function + derived | PASS on 3 revisions | see below |

## The row FAIL, which was the most consequential correction

W1 had said replay "recovered all eight to 2e-15 from 19 of 60 starts". That **spliced a count taken
at the 1e-2 landing bar onto a tolerance taken from the sub-1e-12 population**. At the rows:

- 19 starts return all eight at the 1e-2 bar;
- 18 of those have replay residual < 1e-12, worst-image error **1.408e-15 to 2.155e-15**;
- the 19th clears the loose bar at **4.371e-03**;
- 2.155e-15 rounds **above** 2e-15, so even the tight population failed the quoted bound.

Source of the error: `replay_best_err_min` = 6.6e-16 is a summary field holding the **best single
image in the best start**, read as a per-start worst case. Corrected in `c0062c3`.

Separately, "coefficient sum 1.000000000, min equal to max" was **literally false** — min
0.99999999999999911, max 1.00000000000000044. The true statement is *stronger*: maximum deviation
8.9e-16. The phrase had propagated to three documents and two lanes.

## The mathematical revisions

1. **Excitation was assumed and derived in the same theorem** — a reader working through it stops
   there. Restructured: excitation is now a **consequence** of `rank B_T = q`.
2. **Two genericity hypotheses were missing and used twice**: `rank(A₀H) = q` needs
   `col(H) ∩ ker A₀ = {0}` (A₀ maps ℝᵈ→ℝʳ with r<d), and the same condition makes
   `ker C = span(H) ⊕ ker A₀` a **direct** sum. Affine-hull dimension `min(N−1,k)` needs affine
   independence.
3. **The upgrade, which was inside W1's own proof**: Π annihilates `col(A₀H)` by construction, so
   `Π A_T = Π A₀ + (Π A₀H)M_T H^T = Π A₀` **exactly**. The quotient-sensing constant is **1**, not
   "some c_T". Consequences: the seed reduction leaves **q·d** unknowns, not q·d+1; the constant is a
   theorem confirmed by measurement (1.000000000000 on job 675031's gate) rather than a fitted
   parameter; and `c_T ≠ 1` becomes a **diagnostic** for weight decay or a non-SGD optimiser.
   Verified three ways: two-line algebra, independent simulation at 6.4e-16, and the gate. (`8d8c2aa`)
4. **The invertibility hypothesis was dropped** — traced to every place it could enter (closure
   induction, `Π A_T = Π A₀`, `CH = 0`, `rank C`, `ker C`) and it enters none. Three hypotheses
   remain. Stress-tested on job **696469**: the identity breaks in exactly one row of a step-size
   sweep, and in that row `rank B_T = 2 ≠ q = 5`, so the surviving hypothesis had failed and the
   theorem never applied. A degenerating `I + M_tG` acts **through** excitation. (`691ac2e`)

## The correction I had to make to my own replacement

I wrote, and the approver endorsed, that excitation is "checkable by the attacker from the released
factor". **It is not.** `row(B_T) ⊆ col(A₀H)` always, so `rank B_T ≤ q` unconditionally and excitation
is the case of *equality*; verifying tightness needs `H`. The attacker holds a **lower bound on q**,
not a test. What survives: the failure has a visible direction — a collapsed `rank B_T` means the
certificate is built from too small a subspace, so `CH ≠ 0` and the attack degrades rather than
silently returning something wrong-but-plausible. (`43f980e`)

## Standing rule this produced, for the handover's list

**Computable-from-X and verifiable-from-X are different properties, and the gap between them is where
an attacker-side claim quietly becomes an experimenter-side one.** Before calling a hypothesis
attacker-checkable, name the quantity on each side of the equality and ask whether the attacker has
both.

## Status

All revisions landed; PDF rebuilt at four pages. **The three PASSes were given by sessions that no
longer exist**, so a lane that needs W1 re-certified should treat this as a record of what was checked
rather than as a live sign-off — the corrections are in the commits and can be re-checked from them.
Nothing from Tracks I–III is in W1, and the sockets for them are left open.
