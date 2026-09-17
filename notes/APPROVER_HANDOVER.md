# Approver handover — read before touching a claim (yoado-a8, 2026-09-17)

I am the **math/science approver**. The user assigned this seat: every scientific or mathematical claim gets my
sign-off before it enters RESULTS.md, STATUS.md, LESSONS_LEARNED.md, the `.tex`, the deck, the ledger, or anything
for Gal or his co-authors. Send claims as **Claim / Why / What ran (job id) / Found / Bottom line**; I reply
APPROVE, REVISE or REJECT. I also work the maths on half-formed ideas.

## The two files that carry everything

- `notes/math_rulings_2026-09-06.md` — **R1–R20**, the science rulings.
- `notes/plan_audit_2026-09-07.md` — **A1–A14**, the audit of the 7 Sept brief, plus everything since.

Read them before writing a claim. Several statements in circulation were withdrawn there and will otherwise be
reinstated by someone who has not seen the correction.

## STATE, and it is not good

**Nothing has run on the cluster for ten days.** Queue empty. The last commit before mine today was my own, on
7 Sept. Work from that day was staged and never committed.

**The most consequential result of 7 Sept was never read by anyone.** Job 697344 — see A13. It does two things:

1. **It refutes my own derivation twice.** I claimed the seed-free replay arm has a 32-dimensional solution family
   and that the reduced parametrisation removes it. Measured nullity is **232 in both arms**, twelve-order
   singular gap. The reduction shrinks the search and does **not** change identifiability. Withdrawn.
2. **It exposes a defect in the running harness that is free to fix.** E1B fits the product `A@H` — 192 equations
   — when the release contains `A_T` in full, 1536. Using what is already there drops the nullity **1704 → 232**
   unreduced and **681 → 232** reduced. Both arms burned four and a half hours each on a far weaker problem than
   the release supports.

**And the consequence:** nullity is positive in every cell, so the truth is **not locally isolated in either arm**.
Recovery of `H` from this release is not unique — an *information* property, not a solver property. E1B cannot
report "dynamics invertible from H: yes" in this configuration however well the solver runs.

## The first job, if you are the executor lane

**Refit E1B against the full `A_T` rather than the product**, re-measure the nullity in the configuration actually
run, and if it stays positive report E1B as an **identifiability negative** — a legitimate outcome the brief's kill
criteria anticipate. The 232 then wants its own explanation, being 200 beyond the counting deficit, but the
decision does not wait on that.

## Standing rules that will otherwise be re-broken

- **A count is never identifiability.** "Not ruled out by count" is followed in the same sentence by the fact that
  identifiability additionally needs the chart to meet the private span only at the private points. A cell with the
  count satisfied by two orders recovered nothing.
- **The routes are an equivalence, not a race.** The certificate and the linearised representer share a zero set
  where every image is recorded; replay is strictly stronger. "Cannot identify by either route" is **false** — a
  cell falsified it an hour before it would have been published.
- **The counting rule is one-sided and is NOT a defence.** Sound when it says closed, silent when it says open, and
  the remedy belongs to the attacker, who picks the chart. Geometry, not protection.
- **Every audit entry carries a register**: read-function / read-rows / derived (hypotheses named) /
  read-prose (**never a PASS**). A claim about what an experiment *did* needs the function or the row.
- **A number sourced only to our own prose — including a docstring — is not a measurement.** Three failures of that
  family in one day, one of them a mislabelled quantity in the brief itself (§6.1's 0.25 is a *certificate*
  residual, not a projection residual).
- **A comparison carries the construction of both sides**, not just their values. A claim died on two sound numbers
  whose charts were fitted on different data.
- **Absolute rank floors, never relative-only**, in any defence evaluation: merged and balanced releases give a
  zero certificate, and a relative threshold calls a zero matrix full rank — reporting a closed channel as open.
- **Verdicts never merge** "residual not zero" (optimisation failure) with "residual zero, wrong answer" (alias).

Tell me your lane and I will route only what is yours.
