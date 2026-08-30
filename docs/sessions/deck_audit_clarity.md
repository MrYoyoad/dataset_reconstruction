# Deck clarity / flow audit — supervisor_meeting_2026_08_31.pptx (28 slides)
Auditor: yoado-23 (metric-rigor / clarity lens), for yoado-72. Date 2026-08-30.
Read as Gal would (implicit-bias / NTK theorist, sit-down discussion). Watermark = Spire eval artifact, ignored.

## Overall verdict
Strong, theorist-appropriate deck: point-first titles, one figure + ≤1 critical formula per slide, numbers in notes.
The honesty scaffolding (pre-registration table S15, three-worlds S11/S23, retraction list S27, weakest-attacker posture)
is exactly what earns a theorist's trust. Most slides answer "what are we measuring / why this function / why representative"
from visuals alone. Below: one substantive correctness catch (atlas class-vs-instance), a few density/label fixes, and cuts.

## TOP-10 RANKED FIXES
1. [CORRECTNESS] Atlas class-vs-instance: the DECK (S22) is RIGHT, the NOTE is WRONG — do not sync them the wrong way.
   Job 838868 (S22 + S28 provenance, figures/atlas/atlas.png) is the DEFAULT zoo: each composition = a DISTINCT
   DIGIT-SUBSET (build log comp0={1,6,7,8}, comp1={0,1,7}, comp2={1,6,7}, comp3={0,1,4,9}, comp4={3,4,8,9}).
   => it recovers WHICH DIGITS were present = CONTENT/CLASS-level. S22's "which digits ... not which exemplar" is CORRECT.
   thesis_note_v2 E6 currently says "same odd/even task, different samples -> which-sample, coarse INSTANCE-level, not
   class-level" — that is wrong for 838868 (that description is the SEPARATE --same_digits variant, job 838930_sd,
   atlas_samedigits.png). ACTION: fix the NOTE back to "which digit-subset / content-level"; keep S22 as-is. A theorist
   WILL ask "what are the 5 compositions?" — the answer changes the claim from class-level (mild) to instance-level (strong).
2. [CLARITY] S22 shows only the t-SNE, not the number that defends it. Add the cross-fit result on-slide
   ("+0.989 above the fitted-recipe baseline, cross-fitted, CI excludes 0; G=30") to preempt the exact "it's just a t-SNE"
   objection. Currently the number lives only on S28. Keep the "which exemplar is open -> same-digits variant" scope line.
3. [CORRECTNESS/LABEL] S10 vertical line "r=N=10 (LoRA approx full fine-tune)". The deck's own S27 retracts "r~N" in
   favour of Jang r~sqrt(N). r=N=10 is only the 10-CLASS threshold (sqrt(K.N)=sqrt(100)=10); the binary task threshold is
   sqrt(N)~3. Relabel to "10-class threshold (sqrt(K.N))" or footnote, or a theorist catches a self-contradiction with S27.
4. [HONESTY] S19 subtitle "rises monotonically with d" but the displayed s-values wobble (row1: 0.53 @ d=3.0 > 0.35 @
   d=5.3). Soften to "rises with swap distance (mid-ladder wobble is cross-exemplar noise)" so the picture matches the claim.
5. [DENSITY] S14 is the busiest slide (2 plots + 3 boxes + 2 formulas). Content is important (the winner's-curse
   retraction). Move one box (K-convergence gate) to notes or lean on S25, so the retraction reads cleanly at meeting pace.
6. [DENSITY] S21 packs 4 subplots (ViT / Fashion / full-vs-LoRA / valley) + P_LoRA. Each is legible but tight at meeting
   distance. Trim to the 2 strongest (valley + ViT) or split; the "~5x more signal, same resolution" point survives.
7. [POLISH] S8 face reconstructions have a strong red cast (low-fidelity colour recovery) — caption it ("colour is the
   ceiling's weakest channel") or a theorist asks "why red?". Also confirm face-identity provenance/consent before ANY
   circulation beyond this meeting (flagged in thesis_note_v2 to-resolve list).
8. [PRODUCTION] Confirm the "Evaluation Warning" Spire watermark is ABSENT when the .pptx is opened in PowerPoint; if it
   is embedded, re-render via a licensed / alternative path before presenting. (Low priority per request, but verify.)
9. [FLOW/CUTS] If short: merge S16 -> S17 (composition-knobs null = one line: "dilution flat, duplication sub-linear,
   context ~nothing; what matters is the image") and compress S5 to the G4 one-liner (full anchor curves already live in
   S27 appendix). Frees ~2 slides without losing a result.
10. [TITLES] Most titles are already point-first and good. Lead two with the conclusion: S10 -> "Only the spectrum
    measures usable leakage"; S18 is fine; S2/S3/S6/S12 titles are exemplary — leave them.

## MUST-NOT-SKIP (3): S3 (crux — Gal's top ask, the money answer), S6 (the mechanism: activation enters only via sigma'),
S12 (the secret-swap instrument — the whole method). Close seconds: S15 (pre-registration), S20 (H-gate licensing).

## CUT/MERGE IF SHORT (3): S16 (merge into S17), S5 (compress to G4 line; detail in S27), S21 (trim 4 panels -> 2).

## PER-SLIDE (a: visuals-alone? b: one point clear? c: defects? d: flow/missing)
- S1 title. Clean. a n/a b yes c none d good opener.
- S2 asks x answers table. a yes b yes c none d excellent Part-1 frame; G6 answer -> "notes" ok.
- S3 leakage bars, red/green, oracle diamonds, ssim_n formula. a yes b yes (kinked ~5x smooth) c none d money slide.
- S4 feature-stability curves. a yes b yes (smooth linearizes, still doesn't leak) c none d clean dissociation.
- S5 anchor two-curve, dual-axis, 4 lines. a yes b yes c slightly busy d answers G4; compressible (see #9). "why cap alpha<1" only in notes.
- S6 mechanism two-box (kinked wins / smooth wins) + Omega=GX^T, dM~sigma''. a yes b yes c none d THE mechanism; excellent.
- S7 direct inversion N=4 vs N=10 image grids. a yes b yes (superposition wall) c none d strong.
- S8 full-gradient ceiling MNIST/CIFAR/Flowers + ViT faces. a yes b yes c faces red-tinted (#7) d "CEILING not adapter-only" caveat good.
- S9 measurement-system flow z->x->adapter. a yes b yes c none d clean Part-2 setup.
- S10 dim->rank->spectrum + rank-sweep gap 23/13/0. a yes b yes c r=N=10 label (#3) d good; lead-with-conclusion title (#10).
- S11 SNR spectrum + three-worlds A/B/C. a yes b yes c none d worlds introduced here, revisited S23 (mild, ok).
- S12 secret-swap diagram D/D', clouds, d^2. a yes b yes c none d core method; excellent.
- S13 raw vs whitened + d^2 four-readings + adapter-space caveat. a yes b yes c none d strong honesty.
- S14 estimator-honest: 2-way vs 3-way, null-reads-zero, 3 boxes. a yes b yes c DENSE (#5) d retraction = credibility.
- S15 pre-registration table (arm/question/prediction/outcome, checks+crosses). a yes b yes c none d exactly what a theorist wants.
- S16 composition knobs, 3 panels + beta~0.23. a yes b yes c panels small d null result; mergeable (#9).
- S17 class identity role-swap, 4 lines + 2 boxes. a yes b yes c none d clean inversion logic.
- S18 g0 scatter rho=+0.78 (n=24, CI[.53,.91]) + mechanism. a yes b yes c none d honest (shows weaker large-n + saturation).
- S19 similarity ladder, 2 rows of swaps + s(d). a yes b yes c "monotonic" wobble (#4) d visually strong.
- S20 H-gate mem vs sensitivity rho=+0.88 (n=12). a yes b yes c none d the leakage-licensing slide.
- S21 4-panel generalization + P_LoRA. a yes b yes c DENSE (#6) d survey slide; trim.
- S22 atlas t-SNE (dW by digits / raw B,A by seed) + gauge. a yes b yes c none d FRAMING CORRECT (#1); add the number (#2).
- S23 three-worlds placement + decisions box. a yes b yes c none d strong close; answers "what I need from you".
- S24 gate-matrix rank theorem (appendix). a yes(math) b yes c dense-math (ok for appendix) d backup.
- S25 d^2 four-names + six rules (appendix). a yes b yes c none d good reference.
- S26 q_eff on col(J) construction (appendix). a yes b yes c dense (ok appendix) d backup.
- S27 anchor two-curve all-acts + "retracted, and why" (9 items incl. r~N->sqrt(N), atlas fold bug, crux sign). a yes b yes c none d EXCELLENT trust-builder.
- S28 provenance table (every headline + job). a yes b yes c none d comprehensive; E6 +0.989/838868 correct.

## APPENDIX — atlas verification (evidence for fix #1)
atlas_zoo.py:37 COMPOSITIONS = distinct digit-pairs (default); :52,:71-72 --same_digits = fixed {0,1}, comp varies IMAGE
sample (instance-level, SEPARATE). Default analyze = run_atlas_analyze_wexac.sh -> zoo_bank.pth -> atlas.png (job 838868).
Same-digits = run_atlas_analyze_sd_wexac.sh -> zoo_bank_samedigits.pth -> atlas_samedigits.png (job 838930_sd).
Zoo build log atlas_zoo_808715.out: comp0={1,6,7,8} comp1={0,1,7} comp2={1,6,7} comp3={0,1,4,9} comp4={3,4,8,9}
=> DEFAULT composition = which digits = content/class-level. S22 correct; thesis_note_v2 E6 mislabels it instance-level.
