# Arm B result — per-image adapter sensitivity vs dataset size N (whitened detection)

Status: CONCLUDED & honest (yoado-34 final call, 2026-08-27). mnist/gelu/binary, N in {2..64},
T=1000 lr=0.5 rank=8, K=50, whitened detection metric (Mahalanobis + permutation null + K-fold cross-fit).

## Headline (what survives)
1. **The whitened detection metric is the RIGHT object.** It detects a one-image swap that total-energy
   SNR calls "masked": at N=8, energy-SNR = 0.25 ("masked") vs permutation p = 0.008 (clearly detectable).
2. **Detection is REAL and ~FLAT in N (2 to 64).** A one-image swap stays significantly detectable
   (p < 0.01, null-confirmed) at every N. The energy-SNR "dilution" (||dmu|| 1.96 -> 0.95) is a RED HERRING:
   detection lives in the DIRECTION, not the energy. This is the honest "whitening reveals what energy masks"
   result — N-flat, not N-growing. Report the p-value (rank-based, floor-free), NOT an absolute d² magnitude.
3. **"Sharpens with N" (d² 14 -> 63) was a SMALL-EIGENVALUE ARTIFACT — retracted before claiming.** The
   growth was entirely lambda[0] -> 0 in the undersampled tail (downward Marchenko-Pastur bias). Floor check:
   the growth VANISHES under any sensible lambda floor; the cross-fit numerator was flat all along.

## The three-check protocol (how it was settled)
- **Null (reseed-vs-reseed, no swap):** FLAT ~0 at every N (0.002, -0.001, 0.000, 0.005, -0.004, -0.005).
  Cleared generic estimator artifacts (N-dependent subtraction bias, generic denominator inflation).
- **Decomposition (cross-fit numerator):** ||dmu|| dilutes (1.89->1.31, matches energy); numerator[0] flat
  (0.49-0.74); denominator lambda[0] shrinks (0.038->0.011). => growth is DENOMINATOR-driven.
- **Floor check (decisive):** d²(N) under lambda floored = flat (median-floor 2.4-3.7; 0.05-floor 9.7-15).
  Growth dies => it was the lambda->0 tail. (Confirmation: lambda[0] {K,2K} at K=100, N=32/64 — job 102608.)

## Methodological lesson (durable — see LESSONS_LEARNED)
Third whitening/normalizer artifact this session, all one class: energy-mask motivation -> ssim_norm
inflation -> lambda-tail inflation. **The DENOMINATOR/normalizer is where these hide.** Always: report the
floor-free permutation p-value, not the floor-dependent whitened magnitude; and floor / convergence-check
the small eigenvalues before trusting any whitened magnitude OR its N-trend.

## Carry-forward for the battery
- The metric (whitened detection + permutation null + K-fold cross-fit) is the standard measurement.
- Report dataset-composition sensitivity as: (a) permutation p-value (is it detectable), (b) flat-vs-shape
  in the whitened effect-size — never an absolute d². Small-eigenvalue floor is mandatory.

## K=100 confirmation (job 102608) — artifact confirmed via NON-CONVERGENCE, not MP-correction
Predicted (MP downward-bias): lambda[0] rises K=50->100, d² flattens. WRONG in mechanism:
  N=32: d²  53 -> 83 ;  lambda[0] 0.0091 -> 0.0072
  N=64: d²  63 -> 161;  lambda[0] 0.0118 -> 0.0055
lambda[0] DROPPED and d² GREW (>2x at N=64) with more samples. A real quantity CONVERGES with K; this
INFLATES => definitive artifact, stronger than the MP story. Mechanism: the denominator lambda is estimated
from the SAME split that defines the signal subspace (numerator cross-fit, denominator NOT), so it is
selection-biased small and the bias sharpens with more samples. Fix for any future use: cross-fit the
denominator too (lambda from a disjoint split). Robust conclusion is UNCHANGED because the reported quantity
is the rank-based permutation p-value (floor-free AND K-stable), not the magnitude: detection real + flat in N.
