#!/bin/bash
# =====================================================================
# TEST 6 — the in-band handoff, THREE-BATCH attributive design. NOT LAUNCHED until 7e's round-2 controls land.
# My two-batch design was wrong: hard1_diff's band is uniformly shifted relative to confident's, so a BATCH effect
# and a BAND effect make the same prediction at every k, and same-k contrast controls the dimension but not the
# batch. b9's fix: each batch should degrade at ITS OWN line, not at a common k.
#   confident   N'=7  in-band ends 66  -> predicted onset 67
#   repeated    N'=6  in-band ends 67  -> predicted onset 68
#   hard1_diff  N'=1  in-band ends 72  -> predicted onset 73
# Sweep all three across a common k range 62..75. Onsets at 67/68/73 attribute the degradation to crossing the
# line; a COMMON onset attributes it to the chart dimension and the line has nothing to do with it.
# Reporting: per-cell verdicts and the attributive contrast are DIFFERENT CLAIMS ON DIFFERENT EVIDENCE and get two
# rows -- the contrast does not inherit the per-cell bar, and the three onsets are named either way.
# hard1_diff is a certified N' = 1 cell, so anything scored on it is NEGATIVE-SIDE ONLY: the algebraic identity
# makes the member residual zero whatever the release contains. It is also the CLEANEST per-cell version of the
# in-band question, because every multi-image confound the pre-audit named -- which landing picks which image,
# coverage of the recorded set, one-to-one matching, imprint alignment -- vanishes at one recorded image. The
# three-batch sweep and that cleanliness are compatible: the sweep carries the ATTRIBUTION and hard1_diff carries
# the cleanest PER-CELL verdict, so both are run rather than one chosen.
# PROVENANCE, required on every row from this cell: FOUND, NOT CONSTRUCTED. A natural batch at r = 64 happened to
# record a single image and we used it. The earlier conclusion that such a cell could not be CONSTRUCTED at rank 8
# stands and is unaffected; a write-up saying "we obtained the one-image in-band cell" would imply a capability
# the record does not support.
# STILL OUTSTANDING before this runs: 7e's exact form for the matched scramble arm, and the achievability-floor
# companion solve (a from-truth start under the same solver and budget), which this harness does not yet have.
# =====================================================================
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START test 6 three-batch $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
for SET in confident repeated hard1_diff; do
  echo "=== batch $SET ==="
  python -u -m experiments.exact_inversion.constrained_replay --set $SET --r 64 --N 8 \
      --ks 62 64 66 67 68 70 72 73 75 --seed 1 --arms floor d0 constrained random null \
      --out $OUT/step122_test6_${LSB_JOBID}.jsonl
done
echo "=== DONE $(date) ==="
