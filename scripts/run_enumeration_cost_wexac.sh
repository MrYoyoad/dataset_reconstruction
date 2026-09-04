#!/bin/bash
# The enumerability argument (yoado-cd), COSTED rather than asserted. A certificate test is per-candidate, so an
# attacker facing an augmentation they cannot invert simply tests every element of the group. The question is what
# that multiplier does to the false-positive budget, and "free" is a claim that needs a number.
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START enumeration cost $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
python -u - <<'PY' | tee -a $OUT/step119_enum_${LSB_JOBID}.log
from scipy.stats import beta
# measured: 0 false positives over 1000 same-dataset non-members, in every one of 20 draws.
n, k = 1000, 0
p_hi = float(beta.ppf(0.95, k + 1, n - k))      # one-sided 95% upper bound on the per-candidate FPR
print(f"measured FPR 0/{n}; one-sided 95% upper bound on the per-candidate rate: {p_hi:.2e}\n")
print(f"{'augmentation group':28s} {'|G|':>6s} {'bound at |G| candidates':>24s}  reading")
groups = [("horizontal flip", 2), ("flip x 5-blur family", 10), ("5x5 crop grid", 25),
          ("flip x 5x5 crop grid", 50), ("flip x crop x blur", 250),
          ("flip x crop x blur x scale", 1250), ("full RandAugment-scale stack", 10000)]
for nm, g in groups:
    b = 1 - (1 - p_hi) ** g
    read = ("free" if b < 0.01 else "cheap" if b < 0.05 else "a real cost" if b < 0.5 else "defeats the test")
    print(f"{nm:28s} {g:6d} {b:24.3f}  {read}")
print("\nSo the multiplier is FREE for a small group and is NOT free for a full stack: the per-candidate rate is")
print("bounded at ~3e-3 by 1000 non-members, so a few hundred candidates per image already carries a real")
print("false-positive budget and ~1e4 defeats the test outright. 'Enumerable therefore free' holds only where")
print("|G| is small. Tightening it needs more non-members, not a better argument: the bound is 1/n-limited.")
PY
echo "=== DONE $(date) ==="
