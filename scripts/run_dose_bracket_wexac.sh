#!/bin/bash
# The scoreability condition is stated as ">=1 order of magnitude in best-s-term approximation error", and that
# phrase has two readings which differ by an order: the error at a FIXED s (measured x7.2 across 12 cells), or the
# s required for a FIXED error (s95 runs 10 to 593, x59). Which one is the axis decides whether test (3) is
# scoreable at all. I am not choosing it -- the reading that lets the run proceed is the one I have an interest in.
# This computes both spans and, separately, each cell's position relative to the transition the finding is now
# pre-registered on: ratio = (r - N') / (s.log(n/s)), which must be BRACKETED by the cell list.
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START dose bracket $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
python -u - <<'PY' | tee -a $OUT/step113_bracket_${LSB_JOBID}.log
import json, math, glob
rows = [json.loads(l) for f in glob.glob("results/exact_inversion/step112_dose_*.jsonl") for l in open(f)]
cells = [c for r in rows if r.get("part") == "DOSE" for c in r["cells"]]
n, cond = 784, 248                      # pixels, and the conditions supplied at r=256 with N'=8
print(f"{'dataset':11s} {'basis':6s} {'err@50':>8s} {'s95':>5s} {'s.log(n/s)':>11s} {'ratio':>7s} {'side':>6s}")
ratios = []
for c in sorted(cells, key=lambda z: z["s95"]):
    m = c["s95"] * math.log(n / max(c["s95"], 1))
    ratio = cond / max(m, 1e-9)
    ratios.append(ratio)
    print(f"{c['dataset']:11s} {c['basis']:6s} {c['err_at_s']:8.4f} {c['s95']:5d} {m:11.1f} {ratio:7.2f} "
          f"{'ABOVE' if ratio > 1 else 'BELOW':>6s}")
errs = [c["err_at_s"] for c in cells]; s95 = [c["s95"] for c in cells]
print(f"\nspan reading A -- error at FIXED s=50 : x{max(errs)/min(errs):.1f}")
print(f"span reading B -- s95 for FIXED energy : x{max(s95)/min(s95):.1f}")
print(f"brackets the transition: {'YES' if any(r>1 for r in ratios) and any(r<1 for r in ratios) else 'NO'}"
      f"  ({sum(r>1 for r in ratios)} above, {sum(r<1 for r in ratios)} below)")
print("\nReading A fails the >=1 order bar; reading B clears it by a wide margin. The lanes fix which is the axis.")
PY
echo "=== DONE $(date) ==="
