#!/bin/bash
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START bracket obstruction $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
python -u - <<'PY' | tee -a $OUT/step114_obstruction_${LSB_JOBID}.log
import math
# Can the basis axis alone bracket the transition ratio (r - N') / (s.log(n/s)) = 1 ?
# s.log(n/s) is CONCAVE in s with a maximum at s = n/e, so the requirement has a CEILING that does not depend on
# the basis at all. If the supplied conditions exceed that ceiling, no basis can put the cell below the boundary.
for n, name in ((784, "MNIST 28x28"), (3072, "CIFAR 32x32x3"), (150528, "224x224x3")):
    s_star = n / math.e
    m_max = s_star * math.log(n / s_star)
    print(f"{name:16s} n={n:7d}   max_s s.log(n/s) = {m_max:9.1f}  at s = {s_star:8.1f}")
    for cond, lbl in ((8, "r=16"), (56, "r=64"), (248, "r=256"), (1016, "r=1024")):
        if cond > m_max:
            print(f"    {lbl:7s} {cond:6d} conditions  ratio_min = {cond/m_max:6.2f}  -> ALWAYS ABOVE: no basis "
                  f"can bracket the transition from below")
        else:
            s_lo = None
            for s in range(1, n):
                if s * math.log(n / max(s, 1)) >= cond: s_lo = s; break
            print(f"    {lbl:7s} {cond:6d} conditions  ratio_min = {cond/m_max:6.2f}  -> crossable, needs s >~ {s_lo}")
PY
echo "=== DONE $(date) ==="
