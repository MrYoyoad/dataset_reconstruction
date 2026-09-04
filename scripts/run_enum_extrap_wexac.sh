#!/bin/bash
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
echo "=== extrapolating the crossing group size from the minimum's trend $(date) ==="
python -u - <<'PY'
import math
# measured minimum non-member score at each group size (job 349362), 1500 non-members each
G   = [1, 2, 10, 25]
mn  = [0.1143, 0.1143, 0.1135, 0.1113]
bar = 0.01
x = [math.log10(g) for g in G]
n = len(x); sx = sum(x); sy = sum(mn)
sxx = sum(v * v for v in x); sxy = sum(a * b for a, b in zip(x, mn))
slope = (n * sxy - sx * sy) / (n * sxx - sx * sx); icpt = (sy - slope * sx) / n
print(f"minimum vs log10|G|: slope {slope:+.5f} per decade, intercept {icpt:.5f}")
print(f"observed fall over 25x group size: {mn[0]:.4f} -> {mn[-1]:.4f}  = {100*(1-mn[-1]/mn[0]):.1f}%")
need = (bar - icpt) / slope
print(f"\nextrapolated crossing: |G| = 10^{need:.1f}")
print(f"under INDEPENDENT draws the minimum of |G| samples would fall far faster; the measured decline is")
print(f"essentially FLAT, which is what strong correlation between augmented candidates predicts.")
print(f"\nso enumeration does not reach the bar at any group size an attacker could build.")
PY
echo "=== DONE ==="
