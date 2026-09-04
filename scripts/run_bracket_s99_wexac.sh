#!/bin/bash
# Re-check the bracket under the DECLARED scalar (7e): s = smallest term count whose best-s-term error is within
# 1% of signal energy, i.e. the s99 column of the pre-published dose table -- not a re-read one.
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START bracket under declared s99 $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
python -u - <<'PY' | tee -a $OUT/step115_brackets99_${LSB_JOBID}.log
import json, math, glob
rows = [json.loads(l) for f in glob.glob("results/exact_inversion/step112_dose_*.jsonl") for l in open(f)]
cells = [c for r in rows if r.get("part") == "DOSE" for c in r["cells"]]
n, cond = 784, 248
print(f"{'dataset':11s} {'basis':6s} {'s99':>5s} {'s.log(n/s)':>11s} {'ratio':>7s} {'side':>6s} {'valid':>6s}")
above = below = 0; rs = []
for c in sorted(cells, key=lambda z: z["s99"]):
    s = c["s99"]; m = s * math.log(n / max(s, 1)); ratio = cond / max(m, 1e-9)
    valid = "yes" if s <= n / math.e else "NO"          # beyond n/e the bound is outside its validity range
    if valid == "yes":
        rs.append(ratio); above += ratio > 1; below += ratio <= 1
    print(f"{c['dataset']:11s} {c['basis']:6s} {s:5d} {m:11.1f} {ratio:7.2f} "
          f"{'ABOVE' if ratio>1 else 'BELOW':>6s} {valid:>6s}")
print(f"\nvalid cells: {above} above, {below} below")
print(f"ratio range over valid cells: {min(rs):.2f} .. {max(rs):.2f}   (below-side spread {1-min(rs):.0%})")
print(f"7e notes the CS constant may be ~3x off. A crossing is locatable only if the ratio range EXCEEDS that")
print(f"uncertainty on both sides; here the below side spans {1-min(rs):.0%}, so it does not.")
PY
echo "=== DONE $(date) ==="
