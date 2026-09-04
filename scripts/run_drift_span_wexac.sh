#!/bin/bash
# =====================================================================
# Does the defender's "train longer" lever EXIST? (audit 02b93e8, objection from yoado-81.)
# The recorded count at a drifting layer grows with training steps, which costs certificate margin. But if the
# trajectory is smooth and confined to a low-dimensional manifold, the number of DISTINCT recorded directions
# saturates, the count stops climbing, and there is no lever at all. Measured directly: rank B_T per layer
# against T, at fixed r = 256 (well above the ~90 directions seen at T = 400, so the rank cannot cap the answer)
# and fixed N = 8.
# PRE-REGISTERED, both outcomes:
#   CLIMBS    rank B_T keeps rising with T towards r -> a crossing exists, the defender has an action, and the
#             crossing T is the number to report.
#   PLATEAUS  rank B_T flattens below r -> the drift lives in a subspace, longer training buys nothing, and the
#             lever does not exist. This is the outcome that kills a defence we have not yet claimed.
#   bsub -q long-gpu -gpu "num=1" -R "rusage[mem=32768] select[ngpus>0]" -J ei94_driftspan \
#        -o scripts/wexac_logs/ei94_driftspan_%J.out -e scripts/wexac_logs/ei94_driftspan_%J.err \
#        bash scripts/run_drift_span_wexac.sh
# =====================================================================
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START drift-span sweep $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
MODEL=models/exact_inversion/mnist_mlp_d15w1000.pth
for T in 10 25 50 100 200 400 800 1600 3200; do
  echo "=== T=$T ==="
  python -u -m experiments.exact_inversion.layer_curve --model $MODEL --N 8 --r 256 --k 16 --T $T --lr 0.01 \
      --seed 1 --layers 1 --out $OUT/step94_driftspan_${LSB_JOBID}.jsonl
done
echo "=== DONE $(date) ==="
