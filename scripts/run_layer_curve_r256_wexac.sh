#!/bin/bash
# =====================================================================
# THE EXTENDED-LAYER CURVE, REDONE ABOVE THE DRIFT PLATEAU.
# Job 199890 found 12 of 15 layers vacuous and I reported the curve as FLATTENING. That was r = 64 sitting BELOW
# the drift plateau of ~90-110 recorded directions, not a property of depth: at r = 256 the same 15-layer stack
# has ALL FIFTEEN layers usable at every training length from T = 10 to T = 3200 (job 205888). So the curve has
# never actually been measured in a configuration where the layers are live. This is that measurement.
# PRE-REGISTERED: pixel rank rises as layers are added, capped by rank DF_1 = 784. The deeper layers' pixel maps
# run through encoders whose Jacobian rank collapses with depth (784, 784, 220, 96 at 1, 2, 4, 7 frozen layers,
# job 207671), so the honest prediction is that the increments SHRINK with depth rather than staying additive,
# and that the usable rank above 1e-8 falls behind the formal rank early. Report both.
#   bsub -q long-gpu -gpu "num=1" -R "rusage[mem=49152] select[ngpus>0]" -J ei96_curve256 \
#        -o scripts/wexac_logs/ei96_curve256_%J.out -e scripts/wexac_logs/ei96_curve256_%J.err \
#        bash scripts/run_layer_curve_r256_wexac.sh
# =====================================================================
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START layer curve at r=256 (above the drift plateau) $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
MODEL=models/exact_inversion/mnist_mlp_d15w1000.pth
for T in 100 400; do
  echo "=== T=$T ==="
  python -u -m experiments.exact_inversion.layer_curve --model $MODEL --N 8 --r 256 --k 16 --T $T --lr 0.01 \
      --seed 1 --layers 1 2 3 4 6 8 10 12 15 --out $OUT/step96_curve256_${LSB_JOBID}.jsonl
done
echo "=== DONE $(date) ==="
