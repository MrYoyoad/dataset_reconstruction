#!/bin/bash
# =====================================================================
# WHY the deep curve stops: at depth 15, r=64, layers 3..14 all came back N' = r, margin 0 (VACUOUS), and only
# layers 1, 2 and the head were usable. Layer 1's input is the IMAGE and never moves, so its recorded span is at
# most N. Every deeper layer's input MOVES during training once an earlier layer is adapted, so its recorded span
# accumulates (image, step) pairs rather than images, and fills the rank.
# This tests that mechanism on its two handles: the rank r it has to fill, and the number of steps T that fill it.
#   bsub -q long-gpu -gpu "num=1" -R "rusage[mem=24576] select[ngpus>0]" -J ei91_depthmargin \
#        -o scripts/wexac_logs/ei91_depthmargin_%J.out -e scripts/wexac_logs/ei91_depthmargin_%J.err \
#        bash scripts/run_depth_margin_wexac.sh
# PRE-REGISTERED: if the mechanism is drift, margin at the deep layers is positive at small T and dies as T grows,
# and rank B_T tracks min(r, "recorded (image, step) directions") rather than N -- so raising r does NOT buy margin
# back beyond the point where the trajectory span itself runs out. If instead margin returns proportionally with r,
# the span is genuinely low rank and the deep certificate is only rank-starved, not dead.
# =====================================================================
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START depth-margin sweep $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
MODEL=models/exact_inversion/mnist_mlp_d15w1000.pth
for R in 64 128 256 512; do
  for T in 25 50 100 400; do
    echo "=== r=$R T=$T ==="
    python -u -m experiments.exact_inversion.layer_curve --model $MODEL --N 8 --r $R --k 16 --T $T --lr 0.01 \
        --seed 1 --layers 1 2 3 --out $OUT/step91_depthmargin_${LSB_JOBID}.jsonl
  done
done
echo "=== DONE $(date) ==="
