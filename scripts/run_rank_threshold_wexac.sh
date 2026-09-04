#!/bin/bash
# =====================================================================
# THE RANK THRESHOLD (yoado-cd, §18): r = 256 is about a third of full rank against a deployed 8-64, so the
# deployment gap needs to be a number rather than a caveat. Four adapted layers (1..4, everything above frozen),
# r in {64, 128, 192, 256}, and the question is where the pixel count saturates.
# NOTE ON r = 64: it sits BELOW the ~90-110 drift plateau, so layers 2-4 will be rank-starved and that cell
# measures starvation, not the rank threshold. It is included as a labelled floor, not as a data point on the
# curve. sigma_min is reported beside every count.
#   bsub -q long-gpu -gpu "num=1" -R "rusage[mem=49152] select[ngpus>0]" -J ei98_rankthresh \
#        -o scripts/wexac_logs/ei98_rankthresh_%J.out -e scripts/wexac_logs/ei98_rankthresh_%J.err \
#        bash scripts/run_rank_threshold_wexac.sh
# =====================================================================
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START rank threshold at 4 adapted layers $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
MODEL=models/exact_inversion/mnist_mlp_d15w1000.pth
for R in 64 128 192 256; do
  echo "=== r=$R (layers 1..4 adapted, 5..15 frozen) ==="
  python -u -m experiments.exact_inversion.layer_curve --model $MODEL --N 8 --r $R --k 16 --T 100 --lr 0.01 \
      --seed 1 --adapt 1 2 3 4 --layers 1 2 3 4 --out $OUT/step98_rankthresh_${LSB_JOBID}.jsonl
done
echo "=== DONE $(date) ==="
