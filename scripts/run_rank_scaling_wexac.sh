#!/bin/bash
# =====================================================================
# The FIRST adapted layer is the only one whose recorded count is the IMAGE COUNT: its input is the image and
# never moves during training, so N' = N at every T (measured: N'=8 at T=25,50,100,400 while layers 2 and 3 ran
# 32->63 and 36->64). Its margin is therefore r - N, its certificate holds at the truth to ~1e-14, and its
# conditions land directly on PIXELS with no chart. So the pixel-space constraint count should scale with the
# ADAPTER RANK, not with depth.
# PRE-REGISTERED: pixel rank = min(r, 784) - N exactly, at every r, with all conditions independent; the curve
# meets rank DF_1 = 784 at r ~ 792. The number that decides whether it MEANS anything is the usable rank above
# 1e-8 and sigma_min: squeezing r - N conditions into 784 pixel directions must degrade conditioning, and the
# honest result is where usable rank stops tracking formal rank.
#   bsub -q long-gpu -gpu "num=1" -R "rusage[mem=32768] select[ngpus>0]" -J ei92_rankscale \
#        -o scripts/wexac_logs/ei92_rankscale_%J.out -e scripts/wexac_logs/ei92_rankscale_%J.err \
#        bash scripts/run_rank_scaling_wexac.sh
# =====================================================================
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START rank-scaling (first adapted layer) $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
MODEL=models/exact_inversion/mnist_mlp_d15w1000.pth
for R in 16 32 64 128 256 400 600 784 900; do
  echo "=== r=$R ==="
  python -u -m experiments.exact_inversion.layer_curve --model $MODEL --N 8 --r $R --k 16 --T 100 --lr 0.01 \
      --seed 1 --layers 1 --out $OUT/step92_rankscale_${LSB_JOBID}.jsonl
done
echo "=== DONE $(date) ==="
